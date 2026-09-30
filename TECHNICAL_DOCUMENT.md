# miniGPT + BERT from Scratch — Technical Document

**Project:** GenAI Roadmap Phase 1 (P1) — "Transformers Built from 200 Lines"
**Scope:** A from-scratch (no HuggingFace model loading) PyTorch implementation of a
GPT-2-style decoder-only language model and a BERT-style bidirectional encoder,
including a from-scratch BPE tokenizer, training loops, and attention visualization.

---

## 1. Architecture Overview

Both models are built from the same small set of primitives (`src/layers/`,
`src/blocks/`), verified independently before being assembled into the two
top-level models (`src/model/`):

```
Primitives (Steps 1–6)              Blocks (Steps 7–8)         Models (Steps 10–11)
─────────────────────────           ─────────────────          ────────────────────
scaled_dot_product_attention   ──┐
causal_mask / padding_mask       │─▶  EncoderBlock (Step 7) ──▶ MiniBERT (Step 11)
MultiHeadAttention                │─▶  DecoderBlock (Step 8) ──▶ MiniGPT  (Step 10)
PositionalEncoding                │
FeedForward                       │
AddNorm (residual + LayerNorm)  ──┘
```

```
src/
├── config.py                  # MiniGPTConfig, MiniBERTConfig, TrainingConfig
├── layers/
│   ├── scaled_dot_product.py  # Step 1 — Attention(Q,K,V) = softmax(QKᵀ/√d_k)V
│   ├── multi_head_attention.py# Step 3
│   ├── positional_encoding.py # Step 4 — sinusoidal PE
│   ├── feed_forward.py        # Step 5 — position-wise FFN
│   └── layer_norm_residual.py # Step 6 — Add & Norm (Post-LN)
├── masks/
│   └── mask_utils.py          # Step 2 — causal_mask, padding_mask
├── blocks/
│   ├── encoder_block.py       # Step 7 — bidirectional, self-attn + FFN
│   └── decoder_block.py       # Step 8 — causal, self-attn + FFN (GPT-style)
├── tokenizer/
│   └── bpe_tokenizer.py       # Step 9 — from-scratch BPE
├── model/
│   ├── mini_gpt.py            # Step 10 — decoder-only causal LM
│   └── mini_bert.py           # Step 11 — bidirectional encoder + MLM/NSP heads
└── training/
    ├── train_gpt.py           # Step 12 — AdamW + cosine warmup + grad clip
    └── train_bert.py          # Step 13 — MLM (80/10/10) + NSP data prep + training

notebooks/
└── attention_visualization.ipynb  # Step 14 — per-head heatmaps for both models

tests/                          # one file per module, pytest
```

Project structure is **unchanged** from the original scaffold — no directories were
added or removed; only the previously-empty stub files were implemented.

---

## 2. Design Decisions

### 2.1 Post-LN, not Pre-LN
`AddNorm` implements `LayerNorm(x + Dropout(Sublayer(x)))`, matching Vaswani et al.
(2017) exactly, rather than GPT-2's Pre-LN (`x + Sublayer(LayerNorm(x))`). This is
deliberate: the P1 learning goal is fidelity to the original architecture. Pre-LN is
noted in code comments as the modern alternative and why it trains more stably at
depth, but was not substituted in.

### 2.2 MiniGPT's decoder has 2 sub-layers, not 3
The original paper's decoder (Figure 1, right half) has three sub-layers: masked
self-attention, encoder-decoder cross-attention, and a feed-forward network — because
it is a sequence-to-sequence model translating from a separate encoder's output.
MiniGPT follows Radford et al. (GPT/GPT-2) instead: a **decoder-only** architecture
with no separate encoder to attend to, so the cross-attention sub-layer is dropped
entirely. `DecoderBlock` therefore has exactly two sub-layers (masked self-attention +
FFN), structurally identical to `EncoderBlock` except for which mask is forced in.
This was raised explicitly during development (given the paper's diagram) and
confirmed as the intended design — not an oversight.

### 2.3 BPE tokenizer, word-level (not byte-level)
`BPETokenizer` implements Sennrich et al. (2015)'s character-merge algorithm at the
**word** level (splitting on whitespace, tracking an explicit `</w>` end-of-word
marker) rather than GPT-2's byte-level BPE. Byte-level BPE guarantees no `<UNK>` can
ever occur (any byte sequence is representable), at the cost of implementation
complexity; the word-level variant here is the more direct implementation of the
paper's original algorithm and is sufficient for the toy corpora this project trains
on. This is a stated simplification, not a missing feature.

### 2.4 Weight tying
Both `MiniGPT.lm_head` and `MiniBERT.mlm_head` share their weight matrix with the
token embedding table (`lm_head.weight = token_embedding.weight`). This is the
standard GPT-2 / BERT practice: it roughly halves the parameter count contributed by
embeddings and is reported to improve output quality, since the same matrix is used
to both encode and decode token identity.

### 2.5 Data/Model/Training separation
Per the project's layering standard:
- **Model layer** (`mini_gpt.py`, `mini_bert.py`) owns architecture only — embeddings,
  block stacking, output heads, and (for MiniGPT) autoregressive `generate()`.
- **Training layer** (`train_gpt.py`, `train_bert.py`) owns everything data- and
  optimization-related: building windows for causal LM, the MLM 80/10/10 masking
  policy, NSP pair sampling, AdamW, the cosine warmup schedule, gradient clipping, and
  the training loop itself.

This keeps the model classes reusable for inference/fine-tuning without dragging in
data-preparation code, and keeps every reproducibility knob (seed, batch size, LR
schedule) in one place (`TrainingConfig`).

### 2.6 Reproducibility
`train_gpt.set_seed()` (reused by `train_bert`) seeds Python's `random`, `torch`, and
`torch.cuda` (when available) from `TrainingConfig.seed` (default 42) at the start of
every training run, so results are reproducible run-to-run given the same config and
corpus.

---

## 3. Bugs Found and Fixed During This Update

Three defects were found in the on-disk code while verifying parity with the
project's tested reference implementation, prior to adding any new files. All three
were fixed; none required a design change, only a correction of the implementation.

| # | File | Defect | Impact |
|---|------|--------|--------|
| 1 | `src/masks/mask_utils.py` | Function was named `casual_mask` (typo) instead of `causal_mask` | `ImportError` anywhere the correctly-spelled name is imported — including `decoder_block.py` and the new `mini_gpt.py` |
| 2 | `src/layers/scaled_dot_product.py` | `F.softmax(scores, dim=1)` instead of `dim=-1` | Normalizes attention weights over the wrong axis for any real (non-2D) tensor — attention weights would not sum to 1 over keys, silently corrupting every attention computation |
| 3 | `src/layers/positional_encoding.py` | `forward()` was indented at module level, outside the `PositionalEncoding` class body, and used `x.size()` (a `torch.Size` tuple) instead of `x.size(1)` | `PositionalEncoding()(x)` raised `NotImplementedError` unconditionally — the class had no working forward pass at all |

Bug #3 was the most severe: it meant no model using positional encoding (including
the about-to-be-built MiniGPT/MiniBERT) could run a forward pass at all. It was
caught only once `PositionalEncoding` was actually invoked end-to-end while building
`MiniGPT` — its prior test file apparently did not call `forward()` directly. A
regression test (`tests/test_positional_encoding.py::test_forward_is_callable_and_returns_correct_shape`)
now exercises the real call so this class of bug (a method silently excluded from the
class) cannot pass silently again.

Also flagged: a stray, empty, misnamed test file `tests/test_reference_comparision.py`
(typo — "comparision") duplicates the correctly-named `test_reference_comparison.py`
and is empty. Left in place pending your confirmation to delete it (device file
deletion requires explicit permission).

---

## 4. New Files Added

| File | Step | Contents |
|---|---|---|
| `src/config.py` | — | `MiniGPTConfig`, `MiniBERTConfig`, `TrainingConfig` dataclasses |
| `src/tokenizer/bpe_tokenizer.py` | 9 | Populated with the previously-designed, tested `BPETokenizer` (was empty on disk) |
| `src/model/mini_gpt.py` | 10 | `MiniGPT`: embeddings + N×`DecoderBlock` + LM head + `generate()` |
| `src/model/mini_bert.py` | 11 | `MiniBERT`: embeddings + segment embeddings + N×`EncoderBlock` + MLM head + NSP head |
| `src/training/train_gpt.py` | 12 | Causal-LM training loop: AdamW, cosine warmup, grad clipping, seeded |
| `src/training/train_bert.py` | 13 | MLM (80/10/10 masking) + NSP (sentence-pair sampling) training loop |
| `notebooks/attention_visualization.ipynb` | 14 | Trains both toy models, renders per-head attention heatmaps, notes on extending to W&B logging |
| `tests/test_decoder_block.py`, `test_bpe_tokenizer.py`, `test_positional_encoding.py`, `test_mini_gpt.py`, `test_mini_bert.py` | — | New pytest coverage for every module added or fixed in this pass |
| `tests/conftest.py` | — | `sys.path` setup so `from src...` imports resolve regardless of the invocation directory |

All 24 new/updated tests pass (`python -m pytest tests/`).

---

## 5. How to Run

```bash
pip install -r requirements.txt

# Run the full test suite
python -m pytest tests/ -v

# Train MiniGPT on the built-in toy corpus (or pass --corpus path/to/file.txt)
python -m src.training.train_gpt --epochs 5 --batch-size 4

# Pretrain MiniBERT (MLM + NSP) on the built-in toy documents
python -m src.training.train_bert --epochs 10 --batch-size 2

# Visualize attention (Jupyter required: pip install notebook)
jupyter notebook notebooks/attention_visualization.ipynb
```

Both training scripts use small, hardcoded toy corpora (`TOY_CORPUS`,
`TOY_DOCUMENTS`) so the project runs end-to-end with no external dataset — pass
`--corpus` (GPT) or edit `TOY_DOCUMENTS` (BERT) to train on real text.

---

## 6. Limitations

- **Toy scale only.** Default configs (`d_model=64`, `num_layers=2`) and corpora are
  sized to run in seconds on CPU for learning purposes, not to produce a competitive
  language model or strong downstream classification accuracy.
- **No W&B integration wired in.** The roadmap's P1 deliverable mentions a "W&B
  experiment dashboard"; this was deliberately left out of `train_gpt.py` /
  `train_bert.py` to avoid a hard dependency on an external account for the base
  project to run standalone. Adding `wandb.init()` / `wandb.log({...})` around the
  existing `print()` logging calls is a small, additive change — see the "Next
  Steps" note in the training scripts' log statements.
- **No perplexity-curve plotting utility yet.** Loss/perplexity are printed per
  step; persisting them to a list and plotting with matplotlib (or W&B) is a
  straightforward follow-on, not yet automated.
- **`BPETokenizer` is word-level**, not byte-level (see §2.3) — `<UNK>` is possible
  for characters never seen during `.train()`.
- **No fine-tuning script for classification** yet (the roadmap mentions
  "fine-tune the encoder on classification"). `MiniBERT`'s NSP head demonstrates
  attaching a task-specific head to the `[CLS]` representation; a dedicated
  `train_classifier.py` reusing that same pattern (replace the 2-way NSP head with
  an N-way class head, freeze or fine-tune the encoder) is the natural next module.

## 7. References

- Vaswani et al. (2017), *Attention Is All You Need*
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2)
- Devlin et al. (2018), *BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding*
- Sennrich et al. (2015), *Neural Machine Translation of Rare Words with Subword
  Units* (BPE)
- He et al. (2015), *Deep Residual Learning for Image Recognition* (residual
  connections)
- Ba et al. (2016), *Layer Normalization*
