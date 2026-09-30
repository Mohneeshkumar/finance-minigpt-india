# miniGPT + BERT from Scratch — GenAI Roadmap Phase 1 (P1)

This repo started as a standalone "attention mechanism from scratch" learning
exercise and has been folded into **Phase 1** of the GenAI Engineering
Roadmap:

> **P1 — GenAI Foundations + Early LLMs from Scratch**
> *Project: miniGPT + BERT Classifier — Transformers Built from 200 Lines*
> Modules covered: MODULE 1 (Foundations), MODULE 2 (Early LLMs),
> MODULE 5 concepts (Training an LLM from Scratch)

The attention primitives already built (scaled dot-product attention,
masking, multi-head attention, positional encoding) are not a separate
side project — they are the shared engine that both the GPT decoder and
the BERT encoder are built on top of.

## Goal

Implement a full GPT-2-style decoder and a BERT-style encoder entirely
from scratch in PyTorch — no HuggingFace model loading, only raw math —
train miniGPT on text, fine-tune the encoder on classification, and
visualize attention heads across layers.

## Project Structure

```
attention-from-scratch/
├── README.md
├── requirements.txt
├── src/
│   ├── layers/
│   │   ├── scaled_dot_product.py      [done]  Step 1
│   │   ├── multi_head_attention.py    [done]  Step 3
│   │   ├── positional_encoding.py     [done]  Step 4
│   │   ├── feed_forward.py            [next]  Step 5
│   │   └── layer_norm_residual.py             Step 6
│   ├── masks/
│   │   └── mask_utils.py              [done]  Step 2
│   ├── blocks/
│   │   ├── encoder_block.py                   Step 7  (BERT's building block)
│   │   └── decoder_block.py                   Step 8  (GPT's building block, causal-only)
│   ├── tokenizer/
│   │   └── bpe_tokenizer.py                   Step 9  (BPE from scratch)
│   ├── model/
│   │   ├── mini_gpt.py                        Step 10 (stacked decoder blocks + LM head)
│   │   └── mini_bert.py                       Step 11 (stacked encoder blocks + MLM head)
│   └── training/
│       ├── train_gpt.py                       Step 12 (AdamW, cosine warmup, grad clipping)
│       └── train_bert.py                      Step 13 (MLM objective)
├── notebooks/
│   └── attention_visualization.ipynb          Step 14 (attention heads across layers)
└── tests/                                      mirrors src/, oracle-tested against PyTorch reference
```

## Build Order & Roadmap Alignment

| Step | Module | P1 Deliverable |
|---|---|---|
| 1 | `layers/scaled_dot_product.py` | Core transformer math |
| 2 | `masks/mask_utils.py` | Causal masking foundation |
| 3 | `layers/multi_head_attention.py` | Multi-head attention (oracle-tested vs `nn.MultiheadAttention`) |
| 4 | `layers/positional_encoding.py` | Sinusoidal positional encoding |
| 5 | `layers/feed_forward.py` | Position-wise FFN, shared by GPT & BERT blocks |
| 6 | `layers/layer_norm_residual.py` | Add & Norm, shared by GPT & BERT blocks |
| 7 | `blocks/encoder_block.py` | BERT's building block |
| 8 | `blocks/decoder_block.py` | GPT's building block (causal self-attention only) |
| 9 | `tokenizer/bpe_tokenizer.py` | "BPE tokenizer built from scratch on a custom corpus" |
| 10 | `model/mini_gpt.py` | "Full GPT decoder: embedding → MHA → FFN → residuals → causal mask" |
| 11 | `model/mini_bert.py` | "BERT encoder with MLM pre-training objective and NSP task" |
| 12 | `training/train_gpt.py` | "Training loop: AdamW + cosine LR warmup + gradient clipping + seed control" |
| 13 | Perplexity / loss tracking | "Perplexity curves, loss analysis, model comparison (your GPT vs GPT-2 small)" |
| 14 | `notebooks/attention_visualization.ipynb` | "Attention pattern visualization across layers and heads" |

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

Every module is validated against a test file before the next one is built.
`tests/test_reference_comparison.py` cross-checks `MultiHeadAttention`
against `torch.nn.MultiheadAttention` with identical weights — the
strongest correctness signal in the repo so far.

## Reproducibility

All tests and training scripts set `torch.manual_seed(42)`.

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need.*
- Radford, A. et al. GPT-2 (decoder-only architecture reference).
- Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding.*
