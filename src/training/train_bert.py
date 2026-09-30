"""
Training loop for MiniBERT (Step 13): joint MLM + NSP pretraining on a
toy sentence-pair corpus.

Owns the two data-preparation policies that mini_bert.py deliberately does
NOT own (see mini_bert.py's module docstring):
  1. MLM masking policy -- which tokens get corrupted and how (the
     "80/10/10" rule from Devlin et al., Section 3.1).
  2. NSP pair-sampling policy -- constructing (sentence_A, sentence_B)
     pairs, 50% of the time B genuinely follows A (label = IsNext), 50%
     of the time B is a random unrelated sentence (label = NotNext).

Run:
    python -m src.training.train_bert
"""

from __future__ import annotations

import argparse
import math
import random

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.config import MiniBERTConfig, TrainingConfig
from src.model.mini_bert import MiniBERT
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.training.train_gpt import make_cosine_warmup_scheduler, set_seed


# Each entry is one short "document" -- consecutive sentences within a
# document are true NSP pairs; sentences from different documents are not.
TOY_DOCUMENTS: list[list[str]] = [
    [
        "the transformer replaced recurrent networks for sequence modeling",
        "it relies entirely on attention mechanisms",
        "no convolution or recurrence is used at all",
    ],
    [
        "bert is a bidirectional encoder representation",
        "it is pretrained with masked language modeling",
        "next sentence prediction is a second pretraining objective",
    ],
    [
        "byte pair encoding builds a vocabulary from character merges",
        "frequent adjacent pairs are merged iteratively",
        "this balances vocabulary size against sequence length",
    ],
]


def build_nsp_pairs(
    documents: list[list[str]], seed: int
) -> list[tuple[str, str, int]]:
    """Builds (sentence_a, sentence_b, is_next_label) triples.

    label = 1 means sentence_b genuinely follows sentence_a in the same
    document (IsNext); label = 0 means sentence_b was swapped in from a
    different, randomly chosen document (NotNext). 50/50 split, matching
    the original BERT pretraining recipe.
    """
    rng = random.Random(seed)
    pairs: list[tuple[str, str, int]] = []

    for doc_idx, document in enumerate(documents):
        for i in range(len(document) - 1):
            sentence_a = document[i]
            if rng.random() < 0.5:
                sentence_b = document[i + 1]
                label = 1  # IsNext
            else:
                other_docs = [d for j, d in enumerate(documents) if j != doc_idx]
                random_doc = rng.choice(other_docs) if other_docs else document
                sentence_b = rng.choice(random_doc)
                label = 0  # NotNext
            pairs.append((sentence_a, sentence_b, label))

    return pairs


def apply_mlm_masking(
    token_ids: list[int],
    tokenizer: BPETokenizer,
    rng: random.Random,
    mask_prob: float = 0.15,
) -> tuple[list[int], list[int]]:
    """Applies the BERT 80/10/10 masking rule to one sequence of ids.

    For each of the ~15% of positions selected for masking:
      - 80% of the time: replace with <MASK>
      - 10% of the time: replace with a random vocabulary token
      - 10% of the time: leave unchanged
    In every one of these selected positions, the ORIGINAL token id
    becomes the MLM training label; all other positions get label -100
    (PyTorch's default `ignore_index` for CrossEntropyLoss), so the loss
    is only computed where masking was actually applied -- exactly like
    the paper's "only the masked words are predicted" scheme.

    Returns:
        masked_ids: list[int], same length as token_ids
        labels:     list[int], same length as token_ids, -100 where the
                    position was not selected for masking
    """
    mask_id = tokenizer.token_to_id.get("<MASK>", tokenizer.token_to_id["<UNK>"])
    vocab_ids = list(tokenizer.id_to_token.keys())

    masked_ids = list(token_ids)
    labels = [-100] * len(token_ids)

    for i, original_id in enumerate(token_ids):
        if rng.random() >= mask_prob:
            continue
        labels[i] = original_id
        roll = rng.random()
        if roll < 0.8:
            masked_ids[i] = mask_id
        elif roll < 0.9:
            masked_ids[i] = rng.choice(vocab_ids)
        # else: leave unchanged (the remaining 10%)

    return masked_ids, labels


def build_dataset(
    pairs: list[tuple[str, str, int]],
    tokenizer: BPETokenizer,
    max_len: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Turns (sentence_a, sentence_b, label) triples into padded tensors:
    input_ids, segment_ids, attention_mask (True = padding), mlm_labels,
    nsp_labels -- with [CLS] prepended and [SEP] separating/ending the two
    sentences, following BERT's input format:
        [CLS] sentence_a [SEP] sentence_b [SEP]
    """
    rng = random.Random(seed)
    cls_id = tokenizer.token_to_id.get("<BOS>")  # reused as [CLS]
    sep_id = tokenizer.token_to_id.get("<EOS>")  # reused as [SEP]
    pad_id = tokenizer.token_to_id["<PAD>"]

    all_input_ids, all_segment_ids, all_attention_mask = [], [], []
    all_mlm_labels, all_nsp_labels = [], []

    for sentence_a, sentence_b, nsp_label in pairs:
        ids_a = tokenizer.encode(sentence_a)
        ids_b = tokenizer.encode(sentence_b)

        sequence = [cls_id] + ids_a + [sep_id] + ids_b + [sep_id]
        segments = [0] * (len(ids_a) + 2) + [1] * (len(ids_b) + 1)
        sequence = sequence[:max_len]
        segments = segments[:max_len]

        masked_sequence, mlm_labels = apply_mlm_masking(sequence, tokenizer, rng)

        pad_len = max_len - len(masked_sequence)
        attention_mask = [False] * len(masked_sequence) + [True] * pad_len
        masked_sequence = masked_sequence + [pad_id] * pad_len
        segments = segments + [0] * pad_len
        mlm_labels = mlm_labels + [-100] * pad_len

        all_input_ids.append(masked_sequence)
        all_segment_ids.append(segments)
        all_attention_mask.append(attention_mask)
        all_mlm_labels.append(mlm_labels)
        all_nsp_labels.append(nsp_label)

    return {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "segment_ids": torch.tensor(all_segment_ids, dtype=torch.long),
        "attention_mask": torch.tensor(all_attention_mask, dtype=torch.bool),
        "mlm_labels": torch.tensor(all_mlm_labels, dtype=torch.long),
        "nsp_labels": torch.tensor(all_nsp_labels, dtype=torch.long),
    }


def train(
    documents: list[list[str]] | None = None,
    bert_config: MiniBERTConfig | None = None,
    train_config: TrainingConfig | None = None,
) -> MiniBERT:
    documents = documents or TOY_DOCUMENTS
    train_config = train_config or TrainingConfig()
    set_seed(train_config.seed)

    flat_corpus = [s for doc in documents for s in doc]
    tokenizer = BPETokenizer()
    tokenizer.train(flat_corpus, vocab_size=128)
    # Reuse <BOS>/<EOS> as [CLS]/[SEP]; add a dedicated <MASK> token.
    if "<MASK>" not in tokenizer.token_to_id:
        new_id = len(tokenizer.token_to_id)
        tokenizer.token_to_id["<MASK>"] = new_id
        tokenizer.id_to_token[new_id] = "<MASK>"

    bert_config = bert_config or MiniBERTConfig(
        vocab_size=len(tokenizer.token_to_id),
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_seq_len=32,
        dropout=0.1,
        pad_token_id=tokenizer.token_to_id["<PAD>"],
    )

    pairs = build_nsp_pairs(documents, seed=train_config.seed)
    dataset = build_dataset(
        pairs, tokenizer, max_len=bert_config.max_seq_len, seed=train_config.seed
    )

    device = torch.device(
        "cuda" if train_config.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = MiniBERT(bert_config).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    num_examples = dataset["input_ids"].size(0)
    num_batches_per_epoch = max(1, num_examples // train_config.batch_size)
    total_steps = num_batches_per_epoch * train_config.num_epochs
    scheduler = make_cosine_warmup_scheduler(
        optimizer, train_config.warmup_steps, total_steps
    )

    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_criterion = nn.CrossEntropyLoss()

    model.train()
    global_step = 0
    for epoch in range(train_config.num_epochs):
        permutation = torch.randperm(num_examples)
        epoch_loss = 0.0

        for start in range(0, num_examples, train_config.batch_size):
            batch_idx = permutation[start : start + train_config.batch_size]

            input_ids = dataset["input_ids"][batch_idx].to(device)
            segment_ids = dataset["segment_ids"][batch_idx].to(device)
            attention_mask = dataset["attention_mask"][batch_idx].to(device)
            mlm_labels = dataset["mlm_labels"][batch_idx].to(device)
            nsp_labels = dataset["nsp_labels"][batch_idx].to(device)

            outputs = model(
                input_ids, segment_ids=segment_ids, attention_mask=attention_mask
            )

            mlm_loss = mlm_criterion(
                outputs["mlm_logits"].reshape(-1, outputs["mlm_logits"].size(-1)),
                mlm_labels.reshape(-1),
            )
            nsp_loss = nsp_criterion(outputs["nsp_logits"], nsp_labels)
            # Combined loss, equal weighting -- matches the original paper,
            # which sums the two objectives without extra scaling.
            loss = mlm_loss + nsp_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % train_config.log_every == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"step {global_step:5d} | epoch {epoch+1} | "
                    f"loss {loss.item():.4f} | mlm {mlm_loss.item():.4f} | "
                    f"nsp {nsp_loss.item():.4f} | lr {current_lr:.6f}"
                )

        avg_loss = epoch_loss / num_batches_per_epoch
        print(f"== epoch {epoch+1} done | avg loss {avg_loss:.4f} ==")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain MiniBERT with MLM + NSP.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_config = TrainingConfig(
        seed=args.seed, num_epochs=args.epochs, batch_size=args.batch_size
    )
    train(train_config=train_config)


if __name__ == "__main__":
    main()
