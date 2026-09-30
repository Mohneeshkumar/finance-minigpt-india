"""
Training loop for MiniGPT (Step 12): causal language modeling on a toy
text corpus.

This is the Training Layer -- it owns the optimizer, learning-rate
schedule, gradient clipping, the training/validation split, the loss
function, and the logging/checkpointing loop. It does NOT own model
architecture (mini_gpt.py) or the tokenizer (bpe_tokenizer.py); it only
consumes them.

Run:
    python -m src.training.train_gpt --corpus path/to/text.txt

With no --corpus given, a small built-in toy corpus is used so the script
runs end-to-end out of the box for learning/demo purposes.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import replace

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.config import MiniGPTConfig, TrainingConfig
from src.model.mini_gpt import MiniGPT
from src.tokenizer.bpe_tokenizer import BPETokenizer


TOY_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "attention is all you need for transformers",
    "a decoder only model predicts the next token",
    "byte pair encoding merges frequent character pairs",
    "gradient clipping prevents exploding gradients during training",
    "positional encoding injects order information into embeddings",
    "the transformer architecture uses self attention layers",
    "masked language modeling corrupts tokens for bert pretraining",
]


def set_seed(seed: int) -> None:
    """Seed every RNG the training loop touches, for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    """Linear warmup for `warmup_steps`, then cosine decay to 0 over the
    remaining steps. Returns a LambdaLR so the schedule is just a
    multiplier on the base learning_rate, easy to plot and reason about.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def build_dataset(
    corpus: list[str], tokenizer: BPETokenizer, seq_len: int
) -> torch.Tensor:
    """Concatenates every encoded sentence with an <EOS> boundary, then
    slices the flat token stream into fixed-length (seq_len + 1) windows
    -- input is window[:-1], target is window[1:] (next-token prediction).

    Returns a single (num_windows, seq_len + 1) tensor for simplicity;
    a real project would use a torch.utils.data.Dataset/DataLoader, but
    a toy in-memory tensor is adequate for a from-scratch learning demo.
    """
    eos_id = tokenizer.token_to_id["<EOS>"]
    all_ids: list[int] = []
    for sentence in corpus:
        all_ids.extend(tokenizer.encode(sentence))
        all_ids.append(eos_id)

    window = seq_len + 1
    num_windows = max(1, len(all_ids) - window + 1)
    windows = [all_ids[i : i + window] for i in range(num_windows)]
    # Pad the last window if the corpus doesn't divide evenly.
    pad_id = tokenizer.token_to_id["<PAD>"]
    windows = [w + [pad_id] * (window - len(w)) for w in windows]
    return torch.tensor(windows, dtype=torch.long)


def train(
    corpus: list[str] | None = None,
    gpt_config: MiniGPTConfig | None = None,
    train_config: TrainingConfig | None = None,
) -> MiniGPT:
    corpus = corpus or TOY_CORPUS
    train_config = train_config or TrainingConfig()
    set_seed(train_config.seed)

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=128)

    gpt_config = gpt_config or MiniGPTConfig(
        vocab_size=len(tokenizer.token_to_id),
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        max_seq_len=32,
        dropout=0.1,
        pad_token_id=tokenizer.token_to_id["<PAD>"],
    )
    if gpt_config.vocab_size != len(tokenizer.token_to_id):
        # Keep model and tokenizer vocab sizes consistent even if a caller
        # passed a pre-built config with a stale vocab_size.
        gpt_config = replace(gpt_config, vocab_size=len(tokenizer.token_to_id))

    data = build_dataset(corpus, tokenizer, seq_len=gpt_config.max_seq_len - 1)
    inputs, targets = data[:, :-1], data[:, 1:]

    device = torch.device(
        "cuda" if train_config.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = MiniGPT(gpt_config).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    num_batches_per_epoch = max(1, inputs.size(0) // train_config.batch_size)
    total_steps = num_batches_per_epoch * train_config.num_epochs
    scheduler = make_cosine_warmup_scheduler(
        optimizer, train_config.warmup_steps, total_steps
    )

    # ignore_index skips the padding token when computing the loss, so
    # padded positions contribute no gradient signal (same rationale as
    # padding_idx on the embedding table).
    criterion = nn.CrossEntropyLoss(ignore_index=gpt_config.pad_token_id)

    model.train()
    global_step = 0
    for epoch in range(train_config.num_epochs):
        permutation = torch.randperm(inputs.size(0))
        epoch_loss = 0.0

        for start in range(0, inputs.size(0), train_config.batch_size):
            batch_idx = permutation[start : start + train_config.batch_size]
            batch_inputs = inputs[batch_idx].to(device)
            batch_targets = targets[batch_idx].to(device)

            logits, _ = model(batch_inputs)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), batch_targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping: caps the L2 norm of all gradients combined,
            # preventing a single bad batch from taking an oversized step.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % train_config.log_every == 0:
                perplexity = math.exp(min(loss.item(), 20))  # cap to avoid overflow
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"step {global_step:5d} | epoch {epoch+1} | "
                    f"loss {loss.item():.4f} | ppl {perplexity:.2f} | "
                    f"lr {current_lr:.6f}"
                )

        avg_loss = epoch_loss / num_batches_per_epoch
        print(f"== epoch {epoch+1} done | avg loss {avg_loss:.4f} ==")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MiniGPT on a text corpus.")
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to a text file (one training example per line). "
        "If omitted, a small built-in toy corpus is used.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus = TOY_CORPUS
    if args.corpus:
        with open(args.corpus, "r", encoding="utf-8") as f:
            corpus = [line.strip() for line in f if line.strip()]

    train_config = TrainingConfig(
        seed=args.seed, num_epochs=args.epochs, batch_size=args.batch_size
    )
    train(corpus=corpus, train_config=train_config)


if __name__ == "__main__":
    main()
