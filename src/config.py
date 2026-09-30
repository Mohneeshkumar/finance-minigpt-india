"""
Shared configuration dataclasses for MiniGPT and MiniBERT.

Kept deliberately separate from model/training code (single-responsibility):
this module ONLY defines hyperparameters and their defaults. It has no
torch.nn imports and no training logic, so it can be imported by data,
model, training, and evaluation code alike without circular dependencies.

Reproducibility: `seed` lives here so every entry point (train_gpt.py,
train_bert.py, notebooks) reads it from one place instead of hardcoding
a magic number in several files.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MiniGPTConfig:
    """Hyperparameters for the decoder-only MiniGPT model (Step 10).

    Defaults are intentionally small (toy-scale) so the model trains in
    seconds to minutes on CPU for learning purposes -- this is NOT meant
    to be a competitive language model, only a correctness-verified,
    from-scratch implementation of the architecture.
    """

    vocab_size: int = 512          # set from BPETokenizer.vocab_size after training
    d_model: int = 128             # embedding / residual stream width
    num_heads: int = 4             # must divide d_model evenly
    d_ff: int = 512                # feed-forward inner dimension (paper uses 4x d_model)
    num_layers: int = 4            # number of stacked DecoderBlocks
    max_seq_len: int = 128         # maximum context length (positional encoding table size)
    dropout: float = 0.1
    pad_token_id: int = 0          # index of <PAD> in the tokenizer vocabulary

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )


@dataclass(frozen=True)
class MiniBERTConfig:
    """Hyperparameters for the bidirectional MiniBERT encoder (Step 11)."""

    vocab_size: int = 512
    d_model: int = 128
    num_heads: int = 4
    d_ff: int = 512
    num_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1
    pad_token_id: int = 0
    type_vocab_size: int = 2        # segment embedding: sentence A / sentence B (for NSP)

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer / scheduler / loop hyperparameters shared by both trainers.

    Per the project's AI/ML Standards: AdamW + cosine LR warmup + gradient
    clipping + explicit seed control, so every run is reproducible.
    """

    seed: int = 42
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    log_every: int = 20
    device: str = "cpu"             # overridden at runtime if CUDA is available
