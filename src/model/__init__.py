"""
src.model -- the two pretrained architectures this project implements
(Step 10: MiniGPT, Step 11: MiniBERT).

Both classes are built out of src/blocks (EncoderBlock / DecoderBlock),
which are in turn built out of src/layers. This package is the top of
that stack: it owns token/segment/positional embeddings, the stack of
blocks, and the output head(s) (LM head for MiniGPT; MLM + NSP heads for
MiniBERT). It does NOT own training loops, optimizers, or the masking/
NSP-sampling policy used during pretraining -- see src/training for that
split (mask_utils.py / train_gpt.py / train_bert.py).

    from src.model import MiniGPT, MiniBERT
"""

from __future__ import annotations

from src.model.mini_gpt import MiniGPT
from src.model.mini_bert import MiniBERT

__all__ = [
    "MiniGPT",
    "MiniBERT",
]