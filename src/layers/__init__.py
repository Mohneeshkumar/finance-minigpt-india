"""
src.layers -- reusable Transformer building blocks (Steps 3-7).

This package holds the *stateless-config* layers that both MiniGPT
(decoder-only) and MiniBERT (encoder-only) are assembled from in
src/blocks/*.py. Nothing in here is model-specific: a layer must not
know whether it is being used causally (GPT) or bidirectionally (BERT) --
that decision is made by the caller via the `mask` argument.

Re-exporting the public classes/functions here lets the rest of the
project write:

    from src.layers import MultiHeadAttention, PositionalEncoding

instead of reaching into each submodule individually.
"""

from __future__ import annotations

from src.layers.scaled_dot_product import scaled_dot_product_attention
from src.layers.multi_head_attention import MultiHeadAttention
from src.layers.positional_encoding import PositionalEncoding
from src.layers.feed_forward import FeedForward
from src.layers.layer_norm_residual import AddNorm

__all__ = [
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "PositionalEncoding",
    "FeedForward",
    "AddNorm",
]