"""
Encoder Block.
Reference: Vaswani et al. (2017), Figure 1 (left half) and Section 3.1.

    x -> MultiHeadAttention(x, x, x, mask) -> AddNorm_1
      -> FeedForward                        -> AddNorm_2
      -> output
"""

import torch
import torch.nn as nn

from src.layers.multi_head_attention import MultiHeadAttention
from src.layers.feed_forward import FeedForward
from src.layers.layer_norm_residual import AddNorm

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff,dropout=dropout)

        # Two INDEPENDENT AddNorm instances -- each gets its own learnable 
        # gamma/beta, trained seperately. Never share one instance between 
        # both calls below.
        self.add_norm_1 = AddNorm(d_model, dropout=dropout)
        self.add_norm_2 = AddNorm(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args: 
            x:      (batch, seq_len, d_model) -- already has positional
                    encoding added (Step-4), from the caller
            mask:   optional bool tensor. For an encoder, this is typically
                    a padding mask only (bidirectional attention allows 
                    every position to see every other real position).

        Returns:
            output:            (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # Sub-layer 1: self-attention. Q, K, V are all 'x' -- this IS
        # what makes it "self" attention rather than cross-attention.
        attn_output, attention_weights = self.self_attention(x, x, x, mask=mask)
        x = self.add_norm_1(x, attn_output)

        # Sub-layer 2: position-wise feed-forward.
        ff_output = self.feed_forward(x)
        x = self.add_norm_2(x, ff_output)

        return x, attention_weights

    