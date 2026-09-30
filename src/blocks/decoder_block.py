"""
Decoder Block (GPT-style, decoder-only — no cross-attention).

Reference: Vaswani et al. (2017), Figure 1 (right half), adapted per
Radford et al. (GPT-2) which drops the encoder-decoder cross-attention
sub-layer entirely, since a decoder-only model has no separate encoder
output to attend to.

    x -> MultiHeadAttention(x, x, x, causal_mask) -> AddNorm_1
      -> FeedForward                                -> AddNorm_2
      -> output

Structurally identical to EncoderBlock (Step 7) -- the only difference is
WHAT MASK gets passed in. This block doesn't build the mask itself; the
caller (the full MiniGPT model, Step 10) is responsible for constructing
a causal mask -- and combining it with a padding mask when batches have
padded sequences -- and passing it in here.
"""

import torch
import torch.nn as nn

from src.layers.multi_head_attention import MultiHeadAttention
from src.layers.feed_forward import FeedForward
from src.layers.layer_norm_residual import AddNorm


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()

        # Same three components as Encoder block --  this class adds no new
        # primitives, only enforces how the mask must be used at the call time.
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)

        self.add_norm_1 = AddNorm(d_model, dropout=dropout)
        self.add_norm_2 = AddNorm(d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:           (batch, seq_len, d_model) -- already has
                         positional encoding added, from the caller.
            causal_mask: REQUIRED (not optional, unlike EncoderBlock's
                         mask) -- (batch, seq_len, seq_len) or
                         (seq_len, seq_len) bool tensor, True = blocked.
                         Must at minimum encode "no attending to future
                         positions"; the caller may OR this together with
                         a padding mask for batched, variable-length
                         sequences.

        Returns:
            output:            (batch, seq_len, seq_len)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # Self-attention with the causal mask forced in. Q, K, V are all
        # 'x' -- still self-attention, just constrained to only look
        # backward in time.
        attn_output, attention_weights = self.self_attention(
            x, x, x, mask=causal_mask
        )
        x = self.add_norm_1(x, attn_output)

        # Feed-forward sub-layer -- identical to EncoderBlock, no masking
        # concept applies here since FFN never looks across positions.
        ff_output = self.feed_forward(x)
        x = self.add_norm_2(x, ff_output)

        return x, attention_weights

