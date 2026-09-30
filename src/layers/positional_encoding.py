"""
Sinusoidal Positional Encoding

    PE(pos,2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos,2i+1) = cos(pos / 10000^(2i / d_model))

Where:
    pos = the token's position in the sequence (0,1,2, ...)
    i   = the dimension index within the embedding (0,1,..., d_model/2 - 1)

Why sine AND cosine, alternating? Two reasons the paper gives:
  1. Every dimension pair (2i, 2i+1) oscillates at a different frequency
     (the 10000^(2i/d_model) term), so nearby positions are similar
     encodings while far-apart positions get very different ones - 
     the model can learn to attend by *relative* position, because
     PE(pos + k) can be expressed as a linear function of PE(pos) for
     any fixed offset k (a property of sin/cos addition formulas).
  2. Bounded output range ([-1,1]) keeps the encoding on a similar scale
     to the token embeddings it gets added to, so it doesn't dominate or
     vanish relative to the semantic content.

This encoding is precomputed once (it depends only on position and
d_model, never on the input data) and simply added to the token
embeddings, not concatenated — this keeps the dimensionality unchanged
going into the rest of the network.
"""

import math

import torch 
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Precompute the (max_len, d_model) table of positional encodings
        # once, up front, since it never depends on the actual input data
        # - only on position index and dimension index.
        pe = torch.zeros(max_len, d_model)

        # (max_len, 1) column vector: [0,1,2,3,..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term computes 1 / 10000^(2i/d_model) for each even dimension
        # index i, but via exp(log(...)) instead of a direct power - this 
        # is a standard numerical-stability trick: 10000^(2i/d_model) can 
        # overflow for large d_model computed directly, but
        # exp(2i * -log(10000)/d_model) stays well-behaved.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Even dimensions (0, 2, 4, ...) get sine; odd dimensions
        # (1, 3, 5, ...) get cosine of the SAME base frequency — they're
        # paired, not independent, which is what gives this encoding its
        # relative-position-recoverable property.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: (1, max_len, d_model). This lets 'pe'
        # broadcast against any batch size in forward() without 
        # modification.
        pe = pe.unsqueeze(0)

        # register_buffer stores `pe` as part of the module's state
        # (moves with .to(device), appears in state_dict) but explicitly
        # NOT as a learnable parameter — it has no gradient and is never
        # updated by the optimizer, since it's a fixed mathematical
        # function, not something learned from data.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) token embeddings.

        Returns:
            (batch, seq_len, d_model) - x with positional information
            added, same shape throughout.
        """
        seq_len = x.size(1)

        # Slice out only as many positions as the current sequence needs,
        # then add (not concatenate) to the token embeddings. Slicing
        # rather than indexing keeps this correct for any seq_len <= max_len.
        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)