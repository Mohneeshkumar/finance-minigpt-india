"""
Position-wise Feed-Forward Network.
Reference: Vaswani et al. (2017), Section 3.3

    FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
"""

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model :int, d_ff :int, dropout: float = 0.0):
        super().__init__()

        # Expand d_model -> d_ff (typically 4x wider) to give the 
        # non-linearity room to represent complex per-token transformers.
        self.linear_1 = nn.Linear(d_model, d_ff)

        # ReLU is the activation layer
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        # Compressing d_ff back to d_model so, the output can be added 
        # residually  in the next step (Add & Norm)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) - same shape, transformed
            independentlt at each postion.
        """
        # nn.Linear only ever operates on the last dimension, broadcasting
        # across (batch, seq_len) automatically - this IS the 
        # "position-wise" property: no mixing across seq_len anywhere here.
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
    