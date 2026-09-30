"""
Add & Norm (residual connection + Layer Normalization)

Reference: Vaswani et al. (2017), "Attention Is All You Need", Section 3.1
(end of the section) and Section 5.4 (residual dropout).

Every sub-layer in the Transformer (multi-head attention, feed-forward)
is wrapped identically:

    output = LayerNorm(x + Dropout(Sublayer(x)))

Two separate ideas are fused here:

1. Residual connection (x + Sublayer(x)):
   Lets gradients flow directly around the sub-layer via the "+x" path
   during backpropagation — d(output)/dx includes a direct +1 term
   regardless of how complex Sublayer is. This is exactly the insight
   from ResNets (He et al., 2015), applied to Transformers: without it,
   stacking many blocks makes gradients vanish as they multiply through
   each sub-layer's Jacobian.

2. Layer Normalization (Ba et al., 2016):
   Normalizes each position's feature vector independently (across the
   d_model dimension) to zero mean and unit variance, then applies a
   learnable per-dimension scale (gamma) and shift (beta). This keeps
   activation magnitudes stable as they flow through many stacked
   layers, which is what makes deep Transformer stacks trainable at all.

Design choice — Post-LN (this implementation) vs Pre-LN:
  Post-LN (used here, matches the original paper exactly):
      output = LayerNorm(x + Sublayer(x))
  Pre-LN (used by GPT-2 onward, and most modern LLMs):
      output = x + Sublayer(LayerNorm(x))
  Pre-LN is generally easier to train at very large depth because the
  residual path itself is never normalized, keeping gradient flow more
  direct. This project uses Post-LN deliberately, to stay faithful to
  the original 2017 architecture as the P1 learning goal requires.
"""

import torch
import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self, d_model: int,dropout: float = 0.0):
        super().__init__()

        # nn.LayerNorm normalizes over the LAST dimension by default
        # (here, d_model) — i.e. independently for every (batch, position)
        # pair, unlike BatchNorm which normalizes across the batch
        # dimension. This is what makes LayerNorm suitable for variable-
        # length sequences: normalization doesn't depend on batch
        # statistics or sequence length at all.
        self.norm = nn.LayerNorm(d_model)

        # Dropout is applied to the Sublayer's output BEFORE the residual
        # add, per Section 5.4 of the paper ("We apply dropout to the output
        # of each sublayer, before it is added to the sub-layer input and normalized").
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:              (batch, seq_len, d_model) - the sub-layer's input
                            (before Sublayer was applied to it).
            sublayer_output:(batch, seq_len, d_model) - the result of applying 
                            Sublayer(x) (e.g. attention or FFN output), computed 
                            by caller.

        Returns:
            (batch, seq_len, d_model) - LayerNorm(x + Dropout(Sublayer_output))
        """
        # 1. Dropout on Sublayer's output - Regularization
        # 2. Residual add: x + ... - the gradient-highway connection.
        # 3. LayerNorm: renormalize the combined result.
        return self.norm(x + self.dropout(sublayer_output))

    