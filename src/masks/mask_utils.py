"""
Mask utilities for attention

Two masking types: 1. Casual masking, 2. Padding Masking
"""

import torch

def causal_mask(seq_len: int) -> torch.Tensor:
    """Builds a causal (look-ahead) mask so position i cannot attend to
    any position j > i — i.e. the model can't "see the future" when
    predicting the next token.

    Args:
        seq_len: length of the sequence.

    Returns:
        (seq_len, seq_len) bool tensor. True at (i, j) means "block query
        position i from attending to key position j." The diagonal (i == j)
        is always False — a token can always attend to itself.

    Example (seq_len=4):
        False  True   True   True    <- position 0 can only see position 0
        False  False  True   True    <- position 1 can see positions 0,1
        False  False  False  True    <- position 2 can see positions 0,1,2
        False  False  False  False   <- position 3 can see everything
    """

    mask = torch.triu(
        torch.ones(seq_len,seq_len,dtype=torch.bool),
        diagonal=1,
    )
    return mask

def padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Builds a padding mask marking which positions in each sequence are
    padding (and therefore must be excluded from attention).

    Args:
        lengths: (batch,) tensor of actual (unpadded) sequence lengths.
        max_len: the length every sequence has been padded to.

    Returns:
        (batch, max_len) bool tensor. True at (b, i) means "position i in
        batch element b is padding — block it."

    Example:
        lengths = [3, 5, 2], max_len = 5
        ->
        [False, False, False, True,  True ]   # length 3: last 2 are padding
        [False, False, False, False, False]   # length 5: nothing is padding
        [False, False, True,  True,  True ]   # length 2: last 3 are padding
    """
    position_indices = torch.arange(max_len, device = lengths.device)
    mask = position_indices.unsqueeze(0) >= lengths.unsqueeze(1)
    return mask


