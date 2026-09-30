"""
Tests for src/masks/mask_utils.py

Expected interface (implement this in src/masks/mask_utils.py):

    def causal_mask(seq_len: int) -> torch.Tensor:
        \"\"\"Returns a (seq_len, seq_len) bool tensor where True marks
        positions that must be BLOCKED (future tokens), i.e. an
        upper-triangular matrix (excluding the diagonal) of True values.\"\"\"

    def padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        \"\"\"lengths: (batch,) actual sequence lengths.
        Returns a (batch, max_len) bool tensor where True marks PADDED
        (blocked) positions, i.e. indices >= lengths[i].\"\"\"

Convention used throughout this repo: mask value True == "block this
position" (added as -inf before softmax). This matches the convention
expected by scaled_dot_product_attention's mask argument.

Run:
    pytest tests/test_masks.py -v
"""

import torch

from src.masks.mask_utils import causal_mask, padding_mask

torch.manual_seed(42)


def test_causal_mask_shape():
    seq_len = 6
    mask = causal_mask(seq_len)
    assert mask.shape == (seq_len, seq_len)
    assert mask.dtype == torch.bool


def test_causal_mask_blocks_future_positions_only():
    seq_len = 5
    mask = causal_mask(seq_len)

    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                assert mask[i, j].item() is True, f"position ({i},{j}) should be blocked"
            else:
                assert mask[i, j].item() is False, f"position ({i},{j}) should be visible"


def test_causal_mask_diagonal_is_never_blocked():
    """A token must always be able to attend to itself."""
    seq_len = 8
    mask = causal_mask(seq_len)
    diag = torch.diagonal(mask)
    assert not diag.any()


def test_padding_mask_shape():
    lengths = torch.tensor([3, 5, 2])
    max_len = 5
    mask = padding_mask(lengths, max_len)
    assert mask.shape == (3, max_len)
    assert mask.dtype == torch.bool


def test_padding_mask_blocks_correct_positions():
    lengths = torch.tensor([3, 5, 2])
    max_len = 5
    mask = padding_mask(lengths, max_len)

    expected = torch.tensor(
        [
            [False, False, False, True, True],   # length 3 -> last 2 blocked
            [False, False, False, False, False],  # length 5 -> none blocked
            [False, False, True, True, True],     # length 2 -> last 3 blocked
        ]
    )
    assert torch.equal(mask, expected)


def test_padding_mask_full_length_blocks_nothing():
    lengths = torch.tensor([4])
    max_len = 4
    mask = padding_mask(lengths, max_len)
    assert not mask.any()
