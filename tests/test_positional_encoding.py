"""Regression test for PositionalEncoding (Step 4).

A prior on-disk revision of this file had `forward` indented at module
level (outside the class body), which meant PositionalEncoding() had no
working forward method at all -- calling it raised NotImplementedError
immediately. This suite exercises the actual forward() call explicitly so
that class of bug cannot silently return (a shape-only or attribute-only
test would not have caught it).
"""

import torch

from src.layers.positional_encoding import PositionalEncoding


def test_forward_is_callable_and_returns_correct_shape():
    pe = PositionalEncoding(d_model=16, max_len=32, dropout=0.0)
    x = torch.zeros(2, 10, 16)
    output = pe(x)  # must not raise NotImplementedError
    assert output.shape == (2, 10, 16)


def test_encoding_is_added_not_concatenated():
    pe = PositionalEncoding(d_model=8, max_len=16, dropout=0.0)
    x = torch.zeros(1, 4, 8)
    output = pe(x)
    # With a zero input, the output must equal the raw positional table
    # (dimensionality unchanged, values are exactly PE(pos)).
    assert torch.allclose(output, pe.pe[:, :4, :])


def test_different_positions_get_different_encodings():
    pe = PositionalEncoding(d_model=16, max_len=32, dropout=0.0)
    x = torch.zeros(1, 5, 16)
    output = pe(x)
    assert not torch.allclose(output[:, 0, :], output[:, 1, :])


def test_seq_len_shorter_than_max_len_slices_correctly():
    pe = PositionalEncoding(d_model=8, max_len=100, dropout=0.0)
    x = torch.zeros(1, 7, 8)
    output = pe(x)
    assert output.shape == (1, 7, 8)
