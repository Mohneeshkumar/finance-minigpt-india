"""
Tests for src/layers/scaled_dot_product.py

Expected interface (implement this in src/layers/scaled_dot_product.py):

    def scaled_dot_product_attention(
        query: torch.Tensor,   # (batch, ..., seq_len_q, d_k)
        key: torch.Tensor,     # (batch, ..., seq_len_k, d_k)
        value: torch.Tensor,   # (batch, ..., seq_len_k, d_v)
        mask: torch.Tensor | None = None,  # broadcastable to (batch, ..., seq_len_q, seq_len_k)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        \"\"\"Returns (output, attention_weights).

        output:            (batch, ..., seq_len_q, d_v)
        attention_weights: (batch, ..., seq_len_q, seq_len_k), rows sum to 1
        \"\"\"

Formula (Vaswani et al., 2017, Section 3.2.1):
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Run:
    pytest tests/test_scaled_dot_product.py -v
"""

import math

import pytest
import torch

from src.layers.scaled_dot_product import scaled_dot_product_attention

torch.manual_seed(42)


def test_output_shape_matches_value():
    batch, seq_len, d_k, d_v = 2, 5, 8, 8
    q = torch.randn(batch, seq_len, d_k)
    k = torch.randn(batch, seq_len, d_k)
    v = torch.randn(batch, seq_len, d_v)

    output, attn_weights = scaled_dot_product_attention(q, k, v)

    assert output.shape == (batch, seq_len, d_v)
    assert attn_weights.shape == (batch, seq_len, seq_len)


def test_attention_weights_sum_to_one():
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)

    _, attn_weights = scaled_dot_product_attention(q, k, v)

    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_scaling_reduces_variance_of_logits():
    """
    Sanity check for *why* scaling by sqrt(d_k) exists: without it, dot
    products grow with d_k and push softmax into saturated (near one-hot)
    regions. We don't test internals directly, but we can confirm attention
    weights aren't degenerate (all mass on a single position) for random,
    unit-scale inputs with a real d_k.
    """
    d_k = 64
    q = torch.randn(1, 10, d_k)
    k = torch.randn(1, 10, d_k)
    v = torch.randn(1, 10, d_k)

    _, attn_weights = scaled_dot_product_attention(q, k, v)

    max_weight_per_row = attn_weights.max(dim=-1).values
    # With proper scaling, attention shouldn't collapse to ~1.0 on one token
    # for random unit-variance inputs at this d_k.
    assert (max_weight_per_row < 0.9).all()


def test_mask_zeroes_out_masked_positions():
    batch, seq_len, d_k = 1, 4, 8
    q = torch.randn(batch, seq_len, d_k)
    k = torch.randn(batch, seq_len, d_k)
    v = torch.randn(batch, seq_len, d_k)

    # Mask out the last key/value position entirely (True = masked/blocked)
    mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool)
    mask[:, :, -1] = True

    _, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)

    assert torch.allclose(
        attn_weights[:, :, -1], torch.zeros_like(attn_weights[:, :, -1]), atol=1e-6
    )


def test_manual_calculation_matches_reference():
    """
    Small, fully worked example computed by hand logic (not hardcoded
    numbers) to cross-check the formula end-to-end.
    """
    q = torch.tensor([[[1.0, 0.0]]])  # (1, 1, 2)
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2)
    v = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])  # (1, 2, 2)

    d_k = q.shape[-1]
    expected_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    expected_weights = torch.softmax(expected_scores, dim=-1)
    expected_output = expected_weights @ v

    output, attn_weights = scaled_dot_product_attention(q, k, v)

    assert torch.allclose(output, expected_output, atol=1e-6)
    assert torch.allclose(attn_weights, expected_weights, atol=1e-6)


def test_batch_size_one_and_greater_than_one_are_consistent():
    """Guards against broadcasting bugs that only appear at batch_size > 1."""
    d_k = 8
    q1 = torch.randn(1, 3, d_k)
    k1 = torch.randn(1, 3, d_k)
    v1 = torch.randn(1, 3, d_k)

    out1, _ = scaled_dot_product_attention(q1, k1, v1)

    q_batch = q1.repeat(4, 1, 1)
    k_batch = k1.repeat(4, 1, 1)
    v_batch = v1.repeat(4, 1, 1)

    out_batch, _ = scaled_dot_product_attention(q_batch, k_batch, v_batch)

    for i in range(4):
        assert torch.allclose(out_batch[i], out1[0], atol=1e-6)
