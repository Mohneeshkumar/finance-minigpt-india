"""
Tests for src/layers/multi_head_attention.py

Expected interface (implement this in src/layers/multi_head_attention.py):

    class MultiHeadAttention(torch.nn.Module):
        def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
            \"\"\"Raises ValueError if d_model % num_heads != 0.\"\"\"

        def forward(
            self,
            query: torch.Tensor,   # (batch, seq_len_q, d_model)
            key: torch.Tensor,     # (batch, seq_len_k, d_model)
            value: torch.Tensor,   # (batch, seq_len_k, d_model)
            mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            \"\"\"Returns (output, attention_weights).
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, num_heads, seq_len_q, seq_len_k)
            \"\"\"

Run:
    pytest tests/test_multi_head_attention.py -v
"""

import pytest
import torch

from src.layers.multi_head_attention import MultiHeadAttention

torch.manual_seed(42)


def test_output_shape_matches_input():
    batch, seq_len, d_model, num_heads = 2, 7, 64, 8
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    x = torch.randn(batch, seq_len, d_model)
    output, attn_weights = mha(x, x, x)

    assert output.shape == (batch, seq_len, d_model)
    assert attn_weights.shape == (batch, num_heads, seq_len, seq_len)


def test_invalid_head_count_raises():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=65, num_heads=8)  # 65 % 8 != 0


def test_different_num_heads_still_produce_valid_shapes():
    d_model = 128
    for num_heads in (1, 2, 4, 8, 16):
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        x = torch.randn(3, 6, d_model)
        output, attn_weights = mha(x, x, x)
        assert output.shape == (3, 6, d_model)
        assert attn_weights.shape == (3, num_heads, 6, 6)


def test_cross_attention_shapes_differ_for_q_and_kv():
    """Query sequence length may differ from key/value sequence length
    (e.g. decoder cross-attending to encoder output)."""
    d_model, num_heads = 32, 4
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    query = torch.randn(2, 5, d_model)   # seq_len_q = 5
    key = torch.randn(2, 9, d_model)     # seq_len_k = 9
    value = torch.randn(2, 9, d_model)

    output, attn_weights = mha(query, key, value)

    assert output.shape == (2, 5, d_model)
    assert attn_weights.shape == (2, num_heads, 5, 9)


def test_attention_weights_sum_to_one_per_head():
    d_model, num_heads = 32, 4
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(2, 6, d_model)

    _, attn_weights = mha(x, x, x)
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_gradients_flow_through_all_parameters():
    """Every learnable projection (Q, K, V, output) must receive gradients."""
    d_model, num_heads = 16, 2
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(2, 4, d_model, requires_grad=True)

    output, _ = mha(x, x, x)
    loss = output.sum()
    loss.backward()

    for name, param in mha.named_parameters():
        assert param.grad is not None, f"No gradient reached parameter: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in parameter: {name}"
