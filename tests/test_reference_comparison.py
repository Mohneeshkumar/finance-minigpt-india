"""
The single most important test in this repo.

Loads IDENTICAL weights into:
  (a) our from-scratch MultiHeadAttention (src/layers/multi_head_attention.py)
  (b) torch.nn.MultiheadAttention (the reference/oracle implementation)

and asserts the outputs match within floating-point tolerance. Passing this
test is strong evidence the from-scratch implementation is mathematically
correct, not just shape-correct.

Notes on aligning the two implementations:
  - torch.nn.MultiheadAttention defaults to batch_first=False; we pass
    batch_first=True to match our own (batch, seq, d_model) convention.
  - torch.nn.MultiheadAttention stores Q/K/V projections as a single
    combined in_proj_weight/in_proj_bias of shape (3*d_model, d_model).
    We split it into three (d_model, d_model) chunks and copy them into
    our own q_proj/k_proj/v_proj weights.
  - Both implementations' output projection is a single (d_model, d_model)
    Linear layer — copy directly.
  - This test assumes your MultiHeadAttention module exposes q_proj, k_proj,
    v_proj, and out_proj as nn.Linear attributes. If you named them
    differently, update the attribute names below (not the test's intent).

Run:
    pytest tests/test_reference_comparison.py -v
"""

import torch
import torch.nn as nn

from src.layers.multi_head_attention import MultiHeadAttention

torch.manual_seed(42)


def _copy_weights_into_custom_mha(reference: nn.MultiheadAttention, custom: MultiHeadAttention, d_model: int):
    with torch.no_grad():
        w_q, w_k, w_v = reference.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = reference.in_proj_bias.chunk(3, dim=0)

        custom.q_proj.weight.copy_(w_q)
        custom.q_proj.bias.copy_(b_q)
        custom.k_proj.weight.copy_(w_k)
        custom.k_proj.bias.copy_(b_k)
        custom.v_proj.weight.copy_(w_v)
        custom.v_proj.bias.copy_(b_v)

        custom.out_proj.weight.copy_(reference.out_proj.weight)
        custom.out_proj.bias.copy_(reference.out_proj.bias)


def test_matches_pytorch_reference_no_mask():
    d_model, num_heads, seq_len, batch = 64, 8, 10, 2

    reference = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=num_heads, batch_first=True
    )
    reference.eval()

    custom = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    custom.eval()

    _copy_weights_into_custom_mha(reference, custom, d_model)

    x = torch.randn(batch, seq_len, d_model)

    with torch.no_grad():
        ref_output, ref_weights = reference(x, x, x, need_weights=True, average_attn_weights=True)
        custom_output, custom_weights = custom(x, x, x)

    assert torch.allclose(ref_output, custom_output, atol=1e-4), (
        "Custom MultiHeadAttention output diverges from torch.nn.MultiheadAttention "
        "reference. Check projection math, scaling factor, and head split/merge order."
    )

    # custom_weights is (batch, num_heads, seq, seq); reference averages over heads
    custom_avg_weights = custom_weights.mean(dim=1)
    assert torch.allclose(ref_weights, custom_avg_weights, atol=1e-4)


def test_matches_pytorch_reference_with_causal_mask():
    d_model, num_heads, seq_len, batch = 32, 4, 6, 1

    reference = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=num_heads, batch_first=True
    )
    reference.eval()

    custom = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    custom.eval()

    _copy_weights_into_custom_mha(reference, custom, d_model)

    x = torch.randn(batch, seq_len, d_model)

    # PyTorch's attn_mask: True (or -inf) means "block this position"
    causal_bool_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    with torch.no_grad():
        ref_output, _ = reference(
            x, x, x, attn_mask=causal_bool_mask, need_weights=False
        )
        # Broadcast the (seq, seq) mask to our expected (batch, seq, seq) shape
        custom_mask = causal_bool_mask.unsqueeze(0).expand(batch, -1, -1)
        custom_output, _ = custom(x, x, x, mask=custom_mask)

    assert torch.allclose(ref_output, custom_output, atol=1e-4), (
        "Custom MultiHeadAttention with causal masking diverges from the "
        "PyTorch reference. Check that masked logits are set to -inf BEFORE "
        "softmax, not zeroed out after."
    )
