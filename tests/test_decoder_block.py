"""Tests for DecoderBlock (Step 8) -- GPT-style, 2 sub-layers, no
cross-attention. Mirrors the structure of test_encoder_block.py, plus one
test unique to decoders: causality enforcement (no leakage from future
positions).
"""

import torch

from src.blocks.decoder_block import DecoderBlock
from src.masks.mask_utils import causal_mask


def make_block(d_model=16, num_heads=4, d_ff=32, dropout=0.0):
    torch.manual_seed(0)
    return DecoderBlock(d_model, num_heads, d_ff, dropout=dropout)


def test_output_shape():
    block = make_block()
    x = torch.randn(2, 5, 16)
    mask = causal_mask(5)
    output, attn = block(x, causal_mask=mask)
    assert output.shape == (2, 5, 16)
    assert attn.shape == (2, 4, 5, 5)


def test_attention_weights_sum_to_one():
    block = make_block()
    x = torch.randn(2, 5, 16)
    mask = causal_mask(5)
    _, attn = block(x, causal_mask=mask)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_causality_no_future_leakage():
    """The defining property of a decoder block: query position i must
    have exactly zero attention weight on any key position j > i, for
    every head and every batch element.
    """
    block = make_block()
    seq_len = 6
    x = torch.randn(1, seq_len, 16)
    mask = causal_mask(seq_len)
    _, attn = block(x, causal_mask=mask)  # (1, heads, seq_len, seq_len)

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert torch.all(attn[:, :, i, j] == 0.0), (
                f"position {i} attends to future position {j}"
            )


def test_gradient_flows_through_all_submodules():
    block = make_block()
    x = torch.randn(2, 5, 16, requires_grad=True)
    mask = causal_mask(5)
    output, _ = block(x, causal_mask=mask)
    output.sum().backward()

    assert x.grad is not None and not torch.all(x.grad == 0)
    for name, param in block.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"


def test_add_norm_instances_are_independent():
    block = make_block()
    assert block.add_norm_1 is not block.add_norm_2
    assert block.add_norm_1.norm.weight.data_ptr() != block.add_norm_2.norm.weight.data_ptr()
