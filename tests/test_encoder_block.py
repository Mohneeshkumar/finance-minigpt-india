"""
Tests for src/layers/feed_forward.py, src/layers/layer_norm_residual.py,
and src/blocks/encoder_block.py

Expected interfaces:

    class FeedForward(torch.nn.Module):
        def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0): ...
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            \"\"\"x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
            Two linear layers with ReLU in between (Section 3.3).\"\"\"

    class AddNorm(torch.nn.Module):
        def __init__(self, d_model: int, dropout: float = 0.0): ...
        def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
            \"\"\"Residual connection + LayerNorm. Document whether this is
            pre-LN or post-LN in the module docstring — this repo recommends
            pre-LN for training stability, which is the modern standard.\"\"\"

    class EncoderBlock(torch.nn.Module):
        def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0): ...
        def forward(
            self, x: torch.Tensor, mask: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor]:
            \"\"\"Returns (output, attention_weights).
            Composition: MHA -> AddNorm -> FeedForward -> AddNorm
            \"\"\"

Run:
    pytest tests/test_encoder_block.py -v
"""

import torch

from src.blocks.encoder_block import EncoderBlock
from src.layers.feed_forward import FeedForward
from src.layers.layer_norm_residual import AddNorm

torch.manual_seed(42)


def test_feed_forward_output_shape():
    batch, seq_len, d_model, d_ff = 2, 6, 32, 128
    ffn = FeedForward(d_model=d_model, d_ff=d_ff)

    x = torch.randn(batch, seq_len, d_model)
    output = ffn(x)

    assert output.shape == x.shape


def test_feed_forward_is_position_wise():
    """Each position should be transformed independently — shuffling the
    sequence dimension should shuffle the output identically."""
    d_model, d_ff = 16, 64
    ffn = FeedForward(d_model=d_model, d_ff=d_ff)
    ffn.eval()  # disable dropout for a deterministic check

    x = torch.randn(1, 5, d_model)
    perm = torch.randperm(5)

    out_original = ffn(x)
    out_shuffled_input = ffn(x[:, perm, :])

    assert torch.allclose(out_original[:, perm, :], out_shuffled_input, atol=1e-5)


def test_add_norm_preserves_shape():
    batch, seq_len, d_model = 2, 5, 32
    add_norm = AddNorm(d_model=d_model)

    x = torch.randn(batch, seq_len, d_model)
    sublayer_out = torch.randn(batch, seq_len, d_model)

    output = add_norm(x, sublayer_out)
    assert output.shape == x.shape


def test_add_norm_output_is_normalized():
    """After LayerNorm, each position's feature vector should have
    approximately zero mean and unit variance (LayerNorm's defining property)."""
    d_model = 32
    add_norm = AddNorm(d_model=d_model)
    add_norm.eval()

    x = torch.randn(1, 4, d_model)
    sublayer_out = torch.randn(1, 4, d_model)

    output = add_norm(x, sublayer_out)

    mean = output.mean(dim=-1)
    std = output.std(dim=-1, unbiased=False)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-3)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-1)


def test_encoder_block_output_shape():
    batch, seq_len, d_model, num_heads, d_ff = 2, 6, 64, 8, 256
    block = EncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    x = torch.randn(batch, seq_len, d_model)
    output, attn_weights = block(x)

    assert output.shape == x.shape
    assert attn_weights.shape == (batch, num_heads, seq_len, seq_len)


def test_encoder_block_respects_mask():
    batch, seq_len, d_model, num_heads, d_ff = 1, 5, 32, 4, 128
    block = EncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    x = torch.randn(batch, seq_len, d_model)
    mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool)
    mask[:, :, -1] = True  # block last position

    _, attn_weights = block(x, mask=mask)

    assert torch.allclose(
        attn_weights[..., -1], torch.zeros_like(attn_weights[..., -1]), atol=1e-6
    )


def test_encoder_block_gradients_flow():
    d_model, num_heads, d_ff = 16, 2, 64
    block = EncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)

    x = torch.randn(2, 4, d_model, requires_grad=True)
    output, _ = block(x)
    output.sum().backward()

    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient reached parameter: {name}"
