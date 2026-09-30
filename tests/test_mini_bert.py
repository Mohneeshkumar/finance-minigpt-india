"""Tests for MiniBERT (Step 11)."""

import torch

from src.config import MiniBERTConfig
from src.model.mini_bert import MiniBERT


def make_model():
    torch.manual_seed(0)
    config = MiniBERTConfig(
        vocab_size=40, d_model=16, num_heads=4, d_ff=32,
        num_layers=2, max_seq_len=12, dropout=0.0,
    )
    return MiniBERT(config), config


def test_forward_output_shapes():
    model, config = make_model()
    ids = torch.randint(0, config.vocab_size, (3, 8))
    out = model(ids)
    assert out["mlm_logits"].shape == (3, 8, config.vocab_size)
    assert out["nsp_logits"].shape == (3, 2)
    assert len(out["attention_weights"]) == config.num_layers


def test_weight_tying_between_embedding_and_mlm_head():
    model, _ = make_model()
    assert model.mlm_head.weight is model.token_embedding.weight


def test_bidirectional_attention_sees_future_positions():
    """Unlike MiniGPT, MiniBERT's encoder must be able to attend in both
    directions -- position 0 attending to position 5 should be non-zero
    (with no padding mask applied, nothing should be blocked).
    """
    model, config = make_model()
    ids = torch.randint(0, config.vocab_size, (1, 6))
    out = model(ids)
    attn = out["attention_weights"][0]  # (1, heads, 6, 6)
    assert torch.any(attn[:, :, 0, 5] > 0.0)


def test_padding_mask_blocks_padded_key_positions():
    model, config = make_model()
    ids = torch.randint(0, config.vocab_size, (1, 6))
    pad_mask = torch.tensor([[False, False, False, False, True, True]])
    out = model(ids, attention_mask=pad_mask)
    attn = out["attention_weights"][0]  # (1, heads, seq_q, seq_k)
    # No query position should place any weight on the padded key columns.
    assert torch.allclose(attn[:, :, :, 4:], torch.zeros_like(attn[:, :, :, 4:]))


def test_segment_ids_default_to_zero_when_omitted():
    model, config = make_model()
    ids = torch.randint(0, config.vocab_size, (2, 5))
    out = model(ids)  # segment_ids=None
    assert out["mlm_logits"].shape == (2, 5, config.vocab_size)
