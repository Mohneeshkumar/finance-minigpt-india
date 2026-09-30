"""Tests for MiniGPT (Step 10)."""

import torch

from src.config import MiniGPTConfig
from src.model.mini_gpt import MiniGPT


def make_model():
    torch.manual_seed(0)
    config = MiniGPTConfig(
        vocab_size=40, d_model=16, num_heads=4, d_ff=32,
        num_layers=2, max_seq_len=12, dropout=0.0,
    )
    return MiniGPT(config), config


def test_forward_output_shapes():
    model, config = make_model()
    ids = torch.randint(0, config.vocab_size, (3, 8))
    logits, attn = model(ids)
    assert logits.shape == (3, 8, config.vocab_size)
    assert len(attn) == config.num_layers
    assert attn[0].shape == (3, config.num_heads, 8, 8)


def test_weight_tying_between_embedding_and_lm_head():
    model, _ = make_model()
    assert model.lm_head.weight is model.token_embedding.weight


def test_causal_mask_blocks_future_positions_in_every_layer():
    model, config = make_model()
    ids = torch.randint(0, config.vocab_size, (1, 6))
    _, attn = model(ids)
    for layer_attn in attn:
        for i in range(6):
            for j in range(i + 1, 6):
                assert torch.all(layer_attn[:, :, i, j] == 0.0)


def test_generate_appends_requested_number_of_tokens():
    model, config = make_model()
    prompt = torch.randint(0, config.vocab_size, (2, 3))
    generated = model.generate(prompt, max_new_tokens=5)
    assert generated.shape == (2, 3 + 5)
    assert torch.equal(generated[:, :3], prompt)


def test_raises_on_sequence_longer_than_max_seq_len():
    model, config = make_model()
    ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len + 1))
    try:
        model(ids)
        assert False, "expected ValueError for oversized sequence"
    except ValueError:
        pass
