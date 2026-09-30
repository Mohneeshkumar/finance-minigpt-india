"""
End-to-end sanity check for the from-scratch attention implementation.

Runs a full forward pass through the stacked mini encoder (src/model/mini_encoder.py)
on random dummy input, confirms there are no shape errors, prints basic
attention weight statistics, and — if matplotlib is available — saves an
attention heatmap for visual inspection.

This is NOT a training script. There is no loss, no optimizer, no dataset.
Its only purpose is to confirm the architecture is wired together correctly
end-to-end after all `src/` modules have been implemented.

Usage:
    python scripts/run_sanity_check.py
"""

import sys
from pathlib import Path

import torch

# Allow running this script directly via `python scripts/run_sanity_check.py`
# from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.masks.mask_utils import padding_mask
from src.model.mini_gpt import MiniEncoder

SEED = 42
BATCH_SIZE = 2
SEQ_LEN = 12
D_MODEL = 64
NUM_HEADS = 8
D_FF = 256
NUM_LAYERS = 4
VOCAB_SIZE = 1000  # dummy vocab size for the input embedding lookup


def main() -> None:
    torch.manual_seed(SEED)

    print("=" * 60)
    print("Attention-from-Scratch — End-to-End Sanity Check")
    print("=" * 60)
    print(
        f"batch_size={BATCH_SIZE} seq_len={SEQ_LEN} d_model={D_MODEL} "
        f"num_heads={NUM_HEADS} d_ff={D_FF} num_layers={NUM_LAYERS}"
    )

    model = MiniEncoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=SEQ_LEN,
    )
    model.eval()

    # Dummy token IDs (as if already tokenized)
    dummy_input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    # Simulate variable-length sequences within the batch via padding mask
    lengths = torch.tensor([SEQ_LEN, SEQ_LEN - 3])
    pad_mask = padding_mask(lengths, SEQ_LEN)  # (batch, seq_len), True = padded

    # Expand padding mask to the (batch, seq_q, seq_k) shape expected by attention:
    # block attending TO padded key positions, for every query position.
    attn_mask = pad_mask.unsqueeze(1).expand(BATCH_SIZE, SEQ_LEN, SEQ_LEN)

    print("\nRunning forward pass...")
    with torch.no_grad():
        output, attn_weights_per_layer = model(dummy_input_ids, mask=attn_mask)

    print(f"Output shape:            {tuple(output.shape)}")
    print(f"Expected output shape:   {(BATCH_SIZE, SEQ_LEN, D_MODEL)}")
    assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL), "Shape mismatch in encoder output!"

    print(f"Num attention weight tensors returned: {len(attn_weights_per_layer)} (expected {NUM_LAYERS})")
    assert len(attn_weights_per_layer) == NUM_LAYERS

    last_layer_weights = attn_weights_per_layer[-1]
    print(f"Last layer attention weights shape: {tuple(last_layer_weights.shape)}")

    row_sums = last_layer_weights.sum(dim=-1)
    max_deviation = (row_sums - 1.0).abs().max().item()
    print(f"Max deviation of attention row-sums from 1.0: {max_deviation:.6f}")
    assert max_deviation < 1e-4, "Attention weights do not sum to 1 — check softmax/masking."

    padded_mass = last_layer_weights[0, :, :, -1].mean().item()
    print(f"Mean attention mass on a known-padded position (should be ~0): {padded_mass:.6f}")

    print("\nSanity check PASSED — no shape errors, attention weights well-formed.")

    _try_save_heatmap(last_layer_weights)


def _try_save_heatmap(attn_weights: torch.Tensor) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping heatmap; `pip install matplotlib` to enable)")
        return

    # Average over heads for a single, simple 2D heatmap of the first batch element
    weights_2d = attn_weights[0].mean(dim=0).numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(weights_2d, cmap="viridis")
    ax.set_title("Attention Weights (avg over heads) — Layer -1, Batch 0")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()

    output_path = Path(__file__).resolve().parent / "attention_heatmap.png"
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved attention heatmap to: {output_path}")


if __name__ == "__main__":
    main()
