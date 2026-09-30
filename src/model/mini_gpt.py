"""
MiniGPT -- decoder-only causal language model (Step 10).

Reference: Radford et al. (2018/2019), "Improving/Language Models are
Unsupervised Multitask Learners" (GPT / GPT-2), built on top of the
DecoderBlock (Step 8) which already implements the GPT-style 2-sublayer
block (masked self-attention + FFN, no cross-attention).

Pipeline:
    token ids --(token embedding + positional encoding)--> x
              --(N x DecoderBlock, causal mask)-->        x
              --(linear LM head, tied or untied)-->       logits over vocab

This module owns:
  - the token embedding table
  - adding positional encoding (Step 4) to it
  - constructing the causal mask (Step 2, now correctly named) ONCE per
    forward pass and passing it to every DecoderBlock
  - the final linear projection to vocabulary logits (the "LM head")

It deliberately does NOT own: the tokenizer (Step 9, used only to produce
ids before this model sees them), the loss function, or the optimizer --
those live in train_gpt.py (Step 12), keeping Model and Training layers
separate per the project's layering standard.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.blocks.decoder_block import DecoderBlock
from src.layers.positional_encoding import PositionalEncoding
from src.masks.mask_utils import causal_mask
from src.config import MiniGPTConfig


class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config

        # Token embedding: maps a vocabulary index to a d_model-dim vector.
        # padding_idx tells nn.Embedding to zero out (and never update via
        # backprop) the row for <PAD>, so padding contributes no gradient
        # signal to the embedding table.
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
 
        # Scaling the embedding by sqrt(d_model) is the convention from the
        # original paper (Section 3.4): it keeps the embedding's variance
        # comparable to the positional encoding's before they are summed.
        self.embedding_scale = math.sqrt(config.d_model)

        self.positional_encoding = PositionalEncoding(
            d_model=config.d_model, max_len=config.max_seq_len
        )
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Stack of N identical DecoderBlocks. nn.ModuleList (not a plain
        # Python list) so PyTorch tracks each block's parameters for
        # .parameters(), .to(device), state_dict(), etc.
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Final LayerNorm before the LM head. Not present in the original
        # 2017 paper (each sub-layer's Post-LN already normalizes), but
        # GPT-2 adds one final norm after the last block for training
        # stability at depth. Included here as a documented, deliberate
        # GPT-2-style addition on top of the Step 6-8 primitives.
        self.final_norm = nn.LayerNorm(config.d_model)

        # LM head: projects back from d_model to vocab_size logits.
        # Weight tying (sharing token_embedding.weight with lm_head.weight)
        # is a common GPT-2 trick that halves embedding-related parameters
        # and often improves quality; done explicitly below for clarity.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_ids: (batch, seq_len) long tensor of token ids.

        Returns:
            logits:             (batch, seq_len, vocab_size) -- unnormalized
                                 scores for "what is the next token" at
                                 every position.
            attention_weights:  list of length num_layers, each
                                 (batch, num_heads, seq_len, seq_len) --
                                 kept for the visualization step (Step 14).
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len "
                f"({self.config.max_seq_len}) the positional table supports."
            )

        x = self.token_embedding(input_ids) * self.embedding_scale
        x = self.positional_encoding(x)
        x = self.embedding_dropout(x)

        # Build the causal mask once per forward pass and reuse it across
        # every layer -- it depends only on seq_len, not on layer index.
        mask = causal_mask(seq_len).to(input_ids.device)

        attention_weights_per_layer = []
        for block in self.decoder_blocks:
            x, attn_weights = block(x, causal_mask=mask)
            attention_weights_per_layer.append(attn_weights)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, attention_weights_per_layer

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy/temperature-sampled autoregressive generation.

        Args:
            input_ids:      (batch, seq_len) prompt token ids.
            max_new_tokens: how many tokens to generate beyond the prompt.
            temperature:    softmax temperature; 1.0 = unmodified
                            distribution, <1.0 sharpens it (more greedy),
                            >1.0 flattens it (more random).
            eos_token_id:   if given, generation for a sequence stops
                            (further tokens are not appended for that
                            sequence) once it produces this id. For
                            simplicity, this implementation stops the
                            whole batch once ALL sequences have emitted it,
                            since MiniGPT has no per-sequence early-exit
                            bookkeeping -- adequate for the toy scale here.

        Returns:
            (batch, seq_len + generated_len) token ids.
        """
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            # Truncate to the last max_seq_len tokens so the positional
            # table never runs out of room as the sequence grows.
            context = generated[:, -self.config.max_seq_len :]
            logits, _ = self.forward(context)

            # Only the last position's logits matter for "what comes next".
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
