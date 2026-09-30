"""
MiniBERT -- bidirectional encoder with MLM + NSP pretraining heads (Step 11).

Reference: Devlin et al. (2018), "BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding". Built on top of EncoderBlock
(Step 7), which already implements bidirectional self-attention (no causal
mask -- every position can attend to every other real, non-padding
position).

Pipeline:
    token ids --(token + segment + positional embeddings)--> x
              --(N x EncoderBlock, padding mask only)-->      x
              --+-- MLM head:  Linear(d_model -> vocab_size), applied at
              |                every position, predicts the original token
              |                under masked-language-modeling.
              +-- NSP head:   Linear(d_model -> 2), applied ONLY to the
                               [CLS] position (index 0), predicts whether
                               sentence B follows sentence A.

Segment embeddings (a second, small embedding table indexed by 0 or 1)
tell the model which of the two concatenated sentences a token belongs to
-- this is what makes NSP possible; it is additional to, not a
replacement for, token + positional embeddings.

This module owns embeddings, the encoder stack, and both heads. It does
NOT own the MLM masking policy (deciding which 15% of tokens to corrupt,
and how) or the NSP pair-sampling policy (deciding which two sentences to
concatenate and whether B truly follows A) -- those are data-preparation
concerns that belong in train_bert.py (Step 12) / a dataset class, kept
separate from the model definition itself.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.blocks.encoder_block import EncoderBlock
from src.layers.positional_encoding import PositionalEncoding
from src.config import MiniBERTConfig


class MiniBERT(nn.Module):
    def __init__(self, config: MiniBERTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.embedding_scale = math.sqrt(config.d_model)

        # Segment embedding: 2 rows (sentence A = 0, sentence B = 1).
        # Added to the token embedding just like positional encoding is,
        # so the model can distinguish "same sentence" vs "different
        # sentence" tokens for the NSP objective.
        self.segment_embedding = nn.Embedding(config.type_vocab_size, config.d_model)

        self.positional_encoding = PositionalEncoding(
            d_model=config.d_model, max_len=config.max_seq_len
        )
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(config.d_model)

        # MLM head: predicts the original (pre-masking) token id at every
        # position. Tied to the token embedding weights, same rationale as
        # MiniGPT's LM head (Step 10).
        self.mlm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mlm_head.weight = self.token_embedding.weight

        # NSP head: binary classifier (IsNext / NotNext) applied to the
        # pooled [CLS] representation only.
        self.nsp_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Linear(config.d_model, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids:      (batch, seq_len) long tensor of token ids.
                             Position 0 is conventionally [CLS].
            segment_ids:    (batch, seq_len) long tensor of 0/1 values
                             marking sentence A vs sentence B. If None,
                             everything is treated as segment 0 (a single
                             sentence, no NSP pairing).
            attention_mask: optional (batch, seq_len) bool padding mask,
                             True = position is padding (block it). Passed
                             through to EncoderBlock, broadcast internally
                             the same way the multi-head attention layer
                             already broadcasts masks (Step 3).

        Returns a dict with:
            mlm_logits: (batch, seq_len, vocab_size)
            nsp_logits: (batch, 2)
            attention_weights: list[num_layers] of
                                (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_seq_len "
                f"({self.config.max_seq_len}) the positional table supports."
            )

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        x = self.token_embedding(input_ids) * self.embedding_scale
        x = x + self.segment_embedding(segment_ids)
        x = self.positional_encoding(x)
        x = self.embedding_dropout(x)

        # EncoderBlock's mask is broadcast against (batch, num_heads,
        # seq_len_q, seq_len_k) inside MultiHeadAttention; a (batch,
        # seq_len) padding mask needs an extra dim so it reads as "block
        # this KEY position for every query", not the other way around.
        mask = attention_mask.unsqueeze(1) if attention_mask is not None else None

        attention_weights_per_layer = []
        for block in self.encoder_blocks:
            x, attn_weights = block(x, mask=mask)
            attention_weights_per_layer.append(attn_weights)

        x = self.final_norm(x)

        mlm_logits = self.mlm_head(x)

        # [CLS] is always position 0 by BERT convention -- pool it for NSP.
        cls_representation = x[:, 0, :]
        nsp_logits = self.nsp_head(cls_representation)

        return {
            "mlm_logits": mlm_logits,
            "nsp_logits": nsp_logits,
            "attention_weights": attention_weights_per_layer,
        }
