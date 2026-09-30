"""
Multi-Head Attention

Reference: Vaswani et al. (2017), "Attention Is All You Need", Section 3.2.2.

Instead of performing a single attention pass with d_model-dimensional
Q,K,V , multi-head attention:
  1. Linearly projects Q, K, V into 'num_heads' smaller subspaces
     (each of size d_k = d_model / num_heads).

  2. Runs scaled dot product attention independently and in parallel
    within each subspace.

  3. Concatenates all heads' outputs back into a single d_model-sized
     vector and applies one final linear projection.

Why bother splitting into heads instead of using one big attention pass?
A single attention head is forced to average over all types of
relationships a token might have with others. Multiple smaller heads let
the model attend to different kinds of relationships simultaneously
(e.g. one head might specialize in local/adjacent-token patterns, another
in long-range dependencies) — this is an empirical finding from the paper,
not something enforced by the architecture itself; heads are free to learn
whatever the loss encourages.
"""

import torch
import torch.nn as nn

from src.layers.scaled_dot_product import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model :int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        # d_model must split evenly across heads, since every head gets
        # an equal sized slice (d_k) of the total representation
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
                
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimensionality of each head's subspace
                
        # Four learnable projections, exactly as in Figure 2 of the paper.
        # Each is a full (d_model -> d_model) linear layer — the "split
        # into heads" happens by *reshaping* this single projection's
        # output, not by having num_heads separate smaller Linear layers.
        # This is mathematically equivalent to per-head projections but
        # far more efficient as one batched matmul.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
                
        self.dropout = nn.Dropout(dropout)    

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)

        Reshapes the last dimension into (num_heads, d_k), then moves
        num_heads next to batch so that scaled_dot_product_attention can
        treat (batch, num_heads) together as a combined "batch" dimension
        and run every head's attention in parallel with one matmul.
        """
        batch_size , seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1,2)  # (batch, num_heads, seq_len, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)

        The exact inverse of _split_heads: move num_heads back next to 
        d_k, then collapse them into a single d_model-sized dimension.
        .contiguous() is required because .transpose() only changes
        strides/ view metadata - .view() afterward needs contiguous memory.
        """
        batch_size, _ , seq_len, _ = x.shape
        x = x.transpose(1,2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, seq_length_q, d_model)
            key: (batch, seq_length_k, d_model)
            value: (batch, seq_length_v, d_model)
            mask:  optional bool tensor broadcastable to
                   (batch, num_heads, seq_len_q, seq_len_k). True = blocked.
                   Commonly supplied as (batch, seq_len_q, seq_len_k) —
                   we insert a heads dimension of 1 so it broadcasts across
                   every head identically (see unsqueeze below).

        Returns:
            output:           (batch,seq_len_q,d_model)
            attention_weights:(batch,num_heads,seq_len_q,seq_len_k)
        """
        # 1. Project Q,K,V into d_model-dimensional space, then split 
        #    each into num_heads independent subspaces.
        q = self._split_heads(self.q_proj(query)) # (batch, heads, seq_q, d_k)
        k = self._split_heads(self.k_proj(key)) # (batch, heads, seq_k, d_k)
        v = self._split_heads(self.v_proj(value)) # (batch, heads, seq_k, d_k)

        # 2. If a mask was supplied without a heads dimension, add one so
        #    it broadcasts identically across all heads. A mask is a
        #    property of *positions* (which tokens may see which), not of
        #    individual heads, so every head must share the same mask.
        if mask is not None and mask.dim()==3 :
            mask = mask.unsqueeze(1) # (batch, 1, seq_q, seq_k) -> broadcasts over heads

        # 3. Run the Step 1 formula once — it operates identically whether
        #    or not there's a leading "heads" dimension, since it only
        #    ever looks at the last two dimensions.
        attn_output, attention_weights = scaled_dot_product_attention(q,k,v,mask)
        attn_output = self.dropout(attn_output)

        # 4. Concatenate heads back into a single d_model vector, then
        #    apply the final output projection (this lets the model mix
        #    information across heads, not just within each one).
        merged = self._merge_heads(attn_output)
        output = self.out_proj(merged)

        return output, attention_weights





