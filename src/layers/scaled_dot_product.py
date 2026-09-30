"""
Scaled Dot-Product Attention.
Reference: Vaswani et al. (2017), "Attention Is All You Need", Section 3.2.1

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
"""

import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
        query : torch.Tensor,
        key : torch.Tensor,
        value : torch.Tensor,
        mask : torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    query: (batch, ..., seq_len_q, d_k)
        key:   (batch, ..., seq_len_k, d_k)
        value: (batch, ..., seq_len_k, d_v)
        mask:  bool tensor, True = block this position (added as -inf pre-softmax)

    Returns:
        output:            (batch, ..., seq_len_q, d_v)
        attention_weights: (batch, ..., seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)

    # 1. Raw compatibility scores between every query and every key
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=query.dtype, device = query.device)
    )

    # 2. Block masked positions BEFORE softmax
    if mask is not None:
        fill_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask, fill_value)

    # 3. Normalize a probability distribution over keys
    attention_weights = F.softmax(scores, dim=-1)

    # 4. Weighted sum of values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights
    
