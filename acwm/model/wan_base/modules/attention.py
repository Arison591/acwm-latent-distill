# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn.functional as F

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
):
    """
    Standard attention implementation when Flash Attention is not available.
    """
    # q: [B, L, N, D]
    b, lq, n, d = q.shape
    lk = k.shape[1]
    
    if softmax_scale is None:
        softmax_scale = 1.0 / (d ** 0.5)

    # Transpose for batch matrix multiplication: [B, N, L, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # scores: [B, N, Lq, Lk]
    attn = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    
    if causal:
        mask = torch.triu(torch.ones(lq, lk, device=q.device), diagonal=1).bool()
        attn.masked_fill_(mask, float('-inf'))
        
    attn = torch.softmax(attn, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p)
    
    # out: [B, N, Lq, D] -> [B, Lq, N, D]
    out = torch.matmul(attn, v).transpose(1, 2)
    return out


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Wrapper for flash attention or standard attention fallback.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    # Fallback to standard attention if Flash Attention is not available or not on CUDA
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE) or q.device.type != 'cuda':
        return attention(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, causal, window_size).to(out_dtype)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q_flat = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q_flat = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k_flat = half(k.flatten(0, 1))
        v_flat = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k_flat = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v_flat = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q_flat = q_flat.to(v_flat.dtype)
    k_flat = k_flat.to(v_flat.dtype)

    if q_scale is not None:
        q_flat = q_flat * q_scale

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q_flat, k=k_flat, v=v_flat,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None, seqused_k=None,
            max_seqlen_q=lq, max_seqlen_k=lk,
            softmax_scale=softmax_scale, causal=causal, deterministic=deterministic
        )[0].unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q_flat, k=k_flat, v=v_flat,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq, max_seqlen_k=lk,
            dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
            window_size=window_size, deterministic=deterministic
        ).unflatten(0, (b, lq))
    else:
        # Final fallback just in case
        return attention(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, causal, window_size).to(out_dtype)

    return x.to(out_dtype)
