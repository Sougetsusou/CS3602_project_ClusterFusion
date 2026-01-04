"""Mini FlashAttention (forward-only) reproduction in pure PyTorch.

Implements the FlashAttention *algorithmic idea*:
- block-wise attention
- online softmax (running max + running normalization)
- avoid materializing full (T x T) attention matrix

This is forward-only and not fused, so it will be much slower than flash-attn2.

Important detail (matches flash-attn doc):
- causal mask is aligned to the *bottom-right* of the attention matrix when
  q_len != k_len.
  Keep condition: j <= i + (k_len - q_len)

Shapes:
- q: (B, H, q_len, D)
- k,v: (B, H, k_len, D)

References:
- Dao et al., FlashAttention (NeurIPS 2022)
- flash_attn.flash_attn_interface.flash_attn_func docstring
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    scale: Optional[float] = None,
    additive_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Naive reference attention (materializes scores).

    q: (B,H,q_len,D), k,v: (B,H,k_len,D)
    additive_mask: broadcastable to (B,H,q_len,k_len) with 0 allowed and -inf/neg masked.

    Causal behavior matches flash-attn: bottom-right aligned when q_len != k_len.
    """
    b, h, q_len, d = q.shape
    k_len = k.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B,H,Q,K)

    if additive_mask is not None:
        scores = scores + additive_mask

    if causal:
        # bottom-right aligned causal mask
        offset = k_len - q_len
        qi = torch.arange(q_len, device=q.device)
        kj = torch.arange(k_len, device=q.device)
        causal_mask = kj[None, :] > (qi[:, None] + offset)
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


def attention_mini_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    q_block: int = 128,
    kv_block: int = 128,
    scale: Optional[float] = None,
    additive_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FlashAttention-style attention using online softmax + blocking.

    Args:
        q: (B,H,q_len,D)
        k,v: (B,H,k_len,D)
        causal: apply causal mask unless additive_mask is provided
        additive_mask: (B,1,q_len,k_len) or (B,H,q_len,k_len)

    Returns:
        out: (B,H,q_len,D)
    """
    b, h, q_len, d = q.shape
    k_len = k.shape[2]

    if scale is None:
        scale = 1.0 / math.sqrt(d)

    out = torch.empty((b, h, q_len, d), device=q.device, dtype=q.dtype)

    if additive_mask is not None:
        if additive_mask.dim() != 4:
            raise ValueError(f"additive_mask must be 4D, got {tuple(additive_mask.shape)}")
        if additive_mask.shape[0] != b or additive_mask.shape[-2] != q_len or additive_mask.shape[-1] != k_len:
            raise ValueError(
                f"additive_mask mismatch expected (B,*,Q,K)=({b},*,{q_len},{k_len}) got {tuple(additive_mask.shape)}"
            )

    # For bottom-right aligned causal masking
    offset = k_len - q_len

    for qs in range(0, q_len, q_block):
        qe = min(qs + q_block, q_len)
        q_blk = q[:, :, qs:qe, :]
        qb = qe - qs

        m = torch.full((b, h, qb), float("-inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((b, h, qb), device=q.device, dtype=torch.float32)
        o = torch.zeros((b, h, qb, d), device=q.device, dtype=torch.float32)

        for ks in range(0, k_len, kv_block):
            ke = min(ks + kv_block, k_len)
            k_blk = k[:, :, ks:ke, :]
            v_blk = v[:, :, ks:ke, :]

            s = torch.matmul(q_blk.to(torch.float32), k_blk.transpose(-1, -2).to(torch.float32)) * scale

            if additive_mask is not None:
                m_blk = additive_mask[:, :, qs:qe, ks:ke]
                if m_blk.shape[1] == 1:
                    s = s + m_blk
                else:
                    s = s + m_blk
            elif causal:
                # bottom-right aligned causal mask
                qi = torch.arange(qs, qe, device=q.device)
                kj = torch.arange(ks, ke, device=q.device)
                causal_mask = kj[None, :] > (qi[:, None] + offset)
                s = s.masked_fill(causal_mask[None, None, :, :], float("-inf"))

            m_ij = torch.amax(s, dim=-1)
            m_new = torch.maximum(m, m_ij)

            p = torch.exp(s - m_new[..., None])
            alpha = torch.exp(m - m_new)
            l = l * alpha + p.sum(dim=-1)
            o = o * alpha[..., None] + torch.matmul(p, v_blk.to(torch.float32))
            m = m_new

        o = o / l[..., None].clamp_min(1e-9)
        out[:, :, qs:qe, :] = o.to(q.dtype)

    return out
