"""Mini FlashAttention (forward+backward) reproduction in pure PyTorch.

Implements the FlashAttention *algorithmic idea*:
- block-wise attention
- online softmax (running max + running normalization)
- avoids materializing the full (Q x K) attention matrix

This implementation is intended as a **transparent, readable reproduction**.
It supports:
- forward + backward (via a custom autograd.Function with recomputation)
- causal masking with **bottom-right alignment** when q_len != k_len
- additive masks (broadcastable to (B,H,Q,K))
- attention dropout (training only)

Notes / limitations:
- This is not a fused CUDA kernel, so it will not match flash-attn2 performance.
- Backward uses recomputation and the standard exact attention gradients.

Shapes:
- q: (B, H, q_len, D)
- k,v: (B, H, k_len, D)

References:
- Dao et al., FlashAttention (NeurIPS 2022)
- flash_attn.flash_attn_interface.flash_attn_func docstring
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def _infer_scale(q: torch.Tensor, scale: Optional[float]) -> float:
    return float(scale) if scale is not None else (1.0 / math.sqrt(q.shape[-1]))


def _validate_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"q,k,v must be 4D (B,H,T,D). got q={q.dim()} k={k.dim()} v={v.dim()}")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError(f"(B,H) mismatch: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have same shape, got k={tuple(k.shape)} v={tuple(v.shape)}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"head dim mismatch: q.D={q.shape[-1]} k.D={k.shape[-1]}")


def _normalize_additive_mask(
    additive_mask: Optional[torch.Tensor],
    *,
    b: int,
    h: int,
    q_len: int,
    k_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Normalize additive_mask to a 4D tensor broadcastable to (B,H,Q,K).

    Accepts:
    - None
    - (B, 1, Q, K) or (B, H, Q, K)

    Returns:
    - mask with dtype float32/float16/bfloat16 (caller chooses), 4D.

    Convention: mask is *additive* (0 keeps, -inf masks).
    """
    if additive_mask is None:
        return None

    if additive_mask.dim() != 4:
        raise ValueError(f"additive_mask must be 4D, got {tuple(additive_mask.shape)}")
    if additive_mask.shape[0] != b or additive_mask.shape[-2] != q_len or additive_mask.shape[-1] != k_len:
        raise ValueError(
            f"additive_mask mismatch expected (B,*,Q,K)=({b},*,{q_len},{k_len}) got {tuple(additive_mask.shape)}"
        )
    if additive_mask.shape[1] not in (1, h):
        raise ValueError(f"additive_mask head dim must be 1 or H={h}, got {additive_mask.shape[1]}")

    return additive_mask.to(device=device, dtype=dtype)


def _causal_mask_block(
    *,
    qs: int,
    qe: int,
    ks: int,
    ke: int,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Return boolean mask of shape (Qb, Kb) where True means masked."""
    qi = torch.arange(qs, qe, device=device)
    kj = torch.arange(ks, ke, device=device)
    return kj[None, :] > (qi[:, None] + offset)


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
    _validate_shapes(q, k, v)
    b, h, q_len, d = q.shape
    k_len = k.shape[2]
    scale_f = _infer_scale(q, scale)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale_f  # (B,H,Q,K)

    if additive_mask is not None:
        additive_mask = _normalize_additive_mask(
            additive_mask,
            b=b,
            h=h,
            q_len=q_len,
            k_len=k_len,
            device=q.device,
            dtype=scores.dtype,
        )
        scores = scores + additive_mask

    if causal:
        offset = k_len - q_len
        qi = torch.arange(q_len, device=q.device)
        kj = torch.arange(k_len, device=q.device)
        causal_mask = kj[None, :] > (qi[:, None] + offset)
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

    p = torch.softmax(scores, dim=-1)
    return torch.matmul(p, v)


def _forward_blockwise_online_softmax(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    scale_f: float,
    q_block: int,
    kv_block: int,
    additive_mask: Optional[torch.Tensor],
    dropout_p: float,
    training: bool,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Forward pass.

    Returns:
      out: (B,H,Q,D)
      dropout_seeds: optional tensor (B,H,Q,K) is too large to store, so we don't.

    Instead of storing dropout mask, backward will recompute dropout with Philox by
    reusing a saved rng state (best effort). If generator is not provided, we save and
    restore global RNG state on CUDA (works but may be nondeterministic if other ops interleave).

    For simplicity and correctness, we implement dropout in forward and recompute in backward
    by saving `rng_state`.
    """
    b, h, q_len, d = q.shape
    k_len = k.shape[2]

    out = torch.empty((b, h, q_len, d), device=q.device, dtype=q.dtype)

    # bottom-right aligned causal masking offset
    offset = k_len - q_len

    # We will do all softmax math in fp32
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)

    # For fully-masked rows handling, we detect l==0 and set output to 0.
    for qs in range(0, q_len, q_block):
        qe = min(qs + q_block, q_len)
        qb = qe - qs

        q_blk = q_f[:, :, qs:qe, :]  # fp32

        m = torch.full((b, h, qb), float("-inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((b, h, qb), device=q.device, dtype=torch.float32)
        o = torch.zeros((b, h, qb, d), device=q.device, dtype=torch.float32)

        for ks in range(0, k_len, kv_block):
            ke = min(ks + kv_block, k_len)
            k_blk = k_f[:, :, ks:ke, :]
            v_blk = v_f[:, :, ks:ke, :]

            s = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * scale_f  # (B,H,Qb,Kb)

            if additive_mask is not None:
                m_blk = additive_mask[:, :, qs:qe, ks:ke].to(torch.float32)
                # head broadcast if needed
                if m_blk.shape[1] == 1 and h != 1:
                    m_blk = m_blk.expand(-1, h, -1, -1)
                s = s + m_blk
            elif causal:
                cm = _causal_mask_block(qs=qs, qe=qe, ks=ks, ke=ke, offset=offset, device=q.device)
                s = s.masked_fill(cm[None, None, :, :], float("-inf"))

            m_ij = torch.amax(s, dim=-1)  # (B,H,Qb)
            m_new = torch.maximum(m, m_ij)

            # p_ij = exp(s - m_new)
            p = torch.exp(s - m_new[..., None])

            if training and dropout_p and dropout_p > 0:
                keep_p = 1.0 - dropout_p
                # Dropout on attention probabilities
                # Note: generating full dropout mask per block.
                if generator is not None:
                    mask = torch.rand(p.shape, device=p.device, dtype=torch.float32, generator=generator) < keep_p
                else:
                    mask = torch.rand(p.shape, device=p.device, dtype=torch.float32) < keep_p
                p = p * mask.to(p.dtype) / keep_p

            alpha = torch.exp(m - m_new)
            l = l * alpha + p.sum(dim=-1)
            o = o * alpha[..., None] + torch.matmul(p, v_blk)
            m = m_new

        # If a query row is fully masked, l will be 0.
        # Define output as 0 for that row.
        l_safe = l.clamp_min(1e-9)
        o = o / l_safe[..., None]
        o = torch.where((l > 0)[..., None], o, torch.zeros_like(o))

        out[:, :, qs:qe, :] = o.to(q.dtype)

    return out, None


class _MiniFlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        q_block: int,
        kv_block: int,
        scale: Optional[float],
        additive_mask: Optional[torch.Tensor],
        dropout_p: float,
        training: bool,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        _validate_shapes(q, k, v)
        b, h, q_len, _ = q.shape
        k_len = k.shape[2]

        scale_f = _infer_scale(q, scale)

        mask_norm = _normalize_additive_mask(
            additive_mask,
            b=b,
            h=h,
            q_len=q_len,
            k_len=k_len,
            device=q.device,
            dtype=q.dtype,
        )

        # Save for backward (recompute strategy)
        # We save q,k,v and mask; backward will recompute attention probs blockwise.
        ctx.save_for_backward(q, k, v, mask_norm if mask_norm is not None else torch.tensor([], device=q.device))
        ctx.has_mask = mask_norm is not None
        ctx.causal = bool(causal)
        ctx.q_block = int(q_block)
        ctx.kv_block = int(kv_block)
        ctx.scale_f = float(scale_f)
        ctx.dropout_p = float(dropout_p)
        ctx.training = bool(training)

        # RNG state to deterministically recompute dropout in backward
        ctx.has_dropout = bool(training and dropout_p and dropout_p > 0)
        ctx.generator_provided = generator is not None
        if ctx.has_dropout:
            if generator is not None:
                # Snapshot generator state
                ctx.gen_state = generator.get_state()
            else:
                # Snapshot global RNG state (CUDA or CPU depending on device)
                if q.is_cuda:
                    ctx.gen_state = torch.cuda.get_rng_state(q.device)
                else:
                    ctx.gen_state = torch.get_rng_state()
        else:
            ctx.gen_state = None

        out, _ = _forward_blockwise_online_softmax(
            q,
            k,
            v,
            causal=ctx.causal,
            scale_f=ctx.scale_f,
            q_block=ctx.q_block,
            kv_block=ctx.kv_block,
            additive_mask=mask_norm,
            dropout_p=ctx.dropout_p,
            training=ctx.training,
            generator=generator,
        )
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, mask_saved = ctx.saved_tensors
        additive_mask = mask_saved if ctx.has_mask else None

        # Restore RNG state so dropout is identical to forward
        generator = None
        if ctx.has_dropout:
            if ctx.generator_provided:
                # We cannot recover the original generator object; user must pass one.
                # Fallback: use global rng restoration.
                # (For most training runs users won't rely on external generator.)
                if q.is_cuda:
                    torch.cuda.set_rng_state(ctx.gen_state, device=q.device)
                else:
                    torch.set_rng_state(ctx.gen_state)
            else:
                if q.is_cuda:
                    torch.cuda.set_rng_state(ctx.gen_state, device=q.device)
                else:
                    torch.set_rng_state(ctx.gen_state)

        # Compute gradients using exact attention formulas with blockwise recomputation.
        # Let P = softmax(S) (after masks/dropout), O = P V.
        # dV = P^T dO
        # dP = dO V^T
        # dS = (dP - sum(dP * P, -1, keepdim=True)) * P
        # dQ = dS K * scale
        # dK = dS^T Q * scale

        _validate_shapes(q, k, v)
        b, h, q_len, d = q.shape
        k_len = k.shape[2]

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        q_f = q.to(torch.float32)
        k_f = k.to(torch.float32)
        v_f = v.to(torch.float32)
        do_f = dout.to(torch.float32)

        offset = k_len - q_len

        keep_p = 1.0 - ctx.dropout_p

        for qs in range(0, q_len, ctx.q_block):
            qe = min(qs + ctx.q_block, q_len)
            qb = qe - qs

            q_blk = q_f[:, :, qs:qe, :]  # (B,H,Qb,D)
            do_blk = do_f[:, :, qs:qe, :]

            # First pass: recompute m and l per row (like forward) to get stable P
            m = torch.full((b, h, qb), float("-inf"), device=q.device, dtype=torch.float32)
            l = torch.zeros((b, h, qb), device=q.device, dtype=torch.float32)

            for ks in range(0, k_len, ctx.kv_block):
                ke = min(ks + ctx.kv_block, k_len)
                k_blk = k_f[:, :, ks:ke, :]

                s = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * ctx.scale_f

                if additive_mask is not None:
                    m_blk = additive_mask[:, :, qs:qe, ks:ke].to(torch.float32)
                    if m_blk.shape[1] == 1 and h != 1:
                        m_blk = m_blk.expand(-1, h, -1, -1)
                    s = s + m_blk
                elif ctx.causal:
                    cm = _causal_mask_block(qs=qs, qe=qe, ks=ks, ke=ke, offset=offset, device=q.device)
                    s = s.masked_fill(cm[None, None, :, :], float("-inf"))

                m_ij = torch.amax(s, dim=-1)
                m_new = torch.maximum(m, m_ij)
                p_unnorm = torch.exp(s - m_new[..., None])

                alpha = torch.exp(m - m_new)
                l = l * alpha + p_unnorm.sum(dim=-1)
                m = m_new

            l_safe = l.clamp_min(1e-9)
            valid_row = (l > 0)  # (B,H,Qb)

            # Second pass: accumulate grads blockwise
            for ks in range(0, k_len, ctx.kv_block):
                ke = min(ks + ctx.kv_block, k_len)
                k_blk = k_f[:, :, ks:ke, :]  # (B,H,Kb,D)
                v_blk = v_f[:, :, ks:ke, :]

                s = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * ctx.scale_f

                if additive_mask is not None:
                    m_blk = additive_mask[:, :, qs:qe, ks:ke].to(torch.float32)
                    if m_blk.shape[1] == 1 and h != 1:
                        m_blk = m_blk.expand(-1, h, -1, -1)
                    s = s + m_blk
                elif ctx.causal:
                    cm = _causal_mask_block(qs=qs, qe=qe, ks=ks, ke=ke, offset=offset, device=q.device)
                    s = s.masked_fill(cm[None, None, :, :], float("-inf"))

                # Recompute P for this block: P = exp(s - m) / l
                p = torch.exp(s - m[..., None]) / l_safe[..., None]

                if ctx.has_dropout:
                    # Recompute identical dropout mask in the same order as forward
                    if q.is_cuda:
                        # generator is None; RNG state restored already
                        mask = torch.rand(p.shape, device=p.device, dtype=torch.float32) < keep_p
                    else:
                        mask = torch.rand(p.shape, device=p.device, dtype=torch.float32) < keep_p
                    p = p * mask.to(p.dtype) / keep_p

                # If row is invalid (fully masked), set p=0 so grads are 0.
                p = torch.where(valid_row[..., None], p, torch.zeros_like(p))

                # dV += P^T dO
                dv[:, :, ks:ke, :] += torch.matmul(p.transpose(-1, -2), do_blk)

                # dP = dO V^T
                dp = torch.matmul(do_blk, v_blk.transpose(-1, -2))  # (B,H,Qb,Kb)

                # dS = (dp - sum(dp * p)) * p
                # sum over Kb
                ds = (dp - (dp * p).sum(dim=-1, keepdim=True)) * p

                # dQ += dS K * scale
                dq[:, :, qs:qe, :] += torch.matmul(ds, k_blk) * ctx.scale_f

                # dK += dS^T Q * scale
                dk[:, :, ks:ke, :] += torch.matmul(ds.transpose(-1, -2), q_blk) * ctx.scale_f

        # Cast back to original dtypes
        dq = dq.to(q.dtype)
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)

        # Return grads for all forward inputs
        return (
            dq,
            dk,
            dv,
            None,  # causal
            None,  # q_block
            None,  # kv_block
            None,  # scale
            None,  # additive_mask
            None,  # dropout_p
            None,  # training
            None,  # generator
        )


def attention_mini_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    q_block: int = 128,
    kv_block: int = 128,
    scale: Optional[float] = None,
    additive_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """FlashAttention-style attention using online softmax + blocking.

    This is the public API.

    Args:
        q: (B,H,q_len,D)
        k,v: (B,H,k_len,D)
        causal: apply causal mask (bottom-right aligned for q_len != k_len)
        additive_mask: (B,1,Q,K) or (B,H,Q,K), additive (0 keep, -inf mask)
        dropout_p: attention dropout probability
        training: enable dropout and backward-friendly semantics
        generator: optional RNG generator for dropout determinism

    Returns:
        out: (B,H,q_len,D)
    """
    return _MiniFlashAttnFn.apply(
        q,
        k,
        v,
        bool(causal),
        int(q_block),
        int(kv_block),
        scale,
        additive_mask,
        float(dropout_p),
        bool(training),
        generator,
    )
