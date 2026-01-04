"""Monkey-patch GPT-NeoX attention to use our mini FlashAttention implementation.

Transformers 4.57+ GPT-NeoX calls an attention backend with signature:
    (module, query, key, value, attention_mask, scaling, dropout, head_mask, **kwargs)

The eager backend (transformers) does:
- attn_weights = (Q @ K^T) * scaling
- if attention_mask is not None: attn_weights += attention_mask[:, :, :, :K]
- softmax in fp32
- dropout (disabled in eval)
- attn_output = attn_weights @ V
- attn_output = attn_output.transpose(1, 2).contiguous()

Important: GPT-NeoX expects the attention backend to return attn_output in shape:
    (B, q_len, H, D)

Our previous patch returned (B, H, q_len, D), which breaks the model and causes
catastrophic PPL.

This patch fixes the layout to match eager exactly.

Decode/cache (q_len != k_len): fallback to eager for now.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from models.mini_flash_attention import attention_mini_flash


def patch_gpt_neox_attention(
    q_block: int = 128,
    kv_block: int = 128,
    impl_name: str = "mini_flash",
    debug_once: bool = True,
) -> str:
    try:
        import transformers.models.gpt_neox.modeling_gpt_neox as m
    except Exception as e:
        raise RuntimeError(f"Failed to import transformers GPT-NeoX module: {e}")

    if not hasattr(m, "ALL_ATTENTION_FUNCTIONS"):
        raise RuntimeError("transformers GPT-NeoX module does not expose ALL_ATTENTION_FUNCTIONS; cannot patch")

    if impl_name in m.ALL_ATTENTION_FUNCTIONS:
        return impl_name

    state = {"printed": False}

    def mini_flash_attention_forward(
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float = 0.0,
        head_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # query/key/value: (B, H, q_len/k_len, D)
        q_len = query.shape[2]
        k_len = key.shape[2]

        # Decode/cache path: fallback
        if k_len != q_len:
            return m.eager_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=scaling,
                dropout=dropout,
                head_mask=head_mask,
                **kwargs,
            )

        if debug_once and (not state["printed"]):
            state["printed"] = True
            try:
                cache_position = kwargs.get("cache_position", None)
                print(
                    "[mini_flash debug] q,k,v:",
                    tuple(query.shape),
                    tuple(key.shape),
                    tuple(value.shape),
                    "attention_mask:",
                    None if attention_mask is None else (tuple(attention_mask.shape), attention_mask.dtype, float(attention_mask.min().item()), float(attention_mask.max().item())),
                    "cache_position:",
                    None if cache_position is None else (tuple(cache_position.shape), int(cache_position[0, 0].item()), int(cache_position[0, -1].item())),
                )
            except Exception:
                pass

        # Our mini attention expects scale separately. We match eager: scores * scaling.
        out = attention_mini_flash(
            query,
            key,
            value,
            causal=True if attention_mask is None else False,
            q_block=q_block,
            kv_block=kv_block,
            scale=scaling,
            additive_mask=attention_mask,
        )

        # Match eager: softmax weights fp32 then cast; we already accumulate fp32 inside.
        if head_mask is not None:
            # eager multiplies head_mask into attn_weights; multiplying output is a close proxy
            # for eval-only benchmarking (dropout=0). For strict equivalence we'd need weights.
            out = out * head_mask

        # CRITICAL: match eager output layout: (B, q_len, H, D)
        out = out.transpose(1, 2).contiguous()

        return out, None

    m.ALL_ATTENTION_FUNCTIONS[impl_name] = mini_flash_attention_forward
    return impl_name
