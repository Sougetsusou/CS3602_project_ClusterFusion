import os
import math
import torch

from models.mini_flash_attention import attention_reference, attention_mini_flash


def _run_case(device: str, dtype: torch.dtype, causal: bool):
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 257, 64

    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)

    # Compute reference in fp32 for stability
    ref = attention_reference(q.float(), k.float(), v.float(), causal=causal).to(torch.float32)

    # Mini flash in chosen dtype (internally accumulates in fp32)
    out = attention_mini_flash(q, k, v, causal=causal, q_block=128, kv_block=128).float()

    max_err = (ref - out).abs().max().item()
    mean_err = (ref - out).abs().mean().item()
    return max_err, mean_err


def test_mini_flash_attention_cpu_fp32_causal():
    max_err, mean_err = _run_case(device="cpu", dtype=torch.float32, causal=True)
    assert max_err < 5e-4
    assert mean_err < 5e-5


def test_mini_flash_attention_cpu_fp32_noncausal():
    max_err, mean_err = _run_case(device="cpu", dtype=torch.float32, causal=False)
    assert max_err < 5e-4
    assert mean_err < 5e-5


def test_mini_flash_attention_cuda_bf16_causal():
    if not torch.cuda.is_available():
        return
    max_err, mean_err = _run_case(device="cuda", dtype=torch.bfloat16, causal=True)
    # bf16 is looser
    assert max_err < 5e-2
    assert mean_err < 5e-3


def test_mini_flash_attention_cuda_fp16_causal():
    if not torch.cuda.is_available():
        return
    max_err, mean_err = _run_case(device="cuda", dtype=torch.float16, causal=True)
    assert max_err < 1e-1
    assert mean_err < 1e-2

