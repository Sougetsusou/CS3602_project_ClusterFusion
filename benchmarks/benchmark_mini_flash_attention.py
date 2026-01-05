"""Benchmark: reference attention vs mini-flash attention.

This benchmarks the attention kernel in isolation (not full model) to show:
- runtime scaling with sequence length
- peak CUDA memory allocated

Note: mini_flash is implemented in PyTorch loops (not fused), so it may be
slower than optimized kernels. The key point is algorithmic reproduction:
- online softmax
- blockwise computation
- avoids materializing T x T attention matrix
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mini_flash_attention import attention_reference, attention_mini_flash

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_STR = os.environ.get("DTYPE", "bf16").lower()
if DTYPE_STR in ("bf16", "bfloat16"):
    DTYPE = torch.bfloat16
elif DTYPE_STR in ("fp16", "float16"):
    DTYPE = torch.float16
else:
    DTYPE = torch.float32

WARMUP = int(os.environ.get("WARMUP", "5"))
RUNS = int(os.environ.get("RUNS", "20"))

LENS = [int(x) for x in os.environ.get("LENS", "256,512,1024,2048").split(",") if x.strip()]

B = int(os.environ.get("B", "1"))
H = int(os.environ.get("H", "32"))
D = int(os.environ.get("D", "128"))

Q_BLOCK = int(os.environ.get("Q_BLOCK", "128"))
KV_BLOCK = int(os.environ.get("KV_BLOCK", "128"))


def cuda_sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def reset_peak():
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_alloc():
    if DEVICE != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated())


def fmt_bytes(x):
    if x is None:
        return "N/A"
    return f"{x / (1024**3):.3f} GiB"


def bench_one(fn, q, k, v):
    # Warmup
    for _ in range(WARMUP):
        _ = fn(q, k, v)
    cuda_sync()

    times = []
    peaks = []
    for _ in range(RUNS):
        reset_peak()
        cuda_sync()
        t0 = time.time()
        _ = fn(q, k, v)
        cuda_sync()
        t1 = time.time()
        times.append((t1 - t0) * 1000)
        peaks.append(peak_alloc())

    peak = None
    if peaks and peaks[0] is not None:
        peak = int(max(peaks))

    return float(np.mean(times)), float(np.min(times)), peak


def main():
    print(f"Device={DEVICE} dtype={DTYPE}")
    print(f"B={B} H={H} D={D} q_block={Q_BLOCK} kv_block={KV_BLOCK}")

    for T in LENS:
        print("\n" + "-" * 80)
        print(f"T={T}")
        q = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
        k = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)
        v = torch.randn(B, H, T, D, device=DEVICE, dtype=DTYPE)

        # reference uses fp32 for stability; will materialize T x T
        def ref(q, k, v):
            return attention_reference(q.float(), k.float(), v.float(), causal=True)

        def mini(q, k, v):
            return attention_mini_flash(q, k, v, causal=True, q_block=Q_BLOCK, kv_block=KV_BLOCK)

        ref_mean, ref_min, ref_peak = bench_one(ref, q, k, v)
        mini_mean, mini_min, mini_peak = bench_one(mini, q, k, v)

        print(f"Reference: mean {ref_mean:.2f} ms (min {ref_min:.2f} ms), peak {fmt_bytes(ref_peak)}")
        print(f"MiniFlash:  mean {mini_mean:.2f} ms (min {mini_min:.2f} ms), peak {fmt_bytes(mini_peak)}")

        if ref_peak is not None and mini_peak is not None and mini_peak > 0:
            print(f"Peak mem ratio (ref/mini): {ref_peak / mini_peak:.2f}x")
        print(f"Speed ratio (ref/mini): {ref_mean / mini_mean:.2f}x")


if __name__ == "__main__":
    main()
