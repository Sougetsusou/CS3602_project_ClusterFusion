"""Model-level benchmark: Pythia-2.8B with (1) eager, (2) flash-attn2, (3) our mini-flash attention patch.

This satisfies the homework requirement to test on the unified baseline model.

We register a custom attention backend ("mini_flash") for GPT-NeoX via
ALL_ATTENTION_FUNCTIONS, and load the model with attn_implementation="mini_flash".

Metrics per prompt length:
- Prefill latency (ms/prompt)  [TTFT proxy]
- Peak CUDA allocated memory during prefill
- Prompt PPL (teacher forcing over first PPL_EVAL_TOKENS tokens)

Notes:
- Our mini-flash is pure PyTorch (not a fused CUDA kernel), so speed is much
  slower than flash-attn2.
- This benchmark uses a synthetic no-padding prompt (attention_mask all ones).
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.patch_gpt_neox_mini_flash import patch_gpt_neox_attention

MODEL_ID = os.environ.get("MODEL_ID", "EleutherAI/pythia-2.8b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_PROMPT_LENS = [512, 1024, 2048, 4096, 8192]
PROMPT_LENS: List[int] = [
    int(x)
    for x in os.environ.get("PROMPT_LENS", ",".join(map(str, DEFAULT_PROMPT_LENS))).split(",")
    if x.strip()
]

WARMUP = int(os.environ.get("WARMUP", "2"))
RUNS = int(os.environ.get("RUNS", "5"))
PPL_EVAL_TOKENS = int(os.environ.get("PPL_EVAL_TOKENS", "256"))

# Print mini-flash debug only once overall (avoid duplicate lines)
PRINT_MINIFLASH_DEBUG = os.environ.get("PRINT_MINIFLASH_DEBUG", "0").lower() in ("1", "true", "yes")

# dtype
DTYPE_STR = os.environ.get("MODEL_DTYPE", "bf16").lower()
if DTYPE_STR in ("bf16", "bfloat16"):
    DTYPE = torch.bfloat16
elif DTYPE_STR in ("fp16", "float16"):
    DTYPE = torch.float16
else:
    DTYPE = torch.float32

PPL_DTYPE = torch.float32

# mini-flash blocks
Q_BLOCK = int(os.environ.get("Q_BLOCK", "128"))
KV_BLOCK = int(os.environ.get("KV_BLOCK", "128"))


def cuda_sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def reset_peak():
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_alloc() -> Optional[int]:
    if DEVICE != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated())


def fmt_bytes(x: Optional[int]) -> str:
    if x is None:
        return "N/A"
    return f"{x/(1024**3):.2f} GiB"


def make_inputs(tokenizer, prompt_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    base = "The meaning of life is "
    text = (base * max(1, prompt_len // 3)).strip()
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids
    if input_ids.shape[1] > prompt_len:
        input_ids = input_ids[:, :prompt_len]
    input_ids = input_ids.to(DEVICE)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def prefill_time_and_mem(model, input_ids, attention_mask):
    """Return (ms, peak_mem_bytes)."""
    reset_peak()
    cuda_sync()
    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    cuda_sync()
    t1 = time.time()
    return (t1 - t0) * 1000.0, peak_alloc()


def measure_ttft(model, input_ids, attention_mask) -> Tuple[float, Optional[int]]:
    """Measure Time To First Token (TTFT) = prefill + 1 token generation.
    
    Returns:
        (ttft_ms, peak_mem_bytes)
    """
    reset_peak()
    cuda_sync()
    t0 = time.time()
    with torch.no_grad():
        # Prefill
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        # Generate 1 token
        next_token = torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device) * 42  # Dummy token ID
        _ = model(input_ids=next_token, past_key_values=outputs.past_key_values, use_cache=False)
    cuda_sync()
    t1 = time.time()
    return (t1 - t0) * 1000.0, peak_alloc()


def prompt_ppl(model, input_ids, attention_mask) -> float:
    n = min(input_ids.shape[1], max(2, PPL_EVAL_TOKENS))
    ids = input_ids[:, :n]
    mask = attention_mask[:, :n]
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits
        shift_logits = logits[:, :-1, :].to(PPL_DTYPE)
        shift_labels = ids[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="mean",
        )
    return float(torch.exp(loss).cpu())


def load_model(attn_impl: str, patch_mini_flash: bool = False):
    if patch_mini_flash:
        # Register our custom attention backend under ALL_ATTENTION_FUNCTIONS.
        # Only print debug if PRINT_MINIFLASH_DEBUG is enabled.
        attn_impl = patch_gpt_neox_attention(
            q_block=Q_BLOCK,
            kv_block=KV_BLOCK,
            impl_name=attn_impl,
            debug_once=PRINT_MINIFLASH_DEBUG,
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        attn_implementation=attn_impl,
        dtype=DTYPE,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tok


def bench_config(name: str, attn_impl: str, patch: bool, prompt_len: int) -> Dict:
    model, tok = load_model(attn_impl, patch_mini_flash=patch)
    input_ids, attention_mask = make_inputs(tok, prompt_len)

    # Warmup
    for _ in range(WARMUP):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        cuda_sync()

    times = []
    peaks = []
    for _ in range(RUNS):
        t_ms, p = prefill_time_and_mem(model, input_ids, attention_mask)
        times.append(t_ms)
        peaks.append(p)

    # TTFT (prefill + 1 token)
    ttft_ms, ttft_peak = measure_ttft(model, input_ids, attention_mask)


    ppl = prompt_ppl(model, input_ids, attention_mask)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    peak = None
    if peaks and peaks[0] is not None:
        peak = int(max(peaks))

    return {
        "name": name,
        "prompt_len": int(prompt_len),
        "prefill_ms_mean": float(np.mean(times)),
        "prefill_ms_std": float(np.std(times)),
        "peak_mem_bytes_max": peak,
        "ppl": ppl,
        "ttft_ms": ttft_ms,
        "ttft_peak_mem": ttft_peak,
    }


def main():
    print("=== Pythia-2.8B Model-Level Prefill Benchmark (MiniFlash vs Eager vs FlashAttn2) ===")
    print(f"Model={MODEL_ID} device={DEVICE} dtype={DTYPE}")
    print(f"PromptLens={PROMPT_LENS} runs={RUNS} warmup={WARMUP}")
    print(f"MiniFlash blocks: q_block={Q_BLOCK} kv_block={KV_BLOCK}")

    for prompt_len in PROMPT_LENS:
        print("\n" + "=" * 110)
        print(f"Prompt length: {prompt_len}")
        print("=" * 110)

        results = []

        # Baseline
        try:
            print("\n--- Baseline (eager) ---")
            results.append(bench_config("Baseline (eager)", "eager", patch=False, prompt_len=prompt_len))
        except RuntimeError as e:
            print(f"  ❌ eager failed (likely OOM): {e}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # FlashAttention-2
        try:
            print("\n--- FlashAttention-2 ---")
            results.append(bench_config("FlashAttention-2", "flash_attention_2", patch=False, prompt_len=prompt_len))
        except RuntimeError as e:
            print(f"  ❌ flash_attention_2 failed: {e}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # MiniFlash
        try:
            print("\n--- MiniFlash (patched) ---")
            results.append(bench_config("MiniFlash (patched)", "mini_flash", patch=True, prompt_len=prompt_len))
        except RuntimeError as e:
            print(f"  ❌ mini_flash failed: {e}")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # Print table
        print("\n--- Results ---")
        print("-" * 110)
        print(f"| {'Config':<22} | {'Prefill ms':>12} | {'Std':>8} | {'PeakMem':>10} | {'TTFT ms':>10} | {'PPL':>10} |")
        print("-" * 110)
        for r in results:
            print(
                f"| {r['name']:<22} | {r['prefill_ms_mean']:>12.2f} | {r['prefill_ms_std']:>8.2f} | {fmt_bytes(r['peak_mem_bytes_max']):>10} | {r['ttft_ms']:>10.2f} | {r['ppl']:>10.6f} |"
            )
        print("-" * 110)

        # Speedups if possible
        eager = next((x for x in results if x["name"] == "Baseline (eager)"), None)
        flash = next((x for x in results if x["name"] == "FlashAttention-2"), None)
        mini = next((x for x in results if x["name"] == "MiniFlash (patched)"), None)

        if eager and flash:
            print(f"Prefill speedup (FlashAttention-2 vs eager): {eager['prefill_ms_mean'] / flash['prefill_ms_mean']:.2f}x")
        if eager and mini:
            print(f"Prefill speedup (MiniFlash vs eager): {eager['prefill_ms_mean'] / mini['prefill_ms_mean']:.2f}x")

        if eager and flash and eager["peak_mem_bytes_max"] and flash["peak_mem_bytes_max"]:
            print(f"Peak mem ratio (eager/flash): {eager['peak_mem_bytes_max'] / flash['peak_mem_bytes_max']:.2f}x")
        if eager and mini and eager["peak_mem_bytes_max"] and mini["peak_mem_bytes_max"]:
            print(f"Peak mem ratio (eager/mini): {eager['peak_mem_bytes_max'] / mini['peak_mem_bytes_max']:.2f}x")


if __name__ == "__main__":
    main()
