"""Dataset-based perplexity (PPL) evaluation for Pythia-2.8B.

Meets homework baseline requirement:
- Evaluate PPL on wikitext and pg-19 (pg-19 can use a single sample)
- Compare attention implementations: eager / flash_attention_2 / mini_flash

Usage examples:
  # Single attention method
  MODEL_DTYPE=bf16 python evaluate_dataset_ppl.py --dataset pg19 --attn eager
  # All attention methods
  MODEL_DTYPE=bf16 python evaluate_dataset_ppl.py --dataset pg19 --all_attn
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repo root on path for mini_flash patch
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.patch_gpt_neox_mini_flash import patch_gpt_neox_attention


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["wikitext2", "wikitext103", "pg19"])
    attn_group = ap.add_mutually_exclusive_group(required=True)
    attn_group.add_argument("--attn", type=str, choices=["eager", "flash_attention_2", "mini_flash"],
                          help="Attention implementation to use")
    attn_group.add_argument("--all_attn", action="store_true",
                          help="Run all attention implementations and compare")
    ap.add_argument("--split", type=str, default="test", help="dataset split (wikitext: test/validation; pg19: test)")
    ap.add_argument("--num_samples", type=int, default=None, help="how many documents to use (pg19 default: 1)")
    ap.add_argument("--max_tokens", type=int, default=4096, help="max tokens evaluated (truncate) across all docs")
    ap.add_argument("--stride", type=int, default=1024, help="stride for chunked evaluation")
    ap.add_argument("--seqlen", type=int, default=2048, help="sequence length for chunked evaluation")
    ap.add_argument("--batch_size", type=int, default=1)
    return ap.parse_args()


def get_dtype():
    s = os.environ.get("MODEL_DTYPE", "bf16").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_text_dataset(name: str, split: str, num_samples: int) -> List[str]:
    if name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        return [x["text"] for x in ds]
    if name == "wikitext103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        return [x["text"] for x in ds]
    if name == "pg19":
        # Prefer a pre-packaged HF copy that supports streaming.
        # This avoids the original "pg19" dataset script (which may require trust_remote_code
        # and can break depending on your `datasets` version).
        #
        # Columns include: text, short_book_title, ...
        ds = load_dataset("emozilla/pg19", split=split, streaming=True)
        texts: List[str] = []
        for x in ds:
            t = x.get("text", "")
            if t and t.strip():
                texts.append(t)
            if len(texts) >= num_samples:
                break
        return texts
    raise ValueError(name)


@torch.no_grad()
def ppl_on_token_stream(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    seqlen: int,
    stride: int,
) -> float:
    """Compute PPL on a long token stream using sliding window chunks.

    Standard approach:
    - take windows of length seqlen
    - compute loss on the last (seqlen - stride) tokens of each window

    Returns:
        perplexity
    """
    T = input_ids.shape[1]
    nlls: List[torch.Tensor] = []

    # We compute chunk losses in fp32 for stability
    for start in range(0, T, stride):
        end = min(start + seqlen, T)
        if end - start < 2:
            break

        ids = input_ids[:, start:end]
        mask = attention_mask[:, start:end]

        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = out.logits

        # next-token prediction inside chunk
        shift_logits = logits[:, :-1, :].float()
        shift_labels = ids[:, 1:]

        # Only score the last part of the window (exclude context-only tokens)
        keep_from = max(0, (shift_labels.shape[1] - stride))

        loss = F.cross_entropy(
            shift_logits[:, keep_from:, :].reshape(-1, shift_logits.size(-1)),
            shift_labels[:, keep_from:].reshape(-1),
            reduction="mean",
        )

        nlls.append(loss)

        if end == T:
            break

    mean_nll = torch.stack(nlls).mean()
    return float(torch.exp(mean_nll).cpu())


def evaluate_model(
    attn_impl: str,
    dataset: str,
    split: str,
    num_samples: int,
    max_tokens: int,
    seqlen: int,
    stride: int,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
) -> float:
    """Evaluate a single model configuration."""
    print(f"\n=== Evaluating with {attn_impl} ===")
    
    # Patch if needed
    if attn_impl == "mini_flash":
        patch_gpt_neox_attention(impl_name="mini_flash")

    print(f"Dataset={dataset} split={split} attn={attn_impl} dtype={dtype} device={device}")
    print(f"num_samples={num_samples} max_tokens={max_tokens} seqlen={seqlen} stride={stride}")

    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-2.8b",
        attn_implementation=attn_impl,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")

    texts = load_text_dataset(dataset, split, num_samples)
    joined = "\n\n".join(texts)
    enc = tok(joined, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    ppl = ppl_on_token_stream(model, input_ids, attention_mask, seqlen=seqlen, stride=stride)
    print(f"PPL={ppl:.6f}")
    
    return ppl


def main():
    args = parse_args()

    device = get_device()
    dtype = get_dtype()

    # pg19 default to 1 sample unless user overrides
    if args.dataset == "pg19" and args.num_samples is None:
        args.num_samples = 1
    if args.num_samples is None:
        args.num_samples = 50  # wikitext default

    if args.all_attn:
        # Run all attention methods
        attn_methods = ["eager", "flash_attention_2", "mini_flash"]
        results = {}
        
        for attn in attn_methods:
            try:
                ppl = evaluate_model(
                    attn_impl=attn,
                    dataset=args.dataset,
                    split=args.split,
                    num_samples=args.num_samples,
                    max_tokens=args.max_tokens,
                    seqlen=args.seqlen,
                    stride=args.stride,
                    batch_size=args.batch_size,
                    device=device,
                    dtype=dtype,
                )
                results[attn] = ppl
            except Exception as e:
                print(f"Error evaluating {attn}: {str(e)}")
                results[attn] = None

        # Print summary
        print("\n=== Results Summary ===")
        for attn, ppl in results.items():
            if ppl is not None:
                print(f"{attn}: PPL = {ppl:.6f}")
            else:
                print(f"{attn}: Failed to evaluate")
    else:
        # Original single-attention evaluation
        evaluate_model(
            attn_impl=args.attn,
            dataset=args.dataset,
            split=args.split,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            seqlen=args.seqlen,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
            dtype=dtype,
        )


if __name__ == "__main__":
    main()