#!/usr/bin/env python3
"""
Ablation Study: Split Kernel Performance Breakdown

Tests the performance of each kernel branch separately:
- Kernel 1 (Attention): LayerNorm → QKV → RoPE → Attention → Output Proj → Post-LN → MLP Up → GELU
- Kernel 2 (MLP Down): MLP Down → Cluster Reduce → Residual

This helps understand:
1. Which branch dominates execution time
2. The overhead from splitting (kernel launch + global memory transfer)
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model configuration
MODEL_NAME = "EleutherAI/pythia-2.8b"
HIDDEN_DIM = 2560
NUM_HEADS = 32
HEAD_DIM = 80
ROTARY_DIM = 20
NUM_LAYERS = 32
FFN_DIM = 10240

# Test configuration
WARMUP_ITERS = 100
BENCHMARK_ITERS = 500

def print_header():
    print("=" * 80)
    print("Ablation Study: Split Kernel Performance Breakdown")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {NUM_LAYERS}, Hidden: {HIDDEN_DIM}, FFN: {FFN_DIM}")
    print(f"Benchmark: {WARMUP_ITERS} warmup + {BENCHMARK_ITERS} iterations")
    print("=" * 80)

def load_model():
    """Load Pythia-2.8B model."""
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda:0"
    )
    model.eval()
    print(f"Model loaded: {MODEL_NAME}")
    return model, tokenizer

def precompute_rope_embeddings(max_position, rotary_dim=ROTARY_DIM, head_dim=HEAD_DIM, base=10000, device="cuda:0"):
    """Precompute all RoPE embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    padding_size = head_dim - rotary_dim
    cos = torch.cat([cos, torch.ones((max_position, padding_size), device=device)], dim=-1)
    sin = torch.cat([sin, torch.zeros((max_position, padding_size), device=device)], dim=-1)
    return cos, sin

def extract_layer_weights(layer):
    """Extract weights from a single layer."""
    return {
        "qkv_weight": layer.attention.query_key_value.weight.T.contiguous(),
        "qkv_bias": layer.attention.query_key_value.bias.contiguous(),
        "o_weight": layer.attention.dense.weight.T.contiguous(),
        "o_bias": layer.attention.dense.bias.contiguous(),
        "ln_weight": layer.input_layernorm.weight.contiguous(),
        "ln_bias": layer.input_layernorm.bias.contiguous(),
        "post_ln_weight": layer.post_attention_layernorm.weight.contiguous(),
        "post_ln_bias": layer.post_attention_layernorm.bias.contiguous(),
        "mlp_up_weight": layer.mlp.dense_h_to_4h.weight.T.contiguous(),
        "mlp_up_bias": layer.mlp.dense_h_to_4h.bias.contiguous(),
        "mlp_down_weight": layer.mlp.dense_4h_to_h.weight.T.contiguous(),
        "mlp_down_bias": layer.mlp.dense_4h_to_h.bias.contiguous(),
    }

def benchmark_fused_kernel(model, seq_len=64):
    """Benchmark fused kernel (single layer, multiple iterations)."""
    import clusterfusion
    
    device = torch.device("cuda:0")
    layer = model.gpt_neox.layers[0]
    weights = extract_layer_weights(layer)
    
    # Setup
    max_seq_len = seq_len + 100
    all_cos, all_sin = precompute_rope_embeddings(max_seq_len, device=device)
    k_cache = torch.zeros(max_seq_len, HIDDEN_DIM, device=device, dtype=torch.float16)
    v_cache = torch.zeros(max_seq_len, HIDDEN_DIM, device=device, dtype=torch.float16)
    h = torch.randn(1, 1, HIDDEN_DIM, device=device, dtype=torch.float16)
    cos, sin = all_cos[seq_len], all_sin[seq_len]
    
    # Warmup
    for _ in range(WARMUP_ITERS):
        _, _, _ = clusterfusion.pythia_2b8_decoder_layer(
            h, weights["qkv_weight"], weights["qkv_bias"],
            weights["o_weight"], weights["o_bias"],
            k_cache, v_cache,
            weights["ln_weight"], weights["ln_bias"], cos, sin,
            weights["post_ln_weight"], weights["post_ln_bias"],
            weights["mlp_up_weight"], weights["mlp_up_bias"],
            weights["mlp_down_weight"], weights["mlp_down_bias"],
            seq_len
        )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(BENCHMARK_ITERS):
        _, _, _ = clusterfusion.pythia_2b8_decoder_layer(
            h, weights["qkv_weight"], weights["qkv_bias"],
            weights["o_weight"], weights["o_bias"],
            k_cache, v_cache,
            weights["ln_weight"], weights["ln_bias"], cos, sin,
            weights["post_ln_weight"], weights["post_ln_bias"],
            weights["mlp_up_weight"], weights["mlp_up_bias"],
            weights["mlp_down_weight"], weights["mlp_down_bias"],
            seq_len
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / BENCHMARK_ITERS * 1000  # ms per iteration

def benchmark_split_kernel(model, seq_len=64):
    """Benchmark split kernel (single layer, multiple iterations)."""
    import clusterfusion
    
    device = torch.device("cuda:0")
    layer = model.gpt_neox.layers[0]
    weights = extract_layer_weights(layer)
    
    # Setup
    max_seq_len = seq_len + 100
    all_cos, all_sin = precompute_rope_embeddings(max_seq_len, device=device)
    k_cache = torch.zeros(max_seq_len, HIDDEN_DIM, device=device, dtype=torch.float16)
    v_cache = torch.zeros(max_seq_len, HIDDEN_DIM, device=device, dtype=torch.float16)
    h = torch.randn(1, 1, HIDDEN_DIM, device=device, dtype=torch.float16)
    cos, sin = all_cos[seq_len], all_sin[seq_len]
    
    # Warmup
    for _ in range(WARMUP_ITERS):
        _, _, _ = clusterfusion.pythia_2b8_decoder_layer_split(
            h, weights["qkv_weight"], weights["qkv_bias"],
            weights["o_weight"], weights["o_bias"],
            k_cache, v_cache,
            weights["ln_weight"], weights["ln_bias"], cos, sin,
            weights["post_ln_weight"], weights["post_ln_bias"],
            weights["mlp_up_weight"], weights["mlp_up_bias"],
            weights["mlp_down_weight"], weights["mlp_down_bias"],
            seq_len
        )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(BENCHMARK_ITERS):
        _, _, _ = clusterfusion.pythia_2b8_decoder_layer_split(
            h, weights["qkv_weight"], weights["qkv_bias"],
            weights["o_weight"], weights["o_bias"],
            k_cache, v_cache,
            weights["ln_weight"], weights["ln_bias"], cos, sin,
            weights["post_ln_weight"], weights["post_ln_bias"],
            weights["mlp_up_weight"], weights["mlp_up_bias"],
            weights["mlp_down_weight"], weights["mlp_down_bias"],
            seq_len
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / BENCHMARK_ITERS * 1000  # ms per iteration

def estimate_kernel_breakdown():
    """
    Estimate the theoretical breakdown of kernel time based on FLOPs.
    
    Kernel 1 (Attention + MLP Up):
    - LayerNorm: 2 * HIDDEN_DIM (mean, var) + 2 * HIDDEN_DIM (normalize) = 4 * 2560 = 10K FLOPs
    - QKV Projection: 2 * HIDDEN_DIM * 3*HIDDEN_DIM = 2 * 2560 * 7680 = 39.3M FLOPs
    - RoPE: ~4 * NUM_HEADS * ROTARY_DIM = 4 * 32 * 20 = 2.6K FLOPs
    - Attention (Q@K, softmax, @V): 2 * NUM_HEADS * HEAD_DIM * SEQ_LEN = varies with seq_len
    - Output Projection: 2 * HIDDEN_DIM * HIDDEN_DIM = 2 * 2560 * 2560 = 13.1M FLOPs
    - Post LayerNorm: 4 * 2560 = 10K FLOPs
    - MLP Up Projection: 2 * HIDDEN_DIM * FFN_DIM = 2 * 2560 * 10240 = 52.4M FLOPs
    - GELU: ~8 * FFN_DIM = 82K FLOPs
    
    Kernel 2 (MLP Down):
    - MLP Down Projection: 2 * FFN_DIM * HIDDEN_DIM = 2 * 10240 * 2560 = 52.4M FLOPs
    - Residual Add: 2 * HIDDEN_DIM = 5K FLOPs
    
    Total Kernel 1 (seq_len=64): ~104.8M FLOPs + 0.4M (attention) = ~105.2M FLOPs
    Total Kernel 2: ~52.4M FLOPs
    
    Ratio: K1 : K2 ≈ 2 : 1
    """
    # FLOPs breakdown (in millions)
    qkv_proj = 2 * HIDDEN_DIM * 3 * HIDDEN_DIM / 1e6  # 39.3M
    output_proj = 2 * HIDDEN_DIM * HIDDEN_DIM / 1e6   # 13.1M
    mlp_up = 2 * HIDDEN_DIM * FFN_DIM / 1e6           # 52.4M
    mlp_down = 2 * FFN_DIM * HIDDEN_DIM / 1e6         # 52.4M
    
    kernel1_flops = qkv_proj + output_proj + mlp_up  # ~104.8M (excluding attention which varies)
    kernel2_flops = mlp_down
    
    total_flops = kernel1_flops + kernel2_flops
    
    print("\n" + "-" * 60)
    print("Theoretical FLOPs Breakdown (per layer, excluding attention)")
    print("-" * 60)
    print(f"QKV Projection:    {qkv_proj:>8.2f}M FLOPs")
    print(f"Output Projection: {output_proj:>8.2f}M FLOPs")
    print(f"MLP Up Projection: {mlp_up:>8.2f}M FLOPs")
    print(f"MLP Down Projection: {mlp_down:>8.2f}M FLOPs")
    print("-" * 60)
    print(f"Kernel 1 (Attn+MLPUp): {kernel1_flops:>8.2f}M FLOPs ({kernel1_flops/total_flops*100:.1f}%)")
    print(f"Kernel 2 (MLPDown):    {kernel2_flops:>8.2f}M FLOPs ({kernel2_flops/total_flops*100:.1f}%)")
    print(f"Total:                 {total_flops:>8.2f}M FLOPs")
    print("-" * 60)
    print(f"Expected time ratio K1:K2 ≈ {kernel1_flops/kernel2_flops:.2f}:1")

def estimate_memory_overhead():
    """Estimate the memory overhead from splitting."""
    # Intermediate buffer: mlp_intermediate [FFN_DIM] in FP16
    intermediate_bytes = FFN_DIM * 2  # 10240 * 2 = 20KB
    
    # attn_output [1, HIDDEN_DIM] in FP16
    attn_output_bytes = HIDDEN_DIM * 2  # 2560 * 2 = 5KB
    
    total_bytes = intermediate_bytes + attn_output_bytes
    
    print("\n" + "-" * 60)
    print("Memory Overhead from Splitting")
    print("-" * 60)
    print(f"MLP Intermediate Buffer: {intermediate_bytes/1024:.2f} KB")
    print(f"Attention Output Buffer: {attn_output_bytes/1024:.2f} KB")
    print(f"Total Global Memory Traffic: {total_bytes/1024:.2f} KB per layer")
    print("-" * 60)
    
    # RTX 5090 memory bandwidth: ~1.8 TB/s
    # Time to transfer: 25KB / 1.8TB/s = ~14ns
    bandwidth_tbps = 1.8
    transfer_time_ns = total_bytes / (bandwidth_tbps * 1e12) * 1e9
    print(f"Estimated transfer time: {transfer_time_ns:.2f} ns")
    print(f"(RTX 5090 bandwidth: {bandwidth_tbps} TB/s)")

def main():
    print_header()
    
    model, tokenizer = load_model()
    
    # Show theoretical breakdown
    estimate_kernel_breakdown()
    estimate_memory_overhead()
    
    # Benchmark at different sequence lengths
    seq_lengths = [32, 64, 128, 256, 512]
    
    print("\n" + "=" * 80)
    print("Ablation Results: Single Layer Kernel Time")
    print("=" * 80)
    print(f"{'Seq Len':>8} | {'Fused (ms)':>12} | {'Split (ms)':>12} | {'Overhead':>12} | {'Overhead %':>12}")
    print("-" * 80)
    
    results = []
    for seq_len in seq_lengths:
        try:
            fused_time = benchmark_fused_kernel(model, seq_len)
            split_time = benchmark_split_kernel(model, seq_len)
            overhead = split_time - fused_time
            overhead_pct = overhead / fused_time * 100
            
            print(f"{seq_len:>8} | {fused_time:>12.4f} | {split_time:>12.4f} | {overhead:>12.4f} | {overhead_pct:>11.2f}%")
            results.append({
                "seq_len": seq_len,
                "fused": fused_time,
                "split": split_time,
                "overhead": overhead,
                "overhead_pct": overhead_pct
            })
        except Exception as e:
            print(f"{seq_len:>8} | ERROR: {str(e)[:50]}")
    
    print("-" * 80)
    
    # Summary
    if results:
        avg_overhead = sum(r["overhead"] for r in results) / len(results)
        avg_overhead_pct = sum(r["overhead_pct"] for r in results) / len(results)
        
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Average overhead per layer: {avg_overhead:.4f} ms ({avg_overhead_pct:.2f}%)")
        print(f"For {NUM_LAYERS} layers: {avg_overhead * NUM_LAYERS:.4f} ms additional time")
        print()
        print("Overhead sources:")
        print("  1. Extra kernel launch (~2-5 μs per launch)")
        print("  2. Global memory write: mlp_intermediate [10240 * 2 = 20KB]")
        print("  3. Global memory read:  mlp_intermediate [10240 * 2 = 20KB]")
        print("  4. Implicit device synchronization between kernels")
        print()
        print("Note: Fused kernel uses grid.sync() for synchronization within the kernel,")
        print("      while split kernel relies on kernel launch boundary synchronization.")

if __name__ == "__main__":
    main()

