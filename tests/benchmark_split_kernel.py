#!/usr/bin/env python3
"""
Benchmark: Fused Kernel vs Split Kernel (Pythia-2.8B)

Compares three implementations:
1. Fused Kernel: Single kernel with grid.sync() (cooperative launch)
2. Split Kernel: Two kernels without grid.sync() (regular launch)
3. HuggingFace: Baseline implementation

The split kernel separates:
- Kernel 1: Attention + MLP Up + GELU
- Kernel 2: MLP Down + Residual

This tests whether removing grid.sync() and cooperative launch overhead
can compensate for the extra kernel launch overhead.
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
WARMUP_RUNS = 5
BENCHMARK_RUNS = 1
TOKEN_COUNTS = [32, 64, 128, 256, 512]

def print_header():
    print("=" * 100)
    print("Benchmark: Fused Kernel vs Split Kernel (Pythia-2.8B)")
    print("=" * 100)
    print(f"Model: {MODEL_NAME}")
    print(f"Config: HIDDEN_DIM={HIDDEN_DIM}, NUM_HEADS={NUM_HEADS}, HEAD_DIM={HEAD_DIM}")
    print(f"Benchmark: {WARMUP_RUNS} warmup + {BENCHMARK_RUNS} run per config")
    print("=" * 100)

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
    """Precompute all RoPE embeddings up to max_position."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    
    # Pad to HEAD_DIM
    padding_size = head_dim - rotary_dim
    cos = torch.cat([cos, torch.ones((max_position, padding_size), device=device)], dim=-1)
    sin = torch.cat([sin, torch.zeros((max_position, padding_size), device=device)], dim=-1)
    return cos, sin  # [max_position, head_dim]

def extract_all_weights(model, device):
    """Pre-extract all weights for benchmarking."""
    all_weights = []
    layers = model.gpt_neox.layers
    
    for layer in layers:
        weights = {
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
        all_weights.append(weights)
    
    return all_weights

def benchmark_decode_fused(model, input_ids, num_tokens, all_weights):
    """Benchmark fused kernel decode."""
    import clusterfusion
    
    device = torch.device("cuda:0")
    
    # Get model components
    embed_tokens = model.gpt_neox.embed_in
    final_ln = model.gpt_neox.final_layer_norm
    lm_head = model.embed_out
    
    # Initialize
    with torch.no_grad():
        hidden = embed_tokens(input_ids)
    
    max_seq_len = input_ids.shape[1] + num_tokens + 10
    
    # Precompute all RoPE embeddings
    all_cos, all_sin = precompute_rope_embeddings(max_seq_len, device=device)
    
    # Initialize KV caches
    k_caches = [torch.zeros(max_seq_len, HIDDEN_DIM, device=device, dtype=torch.float16) for _ in range(NUM_LAYERS)]
    v_caches = [torch.zeros(max_seq_len, HIDDEN_DIM, device=device, dtype=torch.float16) for _ in range(NUM_LAYERS)]
    
    # Prefill with HuggingFace
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        
        for layer_idx in range(NUM_LAYERS):
            k, v = past_key_values[layer_idx]
            seq_len = k.shape[2]
            for pos in range(seq_len):
                k_caches[layer_idx][pos] = k[0, :, pos, :].reshape(-1)
                v_caches[layer_idx][pos] = v[0, :, pos, :].reshape(-1)
    
    current_seq_len = input_ids.shape[1]
    
    # Warmup
    for _ in range(WARMUP_RUNS):
        pos = current_seq_len
        cos, sin = all_cos[pos], all_sin[pos]
        h = hidden[:, -1:, :].contiguous()
        
        for layer_idx in range(NUM_LAYERS):
            weights = all_weights[layer_idx]
            h, k_new, v_new = clusterfusion.pythia_2b8_decoder_layer(
                h, weights["qkv_weight"], weights["qkv_bias"],
                weights["o_weight"], weights["o_bias"],
                k_caches[layer_idx], v_caches[layer_idx],
                weights["ln_weight"], weights["ln_bias"], cos, sin,
                weights["post_ln_weight"], weights["post_ln_bias"],
                weights["mlp_up_weight"], weights["mlp_up_bias"],
                weights["mlp_down_weight"], weights["mlp_down_bias"],
                pos
            )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.perf_counter()
    
    for step in range(num_tokens):
        pos = current_seq_len + step
        cos, sin = all_cos[pos], all_sin[pos]
        h = hidden[:, -1:, :].contiguous()
        
        for layer_idx in range(NUM_LAYERS):
            weights = all_weights[layer_idx]
            h, k_new, v_new = clusterfusion.pythia_2b8_decoder_layer(
                h, weights["qkv_weight"], weights["qkv_bias"],
                weights["o_weight"], weights["o_bias"],
                k_caches[layer_idx], v_caches[layer_idx],
                weights["ln_weight"], weights["ln_bias"], cos, sin,
                weights["post_ln_weight"], weights["post_ln_bias"],
                weights["mlp_up_weight"], weights["mlp_up_bias"],
                weights["mlp_down_weight"], weights["mlp_down_bias"],
                pos
            )
        
        # Final LN + LM Head (h is [1, HIDDEN_DIM], logits is [1, vocab_size])
        h = final_ln(h)
        logits = lm_head(h)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        hidden = embed_tokens(next_token).unsqueeze(0)  # Make it [1, 1, HIDDEN_DIM]
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    return end_time - start_time

def benchmark_decode_split(model, input_ids, num_tokens, all_weights):
    """Benchmark split kernel decode."""
    import clusterfusion
    
    device = torch.device("cuda:0")
    
    # Get model components
    embed_tokens = model.gpt_neox.embed_in
    final_ln = model.gpt_neox.final_layer_norm
    lm_head = model.embed_out
    
    # Initialize
    with torch.no_grad():
        hidden = embed_tokens(input_ids)
    
    max_seq_len = input_ids.shape[1] + num_tokens + 10
    
    # Precompute all RoPE embeddings
    all_cos, all_sin = precompute_rope_embeddings(max_seq_len, device=device)
    
    # Initialize KV caches
    k_caches = [torch.zeros(max_seq_len, HIDDEN_DIM, device=device, dtype=torch.float16) for _ in range(NUM_LAYERS)]
    v_caches = [torch.zeros(max_seq_len, HIDDEN_DIM, device=device, dtype=torch.float16) for _ in range(NUM_LAYERS)]
    
    # Prefill with HuggingFace
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        
        for layer_idx in range(NUM_LAYERS):
            k, v = past_key_values[layer_idx]
            seq_len = k.shape[2]
            for pos in range(seq_len):
                k_caches[layer_idx][pos] = k[0, :, pos, :].reshape(-1)
                v_caches[layer_idx][pos] = v[0, :, pos, :].reshape(-1)
    
    current_seq_len = input_ids.shape[1]
    
    # Warmup
    for _ in range(WARMUP_RUNS):
        pos = current_seq_len
        cos, sin = all_cos[pos], all_sin[pos]
        h = hidden[:, -1:, :].contiguous()
        
        for layer_idx in range(NUM_LAYERS):
            weights = all_weights[layer_idx]
            h, k_new, v_new = clusterfusion.pythia_2b8_decoder_layer_split(
                h, weights["qkv_weight"], weights["qkv_bias"],
                weights["o_weight"], weights["o_bias"],
                k_caches[layer_idx], v_caches[layer_idx],
                weights["ln_weight"], weights["ln_bias"], cos, sin,
                weights["post_ln_weight"], weights["post_ln_bias"],
                weights["mlp_up_weight"], weights["mlp_up_bias"],
                weights["mlp_down_weight"], weights["mlp_down_bias"],
                pos
            )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.perf_counter()
    
    for step in range(num_tokens):
        pos = current_seq_len + step
        cos, sin = all_cos[pos], all_sin[pos]
        h = hidden[:, -1:, :].contiguous()
        
        for layer_idx in range(NUM_LAYERS):
            weights = all_weights[layer_idx]
            h, k_new, v_new = clusterfusion.pythia_2b8_decoder_layer_split(
                h, weights["qkv_weight"], weights["qkv_bias"],
                weights["o_weight"], weights["o_bias"],
                k_caches[layer_idx], v_caches[layer_idx],
                weights["ln_weight"], weights["ln_bias"], cos, sin,
                weights["post_ln_weight"], weights["post_ln_bias"],
                weights["mlp_up_weight"], weights["mlp_up_bias"],
                weights["mlp_down_weight"], weights["mlp_down_bias"],
                pos
            )
        
        # Final LN + LM Head (h is [1, HIDDEN_DIM], logits is [1, vocab_size])
        h = final_ln(h)
        logits = lm_head(h)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        hidden = embed_tokens(next_token).unsqueeze(0)  # Make it [1, 1, HIDDEN_DIM]
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    return end_time - start_time

def benchmark_decode_hf(model, input_ids, num_tokens):
    """Benchmark HuggingFace decode."""
    # Warmup
    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=model.config.eos_token_id,
                use_cache=True
            )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.perf_counter()
    
    with torch.no_grad():
        _ = model.generate(
            input_ids,
            max_new_tokens=num_tokens,
            do_sample=False,
            pad_token_id=model.config.eos_token_id,
            use_cache=True
        )
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    return end_time - start_time

def main():
    print_header()
    
    model, tokenizer = load_model()
    device = torch.device("cuda:0")
    
    # Pre-extract all weights
    print("\nExtracting weights...")
    all_weights = extract_all_weights(model, device)
    print("Weights extracted")
    
    # Prepare input
    prompt = "The meaning of life is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    print(f"\nPrompt: '{prompt}' ({input_ids.shape[1]} tokens)")
    
    # Results table header
    print("\n" + "-" * 100)
    print(f"{'Tokens':>8} | {'Fused(s)':>10} | {'Split(s)':>10} | {'HF(s)':>10} | "
          f"{'Fused vs HF':>12} | {'Split vs HF':>12} | {'Fused vs Split':>14}")
    print("-" * 100)
    
    for num_tokens in TOKEN_COUNTS:
        try:
            # Benchmark each implementation
            fused_time = benchmark_decode_fused(model, input_ids, num_tokens, all_weights)
            split_time = benchmark_decode_split(model, input_ids, num_tokens, all_weights)
            hf_time = benchmark_decode_hf(model, input_ids, num_tokens)
            
            # Calculate speedups
            fused_vs_hf = hf_time / fused_time
            split_vs_hf = hf_time / split_time
            fused_vs_split = split_time / fused_time
            
            print(f"{num_tokens:>8} | {fused_time:>10.3f} | {split_time:>10.3f} | {hf_time:>10.3f} | "
                  f"{fused_vs_hf:>11.2f}x | {split_vs_hf:>11.2f}x | {fused_vs_split:>13.2f}x")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{num_tokens:>8} | ERROR: {str(e)[:60]}")
    
    print("-" * 100)
    
    # Summary
    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)
    print("""
Fused Kernel:
  - Single kernel launch per decoder layer
  - Uses cudaLaunchCooperativeKernel (for grid.sync())
  - All intermediate results in registers/shared memory

Split Kernel:
  - Two kernel launches per decoder layer  
  - Uses cudaLaunchKernelExC (regular launch with cluster)
  - Kernel launch boundary replaces grid.sync()
  - Intermediate results written to global memory

Fused vs Split comparison shows the overhead of:
  1. Extra kernel launch per layer
  2. Global memory traffic for intermediate tensors (FFN_DIM * 2 bytes = 20KB)
""")

if __name__ == "__main__":
    main()
