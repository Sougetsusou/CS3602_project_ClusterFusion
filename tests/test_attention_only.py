#!/usr/bin/env python3
"""
Test: Attention-Only Kernel for Pythia-2.8B

This tests the Attention + MLP Up + GELU kernel.
The MLP Down projection is done in PyTorch.
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model configuration
MODEL_NAME = "EleutherAI/pythia-2.8b"
HIDDEN_DIM = 2560
NUM_HEADS = 32
HEAD_DIM = 80
ROTARY_DIM = 20
FFN_DIM = 10240
NUM_LAYERS = 32

def precompute_rope(max_pos, device="cuda:0"):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, ROTARY_DIM, 2, device=device).float() / ROTARY_DIM))
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    cos = torch.cat([cos, torch.ones((max_pos, HEAD_DIM - ROTARY_DIM), device=device)], -1)
    sin = torch.cat([sin, torch.zeros((max_pos, HEAD_DIM - ROTARY_DIM), device=device)], -1)
    return cos, sin

def test_attention_only():
    print("=" * 70)
    print("Test: Attention-Only Kernel for Pythia-2.8B")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    
    device = torch.device("cuda:0")
    layer = model.gpt_neox.layers[0]
    
    # Setup
    seq_len = 64
    max_seq_len = seq_len + 100
    all_cos, all_sin = precompute_rope(max_seq_len, device)
    cos, sin = all_cos[seq_len], all_sin[seq_len]
    
    k_cache = torch.zeros(max_seq_len, HIDDEN_DIM, device=device, dtype=torch.float16)
    v_cache = torch.zeros(max_seq_len, HIDDEN_DIM, device=device, dtype=torch.float16)
    
    # Input
    input_hidden = torch.randn(1, 1, HIDDEN_DIM, device=device, dtype=torch.float16)
    
    # CUDA kernel
    import clusterfusion
    
    qkv_weight = layer.attention.query_key_value.weight.T.contiguous()
    qkv_bias = layer.attention.query_key_value.bias.contiguous()
    o_weight = layer.attention.dense.weight.T.contiguous()
    o_bias = layer.attention.dense.bias.contiguous()
    ln_weight = layer.input_layernorm.weight.contiguous()
    ln_bias = layer.input_layernorm.bias.contiguous()
    post_ln_weight = layer.post_attention_layernorm.weight.contiguous()
    post_ln_bias = layer.post_attention_layernorm.bias.contiguous()
    mlp_up_weight = layer.mlp.dense_h_to_4h.weight.T.contiguous()
    mlp_up_bias = layer.mlp.dense_h_to_4h.bias.contiguous()
    
    # Run CUDA kernel
    attn_output, mlp_intermediate, k_new, v_new = clusterfusion.pythia_2b8_attention_only(
        input_hidden,
        qkv_weight, qkv_bias,
        o_weight, o_bias,
        k_cache, v_cache,
        ln_weight, ln_bias,
        cos, sin,
        post_ln_weight, post_ln_bias,
        mlp_up_weight, mlp_up_bias,
        seq_len
    )
    
    print(f"\nOutput shapes:")
    print(f"  attn_output:      {attn_output.shape}")
    print(f"  mlp_intermediate: {mlp_intermediate.shape}")
    print(f"  k_new:            {k_new.shape}")
    print(f"  v_new:            {v_new.shape}")
    
    # Complete the forward pass with PyTorch MLP Down
    mlp_down_weight = layer.mlp.dense_4h_to_h.weight.half()
    mlp_down_bias = layer.mlp.dense_4h_to_h.bias.half()
    
    mlp_down = F.linear(mlp_intermediate.unsqueeze(0), mlp_down_weight, mlp_down_bias)
    output = input_hidden.squeeze(0) + attn_output + mlp_down
    
    print(f"  final_output:     {output.shape}")
    print(f"\n✅ Kernel executed successfully!")
    
    # Benchmark
    print("\n" + "-" * 70)
    print("Benchmark: CUDA Attention vs PyTorch Attention")
    print("-" * 70)
    
    WARMUP = 50
    ITERS = 200
    
    # CUDA benchmark
    for _ in range(WARMUP):
        _ = clusterfusion.pythia_2b8_attention_only(
            input_hidden, qkv_weight, qkv_bias, o_weight, o_bias,
            k_cache, v_cache, ln_weight, ln_bias, cos, sin,
            post_ln_weight, post_ln_bias, mlp_up_weight, mlp_up_bias, seq_len
        )
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(ITERS):
        _ = clusterfusion.pythia_2b8_attention_only(
            input_hidden, qkv_weight, qkv_bias, o_weight, o_bias,
            k_cache, v_cache, ln_weight, ln_bias, cos, sin,
            post_ln_weight, post_ln_bias, mlp_up_weight, mlp_up_bias, seq_len
        )
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - start) / ITERS * 1000
    
    print(f"CUDA Attention+MLPUp: {cuda_time:.4f} ms per layer")
    print(f"For {NUM_LAYERS} layers:  {cuda_time * NUM_LAYERS:.4f} ms")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_attention_only()

