# CS3602 Project: ClusterFusion - Attention-Only Branch

This branch contains **Attention + MLP Up + GELU acceleration** for Pythia-2.8B.

The MLP Down projection is handled separately in the `mlp-only` branch.

## What This Branch Contains

| Component | Implementation | Status |
|-----------|----------------|--------|
| LayerNorm → QKV → RoPE → Attention → Output Proj | ✅ CUDA Kernel | Accelerated |
| Post-LN → MLP Up → GELU | ✅ CUDA Kernel | Accelerated |
| MLP Down Projection | PyTorch | Not accelerated (see `mlp-only` branch) |

## Architecture

```
Input → [CUDA: LayerNorm → QKV → RoPE → Attention → Output Proj]
     → [CUDA: Post-LN → MLP Up → GELU]
     → [PyTorch: MLP Down → Residual] → Output
```

## API

```python
import clusterfusion

# Attention-Only kernel
# Returns: attn_output, mlp_intermediate, k_new, v_new
attn_output, mlp_intermediate, k_new, v_new = clusterfusion.pythia_2b8_attention_only(
    input,              # [1, 1, hidden_dim]
    weight_qkv, bias_qkv,
    weight_o, bias_o,
    k_cache, v_cache,   # KV cache
    ln_weight, ln_bias,
    cos, sin,           # RoPE embeddings
    post_ln_weight, post_ln_bias,
    mlp_up_weight, mlp_up_bias,
    current_seq_len
)

# Complete forward with PyTorch MLP Down
mlp_down = F.linear(mlp_intermediate, mlp_down_weight, mlp_down_bias)
output = input + attn_output + mlp_down  # Parallel residual
```

## Ablation Results (from full fusion experiments)

The Attention+MLPUp branch contributes approximately **88% of total speedup**:

| Configuration | Time (ms) | Speedup |
|---------------|-----------|---------|
| Full PyTorch | 9.67 | 1.00x |
| CUDA Attn+Up + PyTorch Down | 5.90 | **1.64x** |
| Full CUDA Fused | 4.90 | 1.97x |

**Key Finding**: Accelerating Attention+MLPUp provides the majority of speedup because:
1. It accounts for ~91% of total execution time
2. QKV projection, Output projection, and MLP Up are large matrix operations
3. Flash Decoding is compute-intensive

## Branch Time Breakdown

| Branch | PyTorch Time | Contribution to Speedup |
|--------|--------------|-------------------------|
| Attention + MLP Up | 8.84 ms (91%) | **88%** |
| MLP Down | 1.27 ms (13%) | 12% |

## Files

| File | Description |
|------|-------------|
| `kernel_attention.cuh` | Attention + MLP Up kernel implementation |
| `pythia_attention_dispatch.cu` | Kernel dispatch for attention-only |
| `tests/test_attention_only.py` | Test and benchmark for attention kernel |

## Build & Test

```bash
# Install
pip install -e .

# Test
python tests/test_attention_only.py
```

## Related Branches

| Branch | Description |
|--------|-------------|
| `main` | Attention-only acceleration (this branch) |
| `mlp-only` | MLP-only acceleration |
| `full-fusion-backup` | Complete fused kernel (Attention + MLP) |

## Key Optimizations

1. **Fused LayerNorm + QKV Projection**: Single pass with online normalization
2. **RoPE in Registers**: Rotary position embedding applied in-place
3. **Flash Decoding**: Online softmax with running max for numerical stability
4. **TMA Weight Loading**: Hardware-accelerated tensor memory access
5. **Cluster-level Reduction**: Distributed atomicAdd across SM cluster

## Next Steps for Attention Optimization

Potential improvements:
1. Fuse with MLP Down for complete layer fusion (see `full-fusion-backup`)
2. Multi-query attention support
3. Paged KV cache support
4. Speculative decoding integration

---

For the complete fused implementation, see `full-fusion-backup` branch.
