# FlashAttention Reproduction on Pythia-2.8B (GPT-NeoX)

**Project goal (个人部分复现):** reproduce one inference acceleration method without changing model parameters, and evaluate speed / memory / quality on the unified baseline **Pythia-2.8B**.

This project reproduces the *algorithmic core* of **FlashAttention** (blockwise attention + online softmax, avoiding explicit materialization of the full attention matrix) and integrates it into the HuggingFace **GPT-NeoX / Pythia-2.8B** model via runtime patching.

> Important scope note: our reproduction is **forward-only** and implemented in **pure PyTorch** (no fused CUDA kernel). Therefore it is significantly slower than the official **flash-attn2** kernel, but it is mathematically correct and demonstrates the expected memory behavior.

---

## 1. Environment and Baseline

- **Model:** `EleutherAI/pythia-2.8b`
- **Framework:** PyTorch + HuggingFace Transformers
- **Transformers version:** 4.57.3
- **Torch version:** 2.8.0
- **flash-attn:** 2.8.3
- **Precision used in experiments:** bf16 (`MODEL_DTYPE=bf16`)
- **Hardware:** NVIDIA GPU (31.35 GiB VRAM)

Environment is specified in `environment.yml`.

---

## 2. Method: Mini FlashAttention (Algorithm Reproduction)

### 2.1 Reference attention
We define a reference attention computation (materializes the score matrix):

\[
\text{Attn}(Q,K,V) = \text{softmax}(QK^\top \cdot s + M) V
\]

where `s = 1/sqrt(d)` and `M` is an (optional) additive mask.

### 2.2 FlashAttention idea
FlashAttention avoids storing the full \(T\times T\) attention matrix by computing attention in **tiles** and maintaining an **online softmax** per query row:

- maintain running maximum \(m\)
- maintain running normalization \(l\)
- maintain running output accumulator \(o\)

When streaming over key blocks, update \(m,l,o\) with numerically stable rescaling.

### 2.3 Causal alignment detail
The official flash-attn API specifies that for `q_len != k_len`, the **causal mask is aligned to the bottom-right corner** of the attention matrix.

Let `offset = k_len - q_len`. For query row `i` and key column `j`:

- keep if \( j \le i + \text{offset} \)
- mask if \( j > i + \text{offset} \)

Our implementation matches this alignment.

### 2.4 Implementation files
- `models/mini_flash_attention.py`
  - `attention_reference(...)`
  - `attention_mini_flash(...)` (blockwise + online softmax)
- `models/patch_gpt_neox_mini_flash.py`
  - registers a new GPT-NeoX attention backend named `mini_flash` by inserting into `ALL_ATTENTION_FUNCTIONS`
  - ensures output layout matches Transformers `eager_attention_forward` expectation: output is `(B, q_len, H, D)`

---

## 3. Experiments

We report model-level prefill benchmarks on Pythia-2.8B for three configurations:

1. **Baseline (eager)**: Transformers `attn_implementation="eager"`
2. **FlashAttention-2**: Transformers `attn_implementation="flash_attention_2"`
3. **MiniFlash (patched)**: Transformers `attn_implementation="mini_flash"` after patching

### 3.1 Metrics
- **Prefill latency (ms/prompt)**: wall-clock time for one forward pass over the prompt (TTFT proxy)
- **Peak memory (GiB)**: `torch.cuda.max_memory_allocated()` during prefill
- **PPL**: teacher-forcing perplexity on the first `PPL_EVAL_TOKENS` tokens of the prompt (computed from logits)

### 3.2 How to run
Model-level sweep:

```bash
MODEL_DTYPE=bf16 PROMPT_LENS="512,1024,2048,4096,8192" RUNS=5 WARMUP=2 \
python benchmarks/benchmark_pythia_mini_flash.py
```

Attention-only benchmark:

```bash
DTYPE=bf16 LENS=256,512,1024,2048 python benchmarks/benchmark_mini_flash_attention.py
```

---

## 4. Results (Model-level, Prefill)

Below is a representative sweep (bf16). The key findings are:

- FlashAttention-2 achieves large prefill speedups that **increase with sequence length**.
- Baseline eager attention **OOMs at 8192** on this GPU, while FlashAttention-2 and MiniFlash succeed.
- MiniFlash preserves quality (PPL close to baseline) but is slower because it is not a fused kernel.

### 4.1 Prefill latency and memory vs length

| Prompt Len | Eager Prefill (ms) | FlashAttn-2 Prefill (ms) | MiniFlash Prefill (ms) | Eager PeakMem | Flash PeakMem | Mini PeakMem |
|---:|---:|---:|---:|---:|---:|---:|
| 512  | 21.46 | 19.87 | 64.59 | 5.28 GiB | 5.23 GiB | 5.23 GiB |
| 1024 | 55.21 | 35.54 | 224.51 | 5.53 GiB | 5.28 GiB | 5.28 GiB |
| 2048 | 155.85 | 62.73 | 847.15 | 6.51 GiB | 5.38 GiB | 5.38 GiB |
| 4096 | 512.94 | 128.02 | 3320.25 | 10.37 GiB | 5.58 GiB | 5.58 GiB |
| 8192 | OOM | 283.68 | 13106.22 | — | 5.98 GiB | 5.98 GiB |

### 4.2 Speedup summary
- FlashAttention-2 vs eager speedup:
  - 512: 1.08×
  - 1024: 1.55×
  - 2048: 2.48×
  - 4096: 4.01×

- MiniFlash vs eager:
  - slower than eager (0.15×–0.33×) due to Python-level blocking (no fused kernels)

### 4.3 Quality (PPL)

#### 4.3.1 Wikitext-2
On the Wikitext-2 test set (single sample, 2048 tokens, stride 1024) we obtain:
- eager: 9.833123
- flash_attention_2: 9.460898
- mini_flash: 9.456972

MiniFlash achieves essentially the same perplexity as the optimized FlashAttention-2 kernel, and both slightly outperform the baseline eager implementation.

#### 4.3.2 PG-19 (single sample, 2048 tokens, stride=1024)
We evaluate on a single long sample from PG-19 with a sliding window approach:
- eager: 8.844127
- flash_attention_2: 8.776810
- mini_flash: 8.798897

This shows that MiniFlash maintains comparable perplexity to both the eager and FlashAttention-2 baselines on longer sequences, with all three implementations producing very similar results.

---

## 5. Discussion

### 5.1 Why FlashAttention-2 is faster
FlashAttention-2 uses fused CUDA kernels that:
- reduce HBM traffic by tiling and recomputation
- avoid materializing the full attention matrix in GPU memory
- improve parallelism and work partitioning

These benefits become larger as sequence length increases, matching the observed scaling.

### 5.2 Why MiniFlash is slower
Although MiniFlash reproduces FlashAttention’s algorithm (online softmax + tiling), it is implemented in Python/PyTorch loops:
- many small matmul calls per block
- kernel launch overhead
- no fusion

Therefore it is much slower than optimized kernels.

### 5.3 Memory behavior and long-context capability
The baseline eager implementation OOMs at 8192, while FlashAttention-2 and MiniFlash run successfully with ~6 GiB peak allocated during prefill in this setup.

---

## 6. Conclusion

We reproduced the FlashAttention algorithmic core (blockwise attention with online softmax) and integrated it into the baseline Pythia-2.8B GPT-NeoX model. Model-level experiments show:

- FlashAttention-2 provides large and length-dependent prefill speedups (up to ~4× at 4096 tokens).
- Eager attention OOMs at 8192, while FlashAttention-2 runs successfully.
- Our MiniFlash reproduction preserves model quality (PPL unchanged) and matches the memory-efficient behavior, but is slower due to lack of fused kernels.

---

## References
- Tri Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.
- Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. arXiv 2023.
