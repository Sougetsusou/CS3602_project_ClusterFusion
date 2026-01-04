# Prefill Optimization (Pythia-2.8B): FlashAttention Reproduction + Benchmarks

This repository contains a reproducible study of **prefill acceleration** for the unified baseline model **Pythia-2.8B** (GPT-NeoX) under the constraint of **no parameter changes / no training**.

Personal reproduction focus: **FlashAttention (algorithmic core)**.

We provide:

1. **Baseline model-level benchmarks** comparing:
   - Transformers **eager** attention
   - Transformers **flash_attention_2** (FlashAttention-2 kernel)
   - Our **MiniFlash** implementation (pure PyTorch online-softmax + tiling) integrated into GPT-NeoX via runtime patching

2. **Dataset-based PPL evaluation** on **wikitext** and **pg-19**.

3. **Attention-only benchmarks** that isolate the attention kernel behavior.

---

## 1. Environment

This repo includes an exact environment specification:

- `environment.yml` (env name: `nlp-torch2.8-py312`)

Create and activate:

```bash
conda env create -f environment.yml
conda activate nlp-torch2.8-py312
```

Key packages (from `environment.yml`):
- `torch==2.8.0`
- `transformers==4.57.3`
- `flash-attn==2.8.3`
- `datasets==4.4.2`

---

## 2. FlashAttention reproduction (what we implemented)

### 2.1 MiniFlash (pure PyTorch)

- File: `models/mini_flash_attention.py`
- Implements:
  - blockwise attention
  - online softmax (running max + running normalizer)
  - avoids materializing the full attention matrix
- Causal behavior matches flash-attn’s doc: **bottom-right aligned** causal mask when `q_len != k_len`.

### 2.2 Integrating into Pythia-2.8B (Route A monkey patch)

- File: `models/patch_gpt_neox_mini_flash.py`
- Transformers 4.57.3 GPT-NeoX selects attention backend via `ALL_ATTENTION_FUNCTIONS`.
- We register a new backend name: `mini_flash`.

This makes it possible to run the baseline model Pythia-2.8B with:
- `attn_implementation="mini_flash"`

---

## 3. Run: model-level benchmark (baseline requirement)

This is the main benchmark for the homework baseline model.

```bash
MODEL_DTYPE=bf16 PROMPT_LENS="512,1024,2048,4096,8192" RUNS=5 WARMUP=2 \
python benchmarks/benchmark_pythia_mini_flash.py
```

Output per prompt length:
- Prefill latency (ms/prompt)
- Peak CUDA allocated memory during prefill
- Prompt perplexity (teacher forcing)

Enable MiniFlash debug prints (optional):

```bash
PRINT_MINIFLASH_DEBUG=1 PROMPT_LENS="512" RUNS=1 WARMUP=0 MODEL_DTYPE=bf16 \
python benchmarks/benchmark_pythia_mini_flash.py
```

#### 3.1 Prefill benchmark results (bf16)

| Prompt Len | Eager Prefill (ms) | FlashAttn-2 Prefill (ms) | MiniFlash Prefill (ms) | Eager PeakMem | Flash PeakMem | Mini PeakMem |
|---:|---:|---:|---:|---:|---:|---:|
| 512  | 21.46 | 19.87 | 64.59 | 5.28 GiB | 5.23 GiB | 5.23 GiB |
| 1024 | 55.21 | 35.54 | 224.51 | 5.53 GiB | 5.28 GiB | 5.28 GiB |
| 2048 | 155.85 | 62.73 | 847.15 | 6.51 GiB | 5.38 GiB | 5.38 GiB |
| 4096 | 512.94 | 128.02 | 3320.25 | 10.37 GiB | 5.58 GiB | 5.58 GiB |
| 8192 | OOM | 283.68 | 13106.22 | — | 5.98 GiB | 5.98 GiB |

These numbers are a representative run on a single NVIDIA GPU (31 GiB VRAM). FlashAttention-2 achieves increasing speedups with sequence length while keeping memory usage flat, whereas MiniFlash reproduces the memory benefit but is slower due to the lack of fused CUDA kernels.

---

## 4. Run: dataset PPL evaluation (wikitext / pg-19)

This satisfies the requirement to evaluate PPL on standard datasets.

### 4.1 Wikitext-2 (example)

```bash
MODEL_DTYPE=bf16 python evaluate_dataset_ppl.py --dataset wikitext2 --attn eager --split test
MODEL_DTYPE=bf16 python evaluate_dataset_ppl.py --dataset wikitext2 --attn flash_attention_2 --split test
MODEL_DTYPE=bf16 python evaluate_dataset_ppl.py --dataset wikitext2 --attn mini_flash --split test
```

**Recorded result (wikitext2 / test / num_samples=1 / max_tokens=2048 / seqlen=2048 / stride=1024 / bf16 / cuda):**

| Attention backend | PPL |
|---|---:|
| eager | 9.833123 |
| flash_attention_2 | 9.460898 |
| mini_flash | 9.456972 |

### 4.2 PG-19 (single sample)

The assignment allows using a single long sample for pg-19.

```bash
MODEL_DTYPE=bf16 python evaluate_dataset_ppl.py --dataset pg19 --attn eager --split test --num_samples 1 --max_tokens 4096
MODEL_DTYPE=bf16 python evaluate_dataset_ppl.py --dataset pg19 --attn flash_attention_2 --split test --num_samples 1 --max_tokens 4096
MODEL_DTYPE=bf16 python evaluate_dataset_ppl.py --dataset pg19 --attn mini_flash --split test --num_samples 1 --max_tokens 4096
```

**Recorded result (pg19 / test / num\_samples=1 / max\_tokens=2048 / seqlen=2048 / stride=1024 / bf16 / cuda):**

| Attention backend | PPL |
|---|---:|
| eager | 8.844127 |
| flash_attention_2 | 8.776810 |
| mini_flash | 8.798897 |

Notes:
- Increase `--max_tokens` to 8192 if your GPU allows.
- PPL is computed over a token stream using a chunked sliding-window evaluation (`--seqlen` / `--stride`).

---

## 5. Run: attention-only benchmark

This isolates the attention algorithm to highlight memory scaling.

```bash
DTYPE=bf16 LENS=256,512,1024,2048 \
python benchmarks/benchmark_mini_flash_attention.py
```

---

## 6. Files

- `models/mini_flash_attention.py`: MiniFlash implementation (our reproduction)
- `models/patch_gpt_neox_mini_flash.py`: runtime patch to integrate MiniFlash into GPT-NeoX
- `benchmarks/benchmark_pythia_mini_flash.py`: model-level baseline benchmark (eager vs flash2 vs mini)
- `benchmarks/benchmark_mini_flash_attention.py`: attention-only benchmark
- `evaluate_dataset_ppl.py`: dataset-based PPL evaluation (wikitext / pg-19)
- `TECHNICAL_REPORT.md`: markdown technical report

---

## 7. Notes

- MiniFlash is significantly slower than FlashAttention-2 because it is not a fused CUDA kernel.
- Nevertheless, MiniFlash is a valid reproduction of FlashAttention’s **algorithmic mechanism** and is integrated into the baseline model for evaluation.

---

## 8. References

- Tri Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.
- Tri Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. arXiv 2023.
