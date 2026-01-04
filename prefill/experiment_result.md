$ PROMPT_LENS="512,1024,2048" RUNS=3 MODEL_DTYPE=bf16 \
python benchmarks/benchmark_pythia_mini_flash.py
=== Pythia-2.8B Model-Level Prefill Benchmark (MiniFlash vs Eager vs FlashAttn2) ===
Model=EleutherAI/pythia-2.8b device=cuda dtype=torch.bfloat16
PromptLens=[512, 1024, 2048] runs=3 warmup=2
MiniFlash blocks: q_block=128 kv_block=128

==============================================================================================================
Prompt length: 512
==============================================================================================================

--- Baseline (eager) ---

--- FlashAttention-2 ---

--- MiniFlash (patched) ---

--- Results ---
--------------------------------------------------------------------------------------------------------------
| Config                 |   Prefill ms |      Std |    PeakMem |    TTFT ms |        PPL |
--------------------------------------------------------------------------------------------------------------
| Baseline (eager)       |        21.40 |     0.08 |   5.28 GiB |      56.27 |   1.121644 |
| FlashAttention-2       |        19.73 |     0.01 |   5.23 GiB |      38.96 |   1.122010 |
| MiniFlash (patched)    |        65.43 |     1.01 |   5.23 GiB |      73.67 |   1.120434 |
--------------------------------------------------------------------------------------------------------------
Prefill speedup (FlashAttention-2 vs eager): 1.09x
Prefill speedup (MiniFlash vs eager): 0.33x
Peak mem ratio (eager/flash): 1.01x
Peak mem ratio (eager/mini): 1.01x

==============================================================================================================
Prompt length: 1024
==============================================================================================================

--- Baseline (eager) ---

--- FlashAttention-2 ---

--- MiniFlash (patched) ---

--- Results ---
--------------------------------------------------------------------------------------------------------------
| Config                 |   Prefill ms |      Std |    PeakMem |    TTFT ms |        PPL |
--------------------------------------------------------------------------------------------------------------
| Baseline (eager)       |        55.12 |     0.07 |   5.53 GiB |      69.09 |   1.121644 |
| FlashAttention-2       |        35.44 |     0.04 |   5.28 GiB |      46.21 |   1.122010 |
| MiniFlash (patched)    |       226.24 |     2.11 |   5.28 GiB |     231.84 |   1.120434 |
--------------------------------------------------------------------------------------------------------------
Prefill speedup (FlashAttention-2 vs eager): 1.56x
Prefill speedup (MiniFlash vs eager): 0.24x
Peak mem ratio (eager/flash): 1.05x
Peak mem ratio (eager/mini): 1.05x

==============================================================================================================
Prompt length: 2048
==============================================================================================================

--- Baseline (eager) ---

--- FlashAttention-2 ---

--- MiniFlash (patched) ---

--- Results ---
--------------------------------------------------------------------------------------------------------------
| Config                 |   Prefill ms |      Std |    PeakMem |    TTFT ms |        PPL |
--------------------------------------------------------------------------------------------------------------
| Baseline (eager)       |       155.62 |     0.30 |   6.51 GiB |     164.66 |   1.121644 |
| FlashAttention-2       |        62.57 |     0.07 |   5.38 GiB |      76.04 |   1.122010 |
| MiniFlash (patched)    |       844.32 |     1.90 |   5.38 GiB |     851.10 |   1.120434 |
--------------------------------------------------------------------------------------------------------------
Prefill speedup (FlashAttention-2 vs eager): 2.49x
Prefill speedup (MiniFlash vs eager): 0.18x
Peak mem ratio (eager/flash): 1.21x
Peak mem ratio (eager/mini): 1.21x