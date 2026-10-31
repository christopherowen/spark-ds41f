# DSpark Sequential Markov Fast Path (Rework of Rejected PR #701)

## Why PR #701 Was Rejected
PR #701 attempted to reduce `DSparkSpeculator._sample_sequential` latency at `c8`/`c16` by truncating the sequential Markov loop from $K=5$ steps to $K=4$ or $K=3$ via an opt-in environment flag (`SPARK_DS41F_DSPARK_MAX_STEPS_BY_BATCH`). That approach was rejected because:
1. It introduced an environment switch (violating the "One True Way" invariant).
2. It broke batch invariance (a request received 5 draft tokens at `M=1` and 3–4 at `M=8..16`).
3. Profiling `DSparkSpeculator._sample_sequential` on GB10 (`dgx4`) showed the entire 5-step sequential Markov loop takes only `145–156 us`, so dropping steps 4 and 5 sacrificed valid draft-token acceptances to save `< 50 us/step`.

## What Has Merit and How This Rework Solves It cleanly
We achieve the full **27.5–71.5 us (`18.3%–46.6%`) speedup** on `DSparkSpeculator._sample_sequential` across `M = 1..16` **with all $K=5$ draft steps preserved, zero flags, 100% bit-exact outputs, and full batch invariance**:

1. **Direct contiguous slice when `sample_from_anchor` is `True`:**
   - When `self.sample_from_anchor` is `True` (`num_query_per_req == num_speculative_steps == 5` for DeepSeek-V4.1 Flash DSpark), each request's query block has no leading anchor row and `head_hidden[:num_sample]` is already contiguous in `(req, step)` order (`sample_indices[:num_sample] == [0, 1, ..., num_sample - 1]`).
   - Slicing `head_hidden[:num_sample]` directly in `_sample_sequential` and `_sample_sequential_topk` (and skipping `gather_hidden_rows_calls` registration when `sample_from_anchor` is `True`) eliminates 1 kernel launch and up to `800 KB` (`80 × 5,120 × 2B`) of redundant HBM copy traffic per decode step.

2. **Single-stage non-warp-specialized decode GEMM (`bf16_gemm_decode`, `num_stages=1`) for single-K-block BF16 vocabulary heads:**
   - In `vllm/models/deepseek_v4_1/tilelang/vocab.py`, `BF16_HEAD_CONFIGS` configures the 256-wide DSpark Markov projection (`markov_w2`, `K = 256`) with `block_K = 256` for decode tiles (`block_M in (16, 32)`).
   - Because `k_blocks = K // block_K = 1`, there is only **1 K-loop iteration**. Compiling a 1-iteration loop with warp specialization and `num_stages = 2` split each CTA's 4 warps into producer/consumer warps with `mbarrier` synchronization and doubled shared memory per CTA (`80 KB` vs `40 KB`), cutting SM occupancy on GB10 (`99 KB` SMEM/SM) from **2 CTAs/SM** to **1 CTA/SM** across all `505 CTAs × 5 steps = 2,525 CTAs`.
   - Using `bf16_gemm_decode` (`TL_DISABLE_WARP_SPECIALIZED=True`) and `num_stages = 1` when `k <= config["block_K"]` doubles CTA occupancy and uses all 4 warps for the MMA while accumulating in the exact same `k = 16` order as `block_M = 64, 128`.

## Verification on `dgx4` (`spark-ds41f:19f2c20ed4d6-next-20`)

### CUDA-Graph Replay Latency (`_sample_sequential`, 5 steps, `N=32320, K=256, H=5120`)
| Batch Size `M` | Baseline (`old`) | Reworked (`new`) | Saved | Speedup |
|---:|---:|---:|---:|---:|
| `M = 1` (`c1`) | `156.10 us` | **`84.59 us`** | `-71.51 us` | **`-45.8%`** |
| `M = 2` | `151.95 us` | **`81.20 us`** | `-70.76 us` | **`-46.6%`** |
| `M = 4` | `146.09 us` | **`88.17 us`** | `-57.92 us` | **`-39.6%`** |
| `M = 8` (`c8`) | `145.74 us` | **`107.73 us`** | `-38.01 us` | **`-26.1%`** |
| `M = 16` (`c16`) | `150.01 us` | **`122.48 us`** | `-27.53 us` | **`-18.3%`** |

### Test Suite
- `pytest -o addopts="" --noconftest -q tests/v1/worker/test_dspark_greedy_markov.py tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py`: **`71 passed`**.
