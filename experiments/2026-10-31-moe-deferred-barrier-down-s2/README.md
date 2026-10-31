# Defer `grouped_fp8_fp4_gemm` Barrier Init Past Empty-CTA Return & Use 2 Stages (`2 CTAs/SM`) for Decode `down` GEMM

## Summary

This experiment reduces the fixed CTA-prologue and wave-tail overhead of the TileLang routed-expert `FC1` (`gate_up`) + `SwiGLU` + `FC2` (`down`) stream in `vllm/models/deepseek_v4_1/tilelang/moe.py`:

1. **Deferred mbarrier Initialization & Batched Prologue (`with T.sblock("gemm"):` in `grouped_fp8_fp4_gemm`):**
   - Previously, `loaded = T.alloc_barrier([threads] * stages)` was declared in the outer `T.Kernel` block, causing TileLang's `LowerSharedBarrier` pass to emit `loaded[s].init(128)`, `tl::fence_barrier_init()` (`fence.mbarrier_init.release.cluster`), and `__syncthreads()` **before** `if (expert < 0) return;`.
   - At typical speculative decode batch sizes (`rows = 6, topk = 6`, `36` routes over `36` layout tiles, where only `6..36` tiles are live), `FC2` launches `40 x 36 = 1,440` CTAs and `FC1` launches `9 x 36 = 324` CTAs. Under `e = 6` active experts, `1,200` of `1,440` `FC2` CTAs (`83.3%`) and `270` of `324` `FC1` CTAs (`83.3%`) are empty (`expert < 0`), yet every empty CTA paid shared-memory barrier initialization, a cluster barrier release fence, and `__syncthreads()` before returning.
   - Moving `loaded = T.alloc_barrier([threads] * stages)` into a child `with T.sblock("gemm"):` after `if expert < 0: T.thread_return()` and after issuing the `Order`, `SFBase`, and `Narrow` global loads:
     - Allows empty CTAs (`expert < 0`) to exit immediately after `pdl.wait()` / `pdl.chain()` with **zero** barrier initializations, **zero** cluster barrier fences, and **zero** `__syncthreads()`.
     - Batches `Order`, `SFBase`, and `Narrow` global loads into a single memory wave before the barrier-init `__syncthreads()`, so writing `rows_shared` and initializing `loaded` share a single `__syncthreads()`.

2. **Two-Stage (`down_num_stages = 2`, `2 CTAs/SM`) Decode `down` (`FC2`) GEMM (`MoeExpertsV41Plan`):**
   - For decode routed experts (`block_M = 32`), `FC2` (`down`) has a short inner dimension (`K = 576` = 4 full K-blocks + 64-col tail on TP4; `K = 768` = 6 K-blocks on TP3) and a wide output (`N = 5120` = `40` N-blocks per expert).
   - At `num_stages = 3`, `FC2` allocates `64,768` bytes of shared memory (`> 49.5 KiB`), restricting occupancy on GB10 (`SM121`, `99 KiB`/SM) to **1 CTA/SM** (`48` CTAs/wave) and exposing prologue fill and epilogue BF16 store drain on every wave of the 5-iteration K-loop.
   - Setting `down_num_stages = 2` for decode tile height (`block_M = 32`, keeping `num_stages = 3` for `gate_up` and `prefill_down`) lowers `FC2` shared memory to `43,008` bytes (`<= 49.5 KiB`), doubling residency to **2 CTAs/SM (`96` CTAs/wave)** so each SM overlaps one CTA's K-loop with the co-resident CTA's prologue fill and epilogue BF16 store.

3. **Unrolled 2-Element Accumulator Reductions in `moe_swiglu_cast`:**
   - Replaces `T.vectorized(vec)` with `T.unroll(vec)` on scalar accumulator updates in `moe_swiglu_cast`, eliminating TileLang's `legalize_vectorized_loop.cc` fallback warning during JIT compilation.

## Invariants

- **One True Way:** No environment variables, feature flags, or fallback paths.
- **Bit-Exact & Batch-Invariant:** `loaded` barrier placement and `num_stages` only affect asynchronous copy staging and CTA occupancy; the MMA accumulation order (`ko = 0 .. k_blocks - 1`) and epilogue rounding are 100% bit-identical across all batch sizes and TP degrees.
- **PDL Compliant:** `0` PDL source-check violations across all 64 launched kernels.

## Verification & Benchmark Receipts (`dgx4`, `spark-ds41f:19f2c20ed4d6-next-20`)

### 1. Unit Tests & PDL Verification

```text
pytest -q -o addopts="" --noconftest -p tests.kernels.deepseek_v41_pdl_sources \
  tests/kernels/moe/test_deepseek_v41_tilelang_moe.py \
  tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py \
  tests/v1/worker/test_prepare_kernels.py

127 passed in 48.99s
PDL sources: 64 kernels launched, 55 with PDL, 0 violations
```

### 2. Cold-Weight `expert_rate.py` Benchmark (Target TP4: `E=256, topk=6, H=5120, I=576`, 8-copy 272 MB cold weight rotation)

Measured in the same container session against unmodified `rc20-exact` (`18/18` operating points improved E2E and on `FC2`):

| `(rows, active_experts)` | Baseline `FC1` (us) | Patch `FC1` (us) | $\Delta$ `FC1` (us) | Baseline `FC2` (us) | Patch `FC2` (us) | $\Delta$ `FC2` (us) | Baseline `E2E` (us) | Patch `E2E` (us) | $\Delta$ `E2E` (us) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `(1, 6)` | 26.86 | 26.72 | **-0.14** | 25.09 | 24.48 | **-0.61** | 59.30 | 58.37 | **-0.93** |
| `(2, 6)` | 26.78 | 26.78 | **0.00** | 26.85 | 25.44 | **-1.41** | 60.45 | 58.98 | **-1.47** |
| `(2, 12)` | 36.10 | 35.87 | **-0.23** | 34.43 | 33.47 | **-0.96** | 76.29 | 75.52 | **-0.77** |
| `(4, 6)` | 27.97 | 26.78 | **-1.19** | 29.50 | 25.76 | **-3.74** | 64.00 | 59.17 | **-4.83** |
| `(4, 16)` | 45.79 | 45.82 | +0.03 | 43.39 | 40.96 | **-2.43** | 94.50 | 92.67 | **-1.83** |
| `(4, 24)` | 63.36 | 63.33 | **-0.03** | 56.10 | 53.76 | **-2.34** | 124.70 | 121.63 | **-3.07** |
| `(6, 6)` | 28.00 | 26.82 | **-1.18** | 31.42 | 26.05 | **-5.37** | 65.06 | 59.85 | **-5.21** |
| `(6, 9)` | 31.81 | 31.10 | **-0.71** | 35.23 | 29.98 | **-5.25** | 72.10 | 67.20 | **-4.90** |
| `(6, 12)` | 36.67 | 36.03 | **-0.64** | 38.94 | 34.43 | **-4.51** | 80.58 | 76.60 | **-3.98** |
| `(6, 15)` | 43.90 | 42.98 | **-0.92** | 43.52 | 39.63 | **-3.89** | 92.90 | 88.34 | **-4.56** |
| `(6, 18)` | 49.98 | 49.02 | **-0.96** | 47.49 | 43.65 | **-3.84** | 102.72 | 98.97 | **-3.75** |
| `(6, 21)` | 56.70 | 55.62 | **-1.08** | 52.74 | 48.60 | **-4.14** | 114.62 | 110.87 | **-3.75** |
| `(6, 24)` | 63.26 | 62.66 | **-0.60** | 56.86 | 53.24 | **-3.62** | 124.67 | 121.72 | **-2.95** |
| `(6, 30)` | 78.75 | 78.34 | **-0.41** | 67.62 | 63.96 | **-3.66** | 150.72 | 147.84 | **-2.88** |
| `(6, 36)` | 92.00 | 91.58 | **-0.42** | 76.96 | 73.73 | **-3.23** | 173.44 | 170.30 | **-3.14** |
| `(12, 28)` | 74.56 | 74.40 | **-0.16** | 72.90 | 64.37 | **-8.53** | 153.28 | 146.88 | **-6.40** |
| `(12, 48)` | 123.39 | 122.05 | **-1.34** | 109.76 | 94.99 | **-14.77** | 240.32 | 223.94 | **-16.38** |
| `(16, 36)` | 95.30 | 94.94 | **-0.36** | 91.14 | 81.24 | **-9.90** | 194.24 | 186.17 | **-8.07** |

### 3. Cold-Weight `expert_rate.py` Benchmark (Drafter TP4: `E=128, topk=3, H=5120, I=576`)

| `(rows, active_experts)` | Baseline `FC1` (us) | Patch `FC1` (us) | Baseline `FC2` (us) | Patch `FC2` (us) | Baseline `E2E` (us) | Patch `E2E` (us) | $\Delta$ `E2E` (us) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `(1, 3)` | 20.64 | 20.42 | 18.88 | 18.30 | 45.76 | 44.74 | **-1.02** |
| `(2, 3)` | 20.45 | 20.48 | 19.33 | 18.30 | 45.89 | 44.80 | **-1.09** |
| `(2, 6)` | 26.82 | 26.75 | 25.09 | 24.48 | 56.96 | 55.90 | **-1.06** |
| `(4, 12)` | 35.94 | 35.87 | 34.43 | 33.47 | 74.88 | 73.79 | **-1.09** |
| `(6, 6)` | 26.82 | 26.75 | 27.65 | 25.25 | 60.48 | 57.28 | **-3.20** |
| `(6, 9)` | 30.98 | 30.85 | 31.33 | 29.22 | 67.33 | 64.96 | **-2.37** |
| `(6, 12)` | 36.10 | 35.87 | 35.23 | 33.47 | 75.94 | 74.24 | **-1.70** |
| `(6, 18)` | 48.90 | 48.90 | 44.54 | 43.01 | 98.40 | 97.39 | **-1.01** |
