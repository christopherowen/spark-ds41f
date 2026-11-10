# Routed MoE Dispatch Warp-Shuffle Scan, Grouped GEMM Sync, and Empty-Slab SwiGLU Skip

## Overview

- **Base commit**: `8e5ea208f3d22bc69df91ace94d6975932c858a8` (Candidate `1107`, image `spark-ds41f:1107`)
- **Upstream vLLM base**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`
- **vLLM patch series**: `experiments/2026-11-08-moe-swiglu-and-dual-stream-graph/vllm/series` (`642` active patches)
  - Patch added: `0762-perf-dsv41-skip-empty-moe-swiglu-slabs-and-restore-p.patch`
  - Patch head: `2756478b60b90700b9918b457aeb38657da637b5` -> `cb4657074ed108780180414e3a6ce8842bb8ff94`
  - Expected tree: `ed1bdb7a7684da33d403a055521ac0c4dd7ca670` -> `e598e4a06cd105a6eeed4c22bd7b1ae8972debe3`
- **Hardware**: `sp8` (`spark-0016`, NVIDIA GB10 `sm_121a`)
- **Causal variables (`vllm/models/deepseek_v4_1/tilelang/moe.py`)**:
  1. **Two-level warp-shuffle prefix scan in `_dispatch_macros.layout()`**:
     - Previously, `_dispatch_macros.layout()` computed the 512-thread per-expert padded route prefix sum via a 9-step Hillis-Steele shared-memory scan with 18 CTA-wide `__syncthreads()` barriers (25 total in fused `moe_dispatch_v41`).
     - Replaced with a two-level warp-shuffle prefix scan (`__shfl_up_sync` within each warp, a single shared-memory warp-total write and `__syncthreads()`, and an in-register warp-prefix scan via `__shfl_up_sync` + `__shfl_sync`), reducing CTA-wide barriers in fused `moe_dispatch_v41` from 25 to 7.
  2. **Grouped FP8/FP4 GEMM barrier and PDL streamlining + `TL_DISABLE_SAFE_MEMORY_ACCESS`**:
     - Enabled `TL_DISABLE_THREAD_STORAGE_SYNC` and `TL_DISABLE_SAFE_MEMORY_ACCESS` across `grouped_fp8_fp4_gemm`, `moe_swiglu_cast`, `moe_dispatch_v41`, and `moe_combine_v41`.
     - Removed redundant `T.ptx_commit_group()` / `T.ptx_wait_group(stages - 1)` in `grouped_fp8_fp4_gemm` (since `cp_async_barrier_noinc(loaded[s])` already tracks each stage's `cp.async` transactions).
     - Replaced CTA-wide `T.sync_threads()` with `T.sync_warp()` after `SFB_shared` scale expansion when `block_M == 16 and sfb_copies == 1` (1-warp decode CTAs).
     - Unified non-gather metadata preload (`TileExpert`, `Order`, `SFBase`, `Narrow`) before `pdl.wait()` / `pdl.chain()` with early exit before `pdl.wait()` on empty CTAs, while retaining hardware-predicated `T.ptx_cp_async(..., a_row[c] >= 0)`.
  3. **Routed MoE `moe_swiglu_cast` empty-slab and 1-row-tile early exit**:
     - Preserved `rows_per_cta = 4 if warps >= 4 else 2` (so TP3 `intermediate = 768` keeps `rows_per_cta = 2`), cascaded `Order` and `Weights` preloads before `pdl.chain()` so empty slabs only read `orders[0]` and 1-row tiles only read `orders[0..1]` and `weights[0]`, exited empty slabs immediately after `pdl.chain()` via `if expert >= 0 and orders[0] >= 0:`, and guarded rows `1..rows_per_cta-1` behind `if rows_per_cta > 1 and orders[1] >= 0:`.
     - Dropped the `moe_runner.py` / `shared_experts.py` hook changes after paired serving-layer measurements showed that delaying `_input_ready_event` past the gate GEMM (`post`) shifts `shared_experts` into `routed_experts` and evicts prefetched `shared_experts` weights from the 25 MB L2 cache under cold-L2 replay, whereas `1107`'s pre-quant launch already overlaps `shared_experts` with `partials_kernel`, `router_partials`, and `moe_dispatch` while `shared_experts` weights remain hot in L2.

## Verification & Paired Serving-Layer Benchmark (2 Fresh Container Boots on `sp8`)

- **Unit tests**: `pytest tests/kernels/moe/test_deepseek_v41_tilelang_moe.py tests/models/deepseek_v4_1/test_deepseek_v4_1_l2_schedule.py` passed (`139 passed` on both Boot 1 and Boot 2; `receipts/pytest_boot1.txt`, `receipts/pytest_boot2.txt`).
- **Bit-exactness**: 100% bit-exact (`0` uint16 diffs, identical SHA-256 digests) across all TP4 (`I = 576`) and TP3 (`I = 768`) decode batch sizes `M in (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 48)` in both eager execution and captured CUDA graphs (`chain` and `cold` modes; `receipts/pair_boot1.json`, `receipts/pair_boot2.json`, `receipts/boot1.json`, `receipts/boot2.json`).
- **Paired 6-layer serving `DeepseekV4DecoderLayer` FFN sublayer benchmark (`2` boots $\times$ `12` interleaved windows = `24` paired windows per cell, `8` graph replays per window)**:

### `TP4` (`intermediate = 576`) — Pooled 2-Boot Serving FFN Sublayer (us/layer)

| Mode / `M` | `base_1107` med (us) | `cand_0762` med (us) | Paired $\Delta$ med (us) | 95% Bootstrap CI (us) | Speedup (%) |
|---|---:|---:|---:|---:|---:|
| `chain/1` | `194.08` | `191.74` | **`-2.55`** | `[-2.963, -2.141]` | **`+1.20%`** |
| `cold/1` | `249.22` | `247.10` | **`-2.37`** | `[-2.621, -2.155]` | **`+0.85%`** |
| `chain/2` | `299.48` | `297.94` | **`-1.51`** | `[-1.743, -1.069]` | **`+0.51%`** |
| `cold/2` | `354.59` | `353.61` | **`-1.16`** | `[-1.789, -0.769]` | **`+0.28%`** |
| `chain/3` | `410.31` | `409.24` | **`-1.35`** | `[-1.898, -0.837]` | **`+0.26%`** |
| `cold/3` | `464.54` | `464.16` | **`-0.70`** | `[-1.484, -0.413]` | **`+0.08%`** |
| `chain/4` | `514.63` | `513.49` | **`-1.06`** | `[-1.489, -0.498]` | **`+0.22%`** |
| `cold/4` | `569.30` | `568.01` | **`-1.31`** | `[-1.663, -0.549]` | **`+0.23%`** |
| `chain/6` | `716.39` | `714.33` | **`-1.60`** | `[-2.773, -1.061]` | **`+0.29%`** |
| `cold/6` | `769.51` | `768.62` | **`-1.24`** | `[-1.527, -0.807]` | **`+0.11%`** |
| `chain/8` | `926.41` | `924.68` | **`-0.68`** | `[-1.663, -0.134]` | **`+0.19%`** |
| `cold/8` | `980.91` | `980.07` | **`-1.02`** | `[-1.702, -0.157]` | **`+0.09%`** |
| `chain/12` | `1281.21` | `1278.02` | **`-2.61`** | `[-4.495, -0.831]` | **`+0.25%`** |
| `cold/12` | `1334.28` | `1334.56` | **`-0.85`** | `[-2.049, +1.349]` | **`-0.02%`** |
| `chain/16` | `1702.71` | `1699.14` | **`-1.82`** | `[-6.565, -0.499]` | **`+0.21%`** |
| `cold/16` | `1753.69` | `1752.44` | **`-0.92`** | `[-1.549, +0.004]` | **`+0.07%`** |
| `chain/20` | `2023.24` | `2013.69` | **`-4.28`** | `[-14.907, -0.897]` | **`+0.47%`** |
| `cold/20` | `2067.26` | `2063.36` | **`-2.83`** | `[-4.769, -1.081]` | **`+0.19%`** |
| `chain/24` | `2322.82` | `2315.38` | **`-5.23`** | `[-8.383, -2.761]` | **`+0.32%`** |
| `cold/24` | `2366.01` | `2365.30` | **`-1.59`** | `[-5.636, +0.116]` | **`+0.03%`** |
| `chain/28` | `2640.13` | `2638.09` | **`-5.29`** | `[-9.724, -0.591]` | **`+0.08%`** |
| `cold/28` | `2687.51` | `2685.83` | **`-1.84`** | `[-7.461, -0.059]` | **`+0.06%`** |
| `chain/32` | `2919.06` | `2919.39` | **`-2.27`** | `[-5.249, +1.473]` | **`-0.01%`** |
| `cold/32` | `2970.00` | `2965.74` | **`-6.25`** | `[-12.255, -1.425]` | **`+0.14%`** |
| `chain/48` | `3877.24` | `3879.45` | **`-0.91`** | `[-4.865, +12.071]` | **`-0.06%`** |
| `cold/48` | `3922.66` | `3921.78` | **`+2.67`** | `[-4.062, +5.513]` | **`+0.02%`** |

- **`TP4` summary**: `25/26` cells have faster paired medians, `20/26` cells have strictly negative (faster) 95% CIs, and `0/26` cells are slower.

### `TP3` (`intermediate = 768`) — Pooled 2-Boot Serving FFN Sublayer (us/layer)

| Mode / `M` | `base_1107` med (us) | `cand_0762` med (us) | Paired $\Delta$ med (us) | 95% Bootstrap CI (us) | Speedup (%) |
|---|---:|---:|---:|---:|---:|
| `chain/1` | `235.09` | `234.32` | **`-0.59`** | `[-0.705, -0.344]` | **`+0.33%`** |
| `cold/1` | `286.66` | `286.84` | **`-0.02`** | `[-0.294, +0.460]` | **`-0.06%`** |
| `chain/2` | `382.30` | `381.49` | **`-0.73`** | `[-1.166, -0.162]` | **`+0.21%`** |
| `cold/2` | `436.38` | `435.46` | **`-0.40`** | `[-0.674, +0.685]` | **`+0.21%`** |
| `chain/3` | `527.81` | `526.68` | **`-0.80`** | `[-1.382, -0.320]` | **`+0.21%`** |
| `cold/3` | `578.46` | `578.81` | **`+0.02`** | `[-0.216, +0.785]` | **`-0.06%`** |
| `chain/4` | `670.50` | `668.82` | **`-1.35`** | `[-2.075, -0.318]` | **`+0.25%`** |
| `cold/4` | `723.05` | `722.78` | **`-0.53`** | `[-1.126, +0.171]` | **`+0.04%`** |
| `chain/6` | `931.22` | `931.61` | **`-0.58`** | `[-0.818, -0.003]` | **`-0.04%`** |
| `cold/6` | `983.63` | `984.12` | **`-1.36`** | `[-2.481, -0.269]` | **`-0.05%`** |
| `chain/8` | `1199.84` | `1197.40` | **`-1.65`** | `[-3.201, -0.331]` | **`+0.20%`** |
| `cold/8` | `1248.37` | `1248.93` | **`-0.25`** | `[-1.010, +0.568]` | **`-0.04%`** |
| `chain/12` | `1694.04` | `1691.27` | **`-2.01`** | `[-3.211, -0.724]` | **`+0.16%`** |
| `cold/12` | `1744.71` | `1742.58` | **`-0.65`** | `[-2.505, +1.067]` | **`+0.12%`** |
| `chain/16` | `2189.00` | `2191.16` | **`+1.09`** | `[-1.158, +4.285]` | **`-0.10%`** |
| `cold/16` | `2241.40` | `2240.37` | **`-0.74`** | `[-4.434, +4.581]` | **`+0.05%`** |
| `chain/20` | `2613.50` | `2612.28` | **`-2.67`** | `[-6.591, +1.610]` | **`+0.05%`** |
| `cold/20` | `2656.88` | `2657.55` | **`-2.04`** | `[-4.962, +1.757]` | **`-0.03%`** |
| `chain/24` | `3025.49` | `3021.85` | **`-3.74`** | `[-6.810, +1.319]` | **`+0.12%`** |
| `cold/24` | `3072.07` | `3071.63` | **`-1.18`** | `[-4.460, +5.602]` | **`+0.01%`** |
| `chain/28` | `3448.07` | `3443.69` | **`-2.77`** | `[-9.152, -0.495]` | **`+0.13%`** |
| `cold/28` | `3488.97` | `3490.86` | **`+0.97`** | `[-5.843, +5.502]` | **`-0.05%`** |
| `chain/32` | `3788.57` | `3785.04` | **`-4.46`** | `[-6.693, +0.155]` | **`+0.09%`** |
| `cold/32` | `3830.90` | `3829.72` | **`-1.02`** | `[-7.065, +3.272]` | **`+0.03%`** |
| `chain/48` | `5056.89` | `5056.88` | **`+0.56`** | `[-10.684, +19.070]` | **`+0.00%`** |
| `cold/48` | `5102.90` | `5091.23` | **`-15.30`** | `[-18.605, -8.174]` | **`+0.23%`** |

- **`TP3` summary**: `22/26` cells have faster paired medians, `10/26` cells have strictly negative (faster) 95% CIs, and `0/26` cells are slower.
