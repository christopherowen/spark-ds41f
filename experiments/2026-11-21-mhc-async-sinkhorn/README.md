# mHC Async Sinkhorn Split (`mhc_finalize_norm` + `mhc_finalize_sinkhorn`)

- **Date**: 2026-11-21
- **Base commit**: `0eace90e2` (`spark-ds41f:candidate-63-r5`, vLLM tree `5f16a050bb283d3d964ab831de4cdf6fca74dea7`)
- **One intended variable**: Wire `async_sinkhorn()` and event-based `wait_sinkhorn()` across `DeepseekV4Model`, `DeepseekV4DecoderLayer`, and `DSparkDeepseekV4Model` so decode steps (`M <= 128`) run the already-compiled `mhc_finalize_norm` (5 CTAs/token on the critical path) + `mhc_finalize_sinkhorn` (1 warp/token on `aux_stream`) split from `0818` instead of the single `mhc_finalize` kernel.

## Hypothesis and Prediction

In `spark-ds41f:1120` (and `candidate-63-r5`), `0818` compiles and binds `mhc_finalize_norm` and `mhc_finalize_sinkhorn`, but nothing on the serving path enters `async_sinkhorn()`, so all 86 mHC entries per decode step (40 target layers × 2 + 3 DSpark layers × 2) run `mhc_finalize` (`1.085 ms/step` in `c1-floor-now`'s trace).

Wiring `async_sinkhorn()` with per-device reusable CUDA events (`fork_event` / `done_event`) and automatic waits before every consumer of `(post_mix, res_mix, pre_mix)` (`_mhc_pre`, `_mhc_post*`, `_mhc_collapse*`, fused `engram`, `ced_indices` gather, `sp.gather`, `glp.steer`, and `async_sinkhorn` exit) should:

- Save **`-1.8..-3.4 us/call`** (`-0.15..-0.29 ms/step`, ~0.6–1.1% on single-stream `c1` and grid) across `M = 1..128` with **0 new JIT files**.
- Remain **100% bit-exact** (`128/128` greedy identical, `KL = 0`).

## Pre-Serving Kernel Verification (`dgx4`, `spark-ds41f:candidate-63-r5`)

### Correctness and PDL Contracts

- `tests/kernels/test_deepseek_v41_tilelang_mhc.py` + `test_deepseek_v4_1_ced_model.py` + `test_deepseek_v4_1_ced_sp.py` + `test_deepseek_v4_1_sp_prefill.py`: **`423 passed, 0 failed`** (`118.32 s`), including bit-exact (`torch.equal` on all 7 outputs) sync-vs-async checks across `M = 1, 2, 3, 4, 6, 16, 48, 128`, 12-layer CUDA Graph capture + replay with reused events and standalone `post_collapse` / `post_collapse_norm` auto-waits, `M = 160 > 128` fallback, and exception drain.
- `tests/kernels/test_deepseek_v41_tilelang_pdl.py`: **`15 passed, 0 failed`** (`0` PDL violations, `4/4` `LayerChain` bit-exact and programmatic).

### Cold-L2 40-Layer PDL-Chained CUDA Graph Screen (`mhc.post_pre -> mxfp8_gemm_decode(2112x5120)`)

CUPTI durations at `M = 6` (16 cold weight copies):
- **Sync**: `mhc_project_streams = 25.38 us`, `mhc_finalize = 16.70 us`, `gemm_mxfp8 = 55.85 us`
- **Async**: `mhc_project_streams = 16.81 us`, `mhc_finalize_norm = 1.31 us`, `mhc_finalize_sinkhorn = 3.62 us`, `gemm_mxfp8 = 50.02 us`

40-layer Cold-L2 CUDA Graph median wall time (`sync` vs `async` on `aux_stream` with reused `done_ev`):

| `M` | `sync` (ms) | `async_aux` (ms) | Delta (`us/call`) | Est. 86 calls (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.738 | 2.601 | `-3.42` | `-0.294` |
| 2 | 2.684 | 2.589 | `-2.36` | `-0.203` |
| 3 | 2.690 | 2.607 | `-2.07` | `-0.178` |
| 4 | 2.700 | 2.605 | `-2.37` | `-0.204` |
| 6 | 2.706 | 2.636 | `-1.77` | `-0.152` |
| 8 | 2.729 | 2.652 | `-1.91` | `-0.164` |
| 12 | 2.750 | 2.650 | `-2.50` | `-0.215` |
| 16 | 2.743 | 2.643 | `-2.50` | `-0.215` |
| 24 | 2.966 | 2.885 | `-2.04` | `-0.175` |
| 32 | 2.986 | 2.899 | `-2.17` | `-0.186` |
| 48 | 3.339 | 3.246 | `-2.32` | `-0.199` |
| 64 | 3.664 | 3.540 | `-3.09` | `-0.266` |
| 96 | 4.399 | 4.271 | `-3.19` | `-0.274` |
| 128 | 5.180 | 5.125 | `-1.38` | `-0.119` |

*(Note: Because `mhc_finalize_norm` is already `1.31 us` (`0.11 ms/step` total) and its outputs `(y, y_mxfp8)` feed multiple consumers per sublayer, and `vocab_packed_embedding_kernel` is `5 us × 2 = 0.010 ms/step`, forward-fusing `mhc_finalize_norm` into consumer GEMMs or fusing `embed` into Layer-0 `project_streams` was bounded at `< 0.08 ms/step` and not pursued.)*

## Live Serving Runs (`tp3-triangle`, `dgx1..dgx3`)

All runs cooled `< 49 C` on `spark-ds41f:candidate-63-r5` inside lab window `dgx1-1795263633` (`results/private/lab/2026-11-21-campaign/`). Every arm passed `model` `128/128` greedy bit-exact (`outputs_sha256 = b795e6d4c813815a475ab1516faf277a90a86b73614c38f4700189484a8c532d`, `rank1_fraction = 1.0`, `perplexity = 1.3160824`, `failed = 0`). Live CUPTI confirmed `0` `mhc_finalize_kernel` calls.

| Arm | `grid` `mad5` GM vs `ctrl` | `grid` `fl25` GM vs `ctrl` | `1x3 @4K` (ms) | `8x24 @4K` (ms) | `8x48 @4K` (ms) | `c1` (tok/s) | `c1` `tpot_p50` (ms) | `c8` (tok/s) | `c8` `tpot_p50` (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ctrl-b1` | `+0.085%` | `+0.088%` | `16.890` | `22.619` | `27.557` | `69.65 ±3.82%` | `14.46` | `169.52 ±1.03%` | `47.21` |
| `ctrl-b2` | `-0.085%` | `-0.088%` | `16.923` | `22.625` | `27.576` | `69.73 ±3.79%` | `14.42` | `169.54 ±0.81%` | `46.70` |
| **`mhc-b1`** | **`-1.007%`** | **`-0.803%`** | **`16.773`** | **`22.473`** | `27.622` | **`70.07 ±3.67%`** | **`14.06`** | **`169.93 ±0.90%`** | **`46.77`** |
| **`mhc-b2`** | **`-0.994%`** | **`-0.867%`** | **`16.786`** | **`22.454`** | **`27.472`** | **`69.92 ±3.71%`** | **`14.14`** | `169.48 ±0.93%` | **`46.65`** |
| `pynccl-b1` (`Decode #17`) | `-0.043%` | `-0.044%` | `16.928` | `22.559` | `27.795` | `69.79 ±3.85%` | `14.44` | `169.62 ±0.95%` | `46.97` |

### Conclusion

- **`mhc` `async_sinkhorn` (`PR #1264`)**: **2-boot verified win (`-1.00%` `mad5` / `-0.835%` `fl25` grid GM, `14/15` cells faster, `c1 tpot_p50` `14.44 -> 14.10 ms` [`-2.35%`], `c1` `69.69 -> 70.00 tok/s`, 100% bit-exact).** Ready for `candidate-64`.
- **`VLLM_SHARE_PYNCCL_COMMS=1` (`Decode #17`)**: Neutral on Boot 1 (`-0.043%` grid, `c1`/`c8` within noise); dropped after Boot 1.
