# GB10 Persisting-L2 Set-Aside Sizing and Window Budget Clamping for V4.1 L2 Prefetch

## Hypothesis

On GB10 (`SM121`, `24.00 MiB` L2 cache, `18.00 MiB` hardware maximum persisting L2), the CUDA driver initializes `CU_LIMIT_PERSISTING_L2_CACHE_SIZE` to only **`4.50 MiB` (`4,718,592` B)**. Meanwhile, `vllm/models/deepseek_v4_1/l2_prefetch.py`:
1. Never configures `CU_LIMIT_PERSISTING_L2_CACHE_SIZE` (`_l2pf.PERSIST_L2` defaults to `"0"`, leaving the `4.50 MiB` driver default in place).
2. Issues `cp.async.bulk.prefetch.L2` with `L2::evict_last` across per-window budgets of `BUDGET_WO = 20 MB`, `BUDGET_FFN = 16 MB`, and `BUDGET_NEXT = 20 MB`.

Because `L2PrefetchKernel` walks segments sequentially from head to tail using `L2::evict_last`, prefetching `16–20 MB` into a `4.50 MiB` persisting-L2 set-aside causes the **trailing segments of a window to evict the window's own leading segments** (the very first projection the GPU executes after the window closes) out of the persisting L2 partition, while intervening non-persisting collective/combine traffic evicts the spilled lines from normal L2 sets. Even when `CU_LIMIT_PERSISTING_L2_CACHE_SIZE` is raised to `15.00 MiB`, an unclamped `20 MB` window budget still wraps around inside the persisting L2 partition and creates severe run-to-run latency jitter (`p95 - p05 = 14–63 µs`).

We hypothesize that:
- Sizing GB10's primary-context persisting-L2 set-aside via `VLLM_DS41_L2_PREFETCH_PERSIST_MB` (defaulting to `"16.8"`, which the driver rounds to `16.50 MiB` = `17,301,504` B out of `24.00 MiB`), and
- Clamping each window's `make_plan` budget to the active `persisting_l2_bytes` (`VLLM_DS41_L2_PREFETCH_CLAMP_TO_PERSIST=1`) so a window never self-evicts its own leading projections
will keep the leading dense projections (`fused_wqa_wkv`, `wo_a`, `gate_up_proj`) L2-resident across intervening collective/combine traffic and eliminate bimodal L2 eviction jitter.

## Changes

Patch: `0001-l2-prefetch-persisting-budget-alignment.patch` (against `vllm` commit `1f18be14470339c1227ed0432b896a8ca09811b5`).

- `vllm/models/deepseek_v4_1/l2_prefetch.py`:
  - Add `VLLM_DS41_L2_PREFETCH_PERSIST_MB` (`_persist_env_request()`, default `"16.8"` on `SM121`, `"0"` otherwise) and `configure_persisting_l2(device, request)`.
  - Add `VLLM_DS41_L2_PREFETCH_CLAMP_TO_PERSIST` (default `"1"`) and `effective_budget(budget, device)` to clamp `make_plan` budgets to the device's active `persisting_l2_bytes`.
- `tests/models/test_deepseek_v4_1_l2_schedule.py`:
  - Add unit and GPU tests verifying `_persist_env_request`, `effective_budget`, and `make_plan` persisting-L2 configuration and budget clamping on `dgx4`.

## Verification & Benchmark Results (`dgx4`, `spark-ds41f:1030`)

### Correctness

- `pytest --noconftest -q tests/models/test_deepseek_v4_1_l2_schedule.py tests/models/test_glm5next_l2_prefetch_persist.py tests/models/test_glm5next_tilelang_l2_prefetch.py`: `33 passed, 1 skipped`

### Single-Projection Persisting-L2 Sweep (`mxfp8_gemm_decode`, `M=16, N=2112, K=7168`, `15.14 MB` FP8 weight + `6 MB` intervening traffic)

| Configuration | Driver Persisting L2 Limit | `mxfp8_gemm_decode` Median (µs) | `p95` (µs) | `p95 - p05` Spread (µs) |
|---|---:|---:|---:|---:|
| No prefetch (`limit = 0`) | `0.00 MiB` | `79.87` | `145.82` | `66.98` |
| Prefetch with `limit = 0` | `0.00 MiB` | `67.58` | `151.74` | `84.26` |
| Current default (`limit = 4.50 MiB`) | `4.50 MiB` | `57.34` | `61.41` | `4.16` |
| Undersized persisting (`9.00 MiB`) | `9.00 MiB` | `38.91` | `61.92` | `23.14` |
| Undersized persisting (`12.00 MiB`) | `12.00 MiB` | `28.67` | `91.94` | `63.39` |
| **Sized persisting (`15.00 MiB`)** | **`15.00 MiB`** | **`22.53` (`-60.7%`)** | **`22.53`** | **`0.03` (`-99.3%`)** |
| **Sized persisting (`18.00 MiB`)** | **`18.00 MiB`** | **`22.53` (`-60.7%`)** | **`22.53`** | **`2.05`** |

### Multi-Segment `NEXT` Window (`B1 = 15.14 MB` `fused_wqa_wkv` + `B2 = 6.00 MB` `wq_b` + `6 MB` intervening traffic)

| Window Configuration | `B1` (`fused_wqa_wkv`) Median (µs) | `p95` (µs) | `p95 - p05` Spread (µs) |
|---|---:|---:|---:|
| Current default (`persist = 4.50 MiB`, `budget = 20 MB`) | `79.87` | `122.75` | `44.93` |
| Unclamped (`persist = 15.00 MiB`, `budget = 21.5 MB`) | `32.77` | `44.90` | `14.18` |
| **Clamped (`persist = 16.50 MiB`, `budget <= persist`)** | **`26.56` (`-66.7%`)** | **`26.78` (`-78.2%`)** | **`2.34` (`-94.8%`)** |
