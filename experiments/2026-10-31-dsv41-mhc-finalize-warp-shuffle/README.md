# DeepSeek-V4.1 `mhc_finalize` Warp-Shuffle Sinkhorn Reductions & Pre-Wait Loads

## Problem (IKET 43-Layer Forward-Pass Finding)

In the 43-layer IKET forward-pass trace (`table_43l.md`), `mhc_finalize_kernel` (`vllm/models/deepseek_v4_1/tilelang/mhc.py`) exhibits three structural bottlenecks across its two sublayer entries per decoder layer (`86` launches per forward pass):

1. **41 `tl::NamedBarrier<32>` hardware barrier synchronizations in Warp 0:** `mhc_finalize` lowers its 20-step Sinkhorn-Knopp normalization via TileLang's `T.reduce_max` and `T.reduce_sum` on `4 × 4` fragments (`cm`), which emit `tl::AllReduce<..., tl::NamedBarrier<32>>::run(...)` (`41` dynamic `bar.sync` executions per CTA, accounting for `0.95 µs/warp` of `WARP_SYNC` stalls in IKET). By contrast, `project_and_finalize_streams` in the same module already performs the exact same `4 × 4` Sinkhorn row/column reductions in registers using `T.shfl_xor(..., 2)` / `T.shfl_xor(..., 1)` and `T.shfl_xor(..., 8)` / `T.shfl_xor(..., 4)` with zero named-barrier instructions.
2. **Two separate serial loops over `splits = 40` in Warp 0 and delayed `C` loads in Warps 1–3:** Warp 0 reduces `S[0..39, pid, 0]` and `P[0..39, pid, 0, j]` in two separate non-unrolled `T.serial(splits)` loops (`2.34 µs/warp` `MEM_WAIT` in IKET), while Warps 1–3 wait for all 40 `Q[0..39, pid]` loads and `rsqrt` to finish before issuing any of the 7 `uint4` loads of `C[pid, ...]`.
3. **Zero pre-wait loads before `pdl.chain()`:** Static parameters `scale`, `base`, and `weight` do not depend on the predecessor `mhc_project_streams` kernel, yet `mhc_finalize` calls `pdl.chain()` (`cudaGridDependencySynchronize()`) at line 1 before issuing any global loads.

## Solution

1. **Pre-wait loads of `scale`, `base`, and `weight` before `pdl.chain()`:**
   - Before `pdl.chain()`, Warp 0 (`tx < 24`) preloads its assigned element of `scale` and `base` (`mix_idx = tx + 8` for `tx < 16` holding the `4 × 4` `comb` entries directly on lanes `0..15`, and `tx - 16` for `16 <= tx < 24` holding `pre[0..3]` and `post[0..3]`), and Warps 1–3 (`tx >= 32`) preload all 7 `VEC = 8` chunks of `weight` into unrolled registers `wv`.
2. **Fused unrolled `S`/`P` reduction & register `T.shfl_xor` Sinkhorn in Warp 0:**
   - Fuse the `S` and `P` split reductions into a single `T.unroll(splits)` loop with direct 24-lane coalesced mapping (eliminating both the second loop and the `T.shfl_down` permutation), and replace all 41 `T.reduce_max` / `T.reduce_sum` `tl::NamedBarrier<32>` calls with warp-uniform `T.shfl_xor` reductions in the exact reduction tree order (`delta = 2, 1` for rows and `delta = 8, 4` for columns).
3. **Batch-issued `C` loads & unrolled `Q` reduction in Warps 1–3:**
   - Immediately after `pdl.chain()`, Warps 1–3 issue all 7 `uint4` loads of `C[pid, ...]` into `yv` before reducing `Q[0..39, pid]`, overlapping `C`'s L2 load latency with the `Q` reduction and `rsqrtf`.

## Verification & Benchmark (`dgx4`, GB10 SM121)

Verified in `spark-ds41f:19f2c20ed4d6-next-20` on `dgx4`:
- **Bit-identical outputs & PDL source audit:**
  - `pytest -q -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/test_deepseek_v41_tilelang_mhc.py tests/kernels/test_deepseek_v41_tilelang_pdl.py`
  - **`128 passed`**, `PDL sources: 64 kernels launched, 57 with PDL, 0 violations`.
  - Generated CUDA source confirmed `0` `tl::AllReduce` / `tl::NamedBarrier<32>` occurrences (down from `5` static / `41` dynamic per CTA).
- **CUDA-Graph Latency (`20` calls per graph, `200` graph replays on `dgx4`):**

| Configuration | Baseline Median (p10) | Patched Median (p10) | Speedup |
| :--- | ---: | ---: | ---: |
| `mhc_finalize` standalone (`M=1, H=5120`) | `4.259 µs` (`4.229 µs`) | **`3.963 µs` (`3.933 µs`)** | **`-6.9%`** |
| `mhc_finalize` standalone (`M=6, H=5120`) | `4.312 µs` (`4.304 µs`) | **`4.102 µs` (`4.062 µs`)** | **`-4.9%`** |
| `mhc_finalize` standalone (`M=16, H=5120`) | `4.413 µs` (`4.403 µs`) | **`4.206 µs` (`4.106 µs`)** | **`-4.7%`** |
| Chained `mhc_project_streams + mhc_finalize` (`M=1`) | `9.627 µs` (`9.552 µs`) | **`9.422 µs` (`9.347 µs`)** | **`-2.1%`** |
| Chained `mhc_project_streams + mhc_finalize` (`M=6`) | `9.939 µs` (`9.858 µs`) | **`9.733 µs` (`9.723 µs`)** | **`-2.1%`** |
