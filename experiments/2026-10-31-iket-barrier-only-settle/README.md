# Restrict IKET `TL_IKET_SETTLE()` to Block-Barrier Ranges and Loop-Exit Barrier Tails

## Summary

- **Emit `TL_IKET_SETTLE()` Only on Ranges Containing a Block Barrier (`vllm/models/deepseek_v4_1/tilelang/iket.py`)**:
  - `ptxas` compiles block barriers (`__syncthreads`, `tl::__sync_thread_partial`, `bar.sync`, `barrier.sync`, `.sync()`, and `tl::AllReduce`) as `BAR.SYNC.DEFER_BLOCKING`, which releases warp issue only at the next memory instruction.
  - Previously, `annotate()` emitted `TL_IKET_SETTLE()` (`ld.volatile.shared.u32 s, [0x20]`) before the `RANGE_END` event of *every* range — including `load.*`, `wait.cp_async*`, `wait.mbarrier*`, `compute.mma*`, `store.*`, and `signal.*` ranges that contain no `BAR.SYNC`.
  - Across the 29 DS4.1 TileLang hot kernels, restricting `TL_IKET_SETTLE()` to ranges that contain a block barrier reduces settle loads from `294` to `139` (eliminating `155` redundant volatile shared-memory loads, including 9 of 15 per loop in `moe_grouped_fp8_fp4_gemm`, 8 of 10 in `mhc_project_streams`, 9 of 11 in `mhc_finalize`, 8 of 11 in `oneshot_allreduce`, and all settles in `moe_combine` and `moe_swiglu_cast_kernel`).
- **Settle Sampled Loop-Exit Barrier Tails (`vllm/models/deepseek_v4_1/tilelang/iket.py`)**:
  - When a sampled loop (`iters > 0`) ends with a `settled` range (such as `for (int n = 0; n < 4; ++n)` ending with `__syncthreads()` in `mhc_project_streams`), the loop's final unsampled iteration (`n = 3`) executes `BAR.SYNC.DEFER_BLOCKING` without running the loop range's `TL_IKET_SETTLE()`.
  - Emitting `TL_IKET_SETTLE()` before the first range after that loop prevents the loop's final unsampled `BAR.SYNC.DEFER_BLOCKING` from leaking into the post-loop range's start timestamp (`0x6e00 -> 0x6e40` in `mhc_project_streams`).

## Verification

- Verified on `dgx4` inside `spark-ds41f:1030` (holding `/tmp/spark-lab-gpu.lock`):
  - `tests/kernels/test_deepseek_v41_tilelang_iket.py`: `13 passed`.
  - `sass_check.py` across all 29 DS4.1 TileLang hot kernels (`ctas=1, iters=1`): `0` predicated events, `0` hoisted timer reads, and `0` unsettled block barriers across all 29 kernels (fixing `mhc_project_streams.7193574c2197` from `1` unsettled barrier to `0` while eliminating `155 / 294` non-barrier `TL_IKET_SETTLE()` loads).
