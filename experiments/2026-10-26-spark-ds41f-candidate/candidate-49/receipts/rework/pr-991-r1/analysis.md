# PR #991 rework, round 1: hardware e2m1x2 conversion, corrected GEMM epilogue

**Sub-agent verdict:** `corrected-misses-mark`. The corrected patch is `0764-perf-dsv41-hw-e2m1x2-indexer-and-gemm-mxfp4-quantize.patch`, and every cell is in `result.json`.

**Correction.** The PR forced the GEMM epilogue's row loop rolled (`#pragma unroll 1`), which left the decode GEMM at M=1..4 with no gain. Round 1 keeps 1107's `T.serial` loop, which nvcc unrolls. The decode GEMM is then faster at every M:
- 3.7-5.1% by CUPTI kernel time;
- 3.7-4.8% by graph replay.

**What the sub-agent counted as the miss.** In the PR's bench, a loop of identical eager launches, GEMM M=6 and M=8 read slower beyond the interval in all three round 1 boots (6.36-6.52 µs, against 1107's 5.83-6.20 µs). At the same M:
- kernel time and graph replay show round 1 4.6-5.1% faster;
- in one companion boot, the eager loop itself read 5.53/5.54 µs, faster than 1107.

The faster kernel drops the eager loop out of its queued mode, so that loop measures the host launch path, not the kernel. The paged-key "slowdowns" of the original screen were the same effect: their kernel time is 30-57% lower at every M.

## Coordinator's verdict: meets the mark

Decode kernels run from CUDA graphs in serving, so graph replay and kernel time decide. Judged on those, round 1 is faster at every M of the decode GEMM and slower in no cell, so it meets its mark. It joins the next candidate for fresh qualification.

The query-rope M=2048 gain is not claimed.

The lesson went into the workflow (docs/release-benchmark.md, step 7): judge decode kernels by CUDA-graph replay or kernel time, and record the eager loop beside them.

The analysis is the rework sub-agent's; the coordinator wrote the verdict.
