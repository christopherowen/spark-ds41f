# mHC `project_streams` + `finalize` Single-Launch Epilogue & Barrier-Free `finalize`

## Summary

Each DeepSeek-V4.1 decoder layer runs two lagged mHC sublayer entries (attention and MoE), or 122 entries per target step across the 61 layers. Previously, every entry launched two kernels on the PDL chain:
1. `project_streams` (`40 × cdiv(M, 16)` CTAs): folds the previous sublayer's output (`post_pre` or `broadcast`), projects the 4 streams through `fn` on TF32 tensor cores (`load_split_tf32`), collapses the streams with the incoming pre-mix into `C`, and writes per-split partials `P`, `S`, and `Q`.
2. `finalize` (`M` CTAs, one token per CTA): Warp 0 reduces `S` and `P` across the 40 splits and runs the mix split + 20-iteration 4×4 Sinkhorn; Warps 1–3 reduce `Q`, RMS-normalize `C` with the sublayer norm weight into `y`, and write the 5120-column MXFP8 activation cast (`yq`, `ysf`) for the sublayer's block-32 projections.

This patch delivers two improvements while preserving 100% `torch.equal` bit-identity across all batch sizes and modes:

1. **Barrier-free `finalize` (`mhc_finalize`)**:
   - Previously, `finalize` staged Warp 0's 24 reduced mixes through `mixes_shared = T.alloc_shared((MIXES,), T.float32)`. Because `T.copy(mixes, mixes_shared)` writes shared memory, TileLang emitted a CTA-wide `__syncthreads()` after Warp 0's `S` and `P` reduction loops, forcing Warps 1–3 (`else` branch) to stall before starting their `Q` reduction, RMSNorm, and MXFP8 cast.
   - Replacing `mixes_shared` with intra-warp `T.shfl_down(mix_val[0], MULT)` and `T.shfl_down(mix_val[0], 2 * MULT)` inside Warp 0 eliminates `mixes_shared` and removes `__syncthreads()` from `mhc_finalize_kernel` completely.

2. **Single-launch `project_and_finalize_streams` (`mhc_project_finalize`) for single-row decode (`FINALIZE_FUSE_MAX_ROWS = 1`)**:
   - The last split CTA of each token tile (`arrived == splits - 1` via `T.atomic_add(counters[by], 1, memory_order="acq_rel")` after `__threadfence()`) resets `counters[by] = 0` and runs Warp 0's register Sinkhorn (using the exact `shfl_xor(2), shfl_xor(1)` row and `shfl_xor(8), shfl_xor(4)` column reduction tree of `T.reduce_max` / `T.reduce_sum`) beside Warps 1–3's collapse RMSNorm and MXFP8 cast.
   - At `M = 1`, single-launch `project_and_finalize_streams` reduces entry latency on GB10 (`dgx4`, `spark-ds41f:1030`) from `11.16 µs` to **`10.48 µs` (`-0.68 µs/call`, `-83 µs/step` across 122 sublayer entries)**.
   - At `M >= 2`, standalone `finalize` launches `M` parallel CTAs across `M` SMs (versus 1 last-arriving CTA per 16-row tile), so `_mhc_pre` dispatches to `_project_finalize` when `tokens <= FINALIZE_FUSE_MAX_ROWS` (`1`) and to `_project` + barrier-free `_finalize` when `tokens > 1`.

## Verification on `dgx4` (`spark-ds41f:1030`)

- **Bit-exactness (`torch.equal`)**: `112 passed` in `tests/kernels/test_deepseek_v41_tilelang_mhc.py`, including `test_rows_independent_of_batch` (verifying `M = 1` fused launch is bit-for-bit identical to slice `0:1` of `M = 130` two-kernel launch) and `test_project_and_finalize_bits_match_split` across `tokens in [1, 6, 16, 33, 64]` and modes `pre`, `post_pre`, `broadcast`.
- **PDL**: `PDL sources: 30 kernels launched, 23 with PDL, 0 violations`.
