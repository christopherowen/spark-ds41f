# Multi-Token mHC Project+Finalize, DSpark Aux Collapse, and Final Post+Collapse+Norm Fusion

## Summary
Implements Part 2 items **1**, **5**, and **6** of the DeepSeek-V4.1 TileLang kernel fusion roadmap on `rc21` (`a89da720d`), incorporating the best optimizations from PRs **#782** and **#790**:

1. **Multi-Token Fused `project_and_finalize_streams` (`FINALIZE_FUSE_MAX_ROWS = 16`, Item 1)**:
   - Extends single-launch `project_and_finalize_streams` from `rows == 1` to `1 <= rows <= 16` (covering 6-token speculative verify steps at `c=1..2`, multi-stream decode up to 16 rows, and all DSpark draft steps).
   - Uses `mhc_claim_and_wait` so the last `live_m = min(M - by * block_M, block_M)` split CTAs of each tile spin on `ld.global.acquire.gpu.b32` until all 10 `splits` CTAs have arrived, each finalize 1 token (`by * block_M + rank`) in parallel using warp 0's 24-lane warp-shuffle Sinkhorn and warps 1–3's 96-worker norm+MXFP8 cast, and the last finalizer CTA resets `counters[by] = 0` via `st.global.release.gpu.b32`.
   - Saves **128 kernel launches per step** at `1 < M <= 16`.
2. **In-Register DSpark Target `aux_hidden_states` Collapse in `mhc.pre(mode="post_pre")` (Item 5)**:
   - Adds `with_aux: bool = False` (`aux_out` in `TileKernelsMHC.pre` / `post_pre`) to `project_streams` and `project_and_finalize_streams` (`mode="post_pre"`).
   - Accumulates `0.25 * av[v]` across the 4 updated residual streams in registers and writes `Aux[token, h0 + c0 + v]` at layers 3, 26, and 50, eliminating 3 standalone `mhc.post_collapse` launches and `3 x [M, 5120]` stream re-reads per target step.
3. **Final Layer `post_collapse_norm_streams` (`mhc.post + collapse + self.norm` + `sink_columns`, Item 6)**:
   - Adds 1-CTA-per-token `post_collapse_norm_streams` (`TileKernelsMHC.post_collapse_norm`) fusing `RankOrderParts` reduction, 4-stream `post` update, optional `sink_columns` extraction, stream `collapse`, and output `RMSNorm` (`self.norm.weight`) into one launch at the end of `DeepseekV4Model.forward` and `DSparkDeepseekV4Model.forward`.
   - Eliminates 1–2 kernel launches and the `[M, 4, 5120]` + `[M, 5120]` intermediate HBM round-trips at the end of every target and draft forward pass.
4. **Best of PRs #782 and #790 Included**:
   - Pre-wait `R`, `X`, `Fn`, and `weight` loads before `T.pdl_sync()` across `finalize`, `project_streams`, and `project_and_finalize_streams`.
   - 1-CTA-per-token `finalize` with 24-lane warp-shuffle Sinkhorn and 96-worker norm+MXFP8 cast.
   - `world > 1` (`RankOrderParts`) support in `post_streams`, `post_collapse_streams`, and `post_collapse_norm_streams`.

## Verification on `dgx4` (`spark-ds41f:19f2c20ed4d6-next-21`)
- `pytest --noconftest -v -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/test_deepseek_v41_tilelang_mhc.py`: **162 passed, 0 PDL violations**
- `pytest --noconftest -v -p vllm_cfg_plugin -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/test_deepseek_v41_tilelang_pdl.py`: **10 passed, 0 PDL violations**
