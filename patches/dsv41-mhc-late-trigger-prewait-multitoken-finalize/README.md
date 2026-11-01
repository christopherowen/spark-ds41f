# Safe Multi-Token `mhc` Single-Launch Fusion (`M <= 16`) with Late PDL Trigger & Pre-Wait Preload

Builds on top of **PR #821** (`exp/dsv41-mhc-multitoken-finalize-aux-post-collapse-norm`).

## Summary
- **Split lock-free `mhc_claim` + `mhc_wait_splits` (`nanosleep.u32 32` backoff)**:
  - Replaces `mhc_claim_and_wait` with a two-phase protocol so the `40 - live_m` non-finalizer CTAs exit immediately after Stage 1 (`project_streams`) without entering any spin-wait loop, freeing their SMs for downstream kernels.
- **Zero-spin `M = 1` fastpath**:
  - When `live_m == 1` (`M = 1`), the single last-arriving split CTA (`arrived == splits - 1`) resets `counter` to `0` immediately on the first atomic without executing `ld.global.acquire`, a second `__threadfence()`, or a second `atomicAdd`.
- **Late `pdl.trigger()` after Stage 1 (`mhc_claim`)**:
  - Replaces early `pdl.chain()` at kernel entry with `pdl.wait()` at entry and `pdl.trigger()` right after Stage 1 completes and `mhc_claim` finishes. Downstream kernels (`attention_front`, `gemm_mxfp8`, `moe_topk_gate_forward`) begin their CTA launch as the `40 - live_m` non-finalizer CTAs exit and overlap Stage 2 (`finalize`), avoiding SM contention during Stage 1's 40-CTA TF32 GEMM.
- **Pre-barrier `weight` / `scale` / `base` preload overlapped with `mhc_wait_splits`**:
  - In the `live_m` finalizer CTAs, while Thread 0 waits in `mhc_wait_splits` for sibling `M`-row claims to complete, Warp 0 (`tx < 24`) preloads `scale` and `base` and Warps 1-3 (`tx >= 32`) preload all `5,120` BF16 elements (`10.24 KB`) of the RMSNorm `weight` vector into shared memory before the Stage 2 `T.sync_threads()`.

## Patches
- `0001-dsv41-mhc-late-trigger-prewait-multitoken-finalize.patch`: Cumulative patch against `rc21` (`a89da720d`) including PR #821 + this optimization.
- `0002-dsv41-mhc-late-trigger-prewait-multitoken-finalize-incremental.patch`: Incremental commit directly on top of PR #821.
