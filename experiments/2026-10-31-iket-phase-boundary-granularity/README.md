# Add Phase-Boundary IKET Range Granularity (`VLLM_DS41_IKET_GRANULARITY=phase`)

## Summary

- **Phase-Boundary Range Coalescing (`vllm/models/deepseek_v4_1/tilelang/iket.py`)**:
  - `run-iket`'s single bump-allocated trace buffer reserves the run's largest per-warp record across *every warp of every launch*. In the default `"fine"` granularity, every `wait` statement is its own range, pushing `sparse_mla` to `32` ranges (`64` events/warp) and sizing the entire process's `run-iket` pool ~4x larger than any other kernel needs.
  - Adds `granularity="phase"` (`VLLM_DS41_IKET_GRANULARITY=phase`):
    1. Non-jumping blocks (`for`, `if`, braced scopes) whose statements combine a single non-wait phase (`load`, `compute`, or `store`) with internal block synchronization (`wait.sync`, e.g., online-softmax `tl::AllReduce` loops or `ldmatrix`+`mma_sync` scopes in `sparse_mla`) are classified as that phase instead of `"mixed"`.
    2. Consecutive `wait` siblings (such as `tl::cp_async_wait` + `__syncthreads` + `mbarrier.wait` + `__syncthreads` in `moe_grouped_fp8_fp4_gemm` or `roce_wait_peers` + `__syncthreads` in `oneshot_allreduce`) coalesce into a single `wait` range.
    3. Internal `wait.sync` statements between two consecutive siblings of the same phase bridge into that phase run.
- **Worker & Environment Wiring (`vllm/envs.py`, `vllm/v1/worker/gpu_worker.py`)**:
  - Adds `VLLM_DS41_IKET_GRANULARITY` (`"fine"` default, or `"phase"`) and passes it to `iket.enable(...)` in `Worker.init_device`.

## Verification

- Verified on `dgx4` inside `spark-ds41f:1030` (holding `/tmp/spark-lab-gpu.lock`):
  - `tests/kernels/test_deepseek_v41_tilelang_iket.py`: `14 passed` (including `test_phase_granularity_merges_waits_and_internal_sync_blocks`).
  - Across the 29 DS4.1 TileLang hot kernels (`ctas=1, iters=1`), `granularity="phase"` reduces total ranges from `294` to `162` and lowers the worst-case `events_per_warp_bound` across the engine from `64` (`sparse_mla`) to `22` (`mhc_finalize`):
    - `sparse_mla` (unsplit): `32` ranges (`64` events/warp) -> `8` ranges (`16` events/warp, `-75%`).
    - `sparse_mla` (split-KV): `31` ranges (`50` events/warp main + `12` combine) -> `9` ranges (`10` events/warp main + `8` combine, `-80%`).
    - `moe_grouped_fp8_fp4_gemm`: `15` ranges (`30` events/warp) -> `8` ranges (`16` events/warp, `-47%`).
    - `moe_dispatch`: `12` ranges (`24` events/warp) -> `5` ranges (`10` events/warp, `-58%`).
    - `oneshot_allreduce`: `11` ranges (`22` events/warp) -> `9` ranges (`18` events/warp, `-18%`).
  - `sass_check.py`: `0` predicated events, `0` hoisted timer reads, and `0` unsettled block barriers across `sparse_mla`, `moe_grouped_fp8_fp4_gemm`, and `oneshot_allreduce`.
