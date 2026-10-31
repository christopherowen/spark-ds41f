# Add IKET Warp-Level Sampling (`VLLM_DS41_IKET_WARPS`) and Static/Runtime Payloads

## Summary

- **Warp-Level Sampling (`warps` / `VLLM_DS41_IKET_WARPS` in `vllm/models/deepseek_v4_1/tilelang/iket.py`)**:
  - Previously, every warp inside a sampled CTA (`ctas > 0`) fired `TL_IKET_EVENT` trace stores concurrently, causing 4–8 warps per CTA (`128`–`256` threads) to contend on the SM's shared/global store pipe at every range boundary.
  - Adds `warps: int = 0` (`VLLM_DS41_IKET_WARPS`): when `warps > 0`, injects a warp-uniform predicate `__tl_iket_warp` at kernel entry (`(((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) >> 5) % max(1u, ((blockDim.x * blockDim.y * blockDim.z + 31u) >> 5) / warps) == 0u`) and includes it in `TL_IKET_WHEN(...)`.
  - With `warps=2`, a 256-thread CTA (`sparse_mla`) samples warp `0` (warp-group `0`) and warp `4` (warp-group `1`), and a 128-thread CTA (`moe_grouped_fp8_fp4_gemm`) samples warp `0` (including the `tl_shuffle_elect<128>()` TMA issuer) and warp `2`, cutting per-CTA trace store contention by 2x–4x while preserving warp-group coverage.
- **Static Transfer Byte Metadata & Optional `uint32` Runtime Payloads (`payloads` / `VLLM_DS41_IKET_PAYLOADS`)**:
  - Extracts static TMA transaction bytes (`tma_bytes` from `expect_transaction` / `arrive_and_expect_tx`) and per-thread `cp.async` bytes (`cp_async_bytes_per_thread` from `tl::cp_async_gs*<BYTES>`) into each site's manifest entry (`sources/<kernel>.<hash>.json`).
  - Adds optional `payloads: bool = False` (`VLLM_DS41_IKET_PAYLOADS=1`), which registers `uint32` payload metadata (`payload_iket_id=6`), passes `runtime_payloads=True` to `inject_iket_cuda`, and emits `TL_IKET_EVENT_PAYLOAD_U32(1, <expr>, "<token>")` carrying the innermost `for`-loop variable (or CTA linear index outside loops).

## Verification

- Verified on `dgx4` inside `spark-ds41f:1030` (holding `/tmp/spark-lab-gpu.lock`):
  - `tests/kernels/test_deepseek_v41_tilelang_iket.py`: `14 passed` (including `test_sampled_warps_and_payloads`).
  - `sass_check.py` on `sparse_mla`, `moe_grouped_fp8_fp4_gemm`, and `oneshot_allreduce` instrumented with `ctas=1, iters=1, warps=2, payloads=True`: `0` predicated events (`TL_IKET_WHEN` compiles `__tl_iket_cta && __tl_iket_warp` as a uniform branch), `0` hoisted timer reads, and `0` unsettled block barriers.
