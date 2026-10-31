# DeepSeek-V4.1 TileLang MoE: Pre-Wait Metadata Hoisting & Inactive-Tile Early Exit in `moe_swiglu_cast` and FC2 `grouped_fp8_fp4_gemm`

## Summary

The 43-layer IKET forward-pass trace (`table_43l.md` in PR #777) identified two stalls in the routed MoE pipeline (`vllm/models/deepseek_v4_1/tilelang/moe.py`):

1. **`moe_swiglu_cast_kernel` (`41` launches, `38.64 µs/launch`, `20.3%` `MEM_DEPENDENCY / LONG_SCOREBOARD`):**
   - `moe_dispatch` produces `Order`, `Weights`, and `TileExpert` **before** FC1 (`grouped_fp8_fp4_gemm(gather=True)`) even starts, while FC1 only writes `H`.
   - However, `moe_swiglu_cast_kernel` previously called `pdl.chain()` (`cudaGridDependencySynchronize()`) at the very top of the kernel before reading `TileExpert[t]`, `Order[base + r]`, or `Weights[base + r]`.
   - Every CTA therefore stalled on FC1's completion before even checking whether its tile was active (`TileExpert[t] >= 0`), and active CTAs suffered a cold global memory round-trip on `Order` and `Weights` after FC1 completed instead of overlapping those loads with FC1's tail.
2. **FC2 `moe_grouped_fp8_fp4_gemm` (`gather=False`, `41` launches, `88.74 µs/launch`, `43.2%` `WAIT / BARRIER` and `11.1%` `MEM_DEPENDENCY`):**
   - In FC2 (`gather=False`), `TileExpert`, `Order`, `SFBase`, and `Narrow` are all produced by `moe_dispatch` (or are static expert scale metadata), whereas the immediate predecessor (`moe_swiglu_cast_kernel`) only writes `A` (`Out`) and `ASF` (`OutSF`).
   - Waiting on `pdl.wait()` before loading `TileExpert[by]`, `Order[row]`, `SFBase`, and `Narrow` forced inactive tiles (`expert < 0`) to wait for `moe_swiglu_cast_kernel` before exiting and forced active tiles to serialize `rows_shared` setup and scale metadata loads after `pdl.wait()`.

## Changes

- **`moe_swiglu_cast` (`vllm/models/deepseek_v4_1/tilelang/moe.py`):**
  - Load `expert = TileExpert[t]`, `orders[r] = Order[base + r]`, and `weights[r] = Weights[base + r]` (when `expert >= 0` and `orders[r] >= 0`) **before** `pdl.chain()`.
  - Inactive tiles (`expert < 0`) trigger `pdl.trigger()` and exit without waiting on FC1 (`pdl.wait()`), while active tiles wait on FC1 only immediately before loading `H`.
- **`grouped_fp8_fp4_gemm` (`gather=False`, `vllm/models/deepseek_v4_1/tilelang/moe.py`):**
  - Load `expert = TileExpert[by]` before `pdl.wait()`.
  - When `expert >= 0`, populate `rows_shared[i] = Order[by * block_M + i]`, `base_shared`, and `narrow_shared` while FC1/`moe_swiglu_cast` finishes, then call `pdl.wait()` right before `T.sync_threads()` and the `T.Pipelined` loop.
  - When `expert < 0`, trigger `pdl.trigger()` immediately without waiting on `moe_swiglu_cast`.
- **PDL Source Audit (`tests/kernels/deepseek_v41_pdl_sources.py`):**
  - Register `Order`, `TileExpert`, `SFBase`, and `Narrow` for `moe_grouped_fp8_fp4_gemm` and `Order`, `TileExpert`, and `Weights` for `moe_swiglu_cast_kernel` in `EARLY_PARAMETERS`.

## Verification & Benchmark (`dgx4`, GB10 SM121, `spark-ds41f:19f2c20ed4d6-next-20`)

### Correctness & PDL Source Check
- `pytest -q -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/moe/test_deepseek_v41_tilelang_moe.py tests/kernels/test_deepseek_v41_tilelang_pdl.py`:
  - All tests pass with `0` PDL source violations.

### CUDA-Graph Replay Latency (`20` calls/graph, `200` replays)

| Kernel / Plan | Config | Baseline (`rc20`) | Patched | Delta |
|---|---|---:|---:|---:|
| `moe_swiglu_cast` (`576`, `block_M=32`) | `active = 6 / 36` tiles (`rows=1`) | `2.270 µs` (`p10=2.197 µs`) | **`1.763 µs`** (`p10=1.718 µs`) | **`-22.3%`** |
| `moe_swiglu_cast` (`576`, `block_M=32`) | `active = 16 / 36` tiles | `2.877 µs` (`p10=2.816 µs`) | **`2.837 µs`** (`p10=2.795 µs`) | **`-1.4%`** |
| `MoeExpertsV41Plan.run` (`H=5120, I=576`) | `rows = 1` (`topk=6`) | `92.650 µs` (`p10=92.291 µs`) | **`90.192 µs`** (`p10=89.776 µs`) | **`-2.7%` (`-2.46 µs/layer`)** |
