# DeepSeek-V4.1 TileLang Sparse MLA: Preload Sinks, Positions, and CosSin Before `pdl.chain()`

## Summary

The 43-layer IKET forward-pass trace (`table_43l.md` in PR #777) showed that `sparse_mla_kernel` (`extend_body`) and the `decode_body` combine kernel (`sparse_mla_kernel_1`) in `vllm/models/deepseek_v4_1/tilelang/sparse_mla.py` placed `pdl.chain()` (`cudaGridDependencySynchronize()`) at line 1 of the kernel body before loading:
- `Sinks` (`sink_log2(Sinks, ...)`), a static model parameter;
- `Positions[row]` and `CosSin[pos, ...]`, step-level RoPE metadata that was loaded at the very end of the kernel in `wo_input` (`28.4%` `MEM_DEPENDENCY` stall on `wo_input`'s `CosSin` load after `O_shared` was written).

Because the immediate predecessor of `extend_body` (`metadata_decode` / `rope`) only writes `SwaIndices`, `SwaLengths`, `IdxLengths`, `IdxPageTable`, and `Q`, and the immediate predecessor of `sparse_mla_kernel_1` (`sparse_mla_kernel` split stage) only writes `Partial` and `PartialStats`, `Sinks`, `Positions`, and `CosSin` can all be preloaded before `pdl.chain()`.

## Changes

- **`vllm/models/deepseek_v4_1/tilelang/sparse_mla.py`:**
  - In `extend_body` and `decode_body`'s combine kernel (`sparse_mla_kernel_1`), allocate `cs_shared = T.alloc_shared([ROPE_DIM], accum_dtype)` (`256` bytes) and preload `M` from `Sinks` and `cs_shared` from `CosSin[Positions[row], :]` **before** `pdl.chain()`.
  - Update `wo_input` to read `cs_shared` directly from shared memory when `position is None` (while preserving the 8-argument `(O_shared, heads, position, CosSin, X, amax, inv, scale)` signature used by unit tests).
- **`tests/kernels/deepseek_v41_pdl_sources.py`:**
  - Register `CosSin`, `Positions`, and `Sinks` for `sparse_mla` in `EARLY_PARAMETERS`.

## Verification & Benchmark (`dgx4`, GB10 SM121, `spark-ds41f:19f2c20ed4d6-next-20`)

### Correctness & PDL Source Check
- `pytest -q -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/attention/test_deepseek_v41_tilelang.py tests/kernels/test_deepseek_v41_tilelang_pdl.py`:
  - All tests pass with `0` PDL source violations.

### CUDA-Graph Replay Latency (`20` calls/graph, `200` replays)

| Kernel Route | Batch (`rows`) | Baseline (`rc20`) | Patched | Delta |
|---|---:|---:|---:|---:|
| `SparseMlaV41Plan` (`split`, `128` heads) | `rows = 1` | `12.938 µs` | **`12.707 µs`** | **`-1.8%` (`-0.23 µs/layer`)** |
| `SparseMlaV41Plan` (`extend`, `128` heads) | `rows = 14` | `100.321 µs` | **`99.479 µs`** | **`-0.84 µs/layer` (`-36.2 µs / 43 layers`)** |
