# DeepSeek-V4.1 TileLang Compressor: Preload Step Metadata and RMSNorm Weight Before `pdl.chain()`

## Summary

The 43-layer IKET forward-pass trace (`table_43l.md` in PR #777) showed `56.7%` and `43.2%` `MEM_DEPENDENCY / LONG_SCOREBOARD` stalls in the CSA KV-compressor kernels (`vllm/models/deepseek_v4_1/tilelang/compress.py`):
- **`gather_partial_state_kernel` (`10` launches, `56.7%` `MEM_DEPENDENCY`):**
  - Placed `pdl.chain()` at line 1 before reading `counts`, `starts`, `positions`, and `table`, all of which are step-level metadata prepared at the start of the step, whereas the immediate predecessor (`pool` state ring) is only accessed when `valid` is true.
- **`compress_kernel` (`21` launches, `43.2%` `MEM_DEPENDENCY`):**
  - Placed `pdl.chain()` at line 1 before running `_locate` (`counts`, `starts`, `positions`), reading `slots` / `state_slots`, and loading the static `512`-element RMSNorm `weight` tensor (`4` chunks of `4` FP32 values per lane).
  - Only `values`, `gates`, `pending_values`, `pending_gates`, `tags`, `state_ids`, and `pool` depend on the immediate predecessor (`wk_v_gate` projection and `gather_partial_state_kernel`).

## Changes

- **`vllm/models/deepseek_v4_1/tilelang/compress.py`:**
  - In `gather_partial_state_kernel`: hoist the request/slot metadata loads (`counts`, `starts`, `positions`, `table`) before `pdl.chain()` so only the `pool` ring loads wait on the predecessor.
  - In `compress_kernel`: hoist `_locate` (`counts`, `starts`, `positions`), `out_slot = slots[token]`, `ring_slot = state_slots[token]`, and the `4` vectorized loads of `weight` into a register array `w[_CHUNKS, _VEC]` before `pdl.chain()`.
- **`tests/kernels/deepseek_v41_pdl_sources.py`:**
  - Register `counts`, `positions`, `starts`, and `table` for `compress_gather_partial_state` and `counts`, `positions`, `slots`, `starts`, `state_slots`, and `weight` for `compress` in `EARLY_PARAMETERS`.

## Verification & Benchmark (`dgx4`, GB10 SM121, `spark-ds41f:19f2c20ed4d6-next-20`)

### Correctness & PDL Source Check
- `pytest -q -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/test_deepseek_v41_tilelang_compress.py tests/kernels/test_deepseek_v41_tilelang_pdl.py`:
  - All tests pass with `0` PDL source violations.

### CUDA-Graph Replay Latency (`20` calls/graph, `200` replays)

| Kernel Chain | Config | Baseline (`rc20`) | Patched | Delta |
|---|---|---:|---:|---:|
| `gather_partial_state + compress` | `ratio = 2, rows = 14` | `4.916 µs` | **`4.865 µs`** | **`-1.0%`** |
