# Real-row engine-grid profiler writable KV isolation

## Base

- Repository base commit: `f502a0387` (`main`, post-`PR #1269`)
- Container image / vLLM base: `spark-ds41f:candidate-63-r5` (`candidate-63` series, `0820`)
- Single causal variable: isolate writable KV blocks (circular ring blocks and paged tail blocks) across concurrent requests and depths in `GPUModelRunner.engine_grid` / `_fill_real_context` / `_real_dummy_rows`.

## Problem

In `0820`:
1. `_real_dummy_rows` pointed `table[1:num_reqs, :1] = table[:1, :1]` for circular ring groups, causing concurrent requests in the same batch to overwrite one another's ring KV slots and corrupt request 0's ring history.
2. `_fill_real_context(max_context_len)` filled only up to `max_context_len`, leaving circular ring groups at the deepest context's state rather than each requested depth's history.
3. For paged groups, request 0 wrote its new step tokens into `table[0, context_len // block_size]`, overwriting filled history belonging to deeper shapes or `fill_real_context=False` repeats.
4. `engine_grid` unconditionally zeroed every ring block before checking `fill_real_context`, wiping out the ring history that `fill_real_context=False` was meant to preserve.

## Change (`vllm/0001-fix-real-rows-profiler-writable-kv-isolation.patch`)

- **`_fill_real_context(context_len, *, depths=...)`**: steps through chunks stopping exactly at every requested depth `d in sorted(depths)`, writing paged history into immutable group offsets `0 .. history_kv_blocks - 1` and snapshotting circular ring groups from offset `0` into dedicated read-only per-depth snapshot blocks (`1 .. len(depths)`) via `copy_kv_cache_blocks_inplace`.
- **`_real_dummy_rows`**: gives every concurrent request `r` strictly disjoint writable ring blocks (`1 + num_snapshots + r`, copied from its depth's immutable snapshot) and strictly disjoint writable paged tail blocks (`>= history_kv_blocks`), supporting both uniform `int` and per-request `Sequence[int]` depths.
- **`engine_grid`**: preserves saved ring snapshot blocks when `fill_real_context=False` (excluding them from `kv_block_zeroer` and raising `ValueError` if a requested depth was not previously initialized), and restores writable ring blocks via `_prepare_real_row_ring_blocks` right before `start.record()` so timed step measurements exclude copy overhead.

## Verification (`dgx4`, CUDA GPU)

Executed on `dgx4` against `spark-ds41f:candidate-63-r5` with patched `model_runner.py` bind-mounted at `/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/model_runner.py`:

```bash
pytest --noconftest -v \
  /opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_engine_grid_sync.py \
  /tmp/test_real_rows_kv_isolation.py
```

Result: **`6 passed, 0 failed` (`15.54 s`)**:
- `test_engine_grid_synchronizes_tp_ranks_and_pauses_gc`: PASSED
- `test_engine_grid_restores_gc_on_exception`: PASSED
- `test_overlay_loaded`: PASSED
- `test_real_rows_multi_depth_isolation_reverse_and_repeat`: PASSED (distinct histories at `4096` and `8192`; solo `[A]`, solo `[B]`, batched `[A, B]`, reversed `[B, A]`, and `engine_grid(..., fill_real_context=False)` repeat all `torch.equal`; snapshots and paged history byte-for-byte unmodified)
- `test_fill_real_context_false_rejects_uninitialized_depth`: PASSED
- `test_real_rows_virtual_block_splitting_bpk4`: PASSED (`block_size=256, kernel_block_size=64` matching DS4.1 Flash)

## Conclusion

Ready to carry into the next candidate vLLM patch series (`candidate-64`).
