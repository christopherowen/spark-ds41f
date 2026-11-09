# PR #965 rework, round 1: two release-test failures in candidate-47

## Root cause
The original 0766 changed `BlockTables.apply_staged_writes()` (`vllm/v1/worker/gpu/block_table.py:178-181`) to `if self.num_kv_cache_groups == 0 or not self._dirty: return`. `_dirty` is set only inside `append_block_ids()` (`:176`), when that call stages a write or changes `num_blocks`.

`test_apply_fused_staged_writes` (`tests/v1/worker/test_gpu_tilelang_inputs.py:320`) and `test_declared_calls_cover_the_call_sites` (`:1327`) stage writes directly on `tables.block_tables[i]`, with `_stage()` (`StagedWriteTensor.stage_write`/`stage_write_elem`, `:326`, `:1344-1345`). Then they call `tables.apply_staged_writes()` (`:351`, `:1347`). With `_dirty` false, it returns before `fused_writer.apply()`. The staged writes are neither applied nor cleared: `out/diag-c47-named.txt` shows 57 writes staged, 0 of 7 tables written, 57 still staged.

- Test 1 compares the tables to the Triton reference and fails (`:354`).
- Test 2 records 14 launched kernels against 15 declared. Launched minus declared is empty. Declared minus launched is `apply_write_kernel('int32', True)`, the FusedStagedWriter kernel in `tilelang_inputs.staged_write_calls()` (`tilelang_inputs.py:2975-2984`).

So the failure is a declared kernel the test no longer reached, not an undeclared launch.

## Was serving affected? No.
- Production stages block-table writes only through `append_block_ids`:
  - `add_requests` (`model_runner.py:1631`, overwrite=True);
  - `update_requests` (`:1662`, table updates, overwrite=True);
  - `update_requests` (`:1670`, new blocks, overwrite=False).
- A search of `vllm/` finds no other `stage_write` on a block table and no other writer of `num_blocks.np`.
- Every serving write sets `_dirty`, so it is applied in the same step's `apply_staged_writes()` (`model_runner.py:2929`).
- The fused write kernel is declared unconditionally (`declare_staged_write_kernels()`, `model_runner.py:2500`) and still launches on every step that allocates a block. No kernel shape escapes preparation, and nothing compiles after readiness.

The tests are not stale, though. They encode the 1107 rule that `apply_staged_writes()` applies whatever is staged on the tables. The original 0766 narrowed that rule silently: a write staged on a table outside `append_block_ids` would sit until the next allocating step and then land late.

## Correction
Skip only when nothing is pending:
```python
if self.num_kv_cache_groups == 0:
    return
if not self._dirty and not any(t._staged_write_indices for t in self.block_tables):
    return
```
This is exactly 1107's no-op case. `FusedStagedWriter.apply` already returns early when no table has writes (`buffer_utils.py:297`), and the skipped `num_blocks.copy_to_uva()` would only re-copy unchanged values. The failing tests are unchanged. One test is added (`test_block_tables_apply_staged_writes_applies_writes_staged_on_a_table`).

Two alternatives were rejected:
- Rewriting the tests to stage through `append_block_ids`. It cannot express their overlapping writes at arbitrary starts, so the tests would lose that coverage.
- Setting `_dirty` in the tests. That ties kernel tests to a private flag and keeps the late-write hazard.

Bit-exactness: the corrected code differs from the original only when something is staged while `_dirty` is false, which never happens in serving.

## Evidence (sp8, spark-ds41f:1107 with candidate-47's 0762..0769 mounted read-only)
**Tests**

| Run | Result |
|---|---|
| `test_gpu_tilelang_inputs.py`, original 0766 | 2 failed, 680 passed, 1 skipped |
| `test_gpu_tilelang_inputs.py`, corrected 0766 | 682 passed, 1 skipped |
| Diagnosis, corrected 0766 | launched = declared = 15; 7/7 tables written |
| 50 gate files of `tests/v1/worker/` and `tests/v1/spec_decode/`, corrected | 4,567 passed, 1 failed, 8 skipped |

All of #965's own files pass. The one failure, `test_workspace.py::test_v41_tilelang_attention_reserves_scratch_in_its_model_lane`, fails the same way with the original 0766 and passes with the series stopped at 0766. Only 0767 (#1012) touches `deepseek_v4_1/attention.py`; see `../pr-1012-r1`. #965's own README test command never ran `test_gpu_tilelang_inputs.py`.

**Harness** (#965's `bench_step_fastpath.py`, total step overhead mean in µs, boot 1 / boot 2, baseline → candidate)

| Concurrency | Original 0766 | Corrected 0766 |
|---|---|---|
| c1 | 253.13→126.00 / 258.82→125.52 | 256.45→124.90 / 255.59→124.23 |
| c8 | 277.39→137.38 / 278.15→135.53 | 280.89→141.00 / 273.00→135.52 |
| c16 | 288.28→144.17 / 279.42→143.69 | 283.19→143.43 / 279.85→144.51 |

The gain is beyond the interval everywhere, and every row is bit-exact. The cost of the correction is about +0.24 µs per clean step in the 7-group geometry (`out/clean-apply-*.json`).

**Verdict:** corrected-meets-mark.

(Written by the rework sub-agent; saved by the coordinator because the harness blocks sub-agents from writing `.md` files.)
