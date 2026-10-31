# Elide Redundant Decode-Step Logits Gathers, `cu_num_logits` H2D, and Index Expansions

## Problem

On every pure decode step (including speculative verification decode steps where `adaptive_verification is None` and `self.pcp_manager is None`), `total_num_logits == num_tokens` because every request schedules `num_scheduled_tokens[b] == num_logits[b]`. Consequently:
- `cu_num_logits` is identical to `query_start_loc[:num_reqs + 1]`.
- `logits_indices` is identically the contiguous prefix `[0, 1, ..., total_num_logits - 1]`.

Yet `GPUModelRunner.prepare_inputs` and `GPUModelRunner.sample` (`vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/worker/gpu/input_batch.py`) unconditionally execute on every decode step:
1. A duplicate `np.cumsum` + `async_tensor_h2d` allocation and H2D copy for `cu_num_logits` (in addition to `query_start_loc`).
2. Three `torch.empty` allocations in `combine_sampled_and_draft_tokens` (`logits_indices`) and `expand_idx_mapping` (`expanded_idx_mapping`, `expanded_local_pos`).
3. An ungraphed `expand_idx_mapping` TileLang kernel launch even on all-greedy speculative verification steps (`_verifies_greedily`) that never read `expanded_idx_mapping` or `expanded_local_pos`.
4. An ungraphed `tilelang_inputs.gather_hidden_rows` (`ced.copy_rows`) launch + tensor allocation in `sample()` to gather `hidden_states[0:total_num_logits]`.
5. An ungraphed `tilelang_inputs.logits_rows` launch + two tensor allocations in `InputBatch.logits_rows()` to gather `(input_ids[0:total_num_logits], positions[0:total_num_logits])`.

## Change

1. **Preallocate `logits_indices`, `expanded_idx_mapping`, and `expanded_local_pos` on `InputBuffers` (`vllm/v1/worker/gpu/input_batch.py`)**:
   - Pass `out=self.input_buffers.logits_indices` to `combine_sampled_and_draft_tokens` and `out=(self.input_buffers.expanded_idx_mapping, self.input_buffers.expanded_local_pos)` to `expand_idx_mapping`, eliminating 3 per-step `torch.empty` calls.
2. **Reuse `query_start_loc` for `cu_num_logits` on Pure Decode Steps (`vllm/v1/worker/gpu/model_runner.py`)**:
   - When `draft_tokens` is present, `adaptive_verification is None`, and `total_num_logits == num_tokens`, slice `cu_num_logits_np = query_start_loc_np[:num_reqs + 1]` and `cu_num_logits = query_start_loc[:num_reqs + 1]` instead of running a second `np.cumsum` and `async_tensor_h2d`.
3. **Zero-Copy Contiguous Logits Views (`_logits_rows_contiguous`)**:
   - When `adaptive_verification is None and self.pcp_manager is None and total_num_logits == num_tokens`, mark `input_batch._logits_rows_contiguous = True` (`init=False` so `dataclasses.replace` in batch sharding/PCP/ubatching never inherits it).
   - In `GPUModelRunner.sample`, slice `sample_hidden_states = hidden_states[:n]` directly instead of launching `tilelang_inputs.gather_hidden_rows`.
   - In `InputBatch.logits_rows`, return `(self.input_ids[:n], self.positions[:n])` directly instead of launching `tilelang_inputs.logits_rows`.
4. **Skip `expand_idx_mapping` on All-Greedy Verification Steps**:
   - When `_batch_verifies_greedily` is `True` in `prepare_inputs`, defer `expand_idx_mapping` (recording `_deferred_expand_max_len` so `ensure_expanded_idx_mapping` can materialize it lazily if a caller takes the non-greedy path).
5. **Gate `observe_verification` on `acceptance_estimator is not None`**:
   - Avoid calling `self.speculator.observe_verification(...)` in `sample_tokens` when `self.speculator.acceptance_estimator is None`.

## Verification

- Verified on `dgx4` inside `spark-ds41f:19f2c20ed4d6-next-21`:
  - `pytest --noconftest -q tests/v1/worker/test_dead_rows.py`: **4 passed**.
