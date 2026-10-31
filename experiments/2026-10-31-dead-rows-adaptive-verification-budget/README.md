# Align DSpark CPU Adaptive Verification Budget with `dead_rows_tau` Survival Cut

## Summary

- **Survival Cut Alignment in `AdaptiveVerificationManager.get_num_tokens` (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`)**:
  - When `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU > 0` (`self.dead_rows_tau > 0`), `combine_sampled_and_draft_tokens_kernel` on the GPU marks verification rows past a request's survival cut (`survival < dead_rows_tau`) as padding (`is_padding = 1`, skipped by routed MoE) and `tilelang_rejection` / `tilelang_resampled` truncate drafts from the cut onward to `-1` placeholders so they can never be committed.
  - Previously, `AdaptiveVerificationManager.get_num_tokens` on the CPU ignored `self.dead_rows_tau` and scored every draft slot against `1 + row_cost * rows` (`row_cost = 0.02`), admitting draft slots with survival between `~0.04` and `dead_rows_tau`. Those slots inflated `draft_budget` and forced larger CUDA graph shapes (paying dense projections, shared experts, mHC, attention, and indexer across all 61 layers) only to be truncated to `-1` on device.
  - Filtering `valid &= survival_probability >= self.dead_rows_tau` when `self.dead_rows_tau > 0` trims those dead slots from the CPU budget so the step selects a smaller CUDA graph, while preserving `max_draft_budget` from `scheduled_drafts.sum()` so the periodic exploration probe (`EXPLORE_AFTER_IDLE_STEPS`) still fires when all stale confidences fall below `dead_rows_tau`.

## Verification

- Verified on `dgx4` inside `spark-ds41f:1030`:
  - `tests/v1/spec_decode/test_adaptive_verification.py` (including `test_dead_rows_tau_trims_cpu_budget_below_survival_cut`) and `tests/v1/worker/test_dead_rows.py`: `31 passed, 1 skipped`.
