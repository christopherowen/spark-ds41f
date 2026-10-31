# Exact Variable-Length Decode CUDA Graphs for Odd `c2` and Full-Depth Speculative Batches

## Summary

`dense_varlen_decode_shapes` in `vllm/v1/worker/gpu/cudagraph_utils.py` previously generated `(num_reqs, num_reqs * query_len)` only for `num_reqs in (1, 2)`:
- At `num_reqs = 2`, only even token counts (`2, 4, 6, 8, 10, 12` for `decode_query_len = 6`) were captured. Whenever `AdaptiveVerificationManager.get_num_tokens` selected an odd total draft budget across 2 requests (`num_tokens in {3, 5, 7, 9, 11}`), or when a newly prefilled request with 1 token joined a request with 6 tokens (`num_tokens = 7`), `CudaGraphManager.dispatch` padded the batch up to the next even token count — running the extra row across all 61 layers (122 TP `all_reduce` collectives = `+1.22 MiB` RoCE traffic per padded row per step) on a dummy padding token instead of the trimmed draft slot.
- At `num_reqs >= 3` with `decode_query_len = 6` (`K = 5` drafts), full-depth speculative batches (`18, 30, 36, 42, 54, 60, 66, 78, 84, 90` tokens) fell between the power-of-two / step-of-4/8 `cudagraph_capture_sizes` ladder (`16 -> 20`, `28 -> 32`, `32 -> 40`, `40 -> 48`), padding by `2`, `4`, or `6` extra token rows (`+2.44 MiB` to `+7.32 MiB` RoCE `all_reduce` traffic per step) and padding `num_reqs` up to `min(num_tokens, max_num_reqs)` (`8` on TP3 or `16` on TP4).

## Changes

- **`vllm/v1/worker/gpu/cudagraph_utils.py`**:
  - Update `dense_varlen_decode_shapes(max_num_reqs, decode_query_len, max_capture_tokens)` to emit:
    1. All valid token counts `num_reqs .. num_reqs * decode_query_len` for `num_reqs in 1..min(_DENSE_VARLEN_DECODE_MAX_REQS, max_num_reqs)` (adding `(2, 3), (2, 5), (2, 7), (2, 9), (2, 11)` at `decode_query_len = 6`).
    2. Exact full-depth speculative decode shapes `(num_reqs, num_reqs * decode_query_len)` for `num_reqs in (_DENSE_VARLEN_DECODE_MAX_REQS + 1)..max_num_reqs` when `decode_query_len > 1` (covering `(3, 18), (4, 24), (5, 30), (6, 36), (7, 42), (8, 48), ...`).
  - Sort `_capture_descs[mode]` by `(d.num_tokens, d.num_reqs or 0)` descending so the largest `(num_tokens, num_reqs)` descriptor is captured first when multiple descriptors share a token count.
  - Because `vllm/model_executor/warmup/b12x_prepare.py` calls `dense_varlen_decode_shapes` in `_planned_decode_counts`, both the `weights` and `state` preparation stages automatically include the exact decode token counts.
- **`tests/v1/cudagraph/test_cudagraph_manager.py`**:
  - Update `test_varlen_decode_captures_dense_low_concurrency_product` and `test_varlen_dense_rejection_keeps_larger_full_fallback`.
  - Add `test_varlen_decode_covers_odd_c2_and_non_divisor_uniform_batches` verifying zero-padding dispatch across all `1..2`-request token totals and all `1..8`-request `decode_query_len = 6` full-depth batches.

## Verification

Tested on `dgx4` inside `spark-ds41f:1030`:
- `tests/v1/cudagraph/test_cudagraph_manager.py`: passed
- `tests/v1/spec_decode/test_dynamic_sd_cug.py`: passed
- `tests/v1/spec_decode/test_adaptive_verification.py`: passed
- `tests/model_executor/test_b12x_warmup.py`: `55 passed`
- `tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py`: passed
- Total: **`153 passed, 1 skipped`**.
