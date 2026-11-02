# DeepSeek-V4.1-Flash Loop Detector v2: Default Loop Detection, Windowed Hardening & Completion Receipts

Follow-up to [`#787`](https://github.com/christopherowen/spark-ds41f/pull/787) ([`#787 (comment)`](https://github.com/christopherowen/spark-ds41f/pull/787#issuecomment-6559214555)) addressing the detector shortcomings identified during production deployment of `candidate-32..35` (`0656-feat-v1-break-in-think-reasoning-loops-via-dynamic-T.patch`) and making loop detection a default feature across both `<think>` and post-`</think>` / non-`<think>` generation.

## Shortcomings Addressed

1. **Make Loop Detection a Default Feature Across All Phases (`check_stop`)**:
   - **Previous behavior (`0656`)**:
     - `check_stop(request, max_model_len, thinking_loop_break: bool = False)` defaulted `thinking_loop_break` to `False`.
     - After `</think>` (`request.thinking_output_start < 0`), `has_thinking_loop` only ran if `request.thinking_loop_breaks > 0`. If `<think>` closed naturally (`thinking_loop_breaks == 0`) and the model entered a repetition loop in its tool call or final answer, no loop detector ran unless `sampling_params.repetition_detection` was explicitly passed.
     - When `request.thinking_disabled` was `True` (e.g., `thinking={"type": "disabled"}`, prompt ending with `</think>`, or `/v1/completions`), `check_stop` returned `False` early before running any repetition detection, and `to_sampling_params()` suppressed `default_sampling_params["repetition_detection"]`.
   - **Fix**:
     - Default `thinking_loop_break: bool = True` in `check_stop`.
     - Run `detect_thinking_loop` by default across all generation phases (unless explicitly opted out via `RepetitionDetectionParams(max_pattern_size=0)`):
       - **Inside `<think>` (`request.thinking_output_start >= 0`)**: On the first loop (`thinking_loop_break and request.thinking_loop_breaks == 0`), gracefully breaks the loop by setting `request.force_thinking_end = True`, `request.thinking_loop_breaks += 1`, `request.copy_draft_alloc = 0` so the worker forces `</think>` (`128806`) on the next step via `ThinkingBudgetState(budget=0)`. On any subsequent loop inside `<think>`, terminates with `RequestStatus.FINISHED_REPETITION`.
       - **After `</think>` or when thinking is disabled (`request.thinking_output_start < 0`)**: Runs `detect_thinking_loop(request._output_token_ids, start_idx=request.thinking_end_output_idx)` by default (even when `thinking_loop_breaks == 0` or `thinking_disabled == True`) and terminates degenerate loops with `RequestStatus.FINISHED_REPETITION` (`stop_reason="repetition_detected"`), while keeping `request.thinking_disabled` solely to skip `<think>` marker scanning and prevent `force_thinking_end` outside `<think>`.
     - `ChatCompletionRequest.to_sampling_params()` and `ResponsesRequest.to_sampling_params()` now inherit `default_sampling_params.get("repetition_detection")` uniformly.

2. **Eliminate Windowed False Positives on Parallel Enumerations (`_find_windowed_repeating_pattern`)**:
   - **Previous behavior (`0656`)**: `_has_windowed_repeating_pattern` triggered whenever a `5..64`-token suffix `cand` appeared `>= 5` times in the last `160` tokens covering `>= 60%` of the span, even when **zero** occurrences of `cand` were contiguous (`A + A`). In structured reasoning enumerating items in parallel (e.g., `"Rank 0's shard is aligned to 16 bytes.\n"`, ..., `"Rank 4's shard is aligned to 16 bytes.\n"`), the shared 7-token suffix appeared 5 times over 45 tokens (`77.8% >= 60%`) and false-positived.
   - **Fix**: Track `contiguous_pairs` (`nxt == pos - 1`) during the backward window scan and require **`contiguous_pairs >= 1`**. True interspersed loops (`3x A + B + 2x A`, `2x A + B + 2x A + C + 1x A`) always contain at least one back-to-back `A + A` pair, whereas parallel enumerations (`B_0 A + B_1 A + ... + B_4 A`) have `contiguous_pairs == 0`.

3. **Raise Minimum Contiguous Repeated Span from `16` to `24` Tokens (`detect_thinking_loop`)**:
   - **Previous behavior (`0656`)**: `pattern_len = 2` triggered at `8` repeats (`16` tokens) and `pattern_len = 4` triggered at `4` repeats (`16` tokens), which can misfire on legitimate 8-element array initializers (`0, 0, 0, 0, 0, 0, 0, 0`) or markdown table borders inside `<think>`.
   - **Fix**: Require a minimum contiguous repeated span of **`>= 24` tokens** across all pattern lengths (`min_count = max(10, (24 + pattern_len - 1) // pattern_len)` for `1..3`-token patterns; `min_count = max(4, (24 + pattern_len - 1) // pattern_len)` for `4..64`-token phrases).

4. **Raise Windowed Candidate Period Cap from `2` to `4` (`_find_windowed_repeating_pattern`)**:
   - **Previous behavior (`0656`)**: Breaking out of the outer scan at `checked_patterns >= 2` could miss a multi-clause paragraph loop $P = S_1 + S_2 + S_3$ if internal clauses $S_1, S_2, S_3$ shared a `>= 5`-token clause ending that consumed the 2 candidate checks before reaching the previous full paragraph period.
   - **Fix**: Raise the candidate period limit from `2` to `4` (`checked_patterns >= 4`).

5. **Fix Premature `thinking_disabled = True` Latching on Leading Newline/Whitespace (`Request.update_reasoning_markers`)**:
   - **Previous behavior (`0656`)**: When `<think>` was not at the end of the prompt and the model emitted a leading `\n` token on output token 1 before `<think>` on token 2, `cur_num_output >= start_len` (`1 >= 1`) immediately latched `self.thinking_disabled = True` on token 1.
   - **Fix**: Wait until `cur_num_output >= start_len + 3` before latching `self.thinking_disabled = True`.

6. **Close the Observability Gap with Structured Loop Metadata & Completion Receipts (`scheduler.py`)**:
   - **Previous behavior (`0656`)**: The detection log did not record which rule/pattern triggered (`mode`, `pattern_len`, `count`), and `request.request_id` was never logged again on request completion, making it impossible to confirm from production logs whether a loop-broken request recovered (`FINISHED_STOPPED`) or hit `FINISHED_REPETITION`.
   - **Fix**:
     - `detect_thinking_loop` returns `(mode, pattern_len, count)` (`"contiguous_short"`, `"contiguous_phrase"`, or `"windowed_phrase"`), stored on `request.last_loop_info`.
     - The detection log now records `(mode=%s, pattern_len=%d, count=%d)`.
     - When `stopped` is `True` for any request with `request.thinking_loop_breaks > 0` or `request.status == RequestStatus.FINISHED_REPETITION`, `Scheduler.update_from_output` logs a completion receipt with `request_id`, `breaks`, `last_mode`, `pattern_len`, `count`, `status`, `stop_reason`, `output_tokens`, and `post_think_tokens`.

## Patches

- `0001-feat-v1-make-loop-detection-default-and-harden-windo.patch`: Incremental patch on top of `candidate-35` (`0656-feat-v1-break-in-think-reasoning-loops-via-dynamic-T.patch`).
- `0002-feat-v1-break-in-think-reasoning-loops-and-enable-de.patch`: Full squashed replacement for `candidate-33/vllm/0656-feat-v1-break-in-think-reasoning-loops-via-dynamic-T.patch`.

## Verification on `sp8` (`spark-ds41f:candidate-35`)

```text
tests/v1/core/test_repetition_detection.py (29 passed)
  - test_sampling_params_dict_coercion PASSED
  - test_has_thinking_loop_contiguous_and_interspersed PASSED
  - test_windowed_parallel_enumeration_does_not_false_positive PASSED
  - test_short_pattern_min_span_24_tokens PASSED
  - test_windowed_multi_clause_loop_beyond_two_internal_suffix_matches PASSED
  - test_thinking_loop_break_forces_end_then_recovers PASSED
  - test_default_loop_detection_post_think_and_thinking_off PASSED
  - test_leading_newline_before_think_does_not_disable_tracking PASSED
  - test_protocol_to_sampling_params_inherits_default_repetition_detection PASSED
  - test_thinking_enabled_non_looping_state_invariance PASSED
tests/v1/worker/test_gpu_thinking_budget.py (62 passed across triton + tilelang SM121)
tests/v1/core/test_output.py + test_copy_drafts_scheduler.py + test_sink_draft_zero.py (82 passed)
======================= 173 passed in 33.37s =======================
```
