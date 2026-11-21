# Loop Containment & Thinking-Budget Multi-Token Verification Invariance (`#864`)

- **Date:** 2026-11-21
- **Base commit:** `a5109bb24` (`spark-ds41f:candidate-63-r5`)
- **Causal variable:** Multi-token speculative verification (`1..5` DSpark and `16..64` CopyDrafter rows) invariance for reasoning loop containment (`scheduler.py` / `sched/utils.py`) and GPU thinking-budget enforcement (`tilelang_thinking.py` / `_triton_refs.py`).

## Findings

Auditing and testing `check_stop`, `Scheduler`, `AsyncScheduler`, `ThinkingBudgetState`, and `RejectionSampler` (`block` and `standard` verification) across verify widths `K = 0` (AR), `5` (DSpark), `16` (CopyDrafter default), and `64` (CopyDrafter max) confirmed:

1. **Per-token loop truncation is already verify-width invariant**: `Scheduler._update_request_with_output` appends one committed token at a time, runs `update_reasoning_markers` and `check_stop` per token, and trims `del new_token_ids[num_new:]` on stop. Custom `repetition_detection`, answer-after-forced-end loops, persistent answer token-soup (`OUTPUT_SOUP_PERSIST = 256`), and periodic answer loops (`OUTPUT_LOOP_PERSIST = 2048`) all truncate at the exact same token index across `W = 1, 5, 16, 64`.
2. **Thinking-budget cuts across `K = 0, 5, 16, 64` are already verify-width invariant** for greedy, sampled (`temperature + top_p + top_k + min_p`), single- and multi-token `reasoning_end_token_ids`, `dead_row_cut`, and both `block` and `standard` rejection sampling.
3. **Three concrete multi-token / parser-parity bugs were found and fixed**:
   - **Async `copy_draft_alloc` clobber (`vllm/v1/core/sched/scheduler.py`)**: Under `--async-scheduling`, when Step `t` detected a reasoning loop and set `request.force_thinking_end = True` and `request.copy_draft_alloc = 0`, the already-in-flight Step `t+1` returned `model_runner_output.copy_draft_alloc[req_index] > 0` and unconditionally overwrote `request.copy_draft_alloc` before Step `t+2` (the step carrying `force_thinking_end_req_ids`) was scheduled. Fixed by guarding `update_from_output` and `_widen_copy_placeholders` with `not request.force_thinking_end`.
   - **Unscaled `FORCED_REASONING_END_GRACE = 64` (`vllm/v1/core/sched/utils.py`, `scheduler.py`)**: At wider CopyDrafter verification widths (`num_verification_tokens = 32..64`), the remainder of Step `t` plus the in-flight Step `t+1` can emit up to `2 * (num_verification_tokens + 1)` tokens before Step `t+2` lands `</think>`, prematurely terminating with `FINISHED_REPETITION`. Added `forced_reasoning_end_grace: int = FORCED_REASONING_END_GRACE` to `check_stop` and passed `max(FORCED_REASONING_END_GRACE, 2 * (self.num_verification_tokens + 1))` from `Scheduler._update_request_with_output`.
   - **Quoted `<think>` inside open reasoning reset the budget counter (`vllm/v1/worker/gpu/sample/tilelang_thinking.py`, `_triton_refs.py`)**: `request.py:update_reasoning_markers` and `deepseek_v4.py` follow the 2-state reasoning-parser automaton (inside `<think>`, repeated/quoted `<think>` is reasoning text and only `</think>` closes; outside `<think>`, only `<think>` opens and stray `</think>` is text). Previously `marker_cache_kernel` and `thinking_kernel` blindly overwrote `last_start` on every `<think>`, letting a quoted `<think>` in the prompt, committed output, or speculative drafts reset `thinking_token_budget` (reproduced on `dgx4`: 2-token budget overrun in both cold scan and draft scan). Updated cold scan (`last_end = max{end}`, `last_start = min{start > last_end}`, keeping the `budget == 0` 1-chunk early exit), incremental scan, and speculative draft scan in both TileLang and Triton reference kernels.

## Verification (`dgx4`, `spark-ds41f:candidate-63-r5`)

- **Patch**: [`loop_thinking_verify_invariance.patch`](loop_thinking_verify_invariance.patch)
- **New test suite**: [`test_loop_thinking_verify_invariance.py`](test_loop_thinking_verify_invariance.py)
- **Exact command**:
  ```bash
  pytest -q tests/v1/core/test_repetition_detection.py \
            tests/v1/worker/test_gpu_thinking_budget.py \
            tests/v1/sample/test_thinking_budget_state.py \
            experiments/2026-11-21-loop-thinking-verify-invariance/test_loop_thinking_verify_invariance.py
  pytest -q tests/kernels/test_deepseek_v41_tilelang_pdl.py
  ```
- **Results**:
  - **`116 passed, 0 failed`** (`104` existing repetition + GPU Triton/TileLang thinking-budget tests + `12` new multi-token verify-width invariance & quoted-`<think>` tests).
  - **`15 passed, 0 failed`** PDL verifier (`test_deepseek_v41_tilelang_pdl.py`).

## Conclusion

All three `#864` multi-token speculative verification & reasoning-parser parity gaps are fixed, unit- and GPU-verified (`131/131` passed across Triton, TileLang, scheduler, and PDL gates), with zero overhead on unbudgeted requests. Ready to fold into `candidate-64`.
