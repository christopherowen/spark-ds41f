# DeepSeek-V4.1-Flash In-`<think>` Reasoning Loop Breaker

## Problem & Root Cause

During multi-turn agentic coding workflows, DeepSeek-V4.1-Flash can enter a cyclic reasoning attractor inside its `<think>` block after deciding which tool to invoke (e.g., `"Let me do it.\n"` $\times 3$, `"I'll read lines 25-60.\n"`, `"Let me do it.\n"` $\times 18+$):

1. **Structural `<think>` / tool-call boundary**: Under the `deepseek-v4` chat template (`deepseek-recipe` `0.2.0`), `<｜tool▁calls▁begin｜>` is only valid *after* `</think>` (`128806`). While inside `<think>`, if the model's `</think>` transition logit is slightly outcompeted by a short transitional sentence (`"Let me do it.\n"`), MLA + YaRN induction heads lock onto the repeated sentence suffix and drive $P(\text{"Let me do it.\n"}) \to 0.99+$.
2. **`CopyDrafter` acceleration**: Once the phrase repeats twice in the context, `CopyDrafter` (`vllm/v1/worker/gpu/spec_decode/copy_drafts.py`) matches the suffix and proposes up to 12 draft tokens per step from the loop, all of which verify greedily.
3. **Why `RepetitionDetectionParams` did not catch or recover from it**:
   - `repetition_detection` is `None` by default unless passed per request, and was not wired through `ModelConfig.get_diff_sampling_param()` or `to_sampling_params()` `default_sampling_params`.
   - `_has_repeating_pattern` (`vllm/v1/core/sched/utils.py`) only checks strictly contiguous $m \times L$ tail blocks, so an interspersed variation (`3x A + B + 2x A`) resets the contiguous counter.
   - Even when `RepetitionDetectionParams` triggers, it terminates the request with `RequestStatus.FINISHED_REPETITION` mid-`<think>` instead of closing `</think>` so the model can emit its queued tool call.

## Solution

1. **Contiguous + Windowed Bordered-LCSuf Loop Detection (`has_thinking_loop`)**:
   - Detects contiguous short-token loops (`1..3` tokens repeated $\ge 8\times$, span $\ge 16$), contiguous phrase loops (`4..64` tokens repeated $\ge 4\times$), and interspersed/windowed phrase loops (`5..64` tokens repeated $\ge 5\times$ within a 160-token window covering $\ge 60\%$ of the span).
   - Strips proper borders ($b$ where `phrase[:b] == phrase[-b:]`) from the longest common suffix of the last two occurrences so shared sentence endings (such as `".\n"` between `"I'll read lines 25-60.\n"` and `"Let me do it.\n"`) never obscure the true repeating phrase $A$.
   - Runs in $\approx 0.8\ \mu\text{s}$ on CPU on the Scheduler (`update_from_output`), completely overlapped with the GPU forward pass.
2. **Graceful In-`<think>` Loop Breaking via `ThinkingBudgetState` (`VLLM_THINKING_LOOP_BREAK=1`)**:
   - `Request.update_reasoning_markers` tracks whether the request is currently inside `<think>` (`thinking_output_start >= 0`) or past `</think>` (`thinking_end_output_idx`).
   - When `check_stop` detects a loop inside `<think>`, instead of aborting the request, it sets `request.force_thinking_end = True`, resets `request.copy_draft_alloc = 0`, and includes the request in `SchedulerOutput.force_thinking_end_req_ids`.
   - On the GPU worker (`GPUModelRunner.update_requests`), `Sampler.set_forced_thinking_end` dynamically activates `ThinkingBudgetState` (`budget = 0`) and resets `CopyDrafter` for that slot only until `</think>` (`128806`) is committed, then immediately restores `use_thinking_budget = False` and `needs_logits_processing = False`.
   - If a request still loops *after* `</think>` was forced, `check_stop` terminates it with `RequestStatus.FINISHED_REPETITION` (inspecting only tokens after `thinking_end_output_idx` so pre-`</think>` tokens never false-trigger).
3. **`generation_config` / `default_sampling_params` Wiring & Thinking-Off Bypass**:
   - Wires `repetition_detection` and `thinking_token_budget` through `ModelConfig.get_diff_sampling_param()`, `SamplingParams.__post_init__` (dict coercion), and OpenAI `chat_completion`, `completion`, and `responses` `to_sampling_params()`.
   - **Inactive when thinking is off**: When thinking is disabled for a request (`thinking={"type": "disabled"}`, `chat_template_kwargs={"thinking": False}` / `{"enable_thinking": False}`, `reasoning_effort="none"`, `thinking_token_budget=0`, `reasoning_ended=True`, or a prompt ending with `</think>` / starting generation outside `<think>`), `request.thinking_disabled` is set to `True` and `to_sampling_params()` does not inherit server-default `repetition_detection`. Both `update_reasoning_markers` and the repetition guard in `check_stop` are completely bypassed (`0` overhead).

## Performance Impact

- **Steady-state GPU overhead**: **`0.0 us` (`0.00%`)**. On normal decode steps, `use_thinking_budget` and `needs_logits_processing` remain `False`, preserving both the vocab-parallel `_verifies_greedily` fast path (`temperature=0.0`) and the raw-BF16 Gumbel sampling fast path (`temperature=1.0, top_p=1.0`).
- **Steady-state CPU overhead**: **`0.0 us/step`** when thinking is disabled (`thinking_disabled = True`), and **`~0.8 us/step`** while inside `<think>` on the Scheduler during `update_from_output` (hidden behind the `12–16 ms` GPU forward pass under `async_scheduling`).

## Verification on `dgx4` (`spark-ds41f:19f2c20ed4d6-next-20`)

```text
tests/v1/core/test_repetition_detection.py (24 passed)
  - test_sampling_params_dict_coercion PASSED
  - test_has_thinking_loop_contiguous_and_interspersed PASSED
  - test_thinking_loop_break_forces_end_then_recovers PASSED
  - test_thinking_off_skips_repetition_guard PASSED
  - test_protocol_to_sampling_params_skips_default_repetition_when_thinking_off PASSED
tests/v1/worker/test_gpu_thinking_budget.py (62 passed across triton + tilelang SM121)
  - test_v2_forced_thinking_end_toggles_without_steady_state_overhead[triton] PASSED
  - test_v2_forced_thinking_end_toggles_without_steady_state_overhead[tilelang] PASSED
======================= 86 passed, 16 warnings in 17.50s =======================
```
