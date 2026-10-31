# Experiment: Skip Speculative Drafting When No Request Can Use Drafts (`856b98415b`)

## Summary
Ports upstream vLLM commit [`856b98415b`](https://github.com/vllm-project/vllm/commit/856b98415b) (`[Spec Decode][MRV2] Skip drafting in steps where no request can use the drafts (#62784)`) onto the `rc21` candidate stack (`DFlashSpeculator` / `DSparkSpeculator`, `sink_rejection`, `CopyDrafter`, and `adaptive_verification`).

## Motivation
Previously, `GPUModelRunner.sample_tokens` ran the full speculative draft pass on every step where `self.speculator is not None`, even when every scheduled request in the step was either:
- In a non-final chunked prefill (`num_computed_tokens + num_new_tokens < num_tokens`), or
- A single-token generation request (`max_tokens == 1`) that finishes immediately after the target pass.

On long-context AgentX / Claude Code prefill chunks, running the DSpark query pass + Markov chain sampling + draft acceptance/scatter on intermediate prefill chunks wasted GPU time on drafts that the scheduler immediately discarded.

## Changes
1. **`SchedulerOutput.num_spec_tokens_to_schedule`** (`vllm/v1/core/sched/output.py`, `vllm/v1/core/sched/scheduler.py`):
   - Computes `num_spec_tokens_to_schedule = 0 if all(self._drafts_unused(req, scheduled_tokens[i]) for i, req in enumerate(scheduled_reqs)) else self.num_spec_tokens`.
   - Preserves `sink_rejection` semantics: when `self.sink_rejection` is active and a request is on a decode step with `sink_draft_zero` or `num_in_flight_tokens > 0`, `_drafts_unused` returns `False` so `max_tokens=1` decode steps still draft when row 0 is a provisional sink-rejection token.
2. **DP coordination** (`vllm/v1/worker/gpu/dp_utils.py`):
   - Reduces `num_spec_tokens_to_schedule` via `MAX` across DP ranks alongside `num_tokens_after_padding` so dummy DP ranks run matching collective draft calls when any active DP rank drafts.
3. **`GPUModelRunner.sample_tokens` & Speculators** (`vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`, `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py`):
   - When `num_spec_tokens_to_schedule == 0`, skips `CopyDrafter`, `scatter_rows`, and `adaptive_verification.record_confidences`, while still running the context-only KV-cache prefill pass (`num_speculative_tokens == 0`) for `DFlashSpeculator` (which `DSparkSpeculator` subclasses) and `AutoRegressiveSpeculator` so draft model KV caches stay synchronized across chunked prefill steps.

## Patch Stack
- `vllm/0001-Spec-Decode-MRV2-Skip-drafting-in-steps-where-no-req.patch`
