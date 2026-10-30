# Seeded Sampled Batch-Invariant Adaptive Verification

Fixes the `penalties-sampled/c8` seeded concurrent sampling variation across boots
(`fb36e3be` vs `44f3b309` / `9df5dee2` in `../2026-10-26-spark-ds41f-candidate/README.md`)
on `spark-ds41f:1030` (vLLM `seeded-adaptive-verification`, head `a2b579274` over `d5df88d83`).

## Root Cause

Under block verification (`verify_kernel` + `resample_kernel` in `tilelang_rejection.py`),
greedy (`temperature == 0`) verification is invariant to how a request's token positions split
into verified draft blocks (`capacities[b]`), whereas seeded multinomial rejection sampling
(`seed is not None and temperature > 0`) depends on the verified block length
`min(capacities[b], dead_row_cut[b])`.

While `dead_row_cut[b]` is computed strictly per request on GPU from `self._confidence_probs[req]`
and `dead_rows_tau`, `AdaptiveVerificationManager` previously:
1. Pooled `survival_probability` across all active requests in the step (`1.0 + row_cost * rows`)
   in `get_num_tokens`, so co-scheduled high-confidence requests inflated `expected_tokens` and
   raised the marginal utility threshold `(row_cost / (1 + row_cost * rows)) * expected_tokens`
   required to admit a request's lower-confidence draft slots.
2. Ranked all draft slots globally across the batch on GPU (`assign_draft_budget` in
   `reallocate_drafts`), allowing one request's higher-confidence draft slots to displace
   another request's draft slots.
3. Shared a single batch-wide step counter (`_steps_without_drafts`) and global 2-slot ring
   for stale confidences (`_stale_idx`), which advances on steps where a request is paused or
   co-scheduled with a mid-prefill request.

## Fix (`0568`)

In `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` and `vllm/v1/worker/gpu/model_runner.py`:
- `add_request(req_idx, sampling_params)` records whether the request uses seeded sampling
  (`seed is not None and temperature > 0.0`), initializes its per-request idle exploration
  counter (`_req_steps_without_drafts[req_idx]`), and initializes its per-request 1-step-stale
  confidence snapshot (`_req_stale_confidences[req_idx] = 1.0`).
- `record_confidences` advances a request's CPU stale confidence snapshot only when that request
  itself completes a step (`num_computed_tokens + num_scheduled_tokens >= prefill_len`), and
  keeps mid-prefill requests in `_pending_resets` at confidence `1.0` on both CPU and GPU.
- `get_num_tokens` budgets each seeded sampled request independently using its single-request
  utility (`_budget_for_scores` with `num_sampling_requests=1` and its own `num_non_draft_tokens`),
  matching `alone` (`c1`) bit-for-bit, while unpinned (greedy or unseeded) requests continue to
  pool across the batch.
- `reallocate_drafts` preserves each seeded sampled request's exact per-request draft capacity
  on GPU (skipping `assign_draft_budget` when `draft_budget == scheduled_drafts.sum()`, and
  pinning seeded sampled slots via sentinel row `max_num_reqs` (`+inf`) when mixed with trimmed
  unpinned requests).

## Verification on `dgx4`

 Reproducible from the vLLM worktree:

```bash
python3 experiments/2026-10-30-seeded-adaptive-verification/make_bundles.py VLLM_WORKTREE
```

Ran `bundles/seeded-adaptive` on `dgx4` (`spark-ds41f:1030`, `sha256:7918fd6145b5`):
- **1,362 passed, 2 skipped in 26.88s** ([verdict.json](verdict.json)), including:
  - `test_adaptive_verification.py`: `test_seeded_sampled_budget_is_batch_invariant`,
    `test_seeded_sampled_reallocate_drafts_and_mid_prefill_reset`, and
    `test_seeded_sampled_multi_step_schedule_and_batch_invariance` (verifying 8 seeded sampled
    requests across solo `alone`, concurrent `c8`, chunked-prefill, and paused-step schedules).
  - `test_tilelang_rejection.py`: block verification, dead-row cut, and rejection sampling.
  - `test_prepare_kernels.py`: startup `tilelang_calls` remains 247.
