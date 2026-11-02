# CopyDrafter: publish the current step's slot draft limits

## Symptom

With copy drafts enabled, the engine can die with a non-finite MoE gate:

```
Device assert failed: route_partials: gate logits contain nan or inf
  at vllm/models/deepseek_v4_1/tilelang/router.py:973 in router_partials
torch.AcceleratorError: CUDA error: device-side assert triggered
```

- `min_match=8`: deterministic in one boot on TP3 — `copy_probe.py --prompts 3`
  completes, then the single-stream suite dies in its warm-up. Reproduced twice.
- `min_match=16`: the synthetic probe and suite pass, and copy-dense prompts are
  token-identical to the c37 baseline; the engine dies under real agent traffic
  after a few minutes in the gateway pool (issue #852).

## Root cause

`CopyDrafter.sync_slot_draft_limits` publishes the host mirror of
`slot_draft_limits` only when the queued D2H copy has completed
(`self._limits_event.query()`). Under load the copy frequently has not completed,
so the host keeps the **previous proposal's** limits. Adaptive verification caps
the number of verified rows with that value. When the stale limit is wider than
the current proposal, it verifies rows the proposal does not have; those rows are
the scheduler's `-1` placeholders (async scheduling), and the model forward over
them produces non-finite MoE gate logits.

Instrumented on the m=8 repro: **1,209 stale-limit reads before the assert**, and
**zero** alloc-ring reuses. Changing the copy kernel's draft padding from `-1` to
`0` does **not** fix it, which places the bad rows in the scheduler's
placeholders, not in the copy drafts tensor.

## Fix

`repro/patches/0001-slot-draft-limits.patch` (a `git format-patch` commit on
candidate-37's patch head `5b7cfa37f`, tree `af0fb152`):

- `sync_slot_draft_limits` synchronizes the queued D2H event before copying the
  staging buffer. The copy is small and was queued before the drafter ran, so the
  wait is short; the host then always has the current step's limits.
- Adds a regression test: a not-yet-complete event must be waited on and the
  staged limits published. The old code returns the previous values and fails the
  test.

## Measurements (TP3, c37 image, 2200 MHz, PR #842 envelope 6155/59)

- m=8 deterministic repro (`copy_probe --prompts 3`, then single-stream):
  - before: engine assert during the suite, on two boots;
  - after: **suite exit 0, no asserts**.
- `copy_probe` tokens a step, m=8: copy **16.360**, edit **15.792**, tool
  **6.905**, prose 2.651. Before the fix: 16.360 / 15.792 / 6.636 / 2.651 — the
  tool category gains from not verifying rows the proposal does not have.
- Clean m=16 A/B (same image/profile/suite): suite **63.58** vs 64.683 tok/s
  unfixed (−1.7%, CI95 2.78% — level); probe copy 16.014, edit 15.315 unchanged,
  tool 6.083 → 6.232, prose 2.651. The fix is not a speedup.
- m=8 single-stream with the fix: 63.867 vs 64.897 baseline (−1.6%, CI95 2.77%).
  Copy-heavy categories unchanged; the blocking copy costs ~1–2% on this
  synthetic loop.
- m=16 in the live gateway pool (since 17:36Z): gateway traffic plus five
  30k-token real-agent-transcript requests (512-token completions), no asserts,
  engine healthy. The unfixed arm died within minutes on the same pool.
- m=8 in the live gateway pool: 64 minutes, 280 requests, 0 asserts, 0
  inference failures, 51.4% spec-decode acceptance.

## Reproduction

`repro/README.md` is the full record: base commit, image digest, profile,
boot/probe/suite commands, the run matrix, raw receipt logs, and the
instrumentation / `-1`→`0` control patches. Raw receipts are under
`repro/receipts/`; the profile is under `repro/profiles/`.

## Status

The host-sync patch is the validated diagnosis/fix used on our rings. The
upstream review (#855) holds the non-blocking D2H invariant, so the merge form
is the device-side clamp in `repro/patches/0004-device-clamp.patch` (same
candidate-37 base; it supersedes 0001 rather than stacking on it):
`CopyDrafter.propose` zeroes confidence past the device limit, and the combine's
dead-row gate writes the last sampled token for rows past the cut. The host sync
is reverted; the pinned `test_stale_wide_allocation_...` path stays non-blocking.

Validated on ring2 (candidate-37, TP3, direct,
`SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2`):

- `tests/v1/spec_decode/test_copy_drafts.py`: **22 passed**, including the
  pinned no-sync test.
- `tests/v1/core/test_copy_drafts_scheduler.py`: **11 passed**.
- m=8 copy-dense workload (8 concurrent, 10,456-token prompt, 256 completion):
  **159.90 output tok/s** vs **126.05** on the host-sync arm; no engine assert
  and no non-finite MoE gate in the head log.
- The device profile is `config/site-cand37-ring2-copy8-device.local.json`;
  the device files are mounted over `copy_drafts.py` and `tilelang_inputs.py`
  on all three ranks.

Open: the clamp covers the dead-row path, which is the production recipe. The
non-dead-row combine path still has no limit/padding mask, so the review's
"with or without dead rows" is only half-covered. The paired lab suite and the
NaN-producer diagnostic are not yet run. The host-sync ring remains the fallback
if the device arm regresses.

## Integration

The patch is against the vllm tree after the candidate-37 series
(`vllm/v1/worker/gpu/spec_decode/copy_drafts.py` and
`tests/v1/spec_decode/test_copy_drafts.py`). It is independent of #842's
defaulting change and of 0676 (#778 third-pass dead-row pruning); it fixes the
stale limit that verification reads, upstream of the gate that asserts.
