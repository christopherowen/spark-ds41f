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

`vllm/0001-Fix-CopyDrafter-stale-slot-draft-limits.patch`:

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
- Single-stream closed loop: baseline 64.897 tok/s, m=16 without the fix 64.683,
  m=8 with the fix 63.867 (CI95 2.77%). The blocking copy costs about 1.6% on this
  synthetic loop; the copy-heavy categories are unchanged.
- m=16 in the live gateway pool (since 17:36Z): gateway traffic plus five
  30k-token real-agent-transcript requests (512-token completions), no asserts,
  engine healthy. The unfixed arm died within minutes on the same pool.

## Integration

The patch is against the vllm tree after the candidate-37 series
(`vllm/v1/worker/gpu/spec_decode/copy_drafts.py` and
`tests/v1/spec_decode/test_copy_drafts.py`). It is independent of #842's
defaulting change and of 0676 (#778 third-pass dead-row pruning); it fixes the
stale limit that verification reads, upstream of the gate that asserts.
