# PR #980 rework, round 1: TP rank lockstep at temperature > 0, corrected

**Sub-agent verdict:** `corrected-meets-mark`. The corrected patch is `0764-perf-spec-decode-elide-tp-confidence-broadcast-and-lock-sampling-seeds.patch`, a git format-patch with the original author. All figures are in `result.json` and `numbers.json`.

## The correction

1. **The `adaptive_verification.py` hunk is dropped.**
   - Its `callable()` branch is dead code: the model runner passes the speculator's value as a bool.
   - Production already elided the broadcast in 1107: `DSparkSpeculator` has exposed `tp_invariant_confidence` since 0568.
   - Without the hunk, `record_confidences` is byte-identical to 1107's, so the PR's claimed broadcast saving of −17 µs a step does not apply to production.
2. **Unseeded-request seeds follow `--seed`.**
   - In 1107 the global NumPy RNG drew these seeds, seeded from `ModelConfig.seed`.
   - #980's `np.random.default_rng(0)` gave the same stream whatever `--seed` was.
   - Round 1 builds the generator from the configured seed: `model_runner.py` passes it to the sampler (3 lines), and `Sampler` and `SamplingStates` forward it.
   - With no seed configured it falls back to 0, vLLM's default, so the default `--seed` keeps #980's stream.
   - For a given seed the stream is new, not 1107's. That is within #980's declared change to sampled output.

## Evidence

**Lockstep.** The 1,000-step screen, with seed 1234 and the harness fixed to lay out draft rows as the worker does, found 0 rank mismatches in 1,000 steps in every run:

| Boot | TP3 | TP4 |
| --- | --- | --- |
| 1 (TP3 first) | 0 / 1000 | 0 / 1000 |
| 2 (TP4 first) | 0 / 1000 | 0 / 1000 |

- The screen covers every mode (greedy, sampled unseeded, the dflash estimator) across all 7 histories.
- The control arm diverges as designed: 264/1000 at TP3 and 240/1000 at TP4.

**Tests.** 190 passed and 1 skipped on both boots: the PR's tests plus two new ones.
- `test_unseeded_request_seeds_follow_configured_seed_across_tp_ranks`
- `test_sampler_hands_configured_seed_to_sampling_states`

**Greedy path.** A line trace of the final files finds no changed line in any greedy-path component. In-process step-cost A/B: +0.05 / +0.21 µs per step, within the A/A range between byte-identical code.

**Candidate-48's engine-grid offset (+0.27%) is not step work.** The grid's timed steps run none of #980's lines. #980 changes the tokens the startup sampler warm-up leaves in `last_sampled_tokens`, which the dummy steps hand to the MoE drafter. That gives each arm a fixed offset that repeats across boots, the pattern the attribution showed.

**Not part of the mark:**
- The TP2 crash in the screen is a harness bug: it passes 40 draft entries where `rejection_sample` reads 48. It reproduces on 1107, and serving cannot hit it.
- The first round 1, without the seed fix, is kept under `superseded/r1-noseed/`.

## Coordinator's verdict: meets the mark

#980 fixes TP rank divergence at temperature > 0: lockstep holds at TP3 and TP4 on both boots, and the PR's tests pass. No core metric is worse. The grid offset is a warm-up artifact, not added work.

The seed fix restores what 1107 had: unseeded requests take their seeds from `--seed`.

Round 1 joins the next candidate for fresh qualification. Any grid arm must mount all seven `overlay/` files.

The analysis is the rework sub-agent's; the coordinator wrote the verdict.
