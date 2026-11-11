# candidate-54: the `0799` gate, 2026-11-11

The gate declared in [candidate-54's README](../../README.md) for `0799`, the MoE combine's shared-row load after the
PDL wait. Specs: [gate-0799-tp4.json](../../gate-0799-tp4.json) (12 boots at `tp4-ring`),
[gate-0799-tp4-switched.json](../../gate-0799-tp4-switched.json) and
[gate-0799-tp3-switched.json](../../gate-0799-tp3-switched.json) (4 boots each). Each boot runs the recorded
`determinism` and `greedy` scenarios twice through
[token_determinism_record.py](../../../candidate-58/token_determinism_record.py), then the gate's distinct-stream
checks.

**Image:** `spark-ds41f:candidate-54`, image ID `1e556e80`. This is `0799`'s first build: tree `b20b199d`, built on
dgx4 at 08:28Z and copied to sp1 and sp5. Its serving code is the rebuilt image's (tree `ad2dc11f`). The two trees
differ only in `tests/kernels/moe/test_deepseek_v41_tilelang_moe.py`, which `0799` updates after this build failed
that one release test. The TP3 profile ran without the prefetch caps, which were added after this gate
([gate-caps-tp3-switched.json](../../gate-caps-tp3-switched.json) gates them).

## Result: every run of every boot gives the recorded hashes

| Testbed | Boots | Runs identical | Hashes (logprobs; text) | Window |
| --- | --- | --- | --- | --- |
| `tp4-ring` (dgx1..dgx4) | 12 | 48 of 48 | 18d61961/d6568649; 0a2dacdd/12fd8257 | 08:28-11:03Z |
| `tp4-switched` (sp1..sp4) | 4 | 16 of 16 | 18d61961/d6568649; 0a2dacdd/12fd8257 | 10:24-11:18Z |
| `tp3-switched` (sp5..sp7) | 4 | 16 of 16 | 5c22ce41/bce6092e; 26a739a9/12cfee9b | 10:30-11:32Z |

Every scenario (`c8`, `staggered`, `mixed`, `cached`, `again`, `mixed15`) has 0 token and 0 logprob differences in
every run. The hashes are candidate-54's recorded ones: #1030's TP4 hashes, and the TP3 hashes first recorded by
candidate-54's earlier gate.

**Against the record at `tp4-ring`:** before `0799`, 4 of 12 boots failed (3 of 8 in candidate-58's row trace, 1 of 4
in its gate repeat). After it, 0 of 12 did.
- If the failure rate were unchanged at 1 boot in 3, 12 passing boots would have probability (2/3)^12 = 0.8%.
- Fisher's exact test on the 24 boots, one-sided: p = 0.047.
- `tp4-switched` and `tp3-switched` had not failed before (0 of 8 and 0 of 7), so their boots show only that `0799`
  breaks nothing there.

## Distinct-stream checks

Mean tok/s over the boots (range), against candidate-54's earlier gate (one boot, image `e1de71be`, another
session):

| Testbed | c16 | c8 | c4 |
| --- | --- | --- | --- |
| `tp4-ring` | 270.60 (269.92-271.45), earlier 270.62 | 200.81 (199.84-201.29), earlier 200.19 | 155.07 (154.19-156.61), earlier 155.84 |
| `tp4-switched` | 266.13 (265.36-266.65), earlier 266.19 | 196.72 (196.25-197.19), earlier 197.09 | **151.59 (150.32-152.50), earlier 153.22** |
| `tp3-switched` | | 153.35 (152.98-153.94), earlier 154.00 | 116.30 (115.59-117.21), earlier 116.73 |

- c16 and c8 are the same on every testbed.
- **c4 at `tp4-switched` is 1.06% lower, and all four boots are below the earlier one.** At `tp4-ring` the earlier
  c4 falls inside the boots' range (-0.50%). This is a cross-session comparison against a single boot, from a gate
  check, not a screen. The screens on the rebuilt image measure the 4-request grid shapes.

## Files

Per testbed, `<testbed>/`:
- `determinism/`: each run's summary (`*.jsonl`) and telemetry, and each distinct-stream run.
- `lab/`: the window log's part for the gate job, the gate's table, the overlay list (none), and
  `gate-0799-<testbed>-records.sha256`.
- The record files (`record-*.json`: each run's reference outputs, 15 MB at `tp4-ring`) stay on the head under
  `results/private/determinism/candidate-54-gate-0799-<testbed>/`. Their SHA-256s are in that file. No run differed,
  so they hold no differing request.

## Corroboration from before the fix (candidate-58, 2026-11-10)

**c1 at `tp4-switched`** ([c1-drafter-arms-tp4-switched.json](../../../candidate-58/c1-drafter-arms-tp4-switched.json),
7 arms of 3 boots, each request's per-step token counts compared between boots by
[c1_decisions.py](../../../candidate-58/boot-differences/c1_decisions.py)):

| Arm | Boots that agree | Accepted per draft |
| --- | --- | --- |
| shared expert on the main stream (`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`) | 3 of 3 | 1.303 |
| drafter query forward eager | 3 of 3 | 1.301-1.302 |
| row taps on | 3 of 3 | 1.353 |
| late admission off | boots 1 and 2; boot 3 differs from request 34 | 1.302 |
| drafter context preparation eager | boots 2 and 3; boot 1 differs from request 12 | 1.301-1.302 |
| device sync around the drafter | none | 1.301-1.302 |
| L2 prefetch off | none | 1.317-1.320 |

- The drafter's layers are `DeepseekV4DecoderLayer`s, so its MoE runs the same combine, with the shared expert on
  the aux stream, inside its captured query graph.
- The two arms that remove `0799`'s hazard, one with no aux stream and one with no graph replay, kept every boot's
  decisions the same. The arms that keep both did not. Row taps also agreed; their copies may serialize the streams.
- Acceptance is the same with and without the hazard (1.301-1.303), so the race shifted threshold decisions, not
  draft quality.
- [c1-decisions-tp4-switched.json](../../c1-decisions-tp4-switched.json) repeats this on the rebuilt image.

**The trigger arms at `tp4-ring`** (candidate-58's `gate-arms`: control, admission20, shared-stream-off,
admission-off) passed on boot 1 each: 16 of 16 runs on the recorded hashes. That job stopped at the l2-off arm's
first boot on startup memory. At the pre-fix rate, 4 passing boots tell nothing either way.
