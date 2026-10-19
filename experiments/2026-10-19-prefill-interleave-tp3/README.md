# TP3 prefill interleaving with the CED skip (prefill follow-ups, item 6)

While a long prompt prefills, every engine step carries one of its chunks and the other
streams wait for it: with the skip at 4,096 their step is 0.86 s (median) instead of
0.10 s, about 30 s for a 131K prompt (experiments/2026-10-19-prefill-budget-tp3).
`--prefill-schedule-interval N` schedules that prompt's in-progress chunk only every N-th
step while other requests decode, so the steps between are decode-only. It counts steps,
not seconds (no timing in the decision; `--prefill-compute-share` meters measured step
time, so it is not tried). Chunk boundaries do not move, so aligned chunks and the skip's
successor check are unaffected, and a prefill with nobody decoding runs every step.
Expected: the other streams' median gap falls toward (step + (N-1) x 0.1 s) / N; the worst
wait does not change (one step still carries a chunk); the prompt's prefill under load
slows by about (N-1) x 0.1 s a chunk.

Arms, the gates' skip overlays on production r6x TP3 at 1M: intervals 1, 2 and 3 at the
budgets 4,096 and 8,192, then 4,096 at interval 1 again (the repeat).
- `interleave.json`: `suite.json`, the stall of seven decoding streams during a cold 131K
  prefill and that prefill's rate under them, ten repeats per arm (three left the worst
  wait and p99 step unresolved in the budget sweep).
- `determinism.json`: outputs at intervals 2 (4,096) and 3 (8,192), 700- and 2,200-line
  long prompts, must equal the skip's (62313cb5 / 9742ac2c and 62313cb5 / f6171075):
  interleaving moves decode rows between steps, and the mixed scenarios put them beside
  the long prompt's chunks.

No promotion or production change.

## Determinism (2026-10-19 14:15-14:26 UTC)

Interval 2 at 4,096 reproduces the skip's digests at both long prompts (62313cb5 /
9742ac2c and 62313cb5 / f6171075), with zero token and log-probability differences in
every scenario: interleaving moves decode rows between steps without changing a bit.

Interval 3 at 8,192 did not boot: dgx1's startup guard stopped it at 4.78 GiB MemAvailable
(5 GiB floor). The same budget booted at 13:01 (budget sweep), so 8,192's startup margin on
TP3 is thin enough to fail some boots. The chunking check moves to 6,144 (`chunking.json`,
`skip-6144-i1-tp3.json`): long prompts split at 6,096 instead of 4,048 and 8,096, so equal
digests show outputs do not depend on where a prompt splits (the encoder side of item 6's
load-dependent chunks).

## Stall results (2026-10-19 14:26-15:12 UTC, ten repeats each)

What seven decoding streams wait while a cold 131K prompt prefills, with the skip:

| Arm | Mean gap | p50 | p99 (sd) | Worst wait, mean / max (sd) | 131K prefill under them |
| --- | ---: | ---: | --- | --- | ---: |
| 4,096, interval 1 | 0.746 s | 0.858 s | 1.06 s (0.30) | 1.42 / 1.80 s (0.44) | 4,520 tok/s |
| 4,096, interval 2 | 0.461 s | 0.122 s | 0.93 s (0.02) | 1.01 / 1.04 s (0.04) | 4,007 tok/s |
| 4,096, interval 3 | 0.352 s | 0.115 s | 0.92 s (0.01) | 1.01 / 1.04 s (0.04) | 3,599 tok/s |
| 8,192, interval 1 | 1.076 s | 1.441 s | 1.72 s (0.44) | 2.26 / 3.06 s (0.73) | 5,278 tok/s |

The decode step without a prefill is 0.099 s throughout. Interleaving cuts the mean gap
38% (interval 2) and 53% (interval 3) and takes the worst wait from 1.0-1.8 s to a steady
1.0 s (one prefill step), for 11% and 20% slower prefill while others decode; a prefill
with nobody decoding is not throttled. Interval 2 is bit-exact (above).

**Larger budgets fail dgx1's startup guard.** 8,192 at intervals 2 and 3 and the 6,144
chunking check (`chunking.json`) did not boot: the startup guard stopped them at 5.01-5.24
GiB MemAvailable against its 5 GiB floor (journalctl -t spark-memguard, 15:10-15:15 UTC).
Over the day, boots above 4,096 passed three times (6,144 once, 8,192 twice) and failed
three times; every boot at 4,096 passed. Serving memory is not the limit (the budget
sweep: dgx1 at least 4.57 GiB through 1M at 8,192); the startup peak is. A larger TP3
budget, and load-dependent chunks that need one, wait on lowering that peak. The chunking
check and the 8,192 interleaving arms are unmeasured.

**Recommendation.** TP3 keeps 4,096 with the skip (+11% prefill, outputs identical, the
other streams' median step 10% shorter). No promotion or production change.

**Interleaving is rejected** (owner, 2026-10-19): a prefill should pause only when other
work needs the step, not on a fixed cadence. The decoding streams advance in every chunk
step anyway, so holding a chunk back only speeds them at the long prompt's expense. The
interval stays at 1. The way to give the other streams shorter steps without holding the
prompt back is load-dependent chunks (full chunks when nobody else decodes), which wait on
a larger budget and so on dgx1's startup peak.
