# Is DSpark's verification policy worth its machinery?

Base: `origin/main` `fc16764`, the promoted TP4 recipe `config/cluster-tp4.json`
(image `vllm-ds41f-kkref:04c30fa98e79-r6`). Every arm changes one value of that
recipe; image, weights and model arithmetic are unchanged.

## Question

Each decode step DSpark drafts up to five tokens per request, and three
mechanisms decide how many of them the target verifies:

1. **Host budget** (upstream adaptive verification, cost scale 2.0): chooses the
   number of verified rows from the previous step's confidences and per-shape
   step costs that every boot profiles.
2. **Device admission** (upstream): ranks draft slots by this step's
   confidences and admits the best ones up to that budget.
3. **Dead-row cut** (patch 0010, `SPARK3_DSPARK_DEAD_ROWS_TAU=0.2`): marks rows
   whose survival falls below 0.2 as padding, so routed MoE skips them.

The boot's cost profile takes 24 s of the 102 s TP4 boot (dgx1, 2026-10-07
18:45 UTC), and its result is discarded for the pinned table
(`SPARK3_DSPARK_COST_DIR`, patch 0005). Which of these mechanisms still pays on
r6 at the recipe's 1-16 streams?

## Review

The machinery is upstream vLLM's confidence-scheduled verification plus seven
local patches (0005, 0006, 0008, 0009, 0010, 0011, 0020: 1,002 added lines).
Production uses 0005, 0010 and 0020; the other four (655 lines, 65%) are
experiments that ship disabled:

| Patch | Lines | In production | Evidence |
| --- | ---: | --- | --- |
| 0005 pinned cost curves | 127 | yes | pinning did not remove boot-to-boot spread, which follows acceptance (`2026-09-26-dspark-policy`) |
| 0006 marginal rule | 55 | no | rejected: reasoning text 2-5% slower (`2026-09-26-dspark-policy`) |
| 0008 draft trace | 101 | no | offline replay tool |
| 0009 real-row profile | 75 | no | correct costs, not adopted: trimming by them loses (`2026-09-27-dspark-depth5`) |
| 0010 dead-row cut | 165 | yes (0.2) | +3.6% mean inside the host budget, each point within noise; scheduling every draft with the cut lost 5-14% at eight streams (`2026-09-27-dead-rows`, TP3) |
| 0011 dead rows by ratio | 424 | no | +5.5% at one stream, neutral at eight; mean +2.8% against +3.6% for the fixed cut (`2026-09-27-dead-rows-ratio`) |
| 0020 dead-row kernel warmup | 55 | yes | first request 568 ms to first token without it |

Problems in the promoted path:

- **The cache is read after the measurement.** The pinned table's key is
  configuration only (TP, draft depth, limits, graph sizes), yet every rank runs
  six rounds of dummy steps over 38 shapes before rank 0 looks the table up.
  On TP4 that is 24 s per boot, about 78% of it on the eight shapes above the
  graph limit (144 to 8,192 tokens), where every step carries a prefill and any
  plausible cost leaves the decision to verify every draft unchanged.
- **The profile measures the wrong rows.** Dummy rows are padding, which the
  routers skip, so the curves leave out routed MoE, the largest per-row cost
  (about 0.4 ms against 6.5 ms per real row in `2026-09-27-dspark-depth5`).
  Cost scale 2.0 stands in for the missing cost. Its exact value does not
  matter: 1.0 was neutral (`2026-09-24-kk-speed-tuning`). The per-boot
  precision feeds a heuristic.
- **The pinned key ignores the code.** A candidate with faster or slower
  kernels reuses a table measured on others unless its recipe names a new cost
  directory by hand.
- **Disabled experiments carry runtime surface.** Six module-level variables
  (`SPARK3_DSPARK_PROFILE_REPLAYS`, `_COST_DIR`, `_VERIFY_RULE`,
  `_DEAD_ROWS_TAU`, `_DEAD_ROWS`, `_RATE_ALPHA`) are read at import, five
  checks reject invalid values and combinations, class-level defaults and `getattr`
  fallbacks stand in for state that only some modes create, and 0042 reads
  0009's `dummy_input_ids`.

## Arms

One window, one boot per arm, the promoted recipe bracketed:

| Arm | Config | Change |
| --- | --- | --- |
| `base` | `config/cluster-tp4.json` | none |
| `all` | `verify-all.json` | `SPARK3_DSPARK_VERIFY_RULE=all`: no host budget; every draft is verified, the 0.2 cut still applies |
| `nocut` | `no-cut.json` | `SPARK3_DSPARK_DEAD_ROWS_TAU=0`: host budget only |
| `fixed` | `fixed.json` | `enable_adaptive_verification: false`: upstream fixed verification of all five drafts; no manager, no cost profile, no cut |
| `base-end` | `config/cluster-tp4.json` | drift check |

`fixed` runs last before the bracket, because it is the first boot of the
upstream fixed path on this stack. If it fails, the window restores production
and only the drift check is lost.

## Workload

`dspark-verification-tp4.json`: the lab's `lean` bench (prose and JSON at one
stream, 3 samples; 1K and 16K prefill) and `scripts/distinct_streams.py` at 1, 8
and 16 streams (3 samples, 256 tokens, distinct prompts, thinking off). All arms
are output-neutral by construction: they change only how many drafts the target
verifies.

```sh
scripts/lab.py run experiments/2026-10-07-dspark-verification/dspark-verification-tp4.json \
  --production-config config/cluster-tp4.json
```

## Decision rule

A mechanism stays only if removing it loses more than the `base`/`base-end`
drift at 1, 8 or 16 streams. Boot time comes from each arm's lab log.

- `all` ties or beats `base`: drop the host budget and its cost profile. Keep
  the cut, and stop profiling when the rule is `all`. 0005 goes away.
- `nocut` ties or beats `base`: drop the dead-row cut (0010, 0020).
- `fixed` ties or beats `base`: drop adaptive verification and all seven
  patches.
- Whatever survives: remove 0006, 0008, 0009 and 0011, and check the pinned
  table before profiling instead of after.

## Results

Window `dgx1-1791407569`, 2026-10-07 21:12-21:30 UTC, commit `54041be`, run
`dspark-verification-tp4`. Each arm was ready 111-113 s after its start
command, including the stop of the previous arm. Lowest MemAvailable was
20.2-22.1 GiB on every node, with no swap growth.

`fixed` did not boot. Upstream's `SpeculativeConfig` rejects the cost scale
without adaptive verification ("adaptive_verification_cost_scale requires DSpark
adaptive verification"), and the arm still set it. The runner then closed the
window and restored production, so `base-end` was not measured and there is no
drift check for this window. `fixed.json` now leaves the cost scale out.

Distinct prompts, aggregate tok/s, mean of 3 samples (range):

| Streams | `base` | `all` (no host budget) | `nocut` (no dead-row cut) |
| ---: | ---: | ---: | ---: |
| 1 | 86.6 (86.5-86.8) | 87.3 (87.1-87.4), +0.8% | 83.3 (82.9-83.6), -3.8% |
| 8 | 179.8 (178.9-180.6) | 157.9 (154.3-160.2), -12.2% | 174.6 (174.0-175.6), -2.9% |
| 16 | 220.2 (217.7-223.3) | 201.4 (200.8-201.8), -8.6% | 220.7 (218.5-223.2), +0.2% |

Bench at one stream (step ms, tokens per step, tok/s; 3 samples):

| Case | `base` | `all` | `nocut` |
| --- | --- | --- | --- |
| prose | 31.48, 2.064, 65.6 | 32.39, 2.170, 67.0 (+2.2%) | 33.68, 2.188, 65.0 (-0.9%) |
| JSON | 37.41, 4.000, 106.9 | 37.69, 4.062, 107.8 (+0.8%) | 38.87, 3.939, 101.3 (-5.2%) |

- **Host budget: keep.** Without it, 8 and 16 streams lose 12.2% and 8.6%,
  far outside the samples' ranges and the ~3% boot-to-boot spread recorded on
  this stack. One stream is level: a single request verifies nearly every
  draft anyway.
- **Dead-row cut: keep.** Without it, one-stream steps are 1.5-2.2 ms longer
  (+4-7%, against step-time intervals of 0.9% or less). JSON at one stream loses
  5.2%, distinct prompts lose 3.8% at one stream and 2.9% at eight, and 16
  streams are level. The margin is smaller than the budget's and has no
  drift check, but the direction holds at four of five points and on step time.
- **Fixed verification: not measured.** It verifies every draft like `all` and
  also drops the cut, which `nocut` shows pays. To beat `base` it would need the
  manager's per-step work to cost more than the 8-12% `all` lost. Not re-run.

Under the decision rule, adaptive verification, its pinned table (0005) and
the dead-row cut (0010, 0020) stay. 0006, 0008, 0009 and 0011 go, and so does
`SPARK3_DSPARK_VERIFY_RULE=all`, the arm switch 0010 carried, which lost here.
The pinned table is to be looked up before profiling.

## Candidate: the simplified series

`candidate-tp4.json` is the TP4 recipe on image
`vllm-ds41f-kkref:04c30fa98e79-r6-dspark-v2`, built from `vllm/series` through
`upstreams.lock.json` and `source.json`. The series is r6's without 0006, 0008,
0009 and 0011, with three patches rewritten:

- **0005** looks the pinned table up before profiling. Its key is configuration
  alone: TP size, draft count, request and token limits, capture limit,
  profile context length, and the shapes a profile would time, named as the
  profiled curves name them. The existing `ring4-r6-20261005` table therefore
  still matches. Rank 0 reads the table and broadcasts the curves, and every
  rank skips the timed rounds; the eager shapes past the capture limit run
  once, so their kernels compile before the first long prefill. Without a
  table the boot profiles and pins as before. `SPARK3_DSPARK_PROFILE_REPLAYS`
  is gone.
- **0010** keeps the dead-row cut and drops `SPARK3_DSPARK_VERIFY_RULE`. Its
  arguments travel as a `DeadRows` named tuple, the name 0020's warmup uses.
- **0042** treats every dummy drafter row as padding. It no longer reads
  0009's profile token ids, which would otherwise fail on every dummy step.

Against r6 the vLLM source loses 581 lines and gains 202. The tests lose 201
and gain 97. `adaptive_verification.py` reads 2 environment variables instead
of 7, and raises 2 errors instead of 7. Applying the series with the build's
`git am` flags gives patch head `709a159a` and tree `2c2f9d6d`. The same
procedure on this machine reproduces r6's recorded `cf7703fb` and `20c7c758`.
The model's arithmetic and the verification policy are r6's, so output is
unchanged; the boot skips the timed profile.

Expected: the boot about 20 s shorter, since 24 s of profile becomes one pass
over 8 eager shapes, and serving level with `base`.

Gate: build the image on an idle Spark, load it on all four nodes, run the
changed unit tests in it, then one window with `base` and `candidate`
bracketed on this experiment's workload, recording boot time. It is a
promotion candidate if serving is level and the boot is shorter.

### First build (`-r6-dspark`, tree `9c505063`)

2026-10-07, windows `dgx1-1791410569` and `dgx1-1791411479`, driven from dgx1 by a
node-local script. It opened the window, stopped the cluster with its containers kept,
built on dgx4, loaded the image on dgx1-3, ran the tests, and then ran
`scripts/lab.py run dspark-candidate-tp4.json`.

- The build took 7.5 min on dgx4 (22:03-22:10 UTC). The image was loaded on all
  four nodes with one ID, `sha256:a8d61d1c…`.
- The first test run exited with pytest's usage error. vLLM's
  `tests/conftest.py` imports `tblib`, which the serving image does not
  install. The window closed and production was restored. The second window
  ran the tests with `--noconftest`, as the TileLang kernel bundles do:
  `test_adaptive_verification.py` and `test_dead_rows.py` gave 29 passed and 1
  skipped in the candidate, and 32 passed and 1 skipped in r6, whose copies
  still test the removed modes.
- `base` booted in 100.8 s. `candidate` failed during engine initialisation on
  dgx1 with `'tuple' object has no attribute 'cut'`. The dead-row kernel warmup
  (0020) was written after 0011 and reads `dead_rows.cut` from 0011's
  `DeadRows` object; without 0011, 0010 returned a plain tuple. The other
  ranks then failed on the closed collective. The start rolled back to `base`,
  and the window closed with production serving.

The fix gives 0010 the `DeadRows` named tuple: the kernel wrapper reads the same
fields by name, 0020 is unchanged, and `test_dead_rows.py` passes a `DeadRows`.
An audit of the tree for every symbol that only 0006, 0008, 0009 or 0011
defined finds no other reference. The fixed series is image `-r6-dspark-v2`;
`-r6-dspark` is unused.
