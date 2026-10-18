# DSpark draft budget from one static cost table

The richer option of the boot-timing removal (experiments/2026-10-18-dspark-linear-budget
tests one constant): keep the cost model's full shape (graph-padded buckets below the
capture limit, the per-row cost by size, the drafter's cost by request count) but read it
from one static table every build and boot shares, and sweep the cost scale the old model
applied to extra rows (2.0, tuned on earlier builds' noisy curves).

`static-r6v-25x3.json` is the median of three 25-replay profiles of r6v
(experiments/2026-10-18-dspark-profile-replays, `stability.json`), under r6v's pin key; its
48-to-96-row slope is 0.304 ms a row (r6v's own 5-replay pin: 0.289). For this experiment it
sits in the node's cache (`/cache/dspark-trace/static-r6v-25x3`), which the r6v image reads
as pinned curves, so the image runs unchanged; as a recipe feature the table would be part
of the recipe and the boot profile deleted.

## Runs

`screen.json`: the screen, single-stream and sampled suites: the static table at cost scale
2.0 (r6v's scale on r6v's shape, nothing timed), the linear rule at row cost 0.035 (the
reference), the static table at scale 1.5, scale 2.0 again. Outputs are unchanged by any
draft budget (verification is exact).

The sweep first planned scales 1.5 and 1.0 against r6v. The linear screen
(experiments/2026-10-18-dspark-linear-budget) showed that drafting more than r6v buys no
throughput: row cost 0.02 verified 8-12% more tokens a step at the same output rate, with
slower steps and later first tokens. So the question left is whether the cost model's full
shape (graph buckets, the drafter's cost by request count) beats one linear constant at the
same aggressiveness, which scale 2.0 against row cost 0.035 answers.

## Result, from earlier runs: the full shape adds nothing over one constant

The sweep was stopped for r6x's deployment (owner, 2026-10-19: promote without waiting)
after its first arm, the static table at scale 2.0, ran the screen and single-stream suites
(sampled was cut short). Its same-session reference never ran, so the comparison below is
against the linear and r6v arms of experiments/2026-10-18-dspark-linear-budget, an earlier
session the same day: indicative, not a verdict. The engine grid, which no draft budget
touches, moved +0.79% and -0.36% between these sessions, which bounds the resolution.

| Static table, scale 2.0, against | c16 tok/s | c16 tokens a step | c1 tok/s |
| --- | ---: | ---: | ---: |
| linear row cost 0.035 | +0.27% (same) | -1.63% | -0.34% (same) |
| r6v's timed curves (scale 2.0) | -0.28% (same) | -0.86% | -0.10% (same) |

Every budget measured that day sits within about 0.3% on throughput: linear 0.02 and 0.035,
r6v's boot-timed curves and the full static shape. Throughput is flat across this range of
budgets, so the cost model's shape (graph buckets, the drafter's cost by request count)
buys nothing a single constant does not; r6x keeps the constant. The owner chose this
answer from the runs already made over a 2.7-hour same-session run with production down
(2026-10-19); the spec stays in the lab queue's skipped/ folder should a formal verdict ever
be wanted.
