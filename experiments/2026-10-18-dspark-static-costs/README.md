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
