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

`screen.json`: the screen, single-stream and sampled suites: static table at cost scale 1.5,
r6v (its own pin, scale 2.0), static table at scale 1.0, scale 1.5 again. Outputs are
unchanged by any draft budget (verification is exact).
