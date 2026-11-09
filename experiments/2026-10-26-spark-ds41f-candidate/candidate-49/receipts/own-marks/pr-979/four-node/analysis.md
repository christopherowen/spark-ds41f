# PR #979 own mark, four nodes (sp1..sp4)

**Sub-agent verdict:** `pass` on the mark's T≥6144 clause. All figures are in `result.json` and `analysis.json`.

## Setup
- **Ranks:** four, one per node, over the switch, on `spark-ds41f:1107` (sha256:3f32a962...). No build.
- **Candidate:** candidate-49 (0762..0767), bind-mounted.
- **Baseline:** candidate-49 without #979 (0762..0765 and 0767). It differs only in `sp_prefill.py`.
- **Order:** same session, ABBA per topology, two boots per arm, with ≥240 ms of GPU warm-up.

The PR ships no bench scripts. `scripts/bench_979_integrated.py` keeps its workload but calls each tree's own `project_exchange_rows` and `gather_mxfp8`, the way the model calls them. The PR's original scripts, found on the nodes, are in `pr-bench-scripts/`.

## `project_exchange_rows` at T≥6144: faster beyond the interval, bit-exact

Slowest rank's median per iteration, boot 1 / boot 2:

| Topology | T | candidate-49 (µs) | baseline (µs) | Change |
| --- | --- | --- | --- | --- |
| ring4 | 6144 | 9637 / 9883 | 11312 / 11168 | −13.2% |
| ring4 | 8192 | 12682 / 12603 | 15344 / 15447 | −17.9% |
| direct | 6144 | 8931 / 8925 | 10443 / 11124 | −17.2% |
| direct | 8192 | 11537 / 11634 | 14433 / 13843 | −18.1% |

Every rank's output digest matches across all four runs.

## Slower cells of its own bench, on shapes serving uses
- **`gather_mxfp8` on direct:** where it takes #979's new one-shot RoCE route (T≤2048), it is slower than the baseline at every T, by 9% to 50% beyond the interval. At T=2048 it reads 1610/1723 µs against 1085/1309 µs.
- **`project_exchange_rows` below T=6144:**
  - ring4 T=160: +250 µs (slowest rank);
  - ring4 T=2048 and T=4096: +73 / +106 µs;
  - direct T=4096: +4.5% (rank 0 only).
- **Faster, outside the mark:** `gather_mxfp8` on ring4 at T=160 and T=256 (−33%, −24%).
- **Noise check:** two boots per arm gives a narrow interval. At T≥4096 the gather runs identical PyNCCL code in both arms, yet direct rank 0 reads +12–13%. Single small cells can therefore be noise; the 1.5–2.8 ms gains at T≥6144 are not.

## Coordinator's verdict: misses; sent to rework

The gain at T≥6144 is real. Under release-benchmark.md step 8, though, a slower cell of the PR's own bench on a shape serving uses counts as worse. `gather_mxfp8` on direct, the mode of the switched shapes, is slower at every T where the new route is taken.

#979 goes to rework round 1 (`../../rework/pr-979-r1/`): keep the T≥6144 exchange gain, and use 1107's route wherever #979's route is not faster.

The analysis is the own-mark sub-agent's; the coordinator wrote the verdict.
