# r5n candidate: fence dense GEMM stage reads before the TMA refill

Date: 2026-09-30. Base: promoted r5m.

## Change

B12X patch 0004 (`patches/b12x/0004-gemm-fence-stage-reads-before-tma-refill.patch`,
from `experiments/2026-09-29-determinism/0007-…`, tree `693aaed5`) fences the
async proxy before each dense GEMM mainloop stage release on the TMA load
path. Without it the MMA warps' last generic-proxy reads of a stage can observe
its TMA refill when another kernel's CTAs share the SM: DS4.1's shared-expert
down projection returned wrong columns in about 5% of decode calls beside the
routed MoE (`experiments/2026-09-29-determinism`, rounds 5-7). This is a
correctness fix for r5m's outputs; the deterministic MoE patches (0004-0006 of
that experiment) stay experimental and are not part of r5n.

Evidence so far: standalone, the serving-shaped linear beside the routed MoE
went from 20/12000 (6 rows) and 1/12000 (48 rows) wrong to 0/12000 each
(`gemm_race_stress.py`); with the deterministic MoE arm it made serving repeat
exactly with the side-stream overlap on; one-boot lean screen against r5m
showed no step-time change.

## Method

`screen.sh` measures the exact file 0004 ships (SHA-256 `ffc7c4a4…`) as an
overlay on r5m before building: prefill chunk timings at 8K-200K (two rounds)
on both, then decode alternating overlay, r5m, overlay, r5m (prose and JSON
answers at one and eight streams, six samples). `build.sh` builds the image
from the lock and loads it on every node; `run.sh` boots it, requires every
node's `dense_gemm.py` to be the measured file, and runs the quality gate, the
needle check at 180K and `doctor --live`.

Acceptance: overlay prefill within r5m's round-to-round spread at every depth;
decode step times and eight-stream throughput within the alternating spread;
quality 5/5, needle pass, doctor clean.

## Results

`screen.sh` (2026-09-30, overlay SHA-256 `ffc7c4a4…` on every node):

| | overlay (0004) | r5m | change |
|---|---|---|---|
| prefill, 4,096-token chunk at 8K / 64K / 131K / 200K | 1033 / 1085 / 1117 / 1158 ms | 1025 / 1085 / 1116 / 1153 ms | +0.73 / +0.02 / +0.13 / +0.46% |
| prose step, one stream | 42.23, 42.35 ms | 41.61, 42.52 ms | +0.5% |
| JSON step, one stream | 48.06, 48.31 ms | 48.19, 47.93 ms | +0.3% |
| prose, eight streams | 164.0, 165.0 tok/s | 168.9, 166.0 tok/s | -1.8% |
| JSON, eight streams | 241.0, 241.1 tok/s | 241.1, 241.3 tok/s | -0.1% |

Prefill (two rounds each) is within the overlay's own round-to-round spread
(about 1% at 8K and 200K). Decode alternated overlay, r5m, overlay, r5m, six
samples each; step times and eight-stream JSON are level.

The eight-stream prose gap is not a fence measurement. Each boot profiles
fresh adaptive-verification cost curves (`SPARK3_DSPARK_COST_DIR` unset), and
the boots verified different amounts of draft work:

| eight-stream prose | throughput | verified per draft | accepted per draft |
|---|---|---|---|
| overlay, first boot | 163.96 | 1.909 | 1.138 |
| r5m, first boot | 168.93 | 1.824 | 1.130 |
| overlay, second boot | 164.96 | 2.017 | 1.177 |
| r5m, second boot | 165.97 | 2.019 | 1.186 |

The first pair's -2.9% coincides with 4.7% more verified drafts per draft for
the same accepted drafts; the second pair, verifying the same amount, differs
by 0.6%. Across both pairs acceptance is 1.1575 against 1.158. How much of the
gap comes from startup timing, changed outputs or the fence is not
established by these receipts. Later comparisons share one pinned cost table
(`SPARK3_DSPARK_COST_DIR`, vLLM patch 0005) with its hash recorded.

Built image (`build.sh`, `run.sh`): `sha256:23b7b49d…` on all three nodes;
`dense_gemm.py` identical to the measured overlay everywhere; LRU 5/5; needle
3/3 at 152,914 tokens; `doctor --live` clean; display carve-out 842.5 MiB per
rank.
