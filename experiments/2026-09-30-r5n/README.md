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

Pending.
