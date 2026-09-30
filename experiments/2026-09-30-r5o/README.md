# r5o candidate: fence prefill stage reads before the TMA refill

Date: 2026-09-30. Base: promoted r5n.

## Change

B12X patch 0005 (`patches/b12x/0005-fence-stage-reads-three-kernels.patch`,
from `experiments/2026-09-30-proxy-fence-audit`, tree `1a8b9401`) fences the
async proxy before the TMA-refilled stage releases of the BF16 prefill
projection (`gemm/bf16_gemv/_prefill.py`), the mHC TF32 and BF16 TMA prefill
projections (`norm/mhc/_kernels.py`) and the contiguous attention forward
(`attention/_shared/contiguous/forward.py`), as r5n's 0004 did for the dense
GEMM. The audit found their compiled releases with four or five shared loads
pending and reproduced wrong outputs standalone beside co-resident kernels:
mHC TF32 609/12000 (59 beside a plain copy), BF16 prefill up to 45/12000, 0 with
the fence; the attention fence is preventive. mHC runs in every layer's
prefill.

## Method

Lesson from r5n: throughput comparisons share one pinned adaptive-verification
cost table (`SPARK3_DSPARK_COST_DIR`, vLLM patch 0005) so verification policy is
fixed; its hash is recorded.

`screen.sh`, with the three files 0005 ships mounted over r5n (`overlay.sh`,
SHA-256 checked on every node):

1. Long-prompt repeatability on the experimental deterministic MoE without and
   with the fence (`determinism_long.py`: one ~6K-token prompt prefilled in
   4,096-token chunks, five runs). The short determinism probes never reach the
   prefill projections.
2. Prefill chunk timings at 8K-200K, three rounds on each arm; decode
   alternating r5n, overlay, r5n, overlay (prose and JSON answers, one and eight
   streams, six samples), both on the pinned table.

`build.sh` builds the image from the lock; `run.sh` boots it, requires the
shipped files to equal the measured ones on every node, runs `sass_gate.sh`
(the audit's scoreboard checker over the image's compiled BF16 prefill, mHC and
attention builds: no stage arrive with shared loads pending; on r5n it reports
5 pending arrives), then the quality gate, the needle check at 180K and
`doctor --live`.

Acceptance: prefill within the round-to-round spread at every depth; decode
step times and throughput within the alternating spread, read with the
verification counters; SASS gate clean; quality 5/5; needle pass; doctor clean.

## Results

Pending.
