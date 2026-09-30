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

`screen.sh` (2026-09-30, overlay SHA-256s checked on every node; pinned table
`dspark-costs-e9eb8edaf99252b3.json`, SHA-256 `5dc8961a…`, written by the first
r5n boot and reused by the other three):

Long-prompt repeatability (6,705 prompt tokens, five runs): deterministic MoE
1/5 without and 1/5 with the fence, zero first-token logprob difference. The
prompt did not expose the race in serving; 0005 rests on the SASS evidence and
the standalone reproduction, and costs no repeatability.

| prefill, 4,096-token chunk (three rounds) | r5n | overlay | change |
|---|---|---|---|
| 8K | 1031.8 ms (spread 0.12%) | 1032.2 ms (0.54%) | +0.05% |
| 64K | 1085.7 ms (0.29%) | 1090.3 ms (0.39%) | +0.42% |
| 131K | 1115.0 ms (0.43%) | 1117.8 ms (0.50%) | +0.25% |
| 200K | 1155.6 ms (0.95%) | 1154.7 ms (0.27%) | -0.07% |

| decode, alternating (tables.py) | r5n | overlay |
|---|---|---|
| prose step, one stream | 41.86, 41.99 ms | 41.97, 42.06 ms |
| JSON step, one stream | 48.26, 48.40 ms | 48.45, 48.25 ms |
| prose, eight streams (verified / accepted per draft) | 160.4, 160.9 tok/s (2.45, 2.41 / 1.23, 1.23) | 163.4, 161.2 tok/s (2.39, 2.43 / 1.20, 1.23) |
| JSON, eight streams (verified / accepted per draft) | 232.8, 235.2 tok/s (3.55, 3.50 / 2.90, 2.90) | 241.3, 241.3 tok/s (3.55, 3.52 / 2.91, 2.89) |

Prefill is within the round-to-round spread except 64K (+0.42% against 0.3-0.4%
spreads, below half a percent); step times move by at most 0.3%; eight-stream
throughput is level or higher with matched verification work.
