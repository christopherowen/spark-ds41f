# Dead verification rows below a confidence cut

Base: the promoted configuration (`2026-09-27-karmic-kraken-r5c`, five
drafts) on candidate image `vllm-ds41f-kkref:04c30fa98e79-r5d`: the promoted
sources plus vLLM patch 0010 (tree `e9409518`).

## Mechanism

Patch 0010 (`SPARK3_DSPARK_DEAD_ROWS_TAU`): inside the draft combine, each
request's survival (running product of this step's draft confidences) is
computed and rows past the first draft below the threshold are marked as
padding; the routers give padding rows no experts and B12X skips them, so
dead rows read no expert weights. The sampled-token fix-up commits at most
one token past the last live row. `SPARK3_DSPARK_VERIFY_RULE=all` schedules
every draft so the on-device cut alone decides. Adaptive verification's host
budget, by contrast, is chosen from the previous step's confidences.

## Hypothesis

A survival cut at 0.3-0.5 recovers the prose loss of five drafts while
keeping their gain on code answers: an offline replay of the five-draft
trace (`experiments/2026-09-26-dspark-policy`) prices a 0.3 cut at +17% over
verifying three drafts with this step's confidences at one stream.

## Arms

| Arm | Change from the promoted configuration |
|---|---|
| `control` | r5d image only (dead rows off) |
| `dead03` | `SPARK3_DSPARK_VERIFY_RULE=all`, `SPARK3_DSPARK_DEAD_ROWS_TAU=0.3` |
| `dead05` | as `dead03` with 0.5 |
| `dead01` | as `dead03` with 0.1 |
| `budget01`, `budget02` | promoted host budget (ratio rule, cost scale 2) with the on-device cut at 0.1 or 0.2 inside it |

## Workload and gates

`sequence.sh`: control, dead03, dead05, control again in one session, each
with the LRU gate and the lean decode screen (reasoning and answer cases at
one and eight streams, three samples). Memory guards unchanged.

## Results

One session (2026-09-27, 11:50-12:40 UTC), lean screen, three samples per
point. The three `control` runs are pooled (nine samples); single control
runs differ from each other by up to 11% on single-stream prose points.
Throughput change against the pooled control:

| Point | dead03 | dead05 | dead01 | budget01 | budget02 |
|---|---|---|---|---|---|
| prose c1 | +3% | -4% | +11% | +11% | +3% |
| code c1 | +4% | -9% | +4% | +4% | +4% |
| prose answer c1 | +7% | -2% | +15% | +4% | +10% |
| code answer c1 | +7% | +1% | +6% | +1% | +3% |
| prose c8 | -10% | -14% | -11% | 0% | +2% |
| code c8 | +1% | -7% | -5% | +3% | +1% |
| prose answer c8 | -7% | -10% | -7% | +3% | +1% |
| code answer c8 | -1% | -6% | +1% | -1% | +5% |

Single-stream step time (ms) against the pooled control:

| Arm | prose | code | prose answer | code answer |
|---|---|---|---|---|
| control | 54.7 | 58.6 | 55.7 | 60.0 |
| dead03 | 46.6 | 52.7 | 50.3 | 56.8 |
| budget02 | 48.1 | 56.3 | 51.6 | 58.5 |

- Dead rows skip their routed experts as intended: with every draft
  scheduled and a 0.3 cut, single-stream steps are 3-8 ms shorter.
- A cut that schedules every draft loses 5-14% at eight streams. There the
  48 scheduled rows still pay attention, logits and sampling, and rows share
  experts, so a dead row saves less.
- Inside the promoted host budget, the cut no longer loses at eight streams.
  budget02 (0.2) is positive at every point, +3.6% on average; budget01 is
  +3.2%. Each point is within noise.
- τ 0.5 cuts accepted drafts faster than it saves time.
- The break-even survival is the price of a live row times the step's token
  rate, and both change with concurrency. A fixed τ cannot follow it; see
  `experiments/2026-09-27-dead-rows-ratio`.
- Quality gate LRU 5/5 in every arm; dgx1 minimum MemAvailable 6.1-6.5 GiB.
