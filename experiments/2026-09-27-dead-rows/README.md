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

Pending.
