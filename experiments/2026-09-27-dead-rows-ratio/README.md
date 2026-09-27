# Dead verification rows by expected tokens per millisecond

Base: the promoted configuration (`2026-09-27-karmic-kraken-r5c`, five
drafts) with vLLM patches 0010 and 0011 (tree `7b839dc1`). The screen runs
the r5d image (`vllm-ds41f-kkref:04c30fa98e79-r5d`, 0001-0010) with 0011's
three runtime files mounted over its vLLM tree (`overlay.sh`); a promotion
builds the image. 0009 now also leaves `_dummy_run`'s decorators in place,
which only matters at startup.

## Mechanism

Patch 0011 (`SPARK3_DSPARK_DEAD_ROWS=ratio`) replaces the fixed survival cut
of `experiments/2026-09-27-dead-rows` with a cost rule decided on device:

- At startup, adaptive verification profiles the FULL graph shapes a second
  time on real (routed) rows. The host budget keeps the padded-row profile,
  which prices a scheduled row at what a dead row costs.
- Each step, the host stages the step cost of every live-draft count inside
  its budget: the draft, every scheduled row as padding, and the real-row
  increment of the live rows.
- The draft combine ranks the batch's admitted drafts by this step's
  survival and keeps live the count that maximizes expected tokens per
  millisecond. The remaining rows are dead, as with the fixed cut.

## Hypothesis

The fixed cut's break-even survival is the price of a live row times the
step's token rate, which changes with concurrency: about 0.3 at one stream,
where a live row reads about 5 ms of routed experts. The earlier arms show
this. At τ 0.3 with every draft scheduled, single-stream steps are 3-7 ms
shorter and code answers gain 8%, but eight streams lose 1-10%.

An offline replay of the five-draft trace prices each row with the startup
profiles: live rows at their real-row cost, dead rows at their padded-row
cost. It calibrates to the measured τ 0.3 arm: +7.5% modeled against +1-8%
measured at one stream, -5% against -10 to +2% at eight. It predicts the cost
rule at +7.8%, +4.0%, +5.5% and +3.4% at one, two, four and eight streams.

## Arms

| Arm | Change from the promoted configuration |
|---|---|
| `ratio` | `SPARK3_DSPARK_DEAD_ROWS=ratio` |

Controls: the three `control` runs of `experiments/2026-09-27-dead-rows`
(r5d with dead rows off, the same code path as 0011 with them off), from the
same day and protocol.

## Workload and gates

`sequence.sh`: ratio twice, each with the LRU gate and the lean decode
screen (reasoning and answer cases at one and eight streams, three samples).
Memory guards unchanged.

## Results

Pending.
