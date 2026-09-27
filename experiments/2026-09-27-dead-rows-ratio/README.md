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

Two ratio runs (13:48-14:05 local, six samples per point) against the three
pooled r5d controls of `experiments/2026-09-27-dead-rows` (nine samples),
same day and protocol. budget02 (fixed 0.2 cut inside the host budget) for
reference:

| Point | ratio | budget02 |
|---|---|---|
| prose c1 | +7.3% | +3.4% |
| code c1 | +1.4% | +4.0% |
| prose answer c1 | +9.3% | +10.1% |
| code answer c1 | +4.1% | +2.8% |
| prose c8 | -0.5% | +1.8% |
| code c8 | +3.1% | +0.9% |
| prose answer c8 | -0.6% | +0.5% |
| code answer c8 | -1.7% | +5.1% |
| mean | +2.8% | +3.6% |

Single-stream step time (ms): control 54.7 / 58.6 / 55.7 / 60.0 (prose, code,
prose answer, code answer); ratio 47.4 / 53.3 / 49.5 / 56.2; budget02
48.1 / 56.3 / 51.6 / 58.5.

- At one stream the cost rule shortens steps the most (4-7 ms) and gains
  +5.5% on average.
- At eight streams it is neutral: it cuts prose acceptance from 1.17 to
  0.93 accepted drafts per step and saves only enough time to offset that.
  The replay predicted +3.4%. The live-row price comes from a profile of
  random, uncorrelated tokens in graphs of different sizes. Real drafts share
  experts, and a dead row sits inside a fixed graph, so at eight streams a
  dead row saves less than priced and the rule cuts too much.
- Not a completion-stagger effect: aggregate throughput over the summed
  per-request rates (0.89-0.92) and the spread of per-request rates match
  the control.
- Quality gate LRU 5/5; dgx1 minimum MemAvailable 6.4 GiB.
