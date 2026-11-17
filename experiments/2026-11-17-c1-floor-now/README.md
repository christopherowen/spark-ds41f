# The single-stream step against its byte floor, today

Date: 2026-11-17. Base commit: `d4916ca97` (main). A recalculation: no boot, no kernel run.

## Question

How much of today's single-stream step (candidate-61 with the top-2 margin, tp4-ring) is weight
bytes at the attainable rate, and how much is left for step-time work?

## What was already measured

[../2026-10-26-c1-100tps](../2026-10-26-c1-100tps/README.md) measured the rate and built the
floor, for spark-ds41f-1022:

- GB10 streams expert weights at 266.6 GB/s at the margin (FC1's marginal rate at six rows on
  dgx4; FC2 265, a plain `torch.sum` 248 to 254, the packed LM head 240.7 in the traces; the
  specification is 273);
- a rank reads 5.71 GB a step, 21.4 ms at that rate, of a 30.55 ms step: 8.7 ms was not bytes.

Nothing here measures the rate again.

## One variable

The step the model is applied to: today's, in place of 1022's.

## Method

[floor_now.py](floor_now.py) imports that record's `model.py` and `floor.json` and takes the step,
its tokens and its verified drafts from the two boots of
[the margin's integrated screen](../2026-11-17-draft-calibration-sizing/README.md#integration)
(86.23 tok/s, 27.55 ms a step, 2.385 tokens a step, 2.54 drafts verified a step):

```
step = overhead + (dense + drafter_routed + 40 * N(live) * expert) / rate + dead * row
```

The live rows are the anchor and the verified drafts (the dead-row cut is on the device now);
the overhead is what is left. The block's other rows are counted as dead rows (0.37 ms each) or
as free, which brackets it.

```sh
python3 -I floor_now.py <c1.json> <c1.json> > receipts/floor-now.json
```

## Runs

[receipts/floor-now.json](receipts/floor-now.json).

| | spark-ds41f-1022 (the record) | Today |
| --- | ---: | ---: |
| tok/s | 74.3 | 86.23 |
| Step | 30.55 ms | 27.55 ms |
| Tokens a step | 2.271 | 2.385 |
| Live rows (anchor and verified drafts) | 2.89 | 3.54 |
| Distinct experts a layer | 13.8 | 16.1 |
| Bytes a rank reads | 5,708 MB | 6,128 MB |
| Byte floor at 266.6 GB/s | 21.41 ms (70% of the step) | 22.99 ms (83.5%) |
| Not bytes | 8.70 ms | 3.66 to 4.56 ms (13 to 17%) |
| A live row | 2.9 ms | 2.37 ms |

What the remaining overhead is worth, at today's tokens a step:

| Of the overhead removed | Step | tok/s | Tokens a step that 100 tok/s then needs |
| --- | ---: | ---: | ---: |
| none (today) | 27.55 ms | 86.2 | 2.77 (+16%) |
| a quarter | 26.4 to 26.6 ms | 89.2 to 89.9 | 2.65 to 2.67 |
| half | 25.3 to 25.7 ms | 92.4 to 94.0 | 2.54 to 2.58 (+6 to +8%) |
| two thirds | 24.5 to 25.1 ms | 94.6 to 96.9 | 2.46 to 2.52 |
| all of it | 23.0 to 23.9 ms | 99.4 to 103.3 | 2.31 to 2.40 |

## Conclusion

- **The step is 83% bytes.** More than half of 1022's 8.7 ms of overhead is gone; 3.7 to 4.6 ms
  is left.
- **Step-time work alone does not reach 100.** Every remaining millisecond that is not a byte
  would have to go for 99 to 103 tok/s. Half of it, the record's "credible" measure, is 92 to
  94.
- **The floor moved up, not down.** Today's step verifies more live rows (3.54 against 2.89), so
  it reads 0.42 GB more a rank. Tokens a step paid for that; the bytes a token cost barely moved
  (2,513 MB then, 2,569 MB now).
- **So 100 needs tokens a step or fewer bytes a token**: 2.54 to 2.58 tokens a step with half the
  overhead gone (+6 to +8%), 2.77 at today's step (+16%). The first is a more accurate drafter.
  The second is rows that reach fewer new experts, or fewer bytes read for the same rows.

## Limits

- 1022's byte inventory and routes: candidate-61 reads the drafter's LM head through an 8-bit
  screen for single-stream drafts, which the inventory still counts at 12 bits (up to 83 MB,
  0.3 ms, would move from the floor to the overhead).
- `N(r)` is from eight consecutive greedy tokens of the reference forward, interpolated at 3.54
  rows.
- The overhead is a remainder, not a trace: what it is made of today (collectives, launch gaps,
  streaming below the rate) needs a traced step of candidate-61, as `floor.py` read for 1022.
- Prose single stream on tp4-ring; agent traffic adds 0.3 to 0.5 ms of KV and index-key reads a
  step at 64-100K contexts (the record's estimate).

## A traced step of candidate-61

What the remainder is made of needs a trace, as the record's `floor.py` read for 1022.
[trace-tp4.json](trace-tp4.json) boots candidate-61's dev arm on tp4-ring under the torch
profiler ([prof-tp4.json](prof-tp4.json): every rank, no Python stacks or shapes) and runs
[profile_decode.py](profile_decode.py): three of servebench's prose prompts at 256 tokens each
(the single-stream screen's request: temperature 0, a fixed length, thinking off), once timed
from the server's counters and once between `/start_profile` and `/stop_profile`. The lab stops
the cluster after a profile job and summarizes every rank's kernels
([summarize_kernels.py](summarize_kernels.py), the indexer-decode-share experiment's).

```sh
bin/spark candidate queue tp4-ring experiments/2026-11-17-c1-floor-now/trace-tp4.json --apply
```

Queue entry `20261117T220821` at `245509ec7`, dgx1..dgx4, 22:08 to 22:12 UTC: ready in 114.1 s,
no new kernel files. The three prompts untraced: 314 steps, 28.99 ms a step (wall time over
steps, with the requests' own overheads); traced: 308 steps, 28.06 ms. The profiler without
stacks does not slow the step. Receipts: [receipts/trace](receipts/trace) (the workload's lines
and each rank's kernel summary); the traces (25 MB a rank) stay in
`results/private/determinism/c1-floor-now-trace-tp4` on dgx1.

[floor_trace.py](floor_trace.py) runs the record's `window` on them
([receipts/floor-trace.json](receipts/floor-trace.json)). Three names differ from 1022's build:
a forward's mark (`vocab_packed_embedding_kernel`), the drafter's head read through its 8-bit
screen (one byte a weight; 653 µs a step, which is 165 MB at 253 GB/s), and
`gemm_mxfp8_rope_mxfp4_kernel` as a stream. The four ranks agree to 0.05 ms on every line;
rank 0, a step in ms:

| | 1022 (the record, greedy window) | candidate-61, traced |
| --- | ---: | ---: |
| Step | 34.33 | 27.53 (the 305 six-row steps: 27.28) |
| Byte floor at 266.6 GB/s | 25.63 (19.9 experts a layer) | 23.35 (17.2 experts a layer) |
| Streaming below that rate | 5.69 | 2.37 |
| of it: routed experts' launch fixed costs | 1.24 | 1.24 |
| of it: the target's LM head (231.7 GB/s) | 0.20 | 0.14 |
| of it: the rest (prefetched GEMMs and their consumers, small streams) | 4.25 | 0.99 |
| DRAM idle under collectives | 0.95 | 1.03 |
| DRAM idle under other small kernels | 0.96 | 0.57 |
| DRAM idle, no kernel (launch gaps, host) | 0.70 | 0.13 |
| DRAM idle under attention, indexer, cache | 0.40 | 0.08 |
| **Not bytes, total** | **8.70** | **4.18 (15.2% of the step)** |

Kernel time a step by family (kernels overlap across streams): routed experts 12.9, dense
projections 11.1, the L2 prefetch's side stream 8.9, collectives 5.9, mHC 3.5, attention and
indexer 3.3, LM heads and screens 1.7, GLP steer 1.4.

### What the trace says

- **The remainder is 4.2 ms, as the model gave** (3.7 to 4.6): 2.4 ms of streaming below the
  attainable rate and 1.8 ms with DRAM idle.
- **Two items are as they were at 1022.** The routed experts' launch fixed costs (1.24 ms, 86
  launches a step) and the DRAM-idle time under collectives (1.03 ms). Together they are 2.3 ms,
  over half of what is left: +4.7% and +3.9% of throughput if each went to zero.
- **Everything else was cut by two thirds or more**: the prefetched GEMMs' excess 4.25 to 0.99
  ms, small kernels 0.96 to 0.57, launch gaps 0.70 to 0.13, attention 0.40 to 0.08.
- **All of it gone is 101.7 tok/s** at today's tokens a step (+17.9%); the two unchanged items
  alone, 94.0.

### Limits of the trace

- The experts a layer (17.2) are inferred from FC1's time with the record's dgx4 fit of 1022's
  kernel. The routed-expert kernels have changed since (FC2 now streams at 104% of FC1's
  per-byte rate; it was 92%), and the reference routes give 16.1 at 3.54 live rows. One expert a
  layer is 0.7 ms of floor, so the remainder is 4.2 ± 0.7 ms, and the launch fixed costs are the
  old fit's too.
- Three prose prompts, 308 steps, greedy, on one boot.
- Two of the window's steps are the prompts' prefill (20 and 24 rows); they raise the mean step
  by 0.25 ms.

## The expert-rate bundle on today's kernels

The trace's floor and its launch costs used the record's fit of 1022's kernels.
[bundles/expert-rate](bundles/expert-rate) is the record's bundle on candidate-61's plan (the
dispatch and kernels of a call from the plan at its tile height; a layer's three launches also
timed together), on candidate-60's image with candidate-61's overlay.

- sp8, twice, from a copy staged under `~/spark-lab`: pass in 27 s each
  ([runs/expert-rate-sp8-1.txt](runs/expert-rate-sp8-1.txt), `-2`);
- the ring's four nodes, the cluster stopped ([expert-rate-tp4.json](expert-rate-tp4.json), queue
  entry `20261117T225516` at `89297cb31`): pass on each in 24 to 25 s, verdicts agree
  ([runs/expert-rate-dgx1.txt](runs/expert-rate-dgx1.txt) to `-dgx4`);
- the record's own bundle on its own image (`spark-ds41f:19f2c20ed4d6-next-6`), unchanged, on
  dgx1 and dgx4 ([expert-rate-1025-tp4.json](expert-rate-1025-tp4.json), queue entry
  `20261117T225921`; [runs/expert-rate-1025-image-dgx4.txt](runs/expert-rate-1025-image-dgx4.txt)).

Six rows, time a launch = launch cost + time an expert × experts (µs), and a plain `torch.sum`
read of 110 and 220 MiB:

| Run | FC1 | FC1 marginal | FC2 | FC2 marginal | Plain read |
| --- | --- | ---: | --- | ---: | ---: |
| October: 1022's kernels, dgx4 | 14.2 + 11.41 n | 266.6 GB/s | 14.6 + 5.75 n | 265.2 | 253.7, 253.4 GB/s |
| October's bundle and image, dgx4 today | 13.5 + 12.50 n | 243.4 | 16.8 + 5.79 n | 263.4 | 229.4, 235.8 |
| October's bundle and image, dgx1 today | 12.3 + 12.49 n | 243.6 | 15.0 + 5.85 n | 260.9 | 236.7, 235.8 |
| Today's kernels, dgx1 | 10.8 + 12.28 n | 247.7 | 7.9 + 6.03 n | 253.2 | 237.3, 235.6 |
| Today's kernels, dgx2 | 10.9 + 12.16 n | 250.3 | 8.2 + 5.92 n | 257.7 | 235.2, 236.7 |
| Today's kernels, dgx3 | 10.7 + 12.19 n | 249.6 | 8.4 + 5.93 n | 257.4 | 237.7, 236.3 |
| Today's kernels, dgx4 | 9.2 + 12.29 n | 247.6 | 7.9 + 5.95 n | 256.6 | 237.4, 235.1 |
| Today's kernels, sp8 (two runs) | 10.3 + 11.71 n, 2.6 + 12.35 n | 259.7, 246.4 | 8.5 + 5.78 n, 6.4 + 5.96 n | 263.9, 255.9 | 246.4, 245.8 |

The traced step again, each rank with its own node's fit
([receipts/floor-trace-today-fit.json](receipts/floor-trace-today-fit.json)); rank 0, ms:

| | With the record's fit | With today's fit |
| --- | ---: | ---: |
| Attainable rate | 266.6 GB/s | 247.7 GB/s |
| Experts a layer | 17.2 | 16.2 (the reference routes: 16.1) |
| Byte floor | 23.35 | 24.42 (89% of the step) |
| Streaming below the rate | 2.37 | 1.29 |
| of it: routed experts' launch costs | 1.24 | 0.80 |
| DRAM idle | 1.81 | 1.81 |
| Not bytes, total | 4.18 | 3.11 |

### What it says

- **The ring streams 7 to 9% slower than dgx4 did in October, with nothing of ours changed.**
  The record's bundle on the record's image gives 243 GB/s for FC1 where it gave 266.6, and a
  plain read 230 to 236 where it gave 248 to 254. dgx1 reads the same as dgx4. FC2's marginal
  rate did not move (263 against 265).
- **So today's floor is 24.4 ms, not 23.4**, and the step is at 89% of it. What is not bytes is
  3.1 ms: 1.8 ms with DRAM idle (1.03 under collectives), 0.8 ms of launch costs, 0.5 ms of other
  streaming below the rate.
- **The launch costs came down.** 19 µs a layer for the two GEMMs against 29 at 1022 (10.4 and
  8.1 against 14.2 and 14.6): 0.80 ms a step, +3.0% if it went to zero. Timed together, a
  layer's three launches cost 9 to 15 µs over their bytes.
- **The rate itself is now the largest item.** At October's rate the same bytes would take 22.7
  ms: 1.6 ms less, +6.2% (about 91.6 tok/s), more than the launch costs and the collectives'
  idle time are worth separately.

### What changed on the nodes is not established

Read from dgx4, nothing changed: the same kernel booted on 2026-10-25 and today
(`7.0.0-1019-nvidia-64k`, 64 KiB pages), the same driver (580.178.04, installed 2026-10-16), the
`performance` governor, THP `never`. Two differences are known:

- October's run started 17 minutes after a boot (18:45, run at 19:02); today the nodes have been
  up ten days, through every boot of the serving container since 2026-11-07. Free memory is
  spread over small orders (`/proc/buddyinfo`: nothing free at the two largest orders).
- the host setup of the 64 KiB profile (docs/setup.md) and the DGX OS release may have changed
  between the two dates; this record did not check their history.

sp8 (the 4 KiB kernel `7.0.0-1024-nvidia`) reads 246 GB/s and its FC1 rate differed by 5%
between two runs a minute apart, so the rate is not a constant of the hardware.

### Limits

- One run a ring node; the nodes agree within 1% on every rate.
- The October numbers are that record's receipts, not remeasured under October's conditions.
- A cause is not shown. A reboot of one node with the bundle before and after would say whether
  the time since boot is it.
