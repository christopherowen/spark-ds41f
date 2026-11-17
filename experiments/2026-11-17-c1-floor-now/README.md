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
  alone, 93.7.

### Limits of the trace

- The experts a layer (17.2) are inferred from FC1's time with the record's dgx4 fit of 1022's
  kernel. The routed-expert kernels have changed since (FC2 now streams at 104% of FC1's
  per-byte rate; it was 92%), and the reference routes give 16.1 at 3.54 live rows. One expert a
  layer is 0.7 ms of floor, so the remainder is 4.2 ± 0.7 ms, and the launch fixed costs are the
  old fit's too.
- Three prose prompts, 308 steps, greedy, on one boot.
- Two of the window's steps are the prompts' prefill (20 and 24 rows); they raise the mean step
  by 0.25 ms.
