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
timed together), on candidate-60's image with candidate-61's overlay (the bundle as of
`62a1873f9`; from 2026-11-18 it names candidate-61's image, see the last section).

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
  ms: 1.7 ms less, +6.7% (about 92.0 tok/s), more than the launch costs and the collectives'
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

## dgx4 rebooted

One node, one variable: dgx4 rebooted at 2026-11-17T23:07Z (back in 41 s; state before and after
in [runs/reboot-dgx4-before.txt](runs/reboot-dgx4-before.txt) and
[runs/reboot-dgx4-after.txt](runs/reboot-dgx4-after.txt)), dgx1 left up since 2026-11-07. Then
October's bundle on October's image on both, as before
([expert-rate-1025-tp4.json](expert-rate-1025-tp4.json) again;
[runs/expert-rate-1025-image-dgx4-after-reboot.txt](runs/expert-rate-1025-image-dgx4-after-reboot.txt),
`-dgx1-after-reboot`).

| | FC1 at 21 experts | FC1 marginal | Plain read |
| --- | ---: | ---: | ---: |
| dgx4 before, up ten days | 232.3 GB/s | 243.4 GB/s | 229.4, 235.8 GB/s |
| dgx4 after the reboot | 253.5 | 266.4 | 249.6, 253.6 |
| dgx1 before | 233.0 | 243.6 | 236.7, 235.8 |
| dgx1 again, not rebooted | 233.3 | 243.4 | 236.8, 235.5 |

The reboot restores October's rate in this bundle; FC2's marginal rate barely moved (263.5 to
266.0 GB/s). Free memory on dgx4
went from nothing at the two largest orders to 233 blocks of 512 MiB.

## What holds the large blocks

[memory_blocks.py](memory_blocks.py) on copies of `/proc/kpageflags` and `/proc/kpagecount`
([receipts/memory-blocks.txt](receipts/memory-blocks.txt)):

- dgx1, up ten days, idle, 106.6 GiB free: every one of its 248 blocks of 512 MiB holds pages
  the kernel cannot move (slab 1.3 GiB, other kernel pages 5.6 GiB; a median of 41 pages, 2.6
  MiB, a block). 104.9 GiB is free inside those blocks and cannot form a whole one.
- dgx4, an hour after its reboot, with a build's file cache: 97 blocks entirely free, 50 more
  hold only movable pages.
- sp8 (4 KiB pages, never stopped by a cluster start): 30 GiB of slab and 51 GiB of file cache;
  58,332 of its 63,503 blocks of 2 MiB hold unmovable pages.

The unmovable total is about 7 GiB on dgx1 and on dgx4 alike: the pages are not leaked, they are
spread. Which kernel code owns the scattered ones is not shown (it needs the `page_owner` boot
parameter, a host setting).

## Where the engine's memory sits

The loader's sponge (vLLM patch 0800, `direct_loader.py`): the driver takes GPU memory from the
kernel's page allocator in 64 KiB pieces, and the allocator serves its smallest free fragments
first. Before the weights are placed the loader allocates throwaway GPU memory in 128 MiB chunks
until `/proc/buddyinfo` shows blocks of the target size being consumed, places the weight arena
and the expert slabs, keeps the sponge through post-load processing, and releases it. The target
is 512 MiB when the free 512 MiB blocks can hold the weights (64 KiB kernels), else 2 MiB, then
1 MiB, then 512 KiB.

- The ring's logs: 72 sponge lines in the engine logs kept on dgx1, every one with a 2 MiB
  target; in boot `c61-r13` it soaked 4.8 to 8.6 GiB a node for 73.6 GiB of weights. No recorded
  ring boot placed weights in 512 MiB blocks.
- sp1 with an engine loaded (4 KiB pages, candidate-60, 90.6 GiB of GPU memory; a copy of
  `/proc/kpageflags` read at low priority, [engine_pages.py](engine_pages.py),
  [receipts/engine-pages-sp1.txt](receipts/engine-pages-sp1.txt)): 89.3 GiB in 1,463,457
  allocations of 64 KiB; 88% of it in contiguous runs of 2 MiB or more, 73% inside whole aligned
  2 MiB blocks, 11.7% in runs under 2 MiB, 0.7% under 256 KiB. The map does not tell weights from
  the engine's other memory.

**What this says about the bundle.** The bundle and the plain read do not go through the loader:
no sponge, so their buffers take the smallest free fragments. dgx1 holds 10.7 GiB free in
fragments under 2 MiB (2.9 GiB at 256 KiB or less); dgx4 after its reboot held almost none. The
7 to 9% the bundle lost on nodes up ten days may therefore be the case the sponge exists for,
and the engine, whose weights sit in blocks of 2 MiB or more on old and fresh nodes alike, may
not pay it. Not shown either way: the first traced boot had all four ranks up ten days and they
agree within 0.6% ([rank_compare.py](rank_compare.py),
[receipts/rank-compare.json](receipts/rank-compare.json): FC1 206 to 207 µs a call, the target's
LM head 1071 to 1078 µs, step 27.28 ms). The second traced boot
([trace-after-reboot-tp4.json](trace-after-reboot-tp4.json)), with dgx4 rebooted and the other
three not, decides it: a faster rank on dgx4 means serving pays for the time since boot too.

## Runs after the reboot

- `c1-floor-now-expert-rate-after-reboot-tp4` (queue entry `20261118T003759` at `4dea1ea17`):
  **failed on all four nodes in under a second, nothing measured.** The bundle mounted
  candidate-61's overlay files, and the overlay left the repository when candidate-61 was built
  as an image (patches 0817 to 0822): the mount sources did not exist.
- The bundle now names `spark-ds41f:candidate-61` and mounts only its script
  ([expert-rate-c61-image-tp4.json](expert-rate-c61-image-tp4.json), run
  `c1-floor-now-expert-rate-c61-image-tp4`). The image is a second variable against the runs
  above; dgx1 to dgx3, not rebooted, are the control for it.
  Queue entry `20261118T004302` at `4f449bee9`: pass on each node
  ([runs/expert-rate-dgx1-c61-image.txt](runs/expert-rate-dgx1-c61-image.txt) to `-dgx4`; each
  file starts with the node's boot time and its `/proc/buddyinfo` line at collection).

| Node | FC1 marginal, before | on candidate-61's image | FC2 marginal, before | now | Plain read, before | now | Free under 2 MiB | Free 512 MiB blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dgx1 | 247.7 GB/s | 247.1 | 253.2 | 254.5 | 237.3, 235.6 | 238.1, 236.0 | 10.9 GiB | 0 |
| dgx2 | 250.3 | 249.4 | 257.7 | 255.7 | 235.2, 236.7 | 235.6, 237.2 | 6.6 GiB | 0 |
| dgx3 | 249.6 | 249.6 | 257.5 | 257.5 | 237.7, 236.3 | 238.3, 236.4 | 6.4 GiB | 0 |
| dgx4, rebooted 97 minutes before | 247.6 | 253.1 | 256.6 | 260.5 | 237.4, 235.1 | 244.6, 241.3 | 0.65 GiB | 68 |

  - The image is not a variable here: the three nodes that were not rebooted read within 0.4%
    of their runs on candidate-60's image with the overlay.
  - dgx4 reads 2.2% faster in FC1 and 3% faster in the plain read than before its reboot, where
    October's bundle read 9.4% faster four minutes after it. Between the two runs dgx4 ran an
    image build and the release tests; its free 512 MiB blocks went from 233 to 68. Either the
    gain fades as the node's memory is used, or the two bundles respond differently: October's
    bundle again on dgx1 and dgx4
    ([expert-rate-1025-later-tp4.json](expert-rate-1025-later-tp4.json)) separates the two.
  - The bundle allocates the raw weights (1.7 GB), then the prepared FC1 weights (1.17 GB),
    then FC2's (0.58 GB), and the allocator serves its smallest free blocks first. By that
    order (not mapped), FC1's weights sat in fragments under 1 MiB on dgx1 and in blocks of
    about 8 MiB on dgx4 today, and FC2's in larger blocks than FC1's on every node, which is
    consistent with FC2's rate barely moving in any of these runs.
- `c1-floor-now-trace-after-reboot-tp4` (queue entry `20261118T004321` at `4f449bee9`):
  **refused before any boot** ("qualified config drift for tp4-ring: serve_args differ from
  qualified profile"). [prof-tp4.json](prof-tp4.json) carries candidate-61's dev arm's resolved
  arguments, and that arm's KV cache went from 10.5 to 16.5 GiB when candidate-61 was built
  (`--kv-cache-memory-bytes` 11274289152 to 17716740096). The arm now carries the current
  value. So the second traced boot differs from the first in the image, the KV cache size and
  dgx4's reboot; its question is answered inside the boot (dgx4's rank against the other
  three), not against the first trace.
- October's bundle once more, 101 minutes after dgx4's reboot (queue entry `20261118T004758`):
  pass on dgx1 and dgx4
  ([runs/expert-rate-1025-image-dgx4-later.txt](runs/expert-rate-1025-image-dgx4-later.txt),
  `-dgx1-later`).

| October's bundle and image | FC1 marginal | FC2 marginal | Plain read |
| --- | ---: | ---: | ---: |
| dgx4 before its reboot | 243.5 GB/s | 263.5 | 229.4, 235.8 |
| dgx4 four minutes after | 266.4 | 266.0 | 249.6, 253.6 |
| dgx4 101 minutes after | 248.8 | 261.1 | 239.5, 238.3 |
| dgx1, not rebooted: the three runs | 243.6, 243.4, 243.9 | 260.8, 260.1, 261.5 | 235 to 237 |

  **The reboot's gain fades with use, not with the bundle**: the same bundle on the same node
  keeps 2.2% of the 9.4% after 100 minutes in which the node ran an image build, the release
  tests and two bundle runs. The `/proc/buddyinfo` lines in the run files are from collection,
  seconds after each run (once, on dgx4, after the next job's cluster start had begun): they
  are not the state a run saw, and which blocks its buffers took is not recorded.
- The second traced boot (queue entry `20261118T004804` at `8d85f750d`, 105 minutes after
  dgx4's reboot): candidate-61's image; the job ended clean, 305 six-row steps on every rank as
  in the first.

## The rebooted rank in serving

[rank_compare.py](rank_compare.py) on both traced boots
([receipts/rank-compare.json](receipts/rank-compare.json)): dgx4's mean time a call against the
mean of the other three ranks.

| Kernel | ms a step | First trace, all four up ten days | Second trace, dgx4 rebooted | Change |
| --- | ---: | ---: | ---: | ---: |
| FC1 (target routed experts) | 8.24 | +0.2% | -2.0% | -2.2 |
| FC2 (target routed experts) | 3.98 | +0.3% | -0.5% | -0.8 |
| Target LM head | 1.07 | +0.6% | -1.4% | -2.0 |
| Drafter head screen | 0.65 | +0.1% | +1.4% | +1.3 |
| Dense projections | 7.04 | +2.5% | +3.3% | +0.8 |
| L2 prefetch | 8.70 | +0.7% | +2.0% | +1.3 |
| Step | 27.41 (27.28 in the first) | lockstep | lockstep | |

- **In serving the rebooted node is 2% faster in FC1 and the target's LM head, and no faster
  elsewhere.** Summed over the stream kernels (prefetch aside, it runs beside them) that is
  0.17 ms of the 27.4 ms step: +0.6% if every rank had it. The three old ranks differ from each
  other by up to 0.9% in these kernels (0.2% in FC1), so FC1's change stands clear of the spread
  and the LM head's barely. One boot.
- **The 7 to 9% the bundles showed does not carry over to the engine.** The byte floor section
  above called the memory rate the largest item (1.7 ms, +6.7%) on the bundle's figure; on this
  trace a reboot is worth a tenth of that to serving, and the bundle's figure itself falls to
  2.2% within two hours of use.
- **Where the two engines sat** ([receipts/engine-pages-ring.txt](receipts/engine-pages-ring.txt),
  maps read with the engine loaded, the engine taken as the plain kernel pages that were not
  such on the node's idle map):

| | dgx1, up ten days | dgx4, rebooted |
| --- | ---: | ---: |
| Engine memory | 93.1 GiB | 92.1 GiB |
| Sponge target; soaked before the weights | 2 MiB; 7.4 GiB | 2 MiB; 1.9 GiB |
| In contiguous runs of 32 MiB or more | 55.2% | 82.4% |
| In runs of 2 to 32 MiB | 42.0% | 15.7% |
| In runs under 2 MiB | 2.8% | 2.0% |
| Inside whole aligned 2 MiB blocks | 88.6% | 94.1% |
| Inside whole aligned 32 MiB blocks | 22.7% | 55.7% |
| Inside whole aligned 512 MiB blocks | 0 | 0 |

  The sponge did its work on both: nine tenths of the engine in whole 2 MiB blocks, under 3% in
  fragments. dgx4's engine sits in longer runs, and is 2% faster in two kernels. Neither node
  got the 512 MiB target: dgx4 had 68 free 512 MiB blocks (34 GiB) at the last reading before
  the boot, against 73.6 GiB of weights.

### Not measured

- A boot on a node just rebooted, when the 233 free 512 MiB blocks (116 GiB) would give the
  sponge its 512 MiB target. Every recorded ring boot, this one included, placed the weights
  with the 2 MiB target; the bundle's 266 GB/s was read in that just-rebooted state.
- Why contiguity past 2 MiB would matter (the loader's note names only fragments of 64 to 256
  KiB), and why the drafter's head and the dense projections did not gain.
