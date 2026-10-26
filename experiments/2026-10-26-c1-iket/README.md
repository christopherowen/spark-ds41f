# One stream at 100 tok/s, bottom-up: where c1 decode kernels wait (IKET)

The first single-stream study (#598, `2026-10-26-c1-100tps`) worked top-down: the byte
floor is 70% of a c1 step, and everything that is not a byte caps at 103.9 tok/s.
This study works bottom-up from inside the kernels. IKET (#610, `2026-10-26-iket-debug`)
times each TileLang kernel's load, wait, compute and store phases. The questions are
where c1 decode kernels wait, what each wait is for, and what removing it is worth.

Rules, as for every study now: native weights and reference precision; temperature 0
bit-identical; no serving overhead (IKET stays off in production); a ceiling for each
idea from measured traces before building; GPU work through `scripts/lab.py
kernel-local` only, on the node the coordinator assigns.

**Where the runs happened.** dgx1-4 were powered off at 08:10 CET on 2026-10-26 for
re-cabling, and the smoke bundle queued on dgx4 at 08:35 never started. While production
waited for a cable that afternoon, the coordinator gave this study dgx1 alone. Every run
below ran there from 13:50 CET, one bundle at a time under lab.py's GPU lock, from this
branch's checkout in `~/spark-lab/agents/c1-iket/` ([Runs](#runs-on-dgx1)).

## The short answer so far

- **The routed experts' waits are the weight stream itself.** In IKET's hot traces the
  gate-up GEMM's warps spend 44% of their time waiting for TMA weight tiles. The GEMM
  still streams at 256 GB/s (FC2: 266) against the attainable 266.6, so these waits
  are DRAM time. More stages, a warp-specialized producer or fewer block syncs cannot
  shorten the steady stream. The block sync after `cp_async_wait` (19.5%) comes after
  the next K block's loads have issued, so it does not cut the bytes in flight.
- **What c1 pays is each launch's ramp and tail, and the kernel between the GEMMs.**
  At c1 a layer reads about 14 experts, a few waves of 48 SMs. In production, at TP4,
  each layer's routed chain spends:
  - 3.8 µs in the SwiGLU and its two idle gaps;
  - 9.3 µs of FC2 fixed cost (a regression over 400 traced layers, r² 0.96);
  - 6.2 µs an expert in FC2 against 5.7 at the attainable rate;
  - FC1's fixed cost, about 14 µs in the bench.

  Even a plain 18 MiB read pays about 11 µs of ramp and tail on GB10.
- **One continuous FC1-FC2 stream is the largest lever this study owns.** It removes
  FC1's tail, the SwiGLU launch and FC2's ramp, and brings FC2 to the attainable rate.
  Ceiling: TP4 0.93 ms a step, 75.4 → 77.8 tok/s (+3.2%); TP3 1.01 ms, 60.7 → 62.3
  (+2.8%). It sits inside #598's rows 5 and 11 (1.51 ms), not beside them.
- **The all-reduces wait for late peers.** With its peers ready, the kernel spends 68%
  of its time at the barrier after the peer-flag poll; with them 20 µs late, 90%. The
  lateness is compute jitter between ranks (79%, the 10-07 attribution). Its ceiling is
  #598's: at most the 0.95 ms of DRAM-idle time under collectives (+3.2%). It needs the
  cluster, so it is not prototyped here.
- **Prototype 1, the stream grouped GEMM, is built and waits for dgx4.** It is a part of
  the continuous stream, bit-identical by construction, and has a bit-identity test.
  Its gated A/B against next-candidate is ready (`bundles/stream-ab`).

## Inputs

- TP4 production, c1: rank 0's torch-profiled traces of request-inventory-5
  (`results/private/determinism/request-inventory-5/ports`): greedy (10 steps of
  34.6 ms) and sampled (25 of 29.5 ms). These are the traces #598 and the fusion survey
  (#583) read.
- The release suite's single stream, spark-ds41f-1025 (candidate 6): TP4 33.003
  steps/s at 2.284 tokens a step; TP3 26.730 at 2.269
  (`manifests/suites/2026-10-25-spark-ds41f-1025-tp{4,3}-single-stream`).
- The expert-rate bundle (#598, `runs/expert-rate-dgx4-2.txt`): FC1 and FC2 at 6-36
  experts, and plain reads.
- IKET's hot-kernel traces (#610, `runs/hot-5/table.json`): the routed experts, sparse
  MLA, the mHC entry and the one-shot all-reduce, at TP3 shapes and 8 rows with random
  routes (about 45 distinct experts).
- vLLM next-candidate 97e204cd9 (PDL along the chain, late trigger from 16 rows).

## Method

**The per-rank layer harness** (`harness/c1_layer.py`). The production c1 step cannot
be traced with IKET on the cluster. That needs a window, and run-iket allocates one
pool for the whole process. A single GPU cannot hold DS4.1-Flash either. So the harness
runs one c1 decode layer's kernels at one TP4 or TP3 rank's shapes on one GB10. It
builds on vLLM's PDL layer chain (`tests/kernels/deepseek_v41_pdl_chains.py`):
- the attention entry (mHC `post_pre` and finalize with its MXFP8 cast);
- the fused Q-A/KV GEMM (replicated, 1792 × 5120), the q latent's RMSNorm, Q-B (16
  heads a rank at TP4; 24 at TP3, whose 64 heads are padded to 72);
- sparse MLA decode on the route production picks at six rows (the IKET hot harness's
  inputs);
- WO-A's split-K partials and reduce-cast (2 groups, or 3), WO-B;
- the FFN entry; the shared expert on its own stream; the routed experts (dispatch,
  FC1, SwiGLU and cast, FC2, combine);
- with `--head`, the packed 12-bit head at the rank's vocabulary shard (32,320 rows or
  43,136).

Routes are real. Layer *i* takes the reference forward's layer *i* mod 40 routes for
six consecutive greedy tokens (`2026-10-18-json-count/runs/ref-woa1.json`), with rows
from `--live` on dead. Three live rows of six give about 14 distinct experts a layer,
the suite's operating point. Weights rotate through four sets, so no layer finds the
previous one's weights in L2. Every pass hashes every buffer it writes; equal digests
between builds mean equal bits.

**Traces** (`make_bundles.py`, bundle `layer`), for each rank's shapes:
1. the off build's times, as a CUDA graph;
2. nsys of the graph and of eager launches, for absolute kernel times;
3. run-iket with every kernel instrumented, K loops unsampled: the phase table
   (IKET's `iket_table.py`);
4. warp lifetimes, per SM (bundles `lifetimes`, `lifetimes-stream`).

Two things about run-iket, learned on the way:
- It records warp lifetimes only for instrumented kernels. With every kernel skipped
  its trace is empty (smoke run 2), and with only the combine instrumented it holds the
  combine alone (smoke run 3).
- So the lifetimes runs instrument everything, with one CTA a launch recording ranges
  (`VLLM_DS41_IKET_CTAS=1`). Every warp's lifetime is recorded, and the other CTAs pass
  a uniform branch at each event. In a persistent kernel the recording CTA runs slower
  and becomes its tail, so the stream GEMM's lifetimes are not timings; the A/Bs are.

**Validation.** The IKET build writes the same bits with the mode off and with it on and
every kernel skipped (digest `51332953ab203838`, smoke run 2). Graph replays write eager
launches' bits in every run.

**Analysis:**
- `cta_timeline.py` rebuilds from step 3 each launch's SM occupancy and, for the routed
  GEMMs, the weight stream over time. It splits each launch's busy time into ramp,
  steady stream and tail against the attainable 266.6 GB/s.
- `step_shares.py` gives each kernel family's share of the real TP4 step.
- `routed_chain.py` gives the routed chain's anatomy and FC2's serving fit.
- `levers.py` turns per-step savings into tok/s ceilings.

## Each kernel family's share of a c1 step (TP4 production)

`step_shares.py`, c1 greedy, 10 steps of 34.63 ms. "Critical" is the time a kernel runs
with nothing but the L2 prefetch beside it, plus the idle gap before it (the fusion
survey's measure).

| family | launches | kernel µs | critical µs | share |
|---|---:|---:|---:|---:|
| routed FC1 | 40 | 9,802 | 9,051 | 26.1% |
| routed FC2 | 40 | 5,357 | 5,377 | 15.5% |
| all-reduce | 81 | 4,172 | 4,232 | 12.2% |
| mHC (project, finalize, post) | 170 | 1,489 | 1,815 | 5.2% |
| attention small kernels (indexer, metadata, cache, RoPE, compress) | 228 | 986 | 1,214 | 3.5% |
| Q-B GEMM | 48 | 1,098 | 1,134 | 3.3% |
| target LM head | 1 | 1,049 | 1,052 | 3.0% |
| sparse MLA | 80 | 930 | 1,035 | 3.0% |
| drafter LM head | 1 | 1,012 | 1,013 | 2.9% |
| WO-A (partials, reduce-cast) | 80 | 901 | 986 | 2.9% |
| WO-B and shared down GEMMs | 80 | 1,616 | 905 | 2.6% |
| Q-A/KV GEMM | 40 | 775 | 818 | 2.4% |
| casts and norms | 303 | 606 | 624 | 1.8% |
| DSpark main projection | 2 | 359 | 361 | 1.0% |
| routed dispatch, SwiGLU, combine | 120 | 372 | 276 | 0.8% |
| shared gate/up GEMM | 40 | 515 | 52 | 0.1% |
| other target kernels (all-gathers, router gate, compressors, Engram `wkv`, small) | | | 1,149 | 3.3% |
| the drafter, but its LM head | | | 1,989 | 5.7% |
| launch gaps and host | | | 1,550 | 4.5% |

The sampled window is in `step_shares.json`: 29.45 ms, FC1 16.4%, the all-reduces
13.8%, FC2 10.5%.

## Inside the kernels: the phase split (IKET)

From IKET's hot traces (#610 run 5: TP3 shapes, 8 rows, mean warp shares). Each row
draws six random experts, so a layer reads about 45 distinct experts (the expectation
for 8 × 6 draws from 384); the rates below use that number. These are not c1's routes. The c1 tables at both ranks' shapes are the
`layer` bundle's job.

| kernel | step share (TP4) | wait | load | compute | store | the wait is for |
|---|---:|---:|---:|---:|---:|---|
| routed gate-up GEMM (FC1) | 26.1% | 73% | 8% | 13% | 0% | TMA weight tiles 44% (`loaded[s].wait`); the block sync after `cp_async_wait` 19.5% (the slowest warp's activation and scale-code copies); `cp_async_wait` 6% |
| routed down GEMM (FC2) | 15.5% | 54% | 21% | 14% | 2% | TMA weight tiles 21%; the sync after `cp_async_wait` 19%; besides the waits, the tile's prologue loads (expert, routes, base exponents) take 16% |
| one-shot all-reduce | 12.2% | 68% | 1% | 14% | 8% | the barrier after six lanes poll the peers' flags (90% with peers 20 µs late) |
| mHC `project_streams` | (5.2% with the rest of mHC) | 5% | 34% | 45% | 2% | little: TF32 MMA and the streams' loads |
| mHC `finalize` | | 15% | 19% | 21% | 10% | warp 0's mix reduction; 35% unattributed |
| sparse MLA split | 3.0% | 16% | 11% | 31% | 5% | MMA, the softmax reduction; 38% unattributed (27 ranges) |
| SwiGLU and cast | 0.8% (with dispatch, combine) | 0% | 64% | 13% | 3% | none: reading gate/up |
| MoE dispatch | | 40% | 2% | 0% | 34% | its nine block syncs |

What the waits mean for c1:
- **Weight-tile waits are DRAM time.** At 45 experts, FC1's 576 CTAs stream
  182.6 MB in 712.9 µs (256 GB/s) and FC2's 1,920 CTAs 91.4 MB in 344.0 µs (266 GB/s).
  The expert-rate bundle gives FC1 11.6 µs an expert at every count from 9 to 36 at six
  rows: 262 GB/s, 98% of the attainable rate. There is no wave staircase, so the
  steady stream has no wait to remove.
- **The block syncs do not cut the bytes in flight.** Each K block's loads are issued at
  the top of the iteration, before `cp_async_wait`, its sync and the MMA. A warp held
  at the sync holds no load back. With three stages, two K blocks of weights (16 KB) are
  in flight per SM, so the 48 SMs keep about 770 KB in flight against the 266 GB/s
  stream. A faster per-SM rate matters only while fewer SMs stream: in the ramp and the
  tail.
- **The per-tile prologue matters where tiles are few.** An FC2 tile streams 37-51 KB.
  Its prologue (reading its expert, routes and base exponents, then a block sync) and
  its store are about 2.7 µs of an 11.6 µs CTA under the profiler. At 45 experts, 40
  waves of tiles hide that behind other SMs' streams. At c1's 14 experts they hide it
  less: in serving, FC2 costs 6.2 µs an expert against 5.7 at the attainable rate.

## The routed chain in production (TP4)

`routed_chain.py`, c1 greedy, 400 layers; medians a layer:

| launch | µs | idle gap before, µs | critical µs |
|---|---:|---:|---:|
| dispatch | 3.3 | 0.0 | 0.5 (hidden behind the shared expert) |
| FC1 | 253.9 | 0.0 | 231.5 |
| SwiGLU and cast | 3.1 | 0.19 | 3.1 |
| FC2 | 140.5 | 0.48 | 140.5 |
| combine | 2.7 | 0.19 | 2.7 |

FC2 on FC1 across the 400 layers: FC2 = 1.6 + 0.540 × FC1 µs (r² 0.96). With FC1 at the
bench's 14.2 µs + 11.41 µs an expert, that is FC2 = 9.3 µs + 6.16 µs an expert in
serving. The bench gives 14.5 + 5.77 under event timing, and the attainable rate 5.72.
Sampled (1,000 layers, 11.5 experts a layer): 3.1 µs + 6.45 µs an expert.

GB10 streams with a fixed cost even with no GEMM around it. A plain read reaches 221.5
GB/s on 18 MiB and 253.7 on 110 MiB (the expert-rate bundle), about 11 µs of ramp and
tail. A launch boundary on the weight stream costs about that, wherever it sits. The
routed chain has two in each layer, FC1's end and FC2's start, and the SwiGLU launch
between them.

## Ranked levers, with ceilings at c1

`levers.py`, one lever at a time, from the suite's step (TP4 30.30 ms at 2.284 tokens a
step, 75.4 tok/s; TP3 37.41 ms at 2.269, 60.7 tok/s) at its 13.8 experts a layer and
43 MoE launches a step. Every saving is non-byte time in #598's terms, so none
overlaps the byte floor.

| # | lever | owner | TP4 | TP3 | evidence |
|---|---|---|---|---|---|
| 1 | L2 prefetch timed against its consumers | l2-prefetch-timing session | 1.67 ms (+5.8%, 79.8 tok/s) | - | moe-overlap and PDL records; not counted here |
| 2 | Routed experts as one continuous FC1-FC2 stream | this study | 0.93 ms (+3.2%, 77.8) | 1.01 ms (+2.8%, 62.3) | a layer: SwiGLU and gaps 3.8 µs, FC2 ramp 4.6, FC1 tail 7.1, FC2 above rate 6.0 (TP3 8.0) |
| 2a | of which the stream GEMM (prototype 1) | this study | FC2 above rate and the launches' tails: about 0.3-0.5 ms | about 0.4-0.6 ms | to be measured (`stream-ab`) |
| 3 | All-reduce: peer skew and staging | sparknet; cluster-only | at most 0.95 ms (+3.2%, 77.8) | at most 0.95 ms (+2.6%, 62.2) | #598: DRAM idles 0.95 ms under collectives; IKET: the time is the peer wait |
| 4 | mHC boundary (project, finalize, post) | fusion survey | its 3.1% c1 restructure ceiling | - | 5.2% of the step; IKET: small compute- and load-bound kernels |
| 5 | LM heads at the attainable rate | unclaimed | 0.20 ms (+0.7%, 75.9) | 0.27 ms (+0.7%, 61.1) | #598 row 12 |
| 6 | Routed GEMMs' steady stream: deeper stages, warp-specialized producer, fewer syncs | none | 0 | 0 | 98% of the attainable rate; the syncs follow the loads' issue |
| 7 | FC2's first weight loads before its PDL wait | - | at most 0.13 ms (+0.4%), part of row 2 | | the SwiGLU it would overlap lasts 3.1 µs |

Rows 2, 3 and 5 together: TP4 80.9 tok/s, TP3 64.5. Row 2's split between ramp and
tail (each half of a launch's fixed cost here) is an assumption until the `layer`
traces measure it.

Reconciled with #598: row 2 is a part of its rows 5 and 11 (launch fixed costs 1.24 ms,
FC2's rate 0.27 ms). Its remaining 0.58 ms is each MoE chain's first ramp (FC1's start
after the dispatch) and last tail (FC2's end before the combine). Neither goes away
while the routed experts are their own launches. Row 3 is #598's row 7. Row 1 is most
of #598's 3.4-3.7 ms of prefetch and consumer time.

## Prototype 1: the stream grouped GEMM

vLLM branch `c1-stream` (worktree ~/projects/vllm-c1-stream; 3360247fe and 06b74e7ff
over next-candidate 97e204cd9), not pushed:
`moe.grouped_fp8_fp4_gemm_stream`. Below 16 rows, the plan's gate-up and down GEMMs use
it in place of the grid kernel.

- The grid kernel launches a CTA for each (tile, column block) item: 324 for FC1 at six
  rows, of which about 126 are live. Each CTA fills its pipeline, streams one
  `[128, K]` weight slab and drains. The CTAs past the last used tile exit after their
  PDL wait.
- The stream kernel launches 48 CTAs, one an SM. They split the live items into
  contiguous runs of `ceil(items / 48)`. At 14 experts that is 3 FC1 items on each of
  42 SMs, and 12 FC2 items on each of 47. A CTA's pipeline runs through its items' K
  blocks without draining: an item's first stages load while the previous item's last
  ones multiply. Its weight slabs are consecutive in memory.
- Each item's tile multiplies its K blocks in order into a cleared FP32 accumulator,
  with the grid kernel's MMAs, operands and expanded scale words. Every output
  therefore has the grid kernel's bits. `test_stream_gemm_gives_the_grid_gemm_bits`
  compares both kernels bit for bit: TP4's and TP3's FC1 and FC2 shapes (TP4's FC2 has a
  64-wide K tail), at 1, 3, 6, 12 and 15 rows, padding rows included.
- What it can remove: FC2's per-tile prologue and fill, about 0.4 µs an expert above
  the attainable rate; the uneven last wave of both GEMMs; and the exiting CTAs. What it
  cannot: each launch's own ramp and tail, and the SwiGLU between the GEMMs (row 2's
  rest).

Bundles: `stream-tests` runs the routed-expert tests, the PDL tests and the source
check. `stream-ab` runs the layer harness on next-candidate and the stream build in
alternation, six rounds, at TP4 with 3 and 6 live rows and TP3 with 3, and compares
bits and µs a layer. `layer-stream` traces its CTA timelines.

## What runs next

In order, once a test box is free. `run_queue.sh` (copied to `~/spark-lab/agents/c1-iket/`
on the node, with bundle:output pairs after the smoke run) stops at a failed smoke run or
when a release request appears:

1. `smoke`: the harness at both ranks' shapes, and one lifetimes trace.
2. `stream-tests`: prototype 1's bits and the kernel tests.
3. `layer`: the c1 tables at TP4 and TP3. They measure each routed launch's ramp,
   steady stream and tail, and decide row 2's split.
4. `stream-ab`: prototype 1 against next-candidate.
5. `layer-stream`, `layer-live6`, `layer-base`: prototype 1's timelines, the 24-expert
   operating point, and next-candidate's own bits and times in the harness.

About 2-3 hours of single-GPU time.

What the results decide:
- If FC1's tail and FC2's ramp are near the 7 and 5 µs assumed, the continuous FC1-FC2
  stream is worth building for the release after next. It is a persistent kernel: FC2
  tiles of an expert start when its FC1 tiles are done, with the SwiGLU and cast fused,
  and a deadline trap for liveness.
- If prototype 1 alone recovers most of row 2a, take it first: it is a small,
  bit-identical change.

## Files

| file | what |
|---|---|
| `harness/c1_layer.py` | the per-rank c1 layer harness |
| `cta_timeline.py` | SM occupancy and weight stream from run-iket's warp lifetimes |
| `step_shares.py`, `step_shares.json` | kernel families' shares of the TP4 production step |
| `routed_chain.py`, `routed_chain.json` | the routed chain in production; FC2's serving fit |
| `levers.py`, `levers.json` | ceilings in tok/s |
| `ab.py` | base against prototype on the harness |
| `make_bundles.py`, `bundles/` | the kernel-lab bundles (IKET experiment's mechanics) |
| `run_queue.sh` | the node-side driver: the bundles one after another |
