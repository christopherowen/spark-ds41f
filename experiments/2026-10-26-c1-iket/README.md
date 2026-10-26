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

## The short answer

- **Where c1 kernels wait.** At a TP4 rank's shapes and the suite's routes:
  - The routed FC1 GEMM's warps spend 65% of their time waiting: 36% for TMA weight
    tiles, 20% at the block sync after the activation copies.
  - FC2's CTAs live 4 µs each, and 13% of that is their prologue's base-exponent loads.
  - The dense GEMMs wait 50-65% on their cp.async weight copies, and 14-21% at the sync
    after them.
  - The all-reduce waits at the barrier after its peers' flags: 68% of its time with the
    peers ready, 90% with them 20 µs late (IKET's hot traces, simulated peers).
- **The routed experts' waits are DRAM time, and the chain is already close to an
  ideal stream.** On one GPU, next-candidate's routed chain (dispatch, FC1, SwiGLU, FC2,
  combine) runs within 0.9-6.8 µs a layer of an ideal one-launch read of the same
  expert bytes:
  - TP4: 251.8 against 245.0 µs at 13 experts, 446.3 against 441.6 at 24;
  - TP3: 324.2 against 318.0, and 584.4 against 583.5.

  So no restructuring of the routed experts recovers more than about 0.3 ms a step:
  TP4 ≤ +1.0%, TP3 ≤ +0.7%. That covers a continuous FC1-FC2 stream, persistent or
  stream GEMMs, deeper stages, a warp-specialized producer, a fused SwiGLU and fewer
  syncs.
- **What remains of #598's "launch fixed costs" is the ramp and tail of any DRAM
  stream.** An ideal read pays it too: one launch a layer is 5-17 µs slower than one
  launch for all layers. The next layer's experts are unknown until routing, so the
  routed stream cannot cross it. Known traffic, such as the next prefetch window, can
  fill it, and that is the L2 prefetch session's lever.
- **Prototype 1, a stream grouped GEMM (one CTA an SM over runs of tiles), is slower:**
  - v1 lost 0.4-1.8% in the full layer;
  - v2 is 9-28 µs a layer slower than the grid chain;
  - both kept the bits exactly.

  It is dropped. Its lesson: one SM with two K blocks in flight cannot carry more than
  its forty-eighth of the stream, so a schedule that idles SMs loses.
- **The c1 levers this study can rank, from the measured tables** ([Ranked
  levers](#ranked-levers-with-ceilings-at-c1)):
  - L2 prefetch timing (owned elsewhere, +5.8% TP4);
  - the latency-bound small kernels (at most +4.7% TP4, +3.7% TP3; the fusion
    survey's);
  - the all-reduce's peer waits (at most +3.2%, cluster only);
  - filling stream boundaries with prefetch traffic (+1.6% / +0.6%);
  - the routed-expert pipeline (+1.0% / +0.7%);
  - the heads (+0.3%).

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

**Validation.**
- The IKET build writes the same bits with the mode off and with it on and every kernel
  skipped (digest `51332953ab203838`, smoke run 2).
- next-candidate itself writes the IKET build's bits at both ranks' shapes (layer-base-1
  against layer-1: `cc4d115b6d040964` at TP4, `e2d08a7c6ba6b1cb` at TP3).
- Graph replays write eager launches' bits in every run.

**Analysis:**
- `layer_table.py` gives each kernel family's cost, its excess over its bytes, its IKET
  phase split and what its waits are for, from a `layer` run.
- `cta_timeline.py` rebuilds from the lifetimes each launch's SM occupancy and the routed
  GEMMs' weight stream over time. Under run-iket the eager launches and the recording
  CTAs shift the timings, so its numbers are only qualitative; the bound and the A/Bs
  carry the measurements.
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

## The c1 layer at one rank's shapes, measured (dgx1)

`layer_table.py` over run layer-1 (TP4 and TP3, 3 of 6 rows live, 13.25 distinct experts
a layer) and layer-live6-1 (TP4, all 6 rows live, 24.38). Columns:
- **cost:** each kernel's end less the previous main-chain kernel's end, in CUDA graph
  replays, summed over a family's launches in a layer, median over layers;
- **excess:** cost less the family's bytes at 266.6 GB/s;
- **phases:** IKET's split of a warp's life with the PDL wait taken out;
- **waits:** what each wait range waits for, named from its statement.

The dense GEMMs read cold weights here; in production the L2 prefetch fills them first.

**TP4, 13.25 experts a layer: 519.8 µs a layer**

| kernel | TP4 production share | cost µs | excess µs | wait | load | compute | store | the waits are for |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MoE block: dispatch, FC1, SwiGLU, FC2, combine, with the shared expert beside them | 42.5% | 287.6 | 32.5 | | | | | |
| routed FC1 (its window also carries the shared gate/up's bytes) | 26.1% | 181.0 | | 65% | 8% | 16% | 0% | TMA weight tiles 36%; the sync after cp.async 20%; cp.async 6% |
| routed FC2 | 15.5% | 82.1 | 8.3 | 46% | 30% | 16% | 3% | the sync after cp.async 19%; TMA weight tiles 10%; cp.async 9% |
| routed combine (waits for the shared down GEMM, which runs after FC2 under PDL at six rows) | 0.8% | 15.5 | | 0% | 89% | 0% | 3% | |
| routed dispatch | | 6.4 | | 42% | 1% | 0% | 52% | its block syncs |
| routed SwiGLU and cast | | 1.6 | | 0% | 60% | 4% | 3% | |
| WO-B GEMM | 2.6% | 46.6 | 7.2 | 80% | 12% | 5% | 0% | cp.async weight copies 65%; the sync after them 14% |
| Q-B GEMM | 3.3% | 45.0 | 5.6 | 74% | 19% | 4% | 0% | cp.async 57%; sync 16% |
| WO-A partials and reduce-cast | 2.9% | 40.2 | 8.7 | 68% | 17% | 3% | 0% | cp.async 49%; sync 15% |
| Q-A/KV GEMM | 2.4% | 39.6 | 5.1 | 86% | 8% | 6% | 0% | cp.async 62%; sync 19% |
| mHC project and finalize (4 launches) | 5.2% | 22.3 | 7.6 | 11% | 44% | 15% | 3% | block syncs |
| sparse MLA (split and combine) | 3.0% | 15.6 | | 22% | 27% | 20% | 3% | block syncs 19% |
| q-latent norm and Q-B cast | 1.8% | 2.8 | | 3% | 16% | 4% | 53% | |
| LM head (once a step) | 3.0% | 972.4 | 41.3 | 40% | 25% | 7% | 0% | cp.async 24%; sync 14% |

**TP3, 13.25 experts: 673.7 µs a layer.** MoE block 350.9 µs, 28.7 over its bytes. FC1
224.5 µs, and FC2 100.3 µs (6.7 over). WO-B 76.6 (17.6 over), Q-B 68.1 (9.1),
WO-A 60.5 (13.3), Q-A/KV 38.6 (4.2), mHC 22.1 (7.4), MLA 17.3. LM head 1,297.8 µs (55.2
over). The phase splits match TP4's within a few points.

**TP4, 24.38 experts: 703.0 µs a layer.** MoE block 465.6 µs, 34.7 over its bytes; FC1
298.8, FC2 140.1 (8.2 over).

What the tables say:
- **The MoE block's non-byte time is a fixed ~33 µs a layer.** At TP4 it is 32.5 µs at
  13 experts and 34.7 at 24; at TP3, 28.7. It is the launch boundaries' ramps and tails
  and the small kernels between them, not a per-expert loss. Each expert streams at the
  attainable rate once the GEMMs run.
- **FC2 gives away about 8 µs a layer to its tiles.** Each CTA lives 4 µs (TP4), and
  13% of that is the base-exponent loads of its prologue.
- **The shared expert's down GEMM finishes after FC2,** and the combine waits for it:
  about 12 µs a layer on the chain at TP4. Its bytes count in the floor anyway. PDL's
  order at six rows trades this against FC1's last wave keeping DRAM to itself
  (next-candidate 97e204cd9).
- **The latency-bound kernels add about 37 µs a layer at TP4:** mHC's non-byte 7.6,
  MLA 15.6, the norm and cast, dispatch, SwiGLU, combine. They are the fusion survey's
  territory.

## Earlier evidence: IKET's hot traces at 45 experts

Before dgx1 was free, the study worked from IKET's hot traces (#610 run 5: TP3 shapes,
8 rows, mean warp shares). Each row
draws six random experts, so a layer reads about 45 distinct experts (the expectation
for 8 × 6 draws from 384); the rates below use that number. These are not c1's routes;
the c1 tables are in the section above.

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
tail. That is what made the boundary between FC1 and FC2 look like a lever. The bound
below measures it: the chain's five launches cost only 1-7 µs a layer more than one
ideal launch of the same bytes, so the plain-read ramps overlap well inside the chain.

## The routed experts against an ideal stream (the bound)

`harness/stream_bound.py` (runs bound-2 on next-candidate, bound-stream-1 on the stream
GEMM's build) times four CUDA graphs of eight layers at each rank's shapes, with the
reference routes and no shared expert:
- the routed chain as the plan runs it;
- an ideal read of exactly the layer's live experts' FC1 and FC2 bytes, as one launch
  a layer; as two launches a layer; and as one launch for all eight layers.

The read kernel is a plain stream: 48 CTAs, contiguous 32 KB chunks, eight 16-byte
loads a thread a chunk.

| rank, experts a layer | chain µs a layer | GB/s | ideal, one launch a layer | ideal, all layers in one | chain - ideal | one launch - all |
|---|---:|---:|---:|---:|---:|---:|
| TP4, 13.25 | 251.8 | 240.7 | 245.0 | 234.0 | **6.8** | 11.0 |
| TP4, 24.38 | 446.3 | 249.8 | 441.6 | 428.3 | **4.6** | 13.3 |
| TP3, 13.25 | 324.2 | 248.9 | 318.0 | 312.7 | **6.2** | 5.3 |
| TP3, 24.38 | 584.4 | 254.0 | 583.5 | 566.3 | **0.9** | 17.2 |
| TP4, 13.25, stream GEMM v2 | 261.2 | 232.0 | 241.7 | 233.2 | 19.5 | 8.4 |

- The grid chain's five launches cost 1-7 µs a layer more than one ideal launch of the
  same bytes. Restructuring the routed experts is worth at most that: 43 MoE layers ×
  6.8 µs = 0.29 ms a step at TP4 (+1.0%), and 0.27 ms at TP3 (+0.7%).
- An ideal stream of one layer's bytes still pays 5-17 µs of ramp and tail against
  one stream of every layer's bytes. The routed experts cannot cross that boundary: the
  next layer's experts are known only after its routing. Traffic known in advance can
  fill it, which the L2 prefetch session's schedule does.
- The stream GEMM v2 (all 48 SMs) is 9-28 µs a layer slower than the grid chain at every
  shape: 261.2 / 464.9 / 341.9 / 612.0 µs against 251.8 / 446.3 / 324.2 / 584.4.

## Ranked levers, with ceilings at c1

`levers.py`, one lever at a time, from the suite's step (TP4 30.30 ms at 2.284 tokens a
step, 75.4 tok/s; TP3 37.41 ms at 2.269, 60.7 tok/s). The inputs are the measured bound
and layer tables above, over 43 MoE layers (40 target, 3 drafter) or the 40 target
layers. Every saving is non-byte time in #598's terms, so none overlaps the byte floor
or another row.

| # | lever | owner | TP4 | TP3 | evidence |
|---|---|---|---|---|---|
| 1 | L2 prefetch timed against its consumers | l2-prefetch-timing session (#615) | 1.67 ms (+5.8%, 79.8 tok/s) | - | moe-overlap and PDL records; its prototype measured −1.5% c1 step |
| 2 | Latency-bound small kernels: mHC's non-byte time, sparse MLA, the norm and cast, dispatch, SwiGLU | fusion survey (#583) | at most 1.36 ms (+4.7%, 78.9) | at most 1.33 ms (+3.7%, 62.9) | layer tables: 34 µs a layer at either rank; an upper bound (fused kernels still cost something) |
| 3 | All-reduce: peer lateness and staging | sparknet; cluster only | at most 0.95 ms (+3.2%, 77.8) | at most 0.95 ms (+2.6%, 62.2) | #598: DRAM idles 0.95 ms under collectives; IKET: the peer wait. 10-03 four-rank traces: each rank waits 1.0-1.7 ms a step for the last peer, and dgx4 is last in 39% of collectives (a ring-only lever, #616) |
| 4 | Stream boundaries' ramps and tails filled with known traffic | l2-prefetch-timing session | 0.47 ms (+1.6%, 76.6) | 0.23 ms (+0.6%, 61.0) | the bound: one ideal launch a layer against one for all layers |
| 5 | Routed experts restructured (continuous FC1-FC2, persistent or stream GEMMs, deeper stages, warp-specialized producer, fused SwiGLU) | this study | 0.29 ms (+1.0%, 76.1) | 0.27 ms (+0.7%, 61.1) | the bound: the chain less one ideal launch of the same bytes |
| 6 | LM heads at the attainable rate | unclaimed | 0.08 ms (+0.3%, 75.6) | 0.11 ms (+0.3%, 60.8) | layer tables: 41 µs (TP4) and 55 µs (TP3) over a head's bytes |

Reconciled with #598:
- Row 5 replaces this study's first estimate (+3.2%, from #598's rows 5 and 11). That
  estimate assumed the attainable 266.6 GB/s within a layer; an ideal one-layer launch
  reaches 247-254 GB/s.
- Row 4 is the rest of #598's launch fixed costs, and only prefetch traffic can claim it.
- Row 3 is #598's row 7. Rows 1 and 4 are most of #598's 3.4-3.7 ms of prefetch and
  consumer time.
- Row 2 overlaps the fusion survey's mHC and attention-front rows.

## Prototype 1: the stream grouped GEMM (dropped)

vLLM branch `c1-stream` (worktree ~/projects/vllm-c1-stream), not pushed:
`moe.grouped_fp8_fp4_gemm_stream`. v1 is 3360247fe and 06b74e7ff, and v2 adds 59b2ac540,
over next-candidate 97e204cd9. Below 16 rows, the plan's gate-up and down GEMMs use it in
place of the grid kernel. It was built before the bound existed, on the first estimate's
+3.2% ceiling.

| run | arm | TP4, 13 experts | TP4, 24 experts | TP3, 13 experts | bits |
|---|---|---:|---:|---:|---|
| stream-ab-1 (layer harness, six rounds) | v1: runs of `ceil(items / 48)` | −0.36% (645.6 → 652.0 µs a layer) | −1.77% (840.2 → 855.4) | −0.53% (847.5 → 854.2) | equal in every round |
| bound-stream-1 (the routed chain alone) | v2: every SM streams | 261.2 against 251.8 µs | 464.9 against 446.3 | 341.9 against 324.2 | - |
| stream-tests-1 | v1 | 115 tests passed, bit-identity test included | | | four PDL tests not run (conftest fixture) |

Why it loses:
- v1 balanced the items onto fewer SMs: 39 of 48 for FC1 at 13 experts. One SM keeps
  only two K blocks (16 KB) of weights in flight, too little for its share of the
  stream on fewer SMs.
- v2 streams on all 48 and is still slower. The likely reason, not traced: the grid
  kernel's CTA turnover costs less than the stream kernel's longer issue path (a
  division and shared-memory reads every K block, and the base words at each item),
  and the stream keeps no more bytes in flight. The bound leaves at most 6.8 µs a layer
  to win anyway.

- The grid kernel launches a CTA for each (tile, column block) item: 324 for FC1 at six
  rows, of which about 126 are live. Each CTA fills its pipeline, streams one
  `[128, K]` weight slab and drains. The CTAs past the last used tile exit after their
  PDL wait.
- The stream kernel launches 48 CTAs, one an SM. They split the live items into
  contiguous runs: v1 of `ceil(items / 48)` (3 FC1 items on each of 39 SMs at 13
  experts), v2 differing by at most one item over all 48. A CTA's pipeline runs through its items' K
  blocks without draining: an item's first stages load while the previous item's last
  ones multiply. Its weight slabs are consecutive in memory.
- Each item's tile multiplies its K blocks in order into a cleared FP32 accumulator,
  with the grid kernel's MMAs, operands and expanded scale words. Every output
  therefore has the grid kernel's bits. `test_stream_gemm_gives_the_grid_gemm_bits`
  compares both kernels bit for bit: TP4's and TP3's FC1 and FC2 shapes (TP4's FC2 has a
  64-wide K tail), at 1, 3, 6, 12 and 15 rows, padding rows included.
- What it aimed at: FC2's per-tile prologue and fill, the uneven last wave of both
  GEMMs, and the exiting CTAs.

Bundles: `stream-tests` runs the routed-expert tests, the PDL tests and the source
check (with conftest from stream-tests-2 on). `stream-ab` runs the layer harness on
next-candidate and the stream build in alternation, and compares bits and µs a layer.
`layer-stream` and `lifetimes-stream` trace its CTA timelines.

## The weight-stream variants (stages-ab)

With the stream GEMM losing, the other routed-pipeline levers were run as one variable
each against next-candidate. Run stages-ab-1 is the layer harness in rotation over five
builds, four rounds, every buffer's bits equal in every build. It gives µs saved a layer
(median of rounds; the rounds spread about ±10 µs a layer, so the resolution is about
±0.5%):

| arm (vLLM branch) | TP4, 13 experts | TP4, 24 experts | TP3, 13 experts |
|---|---:|---:|---:|
| the grid GEMMs at four stages in decode (`c1-stages4` 4d77b5e71) | −0.7 µs (−0.11%) | +1.8 (+0.22%) | −0.9 (−0.11%) |
| base words loaded after the first stages are issued (`c1-prologue` 4981583ba) | −0.2 (−0.02%) | +4.9 (+0.59%) | +0.5 (+0.05%) |
| the stream GEMM on all 48 SMs (`c1-stream` 59b2ac540) | −4.2 (−0.65%) | −15.3 (−1.81%) | −15.7 (−1.87%) |
| that at four stages (`c1-stream-s4` 8023b0d85) | −2.3 (−0.36%) | −13.1 (−1.56%) | −8.5 (−1.02%) |

None beats next-candidate beyond the run's resolution, which is what the bound predicts.
A fourth stage keeps three K blocks in flight an SM and gains nothing, so the grid
kernel's stream is not short of bytes in flight. The prologue reorder is within noise
too.

## Recommendation for the release after next

- **Take no routed-expert pipeline change.** The chain is within 1-7 µs a layer of an
  ideal stream (at most +1% at c1), and none of the five variants measured above the
  noise.
- **Spend the c1 effort where the measured non-byte time is:**
  1. **The L2 prefetch schedule** (#615): already prototyped, −1.5% at c1. The bound
     adds a reason: an ideal stream pays 5-17 µs of ramp and tail at each layer's MoE
     boundary, and prefetch traffic known in advance is what can fill it.
  2. **The latency-bound small kernels:** 34 µs a layer at either rank, at most 1.3 ms a
     step. They are mHC's non-byte time, sparse MLA, the norm and cast, dispatch and
     SwiGLU, and the fusion survey's attention-front and mHC boundary fusions target
     them.
  3. **The all-reduce's late peers** on the ring (dgx4 last in 39% of collectives): a
     cluster study, at most +3.2%.
- **For the 100 tok/s question this confirms #598 from the other side.** The weight
  stream itself is at the attainable rate, inside the kernels as well as overall. A
  step only gets shorter through fewer bytes (the exact byte cuts, a stronger drafter's
  tokens a step) or through less non-byte time between the streams.
- **IKET's per-rank decode-step tables** (#610's step bundles at TP4 and TP3 rank
  shapes, on dgx3) are still running. They cover the per-step kernels this harness
  leaves out: router, metadata, sampler, drafter. When they land, they attribute the
  remaining part of #616's 6.5 ms grid-to-serving gap.

## Lever 2: the latency-bound small kernels, starting with the mHC boundary

The coordinator then gave this study lever 2 (row 2 above), which nobody owned.

**Bounds, from the layer tables and the production traces.** The mHC boundary is the
largest small-kernel family:
- In production (TP4, c1 greedy, the ports build before PDL), `project_streams` takes
  1.16 ms a step and `finalize` 0.60 ms: 14.4 and 7.5 µs a boundary, 80 boundaries,
  5.0% of the step.
- In the harness with PDL, at either rank (mHC is replicated), it is 11.9 and 5.2 µs a
  boundary: 1.37 ms a step.
- The attention front is next: RoPE 0.26 ms, the cache write 0.25 ms, the q and kv
  norms and casts 0.28 ms, metadata 0.31 ms a step.

A boundary moves about 2.6 MB:
- `fn`, FP32, 1.97 MB;
- the four streams read and written, about 0.5 MB;
- the collapse and the output.

At 266.6 GB/s that is 9.7 µs of its 17.1 µs. So 7.4 µs a boundary, 0.59 ms a step
(TP4 +2.0%, TP3 +1.6%), is non-byte time. Most of it is `finalize`'s 5.2 µs, six CTAs
that reduce the partials, run the Sinkhorn and normalize.

**Where `project_streams` waits** (IKET, TP4): 51% of a warp's life is the `fn` loads,
15% the TF32 MMAs, 11% a block sync. It reads `fn` at about 200 GB/s across its 40 CTAs.

**Two prototypes on `fn`'s loads, bit-identical** (vLLM branches from next-candidate
c579b3fe1). Runs fusion-ab-2 and fusion2-ab-1: the layer harness with all-reduce stand-ins
(`--ar`), four rounds, at c1 (6 rows, 3 live), c8 (24 rows) and c16 (48 rows), every
buffer's bits equal:

| arm | TP4 c1 | TP4 c8 | TP4 c16 | TP3 c1 | TP3 c8 | TP3 c16 | tests |
|---|---:|---:|---:|---:|---:|---:|---|
| A1: every stream's `fn` loaded at once, before the first GEMM (`c1-fusion` f17fb3646) | −0.82% | −0.51% | −0.44% | −0.75% | −0.56% | −0.12% | 106 passed, kernels source-checked |
| A2: the next stream's `fn` loaded during the current stream's GEMMs (`c1-fusion2` b84e421b2) | −0.42% | −0.19% | −0.06% | +0.21% | −0.17% | −0.08% | 106 passed |

Neither helps; A1 is a little slower, probably from its scalar loads and 96 extra
registers a thread. The `fn`
read is bound by DRAM, not by serialized round trips: about 200 GB/s over 40 SMs is
already near the rate. Both are dropped.

**What is left at the boundary** is `finalize`'s latency and the short serial phases
around it: about 7 µs a boundary of non-byte time, including about 3 µs of
`project_streams`' own GEMM and syncs.

The next fusion to take is `finalize` folded into `project_streams`:
- Each of the 40 CTAs arrives on a counter once its partials are written. The 40 fit
  one an SM, so all are resident.
- The last arrival reduces the partials in today's order and runs the Sinkhorn.
- All 40 then normalize their own 128 columns and write the output and its cast.
- Every wait has a deadline that traps.

It saves the launch and part of the serial phases, about 3-4 µs a boundary: 0.24-0.32 ms
a step, +0.8-1.1%. The risk is low: the same operations in the same order.

The attention front's horizontal fusions follow at about the same size, about 8 µs a
layer: both RoPEs with the cache write in one kernel (+0.3% for the RoPE pair alone),
and both norms with their casts in one. The norms are TileKernels', so a bit-exact port
comes first.

## Runs on dgx1

dgx1, 2026-10-26, image `spark-ds41f:19f2c20ed4d6-next-6`, `scripts/lab.py kernel-local`
from `~/spark-lab/agents/c1-iket/spark-ds41f`, outputs in `~/spark-lab/agents/c1-iket/out/`.
`run_queue.sh` ran the bundles in order and stopped at a failed smoke run. Receipts
(`output.txt`, `verdict.json`, and for the traces the derived tables) are in `runs/`.

| run | started (CET) | bundle | result |
|---|---|---|---|
| smoke-1 | 13:50 | smoke | FAIL in 4 s: the harness read its default routes at import |
| smoke-2 | 13:58 | smoke | pass, 171 s; the lifetimes trace was empty (no kernel instrumented) |
| stream-tests-1 | 14:01 | stream-tests (v1) | 115 passed, bit-identity test included; four PDL layer-chain tests lacked conftest's fixture |
| layer-1 | 14:02 | layer | pass, 572 s: the TP4 and TP3 tables |
| smoke-3 | 14:11 | smoke | pass, 113 s; the lifetimes held the combine alone |
| stream-ab-1 | 14:12 | stream-ab (v1) | pass, 453 s: bits equal, v1 0.4-1.8% slower |
| bound-1 | 14:13 | bound | FAIL in 46 s: inputs cast after the kernel cache froze |
| layer-stream-1 | 14:21 | layer-stream (v1) | pass, 417 s |
| lifetimes-1 | 14:22 | lifetimes | pass, 255 s |
| layer-live6-1 | 14:29 | layer-live6 | pass, 302 s: TP4 at 24 experts |
| smoke-4 | 14:29 | smoke | pass, 115 s |
| bound-stream-1 | 14:33 | bound-stream (v2) | pass, 50 s |
| layer-base-1 | 14:38 | layer-base | pass, 78 s: next-candidate's digests equal the IKET build's at both ranks |
| bound-2 | 14:40 | bound | pass, 41 s: the bound |
| lifetimes-stream-1 | 14:41 | lifetimes-stream (v2) | pass, 255 s |
| stages-ab-1 | 14:43 | stages-ab | pass, 730 s: no arm above noise, bits equal |
| stream-tests-2 | 14:47 | stream-tests (v2) | 119 passed, every launched kernel source-checked |
| smoke-5 | 15:29 | smoke | pass, 116 s |
| fusion-tests-1 | 15:33 | fusion-tests (A1) | 106 passed, every launched kernel source-checked |
| fusion-ab-1 | 15:34 | fusion-ab (A1) | FAIL in 95 s: at 48 rows sparse MLA takes the extend route, which the harness did not run |
| smoke-6 | 15:40 | smoke | pass, 117 s |
| fusion-ab-2 | 15:43 | fusion-ab (A1) | pass, 577 s: bits equal, 0.1-0.8% slower |
| smoke-7 | 16:04 | smoke | pass, 115 s |
| fusion2-tests-1 | 16:06 | fusion2-tests (A2) | 106 passed |
| fusion2-ab-1 | 16:07 | fusion2-ab (A2) | pass, 615 s: bits equal, within ±0.4% |

## Files

| file | what |
|---|---|
| `harness/c1_layer.py` | the per-rank c1 layer harness |
| `harness/stream_bound.py` | the routed chain against an ideal read of the same bytes |
| `layer_table.py` | per-kernel cost, excess over bytes, IKET phases and waits, from a `layer` run |
| `cta_timeline.py` | SM occupancy and weight stream from run-iket's warp lifetimes (qualitative under the profiler) |
| `step_shares.py`, `step_shares.json` | kernel families' shares of the TP4 production step |
| `routed_chain.py`, `routed_chain.json` | the routed chain in production; FC2's serving fit |
| `levers.py`, `levers.json` | ceilings in tok/s from the measured bound and tables |
| `ab.py` | base against prototype builds on the harness, in rotation |
| `make_bundles.py`, `bundles/` | the kernel-lab bundles (IKET experiment's mechanics) |
| `run_queue.sh` | the node-side driver: the bundles one after another |
