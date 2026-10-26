# L2 prefetch schedule: windows that land before their consumers

Decode prefetches each layer's weights into L2 ahead of the GEMMs that read them
(`vllm/models/deepseek_v4_1/l2_prefetch.py`): three windows a layer, WO, FFN and NEXT,
filled by one side stream at grid 2. The windows are worth about 5% of a single-stream
step (experiments/2026-10-06-prefetch-fill-rate). The moe-overlap experiment
(2026-10-25, `l2_windows.py`) found their timing to be the largest L2 loss left:
consumers that start while their window still fills cost 1.67 ms a c1 step.

This experiment asks where that loss comes from, layer by layer, what a different issue
point could recover, and measures prototypes on next-candidate with PDL (vLLM
1bc005688).

**Result (dgx2, forty TP4 decode layers at one rank's shapes):**
- **Moving WO and FFN earlier loses 8-10%.** A window issued before the lines of the
  one ahead of it are read evicts those unread lines.
- **The owner and Engram layers' own points gain 1.5-1.6% at c1, 0.7% at c8 and 0.3% at
  c16**, bit-identical. Those are the layers whose lines were lost before use.
- TP3 shapes give 1.5 / 0.5 / 0.3%.
- The bench's spin stand-ins make the prefetch worth 3-4 times what it is on the ring
  (prefetch off: +19% here at c1, +5% on the ring), so expect about 0.4% at c1 and 0.2%
  at c8 and c16 there. The cluster A/B (specs below) decides.
- **On next-candidate c579b3fe1 (dgx1) the windows give the same**: TP4 -1.60 / -0.58 /
  -0.30%, TP3 -1.54 / -0.50 / -0.28%, bit-identical, tests passing.
- **Prefetch cannot fill the routed experts' ramp and tail** (dgx1). A fill there finds
  DRAM time for at most about 2 MB a layer, but the lines it lands are evicted before
  their consumers read them. Every split of NEXT into the routed chain loses at c8 and
  c16. At c1 the best gains 0.15% (TP3 on 1bc005688) and at most 0.08% on next-candidate,
  inside the rounds' spread.

Branches (vLLM ones local, on the shared object store, as `pdl-next`):
- vLLM `l2-prefetch-schedule`: the prototype, 650fe3679 on next-candidate 1bc005688;
- vLLM `l2-prefetch-pertype`: the prototype's package diff on next-candidate c579b3fe1
  (530205c6a; the benchmark d1bfa0372): **what the next candidate would take**;
- this branch, `l2-prefetch-schedule`.

## The loss, by layer type (cluster traces)

[schedule_bounds.py](schedule_bounds.py) reads request-inventory-5's rank-0 decode
traces (TP4 ring, ports build, before PDL) and splits each graph launch's main chain into
decoder layers by its anchor kernels in capture order ([traces.py](traces.py) is the PDL
experiment's loader). A layer's type follows its attention core:

| Type | Layers | Attention core between Q-B and WO-A |
| --- | --- | --- |
| swa | the other 30, and the drafter's 3 | RoPE, cache writes, metadata, sparse MLA: 44.5 us at c1 |
| idx | 24, 28, 32, 36 | the indexer's Q-B (DRAM), its head weights, top-k, sparse MLA |
| own | 2, 8, 14, 20 | the compressor's BF16 projection (64 us, DRAM), compression, the indexer, top-k, sparse MLA |
| eng | 1, 14 (before the mHC entry) | the Engram row wait and all-gather, its 6400 x 6144 projection (178 us), an all-gather |

```sh
python3 experiments/2026-10-26-l2-prefetch-schedule/schedule_bounds.py \
  results/private/determinism/request-inventory-5/ports
```

Each consumer's floor is the 25th percentile of its calls that started after their fill
ended, in layers whose lines survive. Excess over the floor a step
([schedule_bounds.json](schedule_bounds.json)):

| c1 greedy, 36.5 ms profiled step | WO | FFN | NEXT |
| --- | ---: | ---: | ---: |
| consumer started before its fill ended | 810 us | 517 us | 512 us |
| fill ended, lines lost before use | 115 us (own layers) | | 93 us (Engram layers) |
| fill ended, lines kept | 36 us | 414 us (shared down beside FC1) | 51 us |

At mixed-c8 (62.3 ms) the same rows are 188 / 140 / 109 us late and 54 / - / 27 us lost.
The late-or-lost pool is **2.05 ms a c1 step (5.6%)** and 0.52 ms at c8 (0.8%), the
moe-overlap and PDL experiments' 1.6-1.7 ms counted the same way. The shared expert's
down projection is never late: its excess is FC1's contention, not the schedule's.

What the layer types show (`by_layer`, c1 medians):
- **swa layers start WO-A 40-60 us and WO-B 10-50 us before the WO fill ends.** The fill
  needs about 92 us; the attention core is 44.5 us.
- **own layers end the WO fill 65-125 us before WO-A, and still read WO at DRAM speed**
  (WO-A 24-33 us against 12-15 us landed, WO-B 15-24 against 7.4). Between fill and use
  the compressor (10.5 MB BF16) and the indexer's Q-B (5.2 MB) stream through L2, and the
  oldest fill lines, WO-A's, go first. At c8 the attention core is longer and WO-A takes
  42-56 us.
- **idx layers end the WO fill 3-16 us before WO-A, and WO lands** (WO-B at its 7.4 us
  floor). The indexer's Q-B beside the fill takes 39-46 us instead of 28.
- **Engram layers lose NEXT entirely:** the next layer's fused Q-A/KV takes 39 us
  (its DRAM time) and Q-B 28 us, because the 39 MB Engram projection runs between
  NEXT's fill and them.
- **NEXT is late in every layer** by about 45 us at the mHC entry and 20 us at Q-A/KV;
  Q-B mostly starts after the fill ends but always reads the 1.7 MB the 20 MB budget
  leaves out.

## What an issue point can recover (bounds)

A consumer's late bytes come from DRAM either way; a different schedule gains only where
it moves fill bytes into time when DRAM is otherwise idle: latency-bound kernels, gaps,
collectives, and GEMMs that read L2. Each option's bound is the unused DRAM-idle time it
reaches times R / BW (fill 215 GB/s, consumer DRAM read 250 GB/s), capped by the excess of
the consumers it serves. Collectives count as idle, so the bounds do not charge a fill's
collision with the all-reduce (2026-10-06: the MoE all-reduce takes 26-31 us with the
prefetch off and 39-41 at grid 2).

| Option | c1 | c8 | What it reaches |
| --- | ---: | ---: | --- |
| S1 WO issued after the fused Q-A/KV (swa layers) | 1.29% | 0.23% | NEXT's fill ends about 8 us before Q-B; the norms, cast and Q-B (8.8 of 10.5 MB in L2) leave DRAM idle until WO is issued: 22.7 us a layer (p50) |
| S2 FFN issued after WO-A | 0.83% | 0.15% | WO-B's L2-fed time once WO's fill has ended, plus what S1 frees: 9.8 us a layer |
| S3 owner layers: an IDX window (compressor, indexer projections) after Q-B, WO right after the compressor behind it | 0.66% | 0.16% | an estimate, not a bound (below): IDX lands the compressor's first bytes and the indexer's 5.6 MB; WO's fill then has 88 us (c1) to 114 us (c8) to WO-A, enough for its 20 MB |
| S4a NEXT between the Engram projection and its all-gather | 0.17% | 0.04% | the all-gather and gate, 34 us |
| S4b an ENGRAM window in NEXT's place before an Engram layer | 0.46% | 0.08% | the all-reduce, mHC post, row wait and all-gather before the projection, 104 us |
| S6 the rest of Q-B in NEXT's budget | 0.76% | 0.13% | the same idle time as S1 (one or the other) |

S3 is accounted differently from the others. An owner layer's lines are lost, not late, so
a new point must land more than the roughly 13 of 20 MB that survive today. The first
point tried, WO after the index query, gives its fill only the 49 us top-k chain at c1:
about 10.5 MB, a loss against today (the first typed commit, 9e6d9d78a). Issued right after
the compressor, WO's fill starts once IDX has landed and reaches WO-A in time
(353085fb7). The S3 row is today's lost-line excess minus the bytes the new point leaves
unlanded, plus the compressor's and the indexer Q-B's own gains.

A fill-landing model on the same traces (fills progressing at one rate while DRAM is idle
and another beside DRAM-bound kernels) does not fit: the prefetch kernels run at about
210 GB/s whatever the main chain does (c1: 91.8 us for WO with 26.8 us of it beside
DRAM-bound kernels; c8 alike), so a fill kernel's end is when its requests were issued,
not when its lines landed. The bounds above use only the main chain's idle time and the
consumers' measured excess.

Options the brief named, bounded elsewhere or here:
- **Issuing FFN or NEXT earlier inside their windows:** at most 0.30% at c1 (the PDL
  experiment's `prefetch_windows.py`): NEXT already starts right after FC2 and FFN right
  after WO-B, which read DRAM at bandwidth. S2 reaches further only because WO-B reads
  L2 once WO has landed.
- **Splitting windows into fills placed in every latency-bound stretch:** the PDL
  experiment's 1.51% (FFN, NEXT) and 0.55% (WO) at c1 counted only latency-bound
  kernels. S1 and S2 reach L2-fed GEMM time that bound left out; the routed experts'
  latency-bound stretches (dispatch, casts, SwiGLU, combine, 3 us each) are shorter than
  a fill's launch and ramp.
- **Triggering the prefetch from the preceding kernel under PDL:** at most 0.01% for
  window consumers (PDL experiment): `pdl.chain()` overlaps only the 1-2 us kernel before
  a consumer.
- **Grid per window:** not boundable from traces. The fill's rate trades against the
  all-reduce and the latency-bound kernels it slows; grid 2 beat 4 by 1-1.5% on the ring
  (2026-10-06). The cluster specs below leave it at the recipe's 2.
- **Evict-first, pacing around the all-reduce, FFN/NEXT after their all-reduce:** dead
  ends (moe-overlap #587, 2026-10-06), not revisited.

## Prototypes (vLLM `l2-prefetch-schedule` on next-candidate 1bc005688)

| Commit | What | Outcome on dgx2 |
| --- | --- | --- |
| 5816ad749 | the schedule benchmark (below); no package file | |
| 980aed9a4 | **chained** (S1, S2): WO right after the fused Q-A/KV in a layer without an indexer, FFN between WO-A and WO-B | +9.8 / +8.4 / +5.4% at c1 / c8 / c16: dropped |
| 9e6d9d78a, 353085fb7 | **typed**: chained, plus an owner layer's IDX window after Q-B and WO after its compressor; before an Engram layer an ENGRAM window, and NEXT between the Engram projection's GEMM and all-gather | +8.1 / +7.9 / +5.1%: the chaining's loss; its own points pay (below) |
| 95a067b0d, 71314ad6e | tests of every layer type's issue points on fake layers; benchmark fixes | |
| 5e509b53a | **per-type**: typed without the chaining (WO and FFN back at their points) | -1.5 / -0.6 / -0.3% |
| d2a196527 | Engram's projection as a module helper, `project_rows` (an SP test calls `Engram.forward` on a fake module) | |
| **650fe3679** | **the prototype**: per-type with the owner layer's window before WO holding the compressor alone (COMP) | **-1.5 to -1.6 / -0.7 / -0.3%** |

The prototype's package diff against next-candidate is four files, +93 / -9 lines:
`attention.py` (WO's point and the COMP window), `nvidia/model.py` (the COMP and ENGRAM
plans, NEXT's hook on the Engram module), `common/engram.py` (the hook between the GEMM
and the all-gather), `l2_prefetch.py` (the windows' docstring). Cache hints only.

## dgx2: the schedules on forty layers at kernel shapes

The harness (vLLM `benchmarks/kernels/benchmark_deepseek_v41_l2_schedule.py` over
`tests/kernels/deepseek_v41_l2_schedule_chain.py`) builds the target model's forty
decoder layers at one rank's TP4 or TP3 shapes, each with its own weights, the Engram,
owner and indexer layers in their places:
- real kernels for every projection a window prefetches or competes with;
- spin kernels (`torch.cuda._sleep`, no memory traffic) for the latency-bound work around
  them (attention core, router top-k, all-reduces, Engram waits), each as long as rank
  0's median in request-inventory-5's traces;
- the model's own plans, budgets, grid 2 and side stream, issued at each schedule's
  points.

One CUDA graph per schedule; 8 rounds of 10 replays, the schedules' order rotating each
round. Before timing, every schedule's replay must give the eager run's bits in every
buffer the layers write.

```sh
scripts/lab.py kernel-local experiments/2026-10-26-l2-prefetch-schedule/bundles/schedule-tp4 --out OUT
python3 experiments/2026-10-26-l2-prefetch-schedule/harness_table.py OUT/output.txt --schedules pertype2
```

Runs (dgx2, image `spark-ds41f:19f2c20ed4d6-next-6`, 2026-10-26 13:50-14:30 CET; outputs in
[runs/](runs)):

| Run | What |
| --- | --- |
| smoke-1 | 6 rows, 2 rounds: head, chained, typed, off |
| tp4-1 | TP4: head, pertype, typed, chained, off |
| tp4-2, tp3-1 | TP4, TP3: head, pertype, own, eng, off (own and eng: pertype's owner and Engram points alone) |
| tp4-3 | TP4: adds pertype2 (the owner window of the compressor alone) |
| tp4-4, tp3-2 | TP4, TP3 on the prototype tree 650fe3679: head, pertype, pertype2, own, eng, off |
| tp4-5 | TP4: pertype3 (pertype2 with the indexer's Q-B and head weights prefetched right after the compressor, ahead of WO) |
| tests-head-2, tests-pertype-3, tests-pertype-4 | the prototype's test files on next-candidate and on the prototype |

**Bits:** every schedule's replay equals the eager run in every buffer, at every row count,
in every run (`"equal": true`).

**Step time against the head** (pertype2 is the prototype; each cell the run's median, the
rounds' range in brackets):

| | c1 (6 rows) | c8 (48 rows) | c16 (96 rows) |
| --- | ---: | ---: | ---: |
| TP4, tp4-4 | **-1.52%** [-1.56, -1.46], -397 us | **-0.66%** [-0.69, -0.63], -370 us | **-0.32%** [-0.35, -0.29], -280 us |
| TP4, tp4-3 | -1.61% [-1.68, -1.55] | -0.69% [-0.75, -0.67] | -0.31% [-0.34, -0.23] |
| TP3, tp3-2 | **-1.48%** [-1.58, -1.38], -481 us | **-0.52%** [-0.57, -0.48], -368 us | **-0.27%** [-0.30, -0.25], -292 us |
| chained (tp4-1) | +9.79% | +8.44% | +5.40% |
| prefetch off (tp4-4) | +18.91% | +8.89% | +4.72% |

Owner and Engram points apart (tp4-4): owner layers -0.92 / -0.29 / -0.17%, Engram layers
-0.55 / -0.30 / -0.13%; they add up.

pertype3 (tp4-5): -1.63 / -0.74 / -0.31% against the prototype's -1.52 / -0.68 / -0.31%
in the same run. It prefetches the indexer's Q-B and head weights right after the
compressor, ahead of WO. That is 0.1 point at c1, 0.06 at c8 and none at c16 for a third
window, a few hundredths of a percent on the ring at the harness's scale: not taken.

**Where the time goes** (tp4-4, c1, head to prototype, a step): the compressor -185 us
(87.9 to 39.8 us: COMP lands its first bytes, and WO's fill no longer runs beside it),
owner-layer WO-A -122 us (46.4 to 13.2), WO-B -33 us, the Engram projection -94 us (163 to
110 us), and the indexer's Q-B +48 us (22.9 to 34.8: WO's fill now runs beside it). The
other layers' consumers are unchanged within noise.

**Per-consumer L2-speed share** (the consumer GEMM's floor with its weight in L2, measured
alone, over its own time in the step; `harness_table.py`), head and prototype, TP4:

| Consumer | c1 head | c1 prototype | c8 head | c8 prototype | c16 head | c16 prototype |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention mHC | 0.44 | 0.46 | 0.61 | 0.60 | 0.69 | 0.69 |
| fused Q-A/KV | 0.64 | 0.68 | 0.71 | 0.77 | 0.46 | 0.47 |
| Q-B | 0.49 | 0.49 | 0.51 | 0.50 | 0.67 | 0.67 |
| WO-A | 0.31 | 0.35 | 0.58 | 0.62 | 0.91 | 0.96 |
| WO-B | 0.62 | 0.65 | 0.70 | 0.72 | 0.78 | 0.76 |
| FFN mHC | 0.57 | 0.56 | 0.79 | 0.79 | 0.77 | 0.78 |
| router | 0.34 | 0.34 | 0.88 | 0.86 | 0.28 | 0.28 |
| shared gate/up | 0.57 | 0.57 | 0.80 | 0.79 | 0.50 | 0.49 |
| shared down | 0.53 | 0.55 | 0.62 | 0.62 | 0.61 | 0.60 |
| compressor (COMP) | - | 0.26 | - | 0.35 | - | 0.61 |
| Engram projection (ENGRAM) | - | > 1 | - | > 1 | - | > 1 |

(The Engram projection's 39 MB cannot sit in L2, so its "floor" is its DRAM time and the
prefetched run beats it.)

The c1 IKET study ([#614](https://github.com/christopherowen/spark-ds41f/pull/614),
experiments/2026-10-26-c1-iket, one GB10, 6 rows, next-candidate with PDL, no prefetch)
gives these consumers' cold costs (TP4):

| Consumer | Cold cost | Time spent waiting on its own weight copies | This harness, head | This harness, prototype |
| --- | ---: | ---: | ---: | ---: |
| fused Q-A/KV | 39.6 us | 62% | 13.0 us (window-only layers) | 12.7 us |
| Q-B | 45.0 us | 57% | 19.6 us | 19.6 us |
| WO-A and its reduce | 40.2 us | 49% | 32.8 us (window-only), 46.4 us (owner layers) | 33.4 us, 13.2 us |
| WO-B | 46.6 us | 65% | 10.1 us (window-only), 29.3 us (owner layers) | 10.7 us, 18.7 us |

- Cold, they run at 78-87% of the 266.6 GB/s expert-stream rate.
- The owner layers' head WO-A at or above its cold cost is the lost-lines case: all of
  WO-A read from DRAM, beside the WO fill.
- The window-only layers sit between cold and L2 speed in both schedules. Under PDL a kernel's own time is its end less its predecessor's
end, so a consumer's share also carries its prologue; the shares compare schedules, not
absolute L2 speed.

**What the harness shows that the traces could not:** a window issued before the one ahead
of it has been read evicts that window's unread lines. The consumers' reads keep the lines
they read, so the unread ones are the oldest. Chaining WO behind NEXT took the window-only
layers' Q-B from 19.6 to 57.9 us, and chaining FFN behind WO took WO-B from 20.7 to 38 us.
The owner-layer IDX window lost its indexer lines the same way to the WO fill behind it.
Every point the prototype keeps starts a fill only once the window before it has been read,
or puts the first-read bytes first.

**How far the harness overstates:** with the prefetch off it runs 18.9% slower at c1, 8.9%
at c8 and 4.7% at c16. On the ring the prefetch was worth 5.0%, 2.8% and 3.1%
(2026-10-06). The spin stand-ins leave DRAM idle where real kernels contend, and there
is no all-reduce to collide with. Scaled by those ratios, the prototype's 1.5 / 0.7 /
0.3% would be about 0.4 / 0.2 / 0.2% on the ring.

**Tests on dgx2:** next-candidate's and the prototype's test files
(`tests-head-2`, `tests-pertype-4`):
- next-candidate: 152 passed, 7 failed;
- the prototype: 158 passed (the same 152 and its 6 new tests), the same 7 failed.

The 7 are the disk-Engram loader tests, which need io_uring; the kernel-local container
does not permit it ("io_uring initialization failed: Operation not permitted").

## dgx1: filling the routed chain's ramp and tail

The c1 IKET study (#614) finds DRAM under-used for about 15.5 us a layer around the
routed experts at c1: FC1's tail (7.1 us), the SwiGLU and gaps (3.8 us), FC2's ramp
(4.6 us). That is up to 4 MB of reads at the 266.6 GB/s expert-stream rate. Can the
prefetch use that time?

The harness's routed chain is the model's own: `_moe_run` is `MoeExpertsV41Plan.run`'s
launches (dispatch, FC1, SwiGLU, FC2, the shared expert on the aux stream, combine) with
the plan's kernels at its row count and a per-rank pool of 14 / 48 / 80 experts at
6 / 48 / 96 rows. Only the attention core, top-k, all-reduces and Engram waits are spins.
Two kinds of schedule, each on top of the prototype (pertype2), issued from the side
stream after a routed launch (`dispatch`, `fc1` after FC1, `fc2` after FC2):

- **junk:POINT:4**: a 4 MB fill of bytes nobody reads (a per-layer slice of a 160 MB
  buffer), to measure what a fill there costs;
- **split:POINT:MB**: NEXT's first 2, 4 or 6 MB issued there, the rest after the combine
  as now: the useful version.

Runs (dgx1, 2026-10-26 15:20-15:31 CET): `ramp-tp4-2`, `ramp-tp3-2`, on the prototype tree
650fe3679 over next-candidate 1bc005688. Every schedule's bits equal the eager run's.

**What a fill costs** (step time against pertype2, a step; free = the 4 MB less the time it
added at 266.6 GB/s, a layer):

| 4 MB junk | TP4 c1 | TP4 c8 | TP4 c16 | TP3 c1 | TP3 c8 | TP3 c16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| after dispatch (beside FC1) | +357 us, 1.6 MB free | +375, 1.5 | +322, 1.9 | +341, 1.7 | +484, 0.8 | +642, none |
| after FC1 (SwiGLU, FC2) | +256 us, **2.3 MB free** | +286, 2.1 | +289, 2.1 | +328, 1.8 | +387, 1.4 | +624, none |
| after FC2 (shared expert, combine) | +355 us, 1.6 MB free | +475, 0.8 | +469, 0.9 | +417, 1.2 | +461, 0.9 | +461, 0.9 |

Window consumers do not slow under junk; the cost is in the routed kernels the fill runs
beside (after dispatch FC1 +309 us at TP4 c1; after FC1 FC2 +177; after FC2 the combine,
+430 at c8). **So prefetch traffic can fill at most about 2 MB a layer of the ramp and
tail, 8 us of the 15.5, at TP4, and less at TP3.** If all of it came off NEXT's fill, the
harness would gain about 8.6 us a layer, 1.5% at c1, about 0.4% on the ring.

**What a split gains** (against pertype2; the rounds' range in brackets):

| NEXT's first bytes | TP4 c1 | TP4 c8 | TP4 c16 | TP3 c1 | TP3 c8 | TP3 c16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 MB after dispatch | +1.66% | +0.43% | +0.12% | +1.74% | +0.42% | +0.32% |
| 2 MB after FC1 | +1.41% | +0.31% | +0.08% | +1.54% | +0.37% | +0.31% |
| 2 MB after FC2 | -0.04% [-0.15, +0.04] | +0.30% | +0.08% | -0.04% [-0.14, +0.29] | +0.26% | +0.25% |
| 4 MB after FC2 | -0.02% [-0.08, +0.01] | +0.34% | +0.29% | -0.08% [-0.12, -0.04] | +0.31% | +0.22% |
| 6 MB after FC2 | -0.04% [-0.08, +0.60] | +0.79% | +0.44% | **-0.15%** [-0.23, -0.08] | +0.31% | +0.33% |
| 4 or 6 MB after dispatch or FC1 | +0.24 to +2.74% everywhere | | | | | |

The bytes landed in the routed chain are gone when NEXT's consumers read them. The next
layer's attention mHC, which reads NEXT's first bytes, slows from 412 to 1447 us a step
(TP4 c1, 2 MB after FC1) and to 749 us (2 MB after FC2); its fused Q-A/KV from 349 to
617 and 543 us at c8. Before FC2 the 21 MB expert stream passes through L2 between fill
and use. After FC2 the early lines are the oldest unread ones when the rest of NEXT
lands: the eviction order that sank chaining. Issued after FC2, the fill also slows
the combine (+254 us a step at TP4 c8).

**Ramp and tail are not prefetch's to fill.** There is DRAM time, at most 2 MB a layer,
but L2 cannot keep what lands there until it is read. The best schedule, 6 MB after FC2,
is 0.15% at TP3 c1 in the harness (about 0.04% on the ring) and costs 0.3% at c8 and c16.
Not prototyped in vLLM: a row-count-dependent issue point for 0.04% would be an arm of
its own. The ramp and tail have to shrink at their source: the routed GEMMs' own loads
(#614's continuous FC1-FC2 stream).

### On next-candidate c579b3fe1

Next-candidate moved on after 1bc005688. From 16 rows its routed grouped GEMMs trigger
their dependent's launch after their K loop (97e204cd9), which changes the c8 and c16
tail; the other commits are CED and the rejection sampler. None touches the per-type
windows' four files, which apply unchanged: vLLM `l2-prefetch-pertype` 530205c6a, and
the benchmark d1bfa0372. The bundles now build these trees (`make_bundles.py`), and dgx1
re-ran the tests and both sweeps on them (2026-10-26 15:36-15:49 CET, `next-*-1`).

**Tests** (`next-tests-head-1`, `next-tests-pertype-1`): next-candidate 152 passed;
with the windows 158 passed, its 6 new tests included. The same 7 disk-Engram loader
tests fail on both (io_uring, as above).

**Bits:** every schedule equals the eager run, at every row count, TP4 and TP3.

**The per-type windows against next-candidate** (step time, the rounds' range in
brackets):

| | c1 (6 rows) | c8 (48 rows) | c16 (96 rows) |
| --- | ---: | ---: | ---: |
| TP4, next-schedule-tp4-1 | **-1.60%** [-1.71, -1.51], -373 us | **-0.58%** [-0.61, -0.55], -325 us | **-0.30%** [-0.35, -0.29], -261 us |
| TP3, next-schedule-tp3-1 | **-1.54%** [-2.02, -1.45], -446 us | **-0.50%** [-0.55, -0.47], -348 us | **-0.28%** [-0.29, -0.23], -309 us |

**Ramp and tail on next-candidate:** the same picture. A 4 MB fill after FC1 hides
2.2 / 2.1 / 2.1 MB a layer at TP4 c1 / c8 / c16. No split gains: NEXT's first 2-6 MB after
FC2 is +0.01 to +0.11% at c1 and +0.09 to +0.80% at c8 and c16; at the other points
+0.06 to +2.95%. TP3: a 4 MB fill after FC1 hides 1.9 / 1.4 MB a layer at c1 / c8 and
nothing at c16. NEXT's first bytes after FC2 are -0.03 to -0.08% at c1 (rounds -0.14 to
+0.60), and +0.17 to +0.50% at c8 and c16; at the other points +0.24 to +2.45%.

## Cluster A/B (specs, not run)

Arms are candidate 6's recipe with next-candidate c579b3fe1 (`head-*.json`) and with
the per-type windows on it, 530205c6a (`pertype-*.json`), mounted over the image, at
tp4-ring and tp3-triangle; `-dev` adds the dev-mode RPC the suites use. (Until the
ramp/tail runs they were on 1bc005688 and 650fe3679.)
- [screen-tp4.json](screen-tp4.json): the determinism gate on head and pertype (the
  prototype's hashes, sampled and controls digests and 16/8/4 distinct streams must be the
  head's), then the screen and single-stream suites on pertype, head, pertype again.
- [screen-tp3.json](screen-tp3.json): the same on the triangle (8/4 distinct streams,
  `screen-tp3`).
- [profile-tp4.json](profile-tp4.json): both arms under the torch profiler, decode request
  types; then `schedule_bounds.py results/private/determinism/l2-schedule-profile/<label>`
  on each to see the owner and Engram layers' consumers on the ring.

```sh
scripts/lab.py run experiments/2026-10-26-l2-prefetch-schedule/screen-tp4.json --production-config config/cluster.json
scripts/lab.py run experiments/2026-10-26-l2-prefetch-schedule/screen-tp3.json --production-config config/cluster.json
scripts/lab.py run experiments/2026-10-26-l2-prefetch-schedule/profile-tp4.json --production-config config/cluster.json
```

They plan cleanly with `--dry-run` and need this branch on origin/main (the lab syncs node
checkouts to it). Acceptance: bits identical in the gate; c1 decode step faster on the
screen; no suite row slower beyond the run's own repeat.

## Files

- [traces.py](traces.py), [schedule_bounds.py](schedule_bounds.py),
  [schedule_bounds.json](schedule_bounds.json): the trace analysis.
- [make_bundles.py](make_bundles.py): overlays, bundles, arms and specs from the vLLM branch.
- [harness_table.py](harness_table.py): step times, consumer excess and L2-speed shares
  from a benchmark run's output; [runs/](runs): every dgx2 and dgx1 run's output and
  verdict.
- [harness_check.py](harness_check.py): the benchmark's bookkeeping checked on the CPU, no
  torch: the chain's own `_Layer.run()` with recording fakes, through the benchmark's
  `roles()`, `windows()` and `summarize()`. Per schedule it prints each window's fills and
  their anchors; every fill must find all its consumers.
  `python3 -I experiments/2026-10-26-l2-prefetch-schedule/harness_check.py VLLM_TREE`.
- [overlay/](overlay), [overlay-pertype/](overlay-pertype): next-candidate c579b3fe1's
  73 package files over the image's tree, and the 4 the per-type windows change from it.
  [bench/](bench): every test and benchmark file the vLLM branch adds or changes over the
  image's tree, at its path (the image's copies are the release's).
- [bundles/](bundles): kernel-lab bundles (`schedule-smoke`, `schedule-tp4`,
  `schedule-tp3`, `tests-head`, `tests-pertype`), run on dgx2 and dgx1. Runs before the
  `next-` ones used them as this branch had them at 4d5e8fc1 (next-candidate 1bc005688,
  66 files; the prototype 650fe3679).
- `{head,pertype}-{tp4,tp3}[-dev].json`, `prof-{head,pertype}-tp4.json`: serving arms.
  The chained and typed arms and overlays were removed once dgx2 measured them (they are
  in this branch's history at 26dbd21d).
- [screen-tp4.json](screen-tp4.json), [screen-tp3.json](screen-tp3.json),
  [profile-tp4.json](profile-tp4.json): cluster specs.
- [profile_decode.py](profile_decode.py): request-inventory's decode profiler, which the
  profile spec runs.
