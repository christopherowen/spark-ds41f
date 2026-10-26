# L2 prefetch schedule: windows that land before their consumers

Decode prefetches each layer's weights into L2 ahead of the GEMMs that read them
(`vllm/models/deepseek_v4_1/l2_prefetch.py`): three windows a layer, WO, FFN and NEXT,
filled by one side stream at grid 2. The windows are worth about 5% of a single-stream
step (experiments/2026-10-06-prefetch-fill-rate). The moe-overlap experiment
(2026-10-25, `l2_windows.py`) found their timing to be the largest L2 loss left:
consumers that start while their window still fills cost 1.67 ms a c1 step.

This experiment asks where that loss comes from, layer by layer, what a different issue
point could recover, and measures two prototypes on next-candidate with PDL (vLLM
1bc005688).

**Status:** RESULTS PENDING (dgx4 runs below). Branches: vLLM `l2-prefetch-schedule`
(local, on the shared object store, as `pdl-next`), this branch `l2-prefetch-schedule`.

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
| S3 WO after the index query (own layers) | 0.32% | 0.09% | the top-k chain and sparse MLA, 49 us at c1 |
| S3b an IDX window in WO's old place (own layers) | 0.25% | 0.05% | the compression stretch before the indexer's Q-B, where the lost WO fill runs now |
| S4a NEXT between the Engram projection and its all-gather | 0.17% | 0.04% | the all-gather and gate, 34 us |
| S4b an ENGRAM window in NEXT's place before an Engram layer | 0.46% | 0.08% | the all-reduce, mHC post, row wait and all-gather before the projection, 104 us |
| S6 the rest of Q-B in NEXT's budget | 0.76% | 0.13% | the same idle time as S1 (one or the other) |

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

| Commit | What |
| --- | --- |
| 5816ad749 | the schedule benchmark (below); no package file |
| 980aed9a4 | **chained** (S1, S2): WO issued right after the fused Q-A/KV in a layer without an indexer, so its fill queues behind NEXT's; FFN issued between WO-A and WO-B (`TileLangWOProjection(after_a=...)`) |
| 9e6d9d78a | **typed** (S3, S3b, S4a, S4b): an owner layer issues WO after the index query and an IDX window (compressor, indexer wk, wq_b, head weights) after Q-B; before an Engram layer the MoE hook issues an ENGRAM window (the projection's first 20 MB) and the Engram layer issues NEXT between its projection's GEMM and all-gather |
| 95a067b0d | tests of every layer type's issue points and plans, on fake layers (`tests/models/test_deepseek_v4_1_l2_schedule.py`) |
| 71314ad6e | benchmark fixes found by reading it before its first run |

Risks the traces cannot settle, for the cluster A/B:
- S1 starts the WO fill while Q-B's 8.8 MB of NEXT lines are unread. Under LRU the fill
  evicts the consumed mHC and Q-A/KV lines first; if GB10's L2 does not, Q-B slows.
- S3 moves the owner layers' WO fill from beside the compressor and indexer projections
  (DRAM-bound, so the fill costs nothing there) onto the top-k chain, which slows beside a
  fill as sparse MLA does (11.3 to 14.8 us). At +20-30% on a 50 us chain that is 10-15 us
  of the 26 us a layer S3 recovers.
- S1 and S2 move fill time off sparse MLA and the attention all-reduce, which both slow
  beside a fill: a gain the bounds do not count, and dgx4's spin stand-ins cannot show.

Cache hints only: no kernel, buffer or value changes, so every output is bit-identical by
construction; the benchmark and the cluster gate check it.

## dgx4: the schedules on forty layers at kernel shapes

RESULTS PENDING.

## Cluster A/B (specs, not run)

RESULTS PENDING.

## Files

- [traces.py](traces.py), [schedule_bounds.py](schedule_bounds.py),
  [schedule_bounds.json](schedule_bounds.json): the trace analysis.
- [make_bundles.py](make_bundles.py): overlays, bundles, arms and specs from the vLLM branch.
- [overlay/](overlay), [overlay-chained/](overlay-chained), [overlay-typed/](overlay-typed):
  next-candidate's 66 package files over the image's tree, and the files each prototype
  changes from it. [bench/](bench): the benchmark and its chain.
- [bundles/](bundles): kernel-lab bundles for dgx4.
- `{head,chained,typed}-{tp4,tp3}[-dev].json`, `prof-*-tp4.json`: serving arms.
- [screen-tp4.json](screen-tp4.json), [screen-tp3.json](screen-tp3.json),
  [profile-tp4.json](profile-tp4.json): cluster specs.
- [profile_decode.py](profile_decode.py): request-inventory's decode profiler, which the
  profile spec runs.
