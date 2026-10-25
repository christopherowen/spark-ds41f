# PDL along the DeepSeek V4.1 decode chain

Owner, 2026-10-25: are PDL, L2 eviction hints and kernel timings being investigated before
fusion? The plan is staged: small-kernel cleanup (the kernel-audit session), `evict_first` and
the act-quant fold (moe-overlap), then fusions ranked by ceiling (fusion-survey). PDL had only
been bounded inside the routed MoE chain (at most 0.6% at c1). This experiment bounds it across
the whole decode step from the critical path, and prototypes it.

Rules: bound each idea's gain from traces before building it; bit-identical outputs; correctness
before speed; no fallback paths.

Status: **analysis done, prototype committed, not run.** No GPU or cluster time was used (a
release window holds the cluster). The lab specs below are ready to queue once this branch is
merged.

## The short answer

- **A decode step is one chain.** Apart from the shared expert beside the routed experts and the
  L2 prefetch side stream, every kernel depends on the one before it. A one-stream greedy step
  (34.33 ms) is 32.64 ms of kernel time on the critical path and 1.69 ms (4.9%) of gaps.
- **The gaps are graph launch latency, not cross-stream waits.** Consecutive kernels of one
  capture stream inside a CUDA graph start a median 0.51 us apart, 1,420 times a step: 3.70% of
  a c1 greedy step, 4.79% c1 sampled, 1.71% at eight mixed streams. Cross-stream forks and joins
  are 0.06% or less, and the L2 prefetch's join never binds. Host-side and eager gaps are
  1.3% at most (with Python stacks on, an upper bound).
- **The first numbers' "busiest stream" was the prefetch stream.** The CUDA graph executor
  spreads the target forward's single capture stream over about 80 internal streams (it changes
  stream at every fork and join), so per-stream statistics cut the main chain into pieces. The
  busiest internal stream (stream 16, 0.9% of the window in gaps under 5 us, median 1.6 us)
  holds the L2 prefetch side chain (1,397 prefetch kernels in the c1 window) and the eager
  metadata kernels, off the critical path.
- **PDL's ceilings** (c1 greedy / c1 sampled / c8 mixed, share of the step):

  | Mechanism | Ceiling | What bounds it |
  | --- | ---: | --- |
  | Same-stream boundaries in the graphs (all) | 3.70 / 4.79 / 1.71% | measured gaps; PDL leaves the wait's release latency |
  | ... of which the prototype chains | 3.36 / 4.36 / 1.50% | receiving kernel begins with `pdl.chain()` or is TileKernels' |
  | Eager boundaries the prototype chains | 0.40 / 0.50 / 0.20% | measured; host-paced in part |
  | Weight prologue before the wait (not built) | < 0.80 / 0.99 / 0.43% | model: 1 us a weight GEMM boundary, capped by the predecessor; [the windows](#the-l2-weight-prefetch-windows) lower it |
  | Restructuring cross-stream joins | <= 0.21 / 0.15 / 0.08% | the shared expert's fork, SM contention and join |
  | Removing tiny kernels | (kernel-audit) | 790 kernels under 5 us on the c1 path, 1.81 ms (5.3%) |

  The gaps are an upper bound for PDL alone: with PDL the next kernel's CTAs wait at
  `griddepcontrol.wait`, so the boundary costs the wait's release instead of a launch. What part
  of the 0.51 us that is on GB10 is the first thing the lab measures
  ([benchmark_deepseek_v41_pdl_mechanism](#benchmarks)).
- **The L2 prefetch windows are late, and neither PDL nor an earlier issue fixes that**
  ([below](#the-l2-weight-prefetch-windows)). 77-87% of window instances still run when their
  first consumer starts; the consumers' time above their L2 speed is 4.67% of a c1 greedy step
  on the critical path. Issuing FFN and NEXT earlier inside their windows is bounded at 0.30%,
  and a PDL prologue prefetch at 0.01% (window consumers) and 0.08% (weight GEMMs in no window).
  The limit is DRAM time where the windows sit, not launch order. The prefetch join never
  binds; the forks cost about 0.04%.
- **Built:** every kernel on the layer chain begins with `pdl.chain()`, TileKernels' casts and
  norms take DeepSeek's PDL switch, the warp-specialized GEMMs keep their launch (vLLM `pdl`
  0b7952de4 and 75e6d02c7 over 26382f5cf). Scheduling only: the tests require identical bits.

## How the critical path was computed

[critical_path.py](critical_path.py) reads request-inventory-5's rank-0 traces (the ports build
of the sixth serving check, vLLM 9554a406c's package; the layers are 26382f5cf's) and writes
[critical_path.json](critical_path.json):

```sh
python3 -I experiments/2026-10-25-pdl/critical_path.py results/private/determinism/request-inventory-5/ports \
  --layer-sequence experiments/2026-10-25-pdl/layer-c1.json
```

- **Steps** as the fusion survey takes them: one target forward plus the drafter forward after
  it, from one target `vocab_embedding_kernel` to the next; the first step of a window (the
  prefill), sampled's host stall at step 24 and mixed-c8's steps outside the full batch left
  out. 11 greedy steps of 34.33 ms, 23 sampled of 27.84 ms, 17 mixed of 63.57 ms.
- **Capture streams.** In a graph replay a kernel's stream is the executor's internal stream.
  The capture streams come from the code instead: the shared expert (input cast, gate/up,
  SwiGLU, cast, down; forked after the FFN entry, joined by `moe_combine`, vLLM's aux stream),
  the L2 prefetch (forked by every window, joined once before the final mHC post) and the main
  stream (everything else).
- **The walk.** Backwards from the step's last kernel, each kernel's binding predecessor is the
  latest-ending of its dependencies: the previous kernel of its capture stream; for
  `moe_combine`, the shared expert's last kernel; for the shared branch's first kernel and each
  prefetch, the main kernel it forks from; for the kernel after the prefetch join, the last
  prefetch; for a launch's first kernel, the previous launch's last kernel. A kernel that starts
  more than 2 us after its dependency finished while another kernel ran is bound to that kernel
  (it waited for SMs, not for a launch). A kernel's share of the path starts where its
  predecessor ends, so path time plus gaps is the step.
- **Gaps** are classified by mechanism: graph chain (and "switch" where the executor changed
  stream), graph fork and join, prefetch join, resource wait, launch boundary, eager chain,
  memcpy; "host" where the launch call was issued after the predecessor had finished.

## Where a step's gaps are

| Gap on the critical path | c1 greedy | c1 sampled | c8 mixed |
| --- | ---: | ---: | ---: |
| Graph chain, same internal stream | 1,200 us, 3.49% (1,299 a step, median 0.51 us, p90 1.9) | 1,260 us, 4.52% | 1,006 us, 1.58% |
| Graph chain at an executor stream switch | 71 us, 0.21% (123, median 0.61) | 75 us, 0.27% | 82 us, 0.13% |
| Eager chain (launches issued in time) | 204 us, 0.59% (77, median 2.66) | 210 us, 0.75% | 285 us, 0.45% |
| Eager chain and memcpy, host late | 106 us, 0.31% | 64 us, 0.23% | 0 |
| Memcpy (device-side) | 61 us, 0.18% | 77 us, 0.28% | 95 us, 0.15% |
| Graph launch boundaries | 23 us, 0.07% | 21 us, 0.08% | 22 us, 0.04% |
| Shared expert fork | 20 us, 0.06% | 12 us, 0.04% | 10 us, 0.02% |
| Shared expert join, prefetch join | 0 | 0.1 us | 0.5 us |
| Total gaps | 1,686 us, 4.91% | 1,719 us, 6.17% | 1,501 us, 2.36% |

The largest graph boundaries (c1 greedy, gap time a step): the all-reduce into the next
sublayer's `mhc_project_streams` (80, median 1.15 us, 144 us), `mhc_project_streams` into
`mhc_finalize` (86, 0.80 us, 103 us), a cast into an MXFP8 GEMM (97, 0.51 us, 87 us), RoPE into
the cache writer (50, 1.18 us, 73 us), the sparse-MLA kernel into its split-KV combine and on
into WO-A (43 each, 0.9 us, 54 and 52 us). Gaps over 1 us are half the graph-chain time; they do not follow a
change of shared-memory carve-out (median 0.54 us when the next kernel's shared memory grows,
0.51 us when it shrinks).

Two effects of the streams the walk found:
- At c8, FC1 fills every SM (2,592 CTAs), so the shared expert's SwiGLU, cast and down GEMM
  start only as FC1's and FC2's CTAs drain. In 2.5 of 40 layers a step the down GEMM ends after
  FC2 and `moe_combine` waits for it: the path runs through the shared expert for 43 us a step,
  0.07%.
- At c1 the shared branch holds SMs the routed chain's next kernel needs in about 7 layers a step:
  the path runs through the shared expert there, 72 us with its fork (0.21%).

Kernel time on the path (c1 greedy): routed expert GEMMs 15.50 ms, dense GEMMs 5.27, one-shot
all-reduces 4.71, LM heads 2.06, other small kernels 2.14, mHC 1.60, sparse MLA 0.99, indexer
0.29.

## Ceilings per mechanism

1. **PDL on same-stream boundaries.** In stream capture a kernel launched with programmatic
   stream serialization gets a programmatic edge from its predecessor: it launches while the
   predecessor runs and its CTAs wait at `griddepcontrol.wait`. The ceiling is the graph chain's
   gaps, 3.70 / 4.79 / 1.71%; the boundaries into kernels the prototype chains carry 3.36 / 4.36
   / 1.50% (1,244 a c1 step). What stays unchained: the one-shot all-reduce and all-gather
   kernels (sparknet's; 57 us a c1 step, 0.17%), the warp-specialized BF16 GEMMs (router and other
   BF16 projections, 41 us, 0.12%), the Engram row wait (3 us) and the eager tail.
2. **PDL's prologue overlap.** `pdl.chain()` waits first, so nothing but the launch and the CTAs'
   start overlaps the predecessor. A weight GEMM could load its first weight tiles before the
   wait (weights never depend on the predecessor). Modelled at 1 us (a DRAM round trip and a
   first tile) per chained weight-GEMM boundary, capped by the predecessor's duration: +0.80 /
   0.99 / 0.43%. It is a model, and the window analysis below shows it is optimistic. Where a
   window has finished, the first tile comes from L2 (about half the latency). Where it is
   late, its own fill holds DRAM while the predecessor runs. Prefetching a consumer's weight in
   its prologue is bounded at 0.01% (window consumers) and 0.08% (weight GEMMs in no window).
   What remains is the first tile's latency, under about 0.4% at c1. Not built.
3. **Restructuring cross-stream joins.** The shared expert's fork, its SM contention and the
   combine's join: at most 0.21 / 0.15 / 0.08%. The join itself never binds at c1 and costs 0.07% at c8. Not worth a
   restructure now; at c16, where FC1 holds the GPU longer, the c8 join cost would grow and is
   worth re-measuring.
4. **Removing tiny kernels.** 790 kernels under 5 us run on the c1 path (1.81 ms, 5.3%), with a
   gap before each. That is the kernel-audit session's work. Each kernel it removes also removes
   one PDL boundary, so the two gains overlap by about 0.5 us a removed kernel.

The ceiling is well above 0.5% at c1, so the prototype was built.

## The L2 weight prefetch windows

Owner's addition, 2026-10-25: coordinate the L2 weight prefetch with PDL. Prefetch the next
operation's weights in the overlap windows, communication included, so they are resident when
the next kernel starts; with PDL, the next kernel's prologue could issue or overlap those loads.
[prefetch_windows.py](prefetch_windows.py) places every `l2_prefetch_kernel` on the critical
path ([prefetch_windows.json](prefetch_windows.json); same traces and steps):

```sh
python3 -I experiments/2026-10-25-pdl/prefetch_windows.py results/private/determinism/request-inventory-5/ports
```

Each prefetch is placed by capture order (the graph node id): it is forked from the main
kernel captured just before it. `l2_prefetch.py`'s three windows a layer are:
- **WO**, forked after Q-B: WO-A and WO-B;
- **FFN**, forked after WO-B, before the attention all-reduce: the FFN entry's mHC
  projection, the router, the shared expert's gate/up and down;
- **NEXT**, forked after the combine, before the MoE all-reduce: the next layer's mHC
  projection, fused Q-A/KV and Q-B.

A consumer's excess is its time above the median of its calls that started after their
prefetch had finished, counted up to its share of the path.

| c1 greedy (sampled, c8) | WO | FFN | NEXT |
| --- | ---: | ---: | ---: |
| Instances a step | 43 | 43 | 41 |
| Prefetch kernel, p50 | 92 us (81, 104) | 72 us (60, 88) | 92 us (79, 104) |
| Wait for the side stream at issue, p50 | 0.5 us | 0.5 us | 0.5 us |
| Side stream idle before the issue, p50 | 27 us | 6 us | 406 us (241, 946) |
| ... of it latency-bound path time just before the issue, p50 | 0 | 0 | 2.9 us |
| Instances still running at the first consumer's start | 77% (65, 77) | 86% (75, 78) | 86% (87, 87) |
| Lateness when late, p50 | 47 us | 27 us | 48 us |
| Consumers' excess on the path | 722 us, 2.10% | 389 us, 1.13% | 493 us, 1.44% |

- **The excess is 1.60 ms a c1 greedy step, 4.67%** (4.56% sampled, 1.39% at c8). The
  moe-overlap session found the same (1.67 ms a step over the whole trace). By consumer at c1:
  WO-B 529 us, fused Q-A/KV 269, the FFN entry's mHC projection 246, WO-A 193, the NEXT
  window's mHC projection 149, the router 123, Q-B 75. The shared expert's down projection is
  never late, and its gate/up mostly runs off the path.
- **The windows are shorter than the fills.** At one stream a window has 27-57 us of
  communication, or about 45 us of RoPE, cache writes, metadata and sparse MLA before WO-A,
  against 72-92 us of fill. Over a c1 step the fills take 10.8 ms (126 a step) and the
  latency-bound path time is 9.8 ms (all-reduce waits, small kernels, mHC, sparse MLA). The
  idle DRAM time is there, but not where the windows' consumers need it.
- **Joins and forks.** The prefetch join before the final mHC post never binds (0 to 0.12 us a
  step). The 127 forks a step are where the graph executor switches the main chain's stream:
  0.61 us median at a fork against 0.51 us elsewhere, about 13 us a step (0.04%). The busiest
  stream's 2.1 us gaps are between the prefetch kernels themselves, off the path.
- **Contention.** Latency-bound path kernels beside a fill take up to 1.0 ms a c1 step
  (2.9%) longer than calls of the same kernel and grid alone. That figure mixes layer types:
  the SWA-only layers' sparse MLA always runs beside the WO fill. The clean comparison is the
  indexer layers' sparse MLA: 14.8 us beside the fill against 11.3 us alone. The recorded A/B
  (the WO window alone nearly neutral) is the measured net.

Bounds:
- **Issuing FFN and NEXT earlier inside their windows** (never after their all-reduce): at most
  0.30% c1 greedy, 0.37% sampled, 0.27% c8. An even fill gives about a third of that (0.09 /
  0.15 / 0.08%). FFN is forked right after WO-B; NEXT right after FC2 and the 3 us combine. An
  earlier issue runs the fill beside WO-B or FC2, which already read DRAM at bandwidth: the
  bytes move from the consumer to that GEMM, and the time does not shrink. If DRAM were free
  during the side stream's idle time (the MoE, mostly), the bound would be 1.98 / 2.70 / 0.75%.
  That is unrealizable while the routed experts stream at bandwidth.
- **PDL-coordinated prefetch**, a consumer issuing its own weight's L2 prefetch in its prologue:
  at most 0.01% for window consumers, and 0.08 / 0.10 / 0.05% for the weight GEMMs in no window.
  With `pdl.chain()` a kernel's prologue overlaps only the kernel before it, usually a 1-2 us
  cast or norm. Where the window is late, its own prefetch already holds DRAM then. PDL cannot
  reach back the 45-100 us a fill needs. It does not replace the side-stream windows.
- **What would move the excess:**
  - fewer bytes per window, in consumer order, so what lands first is read first;
  - lines that survive the weight GEMMs between an earlier fill and its consumer: the
    moe-overlap session's cache-priority hints and persisting-L2 set-aside;
  - fills in latency-bound stretches no window uses now: the Engram layers' row wait and
    all-gather (about 145 us) before the Engram projection (`gemm_mxfp8` 200 CTAs, 180 us,
    two a step, in no window), and the indexer's Q-B (`gemm_mxfp8` 64 CTAs, about 10 a c1 step
    on the path, 384 us, in no window).

  Together they are bounded by the 4.67% excess pool and by L2's 24 MB.

## The prototype (vLLM `pdl`, two commits over 26382f5cf)

`vllm/models/deepseek_v4_1/tilelang/pdl.py`:
- `chain()`: `T.pdl_sync()` then `T.pdl_trigger()`, the first statement of a kernel. The wait
  comes before any global access, so a kernel reads exactly what a serialized launch would read
  and stores in the same order; the trigger after it lets the next kernel launch while this one
  runs, so at most two kernels of the chain are resident. TileLang (b95ee4ff) marks a kernel that
  waits `has_cuda_pdl_sync` and its TVM runtime launches it with
  `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`.
- `tile_kernels()`: TileKernels with `set_pdl(True)`, used by every call site (casts, norms, the
  top-k gate). DeepSeek's kernels place the wait first and the trigger after their stores.

Chained: mHC entry, finalize, post and collapse; the decode MXFP8 GEMM tiles, split-K partials
and reduces, WO-A's W8A16 kernels, the split-K BF16 GEMM; RoPE, cache writes, the sparse-MLA
metadata (decode, chunk, pages, requests, tokens), sparse MLA and its combine, the indexer and
compressor; MoE dispatch, grouped FP8 x FP4 GEMMs, SwiGLU cast and combine; routing masks and
partials; the shared expert's SwiGLU. Not chained: the warp-specialized GEMMs (`gemm_bf16`,
`gemm_bf16_two`, `gemm_bf16_partials`, `mxfp8_gemm`'s prefill tiles), whose producer warps'
TMA loads would have to follow the wait as well; `mxfp8_gemm_decode` always chains and
`mxfp8_gemm` never, so there is no switch.

Why the bits cannot change: every global load and store follows the wait, which returns only
when every kernel the launch depends on has completed and its writes are visible. TileLang drops
`__restrict__` from a kernel that waits (so no non-coherent load can move above it); that changes
load instructions, not arithmetic. The tests check the bits anyway.

## Tests and benchmarks

In vLLM `pdl` (75e6d02c7), run on GB10 by the bundles below; none has run yet.

- `tests/kernels/test_deepseek_v41_tilelang_pdl.py`:
  - a reader that loads before its wait, after a producer that spins tens of microseconds,
    must read the poisoned buffer in some CUDA-graph replays (otherwise the edges are not
    programmatic and PDL does nothing); a reader that waits first never does, in 20 replays;
  - the layer chain (below) at 6, 24 and 48 rows, every intermediate buffer set to NaN before
    each of 20 graph replays, gives its eager digests every time;
  - TileKernels' switch is on.
- `tests/kernels/deepseek_v41_pdl_sources.py`, a pytest plugin: every kernel a session launches
  is checked in its CUDA source. A chain kernel calls `cudaGridDependencySynchronize()` outside
  any block before any statement naming one of its pointer parameters, then
  `cudaTriggerProgrammaticLaunchCompletion()`; a warp-specialized GEMM has no PDL; any other
  kernel that waits does so first. The bundle runs it over the TileLang kernel tests of every
  changed module, which also check each kernel's bits against its reference on the PDL build.
- `tests/kernels/deepseek_v41_pdl_chains.py`: one decode layer's kernel chain at TP4's shapes
  (mHC entries, fused Q-A/KV, q norm, Q-B, WO-A split-K and reduce-cast, WO-B, the shared expert
  on its own stream beside the routed experts, the combine waiting for it): 23 kernels, random
  weights and fixed routes by seed, only APIs that predate PDL, so it runs on either build.
- `benchmarks/kernels/benchmark_deepseek_v41_pdl_chain.py`: that chain, four layers a graph, at
  6, 24 and 48 rows, warm and L2-cold, with every written buffer's digest.
- <a id="benchmarks"></a>`benchmarks/kernels/benchmark_deepseek_v41_pdl_mechanism.py`: synthetic
  dependent kernels (1 to 192 CTAs, no work to about 10 us) in four launch modes (plain, wait
  only, `chain()`, TileKernels' late trigger): the per-boundary saving on GB10; a fork/join like
  `moe_combine`'s; and [layer-c1.json](layer-c1.json), target layer 10's critical path from the c1
  greedy trace (28 kernels, 536 us, 25 chained), replayed as spins of the traced CTAs, threads and
  durations, plain and chained.

Checked here: `ruff` 0.14.0 format and check on every changed file; the source parser on a
synthetic CUDA source (it flags an early read, a wait inside a branch and a chain kernel that
never waits); `lab.py run --dry-run` of the three specs.

## Lab specs

Regenerate the bundles and arms from the vLLM branch with
`python3 experiments/2026-10-25-pdl/make_bundles.py ~/projects/vllm-pdl 26382f5cf pdl`.
Image: candidate 6 (`spark-ds41f:19f2c20ed4d6-next-6`, whose package files are 26382f5cf's).

1. [kernel.json](kernel.json), the cluster stopped, three nodes at once:
   - dgx2 [bundles/pdl-tests](bundles/pdl-tests): the PDL tests, then the 11 changed modules'
     kernel tests with the source check. Pass: exit 0, no violations.
   - dgx3 [bundles/pdl-chain](bundles/pdl-chain): [pdl_chain_ab.py](pdl_chain_ab.py) runs the
     chain benchmark on the image's build and on a tree of links to it with the branch's 14
     package files copied in, three rounds alternating. Pass: every digest equal across builds,
     replays and rounds; the line per shape gives the time saved per layer.
   - dgx4 [bundles/pdl-mechanism](bundles/pdl-mechanism): the synthetic benchmark (a status
     bundle; its numbers are the result).
2. [screen.json](screen.json): the determinism gate on [pdl-tp4.json](pdl-tp4.json) (candidate 6
   with [overlay/](overlay) mounted), whose hashes must be candidate 6's (log probabilities
   `3eab5112` / `db25fde3`, text `c9cf9dbe` / `c6bb4f4e`), with sampled and controls digests,
   16/8/4 distinct streams and the startup lines; then `screen` and `single-stream` on
   [pdl-tp4-dev.json](pdl-tp4-dev.json), candidate 6's dev arm, and the PDL arm again.
3. [profile.json](profile.json): candidate 6 and the PDL arm under the torch profiler
   ([prof-candidate6.json](prof-candidate6.json), [prof-pdl.json](prof-pdl.json)), greedy, sampled
   and mixed-c8 windows ([profile_decode.py](profile_decode.py) with `--scenarios`). Then
   `critical_path.py results/private/determinism/pdl-profile/<label> --scenarios greedy,sampled,mixed-c8
   --window ...` on both. Under PDL a kernel's CTAs start before its predecessor ends, so its gap
   becomes zero and the wait's release moves into its own time: compare step times and the
   chained kernels' path time, not the gap rows alone.

Acceptance, before running: bits identical everywhere (kernel tests, chain digests, the
determinism gate's hashes); the chain benchmark and the screen's c1 decode step faster by at
least 0.5%, no suite row slower beyond the run's own repeat. The specs need this branch merged
(the lab syncs node checkouts to origin/main): after the release window, on the owner's word.

## Interactions with the other sessions

- **kernel-audit** (vLLM `kernel-audit`): it merges `metadata_requests` and `metadata_tokens`
  into one launch, the largest eager boundaries here (136 us a c1 step, 0.40%), and moves the mHC
  entry's fn tile loads. Its `mhc.py` and `metadata.py` edits and this branch's one-line
  `pdl.chain()` insertions touch the same kernels: a textual merge, and its merged metadata
  kernel needs `pdl.chain()` too. Every kernel it removes takes a PDL boundary with it.
- **moe-overlap** (vLLM `moe-overlap`): `evict_first` on the routed experts' weights is
  independent of PDL. Its cache-priority hints and the persisting-L2 set-aside are what could
  let a fill start earlier, across a weight GEMM, and keep its lines; the windows' lateness
  (above) sizes that: 77-87% of instances late by a median 27-48 us, 4.67% of the step in
  excess at c1. The act-quant fold into `mhc_finalize` removes the cast between finalize
  and the projections (two boundaries become one, about 0.5 us a sublayer saved by either); the
  new finalize keeps `pdl.chain()`.
- **fusion-survey**: its "PDL across the decode graphs" ceiling (1.9 / 2.4 / 1.1%) counted gaps up
  to 1.5 us inside the layers plus `min(predecessor, GEMM)` for weight GEMMs. Measured on the
  critical path the boundary gaps alone are 3.7 / 4.8 / 1.7% (half of them over 1 us), and the
  weight prologue is modelled separately (0.8 / 1.0 / 0.4%). Every fusion that removes a
  boundary also removes that boundary's PDL gain: the ceilings overlap.

## Follow-ups (not built)

- sparknet's one-shot all-reduce and all-gather kernels with `T.pdl_sync()` first (57 us a c1
  step, 0.17%): a sparknet change, released and pinned on its own.
- The warp-specialized BF16 GEMMs: PDL needs the wait placed before the producer warps' TMA
  loads, checked in the generated source (41 us, 0.12%).
- The weight prologue (model: 0.8% at c1), after the lab has measured what the wait costs.
- Chaining the eager tail (sampling, drafter glue): after the V2 runner's glue settles.

## Files

- [critical_path.py](critical_path.py), [critical_path.json](critical_path.json),
  [layer-c1.json](layer-c1.json): the analysis and its output.
- [prefetch_windows.py](prefetch_windows.py), [prefetch_windows.json](prefetch_windows.json):
  the L2 prefetch windows on the critical path.
- [make_bundles.py](make_bundles.py), [run_steps.py](run_steps.py),
  [pdl_chain_ab.py](pdl_chain_ab.py), [bundles/](bundles), [overlay/](overlay): the bundles and
  the serving overlay, generated from vLLM `pdl`.
- [pdl-tp4.json](pdl-tp4.json), [pdl-tp4-dev.json](pdl-tp4-dev.json),
  [prof-candidate6.json](prof-candidate6.json), [prof-pdl.json](prof-pdl.json): arms.
- [kernel.json](kernel.json), [screen.json](screen.json), [profile.json](profile.json): specs.
- [profile_decode.py](profile_decode.py), [summarize_kernels.py](summarize_kernels.py),
  [analyze_costs.py](analyze_costs.py): request-inventory's profiling scripts (the first with
  `--scenarios`).
