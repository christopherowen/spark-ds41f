# PDL along the DeepSeek V4.1 decode chain

Owner, 2026-10-25: are PDL, L2 eviction hints and kernel timings being investigated before
fusion? The plan is staged: small-kernel cleanup (the kernel-audit session), `evict_first` and
the act-quant fold (moe-overlap), then fusions ranked by ceiling (fusion-survey). PDL had only
been bounded inside the routed MoE chain (at most 0.6% at c1). This experiment bounds it across
the whole decode step from the critical path, and prototypes it.

Rules: bound each idea's gain from traces before building it; bit-identical outputs; correctness
before speed; no fallback paths.

Status: **analysis done; the final head is vLLM `pdl-next` 1bc005688 on next-candidate
577091997; kernel checks run on dgx4** ([results](#results-on-dgx4-2026-10-25-cet),
[final head](#the-final-head-pdl-next-on-next-candidate-577091997)). Bits are identical and
the source check finds 0 violations. The c1 layer chain is 0.7-1.3% faster against
next-candidate (1.3-1.8% against candidate 6's build). The 48-row chain trends 0.4% slower, an
open finding. The serving screen and profile need the cluster and stay queued.

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
- **Measured on dgx4, kernel level (2026-10-25):**
  - outputs are bit-identical: kernel tests, layer-chain digests across builds, and replays;
  - the source check finds 0 violations;
  - the captured graphs carry programmatic edges;
  - a boundary costs 0.25-0.36 us less with `pdl.chain()`;
  - the traced c1 layer is 0.75-1.42% shorter (two runs). The real c1 layer chain is 1.81% warm
    and 1.30% cold faster on candidate 6's build. On next-candidate, whose finalize cast already
    removed three of the chain's boundaries, it is 0.67-1.31% warm and 0.97-1.12% cold faster;
  - at 24 rows the chain is within noise; at 48 rows PDL trends 0.4% slower (three runs; the
    mechanism is not yet tested; see [the final head](#the-final-head-pdl-next-on-next-candidate-577091997)).

  That is roughly 0.7-1.3% of a c1 step, if serving confirms it.
- **The L2 prefetch windows are late, and neither PDL nor an earlier issue fixes that**
  ([below](#the-l2-weight-prefetch-windows)). 77-87% of window instances still run when their
  first consumer starts; the consumers' time above their L2 speed is 4.67% of a c1 greedy step
  on the critical path. Issuing FFN and NEXT earlier inside their windows is bounded at 0.30%,
  and a PDL prologue prefetch at 0.01% (window consumers) and 0.08% (weight GEMMs in no window).
  The limit is DRAM time where the windows sit, not launch order. Splitting each window into
  fills placed in the latency-bound stretches before it is bounded at 1.51% at c1 (2.04%
  sampled, 0.60% c8, plus 0.55% for WO); evict-last lines survive the GEMMs in between on GB10
  (moe-overlap's probe). The prefetch join never binds; the forks cost about 0.04%.
- **Built:** every kernel on the layer chain begins with `pdl.chain()`, TileKernels' casts and
  norms take DeepSeek's PDL switch for their own calls, and the warp-specialized GEMMs keep their
  launch. That is vLLM `pdl-next` (0ccbe30e5, 3a0ac1afa, b2bae34c9, b3f659d9a, 1bc005688 over
  next-candidate 577091997), first built as `pdl` over 26382f5cf. Scheduling only: the tests
  require identical bits.

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
- **Splitting a window into fills placed in each latency-bound stretch** since the side stream
  came free, across the weight GEMMs in between: at most **1.51% c1 greedy, 2.04% sampled,
  0.60% c8** for FFN and NEXT, and 0.55 / 0.75 / 0.24% more for WO (`split_fills`).
  - It needs the early fills' lines to survive those GEMMs until their consumer runs. The
    moe-overlap session's GB10 probe (2026-10-25 20:07 CET, PR #587) shows evict-last lines do,
    at the recipe's set-aside of 0: 0.91-0.96 of the prefetch kernel's lines survive a 96 MB
    default stream. A persisting set-aside adds nothing.
  - The cost to watch, from the same probe: evict-last lines that are no longer needed crowd out
    default reuse until they are evicted (a 16 MB default re-read at DRAM speed beside 20 MB of
    stale lines). Each fill should hold only lines whose consumer has not yet run.
  - It is the one schedule change here with a ceiling above 0.5%; it is not built.
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

## The prototype (vLLM `pdl-next`, five commits over next-candidate 577091997)

First built as vLLM `pdl` over 26382f5cf (0b7952de4, 62538a5d8, tests in 75e6d02c7, 59b72eb81
and 352b07b52); the measurements in [Results](#results-on-dgx4-2026-10-25-cet) items 1-4 are of
that branch. `pdl-next` carries the same five commits onto next-candidate
([below](#the-final-head-pdl-next-on-next-candidate-577091997)).

`vllm/models/deepseek_v4_1/tilelang/pdl.py`:
- `chain()`: `T.pdl_sync()` then `T.pdl_trigger()`, the first statement of a kernel. The wait
  comes before any global access, so a kernel reads exactly what a serialized launch would read
  and stores in the same order; the trigger after it lets the next kernel launch while this one
  runs, so at most two kernels of the chain are resident. TileLang (b95ee4ff) marks a kernel that
  waits `has_cuda_pdl_sync` and its TVM runtime launches it with
  `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`.
- `per_token_cast`, `norm_forward`, `moe_topk_gate_forward`: TileKernels' kernels with
  DeepSeek's PDL switch on for that call only, used by every chain call site (the casts,
  `linear.cast_mxfp8` among them, the norms and the top-k gate). These kernels place the wait
  first and the trigger after their stores. Every other TileKernels kernel keeps its default (no
  PDL): some read inputs before their wait (results item 2).

Chained: mHC entry, finalize (which on next-candidate also writes its output's MXFP8 cast),
post and collapse; the decode MXFP8 GEMM tiles, split-K partials and reduces, WO-A's W8A16
kernels, the split-K BF16 GEMM; RoPE, cache writes, the sparse-MLA metadata (builder, decode,
chunk, pages), sparse MLA and its combine, the indexer (its re-index over candidate block ids
included) and compressor; MoE dispatch, grouped FP8 x FP4 GEMMs, SwiGLU cast and combine, at the
decode and the 64-row prefill tile heights; routing masks and partials; the shared expert's
SwiGLU. Not chained: the warp-specialized GEMMs (`gemm_bf16`, `gemm_bf16_two`,
`gemm_bf16_partials`, `mxfp8_gemm`'s prefill tiles), whose producer warps' TMA loads would have
to follow the wait as well; `mxfp8_gemm_decode` always chains and `mxfp8_gemm` never, so there
is no switch.

Why the bits cannot change: every global load and store follows the wait, which returns only
when every kernel the launch depends on has completed and its writes are visible. TileLang drops
`__restrict__` from a kernel that waits (so no non-coherent load can move above it); that changes
load instructions, not arithmetic. The tests check the bits anyway.

## Tests and benchmarks

In vLLM `pdl` (75e6d02c7; 59b72eb81 adds the graph-edge and eager checks), run on GB10 by the
bundles below.

- `tests/kernels/test_deepseek_v41_tilelang_pdl.py`:
  - a reader that loads before its wait, after a producer that spins tens of microseconds,
    must read the poisoned buffer in some CUDA-graph replays (otherwise the edges are not
    programmatic and PDL does nothing); a reader that waits first never does, in 20 replays;
  - the layer chain (below) at 6, 24 and 48 rows, every intermediate buffer set to NaN before
    each of 20 graph replays, gives its eager digests every time;
  - TileKernels' switch is on;
  - the early and waiting readers also as eager launches;
  - the captured graphs read back through the driver (`cuGraphGetEdges_v2`): a waiting
    reader's edge from its producer is programmatic and a torch kernel's is not; in the layer
    chain every chain kernel whose one predecessor is a kernel has a programmatic edge.
- `tests/kernels/deepseek_v41_pdl_sources.py`, a pytest plugin: every kernel a session launches
  is checked in its CUDA source. A chain kernel calls `cudaGridDependencySynchronize()` outside
  any block before any statement naming one of its pointer parameters, then
  `cudaTriggerProgrammaticLaunchCompletion()`; a warp-specialized GEMM has no PDL; any other
  kernel that waits does so first. The bundle runs it over the TileLang kernel tests of every
  changed module, which also check each kernel's bits against its reference on the PDL build.
- `tests/kernels/deepseek_v41_pdl_chains.py`: one decode layer's kernel chain at TP4's shapes
  (mHC entries, fused Q-A/KV, q norm, Q-B, WO-A split-K and reduce-cast, WO-B, the shared expert
  on its own stream beside the routed experts, the combine waiting for it), random weights and
  fixed routes by seed, only APIs that predate PDL, so it runs on either build. On `pdl` it
  was 23 kernels. On `pdl-next` it is 20: finalize writes the MXFP8 cast that fused Q-A/KV, the
  shared gate/up and the routed experts read, so their three casts go, and the entries' casts
  are among the digested buffers.
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

Regenerate the bundles and arms from the vLLM branches with
`python3 experiments/2026-10-25-pdl/make_bundles.py ~/projects/vllm-pdl-next 26382f5cf 577091997 pdl-next`.
Image: candidate 6 (`spark-ds41f:19f2c20ed4d6-next-6`, whose package files are 26382f5cf's).
Both builds are that image with files mounted over its tree. The PDL build gets every file
`pdl-next` changes since 26382f5cf ([overlay/](overlay)). The base build, next-candidate
577091997, gets the same files except PDL's: next-candidate's version of the seven package
files PDL changes ([overlay-base/](overlay-base)), and the image's own for the rest. The four
modules next-candidate deletes are mounted in both as modules that fail to import. The bundles
mount these from the node's checkout.

[bundles/pdl-sources](bundles/pdl-sources) runs [sources_cpu.py](sources_cpu.py) through
kernel-local: every chain kernel lowered at its decode shapes and checked without a launch.

1. [kernel.json](kernel.json), the cluster stopped, three nodes at once:
   - dgx2 [bundles/pdl-tests](bundles/pdl-tests): the PDL tests, then the 11 changed modules'
     kernel tests with the source check. Pass: exit 0, no violations.
   - dgx3 [bundles/pdl-chain](bundles/pdl-chain): [pdl_chain_ab.py](pdl_chain_ab.py) runs the
     chain benchmark on the base build and on the PDL build, three rounds alternating. Each
     build is a tree of links to the image's with its files copied in: the base build's 59
     (55 package files and the four stubs), then PDL's 14 on top. A probe checks that each tree
     imports its own mHC module, chained only in the PDL build. Pass: every digest equal across
     builds, replays and rounds; the line per shape gives the time saved per layer.
   - dgx4 [bundles/pdl-mechanism](bundles/pdl-mechanism): the synthetic benchmark (a status
     bundle; its numbers are the result).
2. [screen.json](screen.json): the determinism gate on [base-tp4.json](base-tp4.json) and
   [pdl-tp4.json](pdl-tp4.json), candidate 6's tp4-ring recipe on next-candidate without and with
   PDL. The PDL arm's hashes must be the base arm's, and so must its sampled and controls digests
   and its 16/8/4 distinct streams; the startup lines are recorded. Then `screen` and
   `single-stream` on [pdl-tp4-dev.json](pdl-tp4-dev.json),
   [base-tp4-dev.json](base-tp4-dev.json), and the PDL arm again. (Candidate 6's recipe passes
   `--load-format b12x`, which next-candidate still accepts. Next-candidate no longer reads its
   `VLLM_DS41_KERNEL_BACKEND`, as in retire-conditionals' arms.)
3. [profile.json](profile.json): the base and PDL arms under the torch profiler
   ([prof-base.json](prof-base.json), [prof-pdl.json](prof-pdl.json)), greedy, sampled
   and mixed-c8 windows ([profile_decode.py](profile_decode.py) with `--scenarios`). Then
   `critical_path.py results/private/determinism/pdl-profile/<label> --scenarios greedy,sampled,mixed-c8
   --window ...` on both. Under PDL a kernel's CTAs start before its predecessor ends, so its gap
   becomes zero and the wait's release moves into its own time: compare step times and the
   chained kernels' path time, not the gap rows alone.

Acceptance, before running: bits identical everywhere (kernel tests, chain digests, the
determinism gate's hashes); the chain benchmark and the screen's c1 decode step faster by at
least 0.5%, no suite row slower beyond the run's own repeat. The specs need this branch merged
(the lab syncs node checkouts to origin/main): after the release window, on the owner's word.

## Results on dgx4 (2026-10-25, CET)

dgx4 alone, uncabled, image `spark-ds41f:19f2c20ed4d6-next-6` (`sha256:82b071b57485`), bundles
run with `scripts/lab.py kernel-local BUNDLE --out DIR` from a checkout of this branch under
`~/spark-lab/agents/bound-and-prototype-pdl/` (outputs in its `out/`; the final head's in
`out/1bc005688` and `out/2362f6f8`, 2026-10-26 01:56-02:09 CET). Serving, determinism and
profile checks stay queued: they need the cluster.

1. **Source check without a GPU** ([sources_cpu.py](sources_cpu.py),
   [sources_cpu.txt](sources_cpu.txt)), spark-ds41f 4de3d223 (vLLM `pdl` 352b07b52), the image
   run without `--gpus`: 64 chain-kernel calls at TP4 shapes lowered for sm_121 with each
   factory's pass configs. 62 wait first and trigger after, the two warp-specialized GEMMs have
   no PDL, **0 violations**; 2 calls were skipped: the MoE SwiGLU needs a GPU to pick its
   arguments, and the compress call was the script's own error (ratio 4, which the factory
   rejects; found and fixed at the final head, item 5). The decode MXFP8 GEMM's
   generated code sets up its shared-memory pointers, then calls
   `cudaGridDependencySynchronize()` and `cudaTriggerProgrammaticLaunchCompletion()` before
   any load; its pointer parameters lose `__restrict__`, as TileLang does for a kernel that
   waits.
2. **pdl-tests, first runs.**
   - e29733a0 (smoke, the PDL test file alone): the plugin failed to load. `tilelang.jit` is
     also the decorator tilelang re-exports, so `import tilelang.jit.kernel` resolved to the
     function. Fixed in vLLM 352b07b52 (the module imported by name).
   - 4de3d223 (`out/pdl-tests`, 454 s): **PDL tests 10 passed**, among them the early reader
     reading the producer's buffer unwritten (graph and eager), the waiting reader never, the
     graph edges, and the layer chain bit-equal over poisoned replays. **Kernel tests of the
     11 changed modules: 416 passed, 1 xfailed.**
   - The source check failed the bundle with three distinct findings, each root-caused:
     - `pdl_reader_kernel` reads before its wait: the test's deliberate early reader. It is now
       its own kernel, `pdl_early_reader`, which the check exempts.
     - TileKernels' `mhc_post_fwd` reads its mixes, and its RoPE its positions and cos/sin
       table, before their wait; RoPE also waits inside a branch. DeepSeek's placement is safe
       only where the kernel launched just before did not write those inputs. The deployment
       never launches these two, but `set_pdl(True)` at first use had turned PDL on for every
       TileKernels kernel, the tests' references included. Fixed in vLLM 62538a5d8:
       `pdl.per_token_cast`, `pdl.norm_forward` and `pdl.moe_topk_gate_forward` set
       DeepSeek's switch for their call only, and every other TileKernels kernel keeps its
       default (no PDL). A new test checks that a cast through `pdl` has a programmatic edge
       and a direct call does not.

   - **d5eab5d8 (vLLM 62538a5d8; `out/pdl-tests`, 116 s, pass;
     [runs/tests-d5eab5d8.txt](runs/tests-d5eab5d8.txt)):** PDL tests 10 passed. That includes
     the scoped TileKernels switch and the layer chain's graph, where every chain kernel whose
     one predecessor is a kernel has a programmatic edge from it. Kernel tests: 416 passed, 1
     xfailed. **Source check: 0 violations** over 305 launched kernels, 230 of them with PDL.
3. **pdl-mechanism**, spark-ds41f 4de3d223 (`out/pdl-mechanism`, 92 s, pass;
   [runs/mechanism-4de3d223.txt](runs/mechanism-4de3d223.txt)). Synthetic dependent kernels,
   64 a CUDA graph, microseconds per kernel:

   | CTAs | work | plain | wait only | `chain()` | late trigger | saved by `chain()` |
   | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | 1 | none | 0.704 | 0.613 | 0.551 | 0.579 | 0.15 |
   | 6 | none | 0.811 | 0.709 | 0.538 | 0.651 | 0.27 |
   | 48 | none | 0.812 | 0.683 | 0.495 | 0.651 | 0.32 |
   | 192 | none | 0.845 | 0.715 | 0.595 | 0.796 | 0.25 |
   | 1 / 6 / 48 / 192 | about 2 us | 2.58 / 2.25 / 2.32 / 2.44 | | 2.28 / 1.93 / 1.96 / 2.09 | | 0.31 / 0.32 / 0.36 / 0.35 |
   | 1 / 6 / 48 / 192 | about 10 us | 10.09 / 9.93 / 9.74 / 9.80 | | 9.80 / 9.71 / 9.42 / 9.49 | | 0.29 / 0.22 / 0.32 / 0.31 |

   - A boundary costs 0.7-0.85 us in a graph. The wait alone recovers about 0.1 us of it;
     triggering after the wait (`chain()`) recovers 0.25-0.36 us; TileKernels' trigger after
     its stores, 0.1-0.2 us. So triggering early is what pays.
   - The fork/join (A, B on a side stream, C after both) saves 0.09 us of 29.0 us.
   - **The traced c1 layer** ([layer-c1.json](layer-c1.json): target layer 10's 28 path
     kernels as spins of their traced CTAs, threads and durations, 25 chained) takes 503.3 us
     plain and 496.1 us chained: **7.2 us saved, 1.42% of the layer**, 0.29 us per chained
     boundary, 57% of the trace's 0.51 us median gap.
   - At 0.29 us across the 1,244 chained graph boundaries of a c1 greedy step, that is about
     360 us: **about 1.0% of a c1 greedy step, 1.3% c1 sampled, 0.6% at c8**, from launches
     alone, before the eager boundaries and any prologue overlap. The serving screen measures
     it end to end.

4. **pdl-chain**, spark-ds41f d5eab5d8, vLLM 62538a5d8 (`out/pdl-chain`, 116 s, pass;
   [runs/chain-d5eab5d8.txt](runs/chain-d5eab5d8.txt)). The 23-kernel layer chain, four layers
   a graph, on candidate 6's build (the image's tree) against the same tree with the branch's 14
   package files. Three rounds, alternating which build goes first, 50 replays each; a probe
   confirmed which build each tree imports.
   - **Bits: identical.** Every written buffer's digest matches across builds, rounds and
     replays at all three shapes, and every replay matches its eager run.
   - Time per layer (median of rounds; per-round values in the receipt):

     | Rows | warm, image / PDL | saved | cold, image / PDL | saved | per round, image - PDL (warm) |
     | ---: | ---: | ---: | ---: | ---: | --- |
     | 6 (c1) | 961.9 / 944.5 us | **17.4 us, 1.81%** | 951.7 / 939.3 us | **12.4 us, 1.30%** | +17.4, +23.8, +6.7 |
     | 24 | 2629.1 / 2633.2 us | -0.16% | 2626.1 / 2626.0 us | 0.01% | +17.2, -10.1, +22.2 |
     | 48 | 3928.0 / 3937.2 us | -0.23% | 3926.7 / 3937.8 us | -0.28% | +4.0, -36.0, -9.2 |

   - At one stream PDL is faster in every round, warm and cold, by more than the synthetic
     estimate (17 us against about 21 x 0.29 = 6 us). The real kernels start up more slowly
     than the spins: grid launch, shared-memory and barrier set-up, first loads.
   - At 24 and 48 rows the rounds disagree in sign, and their spread (about 20-40 us) is larger
     than the 6-17 us a layer's boundaries can give. Random routes make these chains 2.6-3.9 ms
     a layer of expert weight streaming (144 to 288 routes), much longer than a serving layer at
     c8 (about 1.5 ms), so a 0.2-0.6% gain is below this benchmark's resolution. The serving
     screen measures c8 on real routes.

### The final head: `pdl-next` on next-candidate 577091997

vLLM `pdl-next` 1bc005688 (2026-10-26) is the five commits rebased onto next-candidate 577091997:

| Commit | |
| --- | --- |
| 0ccbe30e5 | Programmatic dependent launch along the decode chain |
| 3a0ac1afa | PDL: replay tests, a source check and two benchmarks |
| b2bae34c9 | PDL tests: the captured graph's edges, and eager launches |
| b3f659d9a | PDL source check: import `tilelang.jit.kernel` by name |
| 1bc005688 | PDL: DeepSeek's TileKernels switch only for the chain's own calls |

How the overlaps resolved:
- **`metadata.py`:** next-candidate's `builder_kernel` (kernel-audit's requests and tokens in
  one launch, prim_func `metadata_builder`) and live-page `pages_kernel`, with `pdl.chain()` at
  the top of the builder, chunk, decode and pages kernels.
- **`moe.py`:** `run` takes the tokens as `MXFP8Rows` from finalize, so PDL's token cast there
  goes.
- **`linear.py`:** `linear.cast_mxfp8` (the cast left for regrouped rows) goes through
  `pdl.per_token_cast`. `linear._tile_kernels` had no caller left and is gone. The fold lands
  in 1bc005688, the commit that scopes the switch.
- **The layer-chain harness:** finalize's `yq`/`ysf`; Q-A/KV, the shared gate/up and the routed
  experts read the entries' casts; 20 kernels.

Two newer kernel paths were checked against the chain:
- **0504's 64-row tiles (next-candidate 9ad4a1fdb):** `MoeExpertsV41Plan` builds the prefill
  grouped GEMMs and SwiGLU cast from the same factories as the decode ones, and the prefill
  dispatch plan from the same dispatch factory. All begin with `pdl.chain()`. They are
  prefill-only: calls above the decode rows.
- **0507's re-index (577091997):** `indexer_score_v41` with `candidates=True` and the
  candidate-block top-k. These are the same chained `indexer_score` and `indexer_topk_*`
  prim_funcs, so they are chained wherever they run.

Apart from `metadata_builder` replacing `metadata_requests` and `metadata_tokens`, 577091997's
DS4.1 model tree has the same prim_funcs as 26382f5cf. The V2 runner's new `fill_kv_blocks`
(the debug-only `VLLM_KV_CACHE_POISON` fill) is input preparation outside the model's graphs,
unchained like the other input-preparation kernels.

5. **The source check over the final kernel set** ([bundles/pdl-sources](bundles/pdl-sources),
   kernel-local):
   - 3a3792c2 ([runs/sources-3a3792c2.txt](runs/sources-3a3792c2.txt), 62 s): 83 calls, 81
     waiting first and triggering after, the two warp-specialized GEMMs without PDL, 0
     violations. The compress call was skipped again. That was the script's error (ratio 4;
     the factory takes 1 or 2), not a GPU need, so the static check had never covered the
     compress kernel; the kernel tests' launches had, through the plugin.
   - Fixed in 2362f6f8: both ratios are lowered, and a call that fails to lower fails the run.
     **2362f6f8 ([runs/sources-2362f6f8.txt](runs/sources-2362f6f8.txt), 74 s, pass): 85 calls,
     83 with PDL, 0 violations, none unlowered.** The metadata builder (plain and circular),
     the pages kernel at pages of 64 and 256, the re-index and the 64-row dispatch are among
     them.
6. **pdl-tests**, 3a3792c2 ([runs/tests-3a3792c2.txt](runs/tests-3a3792c2.txt), 211 s, pass):
   - PDL tests: 10 passed.
   - The 11 modules' kernel tests on the PDL build: **530 passed**, 1 skipped
     (compute-sanitizer is not in the image), 1 xfailed.
   - **Source check: 0 violations** over 321 launched kernels, 242 of them with PDL.
7. **pdl-mechanism**, 3a3792c2 ([runs/mechanism-3a3792c2.txt](runs/mechanism-3a3792c2.txt),
   70 s, pass):
   - `chain()` saves 0.19-0.37 us a boundary, as in the first run, and the fork/join 0.34 us.
   - The traced c1 layer takes 510.5 us plain and 506.7 us chained: 3.85 us, 0.75% (7.17 us,
     1.42% in the first run). It is a synthetic benchmark of spins, independent of the vLLM
     build, so the two runs bound its run-to-run spread.
8. **pdl-chain:** next-candidate's build against `pdl-next`, both over the image. The
   20-kernel chain, four layers a graph, three rounds alternating, 50 replays each. Two runs:
   3a3792c2 ([runs/chain-3a3792c2.txt](runs/chain-3a3792c2.txt), 96 s) and 2362f6f8
   ([runs/chain-2362f6f8.txt](runs/chain-2362f6f8.txt), 94 s), both pass.
   - **Bits: identical** in both runs: every written buffer's digest, the entries' MXFP8 casts
     included, across builds, rounds and replays at all three shapes, and every replay equals
     its eager run. The probes confirmed that each tree imports its own `mhc.py`, chained only
     in the PDL build.
   - Time per layer (median of rounds; base - PDL per round, warm):

     | Rows | run | warm, base / PDL | saved | cold, base / PDL | saved | per round (warm) |
     | ---: | --- | ---: | ---: | ---: | ---: | --- |
     | 6 (c1) | 1 | 958.9 / 952.5 us | **6.4 us, 0.67%** | 953.6 / 944.4 us | **9.2 us, 0.97%** | +1.2, +9.0, +21.5 |
     | 6 (c1) | 2 | 955.8 / 943.3 us | **12.5 us, 1.31%** | 948.9 / 938.3 us | **10.6 us, 1.12%** | +11.2, +10.2, +12.5 |
     | 24 | 1 | 2632.1 / 2619.8 us | 0.47% | 2628.8 / 2616.4 us | 0.47% | +12.4, +23.0, +12.9 |
     | 24 | 2 | 2619.4 / 2619.8 us | -0.01% | 2616.2 / 2618.2 us | -0.08% | -3.9, -3.1, +2.2 |
     | 48 | 1 | 3947.7 / 3966.5 us | **-0.48%** | 3945.3 / 3965.2 us | -0.50% | +4.4, -6.4, -27.8 |
     | 48 | 2 | 3948.2 / 3963.0 us | **-0.37%** | 3947.5 / 3961.1 us | -0.34% | +11.4, -12.9, -31.9 |

   - **c1: PDL is faster in every round of both runs, 0.67-1.31% warm and 0.97-1.12% cold.**
     That is 6-12 us a layer, about the mechanism's estimate (20 boundaries at 0.29 us). The
     chain on 26382f5cf saved 17 us over 23 kernels. Next-candidate's finalize cast removed
     three of the boundaries PDL would have shortened.
   - 24 rows: within noise (+0.47%, then -0.01%).
   - **48 rows: PDL trends slower.** The median is -0.37% to -0.48% in both runs, PDL is slower
     in 4 of 6 rounds, and the chain on 26382f5cf gave -0.23%. Per round the spread is 40 us,
     so a 15 us effect is at this benchmark's resolution. But the sign holds across three runs,
     so it is recorded as a finding, not as noise.
     - The likely mechanism (not yet tested): a chained kernel's CTAs launch early and hold SM
       slots while they wait. At 48 rows the routed experts' long kernels run beside the shared
       expert's kernels on the aux stream, which then get fewer SMs.
     - The discriminating check: the same A/B with the shared expert inline (no aux stream).
       The regression should vanish if this is the cause.
     - The serving screen's c8 and c16 rows measure it on real routes.

The cluster checks stay queued as [screen.json](screen.json) and [profile.json](profile.json).

## Interactions with the other sessions

- **kernel-audit** (vLLM `kernel-audit`): it merges `metadata_requests` and `metadata_tokens`
  into one launch, the largest eager boundaries here (136 us a c1 step, 0.40%), and moves the mHC
  entry's fn tile loads. Every kernel it removes takes a PDL boundary with it. On next-candidate
  its merged `metadata_builder` begins with `pdl.chain()` (`pdl-next`, resolved in 0ccbe30e5),
  like its live-page `metadata_pages`.
- **moe-overlap** (vLLM `moe-overlap`): its GB10 probe settles the cache-priority question for
  the windows. Evict-last prefetched lines survive the GEMMs between a fill and its consumer
  at set-aside 0; the set-aside adds nothing; routed evict-first is dropped. That makes split
  fills (at most 1.5% at c1, above) possible without a set-aside; the windows' lateness sizes
  the pool (77-87% of instances late by a median 27-48 us, 4.67% of the step). The act-quant fold into `mhc_finalize` (next-candidate a7ba079c7)
  removes the casts between finalize and the projections it feeds (two boundaries become one,
  about 0.5 us a sublayer saved by either); its finalize keeps `pdl.chain()`, and the cast left
  for regrouped rows (`linear.cast_mxfp8`) goes through `pdl.per_token_cast`.
- **fusion-survey**: its "PDL across the decode graphs" ceiling (1.9 / 2.4 / 1.1%) counted gaps up
  to 1.5 us inside the layers plus `min(predecessor, GEMM)` for weight GEMMs. Measured on the
  critical path the boundary gaps alone are 3.7 / 4.8 / 1.7% (half of them over 1 us), and the
  weight prologue is modelled separately (0.8 / 1.0 / 0.4%). Every fusion that removes a
  boundary also removes that boundary's PDL gain: the ceilings overlap.

## Follow-ups (not built)

- The 48-row chain's 0.4% slowdown: the same A/B with the shared expert inline. If the
  slowdown goes away, waiting CTAs are taking SMs from the aux stream. One kernel-local
  bundle on dgx4.

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
  [pdl_chain_ab.py](pdl_chain_ab.py), [sources_cpu.py](sources_cpu.py), [bundles/](bundles),
  [overlay/](overlay), [overlay-base/](overlay-base): the bundles and the builds' files,
  generated from vLLM `pdl-next` and next-candidate.
- [pdl-tp4.json](pdl-tp4.json), [pdl-tp4-dev.json](pdl-tp4-dev.json),
  [base-tp4.json](base-tp4.json), [base-tp4-dev.json](base-tp4-dev.json),
  [prof-base.json](prof-base.json), [prof-pdl.json](prof-pdl.json): arms.
- [kernel.json](kernel.json), [screen.json](screen.json), [profile.json](profile.json): specs.
- [profile_decode.py](profile_decode.py), [summarize_kernels.py](summarize_kernels.py),
  [analyze_costs.py](analyze_costs.py): request-inventory's profiling scripts (the first with
  `--scenarios`).
