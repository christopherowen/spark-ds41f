# IKET debug mode: phase traces from inside the DS4.1 TileLang kernels

Until now we could only see the serving kernels from the outside: nsys gives each
launch's start and end. IKET is NVIDIA's in-kernel event tracing (CuTe DSL's
`run-iket`), and TileLang's `tilelang.tools.cuda.iket` is its TileLang front end. With
this mode, every TileLang kernel carries IKET ranges around its phases: where it loads,
computes, waits and stores. A trace shows, warp by warp, how long each phase takes.

The mode is debug only and off in production. Off means no IKET import, no TileLang
callback, and generated code identical to the build without the mode.

- vLLM: branch `iket-debug` (worktree ~/projects/vllm-iket, local only), cut from
  next-candidate 577091997, head 52862f2ca. Not part of next-candidate.
- TileLang: no change. The image's pin (tile-ai/tilelang b95ee4ff, 0.1.15) and the fork's
  `deepseek-v41-sm120` hold the same IKET code.

## Step 1: what IKET needs, and GB10

The probe bundle (`bundles/probe`, run 2026-10-26 01:33 CET on dgx4,
`runs/probe-1.txt`) found:

| question | answer |
|---|---|
| Where does the profiler come from? | `run-iket` (entry point `iket.cli.main:entrypoint`), from `nvidia-cutlass-dsl-libs-base` 4.7.1, which the image already installs for CuTe DSL. Its native parts (`libsmodel_injection.so`, `iket_lib`, `libiket_cubin_info.so`) are aarch64 builds. |
| Does it run on aarch64 and SM121? | Yes. It profiled TileLang kernels compiled for `sm_121a` inside the plain `kernel-local` container. No extra capability was needed, although the driver has `RmProfilingAdminOnly=1`. |
| `%cluster_ctarank` on SM12x | TileLang emits it for SM90 and later, so for `sm_121a` too. It compiles, and outside a cluster launch it reads 0. |
| Does TileLang's IKET match the fork? | Yes, byte for byte, at the image's pin (b95ee4ff, which contains #2515). The r6u patch series does not touch it. |
| Where do the records go? | Each event's placeholder stores to shared address 0x20. On GB10 the dynamic shared base is 1024, which is the per-block reservation (`cudaDevAttrReservedSharedMemoryPerBlock` = 1024). The placeholder therefore never touches kernel data, and an instrumented kernel computes correct bits outside the profiler too (`smem` probe). `run-iket` patches the placeholders into trace stores. |

What run-iket does:
- It runs the workload twice: a dry run that counts records per warp, then the
  measured run.
- Every launch gets warp lifetimes, whether or not the kernel is instrumented.
- Timestamps come from `%globaltimer`, in 32 ns steps. The event id lives in the low five
  bits, so a module can hold at most 30 events; id 31 ends a range.
- It cannot run alongside nsys or other CUPTI tools, and it has no capture window: it
  profiles the whole process.

The image has no `nvdisasm`; dgx4's host does (`/usr/local/cuda-13.0/bin/nvdisasm`). Nsight
Systems 2025.3.2 is on the host and is mounted into the bundles at `/opt/nsys`.

## Step 2: the switch

`VLLM_DS41_IKET=<directory>` follows `vllm/envs.py`'s `VLLM_DS41_*` convention. Like every
variable there, it is part of vLLM's compile key. The GPU worker turns the mode on first
thing in `init_device`, before any kernel compiles. A harness calls
`vllm.models.deepseek_v4_1.tilelang.iket.enable(dir)` itself.

When on:
- TileLang's kernel and binary caches are off, and `TILELANG_CACHE_DIR` is
  `<dir>/tilelang-cache`. An instrumented kernel is never read from or written to the
  deployment's cache.
  - The in-process caches key on the mode by construction: it is one-way for the process.
  - `enable` refuses once the process has compiled any TileLang kernel, because those
    kernels would keep serving their calls uninstrumented.
- Kernels still go through declare, prepare and freeze (`tilelang.prepare` in its thread
  pool, then `freeze`). Preparation compiles every one of them, so a boot is a cold boot.
- `<dir>/sources/<kernel>.<hash>.cu` and `.json` hold each instrumented module and its
  sites: range name, phase, kinds, source lines, first statement, loop depth and a static
  bound on a warp's events.

Three optional knobs, read only when the mode is on. An engine-wide trace needs them;
the hot kernels did not:
- `VLLM_DS41_IKET_CTAS=N`: only about N CTAs of a launch record ranges, evenly spread.
  The flag is uniform per CTA, so ranges stay balanced.
- `VLLM_DS41_IKET_ITERS=N`: a range inside a `for` loop records in at most N of its
  iterations, each in the middle of its share of the loop, so `N=1` samples the middle
  iteration rather than the pipeline's fill. The guard uses the loop variable, so it is
  uniform within an iteration. Each range then records at most once per sampled
  iteration and group of lanes that reaches it, and the manifests' static bound holds
  where a warp's lanes do not diverge (sparknet's polling lanes: 26 records against a
  bound of 22). A TVM
  `while (1)` loop has no variable to sample: a range inside one records in every
  iteration, the site lists the loop as unsampled, and its kernel has no bound. In the
  decode step that is the logits processor (`bias_kernel`, at most 16 vocabulary tiles)
  and the slot-mapping binary search.

  A sampled event sits in a loop of zero or one iterations (`TL_IKET_WHEN`), never
  behind a plain `if`. run-iket gives every event a warp *issues* a slot in the warp's
  record, predicated off or not, and ptxas predicates a short guarded block whatever its
  condition. Behind `if`s, the sampled sparse MLA kernel's warps took 109 slots for 42
  records (187 while lane 0 alone recorded: a lane's guard predicates more). ptxas does
  not predicate a loop, and an `asm` hides the loop's count from it.
- `VLLM_DS41_IKET_SKIP=REGEX`: kernels whose name matches compile uninstrumented. run-iket
  still records their warps' lifetimes.

Where the marks come from: a pass, not the kernel sources. The pass runs in TileLang's
`tilelang_callback_cuda_postproc` hook, over each module's generated CUDA, then hands the
result to TileLang's IKET codegen (`inject_iket_cuda`).

Why a pass:
- The kernels' Python text stays unchanged, which was the brief's preference.
- Every TileLang kernel the process compiles is covered: DS4.1's, TileKernels'
  (third-party), sparknet's and the samplers'.
- TileLang's lowering has no other hook. A TIR pass would need a TileLang C++ change and a
  new image.

How the pass works:
1. It parses each `__global__` body (TVM's layout: one statement a line, every control
   statement braced) into statements and blocks.
2. It gives each statement a phase:
   - `load`: TMA, cp.async, a read of a pointer parameter;
   - `wait`: `cp_async_wait`, mbarrier `.wait`, `__syncthreads` and named barriers,
     `cudaGridDependencySynchronize`, sparknet's `roce_wait_*`, fences, a spin on a global
     flag;
   - `compute`: MMA, ldmatrix, `AllReduce`, shuffles, sparknet's reduce;
   - `store`: a write through a pointer parameter, TMA store, atomics, sparknet's staging;
   - `signal`: sparknet's doorbell, release stores and tail.
3. A block has its statements' phase when they share one.
4. Consecutive siblings of one phase become one range. A block that mixes phases is
   entered instead, and every wait is a range of its own.
5. Safety rules:
   - A range never contains a `return`, `break`, `continue` or `goto`, so it closes on
     every warp path (the expert GEMM's `if (expert < 0) return;`).
   - `#pragma unroll` stays with its loop.
   - A node that shares a line, or a kernel with an `#if`, is left alone.
6. Names are `<phase>.<kind>#<site>`. A module with more than 30 names folds its sites into
   their first names, then traces at most 30.
7. If the pass meets a source it cannot parse, the compile fails loudly. It never falls
   back to an uninstrumented kernel.

## Step 3: the hot kernels on dgx4

`harness/iket_hot.py` builds four kernel families at TP3 decode shapes, 8 rows each, on
one GPU:
- `moe`: the routed experts, `MoeExpertsV41Plan(384, 6, 5120, 768)`: dispatch, gate-up
  GEMM, SwiGLU cast, down GEMM and combine;
- `mla`: sparse MLA decode at 24 heads a rank, a 128 window and 512 indexed entries
  (`split` and `decode` routes);
- `mhc`: the mHC entry, `project_streams` then `finalize`;
- `allreduce`: sparknet 0.5.0's one-shot all-reduce at TP3 geometry, with a host thread
  in place of the two peers' proxies. The thread writes the peers' flags after the
  doorbell, optionally late.

The kernels go through `prepare.declare`, then `freeze`, as the worker does. The bundles:
- `hot-base` runs next-candidate 577091997 itself.
- `hot` runs the branch:
  1. the mode's tests;
  2. off;
  3. on, outside the profiler;
  4. on, under `run-iket`;
  5. the all-reduce with its peers 20 us late;
  6. nsys of off and of on;
  7. the tables.

Receipts are in `runs/hot-2` (before the settle fix) and `runs/hot-5`.

### Off is identical

Run 2 compared off on the branch with next-candidate (`hot-base`):
- All 29 generated CUDA sources are byte-identical (`diff -r` of the dumped sources).
- The output SHA-256s are equal for all four families.
- Neither `tilelang.tools.cuda.iket` nor `vllm.models.deepseek_v4_1.tilelang.iket` is
  imported, and no `tilelang_callback_cuda_postproc` is registered.

The mode's GPU test checks the same on a small kernel in fresh processes
(`runs/tests-6`, 11 passed):
- off and on give equal bits;
- the deployment cache's file list is unchanged by the on run;
- the on run's cache directory is redirected and stays empty.

Instrumentation does not change results either. On outside the profiler and on under
`run-iket` give the same output SHA-256s as off, for all four families, in both runs.

### What the hot kernels show (run 5, 8 rows, TP3 shapes)

Mean warp share of each phase. "Last warp" is the warp a launch waits for.

| kernel | grid | nsys off us | IKET span us | wait | load | compute | store | unattributed | biggest range |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| routed gate-up GEMM | 12x48 | 712.9 | 901.4 | 73% | 8% | 13% | 0% | 6% | `wait.mbarrier` (TMA weight tiles) 44%, then the block sync after `cp_async_wait` 19.5% |
| routed down GEMM | 40x48 | 344.0 | 480.0 | 54% | 21% | 14% | 2% | 9% | `wait.mbarrier`, then the expert-id/routing reads (`load.global#3`) |
| sparse MLA, `split` | 5x2x8 | 20.6 | 25.4 | 16% | 11% | 31% | 5% | 38% | MMA, then the softmax `AllReduce` |
| sparse MLA, `decode` | 5x1x8 | 12.9 | 18.3 | 20% | 10% | 38% | 8% | 24% | MMA |
| sparse MLA, segment combine | 24x8 | 2.3 | 7.3 | 5% | 90% | 1% | 0% | 3% | reading the partials |
| SwiGLU cast | 16x48 | 3.5 | 11.3 | 0% | 64% | 13% | 3% | 19% | reading gate/up |
| mHC `project_streams` | 40 | 5.5 | 6.1 | 5% | 34% | 45% | 2% | 13% | TF32 MMA, then loading the streams |
| mHC `finalize` | 8 | 4.2 | 3.9 | 15% | 19% | 21% | 10% | 35% | the mix's `AllReduce` (warp 0) |
| one-shot all-reduce | 8 | 7.6 | 8.4 | 68% | 1% | 14% | 8% | 7% | the barrier after the peer-flag wait |
| MoE dispatch | 1 | 4.9 | 5.7 | 40% | 2% | 0% | 34% | 24% | the route-position stores, between nine block syncs |
| MoE combine | 5x8 | 3.8 | 3.7 | 0% | 96% | 0% | 1% | 3% | reading the expert rows |

What the table shows:
- At 8 rows the routed experts are bound by the weight stream: the gate-up GEMM's warps
  spend three quarters of their time waiting.
  - Most of the wait is for the stage's TMA weight tile (`loaded[s].wait`).
  - The block sync after `cp_async_wait` takes another fifth: there the warps wait for
    the slowest warp's activation and scale copies.
- The all-reduce's time is the peer wait. With the peers 20 us late, the span goes from
  8.4 to 35.4 us, and every warp spends 31 us (90%) in `wait.sync#2`.
  - Only six lanes of block 0 poll the flags. The others meet them at the barrier after
    `roce_wait_peers`, so the barrier is where the collective's wait shows.

### Alignment with nsys, and overhead

- **Alignment.** For a kernel IKET hardly slows, a launch's span (first warp start to last
  warp end) matches the nsys duration to within about 0.5 us. It can come out below nsys,
  because the span leaves out launch setup: finalize 3.9 vs 4.2 us, combine 3.7 vs 3.8 us,
  per-token cast 2.0 vs 2.2 us.
  - Warp lifetimes add up: the gate-up GEMM's 576 CTAs run one per SM, 12 waves of a
    73 us warp, against an 901 us span.
- **Overhead in the placeholder build.** These are kernels built in IKET mode but run
  without the profiler (nsys "on" vs "off", run 5):
  - routed GEMMs: +1.2 to +1.5%;
  - SwiGLU and combine: +0 to +3%;
  - mHC finalize: +6%;
  - mHC `project_streams`: +19%; MoE dispatch: +21%;
  - sparse MLA: +22% (27 ranges in a short kernel; each range adds its events and the
    settle load).
- **Overhead under `run-iket`.** The patched trace stores cost more:
  - routed GEMMs: +26% and +40%;
  - sparse MLA: +24% and +42%;
  - the tiny combine and SwiGLU kernels: 3.2x and 3.3x.

  The per-phase shares hold up, but absolute times under the profiler are upper bounds.
- **Host overhead.** Calls under the profiler take 2.7 ms (MoE) instead of 1.1 ms. Do not
  time a run that is being traced.

### Two measurement fixes the hot kernels needed

1. **Deferred barriers.** ptxas emits `__syncthreads()` as `BAR.SYNC.DEFER_BLOCKING`: a
   warp keeps issuing until an instruction touches memory.
   - So a range end's `%globaltimer` read (`CS2R`) ran before the barrier released the
     warp. The warp then stalled in the end event's own shared store, and the wait
     counted toward the next range or as unattributed.
   - `harness/sass_check.py` shows ptxas keeps the instructions in order; the hardware is
     what runs ahead.
   - Each range end now first does a volatile shared load of IKET's own slot
     (`TL_IKET_SETTLE`).
   - Results: the late all-reduce's unattributed time fell from 86% (24.2 us a warp) to
     1.4%, and the gate-up GEMM's from 25% to 6%. The SASS check finds every block barrier
     followed by a memory access before the next timer read.
2. **Divergent waits.** IKET records at warp level, when a warp's first lanes pass an
   event.
   - A wait that only some lanes take, like sparknet's six polling lanes, closes early for
     the rest of the warp.
   - Its time shows at the barrier that follows (`wait.sync#2` above). This is a property
     of IKET's warp granularity, not something the pass can change.

## Step 4: decode steps end to end on one GPU

DS4.1-Flash does not fit one GB10 at native weights. `harness/iket_step.py` runs vLLM's
engine in its own process, on DS4.1-Flash-1017's config with dummy weights at TP1. It
uses the production recipe's kernels and graphs:
- the TileLang attention, linear and MoE backends;
- DSpark with five greedy drafts, block verification and adaptive verification;
- `FULL_AND_PIECEWISE` CUDA graphs;
- the packed BF16 head in FP32;
- async scheduling and aligned prefill chunks.

Six of the 40 layers keep every attention role (callable `hf_overrides`, which reach the
drafter too):
- two SWA layers (ratio 0);
- a CSA layer that is a KV and index source, and one that reuses it (ratio 2);
- the HCA candidate source, and a layer that re-indexes its candidates (ratio 1);
- the drafter's three layers after them.

The Engram layers are left out, and at TP1 nothing all-reduces. Shapes are TP1's: the
sharded GEMMs are three times TP3's width. A marker kernel (`iket_step_marker`) brackets
the measured request: one prefill and eight decode steps after a warm request.
`iket_table.py --between iket_step_marker_kernel` reads only the launches between the
two markers.

What a whole-engine trace needed:
- **CTA sampling (`VLLM_DS41_IKET_CTAS=4`).** In each launch only about four CTAs,
  evenly spread, record ranges. run-iket still records every warp's lifetime, so spans
  stay exact.
  - Unsampled, the hot kernels alone wrote a 1.39 GB trace.json from 61 launches: the
    expert GEMMs' K loops put up to 36 iterations of ranges in every warp.
- **Bounded records (`VLLM_DS41_IKET_ITERS=1`).** run-iket's
  device pool is a bump allocator for the whole process (`ContextBuffer::Allocate`).
  Every instrumented launch reserves the run's largest per-warp record for each of its
  warps (`compute_required_buffer_size`), whether or not the warp records anything. Graph
  replays count as launches of their own (`graph_exec_<id>:<replay>`).
  - The first engine trace's dry run saw 16,894 launches and 86 graph launches, mostly
    the engine's startup. Its largest warp held 5,016 timestamps.
  - The pool came to 898 GB, and the run was killed.
  - With a fixed 8 GB pool, the run exhausted it during the encoder's profile run:
    8.49 GB in, a 695 MB request failed.

  With CTA sampling and the middle iteration of each loop, the sources' static bound is
  at most 64 timestamps a warp (sparse MLA: 32 ranges). Yet the dry run recommended 424
  for the decode steps (runs 14-20). With 26.6 million instrumented warps, 19.7 million
  of them in eager launches (the profile run, the warmups and prefill; the routed
  gate-up GEMM alone 9.1 million), that is a 52 GB pool, more than GB10 has beside the
  engine. Capping it (`--max-ts-cnt-per-warp 128`) crashed the exit dump in
  `iket::to_json(WarpBuffer)`, with the graphs (run 15) and without (run 18).

  Finding the cause:
  - It is not graph replays. The tracker lists each replay as a launch of its own
    (`graph_exec_<id>:<replay>`), and an eager run recommends 424 too.
  - Every kernel family exceeds its bound (run 20: GEMMs 368, MoE 224, attention 424,
    mHC 64, inputs 24, sampler 96).
  - The hot kernels reproduce it (runs 21-23). Their warps record at most 42 timestamps.
    The dry run's own count (`max_count`, logged at `--log-level debug`) is 187, and a
    cap of 64 crashes the same way.
  - Their SASS has the answer (`sass_check.py`): ptxas predicated many guarded events
    (`@!P1 PMTRIG`; sparse MLA: 32 of 64). run-iket patches an event's shared store,
    and a warp that issues the store takes a slot, predicated off or not.
  - Dropping the lane guard (lane 0 alone recorded, which made every guard
    lane-dependent) took the count to 109; ptxas still predicated 26 of sparse MLA's
    events behind uniform guards (run 24).
  - In zero-or-one loops (vLLM 52862f2ca), no event is predicated, the dry run counts
    43 against 42 records, and the run capped at 64 completes with the same records
    (run 25).

  For the decode steps the dry run now recommends 80 a warp (run 26), a 10.3 GB pool
  with the CUDA graphs.
- **Room for the exit dump.** run-iket decodes its records when the process exits,
  on the host: run 26's measured run finished its requests, wrote a 4.0 GB
  `iket.data.json`, and ran out of memory decoding it (the watchdog stopped it at
  7 GiB available) while the engine still held 53 GB. The harness now shuts the engine
  down and empties PyTorch's cache once its result is written.
- **Room on the GPU.** The pool counts against vLLM's startup check, so the harness sets
  `gpu_memory_utilization=0.5`. The KV pool is fixed by bytes. The profile run is kept
  small: 256-token chunks in a 512-token step. Graphs are captured for one and two
  requests' verification rows only (6 and 12), since every capture warmup's launches
  reserve records too.
- **Uninstrumented kernels (`VLLM_DS41_IKET_SKIP=^(vision_|topk_topp)`).** The vision
  tower and the top-k/top-p sampler are not in a greedy text decode step. Left
  uninstrumented, their warps reserve no records.
- **Comparable runs.** vLLM's dummy loader leaves integer tensors uninitialized: the
  packed FP4 experts and their UE8M0 scales. The harness zeroes them. Even then, two off
  runs give different tokens, because more of what the TP1 dummy engine reads is never
  written. So the token comparison (`step-bits`) fills uninitialized tensors with
  `torch.use_deterministic_algorithms(True, warn_only=True)`. The trace run does not,
  since the fills add kernels to the graphs.

  The comparison cannot decide anything at the engine level (run 12, `runs/step-bits-12`).
  With dummy weights the output distribution is nearly flat: every top-5 logprob is
  about −11.7696 (−ln 129,280), with neighbours 1e-6 apart, so greedy tokens are argmaxes
  over near-ties.
  - The first off run's tokens differ from every other run's.
  - The second off run and the three on runs (unsampled, CTAs sampled, CTAs and
    iterations sampled) give the same tokens.
  - Their logprobs are equal exactly in seven of the eight steps. In the first step the
    top-5 sets differ, also between two on runs of the same build.

  So off-path identity rests on step 3: the generated sources and the outputs' bits.
  When the mode is off, no callback is registered, so every module is TileLang's own
  output.

### Step 4 results: c1 decode steps at the ring's and the triangle's rank shapes (dgx3)

dgx3, vLLM 734932a56, bundles `step-tp4-c1` (run 32) and `step-tp3-c1` (run 33).
Receipts are in `runs/step-tp4-c1-32` and `runs/step-tp3-c1-33`: `tables.md` (the
output), `table.json`, `step-diff.json`, and both runs' `result-*.json`.
- **The request:** a 32-token prompt and 32 tokens, as one prefill and 16 decode steps
  (dummy weights accept about 1.9 tokens a step).
- **The grid:** after the request, nine runs of the engine grid's dummy step at the same
  shape (one request, six verification rows), about the request's context.
- Both are traced under IKET, and timed with nsys without it.

| a decode step (nsys, no IKET) | TP4 rank | TP3 rank |
|---|---:|---:|
| GPU busy, request | 14.04 ms | 15.13 ms |
| GPU busy, the grid's step | 8.92 ms | 9.68 ms |
| the grid's own timer (`engine_grid`): step / target / drafter | 10.25 / 2.27 / 6.78 ms | 10.84 / 2.69 / 6.98 ms |

The vocabulary head is whole at TP1: each launch is 4.0 ms, where a TP4 rank's quarter
is about 1.0 ms. So the absolute steps above are long by about 3 ms (the target's
head) plus 3 ms (the drafter's head), and the vocabulary-wide BF16 GEMM below is four
times a rank's too. Every other kernel is at a rank's shapes.

**Where a decode step's time goes** (IKET, 16 decode steps; spans a step, then the
phase split of the launch's last recording warp; loop ranges are estimates, see
Sampling above). TP4 rank:

| kernel (grid) | launches | span a step | last warp: wait / load / compute | its longest wait |
|---|---:|---:|---|---|
| packed vocabulary head (1, 4040) | 2 | 8073 us | 59% / 32% / 10% | the block sync after its cp.async waits |
| L2 prefetch (2) | 25 | 1588 us | load (a side stream; 87% overlapped) | |
| BF16 GEMM (2020), vocabulary-wide (2020 x 64 = 129,280 rows: also whole at TP1) | 5 | 1328 us | 91% / 0% / 4% | its mbarrier wait |
| routed gate-up GEMM (9, 30) | 8.6 | 714 us | 37% / 25% / 36% | the TMA weight tile's mbarrier |
| routed down GEMM (40, 30) | 8.6 | 444 us | 14% / 20% / 0% | |
| MXFP8 GEMMs (80, 64, 28, 36, 40) | 39 | 1425 us | 29-94% wait | `cp_async_wait`, then the sync after it |
| mHC project streams (40) | 18 | 286 us | 17% / 37% / 17% | the block sync |

TP3's is the same order, with larger routed GEMMs: the gate-up at 966 us a step, 58%
of its last warp waiting.

**What a decode step adds to the grid's step** (`step_diff.py`, nsys; "exposed" is time
a kernel runs alone on the GPU, so on the step's critical path):

| a decode step less the grid's | TP4 rank | TP3 rank | exposed |
|---|---:|---:|---|
| the target's verification head: a second packed-head launch the grid does not run | +4.04 ms | +3.99 ms | all |
| the routed GEMMs (dummy rows reach 6 experts, the request's rows more) | +0.96 ms | +1.43 ms | 95% / 87% |
| kernels only a request runs: sampler, sink rejection, verification, inputs | +0.17 ms | +0.16 ms | all |
| the other per-layer kernels (mHC, router, sparse MLA, SwiGLU, combine, rope) | +0.13 ms | +0.21 ms | 31% / none |
| kernels only the grid runs (its dummy fills) | -0.14 ms | -0.12 ms | |

- **The verification head.** A request's decode step runs the packed head twice. The
  target's runs on its verified rows, eagerly after the target's graph (`ced_copy_rows`,
  the head, then `logits_rows`, `sink_cut`, `local_stats`, `verify`). The drafter's runs
  inside its graph. The grid's dummy step runs only the drafter's.
- **The request-only kernels** include the harness's top-5 logprobs (`token_ranks`,
  `topk_log_softmax`, `topk_ids`: 96 us). Serving without logprobs leaves about 76 us:
  sink rejection, rejection sampling, adaptive verification's statistics, inputs.
- **Applied to #616's c1 gap** (TP4, 1025: 11.9 ms, of which routing explains about
  5.3 ms and about 6.5 ms is unattributed GPU work), this accounts for:
  - about 1.0 ms of the target's verification head at a TP4 rank's shard;
  - about 0.08 ms of sampler and sink rejection, all on the critical path;
  - about 0.6 ms of slower per-layer kernels over 43 layers, a third of it exposed.

  That is 1.3-1.7 ms. The rest is not in a one-GPU step at these shapes. It must lie
  in what this harness leaves out: Engram, the collectives with real peers, and the
  request's real context.

**c8 does not run yet: the dummy engine makes NaNs with three or more requests**
(`c8-nan` bundle, runs 33-37; summaries in `runs/c8-nan-*`):
- **Where it shows:** `route_partials` asserts that its gate logits contain NaN or Inf.
  At c8 that happens at boot, in vLLM's sampler warmup (`_warmup_kernels`, eight 7-row
  requests). At c3 and c4 it happens in the requests.
- **Not my choices:**
  - It is the same without IKET.
  - It is the same with the whole model's shapes (TP1).
  - It is the same with the uint8 dummy weights filled with 0x77 instead of zeroed.
  - c1 and c2 run.
- **The first non-finite tensor** (eager, a hook on every module) is the input of layer
  1's `attn.fused_wqa_wkv`: NaN in 36, then 31, of the c3 prefill's 104 rows, scattered
  over all three prompts. Layer 0's module outputs are finite.
- **So** something between layer 0's output and layer 1's attention, in functional mHC
  or norm code that the hooks do not see, reads memory that was never written when a
  batch holds three or more requests. The rows change from run to run.
- **Open:** whether production's real weights hide the same read is not known.

Root-causing it is a follow-up. The c8 bundles (`step-tp4-c8`, `step-tp3-c8`) are ready
for when it is fixed.

## Tracing a multi-rank serving step later (not run)

What run-iket does that matters here:
- It injects through `CUDA_INJECTION64_PATH` and `SMODEL_INJECTION_CONFIG`. vLLM's `mp`
  workers inherit both, so each rank writes its own `iket_pid_*` trace.
- It also sets `CUTE_DSL_COMPILER_OPT=iket`, which keeps CuTe DSL's own IKET ops in any
  CuTe kernel compiled under it.
- It runs the workload twice: the tracker dry run, then the measured run. A server never
  exits, so the dry run must be bounded.

What a serving trace on the cluster would take:
1. **A finite workload.** Wrap each node's `vllm serve` as `run-iket … profile -- vllm
   serve …`, with a driver that sends a fixed request set and then stops the servers. The
   alternative is a dry run of the same workload once, replayed with
   `--use-tracker-dir`. Both passes must run the same launches: run-iket assumes stable
   record counts between them.
2. **Memory.** The pool is one allocation for the whole process, so it grows with every
   instrumented launch of the boot and the workload. Sample (`VLLM_DS41_IKET_CTAS`,
   `VLLM_DS41_IKET_ITERS`), read the size the dry run computes, and lower
   `--kv-cache-memory-bytes` by it.
   - Serving leaves 5–7 GiB free a node, and dgx1's 5 GiB startup guard binds. The pool
     must come out of the KV budget, never out of the guard.
   - A long serving run does not fit at all. Trace a short, fixed workload.
3. **Boot time.** The mode turns TileLang's caches off, so every rank compiles every
   kernel cold. The startup rendezvous waits for alive ranks; the heartbeat timeout only
   matters for a hung rank.
4. **One timeline.** Traces from different GPUs are not on one clock. sparknet's trace
   clock (`sparknet.oneshot._trace_clock`) measures each node's `%globaltimer`-minus-
   `CLOCK_MONOTONIC_RAW` offset with a ping kernel. That ties a rank's IKET timestamps to
   its own host.
   - Across nodes it still needs a shared reference: the hosts' synchronized
     `CLOCK_REALTIME`, or the one-shot all-reduces themselves, whose peer waits end
     within about a microsecond on every rank.
5. **No nsys at the same time.** IKET and nsys both need CUPTI.
6. **A window.** It needs the owner's consent and a cluster window (`bin/spark cluster
   stop/start`), with memguard armed as always. dgx4 cannot do it alone: it is uncabled.

## Found on the way (not fixed here)

1. **A stubbed vision tower still declares its kernels.** With `limit_mm_per_prompt
   {"image": 0}`, DS4.1's TileLang vision tower is a stub with meta weights, but
   `declare_kernels` still declares its calls. Binding them then raises "the TileLang
   vision tower requires CUDA weights" (`nvidia/tilelang_vision.py` `_device`).
   Production allows 4 images and never meets it.
2. **The dead-row warmup assumes two sequences.** `_warmup_dead_row_kernels`
   (`v1/worker/gpu/warmup.py`) launches for one request and for two. At
   `max_num_seqs=1` it copies three `query_start_loc` entries into a two-entry buffer
   and fails. Production runs 16.
3. **vLLM's dummy loader leaves integer tensors uninitialized** (upstream; it zeroes them
   only on ROCm): DS4.1's packed FP4 experts and their UE8M0 scales. More of what the
   TP1 dummy engine reads is never written, so two dummy runs give different tokens.
4. **run-iket's pool does not scale to an engine run** (NVIDIA's tool). It is one
   bump allocation for the whole process, and every warp of every instrumented launch
   reserves the run's largest warp record. Sample, with `VLLM_DS41_IKET_CTAS` and
   `VLLM_DS41_IKET_ITERS`, so that record stays small.
5. **A predicated event takes a record slot** (NVIDIA's tool). run-iket's patch on an
   event's shared store uses a slot each time a warp issues it, whether its predicate
   holds or not, and its dry run counts the same way. A trace never shows those
   slots, and a `--max-ts-cnt-per-warp` below the dry run's count overflows a warp's
   buffer: the exit dump then crashes in `iket::to_json(WarpBuffer)`. ptxas predicates
   short guarded blocks freely, so a guard on an event only saves slots if the event
   cannot be predicated: the IKET mode's zero-or-one loops.
6. **run-iket sizes records from one process's dry run** (NVIDIA's tool). It reads
   `recommended_max_ts_cnt_per_warp` from the first `pid_*` directory it lists and
   stops (`iket.cli.run_iket`). The step harness starts two children that load CUDA and
   launch nothing (count 0). When one lists first, every warp gets a 16-byte header and
   no slots, and the exit dump crashes (run 28 on dgx3: a 0.46 GB pool against 10.3 GB
   needed). `harness/iket_profile.py` runs the dry run alone and passes the largest
   count over its processes to the measured run.
7. **run-iket's perfetto and html postprocesses do not scale either** (NVIDIA's tool).
   They build the whole trace in memory: for a decode-step run's 2.6 GB of decoded
   records they took more than 100 GB (run 29). The json postprocess only copies
   `iket.decoded_results.json`, so the step bundles postprocess nothing and read that
   file.

## Timing a decode step with your own vLLM branch

`step_bundle.py` writes a kernel-local bundle that times DS4.1 decode steps by kernel
and phase, from any vLLM ref that carries the IKET mode.
1. Cherry-pick vLLM branch `iket-debug`'s commits onto your branch. The mode is
   `vllm/models/deepseek_v4_1/tilelang/iket.py`, its variables in `vllm/envs.py`, the
   hook in `vllm/v1/worker/gpu_worker.py` and a test.
2. Write the bundle, here for one TP4 (ring) rank at c1:

   ```sh
   python3 experiments/2026-10-26-iket-debug/step_bundle.py ~/projects/<your-vllm> <ref> <bundle-dir> \
     --tp 4 --requests 1
   ```

   `--tp 3` gives a triangle rank's shapes, and `--requests 8` is c8.
3. On your test node, from a spark-ds41f checkout that has this folder:

   ```sh
   python3 scripts/lab.py kernel-local <bundle-dir> --out <out>
   ```

4. Read `<out>/output.txt`, in this order:
   - the timeline;
   - the engine steps (split at each `prepare_pos_seq_lens_kernel`; the step that
     prepares prefill inputs is the prefill);
   - the overview over the decode steps: each kernel shape's launches and span a step,
     its share, and its last warp's phase split;
   - the waits the launches' last warps spend most time in;
   - each kernel's phase and range tables;
   - the nsys timeline of the same request without IKET;
   - a decode step against the engine grid's dummy step at its shape (`step_diff.py`).
     It gives each kernel's time a step in both, the difference, and its exposed time,
     with IKET's phase split beside it.

   The numbers are in `table.json` and `step-diff.json`, under the node's compile cache
   (`~/.cache/spark-lab/compile/iket/<bundle>-<stamp>/`). A c1 bundle takes 8-10 minutes
   on a free GPU.

What the numbers are:
- **Shapes.** One GPU runs six layers and the drafter with one rank's shapes. The
  vocabulary head and the embedding stay whole: a rank's are a TP-th of them, so the
  head's share of a step is TP times too large.
- **Dummy weights.** Routing, and how many drafts the target accepts, are not real.
- **Under run-iket,** kernels run slower (GEMMs +26-40%, tiny kernels about 3x) and so
  does the host. Read shares and in-kernel phases from IKET, and take absolute times
  from the nsys timeline of the same steps without IKET (the bundle's last step).
- **Sampling.** Four CTAs a launch record ranges, and loops record their middle
  iteration. A loop range with literal bounds is scaled to the whole loop and marked
  "est."; other loop ranges hold their one iteration.

## Reading a trace

```sh
python3 experiments/2026-10-26-iket-debug/iket_table.py RUN_DIR \
  --sources IKET_DIR/sources [--nsys off=off.sqlite on=on.sqlite] [--kernels REGEX] \
  [--between iket_step_marker_kernel [--gap-us 200]] [--json OUT]
```

`RUN_DIR` is run-iket's `--output-dir`; the tool reads its largest `*.trace.json`. The
`.pftrace` beside it opens in https://ui.perfetto.dev.

The table groups launches by kernel name, grid and block. Each group has:
- `span`: first warp start to last warp end, the in-kernel equivalent of nsys's duration;
- `warp`: mean warp lifetime;
- per phase and per range: mean time a warp spends there and its share of the lifetime;
  `unattributed` is the rest of the lifetime;
- the phase split of the warp that ends last, the one the launch waits for.

`--nsys` adds each group's median nsys duration for comparison.

With sampling, the manifests in `--sources` say how each kernel was sampled:
- With sampled CTAs, warp, range and last-warp numbers come from the CTAs that recorded
  ranges. The span stays the whole launch's.
- With sampled iterations, a range inside a loop holds only its sampled iteration. It
  is marked `(sampled iterations)`, and the loop's other iterations count as
  unattributed.

`--between MARKER` keeps the launches between a marker kernel's first and last launch
(`iket_step.py` brackets its measured request with `iket_step_marker`). The output then
starts with a timeline:
- wall time, and the GPU's busy time (the union of the launches' spans);
- the idle gaps, with the kernels on either side of the largest;
- the segments that gaps over `--gap-us` separate (here, the engine's steps).

`nsys_window.py FILE.sqlite` prints the same timeline from an nsys export, with the
kernels by total duration and how many ran in graphs:

```sh
python3 experiments/2026-10-26-iket-debug/nsys_window.py nsys-graphs.sqlite --gap-us 200
```
