# IKET debug mode: phase traces from inside the DS4.1 TileLang kernels

Until now we could only see the serving kernels from the outside: nsys gives each
launch's start and end. IKET is NVIDIA's in-kernel event tracing (CuTe DSL's
`run-iket`), and TileLang's `tilelang.tools.cuda.iket` is its TileLang front end. With
this mode, every TileLang kernel carries IKET ranges around its phases: where it loads,
computes, waits and stores. A trace shows, warp by warp, how long each phase takes.

The mode is debug only and off in production. Off means no IKET import, no TileLang
callback, and generated code identical to the build without the mode.

- vLLM: branch `iket-debug` (worktree ~/projects/vllm-iket, local only), cut from
  next-candidate 577091997, head 736e86a25: the mode, then the MoE output fix (below)
  cherry-picked from next-candidate's e846e64dc. Not part of next-candidate.
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

### Step 4 results: c1 and c8 decode steps at the ring's and the triangle's rank shapes (dgx3)

dgx3, vLLM 736e86a25 (with the MoE output fix below).
- **The requests:** c1 is a 32-token prompt and 32 tokens, as one prefill and 16 decode
  steps (dummy weights accept about 1.9 tokens a step). c8 is eight such requests
  together.
- **The grid:** after the requests, nine runs of the engine grid's dummy step at the same
  shape (one or eight requests, six verification rows each), about the requests'
  context.
- **Timing:** nsys without IKET times the steps; IKET traces them for in-kernel phases.
- **Final timings: run 47** (`time-tp{4,3}-c{1,8}`, receipts `runs/time-*-47`).
  - The recipe's environment (`RECIPE_ENV`): breakable CUDA graphs off, late admission,
    32 device connections.
  - The recipe's sizing: 16 or 8 streams, a 1M context, prefix caching.
  - The recipe's CUDA-graph capture ladder.
- **Phase splits: run 46's IKET runs** (`runs/step-*-46`), which ran the same
  environment at an 8K context.
- **Earlier runs, superseded:**
  - **Runs 42 and 43** (`runs/step-*-42`, `runs/step-tp3-c1-43`) left vLLM to switch
    breakable CUDA graphs on: it does for DS4.1 unless the variable is set. They also
    ran the harness's small sizing and capture sizes.
  - **Runs 32 and 33** predate the MoE fix: each layer read a finite but wrong MoE
    output.
- **What changed between them:**
  - The recipe's environment and sizing moved a c1 step by under 0.2 ms; the 1M
    context adds six `indexer_topk_merge` launches a step.
  - The ladder took about 1.2 ms off a c8 step. With only 6, 12 and 48 rows captured, a
    c8 step's 16 verified rows ran the 48-row graph.

| a decode step (nsys, no IKET) | TP4 c1 | TP3 c1 | TP4 c8 | TP3 c8 |
|---|---:|---:|---:|---:|
| GPU busy, requests (run 47) | 14.30 ms | 15.51 ms | 16.15 ms | 17.34 ms |
| GPU busy, the grid's step (run 47) | 8.87 ms | 9.73 ms | 13.30 ms\* | 12.21 ms\* |
| the grid's own timer (`engine_grid`): step / target / drafter | 10.07 / 2.20 / 6.68 ms | 11.03 / 2.79 / 6.99 ms | 14.73 / 3.53 / 9.90 ms | 13.25 / 4.06 / 8.04 ms |
| runs 42-43, for comparison: requests / grid | 14.26 / 8.76 | 15.48 / 9.57 | 17.12 / 11.38 | 18.45 / 12.31 |

\* **c8's grid is not yet matched.**
- **Rows:**
  - A c8 request step verifies 2 rows a request: adaptive verification, 16 rows. Its
    router launches show grid 8 against the grid's 24.
  - The grid's dummy step verifies 6 a request. So run 47's c8 grid does about three
    times the request's per-layer target work.
  - The c8 bundles now give the grid 2 rows a request (`--grid-rows`). Their re-run is
    held: the nodes went into the production ring at 21:38 CET.
- **TP4's grid:**
  - On the 16-stream engine, the grid's 8-request dummy step runs the drafter's
    head over two row tiles (grid (2, 4040), 5.8 ms) and `partial_argmax` over 16
    rows.
  - The c8 request step runs (1, 4040) at 4.4 ms over 8 rows, and so do TP3's grid
    (8 streams) and every c1 grid.
  - That is about 1.4 ms in TP4's c8 grid that an 8-request step does not do.

The vocabulary head is whole at TP1: each launch is 4.0-4.7 ms, where a TP4 rank's
quarter is about 1.0 ms. So the absolute steps above are long by about 3 ms (the
target's head) plus 3 ms (the drafter's head), and the vocabulary-wide BF16 GEMM is
TP times a rank's too. Every other kernel is at a rank's shapes. The model has 40
target layers and the drafter's 3, where this harness has 6 and 3.

**A decode step by phase** (run 47, nsys, no IKET; kernel time a step, then the part
exposed: time a kernel runs alone on the GPU, so on the step's critical path):

| phase | TP4 c1 | TP3 c1 | TP4 c8 | TP3 c8 |
|---|---:|---:|---:|---:|
| packed vocabulary head (two launches, whole) | 8123 / 8123 us | 8163 / 8163 | 8798 / 8798 | 8834 / 8834 |
| vocabulary-wide BF16 GEMM (whole) | 1312 / 1312 | 1334 / 1334 | 1421 / 1421 | 1437 / 1437 |
| dense GEMMs (MXFP8, FP8 weights, BF16 partials, split-K) | 2097 / 1156 | 2929 / 1750 | 2195 / 1442 | 3125 / 2169 |
| routed experts (router, dispatch, GEMMs, SwiGLU, combine) | 1429 / 1253 | 1929 / 1505 | 1677 / 1491 | 2024 / 1716 |
| L2 prefetch (a side stream) | 1724 / 244 | 1847 / 245 | 1845 / 219 | 1978 / 215 |
| mHC and norms | 622 / 92 | 643 / 87 | 918 / 136 | 911 / 120 |
| attention (sparse MLA, indexer, compressor, cache, rope, metadata) | 591 / 198 | 607 / 203 | 773 / 317 | 784 / 315 |
| sampler, verification, sink rejection, the harness's logprobs | 173 / 173 | 172 / 172 | 367 / 367 | 358 / 358 |
| inputs, copies, others | 55 / 55 | 55 / 55 | 90 / 90 | 87 / 87 |

**Where a decode step's time goes inside the kernels** (run 46's IKET, 17 decode steps;
spans a step, then the phase split of the launch's last recording warp; loop ranges are
estimates, see Sampling above). TP4 c1:

| kernel (grid) | launches | span a step | last warp: wait / load / compute | its longest wait |
|---|---:|---:|---|---|
| packed vocabulary head (1, 4040) | 2 | 8107 us | 65% / 29% / 11% | the block sync after its cp.async waits |
| L2 prefetch (2) | 25 | 1589 us | load (a side stream; 86% overlapped) | |
| BF16 GEMM (2020), vocabulary-wide (2020 x 64 = 129,280 rows: also whole at TP1) | 5 | 1331 us | 89% / 0% / 5% | its mbarrier wait |
| routed gate-up GEMM (9, 30) | 8.6 | 817 us | 46% / 20% / 29% | the TMA weight tile's mbarrier |
| routed down GEMM (40, 30) | 8.6 | 495 us | 14% / 20% / 0% | |
| MXFP8 GEMMs (80, 64, 28, 36, 40, 40) | 48 | 1585 us | 29-77% wait | `cp_async_wait`, then the sync after it |
| mHC project streams (40) | 18 | 293 us | 18% / 31% / 17% | the block sync |

TP3's is the same order, with larger routed GEMMs.

c8's IKET runs replayed the 48-row graph (their capture sizes stop at 6, 12 and 48),
where the recipe's ladder replays a 16-row one. There:
- one of the two head launches computes more: 24% of its last warp, against 10%;
- the routed GEMMs' grids grow from (9, 30) and (40, 30) to (9, 288) and (40, 288);
- the down GEMM's last warp spends 56% loading (TP4, run 42).

**What a decode step adds to the grid's step** (`step_diff.py`, nsys; kernel time,
then exposed).
- **c1** is run 47: both sides verify 6 rows.
- **c8** is run 46, whose short capture list ran both sides at 48 rows. So it is
  matched, but at more rows than a real c8 step verifies.
- **TP4 c8's grid** carries the drafter's 16-request head and argmax, about 1.4 ms (see
  above), so its head and BF16 rows read low.
- `runs/time-tp{4,3}-c1-47/step-diff.md` has each kernel, with run 46's IKET phase split
  beside it.

| a decode step less the grid's | TP4 c1 | TP3 c1 | TP3 c8 (48 rows) | TP4 c8 (48 rows; grid padded) |
|---|---:|---:|---:|---:|
| the target's verification head: a second packed-head launch the grid does not run | +4.05 / +4.05 ms | +4.08 / +4.08 | +4.34 / +4.34 | +3.21 / +3.21 |
| the routed GEMMs (dummy rows reach 6 experts, the requests' rows more) | +1.19 / +1.12 | +1.64 / +1.38 | +1.58 / +1.48 | +1.19 / +1.09 |
| the other per-layer kernels (mHC, router, sparse MLA, SwiGLU, combine, rope) | +0.22 / +0.02 | +0.27 / -0.12 | +0.35 / +0.01 | -0.16 / -0.42 |
| L2 prefetch (a side stream) | +0.10 / -0.02 | +0.12 / -0.01 | +0.20 / 0.00 | +0.15 / +0.01 |
| the vocabulary-wide BF16 GEMM (whole) | +0.02 / +0.02 | +0.05 / +0.05 | +0.10 / +0.10 | -0.20 / 0.00 |
| kernels only a request runs: sampler, sink rejection, verification, inputs | +0.07 / +0.07 | +0.07 / +0.07 | +0.12 / +0.12 | +0.13 / +0.13 |
| the harness's top-5 logprobs (`token_ranks`, `topk_log_softmax`, `topk_ids`) | +0.09 / +0.09 | +0.09 / +0.09 | +0.11 / +0.11 | +0.12 / +0.12 |
| kernels only the grid runs (its dummy fills) | -0.11 / -0.11 | -0.13 / -0.13 | -0.15 / -0.15 | -0.16 / -0.16 |
| **total (GPU busy)** | **+5.43 ms** | **+5.79** | **+6.31** | **+4.13** |

A negative exposed time means the grid's step ran those kernels alone for longer than
the requests' step did.

- **The verification head.** A request's decode step runs the packed head twice. The
  target's runs on its verified rows, eagerly after the target's graph (`ced_copy_rows`,
  the head, then `logits_rows`, `sink_cut`, `local_stats`, `verify`). The drafter's runs
  inside its graph. The grid's dummy step runs only the drafter's.
- **The routed GEMMs** match #616's routing estimate. TP4 c1's +1.19 ms over this
  harness's 9 MoE layers is about 5.7 ms over the model's 43 (+1.12 ms exposed: 5.4
  ms), against #616's 5.3 ms.
- **Applied to #616's c1 gap** (TP4, 1025: 11.9 ms, of which routing explains about
  5.3 ms and about 6.5 ms is unattributed GPU work), run 47 accounts for:
  - about 1.0 ms of the target's verification head at a TP4 rank's shard (4.05 / 4),
    all on the critical path;
  - about 0.07 ms of sampler, sink rejection, verification and inputs, all exposed
    (serving asks no logprobs);
  - about 1.0 ms of slower per-layer kernels over 43 layers (0.22 ms x 43 / 9), of
    which about 0.1 ms is exposed;
  - about 0.5 ms more L2 prefetch over 43 layers, on its side stream (none exposed);
  - under 0.01 ms of the vocabulary-wide BF16 GEMM at a rank's quarter.

  That is about 1.2 ms on the critical path, of up to 2.6 ms of kernel time.
  - **Runs 42 and 43** gave the same 1.2 ms on the critical path, of up to 2.7 ms of
    kernel time.
  - **Runs 32 and 33** gave 1.3-1.7 ms. They scaled the vocabulary-wide BF16 GEMM's
    difference by 43 / 9 with the per-layer kernels. That GEMM runs five times a
    step, not once a layer, and it was most of what was exposed.

  The other 5.3 ms is not in a one-GPU step at these shapes. It must lie in what this
  harness leaves out: Engram (layers 1 and 14), the collectives with real peers, and
  the request's real context.
- **c8 against c1** (TP3, run 46, each matched to its grid): a c8 step adds about 0.3 ms
  more than a c1 step does (+6.31 against +6.00 ms).
  - The head's launches run longer: 4.4 ms against 4.0.
  - The request-only kernels take 0.12 ms.
  - The routed difference stays about the same, because the grid's eight dummy
    requests already reach more experts (0.68 ms against 0.11 at c1).

  A matched c8 at the recipe's ladder (2 rows a request on both sides) waits for the
  held re-run.

**c8 found a vLLM bug: at TP1 the routed experts' output lived in the workspace arena**
(runs 33-41; summaries in `runs/c8-nan-*` and `runs/nan-*`):
- **The symptom:** `route_partials` asserts that its gate logits contain NaN or Inf.
  - At c8 it fires at boot, in vLLM's sampler warmup (eight 7-row requests).
  - At c3 and c4 it fires in the requests.
  - It happens without IKET, at the whole model's shapes, and with other dummy weights.
- **Trees:**
  - next-candidate c579b3fe1 fails (`nan-next`).
  - 1025's runtime tree fails the same way (26382f5cf, `nan-1025`; 0489-0490 change
    only tests). It predates both, so there is nothing to bisect.
- **The cause:**
  - `FusedMoEKernelModularImpl.apply` (`fused_moe/modular_kernel.py`) returned
    `output = workspace[2]` whenever it was handed a workspace.
  - The MoE runner hands it one when the shared expert needs scratch
    (`prepare_workspace`), so the layer's output was a view into the workspace arena.
  - With the shared expert folded into the TileLang combine, the output is BF16, so
    nothing casts it. At TP1 no all-reduce copies it either.
  - The next decoder layer's mHC entry (`TileKernelsMHC.pre`) leases its scratch from
    offset 0 of the same arena (`get_simultaneous`). It overwrote the routed output
    while its CTAs were still reading it.
- **The evidence:**
  - Eager, with a hook on every module and on each layer's functional `mhc.pre`: layer
    1's `mhc.pre` turns its `previous_output` (layer 0's MoE output) non-finite.
  - That tensor lies in workspace lane 0 at c1 too (run 40), where it is silently wrong
    rather than NaN. At three or more requests the corruption reaches NaN, on rows that
    vary from run to run.
- **Production:** TP4 and TP3 are not exposed. The routed output goes through the TP
  all-reduce, whose paths all write a fresh tensor (sparknet's one-shot and
  hierarchical, the two-shot, the rank-order path), and the drafter runs at the
  target's TP. TP1 engines are exposed: single-GPU tests and harnesses like this one.
- **The fix** is vLLM branch `moe-output-fresh` e846e64dc, over next-candidate
  c579b3fe1.
  - The output is always a fresh tensor. The experts still write it directly
    (`output_alias`), so no copy is added.
  - Its test, `tests/kernels/moe/test_moe_output_outlives_workspace.py`, fails on
    c579b3fe1 and passes with the fix.
  - With the fix, c1 shows no alias, and c3 (eager and graphs) and c8 run (`nan-fix`,
    run 41).
- **Reproduction:** bundle `nan-next` (`harness/iket_step.py off --tp-shape 4
  --requests 3 --eager --nan-hooks`), about 2 minutes.

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
8. **A run-iket dry run sometimes stops launching** (runs 30 and 42 on dgx3; under
   investigation).
   - **Run 42's TP3 c1 dry run:**
     - It made the measured request's first 463 launches exactly as run 33's passing
       dry run did. The last was `padding_rows_kernel`, after the drafter's graph.
     - It then never reached the next graph launch's callback. In the passing run that
       launch follows in the same millisecond, with no wait between.
     - It logged nothing more until `iket_profile.py`'s 1800 s timeout killed it.
   - **Catching it:** run 43 repeated that dry run eight times (`dry-loop-tp3-c1`)
     under `hangcatch.sh`. When a dry run's log stops growing, the catcher records:
     - GPU power and clocks;
     - each thread's state, wait channel and kernel stack;
     - py-spy and gdb backtraces;
     - then a cuda-gdb attach.

     Dry runs 2 and 4 hung; the other six took 139-142 s each. With run 42's, that is
     about one dry run in four.
   - **Where it stops:**
     - **The host:** the main thread is in `CUDAGraph.replay` (vLLM's `run_fullgraph`),
       inside `cuGraphLaunch`. There run-iket's injection (`libsmodel_injection.so`)
       calls back into libcuda and busy-waits, before it logs the launch.
     - **The other threads:** idle, including the injection's own worker.
     - **The GPU:** 17 W at 2522 MHz and 96% busy, against 10.9 W and 0% when idle. So
       the host waits for GPU work that never completes.
   - **The common window:**
     - The eager launches since the previous graph's sync are `metadata_builder_kernel`
       and `padding_rows_kernel`. Of the three hangs with logs, two stopped before the
       drafter's query graph and one before the target's graph.
     - Neither kernel can spin: no loop without a bound, no atomic, no flag, no PDL.
     - The tracker's patch replaces each `PMTRIG` marker with
       `UIADD3.64 UR0, UR0, 0x4` (a warp's count in a uniform register pair the
       kernel does not use).
     - It leaves the marker's store to shared address 0x20 executing. That address is
       in the 1 KB each CTA reserves for the system: the compiler starts user shared
       memory at 0x400 (`UMOV UR4, 0x400` in `mhc_project_streams`' SASS).
     - Measured runs, which replace that store, have never hung.
   - **cuda-gdb cannot attach to the hang:** its attach makes an inferior call on the
     stuck main thread. That attach also re-initializes the injection, which truncates
     its log. So the catcher now copies the log first.
   - **The catcher's first capture was a false alarm.** TP3 c1's dry run had finished:
     its log ends in `Finalize Dump Json`. The live harness was the measured run,
     decoding its records at exit (`iket::to_json`, GPU idle). The catcher now skips a
     log that ends that way.
   - **What it needs (runs 44, 46 and 47):** the same TP3 c1 dry run, eight at a time.

     | setup | hung |
     |---|---:|
     | the mode on, with CUDA graphs (runs 42, 43, 45 and 46's dry runs) | 3 of 18 |
     | `metadata_builder` and `padding_rows` uninstrumented (`dry-skip-tp3-c1`, runs 44 and 47) | 2 of 16 |
     | the mode off, so the tracker sees no events (`dry-off-tp3-c1`) | 0 of 8 |
     | no CUDA graphs (`dry-eager-tp3-c1`, run 46) | 0 of 8 |

     So it needs IKET events and CUDA graph launches under the tracker. It does not
     need the events of the two kernels before the graph launch it stops at.
   - **Where the fault is:** run-iket's injection, NVIDIA's closed tool. Its
     graph-launch callback waits for GPU work that does not complete. The kernels
     queued before that launch cannot spin.
     - The tracker leaves each marker's store to reserved shared memory executing.
     - Measured runs replace that store, and none of them has hung.
     - The next step is a reproduction small enough for NVIDIA: one tracked TileLang
       kernel and a CUDA graph, launched in a loop under `run-iket --skip-run`.
   - **Until then, the step bundles' dry run hangs about one time in five.**
     `iket_profile.py` kills it after 1800 s and the bundle fails. Run the bundle
     again.

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
- **The recipe's environment and sizing** (`RECIPE_ENV`, `RECIPE_SIZING` in
  `make_bundles.py`):
  - breakable CUDA graphs off, so torch.compile on, as the recipe serves;
  - late admission at layer 4 of 6 (the recipe's 30 of 40);
  - 32 device connections;
  - for the timed (nsys) run, 16 (TP4) or 8 (TP3) streams, a 1,048,576-token context
    and prefix caching.

  The IKET run keeps an 8K context and as many streams as requests. At 1M, the
  warmup and capture launch the indexer's top-k with up to a million warps.
  run-iket reserves a record for every warp: 37.6M warps and a 17.2 GB pool. Its
  exit decode then ran the node out of memory (run 45). So the indexer kernels' phase
  splits come from 8K-context launches.

  Left out:
  - the multi-rank settings;
  - TP3's display carve-out (a whole head does not fit);
  - the batched-token budget, kept small so a run-iket pool stays small. It shapes
    the profile run and prefill chunks, not a decode step.
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

## Perfetto: a decode step lane by lane

`iket_perfetto.py` turns one decode step of a run-iket run into a Perfetto trace:

```sh
python3 experiments/2026-10-26-iket-debug/iket_perfetto.py \
  RUN/trace/iket/pid_*/iket.decoded_results.json OUT.pftrace [--layer T2] [--step K]
```

- **Streaming.**
  - The 3-4.5 GB records file is never loaded whole. A first pass indexes each launch
    record (its offset, kernel, span and graph exec), from an mmap and one
    `raw_decode` call an element.
  - The chosen step's records and the warp locations they use are then decoded.
  - TracePackets (protobuf, interned names) are written as they are made.
  - About 25 s and 0.44 GB of memory a file, whatever the records' size.
- **Opening it:** drag the file onto https://ui.perfetto.dev, which parses it in the
  browser; nothing is uploaded. A file too large for a tab opens through a local trace
  processor:

  ```sh
  trace_processor_shell --httpd OUT.pftrace
  ```

  Then ui.perfetto.dev offers to use it.
- **What it covers:**
  - what IKET recorded, which is the TileLang kernels only. ATen and Triton kernels and
    copies are absent;
  - from the harness: 6 target layers, the drafter's 3 and the head, at one rank's
    shapes. Not the model's 43 layers;
  - times are IKET's (`%globaltimer`), including its instrumentation. Run 46's TP4 c1
    step spans 35.9 ms under IKET, against 14.5 ms in nsys.
- **Tracks** (one GPU process):
  - **1 model:**
    - **step;**
    - **segment:** graph replays and eager runs, named from the whole step (the target's
      forward, the verification head and sampler, the drafter);
    - **layer:** T0-T5 and D0-D2, cut at each MoE combine, with the all-reduce points as
      markers. One GPU runs none: the markers show where a TP rank would wait;
    - **sublayer,** from kernel names.
  - **2 kernel launches:** first warp start to last warp end, in lanes where launches
    overlap.
  - **3 counters:**
    - CTAs and warps resident;
    - the share of traced warps waiting, by reason (`wait.*` ranges): cp.async,
      mbarrier (TMA), block barrier, PDL dependency, flag or spin, fence.
  - **4 SMs:**
    - SM 00-47, each with CTA slots for CTAs that overlap there;
    - each CTA's slice spans its warps' lifetimes;
    - under a slot, each traced warp's track holds its ranges nested in its lifetime.
- **Sampling:** the run's.
  - Every warp's lifetime is recorded, so CTA slices and counters cover every CTA.
  - Phases cover the 4 traced CTAs a launch (`VLLM_DS41_IKET_CTAS=4`).
  - Loops show their middle iteration, with `*N` the factor to the whole loop.
  - `--max-traced-warps N` thins a CTA's traced warps further.

- **Real times, for now: `--base-nsys EXPORT`.**
  - The timeline becomes an IKET-off run's nsys export of the same step: the interim
    stand-in for a `--warp-trace` base run.
  - **Pairing:** each IKET launch pairs with the base launch at the same ordinal, its
    place among the step's launches of one kernel, grid and block. Shapes whose counts
    differ pair by kernel name; a name whose counts differ stays unpaired.
  - **Rescaling:** the launch's CTAs, warps and phases are rescaled into the base
    launch's span. Kernel spans and overlap are the base's.
  - **Per-launch args:** `real_us`, `instrumented_us`, the scale and the overhead.
  - **Flags:**
    - overhead above 50%: phase proportions unreliable, mostly tiny kernels;
    - a paired grid that differs: phases from a launch of another shape.
  - **Non-TileLang kernels:** a "5 streams" group shows the base's every kernel per CUDA
    stream, ATen and Triton included.
  - **Pairing on run 46's records:**
    - TP4 and TP3 c1 against run 47 (the recipe's ladder): 431 of 432 launches pair. The
      exception is `indexer_topk_merge`, 1 launch at 8K against 7 at 1M. 74 and 77 pairs
      differ in grid: the drafter's 10 rows against 5, and the indexer's context.
    - TP4 c8 against run 47 does not pair: its 16-row graphs against the IKET run's 48
      rows give 178 different grids and other L2 prefetch counts. Against run 46's own
      nsys (48-row graphs, a 1M context) 419 of 420 pair, 8 with other grids.
  - **Small kernels' times:** nsys's duration includes a kernel's launch and drain, which
    IKET's warp spans do not. So a small kernel's instrumented time can read below its
    real time.

Files from run 46 (dgx3, vLLM 736e86a25), in `~/projects/commander/outputs/iket-perfetto/`:

| file | what | size |
|---|---|---:|
| `harness-tp4-c1-layerT2-736e86a2-run46-dgx3-20261026.pftrace` | TP4 c1, target layer T2 (compressor, indexer, sparse MLA, WO, MoE): 51 launches, 2,960 CTAs, 864 traced warps | 1.1 MB |
| `harness-tp4-c1-decode-step-...` | TP4 c1, decode step 9 of 18: 432 launches, 43,793 CTAs, 7,097 traced warps | 12 MB |
| `harness-tp4-c8-decode-step-...` | TP4 c8, decode step 9: 420 launches, 174,866 CTAs | 29 MB |
| `harness-tp3-c1-decode-step-...` | TP3 c1, decode step 9: 432 launches, 49,776 CTAs | 13 MB |

Interim files, timed by an nsys base (`--base-nsys`), labelled INTERIM in their names:

| file | base | step span |
|---|---|---:|
| `harness-tp4-c1-layerT2-INTERIM-timed-by-nsys-run47-...` | run 47 TP4 c1 | 632 us (the layer) |
| `harness-tp4-c1-decode-step-INTERIM-timed-by-nsys-run47-...` | run 47 TP4 c1 | 14.47 ms |
| `harness-tp3-c1-decode-step-INTERIM-timed-by-nsys-run47-...` | run 47 TP3 c1 | 16.20 ms |
| `harness-tp4-c8-decode-step-INTERIM-timed-by-nsys-run46-...` | run 46's own nsys, 48-row graphs | 17.56 ms |

### Next: a full forward pass from one base run and a lean run (needs a test box)

`full-pass-spec.md` is the spec for a lab window: which runs, settings, sizes, times per
configuration, hang retries and stitching. The sketch below is its background.

These need GPU runs:
- **Payloads, for what a warp does:**
  - TileLang's IKET records unsigned 32-bit or float32 runtime payloads
    (`iket.session(runtime_payloads=True)`, `TL_IKET_EVENT_PAYLOAD_U32`).
  - The mode would attach per-kernel values to range starts: the warp's role
    (producer or consumer warp group), the tile, the expert, the row block.
  - Off stays identical, because the payloads exist only when the mode is on.
  - Bytes per load phase come from the load's tile size at codegen time, recorded in
    the manifest rather than at run time.
- **A base run:**
  - The deterministic request: temperature 0, a fixed prompt and batch, a fresh
    prefix cache, the recipe's environment and sizing.
  - It runs under `run-iket profile --warp-trace`: the IKET mode off, no tracker pass.
  - Every kernel launch (ATen, NCCL and sparknet included) records its warps' start
    and end, one slot a warp. That is the timeline of record, with a small pool.
- **Slice runs:**
  - The same request again, with the mode on for one kernel family a run (attention,
    MoE, GEMMs, head and sampler, inputs), so each run-iket pool stays small.
  - `VLLM_DS41_IKET_SKIP` selects the family.
- **Stitching, by launch ordinal:**
  - The key of a launch:
    - eager launches: their order in the step;
    - graph launches: the graph exec, the replay's ordinal in the step and the
      record's index in the replay.
  - Each slice's launch sequence (kernel, grid, block) must match the base run's at
    every ordinal, or the stitch refuses.
  - A slice's warps are scaled into the base launch's span, keeping both the raw and
    the scaled times.
- **The cluster:**
  - The same spec on all four ranks, giving one file per node.
  - A combined file aligns the ranks at the collectives, which are rendezvous, by a
    fitted clock offset and drift with its residual.
  - Each rank's wait at every collective shows directly.
