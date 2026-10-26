# IKET debug mode: phase traces from inside the DS4.1 TileLang kernels

Until now we could only see the serving kernels from the outside: nsys gives each
launch's start and end. IKET is NVIDIA's in-kernel event tracing (CuTe DSL's
`run-iket`), and TileLang's `tilelang.tools.cuda.iket` is its TileLang front end. With
this mode, every TileLang kernel carries IKET ranges around its phases: where it loads,
computes, waits and stores. A trace shows, warp by warp, how long each phase takes.

The mode is debug only and off in production. Off means no IKET import, no TileLang
callback, and generated code identical to the build without the mode.

- vLLM: branch `iket-debug` (worktree ~/projects/vllm-iket), cut from next-candidate
  577091997. Not part of next-candidate.
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
  sites: range name, phase, kinds, source lines and first statement.

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

## Reading a trace

```sh
python3 experiments/2026-10-26-iket-debug/iket_table.py RUN_DIR \
  --sources IKET_DIR/sources [--nsys off=off.sqlite on=on.sqlite] [--kernels REGEX] [--json OUT]
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
