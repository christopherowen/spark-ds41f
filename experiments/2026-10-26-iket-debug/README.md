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

<results below>

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
