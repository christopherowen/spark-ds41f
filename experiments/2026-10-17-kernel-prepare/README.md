# Kernel preparation: no compile inside a distributed phase

## Problem

TileLang kernels compile on first call. About half the DS4.1 factories are compiled ahead
(as side effects of `process_weights_after_loading`, `TileKernelsMHC.__init__`, the metadata
builder and B12X preparation stages); the rest compile inside memory profiling, warm-up
and capture, which are distributed forwards. A rank with a warm cache then waits in a
one-shot collective for a rank that is compiling, and the 10 s wait stops the GPU
(forum topic 384438, post 24; reproduced on r6s with one cold rank, 2026-10-16,
[r6t record](../2026-10-17-r6t/README.md)). Lengthening the wait (r6t) only tolerates the
race; it is not promoted.

## Design

Compiling and distributed execution are separate phases, using TileLang's own machinery.

1. **Variants are TileLang configs.** Each kernel family keeps its `@tilelang.jit` factory
   and publishes `configs(ctx)`: the exact keyword sets the factory can be called with for
   this deployment (model and parallel config, scheduler sizing, KV-cache block sizes and
   dtypes, SM count and L2 size). Row count is always a runtime dimension, so the set
   follows from configuration. The call site picks its config through the same key
   function, so declaration and use cannot drift (this also removes the duplicate keys
   from dtype strings versus TileLang types).
2. **Compile with `JITImpl.par_compile(configs)`** into TileLang's `KernelCache` (memory
   and `TILELANG_CACHE_DIR`), in parallel, instead of one by one inside a forward.
3. **Freeze TileLang's `KernelCache`** (a small patch in `patches/tilelang`, proposed
   upstream): once frozen, a miss that would compile raises with the kernel's name and
   key, identically on every rank. It covers DS4.1's factories, TileKernels and sparknet's
   TileLang collectives.
4. **Startup order:** `load_model` (weights only) -> `prepare_kernels` (local, as long as
   each rank's cache needs) -> freeze -> rendezvous -> profiling, KV init, warm-up,
   capture, serving (no compiles). A first boot compiles; later boots are cache hits.
   No extra command before `cluster start`.
5. **The rendezvous waits on progress, not a deadline.** Each rank reports `prepared n/m`
   through the TCPStore; the rendezvous waits while every missing rank is alive and
   advancing, and fails at once, naming the rank, if it dies or stops advancing.
6. **Weight hooks transform weights only:** block-scale words, the power-of-two fold, the
   MoE tile-major repack, packing the head, the Engram norm product, the mHC broadcast sum.
   Compiles, dispatch tables and registration move out; nothing rides on B12X stages.
7. **Variants depend on configuration only.** The packed head's `exponent_base` (from the
   weights) becomes a kernel argument, pending a never-slower check.

The one-shot collectives keep their 10 s wait throughout: it then means only that a peer
or a link stalled. r6t's profiling and warm-up error naming (a stopped collective named
as the cause) is kept.

## Measurement (before building)

`jittrace.json`: two traced boots of r6s, each followed by the gate's determinism text and
eight distinct streams. `jittrace/sitecustomize.py` (loaded through `PYTHONPATH`, no
computation changed) records every TileLang compile with the factory call that caused it,
TileLang disk loads, Triton and CuTe compiles, and the worker's phase boundaries.

- `tilelang-cold`: an empty TileLang cache on every rank (identical nodes compile in
  near-lockstep, so this boots, as r6n's 409 s empty-cache boot did).
- `all-cold`: every JIT cache empty (TileLang, Triton, CuTe DSL, B12X, TorchInductor,
  vLLM, FlashInfer, the driver's PTX cache).

`summarize_jittrace.py` lists every kernel compiled after `load_model`: the prepare
stage's checklist, by family and phase.
