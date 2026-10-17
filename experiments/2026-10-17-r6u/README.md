# r6u: every kernel prepared before any distributed work

r6u is r6s ([experiments/2026-10-16-r6s](../2026-10-16-r6s/README.md)) with kernel
preparation ([design](../2026-10-17-kernel-prepare/README.md)).

## Why

Kernels compiled on first call, and many first calls landed inside memory profiling and
warm-up, which are distributed forwards: a rank with a warm cache waited in a one-shot
collective for a rank that was compiling, and the 10 s wait stopped the GPU (forum topic
384438, post 24; reproduced on r6s with one cold rank). The traced cold boot
([kernel-prepare](../2026-10-17-kernel-prepare/README.md)) found 73 TileLang compiles
(about 116 s) in memory profiling and one in warm-up, and 12 Triton compiles in profiling.

## Change

- **TileLang 0004** ([tilelang/series](tilelang/series)): `tilelang.prepare(calls)` compiles
  call forms in one thread pool and keeps each kernel under the key its call computes;
  `tilelang.cache.freeze(reason)` refuses every kernel not in memory, so a missing one fails
  by name on every rank alike, whatever each rank's disk cache holds.
- **vLLM**:
  - `tilelang/prepare.py`: owners declare their calls (`declare(calls, bind, warm)`),
    found by a walk of the loaded model (`declare_kernels`, `declare_layer_kernels`).
  - The worker prepares before memory profiling: declared TileLang calls (one pool),
    TileKernels warm calls on placeholders, the JIT registry (Triton), then freezes, binds
    and starts the JIT monitor. Profiling, warm-up, capture and serving compile nothing.
  - Every TileLang family declares; weight processing transforms weights only.
  - The sampler's warmups skip the unaligned-stride variants a 129,280 vocab cannot
    launch (66 of 132). A step's Triton kernels (input preparation, block tables,
    sampling, verification, drafting) register while kernels are prepared, after the
    engine core sets the KV cache layout their block-table variants depend on.
  - The startup rendezvous waits on liveness (each rank's heartbeat: phase and kernels
    prepared), not on a deadline; `VLLM_STARTUP_RENDEZVOUS_TIMEOUT_MS` is gone.
  - Profiling and warm-up name a stopped one-shot collective, as serving steps do.
- **Serving is unchanged**: the same kernels and dispatch, the 10 s collective wait.

## Runs

1. Overlay iterations on the r6s image (changed Python files mounted): boots with every
   TileLang cache empty and with one rank's empty, freeze on, until no kernel is missing.
2. `image.json`: r6u built on dgx2, copied over the CX7 links.
3. `tests.json`: the prepare, rendezvous, worker and kernel tests in the image.
4. `gate.json`: temperature-0 determinism and distinct streams, identical to r6s.
5. `coldboots.json`: one rank's TileLang cache empty, every rank's empty, then every JIT
   cache empty; each must boot with the 10 s collective wait, and the trace must show no
   compile after preparation.
6. `head-exponent.json`: the packed head's exponent base as a runtime argument versus a
   constant (bit-identical; included only if no decode row count is slower).
7. `release.json`: release-tp3, r6u / r6s / r6u again.

## Results

- Overlay iteration 4 (2026-10-17, r6u's Python files over the r6s image, TileLang frozen,
  JIT monitor warn):
  - Every TileLang cache empty: ready in 284 s (r6n cold: 409-426 s).
  - dgx2's alone empty: ready in 282 s. dgx2 prepared for 158 s; dgx1 and dgx3 prepared
    in 7 s and waited 151 s for it on liveness, with the 10 s collective wait unchanged.
  - No rank reported a JIT compile after preparation: profiling, warm-up, capture, the
    determinism scenarios (logprobs included) and 8 distinct streams.
  - Temperature-0 determinism identical to r6s (reference 7748d638, long e27d573d), no
    token or logprob difference in any scenario.
