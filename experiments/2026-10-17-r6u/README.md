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
  - The TileLang compressor offers B12X no preparation unit: B12X prepares its units
    while memory profiling, and that one compiled an unused CuTe DSL kernel there
    when its cache was cold. Binding the compressor's kernels allocates its storage.
  - The sampler's warmups skip the unaligned-stride variants a 129,280 vocab cannot
    launch (66 of 132). A step's Triton kernels (input preparation, block tables,
    sampling, verification, drafting) register while kernels are prepared, after the
    engine core sets the KV cache layout their block-table variants depend on.
  - The startup rendezvous waits on liveness (each rank's heartbeat: phase and kernels
    prepared), not on a deadline; `VLLM_STARTUP_RENDEZVOUS_TIMEOUT_MS` is gone.
  - Profiling and warm-up name a stopped one-shot collective, as serving steps do.
- **DeepSeek's current reference**, for both V4.1 Flash checkpoints (the original dba1be0a and
  1017's 1bae0e10, whose config, tokenizer and tensor layout are the same):
  - WO-A quantizes its input as the reference's 2026-10-12 revision does (`act_quant(o, 32,
    "ue8m0", inplace=True)`), inside the W8A16 GEMM's tile load (0177); outputs differ from
    r6s by that quantization.
  - The 2026-10-16 encoding fix (0178): the current turn's reasoning survives its tool results,
    tool images follow their placeholders, image tokens in text are refused.
  - The reference's `index_k` fix needs nothing: the indexer reads its owner's cache directly.
- **Recipe**: `models`/`model` select the checkpoint; r6u's arm sets DeepSeek's sampling
  defaults (`generation_defaults`: temperature 1.0, top_p 1.0) and the `-1017` arms serve 1017.
- The collective wait stays 10 s.

## Runs

1. Overlay iterations on the r6s image (changed Python files mounted): boots with every
   TileLang cache empty and with one rank's empty, freeze on, until no kernel is missing.
2. `image.json`: r6u built on dgx2, copied over the CX7 links.
3. `tests.json`: the release-test gate in the image (`config/release-tests.json`: the 58
   DS4.1 Flash gate files, [testing](../../docs/testing.md)).
4. `gate.json`: temperature-0 determinism and distinct streams, identical to r6s.
   `release-tests-discovery.json`: every test file `scripts/release_tests.py` selects
   (the stack's, the series', and those importing a module the series changes: 326)
   run once on the image to learn their state; the release-test gate follows from it.
5. `coldboots.json`: one rank's TileLang cache empty, every rank's empty, then every JIT
   cache empty; each must boot with the 10 s collective wait, and the trace must show no
   compile after preparation.
6. `head-exponent.json`: the packed head's exponent base as a runtime argument versus a
   constant (bit-identical; included only if no decode row count is slower).
7. `release.json`: release-tp3, r6u / r6s / r6u again.
8. On 1017 (the promotion target): `gate-1017.json`, `coldboots-1017.json` and
   `release-1017.json` (r6u / r6s / r6u again, both on 1017); `gate.json` on the original
   checkpoint shows it still serves.

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
- Image, first build (vLLM through 0171, 2026-10-17):
  - Gate: temperature 0 (7748d638 / e27d573d) and greedy (97045b48 / 9e94db6b)
    identical to r6s; 8 and 4 distinct streams pass.
  - One rank's TileLang cache empty: ready in 282 s (dgx2 prepared 158 s, dgx1 and dgx3
    waited 151 s). Every rank's empty: 283 s. No JIT compile after preparation.
  - Every JIT cache empty: ready in 319 s, but each rank compiled B12X's CuTe DSL
    `_Compress` in memory profiling (the TileLang compressor's leftover B12X unit, never
    launched). A rank with only its CuTe DSL cache cold would compile it inside a
    distributed forward. Fixed in 0172; the image is rebuilt and the cold boots rerun.
  - The tests bundle stopped at collection (tests/conftest.py imports tblib, absent from
    the image); it now runs with --noconftest and a plugin with the fixtures it uses.
- Packed head exponent base as a runtime argument: bit-identical at every row count, but
  slower at 13 of 20 TP3/TP4 decode row counts (+0.07% to +0.48%), so it is not included.
  The script exited 0; the queue reports FAIL because the exit verdict also expects a
  pytest summary line, which the script does not print.
- Release tests (2026-10-17): the DS4.1 Flash deployment's 115 selected test files classified
  (58 gate, 57 excluded, 6 tests deselected); nine test files fixed (0173-0176: stale hand-built
  objects, the deployment environment, TileLang coverage, DS4.1's config tests in their own file).
  The gate over the image with the fixed files mounted: 58 files, 1,350 passed, 41 skipped
  (39 need offline Hugging Face configs, the indexer's NaN contract, two GPUs), none failing, 6.2 min.
