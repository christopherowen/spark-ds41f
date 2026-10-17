# r6u: every kernel prepared before any distributed work

**Promoted for TP3 on 2026-10-17** as [`2026-10-17-karmic-kraken-r6u`](../../manifests/baselines/2026-10-17-karmic-kraken-r6u.json), serving DeepSeek-V4.1-Flash-1017 (owner: promote on the overlay's numbers, without benching the rebuilt image again), on the overlay release-suite run of 0179 ([suite run](../../manifests/suites/2026-10-17-karmic-kraken-r6u-tp3/)). `tp4-ring` resolves to r6u but stays unqualified until a TP4 release (`r6u-tp4-1017.json`, `gate-tp4-1017.json`, `coldboots-tp4-1017.json`, `release-tp4-1017.json`). The r6s recipe this release ran against is frozen in [r6s/cluster.json](r6s/cluster.json) with its lock; the arms built on it extend the frozen copy.

### Release results (2026-10-17, TP3, 1017)

On the image through 0178, the suite ran r6u (13:51-14:31), r6s (14:33-15:13) and r6u again
(15:15-15:55):

```
verdicts: r6s-1017 -> r6u-1017, against the run's own repeat (r6u-1017 -> r6u-1017-again)
metric                             change    ±95%   repeat  verdict
grid: decode geometric mean (15)   +0.68%   0.14%   -1.52%  same
c8: output tok/s                   -0.35%   1.46%   +0.03%  same
c4: output tok/s                   -0.88%   1.45%   -0.23%  same
poisson: output tok/s              -0.19%  21.12%   +0.02%  same
prefill: prefill 32,768 tok/s      -3.11%   0.89%   -0.17%  worse
prefill: prefill 131,072 tok/s     -3.15%   0.65%   -0.21%  worse
prefill: prefill 262,144 tok/s     -3.01%   0.45%   -0.14%  worse
prefill: prefill 500,000 tok/s     -2.77%   0.28%   -0.10%  worse
```

0177 quantized WO-A's input in the GEMM's activation tile loads, once per column tile.
0179 prepares it once per row in the sparse MLA kernels' output store. On the image through
0178 with 0179 mounted (`wo-overlay.json`, 16:15-16:55), against the r6s arm above:

```
metric                         r6s-1017  r6u-wo-1017   change    ±95%  verdict
decode geometric mean (15)                             -0.97%   0.13%  better
c8 output tok/s                 146.613      145.607   -0.69%   1.29%  same
c4 output tok/s                 112.493      111.307   -1.05%   1.44%  same
poisson output tok/s            117.687      117.303   -0.33%  21.03%  same
prefill 32,768 tok/s           4414.396     4404.513   -0.22%   1.17%  same
prefill 131,072 tok/s          4345.996     4342.602   -0.08%   0.59%  same
prefill 262,144 tok/s          4153.366     4153.556   +0.00%   0.52%  same
prefill 500,000 tok/s          3801.250     3800.698   -0.01%   0.16%  same
```

Temperature-0 (62313cb5 / 9742ac2c) and greedy (4d143977 / 12fd8257) outputs are identical
to the image through 0178's, logprobs included; the release-test gate with 0179 mounted:
1,361 passed, 41 skipped. The decode step time is cross-session (the same build read 1.5%
apart in two sessions). The rebuilt image (`rebuild-wo.json`, `sha256:0aec88ed`) ran the
release tests and the determinism gate before promotion; the cold boots and a second release
were not rerun (0179 changes which kernels are declared, not how they are prepared, and the
overlay boot prepared the changed kernels with the cache frozen and served the whole suite
without a compile after preparation).

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
9. The four-node ring on 1017, the same schedule at `tp4-ring` (16 streams, 1M context):
   `tests.json` (shape-independent), `gate-tp4-1017.json` (16, 8 and 4 distinct streams),
   `coldboots-tp4-1017.json` (`r6u-tp4-{coldrank,cold,allcold}-1017.json`; rank 1, dgx2,
   is the cold rank, its TileLang cache absent while dgx1, dgx3 and dgx4 hold copies of a
   warm r6u cache) and `release-tp4-1017.json` (the `release` suite, r6u / r6n / r6u again,
   r6n being the ring's last release). The ring's arms schedule on their builds' DSpark cost
   curves, as TP3 does: the recipe's ring pins production's curves, which another site does
   not have.
10. The WO-A fix (vLLM 0179): `wo-overlay.json` screens it on the r6u image through 0178 (its
    changed files mounted): the release-test gate, the determinism gate and the release suite
    on 1017. Then the rebuild chain, each its own run: `rebuild-wo.json` (the image, tagged
    `-r6u`, the image through 0178 kept as `-r6u-0178`, and the release tests),
    `gate-1017-wo.json`, `coldboots-1017-wo.json` (fresh cache directories,
    `r6u-tp3-*-1017-wo.json`) and `release-1017-wo.json`.

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
- Release on 1017 (2026-10-17, `release-1017.json`, r6s / r6u / r6u again): decode level (8 and 4
  streams, Poisson; the engine grid +0.68% within its repeat's 1.5% drift), prefill 2.8-3.2%
  slower at every size (32K 4414 -> 4277 tok/s), the repeat within 0.2%. Cause: 0177 quantized
  WO-A's input in the GEMM's activation tile loads, so each of a group's 16 column tiles redid
  it; the image before 0177 was level with r6s. Model behaviour: no greedy response identical to
  r6s's (WO-A's input quantization changes the outputs), r6s's teacher-forced KL on r6u's text
  0.023 nats, top-1 agreement 94.5%, acceptance length 1.1% lower.
- vLLM 0179: the sparse MLA kernels prepare WO-A's input in their output store, once per row
  (`sparse_mla.wo_input`: the inverse RoPE, then `act_quant`'s quantize-dequantize), and WO-A's
  GEMM loads its BF16 tiles as before 0177. Overlay screen (`wo-overlay.json`): the release-test
  gate 1,361 passed, 41 skipped (two new tests: `wo_input` bit for bit against TileKernels'
  rotation and `per_token_cast`); temperature 0 (62313cb5 / 9742ac2c) and greedy (4d143977 /
  12fd8257) identical to r6u's on 1017, logprobs included.

