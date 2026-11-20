# Engine grid: real text in the decode rows (grid version 2)

## Hypothesis

The audit's claim: the engine grid's decode shapes are built from identical dummy tokens,
so the routed MoE sees about 8 of 256 experts a layer, against about 195 of 256 on real
16-stream text. If so, the grid under-represents MoE work: its gains need not show at c16,
and kernels are judged on the wrong expert spread.

## Intended delta

One variable: the grid's decode-row inputs.
- **Version 1** (today's grid): no token ids.
- **Version 2** (the fix): each decode shape's rows take servebench's fixed sample of real
  text (`servebench/grid_tokens.json`), marked real rows.

Shapes, replay method, image and recipe are unchanged.

- Base: spark-ds41f `origin/main` 6a03e0caf0.
- vLLM: candidate-57's tree `d74e38eef` (image `spark-ds41f:candidate-57`, `sha256:ba512cb2662b`). Its
  `engine_grid`, `make_dummy` and router code is candidate-58's.
- Patch: [vllm/0001](vllm/0001-DS4.1-Engine-grid-real-rows-from-token-ids.patch), commit `0cea8c0084` on `d74e38eef`. The runs used `796430272e`, the same diff before its message was amended.
  It is mounted read-only over the image as the `patched` arm. `runs/verify-tree-*.txt` checks both trees file by file.

## Result

**The claim holds in substance, but both of its numbers are wrong.** The grid leaves out
more MoE work than it says, and real text spreads less.
- The grid's dummy rows are token 0, and `InputBatch.make_dummy` also marks every one of them
  padding.
- The MoE routers give padding rows expert -1 (`VLLM_MOE_SKIP_PADDING`, on by default, and set
  by no recipe), and the DSpark drafter does the same for its dummy query rows.
- So today's grid routes **no expert at all: 0 of 384 in every one of the 40 target layers and
  the drafter's 3, at every shape.** The ~8 of the claim is what identical tokens reach when
  they are not masked: 7-9 a layer here.
- The model has 384 routed experts and routes 6 a token, not 256. Through the checkpoint's own
  gates:
  - token 0 reaches 6 experts a layer at one position, and 6.25 over three;
  - real text reaches **87.5 of 384 a layer at 16 requests x 3 rows** (71-116 across layers).
    That is not ~195 of 256: text is far more concentrated than uniform routing (204), and
    16x96 reaches 115, not 299.

The routed-expert kernels still launch on version 1, with every row skipped. They cost
1.6-5.0 ms a step at TP4 rank shapes, against 10.5-95.1 ms with real text's experts.

### Experts hit a layer (of 384; mean over the 40 target layers, [min-max])

| Shape | grid (v1) | token 0, real rows | sample, dummy gates | real text, real gates | uniform random |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1x2 | 0 | 7 | 7 | 9.7 [6-12] | 11.9 |
| 4x8 | 0 | 8 | 8 | 33.5 [26-43] | 45.5 |
| 8x16 | 0 | 9 | 9 | 52.4 [42-69] | 85.5 |
| 16x32 | 0 | 9 | 9 | 69.5 [53-89] | 152.0 |
| 1x3 | 0 | 8 | 8 | 11.1 [7-16] | 17.7 |
| 4x12 | 0 | 8 | 8 | 41.1 [30-55] | 66.1 |
| 8x24 | 0 | 9 | 9 | 62.8 [50-87] | 120.9 |
| 16x48 | 0 | 9 | 9 | 87.5 [71-116] | 203.7 |
| 1x6 | 0 | 8 | 8 | 17.6 [11-28] | 34.6 |
| 2x12 | 0 | 8 | 8 | 38.2 [24-57] | 66.1 |
| 4x24 | 0 | 9 | 9 | 59.1 [44-85] | 120.9 |
| 8x48 | 0 | 9 | 9 | 86.2 [70-117] | 203.7 |
| 12x72 | 0 | 9 | 9 | 99.4 [83-141] | 260.4 |
| 16x96 | 0 | 9 | 9 | 114.8 [95-160] | 299.3 |

### Step time (ms; the mean of each input's boots, four for the grid; each change is against the grid boots of its own ABBA block, 95% Welch over boots)

| Shape | grid (v1) | sample, dummy gates | change | routes (real text's experts) | change | v1 / routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1x2 | 21.60 | 25.05 | +16.0% ±4.9 | 26.64 | +23.2% ±7.5 | 0.81 |
| 4x8 | 22.04 | 25.72 | +16.8% ±3.0 | 46.32 | +110.0% ±4.6 | 0.48 |
| 8x16 | 22.91 | 26.98 | +18.1% ±3.9 | 61.61 | +168.3% ±10.3 | 0.37 |
| 16x32 | 26.64 | 30.90 | +16.3% ±5.4 | 78.20 | +192.6% ±12.3 | 0.34 |
| 1x3 | 21.70 | 25.29 | +16.6% ±1.3 | 27.82 | +28.1% ±2.0 | 0.78 |
| 4x12 | 22.15 | 25.81 | +16.6% ±3.2 | 52.18 | +135.3% ±1.1 | 0.43 |
| 8x24 | 25.79 | 28.81 | +11.7% ±1.2 | 70.68 | +174.1% ±1.8 | 0.36 |
| 16x48 | 23.97 | 34.08 | +42.0% ±2.1 | 93.37 | +289.8% ±8.2 | 0.26 |
| 1x6 | 21.84 | 25.72 | +17.8% ±3.2 | 33.37 | +52.8% ±0.8 | 0.65 |
| 2x12 | 21.99 | 25.84 | +17.6% ±3.9 | 49.58 | +125.3% ±4.5 | 0.44 |
| 4x24 | 25.20 | 28.18 | +12.3% ±7.3 | 67.17 | +165.4% ±2.2 | 0.38 |
| 8x48 | 22.88 | 32.72 | +43.7% ±3.4 | 91.25 | +296.9% ±1.2 | 0.25 |
| 12x72 | 27.11 | 38.11 | +41.0% ±1.3 | 105.01 | +286.1% ±19.2 | 0.26 |
| 16x96 | 29.30 | 43.50 | +47.5% ±1.3 | 120.07 | +312.6% ±2.5 | 0.24 |

Geometric mean over the 14 shapes: sample / grid 1.2325, routes / grid 2.5026. Control (plain image, grid inputs, one boot) / patched grid: 1.0005.

### Routed-expert kernels a step (ms, nsys) and their share of the step

| Shape | grid routed ms (share) | sample routed ms (share) | routes routed ms (share) |
| --- | ---: | ---: | ---: |
| 1x2 | 1.64 (8%) | 9.87 (39%) | 10.52 (40%) |
| 4x8 | 1.81 (8%) | 9.40 (37%) | 28.51 (62%) |
| 8x16 | 2.15 (9%) | 8.31 (31%) | 43.19 (70%) |
| 16x32 | 3.33 (12%) | 9.91 (32%) | 56.34 (72%) |
| 1x3 | 2.06 (10%) | 9.27 (37%) | 11.44 (41%) |
| 4x12 | 2.25 (10%) | 9.15 (35%) | 33.92 (65%) |
| 8x24 | 3.01 (12%) | 9.04 (31%) | 49.76 (70%) |
| 16x48 | 3.19 (13%) | 12.40 (36%) | 70.92 (76%) |
| 1x6 | 1.78 (8%) | 9.38 (36%) | 16.00 (48%) |
| 2x12 | 2.21 (10%) | 9.10 (35%) | 31.65 (64%) |
| 4x24 | 2.95 (12%) | 8.99 (32%) | 46.76 (70%) |
| 8x48 | 3.16 (14%) | 12.13 (37%) | 69.27 (76%) |
| 12x72 | 4.47 (16%) | 15.39 (40%) | 82.61 (79%) |
| 16x96 | 4.97 (17%) | 17.82 (41%) | 95.11 (79%) |

### Real text, experts hit in each of the 40 layers (16 requests x 3 rows, c16-like)

112 116 87 99 82 86 71 87 97 97 76 95 90 82 76 97 81 78 82 111 75 74 87 102 78 80 87 92 79 76 82 72 86 89 92 77 85 94 94 98

Notes on the tables:
- **Census.** On dummy weights, `sample` routes like `token0`, 7-9 experts a layer: the
  dummy gate barely depends on the token. That is why the timing needs `routes`: on dummy
  weights, only injected ids carry text's spread. On a testbed, version 2's ids go through
  the real gates.
- **Step time.** `sample` is the cost of routing about 8 experts a layer, already +3.5 ms
  (+16%) at one request. `routes` is the cost of real text's spread: +23% at 1x2 to +313% at 16x96, 2.50 times the version 1 step on the 14 shapes' geometric mean. The
  plain image's grid, one boot, is within 0.05% of the patched arm's grid: without
  token ids the patch changes nothing.
- **Shares.** These are the routed-expert kernels' summed time, as a fraction of the timed
  step. The router kernel is listed separately in `analysis.json`. Kernel sums can overlap
  the L2-prefetch side stream, so the shares are upper bounds. The step times are not, and
  they rise by about as much as the routed kernels.
- **Scope.** This is one TP4 rank on one GPU, with dummy weights and no collectives. On a
  testbed, version 2's rows go through the real gates, but after the grid's dummy KV context
  rather than their own. Their spread should be near the reference's, but it was not
  measured there.
- **Serving, for scale.** The 1107 TP4 c16 serving step is 125.6 ms (7.96 steps/s, 14.4
  requests and 44.6 verified rows a step). On one rank, version 1's 16x48 is 24.0 ms of
  that and `routes`' 16x48 is 93.4 ms.

## Method

Everything ran on sp8, one GB10. Every GPU run took `/tmp/spark-lab-gpu.lock` after checking that
`~/spark-hold.json` was absent ([harness/gf.sh](harness/gf.sh), which logs each run to
`commands.txt`). The CPU reference forward took the same lock, so that its traffic overlapped no
other job's GPU timing.

- **The engine.** step957.py's one-GPU engine: the 40 target layers and the DSpark drafter's
  3, with TP4 rank-0 shapes (16 heads, 2 output groups, a 576-wide expert intermediate, the
  compressor sharded as at TP4). Dummy weights, the recipe's environment
  (`VLLM_MOE_SKIP_PADDING` unset, so on, as on the testbeds).
  [harness/gridstep.py](harness/gridstep.py) adds the recipe's TP4 capture ladder (1-96 rows), so
  every grid shape replays a full CUDA graph. It times the grid's 14 shapes at 4K through
  `engine_grid` as servebench does: 2 warm-ups, 30 alternating rounds, one settle run before
  each timed run.
- **Inputs:**
  - `grid`: no token ids, which is today's grid;
  - `sample`: grid version 2's ids, as real rows;
  - `routes`: the sample's ids, with each routed row's six experts set to the ones the
    checkpoint's gates gave that row in the reference forward. Dummy weights cannot route like
    text, so this input carries real text's expert spread into the timing.

  `routes` injects the ids inside the captured graphs. Each of its shapes is timed in its own
  `engine_grid` call.
- **Routes census** ([harness/census.py](harness/census.py)). The same engine runs eagerly,
  without adaptive verification, which needs full graphs and plays no part in routing. A
  wrapper around `TileLangGateRouter._compute_routing` records every router call's expert ids
  during one `engine_grid` step per shape and input: rows, rows routed and distinct experts,
  per layer. Input `token0` is token 0 in every row, marked real rows: the identical tokens of
  the claim.
- **Real text through the real routers** ([harness/ref_routes.py](harness/ref_routes.py)).
  DeepSeek's reference forward from the checkpoint (`inference/model.py`, snapshot `1bae0e10`)
  runs on CPU one layer at a time, with every weight read from the HF cache. It uses
  experiments/2026-10-18-json-count's driver and kernel shim. Its input is the 16 sample
  prompts' first 80 tokens (prompt, then the recorded reply), plus one sequence of token 0,
  prefilled together. Position 64 + k of sequence r is window r's token k. These are the
  grid's version 2 rows after their real context, routed by the real gates after 40 real layers.
- **Kernel split.** One nsys boot per input (`--cuda-graph-trace=node`), with one NVTX range per
  shape (1 warm-up, 3 settle and timed pairs). [harness/analyze.py](harness/analyze.py) sums the
  routed-expert kernels per step: `moe_grouped_fp8_fp4_gemm`, `moe_swiglu_cast`,
  `moe_dispatch` and `moe_combine`. It sums the router (`router_partials`) separately.

The sample ([harness/build_grid_tokens.py](harness/build_grid_tokens.py)) draws 16 distinct
default prompts with seed 0. For each, it takes the 16 reply tokens at absolute positions 64-79
of prompt + reply. The prompts are 41-55 tokens, so every window sits inside the reply. The
replies are the 1107 release's TP4 model step (`model.json.gz`, sha256 `65db378d…`, 2026-11-07):
greedy answers to the default prompts, mostly reasoning text. Request r of a shape reads window
r's first rows, split over requests as `InputBatch.make_dummy` splits them.

## The fix: grid version 2

- **vLLM** ([vllm/0001](vllm/0001-DS4.1-Engine-grid-real-rows-from-token-ids.patch), on
  `d74e38eef`, 4 files). `engine_grid` shapes may carry `token_ids`, one id per row. For such a
  shape, `InputBatch.make_dummy` copies the ids into the dummy rows and marks them real (via
  `InputBuffers.dummy_input_ids`). The DSpark drafter marks its dummy query rows live
  (`dummy_rows_live`). Rows past the ids, a CUDA graph's padding, stay token 0 and padding. Each
  shape's result reports `real_rows`. Without `token_ids` nothing changes: capture, the memory
  profile, the cost profile and serving keep token 0 and padding rows. The patch has its own
  CUDA test, which passed with the image's input-batch and dead-row tests: 70 passed
  (`runs/pytest-patched-r2.txt`).
- **servebench** (`engine.py`, `__main__.py`, `suite.py`):
  - `GRID_VERSION = 2`. The fixed sample is [grid_tokens.json](../../servebench/grid_tokens.json).
    `with_token_ids` gives each decode shape its rows' ids; prefill shapes stay padding.
  - `run` records the version of the work the server did. It records 2 when every decode shape
    reports real rows, and 1 when the server ignored the ids (a vLLM before the patch runs exactly
    version 1's work). `--grid-version 2` refuses such a server, `1` sends no ids, and `auto` is
    the default.
  - The result's `workload` carries `grid_version`, the requested version and the sample's
    sha256.
  - `compare`, `spread` and `boot_ratios` raise `VersionMismatch` across versions. A result
    without a version (every recorded grid) is version 1. `suite.compare` records a mismatched
    grid step as skipped, with the reason, and compares the other steps.
  - The suites are unchanged, so their digests and every recorded reference still apply.
- **Docs.** The grid descriptions in [docs/release-benchmark.md](../../docs/release-benchmark.md)
  and [servebench/README.md](../../servebench/README.md). The release page
  (`scripts/render_docs.py`, `docs/index.html`) shows a base/candidate grid delta only within
  one version.
- **Tests:** `tests/test_servebench.py` (CPU):
  - the sample: 16 windows of 16 in-vocabulary ids from 16 distinct prompts, the same in every
    run, on every default decode shape, none on prefill;
  - the row split, matching `make_dummy`;
  - a real-row server records version 2, a server that ignores the ids records version 1, and
    `--grid-version 2` refuses one that ignores them;
  - `compare` and `spread` refuse mixed versions, including on the command line;
  - a suite comparison skips a grid step of two versions and compares the rest;
  - a suite step can pin `grid_version` 1.

## What it changes in recorded verdicts

Every recorded grid is version 1: all of `manifests/suites/*/grid.json` and every candidate
screen. Each one timed a decode step with **no routed expert in it**. Two things follow.

1. **The grid cannot see the routed-MoE decode kernels**, beyond their launch and empty-tile
   cost (1.6-5.0 ms a step). This covers the grouped FP8 x FP4 GEMMs, the SwiGLU cast,
   dispatch, combine, and the work behind the router.
   - A grid number for such a change measured only that empty-launch cost. #589's late
     trigger of the routed grouped GEMMs was judged on its own c1 chain bench, so its
     verdict stands. Any grid line in its record says nothing about it.
   - The open levers that act there can only be judged at version 2, or at c8 and c16:
     - the routed MoE persistent kernel (TODO, decode 1);
     - the shared expert beside the routed MoE on a second stream (decode 5).
2. **Every other gain reads larger on the grid than in serving.** A saving of Δ ms is
   Δ / (version 1 step) on the grid, but Δ / (step with experts) in serving.
   - The version 1 step is 0.34 of the real-routes step at 16x32, and 0.81
     at 1x2.
   - This explains much of the record:
     - candidate-54 at `tp4-switched`: grid 4.29% faster, c16 paired decode +0.94% (inside
       its margin);
     - 1107 against 1030: grid 14.36% faster, c16 paired decode +5.30%, c8 +4.04%.
   - Scaled by the 16-request ratio, the 4.29% is about 1.1-1.5% of a step with real
     routing, which is where c16 landed.
   - The c16 serving step at TP4 is 125.6 ms (1107: 7.96 steps/s, 14.4 requests, 44.6
     verified rows). The version 1 grid at 16x48 is 24.0 ms a rank, and real routing puts
     that shape at 93.4 ms.

Verdicts most likely to move:
- **The L2 prefetch settings** (`VLLM_L2_PREFETCH_GRID=2`, and candidate-58's TP3 prefetch
  caps, confirmed by a grid-only job). The prefetch side stream was tuned against steps that
  read no expert weights. With real routing, expert reads compete with the prefetch for DRAM
  and L2. Re-check them at version 2 or at c8 and c16.
- **Small-batch grid losses and gains in non-MoE kernels** (candidate-54's TP3 small-batch
  grid loss from #961, about 5.5-7.2% at 1-request shapes). They keep their absolute size,
  but their % shrinks to about 0.8 of itself at one request. The sign holds.
- **#987's WO-B part does not move.** It acts at M=97-128 rows. The grid's widest shape is 96
  rows, so neither version exercises it. Its pass rests on its own WO-B bench and on "the
  16-stream cell not worse", a dense path that real routing does not change.
  - Real routing makes WO-B's share of a 16x96 step smaller still.
  - A grid shape above 96 rows (copy widening reaches 107 at 16 streams) would be needed to
    judge it on the grid.
- **MoE kernel verdicts.** The ones taken on the grid saw no expert work, so the grid gave
  them no evidence either way. The ones taken on their own kernel benches or on c8 and c16
  stand.

**Related, not changed here:** every other consumer of dummy steps has the same padding
rows, for example the adaptive-verification cost profile. `0009-dspark-profile-distinct-tokens`
addressed the profile in September and is not in the 1107 series. Whether the pinned cost
curves (`0005`) came from dummy steps is worth checking.

## A recorded floor for version 2

Yes, and it must be recorded fresh, keyed by grid version as well as testbed. This section
describes it for the agent adding per-testbed floors; nothing here implements it.

- **Why version 1's spread does not carry over.**
  - A version 2 step is mostly expert-weight streaming: about 72-79% of the 16-request
    steps here.
  - Its boot-to-boot spread need not match version 1's.
  - Here, two boots each gave these geometric-mean differences: version 1 (`grid`) 0.44% and 0.24%,
    dummy-gate `sample` 0.08%, and `routes` 0.30%. These are one pair each: a hint,
    not a floor.
- **What to record.** At the first version 2 reference on each testbed (a round's frozen
  candidate, or the next release), take at least 3 A/A boot pairs. Store the 95% bound of
  one boot's decode geometric mean against another's as `floors[<testbed>]["grid"][2]`.
  The suite's `with_floor` would then use the larger of that and the run's own repeat for
  the grid row, as it already does for the paired rows.
- **Guards.**
  - A floor without a grid version is a version 1 floor, and must never be applied to a
    version 2 grid.
  - The per-shape rows need floors too: the 16-request shapes now carry most of the step and
    most of the spread.

## Runs

All runs were on sp8, 2026-11-10, in the order below. Every GPU run and the reference forward took
the GPU lock with the hold file absent. [runs/commands.txt](runs/commands.txt) has every run's exact command, load, `docker
ps` and lowest MemAvailable. The scripts are [harness/run_all.sh](harness/run_all.sh),
[run_more.sh](harness/run_more.sh), [run_ref.sh](harness/run_ref.sh),
[run_routes.sh](harness/run_routes.sh), [run_routes2.sh](harness/run_routes2.sh) and [run_tail.sh](harness/run_tail.sh). Another agent's
timing jobs took the lock between some of these runs.

| Run | Arm | What | Result |
| --- | --- | --- | --- |
| ref-smoke | CPU | 2-layer smoke test of `ref_routes.py` (87 s, 6 CPUs) | ok. It ran outside the lock while another agent's GPU job held it. Not in `commands.txt` |
| verify-tree base, patched | CPU | both trees against `git ls-tree`, file by file | 7,831 and 7,832 files match, none differ |
| census | patched | first routes census | **failed**: `enforce_eager` with adaptive verification is refused ("Adaptive verification requires full CUDA graphs") |
| step-grid-boot1 | patched | grid inputs | ok. Another agent's CPU-only container was running (`gallant_noyce`, loadavg 10.6) |
| step-sample-boot1, -boot2 | patched | sample ids (dummy gates) | ok |
| step-grid-boot2 | patched | grid inputs | ok |
| nsys-grid, nsys-sample | patched | kernel split | ok |
| step-grid-base | base | control: the plain image's grid | ok |
| census-r2 | patched | routes census, adaptive verification off | ok |
| pytest-patched | patched | the patch's tests | **failed** at the conftest import: the serving image lacks `tblib` |
| ref-routes | CPU | reference forward, 17 x 80 tokens, 40 layers | ok: 637 s, 189 GiB read |
| step-grid-boot3 | patched | grid inputs, opening the routes block | ok |
| step-routes-boot1 (first) | patched | real text's experts injected | **failed** after boot: the routes buffer, made inside the engine's inference mode, was filled outside it. Log `step-routes-boot1-failed.txt`. The script was stopped, fixed (`torch.inference_mode()` around the fill) and rerun as `run_routes2.sh` |
| step-routes-boot1, -boot2, step-grid-boot4 | patched | real text's experts injected, then the grid, closing the block | ok |
| nsys-routes | patched | kernel split with real text's experts | ok |
| pytest-patched-r2 | patched | the patch's tests, with round 3's `tblib` stand-in | ok: 70 passed |

- **A bug in the progress log.** `run_all.sh`'s progress line read `$?` after a command
  substitution, so it logged rc=0 for every run. The census's real exit code (1) is in
  `commands.txt`. `run_all.sh` is kept as it ran; the later scripts read the code first.
- **Memory.** Every engine boot's lowest MemAvailable was 27-29 GiB (the memguard's floor is
  6). The reference forward ran in a 24 GiB container.
- **Harness edits during the session.** `gridstep.py` gained `--routes` after the four
  timing boots and nsys-grid. Its path without `--routes` is unchanged. nsys-sample, the
  control and the later boots ran on the edited file.

## Files

- [vllm/0001-DS4.1-Engine-grid-real-rows-from-token-ids.patch](vllm/0001-DS4.1-Engine-grid-real-rows-from-token-ids.patch):
  the vLLM change, on `d74e38eef`. It applies cleanly to candidate-58's tree `e5911bdf2`, whose
  three touched files are identical.
- [harness/](harness/):
  - `build_grid_tokens.py`: the sample and the reference sequences;
  - `census.py`, `gridstep.py`: the vLLM harnesses. They import `step957.py` and `omlib.py`,
    which are byte-identical to candidate-57's own-mark copies
    (`../2026-10-26-spark-ds41f-candidate/candidate-57/receipts/own-marks/pr-957/raw/harness/step957.py`
    and `.../own-marks/common/h/omlib.py`). The tests ran with r6u's `tblib_stub.py`
    (`../2026-10-17-r6u/bundles/release-tests/tblib_stub.py`). All three were mounted at `/h`;
  - `ref_routes.py`: the reference forward's routes. It needs
    `../2026-10-18-json-count/reference/shim/kernel.py` beside it as `shim/kernel.py`;
  - `routes_table.py`: the per-shape real ids for `--routes`;
  - `analyze.py`: the tables;
  - `gf.sh` and the `run_*.sh` scripts.
- [runs/](runs/):
  - `commands.txt`, `progress.log` and every run's log;
  - `json/` (census, step and nsys results);
  - `analysis.json`: every table here;
  - `ref-sequences.json`: the reference forward's input;
  - `ref-routes.json`: the reference's expert ids at the window positions 64-79, [40 layers][17 sequences][16][6];
  - `routes-table.json`: the per-shape ids `routes` injected;
  - `verify-tree-*.txt`.
  The nsys reports and the reference's `.pt` stay on sp8 under
  `~/spark-lab/agents/grid-fidelity/out/`.
