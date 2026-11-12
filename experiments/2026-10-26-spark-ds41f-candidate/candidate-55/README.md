# candidate-55

The first candidate on frozen [candidate-54](../candidate-54/README.md) (tag `candidate-54`): six of the owner's PRs
that change the platform under the model rather than one kernel. Chosen by the owner on 2026-11-11 ("check 1134 and
include the cuda 13.5 setup for candidate-55", "also add the b12x removal", "also add the sparknet rework PR": "both
#989 and #982"), with #1134's new head and #1145 added on 2026-11-12 ("Yes, rebuild"), and #738's GLP Spec v3 runtime
added the same day ("prepare the GLP update into candidate-55"). It is a candidate, not a merged base: verified
against candidate-54's recorded runs, not release-qualified.

| PR | What | Carried as |
| --- | --- | --- |
| #1134 | Weight arenas and MoE slabs take whole free blocks of 2 MiB or more at load (both kernel page sizes). The buddy sponge is held from model creation through post-load weight processing, with a 12 GiB `MemAvailable` floor (cut by 1/49 for page tables on 4 KiB-page kernels), and trimmed after the arena binds to keep 16 GiB of `MemAvailable` through load and repacking. The RoCE proxy, TP worker and EngineCore are placed by L3 size | vLLM `0800`, `0801`; sparknet `0001` |
| #1145 | The direct loader's range check accepts the TP3 display carve-out's dma-buf mappings, which CUDA 13.5's compat driver reports as host memory with base 0. Pinned host memory and out-of-range writes are still rejected | vLLM `0804` |
| #1096 | B12X removed: no B12X source in the lock, no B12X import on DS4.1 runtime or loader paths, its startup rounds skipped | vLLM `0802` (rebased on candidate-54 and #1134) |
| #989 | sparknet: one persistent multi-stage kernel per collective, 8 thread blocks, no single-slot cap, so SP prefill collectives up to 16 MiB stay on sparknet | sparknet `0002` |
| #982 | sparknet: PDL chaining of the four one-shot collectives (Part A) | sparknet `0003` (re-placed in #989's stage loop); vLLM `0803` (its PDL-chain test entry) |
| #974 | CUDA 13.5.37 toolchain (nvcc/ptxas, cudart) and cuBLASLt 13.10.0.20 | `sources.cuda_toolkit` in this lock (hash-locked debs) |
| #738 | GLP's reader takes Spec v3 (`hc_fold_pre_engram`) beside Spec v1; a Spec v3 vector folds the stream mean and projects each covered layer's attention write. A Spec v1 vector (GLP-39) steers as before | vLLM `0805` (rebased on candidate-55, [receipts/glp-rebase](receipts/glp-rebase/README.md)) |

## Selection

- **Authors:** every PR but #738 is christopherowen's, and every other patch's `From:` is Christopher Owen's. #1134's
  `0800` at `37a7f0ac` read `test <test@example.com>`; at `6757c7a1` it is Christopher Owen's.
- **#738 is an exception to the owner-only rule, on the owner's instruction** (2026-11-12: "prepare the GLP update
  into candidate-55"). #738 is msuiche's PR, merged to `main` as an experiment folder
  ([2026-10-31-glp-specv3-fold-steering](../../2026-10-31-glp-specv3-fold-steering/README.md)), and its patch's
  `From:` is Matt Suiche, co-authored by Christopher Owen. `0805` keeps that authorship.
- **#738 is carried as its patch:** `0805`, rebased by hand in two `nvidia/model.py` hunks. Its GLP-27 profiles are
  replaced by [glp27-tp4.json](glp27-tp4.json), which extends this candidate's `tp4.json`.
- **#985 is not carried.** It switches the default vector to GLP-27 in `config/cluster.json`, which `spark-ds41f-1107`'s
  image cannot read. Every measured arm here keeps GLP-39, and GLP-27 runs on a qualification arm only. The switch
  belongs in the promotion commit of a release whose image reads Spec v3.
- **#1134 is carried in part:** its vLLM and sparknet patches only, at head `6757c7a1` (owner, 2026-11-12: "Yes,
  rebuild").
  - **This head's `0800`** replaces the one of `37a7f0ac`, which replaced the one first carried (heads `2d0c6c0b` and
    `d9b04765`). Since `37a7f0ac` it keeps the sponge from model creation through `process_weights_after_loading`,
    where the first one released it at the end of `create_model()`. At `6757c7a1` its `MemAvailable` floor is 12 GiB
    rather than 6, cut by 1/49 on 4 KiB-page kernels for page tables, and it budgets from `MemAvailable` alone. After
    the arena binds, it releases the sponge when fewer than 256 MiB of target-size blocks remain, or drops chunks
    until 16 GiB of `MemAvailable` remain.
  - **`0801` and sparknet `0001`** have the same diffs as at `d9b04765`.
  - **Its pre-launch defragmentation is on `main`** (owner, 2026-11-11: "add it default on, and a config parameter
    to disable it to each profile"). `cluster start` runs `sudo sync; drop_caches=3; compact_memory` on every node
    before launch. Its switch is `host.defragment_before_start`: true by default, and set true in each of this
    candidate's serving profiles, which the `-dev` profiles inherit.
  - **Not carried: its other deployment tooling**, which is not image content: `scripts/memguard.sh`,
    `scripts/memguard-start.sh` and `spark_ds41f/lab.py`'s telemetry pinning.
  - **The servebench changes** (`trim_stall_tail`, an untimed prefill warm-up) would have altered the measurement
    this candidate is judged by. They have since left #1134.
- **#1134 contains #1008:** its `0801` adds `vllm/utils/cpu_affinity.py` with #1008's partition, and its sparknet
  `0001` is #1008's sparknet patch (the same tree, `dab8c457`). #1008 is not carried separately.
- **#1145 is carried whole:** its `0804`, at head `c6392b4b`. Its experiment folder and profile are not carried.
- **#982 is carried in part:** Part A (PDL chaining). Part B (`next_send_buffer` zero-copy staging) has no caller in
  vLLM, so it does nothing in this deployment and is left out.
- **#974 is selected through the lock:** its toolchain stage builds only for a lock that lists `sources.cuda_toolkit`,
  so `spark-ds41f-1107`, candidate-54 and every other lock build as before on the base image's CUDA 13.0.2. The debs
  are hash-locked in `requirements/`.
- **#1096's tooling** (`scripts/kernel_backend.py`, `scripts/recipe.py`: a lock without B12X drops its tree label) is
  merged to `main` with #974's selection; its experiment folder is not carried.
- **Round 3's PRs** (#978, #988, #957, #956) are not in this candidate; they re-enter later candidates on frozen
  candidate-54 with fresh validation.

## Source

- vLLM ([vllm/series](vllm/series)): candidate-54's 672 patches by relative path, then `0800`, `0801` (#1134's files,
  unchanged), `0802` (#1096 rebased, [receipts/b12x-rebase](receipts/b12x-rebase/README.md)), `0803` (#982's test
  entry), `0804` (#1145's file, unchanged) and `0805` (#738 rebased, [receipts/glp-rebase](receipts/glp-rebase/README.md)).
  678 patches, `patch_head` `d6d4319e`, `expected_tree` `8aaae3df`, `patchset_sha256` `6fdf03dc`.
- sparknet ([sparknet/series](sparknet/series)) on `1fd6a1fb`: `0001` (#1134), `0002` (#989), `0003` (#982 Part A,
  [receipts/sparknet-stack](receipts/sparknet-stack/README.md)). `patch_head` `1fdf8dc7`, `expected_tree` `e06f6a06`,
  `patchset_sha256` `60bb1942`.
- No B12X source. CUDA 13.5.37 through `sources.cuda_toolkit` ([requirements/cuda-13-5-sbsa.txt](../../../requirements/cuda-13-5-sbsa.txt), 23
  debs, 1.51 GB). Every other source is candidate-54's.
- Recorded by `bin/spark build prepare --lock` on dgx4 for both locks (the same vLLM and sparknet trees), the heads
  copied into each `upstreams.lock.json`; the 23 debs downloaded and verified against the lockfile there. Recorded
  again for #1134 at `37a7f0ac` (vLLM tree `ed91c273`), for #1134 at `6757c7a1` with #1145 (tree `e8c20f54`), and with
  #738's `0805` (tree `8aaae3df`). Each time a `git am` replay on dgx3 gives the same tree. The provenance, SHA-256 and author of each patch are in
  [receipts/selection.json](receipts/selection.json).
- **What #1096 changes on the serving path:** one TileLang kernel, `engram_resident_lookup_kernel`, whose row bound
  is now `0 <= n_tok <= tokens` (bit-identical inside it, recompiled); the rest is renames, dead fields, the deleted
  B12X loader and startup rounds skipped without B12X ([receipts/b12x-rebase](receipts/b12x-rebase/README.md)).

## The control: candidate-55-no974

[no974/](no974) is the same series and sources without `sources.cuda_toolkit`: image
`spark-ds41f:candidate-55-no974`, on CUDA 13.0.2. #974 very likely changes temperature-0 output (new SASS from ptxas
13.5, other cuBLASLt algorithms), so candidate-55's gate alone cannot show that the other PRs leave outputs
unchanged. The control can: it must give candidate-54's TP4 hashes `18d61961`/`d6568649`.

## Marks, declared before the build

| PR | Own mark | Serving |
| --- | --- | --- |
| #974 | Its sp6 study (`experiments/2026-11-08-cuda-13-5-compiler-cublaslt`) on sp8, candidate-55 against the control, 2 boots each: the `sm_121a` `LDGSTS.E.64` encoding fix (outputs bit-equal on `down_384x576_evict_first`), and the dense MXFP8 kernels and cuBLASLt FP8/BF16 matmuls faster beyond the interval | candidate-55 against the control at `tp4-ring`: grid, c16 and prefill not worse |
| #1134 | Its 8-boot `ABBAABBA` single-GPU screen on sp8 (`bench_one_boot.py`): boot-to-boot spread of the target forward lower and its mean not slower, the control against candidate-54's image | Every boot reaches ready (no memory-floor halt from the 10 GiB `MemAvailable` floor); the screens' two boots no further apart than candidate-54's |
| #1096 | Its release-test selection passes; no B12X module is imported at startup | Inert: the control's gate gives candidate-54's hashes; time to ready not worse (three startup RPC rounds skipped) |
| #989 | Its `bench_pipelined_collectives.py` on sp1..sp4 (TP4) and sp1..sp3 (TP3): collectives of 4-16 MiB faster than PyNCCL beyond the interval, outputs bit-exact | Prefill at 32K and 256K not worse at `tp4-ring` and `tp4-switched` |
| #982 | Its `bench_pdl_direct_stage.py` chains on sp1..sp4, Part A only: chain time at M=1..32 faster beyond the interval; the PDL-chain source test passes | The gate on 12 boots at `tp4-ring` and the c1 decision check: every boot identical. #982 makes more graph edges programmatic, the class of hazard `0799` fixed |
| #738 (declared 2026-11-12, before the fourth build) | Its suite passes in the release tests: the GLP kernel and model tests, `glp_steer_write`'s PDL-chain entry, the weightless, CED and SP-prefill model tests. The image's reader accepts GLP-39 and GLP-27 on the head (#1149's GLP reader check in `cluster start`'s preflight, at the boots of `tp4.json` and `glp27-tp4.json`) | **Spec v1 unchanged:** GLP-39's steered replies ([glp39-steered-tp4.json](glp39-steered-tp4.json), `dual_probe.py`) identical in token IDs and log probabilities to the third build's on both boots: steered `739b8abc`, stock `9872053e` ([dual_replies.py](dual_replies.py)). **Stock unchanged:** every gate gives candidate-54's hashes with GLP-39 configured, and [qualify-glp27-tp4.json](qualify-glp27-tp4.json)'s gate runs give them with GLP-27 configured. **GLP-27 qualified at `tp4-ring`:** every `dual_probe.py` check passes on both boots; its steered speed and AdvBench rates are recorded, not gated |
| #1145 (declared 2026-11-12, before the third build) | Its test (`test_external_dmabuf_carveout_loads_and_rejects_out_of_bounds`) passes where `/dev/dri/card0` exists, and the release tests pass | Every `tp3-switched` boot loads its weights (no `batch destination exceeds CUDA device allocation`); inert at `tp4-ring`, where the carve-out is off: the gate gives candidate-54's hashes |

## Verification

- **Build:** one dgx job builds both images on dgx4 and copies them along the ring, then runs the release tests on
  candidate-55's image.
- **Gate:** candidate-55 on 12 boots at `tp4-ring` and 4 at each switched testbed, recording its own hashes (#974's
  declared change); the control on 2 boots at `tp4-ring`.
- **c1 decision check:** 4 boots at `tp4-ring` and `tp4-switched`.
- **Screens:** two boots against candidate-54's recorded screens
  ([candidate-54/receipts/references](../candidate-54/receipts/references), `candidate-54-0799-<testbed>-…`).
- **GLP (from the fourth build):** GLP-39's steered replies on 2 boots at `tp4-ring`, against the third build's
  ([glp39-steered-tp4.json](glp39-steered-tp4.json), run `-b3` before the fourth build and `-b4` after). The GLP-27
  qualification on 2 boots at `tp4-ring` ([qualify-glp27-tp4.json](qualify-glp27-tp4.json)): the temperature-0 gate
  on the stock name, [dual_probe.py](../../2026-10-28-glp-dual-serving/dual_probe.py) `--speed`, 16 distinct streams
  and [advbench.py](../../2026-10-28-glp-dual-serving/advbench.py), as GLP-39 was qualified for `glp-3`
  ([2026-10-28-glp-dual-serving](../../2026-10-28-glp-dual-serving/README.md)). GLP-27 at TP3 is not qualified.
  `advbench.py` prints every reply; its output stays on dgx1 under `results/private`, and the receipts keep only each
  boot's summary line (the rates per name).
- **Pre-launch defragmentation:** candidate-55's and the control's boots defragment before launch, but candidate-54's
  recorded references were taken without it. A difference against those references includes the defragmentation,
  which counts as part of #1134.

## TP4's recipe: L2 prefetch caps, and the ring's collective limit

**Added 2026-11-12, after the fourth build's screens.** Two recipe variables answer the two regressions the screens
found at TP4. Neither changes the image or the locks.

### L2 prefetch caps, at `tp4-ring` and `tp4-switched`

candidate-55's TP4 profiles ([tp4.json](tp4.json), [tp4-switched.json](tp4-switched.json), and their `-dev` files
and [glp27-tp4.json](glp27-tp4.json) through them) set `VLLM_DS41_L2_PREFETCH_WO_MB=15`,
`VLLM_DS41_L2_PREFETCH_FFN_MB=12` and `VLLM_DS41_L2_PREFETCH_NEXT_MB=12`, the caps candidate-54's TP3 profiles carry.
The code defaults are 20/16/20 MB. TP5 to TP8 are not measured and keep the defaults, and so does the control
([no974/tp4.json](no974/tp4.json)).

- **Why:** the grid regression at TP4 is in the 1-request rows: 1x2 rows are 9% slower than candidate-54 and 1x3
  rows 8% slower at `tp4-switched`. #982's `T.pdl_trigger()` before `roce_wait_peers` causes it, and only while the
  prefetch windows are at their defaults ([grid-attribution](grid-attribution/README.md)).
- **Measured** at `tp4-switched` on the fourth build, the grid step alone, two boots an arm in one session: the caps
  make the grid 4.66% faster than candidate-55 as built (1x2 rows 18%, 1x3 rows 13%), and 1.7% faster than
  candidate-54 with the same caps. Moving #982's trigger after the wait gives 2.5% alone and nothing under the caps,
  so sparknet `0003` is unchanged.
- **Measured** at `tp4-ring` the same way: the grid 2.51% faster than candidate-55 as built (margin 0.34%), 1x2 rows
  16.2 to 13.5 ms and 1x3 rows 14.8 to 14.0 ms, where candidate-54's recorded screen has 14.5 ms for both.
- **The windows were already too large for candidate-54 at TP4:** with the caps its 1x2 rows are 9% faster. So the
  caps are a gain of their own as well as the fix, and candidate-54's recorded TP4 screens are without them.
- **No output change:** the windows are cache hints, and the arithmetic is unchanged.
- A recipe variable, not a patch: the image and the locks are the fourth build's.

### The collective limit, at `tp4-ring` only

[tp4.json](tp4.json) sets `SPARKNET_ROCE_MAX_COLLECTIVE_BYTES=2097152`, the ring's slot. Reduce-scatters and
exchanges above one slot go to PyNCCL again, as before #989; #989's default is 64 MiB.

- **Why:** candidate-55's prefill at `tp4-ring` is 3.4% slower than candidate-54 at 262,144 tokens, in the control
  too, and unchanged at `tp4-switched`. #989 keeps those collectives on sparknet. The ring's slot is 2 MiB (4 MiB
  switched), its opposite rank is relayed, and #989's own mark was measured on switched testbeds
  ([prefill-attribution](prefill-attribution/README.md)).
- **Measured** at `tp4-ring` on the fourth build, the prefill step alone, two boots an arm in one session: with the
  limit, prefill is 3.4% faster than candidate-55 as built at 262,144 tokens and 3.3% at 32,768, which is
  candidate-54's on the same suite (+0.2% and -0.7%). The sliced all-gather and the pre-launch defragmentation
  change nothing.
- **What it gives up:** #989's routing of large collectives at `tp4-ring`. Its multi-stage kernels still carry the
  collectives of one slot or less, and `tp4-switched` keeps #989 whole, where prefill is not slower.
- **No output change:** both backends reduce in fixed rank order; the gate below checks it.

### Marks, declared before the runs

With the recipe above, at each TP4 testbed:

- the gate on 2 boots gives candidate-54's TP4 hashes ([gate-caps-tp4.json](gate-caps-tp4.json),
  [gate-caps-tp4-switched.json](gate-caps-tp4-switched.json));
- the screen's grid is not slower than candidate-54's recorded screen, and c16 and prefill are not worse than the
  fourth build's screen without the recipe ([screen-caps-tp4.json](screen-caps-tp4.json),
  [screen-caps-tp4-switched.json](screen-caps-tp4-switched.json));
- at `tp4-ring`, prefill at 262,144 tokens is not slower than candidate-54's recorded screen, and the c1 decision
  check's 4 boots agree with each other and with candidate-54's
  ([c1-decisions-caps-tp4.json](c1-decisions-caps-tp4.json)); its paired c1 decode time is recorded.

Their runs are named `…-caps-…-b4`. The fourth build's own gates and screens ran before the recipe and stay as its
record.

### Verified, 2026-11-12

Details and every run in [receipts/tp4-recipe](receipts/tp4-recipe/README.md) (20:42Z to 22:56Z). All 16 boots
reached ready, and every mark is met.

| Check | `tp4-ring` | `tp4-switched` |
| --- | --- | --- |
| Gate, 2 boots | 8 of 8 runs identical; candidate-54's TP4 hashes 18d61961/d6568649 | the same |
| c1 decisions, 4 boots | agree, and with candidate-54's on the 58 requests both sent | agree, and with candidate-54's on the 57 both sent |
| c1 paired decode against candidate-54 | **-1.62% ± 0.15%** decode time (+1.69% tok/s; 86.07-86.33 tok/s) | **-2.84% ± 0.13%** (+2.95% tok/s; 86.17-86.29 tok/s) |
| Screen against candidate-54: grid | **1.55% faster** (was 0.25% slower, the same, on the fourth build without the recipe; 0.67% slower on the third) | **3.11% faster** (was 1.01% slower) |
| c16 paired decode, output | +0.47%, +0.45% (same) | +1.45%, +1.47% (better) |
| Prefill | 32K the same (+0.78%), **256K the same (-0.04%; was 3.05% slower)** | 32K and 256K the same |

- **Both regressions the screens found at TP4 are gone:** the grid, by the caps, and the ring's 256K prefill, by the
  collective limit. #989's serving mark (prefill not worse at 32K and 256K) is now met at `tp4-ring` as well.
- **A single stream gains little from the caps:** c1 decode is 0.16% (`tp4-ring`, inside the interval) and 0.35%
  (`tp4-switched`) faster than the fourth build without them. The 1x2 and 1x3 grid rows the caps repair are replayed
  back to back in the grid.
- **Still open at `tp4-ring`:** four shapes of 24 and 32 rows are 1% to 2% slower than candidate-54's on both boots
  (1.3% to 3.3% without the caps). Not attributed; c16 is not slower.
- **Still open:** the engine's startup is about 4 s longer than candidate-54's at `tp4-ring`, as the third build
  recorded; nothing here changes it.

## Release tests

[release-tests.json](release-tests.json) is candidate-54's classification plus three gate files the series changes or
adds (`tests/quantization/test_mxfp4_csf.py` and `tests/v1/core/test_contiguous_kv_packing.py` by #1096,
`tests/v1/worker/test_roce_cpu_affinity.py` by #1134), less the exclusion of the deleted
`tests/model_executor/model_loader/test_b12x_loader.py`. `scripts/release_tests.py check` on the prepared tree: 217
selected files (156 gate, 61 excluded), 0 problems, the same at both trees. The gate manifest has 157 files and 16
deselected tests at `003fd4cc`, bundled in [bundles/release-tests](bundles/release-tests). The test files of #1134's
new `0800` (`test_direct_loader.py`, `test_direct_loader_cuda.py`) are gate files. `0803`'s `test_sparknet_oneshot_collectives_pdl_sources` is a gate
test that passes only with sparknet `0003` in the image. #738's test files (`test_deepseek_v41_tilelang_glp.py`,
`test_deepseek_v4_1_glp.py`) were gate files already, and so are the other files of its suite. With `0805` the check
is unchanged (217 files, 0 problems), and the gate manifest changes only its head (`d6d4319e`).

**First build** (image `8793529c`, dgx4, 2,656 s): 2 failed, 10,242 passed, 357 skipped
([receipts/release-tests-8793529c](receipts/release-tests-8793529c)). The two failures were
`tests/v1/worker/test_workspace.py::test_dsv4_metadata_free_profile_does_not_reserve_split_attention[0]` and `[384]`,
both `ModuleNotFoundError: No module named 'b12x'`. They test DeepSeek V4's B12X attention backend
(`vllm/models/deepseek_v4/nvidia/b12x.py`, not on DS4.1's path) and import `b12x.attention` directly. #1096's own
without-B12X run did not select this file. Both are now deselected with that reason, and the file's other tests still
run. The deselection leaves the vLLM tree unchanged.

## Status

Assembled (2026-11-11). `bin/spark doctor` passes on every configuration (the switched shapes with the example site
maps), and the unit suite passes in full on dgx4 (889 tests).

**First build** (vLLM tree `980e3812`, #1134 at `d9b04765`; dgx1's window, job `20261111T193116-build.json`,
deployment `e90b1e8a`):

| Image | Image ID | Layers | Note |
| --- | --- | --- | --- |
| `spark-ds41f:candidate-55` | `8793529c` | 56 | Built by job `20261111T182946`, reused here because its tree label matched |
| `spark-ds41f:candidate-55-no974` | `f870f811` | 53 | Built on dgx4 in 466 s |

- Each image has the same layers on all of dgx1..dgx4.
- **Smoke:** job `20261111T182946` failed only at its smoke, which named a module no B12X-free series has. With
  that fixed (#1138), the smoke passes on both images on dgx4's GPU: the 66 DS4.1 modules import without B12X.
- **Release tests:** see above.

**#1134 at `37a7f0ac` folded in** (2026-11-11): vLLM tree `ed91c273`, both locks re-recorded.

**Second build** (dgx1's window, job `20261111T223443-build.json`, deployment `804cef6c`, 22:34-23:54Z):

| Image | Image ID | Layers | Note |
| --- | --- | --- | --- |
| `spark-ds41f:candidate-55` | `4aff1368` | 56 | Built on dgx4 in 11 min |
| `spark-ds41f:candidate-55-no974` | `6c9d7c02` | 53 | Built on dgx4 in 10 min |

- Each image has the same layers on all of dgx1..dgx4. `4aff1368` was then shipped to sp1..sp4 and sp5..sp7.
- **Release tests on `4aff1368`:** pass, 10,242 passed, 357 skipped, 0 failed, 2,652 s on dgx4
  ([receipts/release-tests-4aff1368](receipts/release-tests-4aff1368)).

### Second build at `tp4-ring`: gates pass, screen mixed

Details and every run in [receipts/tp4-ring](receipts/tp4-ring/README.md) (2026-11-11 23:56Z to 2026-11-12 04:17Z).

| Check | candidate-55 | Control (`no974`) |
| --- | --- | --- |
| Gate | 12 of 12 boots, 48 of 48 runs identical; candidate-54's TP4 hashes 18d61961/d6568649 | 2 of 2 boots, 8 of 8 runs; the same hashes |
| c1 decisions | the same on 4 of 4 boots (59 requests), and the same as candidate-54's 4 boots on the 58 requests both sent | not run |
| c1 paired decode against candidate-54 | -1.59% ± 0.13% decode time per request (+1.64% tok/s; cross-session) | not run |
| Screen against candidate-54 | grid **0.77% slower**, c16 output 0.94% and paired decode 0.91% faster, prefill the same at 32K (-1.52%) and **3.25% slower at 256K** | grid 0.40% slower, c16 output the same, prefill **2.50% slower at 32K and 3.80% at 256K** |
| Time to ready, warm boots | 111.7-124.7 s, one at 176.3 s; candidate-54's gate 110.3-113.8 s | 111.5-119.2 s |

- **#974 does not change temperature-0 output at TP4**, contrary to the expectation above: both images give
  candidate-54's hashes.
- **Against the control on the same window, candidate-55 is the same or better:** grid, c16 and 32K the same, 256K
  0.80% faster, the model step identical (KL 0). #974's serving mark at `tp4-ring` passes.
- **The prefill and grid regressions are in the control too**, so they come from #1134, #1096, #989, #982 or the
  pre-launch defragmentation. The series misses #989's serving mark (prefill not worse at 32K and 256K) at
  `tp4-ring`. Which change causes it is not yet measured.

### Second build at `tp4-switched` and `tp3-switched`: no boot reaches ready

The failed starts' logs are in [receipts/failed-starts](receipts/failed-starts); each failure rolled the testbed back
to candidate-54.

- **`tp4-switched`** (gate and c1 check, 23:59Z and 00:03Z): the startup memguard stopped sp1's engine (exit 137) both
  times, at `MemAvailable` 4.78 GiB and 4.89 GiB (5 GiB threshold), after weight load. sp1's sponge soaked 32,512 MiB
  (254 chunks) with 58,110 MiB available on this 4 KiB-page kernel, and `0800` holds it through post-load processing.
- **`tp3-switched`** (gate, 00:04Z): the TP workers on sp5, sp6 and sp7 all raised `RuntimeError: batch destination
  exceeds CUDA device allocation` during weight load, from the direct loader's `batch_execute` in
  `finish_weight_transfers()`. Memory was not short: the sponges were 10.6-12.5 GiB with 78-86 GiB available.
- **`tp4-switched` fails #1134's serving mark** (every boot reaches ready). It does not appear on `tp4-ring`'s
  64 KiB-page kernel.
- **The `tp3-switched` failure is #974's, not #1134's,** as #1145 later showed on sp8. TP3 shapes put
  `embed_tokens`, `lm_head` and DSpark's `markov_w1` in display carve-out buffers imported as dma-bufs. CUDA 13.5's
  compat driver (`libcuda` 620.70.06 over the host's 580.178.04) reports those pointers as host memory with base 0,
  and the direct loader's range check rejects them. TP4 shapes have the carve-out off.

**#1134 at `6757c7a1` and #1145 folded in** (2026-11-12, owner: "Yes, rebuild"): `0800` fixed for 4 KiB pages,
`0804` added; vLLM tree `e8c20f54`, both locks re-recorded. The second build's images and every result above are of
tree `ed91c273` and stay as its record. The third build replaces both images, and its gates and screens are run
again at every testbed. Their specs' runs are named `<run>-b3`, so the second build's result folders on the nodes
(which the receipts list by hash) are not overwritten.

**Third build** (dgx1's window, jobs `20261112T070706-build.json` and `20261112T073136-build.json`, 07:07-08:38Z):

| Image | Image ID | Layers | Note |
| --- | --- | --- | --- |
| `spark-ds41f:candidate-55` | `ad905a6d` | 56 | Built on dgx4 in 11 min by the first job, which then failed on a stale control image still referenced by exited containers |
| `spark-ds41f:candidate-55-no974` | `1d02f07b` | 53 | Built on dgx4 in 10 min by the second job |

- Each image has the same layers on all of dgx1..dgx4. `ad905a6d` was then shipped to sp1..sp4 and sp5..sp7.
- **Release tests on `ad905a6d`:** pass, 10,243 passed, 357 skipped, 0 failed, 2,628 s on dgx4
  ([receipts/release-tests-ad905a6d](receipts/release-tests-ad905a6d)). The one pass more than `4aff1368` is
  `0804`'s carve-out test, which ran because dgx4 has `/dev/dri/card0`.

### Third build: every boot ready, every gate identical

Details and every run in [receipts/third-build](receipts/third-build/README.md) (2026-11-12 08:02Z to 13:02Z, deployment
`bbfe2618`). Every boot reached ready (44 of 44) and no memory guard fired.

| Check | `tp4-ring` | `tp4-switched` | `tp3-switched` |
| --- | --- | --- | --- |
| Gate | 12 of 12 boots, 48 of 48 runs; control 2 of 2, 8 of 8; candidate-54's TP4 hashes 18d61961/d6568649 | 4 of 4, 16 of 16; the same hashes | 4 of 4, 16 of 16; candidate-54's TP3 hashes 5c22ce41/bce6092e |
| c1 decisions | 4 of 4 boots agree, and with candidate-54's 4 on the 58 requests both sent | 4 of 4, and with candidate-54's on the 57 both sent | not run |
| c1 paired decode against candidate-54 | **-1.59% ± 0.17%** decode time (+1.63% tok/s) | **-2.28% ± 0.08%** (+2.35% tok/s) | `single-stream` +1.42% decode, `sampled` c1 +1.60% |
| Screen against candidate-54: grid | **0.67% slower** (control 0.22%, same) | **1.11% slower** | 2.29% faster |
| c16 (c8 at TP3) | paired decode +0.75% (better), output +0.86% (same) | +1.49% and +1.47% (better) | +0.43% and +0.57% (same) |
| Prefill | 32K the same (-2.31%), **256K 3.37% slower** (control 3.08%) | 32K and 256K the same | 32K 6,401 and 256K 6,329 tok/s (first TP3 measurement); `stall-3` **1.32% slower** |
| candidate-55 against the control, same window | grid, c16 and prefill the same; model step identical (KL 0) | | |
| Time to ready, engine | 105.0 s mean, candidate-54 101.0 s; the control the same as candidate-55 | 117-127 s | 112-126 s |

All screen and c1 comparisons are against candidate-54's recorded runs, so cross-session.

- **The third build fixes both switched failures:** #1134's `0800` at `6757c7a1` passes sp1's memguard on 4 KiB pages,
  and #1145's `0804` lets `tp3-switched` load its weights.
- **#974 changes no temperature-0 output at TP3 either**, and its serving mark at `tp4-ring` passes again: against
  the control, grid, c16 and prefill are the same.
- **The 256K regression at `tp4-ring` is in the control too**, and does not appear at `tp4-switched`. Its cause is
  among #1134, #1096, #989, #982 and the pre-launch defragmentation.
- **The grid is slower on both TP4 testbeds and faster at TP3.** The cause is under investigation separately.
- **Boot times:** the engine's startup is about 4 s longer than candidate-54's at `tp4-ring`, in the control too. In
  3 of 66 boots over both builds, the time before launch (stopping the previous engine and the pre-launch
  defragmentation) was about 57 s longer than usual; the engine's own time was normal.

Marks after the third build:

| PR | Own mark | Serving mark |
| --- | --- | --- |
| #974 | Not run | **Met** at `tp4-ring`: not worse than the control on grid, c16 and prefill |
| #1134 | Not run | Every boot reaches ready: **met** (44 of 44). The screens' repeat boots differ by 0.4% or less on the grid and decode rows, inside the recorded A/A floors; against candidate-54's repeats some rows are further apart and some closer, which two boots cannot separate |
| #1096 | Release tests pass; the no-B12X smoke passed on the first build's images | Inert: **met** (the control gives candidate-54's hashes). Time to ready not worse: **not met as measured** (engine +4 s, cross-session; not attributed between #1134 and #1096) |
| #989 | Not run | Prefill not worse: **met** at `tp4-switched`, **missed** at `tp4-ring` (256K 3.37% slower, the control too; not attributed) |
| #982 | Not run | **Met:** 12 of 12 gate boots identical at `tp4-ring`, and the c1 decision checks agree |
| #1145 | **Met:** its test passes on dgx4 (`/dev/dri/card0`), and the release tests pass | **Met:** every `tp3-switched` boot loads its weights; inert at TP4 (candidate-54's hashes) |

**GLP-39's steered replies on the third build** (2026-11-12, before #738 is folded in): [glp39-steered-tp4.json](glp39-steered-tp4.json)
boots `tp4.json` (the third build's image, GLP-39 configured, as every measured arm) twice at `tp4-ring` and runs
[dual_probe.py](../../2026-10-28-glp-dual-serving/dual_probe.py) without `--speed`. Its steered replies (token IDs and
log probabilities, greedy) are the reference for #738's claim that a Spec v1 vector steers as before. Both boots
pass every check, steering changes all 8 prompts, and both give the same replies: steered `739b8abc`, stock
`9872053e` ([receipts/third-build](receipts/third-build/README.md)).

**#738 folded in** (2026-11-12, owner: "prepare the GLP update into candidate-55"):
- **Series and locks:** `0805` added after `0804`. The vLLM tree is `8aaae3df`, both locks are re-recorded, and
  every profile's `vllm.tree` label follows.
- **GLP-27:** [glp27-tp4.json](glp27-tp4.json) adds it, with [qualify-glp27-tp4.json](qualify-glp27-tp4.json) and
  #738's marks declared above. The file is in the Hugging Face cache on dgx1..dgx4 at revision `4d570926`, with
  sha256 `24e97d41`. AdvBench's data is on dgx1.
- **Third build:** its images and results are of tree `e8c20f54` and stay as its record.
- **Fourth build:** it replaces both images and runs every gate, check and screen again. Its specs' runs are named
  `<run>-b4`.
