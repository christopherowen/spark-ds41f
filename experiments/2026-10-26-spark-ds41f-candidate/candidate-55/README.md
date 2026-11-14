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
| Kernel cache key (owner, 2026-11-13) | TileLang's kernel cache key names the CUDA device compiler and its version, so a node whose cache an earlier toolkit filled compiles this build's kernels instead of loading the old ones | TileLang `0014` ([tilelang/series](tilelang/series): `spark-ds41f-1107`'s thirteen patches, then `0014`) |
| Row-band WO window (owner, 2026-11-13: fifth build "with both patches") | `VLLM_DS41_L2_PREFETCH_WO_BANDS` sizes the WO prefetch window, and COMP's, by the forward's row count; unset, nothing changes. The TP4 recipe sets `17-32=10` | vLLM `0806` ([the experiment](../../2026-11-13-l2-prefetch-row-bands/README.md)) |
| Vision GELU table (2026-11-13, the fifth build's release tests) | The aligner's fused GELU reads torch's own `F.gelu` of each BF16 value from a table instead of calling the compiling toolkit's `erff`, so it is DeepSeek's reference under CUDA 13.5 as under 13.0 | vLLM `0807` |
| Torch for the toolkit (owner, 2026-11-13: "we should update pytorch to match our cuda build", "use tonight's nightly", "just add it to candidate-55") | torch `2.16.0.dev20261113+cu135` over the base image's `2.13.0+cu130`, from 22 hash-locked wheels; every native extension is built against it. PyTorch has no stable cu135 build, and no cu135 torchvision, torchaudio or torchcodec: those are the cu134 nightlies of its date | `sources.torch` in this lock ([requirements/torch-2-16-cu135-aarch64.txt](../../../requirements/torch-2-16-cu135-aarch64.txt)) |
| NCCL 2.32.3 (required by that torch) | Its `libtorch_cuda.so` binds nine NCCL symbols 2.30.7 lacks, so the four NCCL patches are rebased onto `v2.32.3-1` | [nccl/series](nccl/series) |
| Classic NCCL process groups (2026-11-14, [boot-memory](boot-memory/README.md)) | `TORCH_DIST_USE_NCCL2=0` in every shape's environment: torch 2.16's default process group creates an NCCL communicator for each of vLLM's groups at construction, about 1.3 GiB a rank at `tp3-switched`; the classic one creates it on first use, as torch 2.13 did. Inert on a torch without the variable | recipe |
| Split canonicalization (required from torch 2.15) | Upstream vLLM #57299: torch 2.15 regenerates split outputs as slices, which stopped two fusion passes from matching | vLLM `0808` |
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
  belongs in the promotion commit of a release whose image reads Spec v3. **Owner, 2026-11-13: "Yes, at
  promotion"**: if candidate-55 is promoted, its promotion commit makes GLP-27 the default vector (#985's change),
  qualified at all three testbeds ([receipts/glp](receipts/glp/README.md)). Until then `config/cluster.json` keeps
  GLP-39.
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
  678 patches, `patch_head` `d6d4319e`, `expected_tree` `8aaae3df`, `patchset_sha256` `6fdf03dc`, through the fourth
  build. The fifth build adds `0806` (the row-band WO window): 679 patches, `patch_head` `1878808a`, `expected_tree`
  `7d05b12b`, `patchset_sha256` `ece15474`, recorded by `bin/spark build prepare --only vllm --lock` on sp8 for both
  locks. `7d05b12b` is the tree the row-band experiment measured as an overlay. The sixth build adds `0807` (the
  vision GELU table): 680 patches, `patch_head` `0f55693e`, `expected_tree` `e630ad15`, `patchset_sha256` `7ce17bdf`,
  recorded the same way. The seventh build adds `0808` (upstream #57299): 681 patches, `patch_head` `5a190b19`,
  `expected_tree` `3782bf90`.
- NCCL ([nccl/series](nccl/series)) on `v2.32.3-1` (`12df1a11`), from the seventh build: `0001` and `0002` applied
  unchanged, `0003` with one conflict in a log line, `0004` ported by hand to `src/tuning/tuning_general.cc`, where
  2.31 moved the sizing. `patch_head` `d39a5a7b`, `expected_tree` `f6e56acc`. The rebuilt library exports the 60 NCCL
  symbols torch needs. Builds one to six, and the `no974` control, are on `v2.30.7-1` with the 2026-10-03 series.
- torch `2.16.0.dev20261113+cu135` from the seventh build (`sources.torch`); its CUDA libraries are the toolkit's
  versions (13.5.37, cuBLAS 13.10.0.20). Not carried: upstream #58189 (this torch has the fix), and a
  `tokenspeed-mla` bump (nothing on DS4.1's path imports it; it conflicts with Triton 3.9 only when loaded).
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
| #738 (declared 2026-11-12, before the fourth build) | Its suite passes in the release tests: the GLP kernel and model tests, `glp_steer_write`'s PDL-chain entry, the weightless, CED and SP-prefill model tests. The image's reader accepts GLP-39 and GLP-27 on the head (#1149's GLP reader check in `cluster start`'s preflight, at the boots of `tp4.json` and `glp27-tp4.json`) | **Spec v1 unchanged:** GLP-39's steered replies ([glp39-steered-tp4.json](glp39-steered-tp4.json), `dual_probe.py`) identical in token IDs and log probabilities to the third build's on both boots: steered `739b8abc`, stock `9872053e` ([dual_replies.py](dual_replies.py)). **Stock unchanged:** every gate gives candidate-54's hashes with GLP-39 configured, and [qualify-glp27-tp4.json](qualify-glp27-tp4.json)'s gate runs give them with GLP-27 configured. **GLP-27 qualified at `tp4-ring`:** every `dual_probe.py` check passes on both boots; its steered speed and AdvBench rates are recorded, not gated. **GLP-27 qualified at `tp4-switched` and `tp3-switched`** (2026-11-13): the same, on both boots of each; the stock name keeps each shape's hashes, and both TP4 testbeds give the same steered replies ([receipts/glp](receipts/glp/README.md)) |
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
  ([2026-10-28-glp-dual-serving](../../2026-10-28-glp-dual-serving/README.md)). Added 2026-11-13: the same
  qualification at `tp4-switched` and `tp3-switched` ([qualify-glp27-tp4-switched.json](qualify-glp27-tp4-switched.json),
  [qualify-glp27-tp3-switched.json](qualify-glp27-tp3-switched.json)); every GLP run's results are in
  [receipts/glp](receipts/glp/README.md). `advbench.py` prints every reply; its output stays on the testbed's head
  under `results/private`, and the receipts keep only each boot's summary line (the rates per name).
- **Pre-launch defragmentation:** candidate-55's and the control's boots defragment before launch, but candidate-54's
  recorded references were taken without it. A difference against those references includes the defragmentation,
  which counts as part of #1134.

## TP4's recipe: L2 prefetch caps, and the ring's collective limit

**The row band, added with the fifth build (2026-11-13).** `tp4.json` and `tp4-switched.json` set
`VLLM_DS41_L2_PREFETCH_WO_BANDS=17-32=10`: the WO window (and COMP's) at 10 MB for forwards of 17 to 32 rows, 15 MB
elsewhere.

- **Why:** the 24- and 32-row grid shapes; see [the row-band experiment](../../2026-11-13-l2-prefetch-row-bands/README.md).
- **Measured** on the fourth build with the patch mounted and the kernel cache its own compiler filled, two boots each
  ([the window sweep](../../2026-11-13-prefetch-window-sweep/README.md)): 17 to 32 rows 1.7% to 2.7% faster at both
  TP4 testbeds, the grid 0.43% and 0.62% faster, outputs candidate-54's. The sweep found no better WO, FFN or NEXT
  size at any other row band.
- **The fifth build's screen** (below, under Status): 256K prefill is 1.03% lower than candidate-54's at `tp4-ring`,
  beyond the margin; 12x72 is level with the recipe without the band (+0.17%); 16x96 reads 1.07% slower than it again
  and is level with candidate-54 (-0.02%).
- **Not set elsewhere:** `tp3-switched`'s sweep picked `17-32=12` on one boot and two boots did not confirm it (grid
  -0.06%), so TP3 has no band; the triangle and TP5 to TP8 have no measurement.
- `tp4-noband.json`, `tp4-switched-noband.json` and their `-dev` files are the recipe without the band. The arms
  measured before it entered (grid-attribution, rows24-attribution, the kernel cache, row-band and sweep experiments)
  extend those, so they still mean what they measured.
- The fifth build's verification specs are `gate-b5-*.json`, `screen-b5-*.json` and `c1-decisions-b5-*.json`.

**Added 2026-11-12, after the fourth build's screens.** Two recipe variables answer the two regressions the screens
found at TP4. Neither changes the image or the locks.

### L2 prefetch caps, at `tp4-ring` and `tp4-switched`

candidate-55's TP4 profiles ([tp4.json](tp4.json), [tp4-switched.json](tp4-switched.json), and their `-dev` files
and [glp27-tp4.json](glp27-tp4.json) through them) set `VLLM_DS41_L2_PREFETCH_WO_MB=15`,
`VLLM_DS41_L2_PREFETCH_FFN_MB=12` and `VLLM_DS41_L2_PREFETCH_NEXT_MB=12`, the caps candidate-54's TP3 profiles carry.
The code defaults are 20/16/20 MB. TP5 to TP8 are not measured and keep the defaults, and so does the control
([no974/tp4.json](no974/tp4.json)).

- **Why:** the grid regression at TP4 is in the 1-request rows: 1x2 rows are 9% slower than candidate-54 and 1x3
  rows 8% slower at `tp4-switched`, while the prefetch windows are at their defaults
  ([grid-attribution](grid-attribution/README.md)). **Corrected 2026-11-13:** this read "#982's `T.pdl_trigger()`
  before `roce_wait_peers` causes it". The arms behind that mounted nothing. Run with its files mounted at `tp4-ring`
  ([rows24-attribution](rows24-attribution/README.md)), #982 is a gain of 3.1% at 1x2 and 2.8% to 4.5% at 1x3 under the caps, not the cause.
- **Measured** at `tp4-switched` on the fourth build, the grid step alone, two boots an arm in one session: the caps
  make the grid 4.66% faster than candidate-55 as built (1x2 rows 18%, 1x3 rows 13%), and 1.7% faster than
  candidate-54 with the same caps. sparknet `0003` is unchanged. **Corrected 2026-11-13:** the arm read here as
  "moving #982's trigger" (2.5% alone, nothing under the caps) changed only the kernel cache directory; see below.
- **Measured** at `tp4-ring` the same way: the grid 2.51% faster than candidate-55 as built (margin 0.34%), 1x2 rows
  16.2 to 13.5 ms and 1x3 rows 14.8 to 14.0 ms, where candidate-54's recorded screen has 14.5 ms for both.
- **The windows were already too large for candidate-54 at TP4:** with the caps its 1x2 rows are 9% faster. So the
  caps are a gain of their own as well as the fix, and candidate-54's recorded TP4 screens are without them.
- **No output change:** the windows are cache hints, and the arithmetic is unchanged.
- A recipe variable, not a patch: the image and the locks are the fourth build's.

### Found 2026-11-13: candidate-55 has been running kernels compiled before #974

TileLang's kernel cache (`tilelang/cache/kernel_cache.py`) is keyed by the kernel and the TileLang build, not by the
compiler that built its device binary; the CUDA binary cache below it is keyed by `nvcc`'s version too, but is asked
only on a miss. The nodes' caches were filled by CUDA 13.0.2 builds: on dgx1, 84 of the 7,033
files in the cache directory candidate-55 uses are newer than its first build, and on sp1 56 of 8,699. So wherever a
kernel's source is candidate-54's, candidate-55 loads the binary an earlier build compiled, and #974's `nvcc` and
`ptxas` compile only the kernels whose source changed. Every gate, check and screen above was taken that way.

- **A node with an empty cache compiles every kernel with CUDA 13.5** and is a different thing from what was
  verified. Two attribution arms booted that way by accident and were 5.9% and 10.8% faster at 1x2 and 1x3 with the
  default windows ([grid-attribution](grid-attribution/README.md)).
- **#974's marks are open again:** "changes no temperature-0 output" and its serving marks were measured on kernels
  its compiler did not build; they hold for its runtime libraries only.
- [2026-11-13-kernel-cache-toolchain](../../2026-11-13-kernel-cache-toolchain/README.md) measures candidate-55 on its
  own kernels, gate first.
- **Fixed in the series** (owner, 2026-11-13: "make the compiler version part of the cache key"): TileLang `0014`
  adds the device compiler to the kernel cache key. TileLang's tree becomes `741806c7` (`patch_head` `444b0447`,
  14 patches, `patchset_sha256` `c0a8825c`), recorded by `bin/spark build prepare --only tilelang --lock` on sp8 for
  both locks, and every profile's `tilelang.tree` label follows. The fourth build's image has the earlier tree, so
  these profiles need the **fifth build**, whose first boot on each node compiles its whole kernel set.
- **Fifth build** (owner, 2026-11-13: "go", with both patches): TileLang `0014` and vLLM `0806`, built on sp8 from
  [image-sp8.json](image-sp8.json). `0806` is inert until a recipe sets `VLLM_DS41_L2_PREFETCH_WO_BANDS`; no profile
  here sets it. The `no974` control shares the vLLM and TileLang series, so its lock and labels follow; its image is
  not rebuilt and stays the fourth build's record.

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
- **At `tp4-ring`, four shapes of 24 and 32 rows are 1% to 2% slower than candidate-54's on both boots** (1.3% to 3.3%
  without the caps); c16 is not slower. **Attributed 2026-11-13** ([rows24-attribution](rows24-attribution/README.md)):
  it is the WO prefetch window at 15 MB in the 32-row decode tile tier, not a kernel or one PR's setting. With the
  prefetch off candidate-55 is faster than candidate-54 at those shapes, and a 10 MB WO window makes them 2.6% faster
  at the cost of 1.9% at 1x2, so the recipe is unchanged; a WO window sized by row band is the fix, for a later
  candidate.
  The change that moved those shapes is #982: without it they are 1.8% to 3.1% faster and 1 to 16 rows 1.1% to 4.5%
  slower, the grid 0.7% slower, so #982 stays. The row-band window, measured as an overlay at both TP4 testbeds, makes the
  24-row shapes 2.2% to 2.7% faster and 16x32 1.7% to 2.1% with nothing else moved
  ([the experiment](../../2026-11-13-l2-prefetch-row-bands/README.md)); its output gate is still to run, and it is not in this image.
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

**Fifth build** (image `c8c22dc2`, sp8, 5,672 s): **3 failed**, 10,292 passed, 357 skipped
([receipts/release-tests-c8c22dc2](receipts/release-tests-c8c22dc2)). All three are the vision aligner's fused GELU
against DeepSeek's reference (`F.gelu`), in `tests/kernels/test_deepseek_v41_tilelang_vision.py`:
`test_gelu_epilogue_is_the_reference[144]` (3 of 36,864 values) and `test_gelu_epilogue_is_the_reference_everywhere[gemm]`
and `[gemv]` (7 of the BF16 inputs a projection can produce, the same seven in both kernels).

- **Cause: the toolkit (#974), not the fifth build's patches.** The kernel calls CUDA's `erff`, and the image's torch is
  `2.13.0+cu130`, whose `erff` is CUDA 13.0's. CUDA 13.5's differs from it by one FP32 step at these arguments, and the
  seven inputs (-3.140625, -4.03125, -4.28125, -4.5625, -5.0, -5.25, -5.34375) are those where that step moves the BF16
  rounding of `(0.5 * x) * (1 + erff(x * M_SQRT1_2))`. The largest change is one BF16 step at 0.0026; the others are
  below 1.2e-4.
- **Builds one to four passed on kernels an earlier toolkit compiled.** The bundle's kernel cache
  (`/c/tilelang-release-tests`) was filled on sp8 on 2026-11-10, and its key did not name the compiler. There, the third
  build's image gives 0 differing inputs with that cache and the same 7 with an empty one. TileLang `0014` is what made
  the fifth build compile its own.
- **Effect:** image inputs. The text gates are unaffected (candidate-54's hashes at all three testbeds on these
  kernels). Not fixed in this build.
- **Fix: vLLM `0807`, for the sixth build.** `gelu_table` is `F.gelu` of each of the 65,536 BF16 values, built by
  torch on the device; the GEMM's and the GEMV's GELU epilogues take it as an operand and read the entry at the
  rounded biased value's bits. The epilogue is then torch's result under any toolkit. Mounted over the fifth build on
  sp8 with an empty kernel cache: the vision file's 108 tests pass, and the first projection with its epilogue takes
  the same time with either epilogue (404 to 408 us at 4 and 8 rows, 790 to 801 us at 144, 1,948 to 1,964 us at 576).

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
  sha256 `24e97d41`, and on sp1..sp7 at the same revision and sha256. AdvBench's data is on dgx1, and on sp1 and sp5
  (copied from dgx1 on 2026-11-13, sha256 `7cafb159`).
- **Third build:** its images and results are of tree `e8c20f54` and stay as its record.
- **Fourth build:** it replaces both images and runs every gate, check and screen again. Its specs' runs are named
  `<run>-b4`.

### Fifth build: gates identical, the grid ahead at TP4, prefill 1% behind, three release tests fail

Image `spark-ds41f:candidate-55`, ID `c8c22dc2`, the same on dgx1..dgx4, sp1..sp4 and sp5..sp7 (2026-11-13). Every job
ran at `f821d9f4d`. Each node's first boot compiled its kernel set (about 22 minutes), with the earlier toolkit's
entries still in the cache: TileLang `0014` acting. Receipts: [receipts/fifth-build](receipts/fifth-build).

| | `tp4-ring` | `tp4-switched` | `tp3-switched` |
| --- | --- | --- | --- |
| Gate | 2 of 2 boots, 18d61961/d6568649 (text 0a2dacdd/12fd8257) | 2 of 2, the same | 4 of 4, 5c22ce41/bce6092e (text 26a739a9/12cfee9b) |
| Grid against candidate-54's recorded screen | **2.22% faster** (±0.14%, margin 0.35%) | **4.14% faster** (±0.17%, margin 0.29%) | **2.90% faster** (±0.13%, margin 0.56%; 15 shapes) |
| c16 paired decode | +0.36%, the same (margin 0.51%) | **+1.39%, better** (margin 0.47%) | |
| Prefill 32,768 | -0.15%, the same | -0.88%, the same | |
| Prefill 262,144 | **-1.03%, worse** (±0.22%, margin 0.69%; 8,803 and 8,723 tok/s) | -0.84%, the same (margin 1.07%; 8,651 and 8,705 tok/s) | |
| c1 decisions, 4 boots | agree on 59 requests; 82.49 to 82.83 tok/s output | agree on 59 requests; 82.52 to 82.73 tok/s | |
| Warm time to ready | 115 and 118 s | 132 and 133 s | |

- **The grid** is the fourth build's with its own kernels and the band: at `tp4-ring` -2.22% here against -1.96% for
  that pair mounted over the fourth build, and -1.55% recorded on the earlier toolkit's kernels.
- **The 24- and 32-row shapes** are 1.4% to 1.9% faster than candidate-54's at `tp4-ring` (24 rows) and level at
  16x32 (+0.33%); they were 1.3% to 1.9% slower.
- **Forwards of 32 rows and more are 0.18 to 0.34 ms slower than candidate-54's at `tp4-ring`** (16x32, the three
  48-row shapes, 12x72, 16x96), which the drafter and the rest of the step give back (-0.18% to +0.85% a step). Not
  attributed; the prefill loss on these kernels is not attributed either.
- **Release tests:** three fail, above.
- **`tp3-switched`'s screen** (two boots, against candidate-54's recorded screen): c8 paired decode +0.64%, better
  (margin 0.34%); single-stream paired decode +1.50%, better (margin 0.29%; 64.05 and 64.04 tok/s); the sampled c16, c8
  and c1 the same (+0.61%, +0.60%, +1.22%); prefill of 131,072 tokens under decode **2.31% slower at 3 streams**
  (±1.07%, margin 1.11%: worse), the same at 6 and 7 (-1.44%, -1.75%). Prefill is lower on this toolkit's kernels at
  all three testbeds; not attributed.
- **Not done:** the c1 decisions against candidate-54's; TP5 to TP8.
- **Correction:** sp8's `spark-ds41f:candidate-55` was the third build until the fifth replaced it, so the single-GPU
  checks of 2026-11-13 on sp8 that name the fourth build ran on the third (the row-band overlay's smoke run among
  them). The testbed runs are unaffected.

Against `spark-ds41f-1107`, composed from candidate-54's recorded screens against it:

| | `tp4-ring` | `tp4-switched` |
| --- | ---: | ---: |
| Grid | 7.05% faster | 8.20% faster |
| c16 paired decode | +1.71% | +2.54% |
| Prefill 262,144 | +6.24% | +6.66% |

### Sixth build: the fifth with vLLM `0807`

The sixth build's tree differs from the fifth's in three files: the vision kernels, the aligner and their test. Its
specs' runs are named `<run>-b6`. Marks, declared before the build:

1. **Release tests:** none fails on the sixth build's image, with the kernel cache CUDA 13.5 filled.
2. **Outputs:** the gate gives candidate-54's hashes on every boot at `tp4-ring`, `tp4-switched` (18d61961/d6568649)
   and `tp3-switched` (5c22ce41/bce6092e).
3. **Serving:** at each TP4 testbed the screen's grid, c16 and prefill rows are the same as the fifth build's two
   boots, within the margin. No text path changes, so a difference would be a finding about the build, not the patch.

Not run again: the c1 decision checks and `tp3-switched`'s screen, which stand as the fifth build's.

**Verified, 2026-11-13.** Image `spark-ds41f:candidate-55`, ID `0728619c`, vLLM tree `e630ad15`, the same on dgx1..dgx4,
sp1..sp4 and sp5..sp7; every job ran at `c4341e032`. The fifth build's image is kept as `spark-ds41f:candidate-55-b5`
on every node. Receipts: [receipts/sixth-build](receipts/sixth-build),
[receipts/release-tests-0728619c](receipts/release-tests-0728619c).

1. **Release tests: met.** 10,295 passed, 357 skipped, none failed (sp8, 2,954 s); the three GELU tests pass on kernels
   CUDA 13.5 compiled.
2. **Outputs: met.** 2 of 2 boots at `tp4-ring` and `tp4-switched` give 18d61961/d6568649 (text 0a2dacdd/12fd8257),
   4 of 4 at `tp3-switched` give 5c22ce41/bce6092e (text 26a739a9/12cfee9b).
3. **Serving: met.** Against candidate-54's recorded screens, beside the fifth build's:

| | `tp4-ring`, sixth | fifth | `tp4-switched`, sixth | fifth |
| --- | ---: | ---: | ---: | ---: |
| Grid | -2.10% (margin 0.30%) | -2.22% | -3.96% (margin 0.29%) | -4.14% |
| c16 paired decode | +0.45% | +0.36% | +1.48% | +1.39% |
| Prefill 32,768 | +0.17% | -0.15% | -0.15% | -0.88% |
| Prefill 262,144 | -1.02% (8,773 and 8,755 tok/s; worse) | -1.03% | -0.39% (8,697 and 8,739 tok/s; the same) | -0.84% |

Open, carried from the fifth build: 256K prefill at `tp4-ring` is 1% behind candidate-54 beyond the margin.

### Seventh build: torch for the toolkit, NCCL 2.32.3, vLLM `0808`

Three variables that cannot be separated: this torch does not load without NCCL 2.32.3, and vLLM's fusion passes
need `0808` on it. The `no974` control cannot take them (no CUDA 13.5 toolkit) and stays the sixth build's. Specs'
runs are named `<run>-b7`. Built on sp8 as `spark-ds41f:candidate-55-b7`, image ID `71293710`; the sixth build keeps
`spark-ds41f:candidate-55` until this one is distributed.

Before the testbeds, on sp8 (2026-11-13):

- The image build's checks pass: torch is the lock's, reports CUDA 13.5 and NCCL 2.32.3, and loads the patched
  library and no other; the GPU, TileLang and sparknet smokes pass and the 66 DS4.1 modules import.
- torch's compiled architectures stop at `sm_120`, which GB10 (12.1) runs: a BF16 matmul and a CUDA graph capture
  and replay work.
- **torch's own GELU moved with the toolkit.** At the seven inputs of the fifth build's failures, this torch's
  `erff` is CUDA 13.5's and its `F.gelu` gives the values the 13.5-compiled kernel gave before `0807`. The kernel
  reads torch's table, so it differs from torch at 0 inputs on either torch. Image inputs therefore differ from
  candidate-54's at those inputs on this build, with the reference.
- vLLM `0808`'s tests on torch 2.13: 15 passed, 33 skipped (they need a model the node does not hold).
- NCCL 2.32.3 with the rebased patches has run no collective yet.

Marks, declared before the testbed runs:

1. **Release tests:** none fails on the seventh build's image.
2. **Every boot reaches ready** at `tp4-ring`, `tp4-switched` and `tp3-switched` (NCCL's ring and channel patches on
   a reworked tuner).
3. **Outputs:** every boot of a testbed gives the same hashes. The text hashes are candidate-54's
   (0a2dacdd/12fd8257 at TP4, 26a739a9/12cfee9b at TP3). The log-probability hashes are candidate-54's too, or the
   difference is shown to be torch's own arithmetic on the logits, not a kernel's.
4. **Serving:** at each testbed the screen's rows are not worse than the sixth build's two boots beyond the margin
   (the fifth build's at `tp3-switched`), and the c1 decisions agree on 4 of 4 boots at both TP4 testbeds.
5. **Time to ready** not longer than the sixth build's warm boots by more than 5 s.

**The seventh build did not boot at `tp4-ring`** (2026-11-14 04:27Z to 04:30Z; runs `candidate-55-gate-tp4-b7`,
`…-screen-tp4-b7`, `…-c1-decisions-tp4-b7`, each failed in under a minute, deployment `2e46a5a8f`).
`torch.distributed.new_group` raised `NCCL split failed: invalid usage`, NCCL's last error
`SWITCHLESS_BIDIRECTIONAL needs 3 or 4 nodes, one rank per node, an even channel count >=2 and unshared resources`.
torch 2.16's default process group makes each new group by splitting its parent's communicator, which shares the
parent's resources; NCCL `0002` refuses that. It is the same default that cost `tp3-switched` its memory
([boot-memory](boot-memory/README.md)), and `TORCH_DIST_USE_NCCL2=0` selects the classic group, which creates its own
communicator. The ring's jobs run again with it as `…-b7-r2`. The ring's image was built on dgx4 from the same lock
(the copy from sp8 did not complete), so its image ID is not sp8's.

**Seventh build outcomes (2026-11-14; dead end, not promoted):**

1. **Release tests: not met.** 172 failed, 10,123 passed (torch nightly bumped Triton from 3.7.1 to 3.9; all 172 failures compare TileLang sampling, top-k/top-p, and rejection kernels against vLLM's Triton kernels at signed zeros, infinities, NaN, and top-p threshold ties). In addition, the base image's prebuilt DeepGEMM extension fails to load with an undefined torch symbol, and no NIXL wheel is built for torch 2.16.
2. **Every boot reaches ready: not met.** At `tp4-ring`, `-b7` failed in under a minute (`NCCL split failed: invalid usage`). At `tp3-switched`, `screen-b7` failed on boot 2 when `sp5` hit 4.968 GiB lowest `MemAvailable` and tripped the 5 GiB startup guard (5 of 6 boots reached ready across gate + screen). With `TORCH_DIST_USE_NCCL2=0` (`-b7-r2`), `tp3-switched` recovered the 8 extra NCCL communicators (0.90 of the +1.42 GiB head delta; [boot-memory](boot-memory/README.md)), leaving +0.52 GiB on the head (`5.866`–`6.075` GiB lowest boot `MemAvailable` on subsequent `-b7-r2` boots), and `screen-b7-r2` halted its `model` step at `sp5` (`MemAvailable 3.97 GiB` below the 4 GiB bench floor on boot 1).
3. **Outputs: met.** At `tp4-ring` (`-b7-r2`, dgx4 image `03db3ec9`), 2 of 2 boots match candidate-54 (`18d61961`/`d6568649`, text `0a2dacdd`/`12fd8257`). At `tp4-switched` (`-b7`, sp8 image `71293710`), 2 of 2 boots match candidate-54 (`18d61961`/`d6568649`). At `tp3-switched` (`-b7`), 4 of 4 boots match candidate-54 (`5c22ce41`/`bce6092e`, text `26a739a9`/`12cfee9b`).
4. **Serving: not met.** Against candidate-54's recorded screens:
   - `tp4-ring` (`-b7-r2`): grid `-2.18%` (±0.14%, margin 0.30%, better), c16 paired decode `+0.38%` (±0.41%, same), Prefill 32,768 `+0.32%` (same), **Prefill 262,144 `-0.92%`** (8,755 and 8,790 tok/s, ±0.41%, repeat +0.40%, margin 0.62%, **worse**); c1 decisions agree across all 4 boots (59 of 59 requests, and 58 of 58 with candidate-54; `82.62`–`82.82` tok/s).
   - `tp4-switched` (`-b7`): grid `-4.11%` (±0.18%, margin 0.35%, better), c16 paired decode `+1.58%` (±0.43%, better), Prefill 32,768 `-0.50%` (same), Prefill 262,144 `-0.30%` (8,700 and 8,750 tok/s, same); c1 decisions agree across all 4 boots (59 of 59 requests, and 57 of 57 with candidate-54; `82.26`–`82.61` tok/s).
   - `tp3-switched` (`-b7-r2`, without the halted `model` step): `single-stream` c1 paired decode `+1.78%` / `+1.63%` (better); `sampled` c16 paired decode `+1.70%` / `+1.74%` (better), c8 `+0.07%` / `+1.03%` (same); `stall` against candidate-54: `stall-3` prefill 131,072 tok/s under decode `-2.09%` / `-1.96%` (**worse**, `4,768.9` and `4,775.3` vs `4,870.7` tok/s) and decode step p90 `+2.49%` / `+2.20%` (**worse**), `stall-6` p90 `+2.38%` / `+2.51%` (**worse**, boot 2 prefill `-1.86%`, **worse**), `stall-7` p90 `+2.46%` / `+2.82%` (**worse**).
5. **Time to ready: met** where warm boots completed: `tp4-ring` `-b7-r2` `108.8`–`115.4` s (sixth build `115.5`–`116.5` s; first `-b7-r2` boot `103.8` s); `tp4-switched` `-b7` `131.9`–`139.0` s (sixth build `133.3`–`134.4` s); `tp3-switched` `-b7`/`-b7-r2` `123.8`–`129.4` s (fifth/sixth build `127.0`–`135.5` s).

### Eighth build: sixth build's torch 2.13 and NCCL 2.30.7, vLLM `0808`, TP3 collective cap

Reverts PyTorch (`2.13.0+cu130` base image) and NCCL (`2.30.7-1`, `eeacf1c6`) to the sixth build while keeping the CUDA
`13.5.37` toolkit and patches, replaces vLLM `0808` with `0808-perf-dsv41-sink-indexer-candidate-score-reduction-un.patch`
(`b210993fd700`, tree `7914f1f91e56`, sinking the 64-lane `s_frag` sum inside `if slot >= 0` in `_nsa_ScoreKernel` so
`tp4-ring` 256K prefill recovers the CUDA 13.5 regression; `test_deepseek_v41_tilelang.py`: 65 passed, 1 skipped,
1 xfailed in the sixth build's container), and sets `SPARKNET_ROCE_MAX_COLLECTIVE_BYTES=4194304` on TP3 configs
(`tp3-switched.json`, `tp3.json`, `tp3-triangle-4k.json`, matching TP4's 4 MiB cap for `tp3-switched` `stall-3`
prefill under decode). Specs' runs are named `<run>-b8`.

Marks, declared before the build:

1. **Release tests:** 0 failures (target 10,295 passed, 357 skipped).
2. **Every boot reaches ready** at `tp4-ring`, `tp4-switched` and `tp3-switched`, with `sp5` lowest boot `MemAvailable`
   restored to Builds 5 and 6 (~6.3–7.0 GiB).
3. **Outputs:** `18d61961`/`d6568649` (text `0a2dacdd`/`12fd8257`) at `tp4-ring` and `tp4-switched`, and
   `5c22ce41`/`bce6092e` (text `26a739a9`/`12cfee9b`) at `tp3-switched` on every boot.
4. **Serving:** `tp4-ring` Prefill 262,144 and `tp3-switched` `stall-3` prefill under decode are not worse than
   candidate-54 beyond the margin; grid, c16, single-stream and sampled gains are preserved; and `c1-decisions`
   agrees on 4 of 4 boots at both TP4 testbeds and with candidate-54.
5. **Time to ready** not longer than the sixth build's warm boots by more than 5 s.
**Eighth build outcomes (2026-11-14; not promoted — open 256K prefill regression at `tp4-ring`):**

Image `spark-ds41f:candidate-55`, ID `daa18fed` (`sp8`, `sp1..sp4`, `sp5..sp7`) and `6281652f` (`dgx1..dgx4`), `patch_head`
`b210993fd700`, tree `7914f1f91e56`. Release tests receipt: [receipts/release-tests-daa18fed](receipts/release-tests-daa18fed).

1. **Release tests: met.** `10,295 passed, 357 skipped`, 0 failed on `sp8` (`2,910.2 s`).
2. **Every boot reaches ready: met.** All boots completed with `halted: null` and `swap_growth_gib: 0.0` across
   `tp4-ring` (`dgx1` lowest boot `MemAvailable` `23.353–23.361 GiB` across screen boots), `tp4-switched` (`sp1`
   `14.159–14.475 GiB`), and `tp3-switched` (8 of 8 screen boots completed; `sp5` lowest boot `MemAvailable` restored
   to `6.274–6.535 GiB`).
3. **Outputs: met.** `tp4-ring` (2 of 2) and `tp4-switched` (2 of 2) give `18d61961`/`d6568649` (text
   `0a2dacdd`/`12fd8257`); `tp3-switched` (4 of 4) gives `5c22ce41`/`bce6092e` (text `26a739a9`/`12cfee9b`).
4. **Serving: partially met.** Against candidate-54's recorded screens:
   - `tp4-ring`: grid `-2.11%` (±0.14%, margin 0.30%, better), c16 paired decode `+0.39%` (±0.41%, margin 0.50%,
     same; `270.88` and `271.51` tok/s), Prefill 32,768 `-0.40%` (same; `8,676` and `8,656` tok/s), **Prefill 262,144
     `-0.96%`** (±0.70%, repeat `+0.07%`, margin `0.77%`, **worse**; `8,766` and `8,772` tok/s). `c1-decisions` agrees
     on 59 of 59 requests across all 4 boots and 58 of 58 with candidate-54 (`82.41`–`82.92` tok/s).
   - `tp4-switched`: grid `-3.78%` (±0.17%, margin 0.50%, better), c16 paired decode `+1.31%` (±0.40%, margin 0.47%,
     better; `268.02` and `268.56` tok/s), Prefill 32,768 `-0.20%` (same; `8,639` and `8,652` tok/s), Prefill 262,144
     `-0.43%` (same; `8,693` and `8,735` tok/s). `c1-decisions` agrees on 59 of 59 requests across all 4 boots and
     57 of 57 with candidate-54 (`82.36`–`82.66` tok/s).
   - `tp3-switched`: grid (15 shapes) `-2.92%` (±0.12%, margin 0.33%, better), c8 paired decode `+0.66%` (±0.48%,
     margin 0.59%, better; `159.08` and `158.93` tok/s), `single-stream` c1 paired decode `+1.59%` (±0.14%, margin
     0.29%, better; `64.06` and `64.09` tok/s), `sampled` c16 `+1.01%` (same), c8 `+0.43%` (same), c1 `+1.54%`
     (better), `stall` prefill 131,072 tok/s under decode `stall-3` `-1.13%` (`4,833.4` and `4,794.2` tok/s, ±1.37%,
     margin `1.78%`, **same**; p90 `0.9981` and `1.0086` s), `stall-6` `-0.41%` (**same**), `stall-7` `-0.69%`
     (**same**).
5. **Time to ready: met.** `tp4-ring` `115.3`–`115.4 s`; `tp4-switched` `132.1`–`132.1 s`; `tp3-switched` warm
   non-cooled boots `123.8`–`132.7 s` (`184.2 s` on `again-stall` after 15 s cooling).

### Ninth build: vLLM `0809` and `0810` for CUDA 13.5 prefill

Keeps the eighth build's stack (`torch 2.13.0+cu130`, NCCL `2.30.7-1`, CUDA `13.5.37` toolkit, TP3 4 MiB collective
cap) and adds two TileLang prefill fixes (`patch_head` `44db6a403230`, tree `49493238c2ff`, `683` patches total):

- `0809-perf-dsv41-unroll-full-iterations-in-sparse-MLA-stag.patch` (`sparse_mla.py`): splits `issue_stage_records` into
  unpredicated full iterations (`issue_record_copy`) plus a single tail check so `nvcc` under CUDA 13.5 drops the
  per-iteration range branch (`test_deepseek_v41_tilelang.py`: `65 passed, 1 skipped, 1 xfailed` in `candidate-55-b8`;
  standalone `sparse_mla` recovers ~`250`–`330 us` CUDA 13.5 loss at 16,384 rows to within `+0.19%`–`+0.32%` of CUDA
  13.0, `max_abs_diff=0`).
- `0810-perf-dsv41-unroll-MoE-blockscaled-K-loop-across-stag.patch` (`moe.py`): unrolls blockscaled MoE K loop across
  `period = 2 * stages` (`k_step`) and passes `max_ko` so full-K TMA tiles elide `ko < k_full`
  (`test_deepseek_v41_tilelang_moe.py`: `140 passed` in `candidate-55-b8`; `nvcc sm_121a` `0` stack, `0` spills, `191`
  regs; standalone `w13` `E=384` `sorted_seq` `9,753.73 us` vs `10,508.13 us` CUDA 13.0 and `10,776.48 us` CUDA 13.5,
  `E=128` `9,672.40 us` vs `10,451.71 us` CUDA 13.0 and `10,730.82 us` CUDA 13.5, `max_abs_diff=0`).

Specs' runs are named `<run>-b9`. Marks, declared before the build:

1. **Release tests:** 0 failures (target 10,295 passed, 357 skipped).
2. **Every boot reaches ready** at `tp4-ring`, `tp4-switched` and `tp3-switched`, with `sp5` lowest boot `MemAvailable`
   in Builds 5 and 6 (~`6.3`–`7.0 GiB`).
3. **Outputs:** `18d61961`/`d6568649` (text `0a2dacdd`/`12fd8257`) at `tp4-ring` and `tp4-switched`, and
   `5c22ce41`/`bce6092e` (text `26a739a9`/`12cfee9b`) at `tp3-switched` on every boot.
4. **Serving:** `tp4-ring` Prefill 262,144 and `tp3-switched` `stall-3` prefill under decode are not worse than
   candidate-54 beyond the margin; grid, c16, single-stream and sampled gains hold; and `c1-decisions` agrees on
   4 of 4 boots at both TP4 testbeds and with candidate-54.
5. **Time to ready** not longer than the sixth build's warm boots by more than 5 s.

**Ninth build outcomes (2026-11-14):**

Image `spark-ds41f:candidate-55`, ID `c7976f69` (`sp8`, `sp1..sp4`, `sp5..sp7`) and `a8fbd559` (`dgx1..dgx4`), `patch_head`
`44db6a403230`, tree `49493238c2ff`. Release tests receipt: [receipts/release-tests-c7976f69](receipts/release-tests-c7976f69).

1. **Release tests: met.** `10,295 passed, 357 skipped`, 0 failed on `sp8` (`3,060.8 s`; 157 files, `2,868 s` of tests).
2. **Every boot reaches ready: met.** All 28 boots across `tp4-ring` (8: 2 gate, 2 screen, 4 c1-decisions),
   `tp4-switched` (8: 2 gate, 2 screen, 4 c1-decisions), and `tp3-switched` (12: 4 gate, 8 screen) reached ready with
   zero startup memory guard trips and `swap_growth_gib: 0.0` across all screen boots (`dgx1` lowest boot
   `MemAvailable` `23.150–23.158 GiB`, benchmark `min_available_gib` `21.76` and `21.94 GiB`; `sp1` lowest boot
   `MemAvailable` `20.755–20.761 GiB`, cluster min `20.493 GiB` on `sp2`, benchmark `min_available_gib` `19.35` and
   `19.39 GiB` on `sp1`; `sp5` lowest boot `MemAvailable` `6.292–6.558 GiB` across all 8 screen boots, benchmark
   `min_available_gib` `4.51–6.72 GiB`). During `screen-tp3` boot 1 (`candidate-55`), after completing `grid` and `c8`,
   `sp6` halted thermally at `85 C` (`min_headroom_c: 4.0 C`, `fan_curve` service `absent`) on `prefill 32,768` repeat
   1/3 (`6,530 tok/s`); boot 2 (`candidate-55-again`) and boots 3–8 (`single-stream`, `sampled`, `stall-tp3`) all
   completed with `halted: null`.
3. **Outputs: met.** `tp4-ring` (2 of 2 boots) and `tp4-switched` (2 of 2 boots) give `18d61961`/`d6568649` (text
   `0a2dacdd`/`12fd8257`); `tp3-switched` (4 of 4 boots) gives `5c22ce41`/`bce6092e` (text `26a739a9`/`12cfee9b`), with
   0 token or logprob diffs across all scenarios.
4. **Serving: met.** Against candidate-54's recorded screens:
   - `tp4-ring`: grid `-2.70%` (±0.13%, repeat `-0.28%`, floor `0.26%`, margin `0.29%`, **better**), c16 paired decode
     `+0.50%` (±0.43%, repeat `+0.14%`, margin `0.51%`, **same**; `271.12` and `271.75` tok/s; c16 output `+0.58%`
     ±0.78%, **same**), Prefill 32,768 `+1.74%` (±2.09%, repeat `+0.24%`, margin `2.26%`, **same**; `8,841.18` and
     `8,862.57` tok/s), **Prefill 262,144 `+0.37%`** (±0.95%, repeat `+0.03%`, margin `1.01%`, **same**; `8,885.43` and
     `8,887.68` tok/s — Build 8's `-0.96%` regression is resolved). `c1-decisions` agrees on 59 of 59 requests across
     all 4 boots and 58 of 58 with candidate-54 (`82.67`–`83.01` tok/s).
   - `tp4-switched`: grid `-3.99%` (±0.17%, repeat `-0.07%`, floor `0.23%`, margin `0.29%`, **better**), c16 paired
     decode `+1.60%` (±0.34%, repeat `+0.32%`, margin `0.41%`, **better**; `268.82` and `269.82` tok/s; c16 output
     `+1.77%` ±0.74%, **better**), Prefill 32,768 `+1.04%` (±1.64%, repeat `+0.25%`, margin `1.78%`, **same**;
     `8,741.55` and `8,763.72` tok/s), Prefill 262,144 `+0.50%` (±0.71%, repeat `+0.24%`, margin `0.97%`, **same**;
     `8,785.34` and `8,806.33` tok/s). `c1-decisions` agrees on 59 of 59 requests across all 4 boots and 57 of 57 with
     candidate-54 (`82.55`–`82.63` tok/s).
   - `tp3-switched`: grid (15 shapes) `-2.78%` (±0.12%, repeat `-0.38%`, floor `0.30%`, margin `0.33%`, **better**), c8
     paired decode `+0.36%` (±0.36%, repeat `+0.02%`, margin `0.49%`, **same**; `158.65` and `158.71` tok/s; c8 output
     `+0.54%` ±1.27%, **same**); boot 2 standalone `prefill` measured `6,568.41` tok/s (±1.45%) at `32,768` and
     `6,481.75` tok/s (±0.66%) at `262,144`. `single-stream` c1 paired decode `+1.69%` (±0.15%, repeat `+0.03%`, margin
     `0.29%`, **better**; `64.14` and `64.18` tok/s; c1 output `+1.57%` ±4.24%, **same**). `sampled`: c16 paired decode
     `+0.90%` (±1.16%, **same**; `154.94` and `155.91` tok/s), c8 `+0.90%` (±1.81%, **same**; `155.96` and `156.36`
     tok/s), c1 `+1.44%` (±1.22%, margin `1.24%`, **better**; `62.03` and `62.13` tok/s). `stall` prefill 131,072
     tok/s under decode: `stall-3` `-0.35%` (±1.22%, repeat `-0.46%`, margin `1.45%`, **same**; `4,862.82` and `4,840.46`
     tok/s; p90 `0.993`–`0.994` and `0.999`–`1.003` s), `stall-6` `-0.39%` (±2.04%, repeat `-0.17%`, margin `2.42%`,
     **same**; `4,529.18` and `4,521.26` tok/s), `stall-7` `-0.25%` (±2.22%, repeat `-0.13%`, margin `2.30%`, **same**;
     `4,860.48` and `4,854.29` tok/s).
5. **Time to ready: partially met.** `tp4-ring` warm boots (`+0` kernel cache files) took `113.3`–`118.7 s` (`gate`
   boot 2 `116.5 s`, `screen` `113.3` and `116.5 s`, `c1-decisions` `116.5`–`118.7 s`; first boot `194.8 s` compiled
   196 kernel cache files), within `+5 s` of the sixth build (`115.5`–`116.5 s`). `tp3-switched` warm boots took
   `130.6`–`139.2 s` (`gate` warm `130.6`–`136.1 s`, `screen` `131.4`–`139.2 s`; first boot `246.1 s`), within `+5 s`
   of the fifth/sixth build (`127.0`–`135.5 s`). `tp4-switched` warm boots took `135.5`–`143.5 s` (first boot
   `239.4 s`): `gate` boot 2 (`139.1 s`) and `c1-decisions` (`135.5`–`137.5 s`) were within `+5 s` of the sixth build
   (`133.3`–`134.4 s`), while `screen` boots (`139.9` and `143.5 s`) ran `+5.5 s` and `+9.1 s` longer.
