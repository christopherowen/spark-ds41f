# Final names: old project names and B12X references after spark-ds41f-1025

On 2026-10-19 the owner asked for karmic-kraken and spark3 to become spark-ds41f in live names,
"moving forward, not historical". spark-ds41f-1025's promotion (branch `spark-ds41f-1025`)
renames the container, image, labels, `/opt` trees, `SPARK3_*` variables, caches and the boot
memory service. This pass finds what is left, renames or deletes the live references, keeps
history, and plans the B12X pieces serving still uses.

- Base: main 39dd4fc8, with spark-ds41f-1025 promoted (#596) and deployed on the ring and the
  triangle (#597, #603). The branch was first written on 8ee6ce53 and checked against the
  promotion branch's work-in-progress tip (6bd43b8d); rebasing it onto main changed only the
  lab's overlay functions, which #604 moved to `spark.host_path`. The unit suite passes.
- Nothing on this branch ran on a GPU or a node. The direct loader's single-node checks on
  dgx4 are #594's.
- The counts and table below are taken at 39dd4fc8 with this branch applied, outside
  `experiments/`, `manifests/`, `results/`, `patches/` and the 2026-10-01 serving-audit snapshot
  (records, whose names stay as written).

| Pattern | Lines at origin/main | Lines after 1025 and this pass |
| --- | ---: | ---: |
| `karmic-kraken` | 70 | 76 (1025's rollback tests add some) |
| `kkref` | 41 | 15 |
| `spark3` | 112 | 66 |
| `b12x` | 354 | 318 |

Every remaining line belongs to a class below.

## What this branch changes

1. **Names.**
   - `host/recovery/grub-console.cfg` says spark-ds41f manages it.
   - `scripts/host-recovery` drops the `.spark3-migrated` watchdog copies. dgx1 and dgx2's
     copies were removed on 2026-09-22 (docs/recovery.md); dgx3 and dgx4 never had them.
   - The CuTe DSL lockfile's comment names `bin/spark`. Comments are not part of the wheel
     set's identity (`expected_wheel_hashes` skips them).
   - TODO.md: the cost-curve pin and the pinned verification cost table are retired, since
     nothing has read cost curves since vLLM 0184, and the draft budget has been a rule since
     r6x. The draft trace replay (`SPARK3_DSPARK_TRACE`) is gone from the series, so the
     methodology no longer points at it.
   - `experiments/2026-10-17-r6v/rigmark_run.py`, the release notes' RigMark run, names the
     running serving container's image. With `dsv41-karmic-kraken` it would have read the
     stopped 1022 container that the lab keeps.
2. **Lab.**
   - B12X's TMA fence check goes. It compared mounted B12X sources with the image's copy;
     serving launches no B12X kernel and no arm mounts B12X sources.
   - `state/nan_probe.py` goes. It wrapped B12X attention and linear classes in
     `vllm/models/deepseek_v41/nvidia`, a tree that no longer exists.
   - `overlays-<run>.txt` now hashes every file an arm mounts over the image's source trees.
     It matched only sources under `~/spark3-overlay`, a layout no arm has used since r5, so
     current runs recorded no overlay hashes.
   - New arm option **`startup_memory`**: every node's `MemAvailable`, `MemFree` and page cache,
     every 0.25 s from the start command to ready (docs/lab.md). The loader spec below needs it.
     It does the `MemAvailable` part of one-off watchers such as
     `2026-10-23-next-release/memory/watch.sh` inside the lab.
   - New arm key **`serve_options`** (branch `direct-loader`, stacked on this one): the value of
     a flag the base's `serve_args` passes once, so an arm that changes one serving option no
     longer copies the whole list (`set` replaces a list whole). The loader arms use it.
3. **Upstreams.**
   - `docs/upstreams.md` drew r6's chain and image (`vllm-ds41f-kkref:04c30fa98e79-r6`). It now
     names the sources, points at the lock and the recipe for revisions and image, and gives
     B12X its 1025 role.
   - The lock's `b12x` role says the same.
4. **vLLM** (local branch, patch in [vllm/](vllm/)): the indexer's split check is
   `SPARK_DS41F_INDEXER_SPLIT_CHECK`. Branch `direct-loader` adds the weight digest and the
   direct loader (patches in [loader/vllm/](loader/vllm/)).
5. **This record:** the inventory, the loader plan, and the loader's lab spec in [loader/](loader/),
   with the bundles and arms on branch `direct-loader`.

## Every remaining reference class

| Identifier | Where | Action | Why |
| --- | --- | --- | --- |
| `integration/karmic-kraken-beta` | `upstreams.lock.json` (vLLM and B12X `tracking_ref`), README, docs/upstreams.md, docs/current-state.md, docker/Dockerfile and README, docs/inspiration.md | kept | Local Inference Lab's branch name, an upstream's. |
| `vllm-ds41f-kkref:*` image tags | README "Names", docker/README.md, docs/current-state.md (earlier releases), tests/test_recipe.py (1022 arms) | kept | Images keep the tags they were built with. 1022's frozen recipe names one. |
| `dsv41-karmic-kraken`, the `dsv41-` prefix | bin/spark (`LEGACY_SERVING_PREFIX`), scripts/lab.py (`container_name`), docs/lab.md, tests (test_lab, test_names, test_recipe) | kept | 1022 rollback: the guards recognise either container, and the stopped `dsv41-karmic-kraken` containers are kept. Retire with 1022. |
| `dsv41-karmic-kraken` in tests/test_memguard.py | r6m's recorded arm, used as a fixture | kept | The test reads a recorded configuration. |
| `local.spark3.*`, `LEGACY_PREFIX` | scripts/source_labels.py, tests/test_names.py, tests/test_recipe.py | kept | Images before 1025 carry these labels (1022 rollback). |
| `/opt/spark3/candidate` | scripts/lab.py (`SOURCE_ROOTS`), docs/lab.md, TODO.md (overlay bullet), tests/test_lab.py, tests/test_doctor.py (r6's recorded image environment) | kept | Trees in images before 1025. Recorded bundles name them. |
| `~/spark3-hold.json`, `~/spark3-request.json` | scripts/lab.py (`LEGACY_HOLD`, `LEGACY_REQUEST`), README "Names", tests | kept | dgx-spark-networking's AGENTS.md still tells agents to hold the cluster with `~/spark3-hold.json` (follow-up 3). |
| `spark3-vllm-ds41f` (the former GitHub name) | bin/spark repository aliases, tests/test_topology.py, README "Names" | kept | The old URL still redirects. Configurations and node remotes that name it are accepted. |
| `bin/spark3` | README "Names" | kept | Says that experiment records keep the name they ran. |
| `/usr/local/libexec/spark3-roce-marker` | bin/spark (the `rocenante-mesh4` probe), README "Names" | delete with the RoCEnante transports (follow-up 1) | Only B12X's mesh4 probe uses it. The binary on the hosts is a record of that experiment. |
| `spark3-kernel-memory`, `/var/lib/spark3` | host/kernel-memory/README.md | kept | Describes how 1025's installer replaces the old service. 1025's deploy moves the hosts. |
| `SPARK3_*` in docs/current-state.md and docs/lab.md | earlier releases' paragraphs; 1025's own note on cost curves | kept | History, or 1025's text. |
| `SPARK3_*` and `local.spark3.b12x.tree` in tests/test_recipe.py | the 1022 recipe it compares | kept | 1022's recipe is frozen with those names. |
| `SPARK3_DSPARK_COST_DIR` in tests/test_kernel_backend.py | B12X/TileLang twin configurations | conditionals session | That session rewrites this file. |
| `SPARK3_DSPARK_MAIN_PROJ_TP`, `local.spark3.b12x.tree` in tests/test_transport_profiles.py | B12X tuning fixtures | delete with the tuning tooling (follow-up 1) | The fixtures exist only for the B12X tuning catalog. |
| `"build_recipe": "bin/spark3 build (docker/Dockerfile)"` | candidate 6's `source.json` (both copies) and manifests/sources/2026-10-25-spark-ds41f-1025-source.json | kept | Frozen and immutable. Each manifest copies this field from the previous one, so the next source manifest should say `bin/spark build`. |
| `CONTAINER = "dsv41-karmic-kraken"` in experiment scripts | `2026-10-18-scale-codes/mem_snapshot.py`, `2026-10-19-sink-rejection-v2/probes.py` | fixed elsewhere | #586 makes mem_snapshot.py read the running serving container; #582 adds `--container` to both. Not touched here, to avoid conflicts. |
| the same in other experiment scripts | `2026-10-19-sink-rejection-v2/survey.py`, `2026-10-23-next-release/memory/{collect.py,watch.sh}` | kept | Records of the runs that used them. No current spec calls them. |
| `jspark3` | docs/inspiration.md, `upstreams.lock.json` references | kept | Another author's project name. |
| `.lil/changes/vllm-karmic-*`, `jovian-judgement` workflows, `GLM53` | the vLLM tree | kept | Local Inference Lab's files and names, in the base. |
| `SPARK3_*` in patch files | `patches/`, `experiments/*/vllm/*.patch` | kept | Patches apply to their recorded bases. vLLM 0334 renamed the variables in the series. |
| everything in `docs/serving-audit-2026-10-01/` | the audit's source copies | kept | A snapshot; source-index.md records each copy's hash. |
| node state: `cache/kkref -> spark-ds41f`, the stopped `dsv41-karmic-kraken` containers, `experiments/2026-10-23-next-release/1022/cluster.json` | dgx1-4, repository | kept | 1022 rollback, until the owner retires 1022. |
| **B12X, live in serving** | config/cluster.json (`--load-format b12x`, `VLLM_PLUGINS=b12x_loader`, `local.spark-ds41f.b12x.tree`); docker/Dockerfile (source copy, install, import and io_uring checks, labels, `PYTHONPATH`); bin/spark build projects, parts, contexts and image tag; the lock's `b12x` source and the source manifest; validate.yml's B12X series job; docs/setup.md, docker/README.md | kept, plan below | The checkpoint loader. vLLM also imports B12X modules at load time, and B12X's preparation coordinator still runs. |
| `VLLM_ENABLE_PCIE_ALLREDUCE=0` | recipe, `topology.RING_ENV` | kept | It turns the B12X PCIe all-reduce off explicitly; vLLM's default is off too. Both go when the conditionals branch removes the option. |
| `kernel_backend` B12X default and profile | scripts/kernel_backend.py, tests/test_kernel_backend.py, docs/doctor.md (backend row) | conditionals session (#591) | It removes `VLLM_DS41_KERNEL_BACKEND` and the B12X paths. |
| what follows the backend policy | README (backend paragraph), bin/spark (bench identity default), docs/current-state.md ("alternative backend") | kept until #591 lands, then edit | The text describes the policy #591 changes. |
| B12X tests in `config/release-tests.json` | release tests | conditionals session (#591) | They follow the vLLM tree's tests, which that session removes. |
| `b12x` in `scripts/release_tests.py`'s patterns, docs/testing.md's B12X test classes | release tests | kept until the vLLM tree's B12X tests are gone | Both describe test files the tree still has. #591 edits only docs/testing.md's environment contract. |
| B12X modules in bin/spark's GPU import smoke (`b12x_roce_all_reduce`, `b12x_mla_sparse`) | bin/spark (`SMOKE_SCRIPT`, `smoke_script()`) | conditionals session (#591) | Its vLLM 0494 deletes `b12x_roce_all_reduce.py`, so the smoke drops that import for a series with the `tilelang-only` capability. |
| **RoCEnante transports**: `rocenante-direct/ring4/mesh4`, `B12X_VLLM_SETTINGS`, `B12X_ROCE_*` | scripts/topology.py; scripts/mesh_fabric.py and mesh_probe.py; bin/spark's RoCEnante checks, mesh4 probe and NIC inventory; scripts/probe_collectives.py's B12X branch; docs/switchless-topology.md, docs/doctor.md; tests (test_topology's B12X base, test_mesh_fabric) | delete after the conditionals branch (follow-up 1) | B12X-only, and dead: doctor already refuses them on every sparknet image. tests/test_kernel_backend.py still resolves its 10-03 twins through them, so they go after that file changes. |
| **B12X tuning**: `bin/spark tuning`, scripts/transport_profiles.py, scripts/model_layout.py | bin/spark, scripts, tests/test_transport_profiles.py, README, TODO.md | delete after the conditionals branch (follow-up 1) | The catalog's two profiles are B12X recipes (`B12X_ROCE_SPIN_LIMIT` required), and test_kernel_backend.py's tuning test reads them. |
| B12X lab fence check, `state/nan_probe.py` | scripts/lab.py, docs/lab.md, tests | **deleted** | B12X kernels only. |
| `overlay/b12x` in kernel-lab bundles, the default `workdir` `b12x` | scripts/lab.py, docs/lab.md, tests/test_lab.py | kept | Recorded bundles rely on the default (`mhc-variants`, `glp-variants`). Goes when B12X leaves the image. |
| B12X in policy and history text | AGENTS.md, CONTRIBUTING.md (B12X trees are still built), docs/recovery.md (an incident), docs/inspiration.md (watchlist), README and TODO.md (r5p/r6 tables), tests/test_bench.py (recorded manifests) | kept | Policy that holds while B12X is built, and history. |
| `## Boot` in TODO.md: "cache B12X routing metadata and prepared state" | TODO.md | kept, flagged | r5h's breakdown (B12X preparation 24 s). No B12X preparation runs since 1025; re-measure the boot before planning from it. |

## Handed to the conditionals session

`local_c364fbfd` removes `VLLM_DS41_KERNEL_BACKEND`, the B12X model and worker paths and config
options, and the B12X test oracles (vLLM 0493-0496). It leaves the loader alone. Its spark-ds41f
change is draft PR #591 (branch `retire-conditionals`):
- scripts/kernel_backend.py, tests/test_kernel_backend.py, config/release-tests.json;
- docs/doctor.md's Kernel backend row and docs/testing.md's environment contract;
- bin/spark's import smoke.

The RoCEnante and tuning deletions are follow-up 1, built on #591's tests/test_kernel_backend.py
so the two branches never edit the same file. On #591's series vLLM refuses
`VLLM_ENABLE_ROCE_ALLREDUCE=1` at startup, so the `rocenante-*` transports cannot start at all.

## vLLM

- At 26382f5cf the tree has no `spark3`, `kkref` or `karmic` outside Local Inference Lab's own
  files. vLLM 0334 renamed `SPARK3_*` to `SPARK_DS41F_*`.
- That mechanical rename kept one model infix: `SPARK_DS41F_DS41_INDEXER_SPLIT_CHECK`.
- Local branch `final-names` (worktree ~/projects/vllm-final-names, on 26382f5cf) renames it to
  `SPARK_DS41F_INDEXER_SPLIT_CHECK`. It is a validation switch, off unless set. No recipe sets
  it and no test reads it.
- The patch is [vllm/0497](vllm/0497-spark-ds41f-names-the-indexer-s-split-check-is-SPARK.patch),
  numbered after the series in use: candidate 6 (to 0490), the KV capacity session's 0491, the
  drafter window's 0492, and the conditionals session's 0493-0496 (confirmed by that session).
- [vllm/series](vllm/series) is candidate 6's series with 0497. The patch applies on candidate
  6's tree (6f8e062f), and also on the conditionals session's working copy of `attention.py`.
- The branch is not pushed. Pushing it would publish the frozen `tilelang-ports` history, as
  for the other research branches.

## B12X still live in serving: the plan

### What serving takes from B12X

1. **The checkpoint loader.** The recipe passes `--load-format b12x` and `VLLM_PLUGINS=b12x_loader`.
   - The plugin is B12X's entry point `b12x.integration.vllm.loader:register_b12x_loader`. It
     registers the B12X package's own `B12xModelLoader` before vLLM's `get_model_loader` would
     lazily import the in-tree copy (`vllm/model_executor/model_loader/b12x_loader.py`).
   - **So production loads with the B12X package's copy, not vLLM's.** The two differ only
     outside the path GB10 takes (vLLM's adds IPC-exportable allocations for GDS). This is read
     from the code; a serving log's `Registered model loader` line names the class.
2. **Modules vLLM imports at load time.** Examples: `vllm/models/deepseek_v4_1/attention.py`,
   `compressor.py`, `common/engram.py` and `nvidia/dspark.py` import `b12x.attention`,
   `b12x.gemm`, `b12x.norm`, `b12x.sequence` and `b12x.preparation`. B12X's preparation
   coordinator still runs under TileLang, without B12X units. The conditionals session is
   removing these.
3. **The PCIe all-reduce backend.** It is off (`VLLM_ENABLE_PCIE_ALLREDUCE=0`, and off by
   default). It goes with the conditionals branch's B12X config options.

B12X can leave the image only when all three are gone.

### What the loader does (B12X 236ddff0, bounce mode)

- **Metadata only.** The weights iterator yields meta tensors that carry each tensor's
  checkpoint range. vLLM's weight loaders route copies into parameter views through vLLM's
  `weight_transfer` hook (`copy_weight`). The session records each copy:
  `(fd, offset, bytes, destination, strides)`.
- **Supported transforms.** Plain copies, strided TP slices (2-D copies) and BF16 to FP32
  expansion are recorded. Anything else fails closed. Scalars and declared metadata are read
  with O_DIRECT into CPU tensors. A copy of one that the session cannot record goes to Torch:
  "the ones it declines" in [r6o](../2026-10-16-r6o/README.md).
- **Reads.** `read_mode` `auto` selects `bounce`, because GB10 reports pageable memory access
  through host page tables (it selects GDS otherwise).
  - Records execute through io_uring O_DIRECT reads into a 16-slot ring of 1 MiB slots
    (16 MiB, `malloc` plus `cudaHostRegister`, freed when the session closes).
  - A copier thread runs `cudaMemcpy2DAsync` into the destinations, on the copy engine.
  - No kernels run ([2026-10-24 completion](../2026-10-24-tilelang-completion/README.md), run 22).
  - The BF16 to FP32 expansion is done by the CPU in the ring; other dtype transforms use an
    8 MiB GPU scratch.
  - **No payload is memory-mapped or read through the page cache.**
- **Engram tables are never read by the loader.**
  - The model's `checkpoint_file_weight_filter` marks them.
  - The loader yields vLLM's file-source descriptors for them (`weight_utils.file_source_tensor`).
  - Serving reads their rows with vLLM's own io_uring reader
    (`vllm/models/deepseek_v4_1/engram_rows/_reader.c`).
- **The native helper.** It is about 750 lines of C (`_storage.c`, `_direct.c`, `_batch.c`,
  `_bounce.c`). `cc` builds it at first use into `$XDG_CACHE_HOME/b12x/loader/<digest>`. The
  image's build check builds it into root's cache, but serving's `XDG_CACHE_HOME` is the
  mounted `/cache/spark-ds41f/jit`, so each node compiles it once per image.
- **Measured (r5h, TODO.md "Boot").** Shard routing 7 s, then the read at NVMe rate: 101 GB in
  15 s per rank.

### Options

The risks named in the brief, bounded from the repository's evidence:

| Option | Startup time | Startup memory against dgx1's 5 GiB guard | R580 64K-page copy hang | Engram file-backed tables | Verdict |
| --- | --- | --- | --- | --- | --- |
| **B12X (today)** | 7 s routing plus 15 s read per rank | 16 MiB registered ring plus 8 MiB scratch, freed. No page cache. | not exposed: no pageable file-backed source | descriptors through vLLM's hooks | works; its code lives in an archived dependency |
| **vLLM `safetensors`** (the default loader) | unmeasured on GB10; page-fault reads of each rank's ranges | Every byte read stays in the page cache. `MemAvailable` counts the page cache, so the guard does not see it, but the GB10 allocator needs `MemFree` (TODO.md "Boot", pitfalls). The KV pool and graphs allocate after the load, where dgx1's margin at TP3 is boot-to-boot noise: one boot on 2026-10-21 fell 12.7 MiB short ([1022 record](../2026-10-22-spark-ds41f-1022/README.md), "Not covered"). `copy_weight` adds an anonymous copy of each source (transient). | exposed; avoided only because `copy_weight` copies every pageable source into anonymous memory first (r6o, vLLM edb34cd27). A weight loader that bypassed it would hang. | supported (`file_backed_safetensors_weights_iterator`); non-Engram tensors in mixed files read by `get_tensor` (mmap) | reject unless the evidence arm shows no page-cache cost |
| **vLLM `instanttensor` / `fastsafetensors`** | fast whole-file reads | Each shard is staged whole in a GPU buffer, even with filtering (`weight_utils`: "reserve a full-file CUDA buffer"). The largest ordinary shard is 7.4 GB (2026-10-02 defragmentation record). | not exposed (GPU staging) | supported | reject: a new dependency the image lacks (`vllm[instanttensor]`), whose pins must move forward with the shared packages, plus a transient of up to a shard per rank |
| **vLLM `mxfp4_csf`** | n/a | n/a | n/a | keeps file-backed Engram | reject: needs a recompressed CSF checkpoint, not DeepSeek's, and imports B12X (`b12x.moe.fused_moe`) |
| **TileLang / TileKernels path** | n/a | n/a | n/a | n/a | nothing to port: the bounce path launches no kernel, and TileKernels has no I/O. A TileLang BF16-to-FP32 expansion would only move a bit-exact transform. |
| **Our own loader**: B12X's bounce path moved into vLLM | the same reads and copies | the same 16 MiB ring and 8 MiB scratch; no page cache | not exposed | descriptors through vLLM's hooks, as today | **recommended** |

### Recommendation

Move B12X's bounce-mode loader into our vLLM tree as `--load-format direct`. The reads,
copies and transforms stay identical, so the change is who owns the code, not what the node
does. Concretely, as vLLM patches on the next release's base:

1. **The weight digest.** A debug check, off in production: with `SPARK_DS41F_WEIGHT_DIGEST=1`
   each rank logs `weight digest rank <r> <sha256>` over every parameter, in name order, after
   loading. This is the bit-identity gate. The arms of both loaders carry it.
2. **The direct loader.**
   - `vllm/model_executor/model_loader/direct_loader.py`: the in-tree `B12xModelLoader`
     without GDS, the shared read group, the IPC allocations and the progress panel. It logs
     one summary line.
   - `vllm/model_executor/model_loader/direct_io/`: `DirectWeightSession`'s bounce path and
     the C helper's bounce, batch and direct readers. Drop the GDS, PLE and shared-plan files.
     B12X is Apache-2.0, so keep its notices.
   - Register `direct` in `LoadFormats`, and fail with a clear error when the device reports
     no host page tables, instead of falling back to GDS (no fallback paths).
   - The helper builds as it does today: at first use, into the mounted cache, once per node
     and image. One path serves both the lab screen (the C source mounted) and production.
3. **The recipe** passes `--load-format direct` and sets `VLLM_PLUGINS` to an empty value.
   Unset, it would load every installed general plugin, B12X's `b12x_fp6` among them, while
   B12X is installed.
4. **Then B12X leaves the image**, once the conditionals branch has removed the imports and
   the preparation coordinator:
   - the source copy, install, build checks, `PYTHONPATH` entry and `local.spark-ds41f.b12x.*`
     labels in docker/Dockerfile;
   - `b12x` in bin/spark's build projects, parts and contexts and in the image tag;
   - the lock's `b12x` source, validate.yml's series job and the expected label;
   - the lab's B12X bundle support, and the "keep B12X" text in AGENTS.md and CONTRIBUTING.md.

   `liburing` stays: vLLM's Engram reader uses it. That is a new image, so it is a release
   candidate of its own.

### The direct loader: implemented, checked on one node (2026-10-25, CET)

Items 1 and 2 are the vLLM branch `direct-loader` on 26382f5cf (worktree
~/projects/vllm-direct-loader; local, not pushed, like the other research branches). The
patches and a series are in [loader/vllm/](loader/vllm/): candidate 6's series with
[0498](loader/vllm/0498-Model-loading-a-weight-digest-on-every-rank-SPARK_DS.patch) and
[0499](loader/vllm/0499-Model-loading-load-format-direct-B12X-s-bounce-mode-.patch). Both apply on
candidate 6's tree (6f8e062f) and on each branch of 0491-0497 (KV capacity, drafter window,
conditionals, final names).

- **0498, the weight digest** (9b04f1797). `BaseModelLoader.load_model` hashes every
  parameter's bytes in name order (a shared parameter once) right after `load_weights`,
  before post-load processing turns them into kernel layouts, so it covers the loader's
  bytes and every loader's arm carries it (B12X's plugin class runs the same
  `load_model`). Each rank logs
  `weight digest rank <r> <sha256> <model class>: <n> parameters, <GB> GB in <s> s`; the
  DSpark drafter loads through the same loader and logs a second line. The bytes pass
  through one 64 MiB host buffer and the CPU's SHA-256 adds some tens of seconds per rank,
  which "Loading weights took" excludes and the API-ready time of every arm includes. Unset,
  it reads the variable once at import.
- **0499, `--load-format direct`** (6d5881d9c):
  - `vllm/model_executor/model_loader/direct_loader.py` is the in-tree `B12xModelLoader`
    without GDS, the shared read group, the IPC allocations and the progress panel. It
    refuses a device without host page tables before opening a session and logs one line,
    `Direct loader: weights loaded, <GB> GB of checkpoint payload in <s> s ...`, which
    loader_report.py matches.
  - `direct_io/` holds `DirectWeightSession`'s bounce path (`_checkpoint.py`), the build
    (`_native.py`) and B12X's `_storage.c`, `_direct.c`, `_batch.c`, `_bounce.c` and
    `_cuda_range.h`. The C differs from B12X only in the module and capsule names, the PLE
    reader's removal and the `#ifdef`s without liburing: liburing is required, so the build
    refuses to run without it. Every ported file names its B12X source and what changed.
  - The session's descriptor planning moved into functions (`view_offset`, `view_records`,
    `scratch_chunks`) so it can be tested without a GPU. The records they produce are
    B12X's.
  - The helper builds at first use into `$XDG_CACHE_HOME/vllm/direct_io/<digest>`
    (`/cache/spark-ds41f/jit/vllm/direct_io` in serving), as the Engram reader does.
- **Tested** on the laptop (no GPU: macOS, torch 2.14.1, the tree from source, pytest
  `--noconftest`): 55 tests pass. They cover the digest, and the descriptors for TP shards,
  padded and transposed destinations, expert slices, byte casts, E8M0 views, scalars and
  BF16 expansion, executed on host memory with `_batch.c`'s semantics and compared with
  Torch's copies. They also cover the scratch reads, refused views, header checks, selection
  and coalesced metadata reads, the loader's registration, options and refusal, the Engram
  file-backed descriptors in mixed and table-only shards, and the helper's build into the
  XDG cache. The rest of `tests/model_executor/model_loader/` still passes (168; the tests
  that need the conftest's fixtures do not run without it).
- **GPU tests:** the 38 tests in `test_direct_loader_cuda.py` are B12X's `test_direct.py`
  for the bounce ring, the helper's capabilities, a loader-routed module whose digest equals
  a Torch copy's, and file-backed tables beside routed weights. They write under
  `DIRECT_IO_TEST_DIR` (default `/cache/direct-io-test`), because O_DIRECT needs a real
  filesystem. They ran on dgx4 (below). Never run them beside a serving container.

### Single-node checks on dgx4 (2026-10-25, CET)

The owner opened dgx4 (uncabled, serving nothing) to sub-agents' GPU tests on 2026-10-25.
dgx1-dgx3 were not touched. Both checks are kernel-lab bundles (docs/lab.md,
`kernel-local`), written by [make_kernel_bundles.py](loader/make_kernel_bundles.py):
- they mount bundles/direct (vLLM 6d5881d9c, 0498-0499) over candidate 6's image,
  `spark-ds41f:19f2c20ed4d6-next-6` (`sha256:82b071b57485`);
- they ran from this branch at e3def993, checked out in `~/spark-lab/agents/direct-loader/spark-ds41f`;
- the command was `scripts/lab.py kernel-local experiments/2026-10-25-final-names/loader/kernel/<bundle> --out ~/spark-lab/agents/direct-loader/out/<bundle>`;
- their output.txt and verdict.json are in [loader/runs/2026-10-25-dgx4/](loader/runs/2026-10-25-dgx4/).

| Check | Bundle | Finished | Verdict |
| --- | --- | --- | --- |
| The branch's loader tests: test_direct_loader_cuda.py, test_direct_loader.py, test_weight_digest.py | [kernel/gpu-tests](loader/kernel/gpu-tests/) | 20:02 (12 s, after 52 min queued for the GPU lock) | **pass**: 93 passed, 0 skipped. The 14 warnings are Torch's `torch.jit` deprecation. |
| One TP4 rank's share of DeepSeek-V4.1-Flash-1017, the direct loader against B12X's | [kernel/rank-load](loader/kernel/rank-load/) | 20:15 (149 s) | **pass**: equal digests, tensors, descriptors and samples |

The helper built at first use, as serving would, into the bundle's XDG cache
(`~/.cache/spark-lab/compile/xdg-direct-loader/vllm/direct_io/ca191a8547dad4fe7b46d9b9`).

**rank-load** ([rank_load.py](loader/kernel/rank-load/rank_load.py),
[rank_compare.py](loader/kernel/rank-load/rank_compare.py)) loads a stand-in for a TP4
rank's shard set, not the model's own shard set: the model cannot be built without every
rank.
- **The plan:** rank 0's quarter of every 2-D tensor whose dimensions both divide, by rows
  or by columns (every other name), and the rest whole.
  - BF16 vectors are expanded to FP32 and FP32 tensors converted to BF16.
  - The Engram tables become descriptors.
- **The path:** each loader's own weights iterator (`DirectModelLoader`, and B12X's
  `B12xModelLoader` with its bounce-mode session) and `copy_weight`, as weight loaders call
  it. Each load runs in a process of its own.
- **Outputs:** per-tensor SHA-256 lists on dgx4,
  `~/spark-lab/agents/direct-loader/out/rank-load/tensors-{direct,b12x}-tp4-rank0.json`.

| | direct | B12X |
| --- | ---: | ---: |
| tensors loaded / Engram descriptors | 96,081 / 4 | 96,081 / 4 |
| loaded (GPU) | 71.585 GiB | 71.585 GiB |
| weight digest | `a4893a5e…290c6fd` | `a4893a5e…290c6fd` |
| tensors whose SHA-256 differ | 0 | |
| sample checked against safetensors' reads | 89/89 equal | 89/89 equal |
| load, routing 96,081 tensors included | 23.71 s | 23.89 s |
| ring execution | 17.38 s | 17.21 s |
| payload, physical reads | 76.88 GB, 191.86 GB | 76.88 GB, 191.86 GB |
| strided copies, BF16 expansion, GPU scratch | 38.06 GB, 1.5 MB, 42.4 MB | the same |
| MemAvailable before, lowest | 116.10, 43.82 GiB | 117.04, 44.48 GiB |
| loader overhead (MemAvailable drop beyond Torch's 71.60 GiB reserved) | 0.68 GiB | 0.96 GiB |
| lowest MemFree | 17.61 GiB | 18.07 GiB |
| page cache before, highest, after | 25.730, 25.732, 25.732 GiB | 25.924, 25.928, 25.927 GiB |
| weight digest time | 33.6 s | 33.5 s |

Reading the table:
- **The same reads:** reads, submissions within 1%, batches and descriptors are equal.
  The direct loader is B12X's loader in bits, reads and speed.
- **No page cache:** the page cache grows by under 5 MiB while 77 GB is read.
- **Memory:** the overhead counts this script's Python objects for 96,081 tensors as well as
  the loader's 16 MiB ring and 8 MiB scratch. The direct loader's is the smaller of the two,
  so it adds nothing B12X does not to the 5 GiB startup guard on dgx1-dgx3.
- **The digest's cost:** about 34 s per TP4 rank on the CPU, the time `SPARK_DS41F_WEIGHT_DIGEST=1`
  adds to every boot of the loader spec (all arms alike, after "Loading weights took").
- **Read amplification:** physical reads are 2.5 times the payload in both loaders. When a
  strided shard's row is narrower than 4 KiB, B12X's ring packs rows into one read that
  spans the full stride, so a column shard of a narrow-row tensor reads the other ranks'
  columns too (`_bounce.c`, slot rows). Half of this plan's 2-D tensors are column shards.
  Production's ratio depends on the model's own column shards; the loader spec's bytes-read
  figure measures it. This is not something the port changes: it is the same bytes in both.


### Lab spec ([loader/](loader/))

Queued only; nothing runs during the freeze.

- **Prerequisite:** the vLLM branch `direct-loader` above. Its first commit is the weight
  digest.
- **Arms:** `loader/make_arms.py ~/projects/vllm-ds41-tl-ports --shape tp3-triangle` (and
  `tp4-ring`) writes the overlay bundles from that branch and three arms over the recipe at
  that shape. The bundles and both shapes' arms are committed (from 6d5881d9c):
  - `b12x-<tp>`: B12X's loader (`--load-format b12x`, `VLLM_PLUGINS=b12x_loader`), with the
    digest;
  - `direct-<tp>`: the direct loader, with the digest;
  - `safetensors-<tp>`: vLLM's own loader, with the digest (the evidence arm).

  Each arm names its load format in the new overlay key `serve_options` (docs/lab.md), so it
  follows every other serving argument of the recipe at run time: a copy of `serve_args`
  would have kept main's 1022 arguments after spark-ds41f-1025's promotion
  (`--enable-prompt-tokens-details` and `--jit-monitor-mode off` differ). Over both
  recipes, main's and 1025's, each arm differs from its shape only by the load format, the
  digest variable, `VLLM_PLUGINS` and its bundle mounts. The bundles mount over
  `/opt/spark-ds41f/candidate/vllm`, so the arms need 1025's image or a later one.

  The overlays are Python plus the C source, so the screen needs no image build.
- **[loader-tp3.json](loader/loader-tp3.json)** (or loader-tp4.json): six boots, alternating
  direct and B12X, direct first (heat soak counts against later arms). Each arm has
  `startup_memory` on. Per boot it runs, in order:
  - [loader_report.py](loader/loader_report.py): per node, the weight-loading lines and their
    seconds after the container started, the API-ready time, the per-rank digests, the bytes
    the container's cgroup read from disk, and `MemFree`/`Cached` after boot;
  - the temperature-0 determinism texts, with log probabilities and without (greedy);
  - one- and eight-stream distinct prompts, 3 samples each.

  Boots plus measurement take about 6 × 6 = 36 minutes.
- **[loader-evidence-tp3.json](loader/loader-evidence-tp3.json):** `safetensors` then B12X, one
  boot each. Run it only to put numbers on the rejected option. It can fail its boot on the
  page cache; the guard and the rollback are the protection, and failing the boot is the
  finding.
- **[tables_arms.py](loader/tables_arms.py)** prints per boot: load seconds, API seconds, bytes
  read, the lowest `MemAvailable` and `MemFree`, the largest page cache during the boot, the
  page cache after it, and the speeds. Per loader, it gives the means over its boots.
- **Gates, all required:**
  1. the weight digests are equal on every rank across every boot;
  2. the temperature-0 digests are equal (with and without log probabilities);
  3. every node stays above the 5 GiB startup guard.

  Read beside them, against the B12X boots of the same session:
  - direct's slowest-rank load time is within 1 s of B12X's;
  - dgx1's lowest `MemAvailable` is within the B12X boots' own spread;
  - the page cache after boot is no larger;
  - the bytes read are within 1%;
  - c1 and c8 are within the boots' spread.
- **Queue it** after the freeze, with the cluster cabled as the spec's shape, behind the
  release work. Then the direct loader goes into the next candidate's series. Its serving
  check is the candidate's own gate. All four specs dry-run clean on branch `direct-loader`
  (2026-10-25, 17:35 CET).
- **If gate 1 fails:** the digest is one hash per rank, and it also covers parameter bytes no
  loader writes (padding a weight loader leaves unfilled). Find the parameter with a
  per-parameter listing before suspecting the reads.

## GPU verification (queued, not run)

- **spark-ds41f:** nothing on this branch reaches serving. The changes are docs, host-file
  comments, lab tooling and a RigMark metadata field. The lab changes run with the next window:
  - the first run's `overlays-<run>.txt` should list every mounted file;
  - an arm with `startup_memory` should write its summary.
- **Hosts:** run `scripts/host-recovery apply` on dgx1-4 in the next window that may touch the
  hosts. It installs the renamed grub drop-in comment. `check` reports the file as different
  until then, and nothing else changes.
- **vLLM:** the variable rename changes nothing while the variable is unset. It goes into the
  next candidate's series, and that candidate's gate (temperature-0 digests equal to 1025's)
  covers it.
- **The direct loader:** its GPU tests and a one-rank load against B12X's passed on dgx4
  (above). The loader spec needs the cluster.

## Follow-ups

1. **After the conditionals branch lands:** delete the RoCEnante transports and the B12X tuning
   tooling:
   - topology.py's `rocenante-*` and `B12X_*` handling;
   - mesh_fabric.py, mesh_probe.py and bin/spark's mesh4 checks, probe and NIC inventory;
   - bin/spark's RoCEnante source checks;
   - transport_profiles.py, model_layout.py and `bin/spark tuning`;
   - probe_collectives.py's B12X branch;
   - their tests and docs, and the RoCE marker reference.
2. **The direct loader:** items 1 and 2 are implemented (vLLM 0498-0499), and its GPU tests
   and a one-rank load equal to B12X's passed on dgx4. Next the loader spec (the cluster),
   then the recipe (item 3), then the image without B12X (item 4).
3. **dgx-spark-networking's AGENTS.md** still says `~/spark3-hold.json`. Once it says
   `~/spark-hold.json` (it is a public repository: the owner's call), lab.py's legacy hold and
   request names can go.
4. **After the owner retires 1022:**
   - drop the `dsv41-` and `local.spark3.` compatibility, `/opt/spark3/candidate`, and the
     rollback tests;
   - remove `cache/kkref` and the stopped `dsv41-karmic-kraken` containers on the nodes.
5. **The next source manifest** (`source.json`) should say `bin/spark build` in
   `build_recipe`.
6. **#574 (agentx-traffic)** mocks `lab.fence_problems` in a new test. Whichever of #574 and this
   branch lands second drops that one line.
