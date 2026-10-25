# Final names: old project names and B12X references after spark-ds41f-1025

On 2026-10-19 the owner asked for karmic-kraken and spark3 to become spark-ds41f in live names,
"moving forward, not historical". spark-ds41f-1025's promotion (branch `spark-ds41f-1025`)
renames the container, image, labels, `/opt` trees, `SPARK3_*` variables, caches and the boot
memory service. This pass finds what is left, renames or deletes the live references, keeps
history, and plans the B12X pieces serving still uses.

- Base: origin/main 8ee6ce53, where the unit suite passes. The branch also applies to
  `spark-ds41f-1025` at 6bd43b8d without conflicts. There the suite shows the same three
  failures with and without this branch: that tip's TP4 benchmark reference,
  `manifests/suites/2026-10-25-spark-ds41f-1025-tp4`, is not written yet (checked 2026-10-25,
  16:45 CET).
- Nothing ran on a GPU or a node: release freeze, and candidate 6's release comparison is on
  the cluster until about 18:50 CET.
- The counts and table below are taken at 1025's tip with this branch applied, outside
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
3. **Upstreams.**
   - `docs/upstreams.md` drew r6's chain and image (`vllm-ds41f-kkref:04c30fa98e79-r6`). It now
     names the sources, points at the lock and the recipe for revisions and image, and gives
     B12X its 1025 role.
   - The lock's `b12x` role says the same.
4. **vLLM** (local branch, patch in [vllm/](vllm/)): the indexer's split check is
   `SPARK_DS41F_INDEXER_SPLIT_CHECK`.
5. **This record:** the inventory, the loader plan, and the loader's lab spec in [loader/](loader/).

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
| `kernel_backend` B12X default and profile | scripts/kernel_backend.py, tests/test_kernel_backend.py, bin/spark (bench identity default), scripts/model_layout.py (default), README (backend policy), docs/doctor.md (backend row), docs/current-state.md ("alternative backend") | conditionals session | It removes `VLLM_DS41_KERNEL_BACKEND` and the B12X paths, and has said it changes test_kernel_backend.py and the recipe's backend setting. |
| B12X tests in `config/release-tests.json`, `scripts/release_tests.py` patterns, docs/testing.md's B12X classes | release tests | conditionals session | They follow the vLLM tree's tests, which that session removes. |
| B12X modules in bin/spark's GPU import smoke (`b12x_roce_all_reduce`, `b12x_mla_sparse`) | bin/spark | conditionals session | The smoke must change in the same change that deletes the modules. |
| **RoCEnante transports**: `rocenante-direct/ring4/mesh4`, `B12X_VLLM_SETTINGS`, `B12X_ROCE_*` | scripts/topology.py; scripts/mesh_fabric.py and mesh_probe.py; bin/spark's RoCEnante checks, mesh4 probe and NIC inventory; scripts/probe_collectives.py's B12X branch; docs/switchless-topology.md, docs/doctor.md; tests (test_topology's B12X base, test_mesh_fabric) | delete after the conditionals branch (follow-up 1) | B12X-only, and dead: doctor already refuses them on every sparknet image. tests/test_kernel_backend.py still resolves its 10-03 twins through them, so they go after that file changes. |
| **B12X tuning**: `bin/spark tuning`, scripts/transport_profiles.py, scripts/model_layout.py | bin/spark, scripts, tests/test_transport_profiles.py, README, TODO.md | delete after the conditionals branch (follow-up 1) | The catalog's two profiles are B12X recipes (`B12X_ROCE_SPIN_LIMIT` required), and test_kernel_backend.py's tuning test reads them. |
| B12X lab fence check, `state/nan_probe.py` | scripts/lab.py, docs/lab.md, tests | **deleted** | B12X kernels only. |
| `overlay/b12x` in kernel-lab bundles, the default `workdir` `b12x` | scripts/lab.py, docs/lab.md, tests/test_lab.py | kept | Recorded bundles rely on the default (`mhc-variants`, `glp-variants`). Goes when B12X leaves the image. |
| B12X in policy and history text | AGENTS.md, CONTRIBUTING.md (B12X trees are still built), docs/recovery.md (an incident), docs/inspiration.md (watchlist), README and TODO.md (r5p/r6 tables), tests/test_bench.py (recorded manifests) | kept | Policy that holds while B12X is built, and history. |
| `## Boot` in TODO.md: "cache B12X routing metadata and prepared state" | TODO.md | kept, flagged | r5h's breakdown (B12X preparation 24 s). No B12X preparation runs since 1025; re-measure the boot before planning from it. |

## Handed to the conditionals session

`local_c364fbfd` removes `VLLM_DS41_KERNEL_BACKEND`, the B12X model and worker paths and config
options, and the B12X test oracles. In spark-ds41f that leaves it the `kernel_backend` rows and
the release-test rows above, plus bin/spark's import smoke. The RoCEnante and tuning deletions
are follow-up 1. They wait for that branch's tests/test_kernel_backend.py, so the two branches
never edit the same file.

## vLLM

- At 26382f5cf the tree has no `spark3`, `kkref` or `karmic` outside Local Inference Lab's own
  files. vLLM 0334 renamed `SPARK3_*` to `SPARK_DS41F_*`.
- That mechanical rename kept one model infix: `SPARK_DS41F_DS41_INDEXER_SPLIT_CHECK`.
- Local branch `final-names` (worktree ~/projects/vllm-final-names, on 26382f5cf) renames it to
  `SPARK_DS41F_INDEXER_SPLIT_CHECK`. It is a validation switch, off unless set. No recipe sets
  it and no test reads it.
- The patch is [vllm/0497](vllm/0497-spark-ds41f-names-the-indexer-s-split-check-is-SPARK.patch),
  numbered after the series in use: candidate 6 (to 0490), the KV capacity session's 0491, the
  drafter window's 0492, and the conditionals session's 0493-0496 (its plan at 16:42 CET;
  renumber if that block grows).
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

### Lab spec ([loader/](loader/))

Queued only; nothing runs during the freeze.

- **Prerequisite:** the vLLM branch `direct-loader` from items 1-2 above, on the release's
  base. Its first commit is the weight digest.
- **Arms:** `loader/make_arms.py ~/projects/vllm-ds41-tl-ports --shape tp3-triangle` (or
  `tp4-ring`) writes the overlay bundles from that branch and three arms over the recipe at
  that shape:
  - `b12x-<tp>`: production's loader, with the digest;
  - `direct-<tp>`: the direct loader, with the digest;
  - `safetensors-<tp>`: vLLM's own loader, with the digest (the evidence arm).

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
  check is the candidate's own gate.

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

## Follow-ups

1. **After the conditionals branch lands:** delete the RoCEnante transports and the B12X tuning
   tooling:
   - topology.py's `rocenante-*` and `B12X_*` handling;
   - mesh_fabric.py, mesh_probe.py and bin/spark's mesh4 checks, probe and NIC inventory;
   - bin/spark's RoCEnante source checks;
   - transport_profiles.py, model_layout.py and `bin/spark tuning`;
   - probe_collectives.py's B12X branch;
   - their tests and docs, and the RoCE marker reference.
2. **The direct loader** (items 1-3 of the recommendation), then the loader spec, then the
   image without B12X (item 4).
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
