# C45 native feature gate draft

`compiler_gate.py` and `gpu-gate-plan.json` adapt the preserved C42 feature
contract to the corrected C45 compiler. This is source preparation only. The
plan has `dispatch_ready: false`; the host driver refuses dispatch. Nothing in
this packet opens a window, grants hardware access, changes serving, prepares
source, builds an image or implements the separate all-rank observer work.

The current host adapter does not itself check the live shared hold/window/token,
participating-node claims, idle/cold/memory/thermal guards or CPU/GPU locks at the
Docker boundary. These are explicit parent-owned integration blockers. Neither
`--apply` nor an old passed image receipt proves a current lease. Root must couple
dispatch to those live guards, monitor management/guard loss while children run,
stop the owned active gate conservatively when required and preserve its original
native failure. The present blocking `subprocess.run` is not a standalone guarded
runner. Do not enable dispatch until that integration has been reviewed.

The driver currently binds the validated C40 vLLM head/tree and its complete
136-file release manifest. C46 is unmeasured. Root must recheck the best
validated default-enabled base before actual image freeze. Choosing another
validated base requires explicit profile, manifest and control review; the
compiler source packet itself has no frontend dependency.

The compiler refresh remains maintenance and retained-feature preservation.
Correctness and no material incremental core/serving regression versus both
saved C40 boots are required; no speedup is required or claimed. These feature
tests cannot establish a serving-performance verdict.

## What is retained and added

All23 original canonical test/static/helper records, exact native selectors,
prepare/freeze runner and narrowly allowed SM120-only skips are retained.
The host compares those plan fields with the hash-bound preserved C42 plan
before staging. No compiler code is mounted as an executable overlay. Static
source assertions stay text-only; only isolated canonical example helpers are
put on the examples support path after installed compiler identity is checked.

The additional asset is the exact22-control prepared-map test from C45's clean
canonical source. It is hash-bound and run with
`TILELANG_PREPARED_REWORK_SOURCE` naming the verified installed package's parent.
Its methods are extracted from the installed JIT/cache Python files; compiler
outputs and argument binding remain doubles. The original real native IKET,
cache-bypass, retained-feature and fix tests remain mandatory.

Each complete feature process has a new output directory and distinct initially
absent TileLang, CUDA and Torch-extension cache paths. Absence is recorded both
on the host and inside the container before its first mapped compiler import.
An unexpected existing path fails without clearing it. Both containers run
the same passed immutable image, have distinct IDs, and preserve all process
exits, logs, XML, native case names, ownership and cleanup receipts. The draft
container bounds are CPU2, memory32GiB and no additional swap, with MAX_JOBS2;
root must review them with its actual guarded builder/controller before use.

After import, module version, distribution version, immutable pin and actual
cache base key must agree with the full source-derived C45 version. The actual
namespace must be
`0.1.15_snapshot_35f15411cf3af0fc06a1cb0097215b1372a40464/linux-aarch64`.
The driver reads actual selected TileLang/TVM loader paths and this process's
maps, hashes mapped native libraries and checks installed-distribution origin
and the passed builder identity. A library inventory or discovered unrelated
library is insufficient. Module/JIT/cache hashes and loaded-library identity
must remain stable before/after every child and across both fresh processes.
This is a self-process feature check, separate from #924's all-rank observer.

## Pre-feature authority contract

The host accepts no `--image` or tag argument. Root must supply a new immutable
JSON receipt and its exact SHA256 via `--image-receipt` and
`--image-receipt-sha256`. The receipt stays below the new
`results/private/candidate-rounds/candidate-45-image/` output. Its schema is
`c45-compiler-feature-authority/1`, with candidate45, `passed: true` and stage
`release-and-topk-passed-features-pending`. This says the prerequisite phase
passed; it is explicitly not a completed compiler or serving qualification.

Root derives this artifact from the actual first new-image build, full release
gate and strict top-k initcheck outcomes. The gate checks all eight ordered
build/prepare/release/top-k process outcomes, the complete native per-file
audit, actual image/source labels and the exact C45 compiler head/tree/version/
patchset. Additional resource/identity steps may coexist; none of these eight
gate steps may be repeated. No C42/C44 image or verdict can satisfy the contract.

Required receipt fields are enforced in `authority()`:

- `commit`, actual immutable `image`, exact `source` identities;
- the completed eight ordered `steps`, passed `gate`, complete
  `native_gate_audit` with its original results SHA256, and passed strict
  `topk_initcheck` with the existing exact source hash;
- `release_cache` containing the actual host/container roots, full version and
  namespace, and `namespace_absent_before_import: true` recorded before the
  builder's first mapped release-gate import;
- `source_binding`, produced from the clean canonical C45 tree by the small
  `source_binding()` metadata helper. It binds original assets, exact installed
  JIT/cache source hashes, driver/plan bytes, control bytes/size and unchanged authored
  prepare runner;
- `installed_compiler_identity`, a passed actual-image module/cache/selected-DSO
  snapshot from the builder after its prerequisite gates. The draft's
  container-only `--identity-only --binding PATH --output NEW_PATH` helper can
  record this once. It sends no model request and repeats no numerical gate.
  Root owns its guarded invocation and immutable-image association.

The source binding and identity snapshot are metadata evidence. They do not
replace the native gate. The snapshot and feature identities must describe the
same source-derived namespace and actual new-image compiler provider. Full
qualification is recorded separately only after both feature processes and all
other required phases pass. This avoids a circular demand for already-passed
feature evidence before the first feature process can run.

## Remaining checks before any launch

No syntax parse, tests, replay, native import, image build or dispatch has been
executed for this draft. Root must arrange bounded remote source review and
syntax/control checks, including container-path handling, immutable authority
rejection, stale/absent caches, provider/source mismatches, all23 asset coverage,
the added control asset's binding, failure receipts and owned cleanup. The22
prepared-method controls already passed on canonical source; do not repeat that
CPU run here. Their first actual-new-image invocation remains required. Do not flip dispatch readiness
based on local inspection alone.

`test_compiler_gate_controls.py` authors stdlib-only controls of the real helper
functions with mocked processes/files/modules. It covers authority/source/image
rejection, the exact asset contract, fresh/stale cache refusal, installed provider
and selected-library validation, original failure preservation and exact-owned
cleanup. It has not been executed here. It does not rerun the22 prepared-method
controls, import a compiler or qualify a native image. Root may execute it only
in the separately authorized bounded remote CPU area after freezing its source
snapshot; no local fallback is implied.

Root still owns the full guarded builder/controller adaptation, source/profile
doctors and render review, complete recursive Linux prepare/build, final base
choice, clean published execution identity, actual new image, full native
release/top-k outcomes and first mapped cache-use guards. Before first serving
use on every rank, separately require the actual mapped new namespace absent
and preserve every old cache; feature-process absence does not prove serving
cache absence.

After native acceptance, the unchanged complete functional phase, declared C40
sampled semantics, source-bound model comparison with both C40 boots, both full
release/single-stream/sampled metric boots and independent all-rank restoration
remain required. Do not retag, rebuild or remeasure C42/C44 or old images, repeat
an already completed gate/point, alter exactness, or infer performance from these
source-only controls. All historical failure and source-proof files are intact.
