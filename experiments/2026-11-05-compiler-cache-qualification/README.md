# Compiler cache identity qualification

**Withdrawn unmeasured.** C42 failed required IKET prepare/freeze compatibility.
The VERSION-only C44 proposal cannot fix that independent retained-feature loss.
Retain validated C40; do not build or launch C44 or extend it into another
algorithmic retry. The source preparation and all original feasibility proofs
are retained as evidence, not as a measured C44 failure.

C44 was prepared as a source-only candidate for one bounded compiler provenance
correction on corrected, default-enabled C40. It keeps the exact C42 ten-patch compiler source
and changes only its VERSION file. The new package version is
`0.1.15+snapshot.bae5fe12b58cdf364efeccbabc04cd08205c43b9`.
The suffix names the complete immutable **pre-metadata** compiler tree, avoiding
a circular version hash. No compiler pass, kernel, arithmetic, model, draft,
topology, cooling or hardware policy is changed.

Purpose is maintenance/cache provenance, declared before any image or timing:
neutral performance is acceptable only with correctness and no material core or
serving regression versus both saved C40 boots. This is not a serving-speedup
claim or an additional algorithmic retry. C41 remains rejected; C43 frontend
lifetime work remains separate and its compiler/base choice remains unfrozen.

## Why this candidate exists

The [focused cache review](../2026-11-05-compiler-live-qualification/runs/preflight/cache-provenance-review.json)
shows that C40 and C42 can share the same0.1.15 cache key/namespace when optional
library stamping is off. Their compatible disk loader checks artifact integrity,
not which compiler produced it. A pure source-method counterexample demonstrates
a possible collision; it does **not** establish that an old binary was loaded or
that the compiler corrupted results. Preparation/freeze is not provenance proof.

C42's native release gate subsequently exited1 before top-k initialization,
fresh compiler-feature processes, functional work or serving timings. The
[separate actual-failure review](../2026-11-05-compiler-live-qualification/runs/preflight/iket-retained-feature-review.json)
verifies its sole native failure and the exact source path: default IKET disables
cache, prepare saves the kernel in the wrapper cache, and the #880 bypass ignores
that prepared object. Compilation then correctly refuses while frozen. The native
test uses its own fixture-local deployment cache; shared old-version reuse is not
the cause of this branch. A no-GPU extracted-method witness returns the prepared
object before #880 but reproduces the native frozen error from both C42 and the
byte-identical C44 JIT source. VERSION is not consulted on that path.
This proposal is therefore **withdrawn unmeasured and must not be launched**. C42's source, image, native
statuses, errors and restoration evidence must remain unchanged. There is no C42
serving point to repeat. Do not retag, rebuild or repeat a completed C42 image
point to manufacture new qualification.

## Exact source

| Input | Identity |
|---|---|
| Base vLLM | C40 head5f926df6d3126266a9c88c82c06544060008c33c, treeca1496c6481dc5bc4b4234b743e4ca023a6c7597 |
| TileLang canonical upstream |2123e412ce2217b324c50c4ec967213906456ac5|
| Reproduced ten-patch source |454202b2ba8d0cf10b8caf434a956755e039483f / bae5fe12b58cdf364efeccbabc04cd08205c43b9|
| New eleven-patch head |d297f60c852dab7dd4fd5a8cfd4ed0d02b071f2f|
| New tree |689108ed654d67ce4ac329fcaed754a104c60f8b|
| Patchset |b4b3b644a48b6b0d92a3666f7dcb8c1a4e0760a12e853a51363fd96d2aeaf238|
| Intended image |spark-ds41f:candidate-44; actual digest UNKNOWN|

Fresh named canonical application reproduces C42 before the metadata patch;
a separate fresh eleven-patch replay reproduces the final head/tree. The source
delta from C42 is exactly one VERSION replacement. All ten authored patches,
messages and notices remain byte-identical. Their29 copied provenance artifacts
and all23 canonical test/static/helper assets verify unchanged. The prepared
source trees are disposable caches; the published series/lock/manifest own identity.

The disabled [candidate profiles](../2026-10-26-spark-ds41f-candidate/candidate-44/)
retain C40 vLLM and the recorded four-image request surface; global67 is untouched.
Expected Docker labels name the new compiler head/tree. The existing Dockerfile
has no compiler-version label; version must be checked against the installed
module/distribution/cache identity, rather than inventing a nonexistent label.

## Version and cache feasibility

The actual version provider with NO_VERSION_LABEL=1 returns the full literal
unchanged. PEP440 and wheel-name parsing accept it. TileLang's CMake project has
no VERSION argument and does not consume the package-version variable in its
CMake/source macros. Two source-derived CPU CMake boundary probes pass. The
[pinned scikit-build-core0.11.6 source](https://github.com/scikit-build/scikit-build-core/blob/v0.11.6/src/scikit_build_core/builder/builder.py)
passes numeric release components separately from the complete local version.
These are source feasibility checks, not a complete compiler/wheel/image build.
Full guarded Linux preparation/check and actual-image installation remain required.

Pure unmodified cache methods produce disjoint keys and namespaces for0.1.15 and
the new pinned version, without importing or executing TileLang/TVM/Torch. The
expected target namespace is
`0.1.15_snapshot_bae5fe12b58cdf364efeccbabc04cd08205c43b9/linux-aarch64`.
This source probe does not prove that any real node namespace is absent or that
an installed image uses the expected version; those remain hard preconditions.

## Preserved first-use preconditions for the withdrawn proposal

Before the **first release gate** on the builder and the **first serving use**
on each configured rank, verify the expected namespace is absent in the actual
host cache root that will be mounted. Preserve all old caches. Do not delete a
preexisting new namespace to pass the check; preserve that setup attempt and
review it. The host absence check must precede importing the compiler with that
mount, because imports may create cache directories. Then require in the actual
image that module version equals distribution version equals pinned version
equals cache base-key version, and that the actual namespace equals the expected
one. Record the exact paths, flags and cache roots.

Fresh feature processes still need separate empty per-boot directories, actual
installed module/wheel/loaded-DSO hashes and no source overlay. A bare library-stamp
flag would not suffice: discovery can differ from an overridden loaded TVM path.
All23 canonical assets remain unchanged; none may be deselected to pass.

These preconditions describe the proposed provenance correction only. They do
not address the independent retained-feature regression and authorize no further
qualification here. No C44 image, native gate or serving arm exists. No runtime
controller, launch or runtime controls were created for this withdrawn packet.

## Acceptance and limits

Report compatible saved1030, C36, previous and current builds with signed changes,
both candidate/reference repeat margins and every adverse row. Keep aggregate
and paired per-user decode distinct, and TPOT separate from actual ITL. Include
cold/mixed prefill, first-chunk stalls, engine cost, acceptance, recorded cache/KV,
preemptions, memory/swap, startup and thermals; identify missing data explicitly.
Never remeasure old images/releases or change a tolerance after the result.

This correction does not reset the compiler's prior algorithmic rework allowance.
If the bounded corrected qualification has an actual remaining correctness or
performance miss, highlight/drop it rather than add another algorithmic retry.
Neutral maintenance is acceptable only without a material regression. No promotion
or bypass of outstanding TP4 gates is implied.

Offline doctors, source/asset checks, normalized renders, Python3.12 parsing and
whitespace pass. The local complete build check exits1 for missing prepared build
inputs; that original warning is retained. Full recursive submodule preparation,
compiler execution, image construction and all native metrics were never run.
The proposal is withdrawn; none are queued under this packet.
