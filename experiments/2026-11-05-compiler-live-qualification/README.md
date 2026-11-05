# Compiler interaction qualification

C42 qualifies the TileLang compiler refresh from #875 together with the existing
authored fixes #879–#883 on the corrected, default-enabled C40 base. This is an
explicit **interaction qualification of one runtime dependency, TileLang**.
Its purpose is maintenance and retained-feature preservation. No speedup is
required or claimed; correctness and no material incremental core/serving
regression versus **both saved C40 boots** are required before acceptance.

This packet does not implement another optimization, merge those PRs, change
production, or qualify their combination by source inspection. C41 remains
rejected. C40's vLLM head/tree, kernels, copy/draft policy, model/weights,
topology and hardware policy remain unchanged. C42 explicitly retains the
recorded four-image-per-request surface; the global default of 67 is untouched.

## Exact source packet

Deployment preparation base: `3c7b455eec27c436f76f08ed28efb30135c86f4f`.
The [candidate-42 lock and manifest](../2026-10-26-spark-ds41f-candidate/candidate-42/)
retain every non-TileLang C40 source entry unchanged.

| Input | Identity |
| --- | --- |
| TileLang upstream | `2123e412ce2217b324c50c4ec967213906456ac5` |
| Combined patch head | `454202b2ba8d0cf10b8caf434a956755e039483f` |
| Combined tree | `bae5fe12b58cdf364efeccbabc04cd08205c43b9` |
| Ten-patch fingerprint | `8fb334c1144bb1b15f3709c50ef427f8d6248e9175f211b4d768a9e837c51465` |
| Unchanged C40 vLLM head | `5f926df6d3126266a9c88c82c06544060008c33c` |
| Unchanged C40 vLLM tree | `ca1496c6481dc5bc4b4234b743e4ca023a6c7597` |
| Proposed image tag | `spark-ds41f:candidate-42` |
| Actual image digest | Unknown until independently built and gated |

[tilelang/series](tilelang/series) first carries the five authored #875 patches,
then the existing fixes in PR order 879 through 883. Every patch is copied
byte-for-byte, preserving its message, author, license notices and recorded
quality/test statements. Renumbering destination filenames does not change
patch bytes. [Provenance](runs/preflight/authored-provenance.json) binds each
copy to the fetched PR head, original path and SHA256. No fix was silently
omitted, rewritten or replaced by a fresh implementation.

Fresh canonical application on named disposable branch
`feat/compiler-interaction-qualification` reproduced the five-patch parent
head `b26f48b85790b832993f2c9d240dc3c39d6075e8` and tree
`971d99a61b44a702110cf98bef56ee1a44ddeaf1`, then applied all five fixes without
conflict or manual source edit. The combined source is clean and passes
`git diff --check`. [Original commands and output](runs/preflight/canonical-replay.json)
are retained. The ignored prepared tree is a disposable cache; the published
patch series, lock and manifest are the source authority.

## Focused review of the authored fixes

These are **authored standalone observations**, preserved in the copied
`runs/preflight/authored-*-results.json` files. They were not independently
repeated here and do not establish combined-stack or serving qualification.

| PR | Selected change | Authored evidence and boundary |
| --- | --- | --- |
| #875 | Main refresh, retained MX/prepare/freeze/IKET features and MX scale-byte selector repair | Five-patch preparation alone establishes reproducibility. Exact varying-scale tests must still run on the new image. |
| #879 | Remove unsafe async-copy wait relaxation | Parent fails three transform cases twice; isolated fix passes twice. This is a compiler-transform failure, not observed serving corruption. |
| #880 | Bypass both JIT wrapper caches when cache is disabled | Parent fails the fresh-wrapper-kernel assertion twice; isolated fix passes five cases and normal prepare/freeze compatibility twice. Original missing-import attempt remains in its authored results. |
| #881 | Publish synchronous producer stores before GEMM | Parent numerical reproducer fails one process and passes the other; isolated fix passes 200 poisoned-output launches in both. Its specialized microkernel is slower than specialization disabled; retain the C40 workaround. |
| #882 | Fence generic-proxy consumer reads before stage release | Parent fails both generated-code assertions; its sampled numerical test also passes. Isolated fix passes assertions and the portable numerical check twice. Do not invoke SM90-only WGMMA cases on SM121. |
| #883 | Infer shared-memory alignment from every use | Parent fails six alignment assertions while sampled copies remain correct; isolated fix passes 19 checks twice. No serving-speedup claim follows. |

The two warp-specialization fixes modify different parts of the same pass:
producer publication and consumer release ordering. Both survive in the
combined file. The wait-count repair acts in pipeline injection, while the
alignment repair changes arena requirements and TMA annotations; those changes
can interact with generated code and must be tested together. Clean patch
application is not proof that those interactions are numerically safe.

The cache-bypass repair gates cache reads and writes in the wrapper while
retaining prepare and the cache's named frozen-kernel refusal. The cache module,
MX instruction/selectors and IKET implementation are byte-identical to the
five-patch authored parent. No runtime warp-specialization workaround is
removed and no compiler feature or accuracy threshold is weakened.

## Local validation and limits

Promoted, TP3, TP4 and build-only profile doctors pass. TP3/TP4/build-profile
renders match C40 except the declared compiler/image/source identity and the
explicit recorded image limit of four. All 26 changed compiler Python files
and both copied test harnesses parse as Python 3.12. Authored-copy hashes,
manifest fingerprint, exact source head/tree, clean status and final source
whitespace pass [local checks](runs/preflight/offline-validation.json).

Immediate CUTLASS/TVM gitlinks match the authored manifest. The fresh tree has
not populated or independently checked all recursive submodules. Their exact
authored pins are retained and must be checked by the guarded Linux preparation.
Local `build check` exits **1** with the original incomplete-input warning,
retained in [builder checks](runs/preflight/offline-builder-validation.json).
No full build check, native compiler execution, image build, GPU gate, serving
measurement, node operation or old-image remeasurement is claimed here.
Read-only inspection errors and all preparation history are retained.

Reproduce source-only checks:

```sh
python3.12 -B experiments/2026-11-05-compiler-live-qualification/check_sources.py
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-42/tp3.json doctor
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-42/tp4.json doctor
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-42/image-dgx4.json doctor
```

The owned guarded builder must perform the normal full preparation and build
check before creating the sole image. After a fresh recursive checkout,
`check_sources.py --source PATH --require-submodules` verifies all recorded pins.
Only the coordinator may schedule hardware work and subsequent restoration.

## Required independent image gates

[gpu-gate-plan.json](gpu-gate-plan.json) gives exact source-relative selectors,
12 canonical test-file hashes and runner requirements. The
[portability audit](runs/preflight/test-portability-audit.json) also identifies
three static source-assertion files, two dynamically loaded test/example helpers
and six examples-package helpers. Stage all files read-only outside the installed
package. Verify the actual image ID, compiler
tree label, installed `tilelang.__file__`, wheel and loaded native-library hashes.
Import the installed compiler before loading tests. **No compiler source or
Python overlay is permitted.** The final runtime wheel does not include the
compiler's source/test tree.

Keep `/tests` off `sys.path`: its three canonical compiler source assets are
read as text only, and their assertions must be reported separately from native
compiler/GPU tests. Place the six examples imports beneath a separate
`/test-support` root containing no compiler implementation, then add only that
root after verifying the installed compiler. Preserve the repository-relative
layout for the two helpers loaded by filename. No authored test is edited or
silently deselected to avoid setup dependencies. Do not stage the original
`testing/conftest.py`, which intentionally injects a compiler checkout into
`sys.path`; the driver supplies deterministic seeds, isolated caches and strict
collection/skip checks without that source injection.

Run the complete retained-feature and fix gates in two fresh processes with
separate empty caches, preserving each outcome and log. Include NVFP4, MXFP8,
mixed FP8/FP4, MXFP4 and scale-layout tests; the four varying-scale A/B cases
must produce exactly 480, and the MX reference-product tests retain exact
equality. Include all staged IKET ranges, payloads, dump modes and CLI tests.
Zero selected cases or blanket skips do not satisfy a gate.

Use the unchanged `authored_standalone.py prepare` case for prepare call-form
deduplication/reuse, execution while frozen, named missing-specialization
rejection and thaw. Give each invocation its own writable `/out`; the original
runner writes `standalone.json` and fails the process on any failed check.
Unrelated standalone variants are outside this packet's adoption scope.

For consumer fences, load the staged `test_tilelang_issue_3529.py` and directly
invoke the three portable functions listed in the gate plan. The copied
`authored_consumer_fences.py` is retained as provenance but assumes a source tree
beside the imported compiler; it cannot be run unchanged in the wheel-only
runtime. Do not mask that absence with an alternate compiler package.

The image then needs the complete vLLM release gates, all native functional
outcomes, runtime identities, serving preparation/freeze and IKET debug checks,
and both unchanged full performance boots. Standalone author passes cannot
replace these gates. Preserve the actual failure and stop on lost retained
features, numerical errors, unprepared kernels, failed requests or guard stops.

## Acceptance and report

Use both C40 boots as the primary saved reference. Keep all four candidate/
reference pairings, both repeat margins, adverse rows and missing-data limits.
Report compatible saved 1030, C36, previous C40 and current C42 with signed
over/under and explicit better/worse/no-proven-change verdicts. C36 has one
compatible saved boot; do not invent its repeat. Preserve C39/C41 evidence and
never remeasure their images or released builds.

Correctness/retained-feature failures or material core/serving regressions
beyond the existing paired and repeat-aware method prevent acceptance. Do not
introduce a new tolerance after results or dismiss engine/prefill costs because
their attribution is uncertain. Keep decode, TTFT/TPOT and recorded ITL distinct;
include cold/mixed prefill, first-chunk waits, engine cost, draft acceptance,
recorded cache/KV/preemptions, memory/swap, startup and thermals. Missing ITL/KV
data and absent targeted workload baselines remain explicit limits.

This is the explicit corrected interaction packet after the parent's focused
review, not authorization for speculative retries. If it still misses its mark,
highlight the problem, preserve its image and all outcomes, and drop the
compiler candidate. Necessary default behavior and safety fixes remain on C40.
No promotion or bypass of outstanding TP4 correctness/release gates is implied.
