# Explicit preparation with disabled compiler caching

This is one focused source rework of C42's required IKET prepare/freeze failure,
reserved for candidate45. C42 built successfully but failed a native retained
feature; it has no serving-performance result. The ten authored compiler patches
assembled as C42 were not a prior rework of this newly observed integration
failure. The owner asked to correct and qualify the idea. C42's image, errors and
all original verdicts remain unchanged. C44 remains withdrawn and unmeasured;
its VERSION-only proposal could not fix this failure.

The correction changes the JIT wrapper/preparation contract only. It adds no
compiler pass, kernel schedule, numerical tolerance, model/draft policy, topology,
cooling or hardware change. The compiler refresh retains its original performance
acceptance criteria. Passing these CPU controls does not establish a speedup or
native correctness; the distinct corrected image must receive its own first full
qualification and both metric boots before a performance verdict.

## Correction

Default IKET disables caching. C42's `prepare` stores a compiled kernel in the
wrapper, but #880's disabled-cache call bypass ignores it. Once compilation is
frozen, the next call raises `KernelNotPreparedError`. The native test has its
own cache; shared old-version cache reuse does not explain this branch.

The rework records successful explicit preparations separately. A disabled-cache
call may use one while frozen only when the same kernel object still occupies
the same parsed call key, with the same function/code and compile context.
Ordinary disabled-cache calls remain fresh while thawed. An ordinary cached
object is insufficient permission to run frozen; missing specializations still
raise the original named error. Fresh preparation invalidates its old permission
before elaboration or compilation, so failure cannot authorize an old object.
Explicit replacement also clears stale call-form shortcuts. Tuning arguments
use the same normalization as a JIT call.

Preparation does not blindly promote an implicit wrapper entry. It goes through
the existing global compiler cache, which can reuse a valid IR/context result
when caching is enabled. Repeating a still-valid explicit preparation with cache
enabled remains a hit. With caching disabled, preparation compiles fresh unless
the already-explicit object is being requested while frozen.

The normal cache-enabled hot-call path is unchanged. The additional record is
per wrapper and parsed specialization; it retains references to already retained
kernel/function/code objects plus a small context snapshot. Native memory and
startup effects still require measurement.

## Source identity

| Input | Identity |
|---|---|
| Recorded runtime base for source preparation | Validated default-enabled C40; vLLM head `5f926df6d3126266a9c88c82c06544060008c33c`, tree `ca1496c6481dc5bc4b4234b743e4ca023a6c7597` |
| Canonical compiler upstream | `2123e412ce2217b324c50c4ec967213906456ac5` |
| Reproduced unchanged ten-patch prefix | Head `454202b2ba8d0cf10b8caf434a956755e039483f`, tree `bae5fe12b58cdf364efeccbabc04cd08205c43b9` |
| Focused correction | Head `2eb3e70abcb3ba523ef2d77549fb14646b704196`, tree `35f15411cf3af0fc06a1cb0097215b1372a40464` |
| Final source with namespace metadata | Head `a470d5daf8529fcf2b661f1119c8f4bb1aecc5e2`, tree `80fcfc4521b06e0c9776bbf1b5419849557e1090` |
| Twelve-patch fingerprint | `2018cea5190861fd1ebaf51a5ed95289debdcba4b2f931fd333af95794a90fff` |
| Package version | `0.1.15+snapshot.35f15411cf3af0fc06a1cb0097215b1372a40464` |
| Intended image | `spark-ds41f:candidate-45`; actual digest **unknown** |

The source delta from C42 is exactly `tilelang/jit/__init__.py`, one new CPU
control test file, and VERSION. All ten original patches and all23 canonical
gate/static/helper assets are byte-identical. Fresh named canonical replay
reproduces both the original prefix and the final twelve-patch head/tree.
Disposable prepared trees are caches; the durable series and source identities
are authority. Recursive submodules and the full Linux build remain unqualified.

## CPU controls and their limits

Twenty-two tests execute actual AST-extracted JIT, prepare and freeze methods
without importing TileLang, TVM or Torch. Compiler outputs and argument binding
are doubles. Cases cover explicit frozen reuse, ordinary bypass, missing and
different keys, tuning normalization, positional/keyword agreement, failed
preparation, wrapper clear/replacement/reset, new wrappers, cache toggles, thaw,
changed function/code/default target/backend/compile options, call-form shortcuts,
stream arguments and eager-mode rejection. All22 pass. The positive case fails
on the original C42 source with its frozen error while its ordinary-bypass
negative control passes. Original logs and a corrected test-harness setup error
are retained under `runs/preflight`.

The context guard uses object identity and representations of supported wrapper
compile options and default target/backend. It is a local reuse guard, not a
cryptographic source identity or universal mutation tracker. Mutating captured
Python globals, callbacks or the function's meaning requires a new wrapper;
existing frontend IR memoization does not support arbitrary monkeypatching.
Do not mutate compiler context while preparation or serving is active.

`KernelCache.clear_cache()` clears its global memory/disk cache, not existing
JIT wrapper objects. This patch preserves that existing lifetime: an explicit
prepared object can remain in memory. Clearing/replacing the wrapper entry or
resetting/creating its wrapper removes its permission; `thaw()` permits fresh
compilation and does not claim to destroy live kernels.

## Source-bound cache namespace

The previously reviewed VERSION mechanism is reused with this correction's own
full immutable **pre-metadata** tree, avoiding a circular hash and avoiding
library discovery that could hash an unrelated loaded library. The actual
provider under `NO_VERSION_LABEL=1` and PEP440/wheel parsing accept the new literal.
The provider, packaging/CMake source and numeric-backend mechanism are unchanged
from the recorded source-only feasibility checks. This is not a complete native
configure, wheel build or installation.

Unmodified pure cache methods select a namespace and key distinct from both
plain0.1.15 and the withdrawn C44 version. The expected Linux ARM namespace is
`0.1.15_snapshot_35f15411cf3af0fc06a1cb0097215b1372a40464/linux-aarch64`.

Before first release-gate import on the builder and first serving use on every
rank, the actual mounted host namespace must be absent. Preserve old caches;
never clear a preexisting new namespace to pass. Then require actual installed
module version, distribution version, immutable pin and cache base-key version
to agree, with the actual expected namespace, source labels and loaded-module/
native-library path hashes. Imports can create directories, so absence must be
checked before the mapped compiler import. Feature processes retain separate
empty per-boot caches and no compiler source overlay.

## Native qualification still required

The root coordinator owns candidate45 profiles, reservations, actual immutable
image creation, controller and launch. This packet does none of those operations
and authorizes no runtime change. Its final best validated runtime base may be
selected after the separate C43 round is reviewed; this compiler packet does not
freeze or change that frontend choice. The new image needs full Linux source/build
checks, unchanged complete vLLM release tests including the failing IKET test,
all #880 bypass tests, all other retained-feature/fix/strict top-k and functional
gates, then both unchanged full performance boots. Add the new CPU control asset
without deselecting any of the original23 assets. No C42 or released image is
retagged, rebuilt or remeasured, and no C42 serving point exists to repeat.

The new CPU test reads actual source methods. Locally it resolves the prepared
canonical source root. If staged for an installed-image check, set
`TILELANG_PREPARED_REWORK_SOURCE` to the verified installed package's parent
directory (containing `tilelang/`), and bind its JIT/cache file hashes first.
Do not point it at an overlaid compiler checkout or treat these doubles as a
substitute for the real native IKET and cache tests.

Report saved compatible1030, C36, the previous measured candidate and this
corrected candidate with signed over/under, both candidate/reference repeat
margins and all adverse rows. Preserve absent data as absent. This is the one
focused rework of the integrated IKET miss; another actual miss must be reported
and handled under the owner's bounded rework rule, not silently retried.
