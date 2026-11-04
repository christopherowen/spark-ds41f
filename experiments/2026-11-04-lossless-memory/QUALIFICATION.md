# Qualification sequence

These are measurement candidates. No image has been built, no target GPU test
has run, and no serving improvement or release acceptance is claimed.

## Source identities

Both isolated serving arms start from the frozen 1030 recipe and the promoted
vLLM tree `ea1281f880e7535e3d87725be0156a8e38fc2fb2`.

| Variable | Canonical patch head | Tree | Proposed immutable image |
| --- | --- | --- | --- |
| Expert scales | `30e56e093b33a9f45d6f58d032d65ed5055a2105` | `c3a9bf4be9b672f11c99f0c3098778a901dbae99` | `spark-ds41f:memory-scales-c3a9bf4be9b6` |
| Engram rows | `24fa42dfed8a55e33ed97532a4a3ac0707be20fd` | `833cd629a486fdc78e21cf8d06c916e3f4932ceb` | `spark-ds41f:memory-engram-833cd629a486` |

Fresh canonical preparation reproduced both locked heads and trees. A verified
local Git object cache supplied objects missing from an upstream partial fetch;
no revision, source bytes or check was substituted. Only the vLLM part is prepared
locally. Complete build inputs and target image checks are still required.

Reproduce the source preparation and local configuration checks with:

```sh
bin/spark --cluster-config experiments/2026-11-04-lossless-memory/expert-scales/tp3.json build prepare --only vllm
bin/spark --cluster-config experiments/2026-11-04-lossless-memory/engram/tp3.json build prepare --only vllm
bin/spark --cluster-config experiments/2026-11-04-lossless-memory/expert-scales/tp3.json doctor
bin/spark --cluster-config experiments/2026-11-04-lossless-memory/engram/tp3.json doctor

python3 scripts/release_tests.py check .work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-edba9f62/src/vllm 19f2c20ed4d64ae22bd009ceacadd85e2a81e354 30e56e093b33a9f45d6f58d032d65ed5055a2105 experiments/2026-11-04-lossless-memory/release-tests.json
python3 scripts/release_tests.py check .work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-586802d8/src/vllm 19f2c20ed4d64ae22bd009ceacadd85e2a81e354 24fa42dfed8a55e33ed97532a4a3ac0707be20fd experiments/2026-11-04-lossless-memory/release-tests.json
```

The successful preparations supplied the existing verified object cache through
`GIT_ALTERNATE_OBJECT_DIRECTORIES=$PWD/.work/upstreams/vllm/.git/objects`.
That cache is disposable and is not a build input or source authority.

The current repository release-test catalog also names four test files added
after the promoted source. The experiment's `release-tests.json` records those
absent paths and retains all 132 gate files present in this source. Classification
checks pass for both candidates (170 selected files, zero problems). Global test
classification is unchanged. The generated `full-gate` bundles execute the full
catalog; the smaller `bundle` directories are initial component screens.

## Operations

The current site is TP3 triangle. Do not start a TP4 arm on that cabling. TP4
plans are prepared for its own authorized, hardware-qualified window only.

Before any hardware work, read the hold on dgx1 and obtain the required explicit
authorization for coordinated stops, candidate boots/restarts and restoration.
Publish clean source, inspect every plan's live/shape lines, and use the recorded
production configuration of the acquired window. Never substitute an older
recipe for the profile that the window must restore.

Build with each variable's own `tp3.json` configuration, using the standard
`bin/spark build` workflow on an idle host. Preserve the initial build failure,
if any. Publish the build receipt and source commit, distribute one OCI digest,
and verify that every configured rank resolves that exact image. Every apply
names its cluster configuration explicitly.

For each variable, complete `build prepare`, `build check`, then
`build image --apply` with that explicit `--cluster-config`. The latter runs
GPU smoke tests and belongs inside the authorized cluster window. None of these
complete builds has run for this experiment.

Within the window, run jobs back to back with `--keep-open`, restoring production
once at the end. The FP8 first/repeat specs have different run IDs, bundles and
native output filenames, so neither overwrites the other's observations.

```sh
scripts/lab.py run experiments/2026-11-04-lossless-memory/fp8/kernel-32.json --keep-open
scripts/lab.py run experiments/2026-11-04-lossless-memory/fp8/kernel-32-repeat.json --keep-open
# Repeat independently for 64 and 128 lanes if the 32-lane gate remains viable.

scripts/lab.py run experiments/2026-11-04-lossless-memory/expert-scales/kernel.json --keep-open
scripts/lab.py run experiments/2026-11-04-lossless-memory/expert-scales/full-gate.json --keep-open
scripts/lab.py run experiments/2026-11-04-lossless-memory/expert-scales/release-tp3.json --keep-open

scripts/lab.py run experiments/2026-11-04-lossless-memory/engram/kernel.json --keep-open
scripts/lab.py run experiments/2026-11-04-lossless-memory/engram/full-gate.json --keep-open
scripts/lab.py run experiments/2026-11-04-lossless-memory/engram/release-tp3.json --keep-open

scripts/lab.py window close
```

These commands are **not** a request to execute before those guards are met.
Screen one variable at a time and stop its progression on a failed exactness
gate, an unqualified memory condition or an unfavorable timing result.

The serving plans run the candidate's release suite, single-stream suite and
sampled suite on two boots. They compare with immutable recorded 1030 references;
the released image is never measured again. The primary release definitions are
copied exactly from their recorded `suite.json`, preserving their fingerprints.

Before promotion, also run startup/cold-cache compilation checks, prefix and
admission checks, and an ordinary agent workload. Keep stock and steered request
identity separate. Require unchanged output digests and target/draft arithmetic,
no failed requests, no memory/thermal stops, and confidence intervals supporting
the claimed benefit. A capacity gain is reported as capacity, not invented TPS.

## Remaining conditional work

- FP8: the GPU decoder is an uncompiled probe. Even a faster byte consumer does
  not qualify a fused GEMM. Port only a measured viable layout, retain exact
  accumulation order, then give that new image its own full serving qualification.
- Cold prefix: the store is a bounded CPU prototype. First export only written,
  fenced checkpoint records from a real workload. Include every group and
  private target/draft auxiliary state. Measure complete restoration after KV
  eviction, never treating a page-only round trip as a continuation proof.
- Engram: include initial packing time, bytes written, source/cache identity,
  lowest memory and the repeat boot's reuse. The packer uses buffered I/O with
  per-chunk synchronization; its costs are not the older experiment's numbers.
- Every accepted variable needs owner acceptance and a new promotion commit
  updating configuration, documentation and immutable baseline evidence together.
