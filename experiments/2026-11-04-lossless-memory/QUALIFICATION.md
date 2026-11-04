# Qualification sequence

These are measurement candidates. The standalone FP8 GPU screen ran and rejected
all three tested Huffman layouts and all three fixed-field layouts. Each of the
twelve fresh containers passed its exact-byte gate. Neither isolated serving image has been built;
no serving improvement or release acceptance is claimed.

## Source identities

Both isolated serving arms start from the frozen 1030 recipe and the promoted
vLLM tree `ea1281f880e7535e3d87725be0156a8e38fc2fb2`.

| Variable | Canonical patch head | Tree | Proposed immutable image |
| --- | --- | --- | --- |
| Expert scales | `30e56e093b33a9f45d6f58d032d65ed5055a2105` | `c3a9bf4be9b672f11c99f0c3098778a901dbae99` | `spark-ds41f:memory-scales-c3a9bf4be9b6` |
| Engram rows | `8bcfc031f1c4c4f40fdebe3ea1099c8cc8076aea` | `05a895a269246adf3c8a573562dd9eda8b1372b9` | `spark-ds41f:memory-engram-05a895a26924` |

Fresh canonical preparation reproduced both locked heads and trees. A verified
local Git object cache supplied objects missing from an upstream partial fetch;
no revision, source bytes or check was substituted. Complete local inputs for both arms are now prepared, including the pinned
auxiliary sources and hash-locked aarch64 wheels. Both `build check` commands
pass. `runs/full-input-preparation/` retains the initial fetch failure and retry;
`runs/engram-observation-preparation/` records the final instrumented Engram tree.
Target image builds and checks are still required. The first image attempt was
rejected before stopping service because its plan used a non-resolving profile
path; `runs/component-plan-failure/` retains the original receipt and logs.
Both plans now use complete repository-relative paths, and the controller
validates every selected lab plan before image or serving operations.

Reproduce the source preparation and local configuration checks with:

```sh
bin/spark --cluster-config experiments/2026-11-04-lossless-memory/expert-scales/tp3.json build prepare --only vllm
bin/spark --cluster-config experiments/2026-11-04-lossless-memory/engram/tp3.json build prepare --only vllm
bin/spark --cluster-config experiments/2026-11-04-lossless-memory/expert-scales/tp3.json doctor
bin/spark --cluster-config experiments/2026-11-04-lossless-memory/engram/tp3.json doctor

python3 scripts/release_tests.py check .work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-edba9f62/src/vllm 19f2c20ed4d64ae22bd009ceacadd85e2a81e354 30e56e093b33a9f45d6f58d032d65ed5055a2105 experiments/2026-11-04-lossless-memory/release-tests.json
python3 scripts/release_tests.py check .work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-5012ef72/src/vllm 19f2c20ed4d64ae22bd009ceacadd85e2a81e354 8bcfc031f1c4c4f40fdebe3ea1099c8cc8076aea experiments/2026-11-04-lossless-memory/release-tests.json
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

The Engram arm adds a guard for invalid negative column deltas before signed
subtraction. `engram/check_row_plan.c` reproduced the original overflow under
UndefinedBehaviorSanitizer (exit 134), then passed with the guard (exit 0).
It also checks all bytes of 1,025 valid packed rows, including page splits.
`runs/engram-c-bounds/` preserves both logs and exact commands. This portable
check substitutes neither Linux direct-I/O nor the native GPU regression tests.

The final Engram tree also records cold packing seconds, physical bytes, bytes
written and repeat-boot reuse once per rank/layer. This adds startup observation;
the bounds-checked C reader, file identity, row bytes and hot path are unchanged.
After each boot, `engram/check_live_cache.py` checks every packed byte against
the immutable source and confirms the serving reader holds those files open.
It runs in a separate 1 GiB CPU-only container with read-only source/cache mounts,
so its buffers do not consume serving's memory allowance. Reads are bounded and
drop their clean page-cache pages; the verifier stops at a 4 GiB host floor.

## Operations

The current site is TP3 triangle. Do not start a TP4 arm on that cabling. TP4
plans are prepared for its own authorized, hardware-qualified window only.
Kernel job targets must belong to the window's production node map: the FP8
probe uses dgx1; scale and Engram component/full-gate tests use dgx1/2/3.

The owner has authorized coordinated candidate boots/restarts and production
restoration (`runs/authorization.json`). Candidate 36's TP3 dev profile currently
matches the live cluster and is the restoration profile for this window. The
publish, idle and hold guards still apply; promotion is not authorized.

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

For either serving component, prefer the strict experiment controller after
opening the normal window with note `owner-authorized lossless memory qualification`:

```sh
python3 experiments/2026-11-04-lossless-memory/run_component.py \
  --component expert-scales --window WINDOW_ID \
  --out results/private/lossless-memory-scales-round1 --apply
# In a subsequent owned window, use --component engram and a distinct --out.
```

The controller builds and distributes one image, requires its original successful
build/import-smoke receipt, and checks every rank's native component and full-gate
verdict. A zero lab exit with a failed test verdict cannot start serving measurement.
It then boots the candidate twice through inspected coordinated plans, without a
shape override. It requires startup memory samples, complete native suite exits,
the gated image on every rank, cool starts and unchanged boot identity across
companion suites. Engine logs are retained before another boot can replace them.
Complete original logs stay on each node; the controller exports at most 32 MiB
per rank and rejects a run exceeding that evidence bound.

The same second boot runs quality, prefix reuse, four concurrent 485K-context
requests with forced output, and token/logprob equivalence across traffic shapes
and greedy/gathered paths. No logprob fallback or quality tolerance is accepted.
The controller stops on a failed gate. A management failure latches a review hold
instead of starting another serving container. A foreign window is never closed.
Production is restored once at the end of the owned window.

After both boots, also compare each complete model report with the recorded
1030 model evidence pinned by `model-reference.json`. This is an offline check;
the released build is never run again. The original report covers all 256
sequences and 65,536 scored positions. Its model step and all other index fields
match the published reference; the original and published suite definitions
differ in later prefill extents/description, as the provenance records. This
model evidence supplies no substitute throughput observation.

```sh
python3 experiments/2026-11-04-lossless-memory/model_identity.py \
  --reference INPUTS/recorded-1030-model.json.gz \
  --provenance experiments/2026-11-04-lossless-memory/model-reference.json \
  --candidate INPUTS/first-model.json.gz INPUTS/repeat-model.json.gz \
  --out INPUTS/model-identity.json
```

The comparison requires complete generated-token, rank, given-token score and
returned top-token vector identity. Missing sequences, failed requests, truncated
vectors, non-finite scores, a different corpus/workload and any numeric change
fail the gate. Returned top/given-token scores are not a complete vocabulary
distribution; the exact kernel/scale/row tests remain required too. The recorded
reference self-check validates this analysis path and is not candidate evidence.
Both candidate comparisons must pass before a release proposal claims unchanged
recorded model behavior.

`--through gate` ends after the image tests. A later window may use
`--reuse-gates-from RECEIPT` only when the prior window restored production,
no serving measurements began, and image identities, native gate hashes and
test-bundle hashes still match. Prior serving observations are never overwritten.

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

Each serving arm also has an `image.json` plan that builds once on dgx1 and
copies that image over the verified direct-peer SSH links. Expected host public
keys were read through the existing trusted management connections before
installing their direct-address aliases; no private key or credential enters Git.
The planned routes are dgx1 to `10.12.2.2`, then dgx2 to `10.23.2.3`.

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

- FP8: all three tested GPU Huffman decoders were byte exact and slower than
  their paired raw consumers. Reject these layouts. Even a faster future byte
  consumer would not qualify a fused GEMM. Port only a measured viable layout, retain exact
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
