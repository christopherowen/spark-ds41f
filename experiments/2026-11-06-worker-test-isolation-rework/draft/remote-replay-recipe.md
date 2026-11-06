# Bounded remote source replay and actual-image fixture proof

This is a draft recipe only. No clone, replay, new reservation, container or test
has been started for it. Root must supply the specific host claim and execution
scope. C43 remains a failed image, with no serving or performance result. A
successful source replay can precede the actual-image witness, but then remains
proposed source rather than a proven native diagnosis or qualified candidate.

## Fixed inputs

- Published deployment/control commit: `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`.
- Source upstream: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`.
- C43 series: `experiments/2026-10-26-spark-ds41f-candidate/candidate-43/vllm/series`.
- Exactly 576 patches, preserving every series byte, order, path and patch byte.
- C43 patchset: `05586bfb7487c6436f75bc240e934ff7d1c2b81877f8c0014ae985a9f9d6245a`.
- Reproduced prefix head: `99029e6cbdb966ff198dd13902a0825e8d507e27`.
- Reproduced prefix tree: `082bda0f13e94290f462eaf55d143cdb7931f9e5`.
- Corrected draft: `draft/fixture-cleanup.diff`, SHA256
  `e7950e488a1b874427a65e3e57e9efdf252a593e5679d93a4d6a1d3433020b92`.
- Only allowed changed path:
  `tests/v1/worker/test_gpu_tilelang_logprob_softmax.py`.
- Expected old/new producer byte hashes:
  `36ebabc33c3870b19355f89ce56d35d4d8137f4b20d22b19ba62317c32111f66` /
  `7c15eb3a5704e43b08ca06f91f7a4145d59efdbf3e7084f0dce025b09b6a91c0`.

## Capacity and isolation prerequisites

Read actual shared holds/requests and host reservations immediately before the
job. Root must explicitly confirm no exclusive/GPU timing conflict and a
host-scoped CPU-only claim for this work. Do not change a foreign hold/request.
The current sp8 foreign image build must be left running and unmodified. Its
image-build lock is not source-job authority and must not be taken or removed.

Use both agreed common CPU-work and offline-work locks, acquired nonblockingly;
root must verify the actual paths/schema before launch. A blocked lock, unclear
claim, insufficient memory, unexpected GPU work or exclusive timing means defer.
Do not interpret a vanished shared hold as a grant. Keep the claim and locks
through all children; remove only this exact claim normally on exit.

Run the entire process and descendants with:

```
CPUQuota=100%; affinity=one allowed CPU
MemoryMax=8G; MemorySwapMax=0; LimitAS=8589934592; TasksMax=64
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
GIT_NO_LAZY_FETCH=1 GIT_TERMINAL_PROMPT=0 PYTHONDONTWRITEBYTECODE=1
```

A parent-approved `systemd-run --user --scope` can enforce the cgroup limits;
verify actual effective limits before starting Git. No environment installation,
Docker, GPU devices, kernel tests, serving commands, node synchronization or
profile edits belong in this source-only job. No heavy workstation fallback.

Proposed fresh isolated root:
`/home/chris/spark-work/worker-test-isolation/c46-source-attempt-1/`.
Refuse preexisting output. Write stdout, stderr, actual exit and step receipts
incrementally there. Preserve the failed state on error; do not reset/am-abort,
erase outputs or silently retry. Bound each Git command and the total job with
timeouts; a timeout is an explicit incomplete attempt.

## Source-only replay sequence

1. Record host, claim, hold/request snapshot, effective cgroup/affinity/AS limits,
   lock ownership and memory/swap observation. No credentials or environment dump.
2. Prepare an isolated deployment packet from the exact fixed execution commit.
   Existing sp8 checkout `/home/chris/projects/spark-ds41f` is read-only. A shared
   clone may borrow its objects; any required fetch/checkouts occur only in the
   fresh isolated clone. Name the deployment remote `deployment`. Record packet
   commit and all 576 patch paths/byte counts/hashes, the series hash and the
   corrected-draft hash. No commits/pushes in the deployment repository.
3. Recompute the repository's patchset algorithm: hash applied-upstream revisions
   plus NUL if any, the literal series bytes, then for each active line the literal
   patch name, NUL, and full patch bytes. The result must equal the fixed C43
   patchset and the active count must be 576. Resolve every path within the packet;
   no TODO placeholder, missing patch, rewrite, reordered patch or repair is allowed.
4. Select an existing vLLM object cache read-only. sp8 inventory found 13 prepared
   trees under `/home/chris/projects/spark-ds41f/.work/build/*/src/vllm`.
   Exact object availability is not yet verified: a bounded 5-second C40 commit
   probe timed out. Set `GIT_NO_LAZY_FETCH=1` before any new cache probe. Never
   fetch/configure/update the original cache. Use a fresh `git clone --shared
   --no-checkout CACHE source`, then remove its auto-created `origin`. Add
   `upstream` using the exact URL in C43's published lock. If pinned upstream
   objects are missing, a bounded explicit fetch is permitted only into this
   isolated source clone under root's scope; retain all fetch errors. Do not
   assume a cache's current HEAD is an authority.
5. Create named source branch `feat/worker-test-isolation-rework` from the exact
   upstream base. Replay all 576 packet patches with the original controls:

   ```
   git -c user.name='Christopher Owen' \
       -c user.email='3221756+christopherowen@users.noreply.github.com' \
       am --quiet --committer-date-is-author-date PATCH_1 ... PATCH_576
   ```

   Use an argument array with resolved patch paths, not shell interpolation.
   Retain the historical whitespace warning if emitted; do not edit the prefix.
   Require exact expected head/tree and a clean source worktree at this point.
   Record all source commits and byte hashes; do not infer success from tree alone.
6. Export only the seven source files needed by `fixture_witness.py` to a fresh
   `before-source/` receipt directory. No dependency installation or imports.
   Check corrected draft with `git apply --check`, then apply it in the isolated
   source clone. Require exactly the one allowed file changed, the old/new
   producer hashes above, every original assertion and `_assert_same` call
   preserved, post-context laziness assertion added, and `git diff --check` zero.
   Parse the proposed Python source as syntax only. Runtime/consumer/frontend
   file bytes and all 138 manifest names must remain identical. Do not change
   chunking, tolerances, deselections or the lazy-reference consumer invariant.
7. Make one source-only commit on the named branch with neutral subject
   `Isolate lazy-reference poisoning in the worker test`. Use the same author and
   committer identity, with one recorded explicit author/committer timestamp.
   Its message must carry these headers/body fields:

   ```
   Upstream base: 19f2c20ed4d64ae22bd009ceacadd85e2a81e354
   Source base: C43 99029e6cbdb966ff198dd13902a0825e8d507e27
   Rationale: restore the test's original absent module-dictionary entry without
   resolving a lazy reference during monkeypatch cleanup.
   Quality: test fixture isolation only; preserve numerical/routing assertions,
   lazy-reference consumer checks, runtime bytes, manifest and runner grouping.
   Tests: list only actually completed source checks and their exact receipts;
   actual-image pytest witness and fresh native qualification remain pending.
   Upstream status: local, not submitted.
   ```

8. Produce `git format-patch --stdout --full-index -1 HEAD` into a new patch file;
   preserve license notices. Record its SHA256, commit ID, source tree, complete
   diffstat, per-file old/new hashes and exact command/exits. Reapply this format
   patch with `--committer-date-is-author-date` on a fresh named branch at the
   already freshly reproduced C43 prefix; require identical new head and tree.
   If replay differs, preserve it as failure rather than modifying commit dates
   until it matches. A new full repository Linux prepare/build check and actual
   image/native gates remain later candidate requirements.
9. Export minimal `after-source/` witness files; require unchanged files identical.
   Publish no runtime/profile edits. Return only the source patch, hashes, precise
   source proof receipt, preserved errors and pending qualification limits to root.
   Root owns the durable candidate46 series/manifest/control packet and publication.

## Actual immutable C43 image: metadata before fixture assertions

Separate root-coordinated DGX4 CPU-only scope is required. This is source fixture
proof using the actual pytest dependency, never a repeat native C43 gate. First
assert Docker resolves this exact image ID without pulling:
`sha256:ac3c5a2ca2e934295ca7be60d69566d4c406aff22057b0a911661d49e5664905`.
Do not use an image tag or `docker exec` into any running container.

The common container argument array is:

```
docker run --rm --pull=never --runtime=runc --read-only --network=none
  --cpus=1 --memory=2g --memory-swap=2g --pids-limit=64
  --cap-drop=ALL --security-opt=no-new-privileges
  --tmpfs=/tmp:rw,noexec,nosuid,size=64m
  --env=NVIDIA_VISIBLE_DEVICES=void --env=CUDA_VISIBLE_DEVICES=
  --env=PYTHONDONTWRITEBYTECODE=1 --env=OMP_NUM_THREADS=1
  --env=OPENBLAS_NUM_THREADS=1 --env=MKL_NUM_THREADS=1 --env=MAX_JOBS=1
  --env=EXPECTED_IMAGE_ID=sha256:ac3c5a2ca2e934295ca7be60d69566d4c406aff22057b0a911661d49e5664905
  --mount=type=bind,src=ABS_WITNESS,dst=/work/fixture_witness.py,readonly
  --entrypoint=python3 EXACT_IMAGE_ID /work/fixture_witness.py --metadata-only
```

Use an argument-safe supervisor; save metadata stdout/stderr and actual wait exit
to fresh distinct files on the host. The script refuses any `/dev/nvidia*`, reads
real pytest version and installed monkeypatch module path/SHA256, then flushes
metadata before source checks. It emits dependency-capture errors before exiting.
No GPU flags, devices, network, persisted container writes or imports of actual
Torch/Triton/kernel modules are used.

Only after successful metadata capture, a second bounded invocation may mount
minimal verified `before-source/` and `after-source/` trees read-only and replace
`--metadata-only` with `--before-source /work/before --after-source /work/after`.
This invocation again emits metadata before any witness assertion. Require
original fixture red, corrected fixture green, and preexisting pollution still
detected. Persist failure output and do not reinterpret a modern-pytest result
that does not reproduce the old behavior as confirmation. Synthetic module and
reference doubles prove cleanup semantics only, not numerical/kernel behavior
or the full native gate; later C46 image/gates must provide that evidence.
