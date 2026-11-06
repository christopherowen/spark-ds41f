# PR924 observer scope and lifetime rework

This is a CPU-verified source rework against PR924 head
`e893def6f8363760b5e4af88bf99239dd7a52bd4`. It changes validation tooling,
not the Engram implementation, serving workload, hardware limits, or runtime
profiles. There is no performance verdict.

The PR's reported 738 repository checks and native permission/recovery evidence
are provided evidence. They were not independently rerun here. The reported
native controller ended on a publication guard and restored C36; that does not
qualify or reject the Engram performance idea.

## Source findings addressed

1. Privileged observer entrypoints now require the exact captured holder,
   window and scope token, complete C36 restoration identity, actual DGX1–3
   node map and user, and readable participating-node holds. A three-node count
   or matching holder prefix cannot authorize a root/IPC_LOCK observer.
2. Each observation captures the inspected immutable serving container ID and
   image, init PID and start time, explicit process set, PID birth ticks, and
   host-to-container PID map. The probe verifies PID namespace and cgroup
   membership before and after access. A second immutable-container inspection
   rejects replacement, restarts, PID reuse or process-set changes. The cache
   verifier scans only those captured serving PIDs.
3. I/O probes, FD permission controls and packed-cache verifiers use unique
   window/nonce-labelled container names. They persist a CID receipt before
   start, retain the actual process outcome, and inspect exact ID, immutable
   image and ownership labels before stopping/removing only that container.
   Unknown cleanup retains a sticky observer-review scope through the existing
   controller path. Foreign containers or holds are never overwritten.

The child process has an explicit maximum 600-second wall deadline, distinct
from the cache verifier's retained 1800-second transport timeout. This leaves
margin before the 900-second stale-heartbeat restoration watchdog. An observer
deadline produces a failed/incomplete instrumentation verdict; it cannot pass
partial data or classify an Engram implementation/performance failure. Any
legitimate verifier that later exceeds this cap needs preserved evidence and a
reviewed distinct continuation, never an automatic rerun.

The root/IPC_LOCK peer permission set is retained because the PR's native
controls report it is needed for the serving root process's protected `/proc`
reads. Runtime runc, no GPU devices, read-only mounts/rootfs, CPU, memory, swap
and PID bounds remain explicit. This draft does not change unrelated build,
model-scoring, serving or recovery protocols.

## Review packet

- `observer-scope-identity-lifecycle.diff`: unified source patch, including the
  new pure-standard-library helper and CPU/mock tests.
- `source-rework.json`: exact source/export hashes, proposed hashes, patch hash,
  original PR/tested identities and outstanding checks.
- `test_lossless_observer_scope.py`: review copy of the new tests (already in
  the unified patch). Tests use an in-memory Docker response script; they do
  not connect to a node or inspect real host `/proc`.

The existing positive I/O-command test is updated to require captured target
identity. New tests cover wrong/missing scopes, wrong node maps/local holds,
PID births/namespaces/cgroups, container replacement, exact owned cleanup,
partial timeout receipts, lost create replies, foreign-container refusal,
pre-start scope change, nonzero exit preservation and child/watchdog margin.

## Remote CPU verification

Applied and verified in an isolated sparse checkout of PR924 head
`e893def6f8363760b5e4af88bf99239dd7a52bd4` on `sp8` under bounded
`systemd-run --user` (`CPUQuota=100%`, `MemoryMax=4G`, `MemorySwapMax=0`,
`TasksMax=128`, `RuntimeMaxSec=600s`) at rework commit
`985ad90b8761f0b124b855224b26664c74fe67cc`
(`runs/observer-controls-attempt-7/receipt.json`):

- Verified all 6 original SHA256s, `git apply --check`, `git apply`,
  `git diff --check`, all 8 proposed SHA256s, and AST parsing of all 8 files.
- Passed `tests.test_lossless_observer_scope` (17 tests) and
  `tests.test_lossless_component_runner` (29 tests), then restored the private
  checkout cleanly.

## Still required

No build, Docker, SSH, node operation or hardware work was executed on the
serving cluster for this rework.

Native CPU permission and owned-timeout/removal controls must still be scheduled
under an exact authorized lease to confirm the actual PID-namespace/cgroup
rendering, durable CID receipts and daemon cleanup. Source/mock tests cannot
prove native permissions or daemon cleanup. No existing completed native point
or failed receipt is to be replaced or rerun by this rework.
