# CopyDrafter defaults and measured candidate rounds

The owner requested PR cleanup, CopyDrafter enabled by default, and successive
candidate measurements in core metrics, using Spaceport to help. Base deployment
commit: `d9b557cb8e87bd31331afc27bec9ed953cd0de8d`. Candidate-38 source and its
recorded two-boot TP3 standard/copy qualification are the starting references.
Released builds and already measured earlier candidates are not remeasured.

## PR review

Closed #850: its recipe-form cleanup changes no serving command and introduces
a known frozen-release equality-test failure. Its patch remains on the PR.
Retain #851's unique startup/image safeguards, #865's unique external-site
receipts, #875's active compiler candidate, and #842 for this default-enablement
work. Reuse #842 rather than opening a duplicate default-enablement PR.

## Round order and measurements

1. Candidate-39 rebases #842 on the exact candidate-38 tree. CopyDrafter becomes
   derived default behavior for greedy adaptive DSpark, with automatic decode-row
   and capture-envelope sizing. This is an explicit interaction test of enabling
   the path and overlapping its bit-identical copy work; compare both recorded
   candidate-38 standard and copy recipes to distinguish those questions.
2. After candidate-39's quality and performance review, a separate round may test
   minimum match 8 against 16, changing only that variable. It needs new source
   identity and receipts, including a mixed copy-dense/tool-like workload that
   binds the token budget. A failed or slower round is retained and rejected.
3. The compiler refresh in #875 stays a separate candidate and quality gate:
   its standalone tests do not establish model-output or serving performance.

Every measured arm needs a boot and repeat boot. Record servebench release,
single-stream and sampled suites with their unchanged workload definitions:
paired output throughput, per-user throughput, TTFT and ITL tails, short-request
wait beside long prefill, cold prefill, engine-step time, accepted/verified drafts,
cache reuse, KV occupancy, errors and output integrity, boot time, minimum host
memory, swap allocation movement and thermals. Judge differences against saved
reference variability and retain every adverse row. If a decision hinges on a
small difference, use the repository's null-arm discipline. No isolated kernel
speedup is an end-to-end serving gain.

## Hardware and restoration

Spaceport access currently fails bounded SSH/TCP checks through the running
operator relay and from DGX1. Its TP4 cooling failure and prior no-content
response also remain unresolved. No reservation, running service, hardware,
network or cooling policy there has been changed. Do not launch there until
management access is isolated and independently verified; a target management
failure requires renewed explicit hardware authorization. Use its capacity only
when those conditions and its reservation allow it.

DGX1–3 are reachable, clean and serving the recorded candidate-36 TP3 dev profile
with identical image `sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11`.
DGX4 is reachable and has no GPU containers. Before any GPU work, re-read the
shared hold, publish the exact source, coordinate all node checkouts through
`bin/spark cluster sync`, use a named restoration profile, and open the guarded
lab window. Restore candidate-36 once at window end and verify every rank and
live doctor. No topology change or released-build remeasurement is authorized
by these experiment files. The owner's request authorizes the candidate
measurement work; hardware recovery remains separate.

## Current state

Candidate-39 image preparation and build are running on DGX4 in attempt 2.
No new serving-performance measurement has completed.
The fresh full vLLM preparation reproduced candidate-38 tree `c061e5f6` exactly;
#842 applies without conflicts. The Mac partial-clone canonical lock replay
failed on a missing intermediate local GLP blob. Its failure is retained in
`runs/preflight/review.json`; a first Linux cache-backed fetch also failed because its partial clone
could not serve every promised object. A fresh direct fetch of the pinned
upstream then completed the full canonical replay and preparation. All logs
are retained. Both candidate profile doctors and 626 repository tests pass.
No GPU result is inferred.

## Guarded execution

`run_round.py --apply`, launched once on DGX1 from the exact published clean
commit on all serving ranks, opens the lab reservation against candidate-36.
It maintains the heartbeat while `build_gate.py` builds and tests on idle DGX4.
The builder uses cold-start, memory and thermal guards and keeps its original
logs and per-file test outcomes. Only a passing image/gate receipt allows the
same image digest to be seeded and distributed to DGX1–3. Then the functional
spec and unchanged release/single-stream/sampled suites run with fresh repeat
boots. The controller stops on a failed phase and restores candidate-36 once.
Management loss is latched, preserves a hardware-review reservation and blocks
a restoration launch even if a later probe recovers. No automatic retry replaces
a failed run.

Native controller receipt: `results/private/lab/candidate-39-round/receipt.json`
on DGX1. Image/gate receipt: `results/private/candidate-rounds/candidate-39-image/receipt.json`
in the clean DGX4 build worktree. Both scripts default to a read-only plan; no
serving change is made by reading this experiment. Collect and analyze native
exits, digests, conditions and both boots before a verdict or the next round.

## First dispatch: yielded before GPU work

Published #842 merged at `f3795349`; execution controls were finalized at
`f9ba89ea` and synchronized cleanly to DGX1–3. The DGX4 build worktree used that
same commit. Staging left candidate-36 running and its live doctor passed.
The first controller opened at 08:07:37 UTC on 2026-11-04, then honored a
clustered-indexer request before starting any phase. It ended at 08:07:55 UTC,
exit 1, with zero build, GPU or measurement steps. Candidate-36 was confirmed
live and the window closed. The other task subsequently acquired the window.
All receipts and the setup supervisor's actual exit are retained under
`runs/2026-11-04-dgx/attempt-1`. This is a scheduling outcome, not a candidate
performance result. The supervisor retained the actual process exit status.

Resume the unchanged matrix with `run_round.py --attempt 2 --apply` only after
the shared reservation and request queue allow it, from a newly published clean
commit on all nodes. The executor refuses to overwrite prior outputs or resume
if any previous phase started. No new GPU measurement exists yet.

## Second dispatch: local qualification running

The owner confirmed local DGX4 as the fallback if the VPN could not be repaired,
with normal shared scheduling. The isolated operator VPN gateway had no recent
handshake; its configured endpoint still resolved to the active endpoint. A
restart with its existing configuration preserved all isolation rules but did
not establish a handshake or reach sp1 in five bounded checks. No Spark service,
hardware, key, site routing policy or reservation was changed by the repair.
The fallback and exact diagnostic results are in `runs/preflight/vpn-recovery`.

DGX source staging synchronized all three serving nodes and the owned DGX4 build
worktree to published commit `b464e9ad70090af532e44468aae739670eb9227d`.
A new indexer request deferred the first launch command before it created a
controller. Our request was requeued rather than replacing that window. Once
the repeat/sanitizer job released the reservation, attempt 2 launched at
09:12:04 UTC on 2026-11-04 (controller PID 2926115). Window
`dgx1-1793783564` opened at 09:12:44 UTC. Its heartbeat is live.
Complete image preparation and `build check` passed; the image build began at
09:14:38 UTC. These are preparation results, not serving-performance results.

Current native log: `~/tl-logs/copy-default-screening/candidate-39-window-attempt-2.log`
on DGX1; setup uses `launch-attempt-2.json` and `exit-attempt-2.json`. The
controller receipt is `results/private/lab/candidate-39-round-attempt-2/receipt.json`.
The initial source/launch/reservation/image snapshots are retained in
`runs/2026-11-04-dgx/attempt-2`. Keep execution pinned at `b464e9ad`; later
publication of evidence is not a reason to synchronize an active window.
