# Top-k liveness qualification on the default-enabled base

Base deployment: `5962b17a` (the complete candidate-39 measured review).
One intended runtime variable: the lazy inactive-list scratch-read guard from
merged #878, applied after the exact candidate-39 source. Candidate-39 remains
unchanged, with both boots and all unfavorable evidence retained.

Candidate-40 full canonical replay from pinned upstream base `19f2c20ed4d6`
reproduced candidate-39 head/tree before applying the guard. The resulting head
is `5f926df6d3126266a9c88c82c06544060008c33c`, tree
`ca1496c6481dc5bc4b4234b743e4ca023a6c7597`, with 574 active patches.
Commands, original log hash and every preparation result are under
`runs/preflight`. The imported canonical objects are a disposable cache verified
against the tracked candidate-39 identities; running state is not configuration
authority. The published patch series and manifests remain the source of truth.

Use the normal shared DGX reservation. Build/release-gate on idle DGX4, including
strict initcheck of the actual new image with the independent #878 reproducer
and no source overlay. Only then distribute the identical image to all three
serving ranks. Require every functional extra's actual exit before timed work.
The runtime collector must receive an explicit expected immutable image or
passing image-gate receipt. Keep the current experiment's specs/data distinct.

Run the unchanged functional and release-tp3/single-stream/sampled workload
families on a fresh boot and repeat boot; preserve errors and adverse rows.
Compare both new boots against both candidate-39 boots, both saved candidate-38
recipes, and immutable release references. Never remeasure those images. Do not
claim a small gain from one boot or a favorable reference selection.

Restore `experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json`,
image `sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11`,
once at window end and independently verify all ranks/live doctor. Clean
published same-commit checkouts, named plans, guards and foreign/hardware-review
holds remain mandatory. No Spaceport hardware operation is authorized here.

Candidate-40 attempt 1 started at 15:34:15 UTC, with window
`dgx1-1793806495` opened at 15:34:55 UTC. All eight image/gate steps passed at
16:21:53 UTC. The image is
`sha256:7ae96e98e361601d8232cd99e6af1c6044b7c282bf6904c989e6ad033c147f82`.
The controller then yielded for the queued shared request, before seeding,
functional work or timing. Serving measurements remain pending. The next
copy-match and compiler rounds wait for this candidate's measurement review.
No promotion or bypass of TP4 correctness/release gates.

## Execution controls

Both profile doctors, Python 3.12 parsing and eight offline control tests pass.
Repository `build prepare --only vllm` reproduces the exact 574-patch head/tree.
A historical intermediate whitespace warning is retained in its original log;
the final canonical source passes whitespace checks. Complete Linux preparation
and `build check` run before the image build, when all inputs are available.

`build_gate.py --window WINDOW --apply` requires the owned head reservation and
idle DGX4, refuses an existing candidate-40 image tag, builds once and runs the
full release tests. It then runs strict initialization checking of the new
image's actual top-k module: no source overlay, exact module-byte hash, expected
tie-ordered output, and zero reported initialization errors are all required.
A failure stops before deployment.

`run_round.py --apply` defaults to attempt 1 and refuses existing output, foreign
reservations or queued requests. It owns heartbeat/watchdog/restoration. Only a
passing image and sanitizer receipt permits distribution and functional boots.
The generic `scripts/collect_runtime.py` accepts an explicit immutable image ID
or a passing source-bound image receipt; this spec names the latter. The
controller checks all 40 ordered functional exits before starting performance,
so the stale-collector failure seen in C39 cannot pass through a parent zero.
Native quality summaries still receive the final independent artifact audit.

Preserve launch PID, actual wait exit and all native outputs under distinct
attempt labels. A safe scheduling close after all image gates and before any
functional/timed work can resume with `--resume-image-gate-from` and a new
attempt, using the unchanged image/receipt. Other prior phases or failures are
refused; no benchmark or failed gate is silently repeated. Management or
telemetry loss latches a hardware-review hold and prevents a restoration launch.

The unchanged full workload families are used here, rather than comparing a
lean protocol with C39's full protocol. The shared window is expected to take
about 4–5 hours, without a time cap; ordinary requests close it after its current
job. A released build and already measured earlier images are never remeasured.

## Coordinated queue

The tested source/control packet is published at
`2922052677cb01c959af3461efa617cc7aa88b7b`. A new foreign lossless-memory
window opened before our source-staging dispatch. The dispatch honored it and
exited before creating a setup directory, reservation, controller or node
checkout change. This is scheduling history, not a measured candidate failure.

An atomic `who=copy-default-screening` request was queued at 15:16:03 UTC behind
window `dgx1-1793805208`, asking for about 4–5 hours after its current job. The
original request/hold snapshot and deferral are preserved in `runs/preflight`.
The foreign window subsequently released. Only our matching request was
fulfilled and removed. Guarded staging window `dgx1-1793806106` synchronized all
three serving checkouts and Git-updated the owned DGX4 build worktree to exact
published execution commit `1f134c7aad82bc63c95afed36f488becf2959a41`. All four
checkouts were clean; live C36 verification passed. Staging closed at 15:33:45
UTC without a serving restart.

The attempt-1 detached controller was PID `3431611`, supervised by PID `3431610`,
under `/home/swank/tl-logs/topk-live-qualification/candidate-40-window.log` on
DGX1. Native controller output is
`results/private/lab/candidate-40-round/receipt.json`; builder output is
`results/private/candidate-rounds/candidate-40-image/receipt.json` in the owned
DGX4 worktree. Operator setup, initial liveness/phase receipts and hashes are
preserved under `runs/2026-11-04-dgx/attempt-1`. The supervisor recorded actual
exit 1 at 16:22:13 UTC, solely `RuntimeError: window changed or closure requested`;
the image/gate child exited 0 and no serving phase started. Later evidence commits do not change
the active execution identity and must not trigger synchronization or restart.

At 15:37:11 UTC a separate lossless-memory request entered the shared queue,
asking for a short screen after the current image-and-gate job. The request is
retained in `shared-request-1.json`; it does not interrupt the build or gates.
The controller yielded safely before functional/timed work. Preserve its actual
exit/restoration and the completed immutable gate. Local explicit resume review
passed; after the foreign request is fulfilled and its window releases, use a
distinct attempt with the unchanged gate. Do not rebuild, repeat gates or remove
the foreign request.

## Completed image gate; serving qualification pending

The independent local artifact audit verified 136 unique per-file receipts
exactly cover the manifest: 8,074 passed, 360 skipped, zero failures/errors.
All 14 original XML reports agree. The actual-image top-k initialization check
verified the native module path and byte hash, exact tie-ordered output and
`ERROR SUMMARY: 0 errors`, without a source overlay. Its original log and hash
are retained. The actual DGX4 image still matches the frozen digest, and both
gate containers are stopped.

Build/gate conditions: max GPU 64 C, minimum thermal headroom 22 C, maximum
system 73.5 C, minimum available host memory 78.96 GiB, zero swap growth and
thermal slowdown; recorded power-cap time 434,777.5 ms. These are build/gate
conditions, not serving-performance measurements.

Independent named C36 live doctor exited 0 after closure. All three running
containers independently matched the C36 image and exact restoration-profile
label. Native controller/error, actual exit, original logs, image receipt,
XML/JSONL, all export hashes and separate restoration receipts are preserved in
the attempt directory. Full raw exports remain private under `gate-export`.

## Resume attempt 2

The foreign FP8 screen released window `dgx1-1793809624`; only our exact matching
return request was fulfilled. Named guarded staging window `dgx1-1793810908`
read the unchanged TP3 live/shape and synchronization plans, synchronized all
three serving checkouts back to the frozen execution commit `1f134c7a`, and
verified the owned DGX4 build worktree already matched it. All four checkouts
were clean. Staging closed at 16:49:25 UTC with C36 still live and no restart.

The sole resume controller started at 16:49:55 UTC (PID `3534669`, supervisor
`3534668`), opening window `dgx1-1793811034` at 16:50:34 UTC. It explicitly
reuses attempt 1's unchanged completed image/gate receipt and actual image.
At the initial compact check it was seeding the image, with both processes alive,
fresh heartbeat and no final exit or request. No image build or gate repeated.
The native receipt is `results/private/lab/candidate-40-round-attempt-2/receipt.json`;
the log is `/home/swank/tl-logs/topk-live-qualification/candidate-40-window-attempt-2.log`.
Fresh operator setup, staging, PID and initial hash receipts are preserved in
`runs/2026-11-04-dgx/attempt-2`. Functional and serving-performance results remain
pending. The owner paused recurring checks and requested direct waiting here.
