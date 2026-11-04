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

Preparation only: no candidate-40 image, GPU run or serving change exists yet.
The next copy-match and compiler rounds wait for this candidate's quality and
measurement review. No promotion or bypass of TP4 correctness/release gates.

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
No candidate-40 GPU work has started. Re-read actual state, preserve foreign
requests/holds, and remove only our fulfilled request before guarded staging.
Refresh published changes and select one exact clean source/control commit on
all serving checkouts and the owned DGX4 build worktree; record actual execution
identity separately from later evidence commits.
