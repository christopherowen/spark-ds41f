# Pending qualification

The owner authorized hardware tests and coordinated service changes. This draft
does not promote a configuration or authorize overlapping another holder. Read
the head's hold, request and advisory coordination record before any GPU action.
Preserve the explicit collective qualification then verification-budget turn.
No acquisition waiter is running for this experiment.

## First bounded job

Reserve 20–30 minutes on one idle SM121 GPU for native source tests, after the
existing reservations. Serving remains unchanged. Use the exact candidate-38
image ID in `base.json`, the exact published source head, and read-only test and
source-file mounts. Run the same tests against the control and candidate files
in separate processes, then repeat the candidate in another fresh process.

Run the independent group suite, CED metadata suite, GLP policy tests, and
`tests/kernels/attention/test_deepseek_v41_ced_replay.py`. Preserve each exit code,
all failures, image/source hashes, GPU memory and temperature observations. The
old source must reach the asserted state mismatch; a fixture, import, compilation
or unwritten-memory failure is not a valid old-fails reproducer. The finite-cone
test-only policy is exploratory; it is not the production candidate.

Use an exact owned window ID, heartbeat and a supervised container with a
bounded timeout and memory limit. Stop only this experiment's container on
ownership loss. Verify that it has exited and serving identity is unchanged
before closing the hold. A CPU-only container uses `--runtime=runc`, no GPU
devices, at most two CPUs and 4 GiB; it is preparation, not native evidence.

## Serving comparison after native gates

Capture the actual original clean deployment commit, live profile, image digest,
site map and shape on all ranks. Install and test `branch_watchdog.py` against
that captured state before changing any checkout. It restores through a separate
clean controller at the original commit. Each operation runs under the recorded
operation lock; each long-running child has an activity marker until confirmed
exited. Never remove the hold with unresolved children or failed restoration.

Publish this experiment commit, then inspect the coordinated sync plan:

```sh
bin/spark --cluster-config experiments/2026-11-06-prefix-group-validity/screen-tp3.json cluster sync --revision PUBLISHED_COMMIT
```

Read the plan's `live:` and `shape:` lines and inspect all nodes. Apply only that
reviewed plan, with its explicit cluster config. Every rank must be clean and
detached at the exact published commit. Do not use `--change-shape`.

Use `screen-tp3.json` for candidate Python mounts and `control-tp3.json` for the
unreleased parent. Both use the same OCI digest and runtime arguments.
`guarded_probe.py` checks the all-rank source hashes, mounts, deployment identity,
cooling, fans, memory and thermal conditions before issuing requests. Run it
under `window_run.py` with the same owned window-state file.

Use candidate A, control A, control B, candidate B. Keep the window throughout;
restore once when done. The recovery workload is new and has no recorded control,
so this unreleased parent is the causal control. Do not rerun released builds or
replace immutable recorded measurements. The probe uses uninterrupted cold A/A,
ordinary reuse, explicit preemption after 129/257/1025 observed output tokens,
and two concurrent admission orders. A successful reset plus the server's
preemption delta is required. Unstable uninterrupted A/A makes exact output
comparison inconclusive, not a pass. Actual streaming-input sessions retain baseline behavior; the candidate does
not claim their recovery correctness. Native metadata gates must verify that
this exclusion survives the final input sentinel.

On both candidate boots, run `runtime-guard-suite.json` with the existing
`bin/spark bench --suite` under `window_run.py`, named profile, `--compare none`
and unique output directories. Compare each exact step against the two recorded
candidate-38 runs in `runtime-guard-references.json`; the combined suite is not an
identical whole-release suite. Report intervals, A/A floors, boot drift, memory
and recovery costs separately. An interval permitting a material regression is
inconclusive even if it includes zero.

Finish with coordinated stop, exact original-commit sync and original-profile
start, inspecting each plan before apply. Require all original ranks, clean
checkouts, image digest, profile, origin and API readiness, plus `doctor --live`,
before releasing the hold. No full-model, performance or restoration result has
been recorded for this experiment yet.
