# Pending qualification

The owner authorized hardware tests and coordinated service changes. This draft
does not promote a configuration or authorize overlapping another holder. Read
the head's hold, request and advisory coordination record before any GPU action.
Preserve all earlier claims in the current coordination record, including the
Engram serving window and the expert-routing native reservation.
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

The test source is the clean published candidate checkout at the head recorded
in `source.json`. Mount its entire `tests/` directory read-only at
`/opt/spark-ds41f/candidate/vllm/tests`. Keep `PYTHONPATH` and the working directory
at `/opt/spark-ds41f/candidate/vllm`, so imports use the image's compiled package.
For the candidate only, mount the four files enumerated in `overlay.json` at
their recorded container paths; verify their hashes first. Do not mount a whole
source checkout over the compiled package. Candidate tests must not come from
the older image: its metadata and GLP stand-ins lack the new recovery field.

Run these separate processes, saving a unique log and receipt for each. Each
command starts with `python3 -m pytest -q -p no:cacheprovider`. The container's
entrypoint must impose a wall-clock timeout; the supervisor must remove only its
own named container on ownership loss and confirm that it exited.

| Arm | Additional pytest arguments | Required result |
| --- | --- | --- |
| Control | `--confcutdir=tests/kernels/attention tests/kernels/attention/test_deepseek_v41_ced_replay.py -k test_preempted_decode_rebuilds_original_decoder_history` | Four finite output mismatches at the final equality assertion; no setup failures or skips |
| Candidate A | Same native reproducer command | Four passes |
| Candidate A | `--confcutdir=tests/kernels/attention tests/kernels/attention/test_deepseek_v41_ced_replay.py` | All 32 native cases pass, including final KV/auxiliary state and too-short replay controls |
| Candidate A | `--confcutdir=tests/v1/attention tests/v1/attention/test_b12x_v41_ced_metadata.py` | Full metadata suite passes, including the fixed-input/session scope check |
| Candidate A | `--confcutdir=tests/models tests/models/test_deepseek_v4_1_glp.py` | GLP policy and lifecycle suite passes |
| Candidate A | `--confcutdir=tests/v1/core tests/v1/core/test_prefix_group_validity.py` | All 183 group/lifecycle and input-session identity cases pass |
| Candidate B | Repeat the complete candidate matrix in fresh processes | Same results; preserve independent process identities and receipts |

Run the four outstanding HiSparse cases on both arms with the exact selection
recorded in `runs/hisparse-{control,candidate}-cpu.json`, now with the qualified
GPU. Those prior device-inference failures remain in the evidence. Native skips
are unqualified results, even when pytest exits successfully. Do not count new
metadata API argument failures on the old source as a runtime reproducer: those
tests intentionally require interfaces absent from the control.

The native stack uses fixed encoder rows and captures the decoder layer inputs
needed by auxiliary consumers. It does not execute actual DSpark context
projection/storage or compressed/indexer reconstruction. The serving gates must
still qualify the checkpoint, cancellation and full private-state lifecycle.

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
two concurrent admission orders, and a cancelled producer followed by same-prefix
reuse. Cancellation targets only the exact producer ID; its acknowledgement,
an increased abort counter and idle server are all required. The follower must
report cached tokens and match the uninterrupted tokens and scores.
A successful reset plus the server's
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
