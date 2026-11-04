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
receipts, #875's active compiler candidate, and the separate active research in #877. PR #842 merged this default-enablement
work; no duplicate PR was opened.

## Round order and measurements

1. Candidate-39 rebases #842 on the exact candidate-38 tree. CopyDrafter becomes
   derived default behavior for greedy adaptive DSpark, with automatic decode-row
   and capture-envelope sizing. This is an explicit interaction test of enabling
   the path and overlapping its bit-identical copy work; compare both recorded
   candidate-38 standard and copy recipes to distinguish those questions.
   These have separate success criteria: default enablement is required behavior;
   the overlap proposal is a performance hypothesis and did not establish a
   serving-speed gain. It must not be relabeled as cleanup after measurement.
2. Review the independent scratch-read fix in #878 and qualify it as its own
   corrected candidate before further tuning. Its lazy liveness guard applies
   to the exact candidate-39 tree; the current image and matrix stay frozen.
3. The proposed minimum match 8 versus 16 round was
   [dropped at review](../2026-11-04-copy-policy-review/README.md). Adverse saved
   simulations and no compatible targeted timing baseline do not justify a new
   GPU round. No patch or serving speedup is claimed. Any future hypothesis
   needs its own defensible causal mechanism and compatible recorded baseline.
4. The compiler refresh in #875 stays a separate candidate and quality gate:
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

Candidate-39 image preparation, build and the release-test gate passed in attempt 2.
The window yielded for the requested DGX4 slot after preserving the completed
gate and verifying candidate-36 on every serving rank. That slot and the full attempt-3 matrix are complete. Candidate-36 restoration
is independently verified. The measurements below show no proven serving
throughput gain versus either C38 recipe; a separate correctness fix is required.
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

## Second dispatch: image gate before the shared slot

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
`dgx1-1793783564` opened at 09:12:44 UTC. Its heartbeat remained live through closure.
Complete image preparation and `build check` passed; the image build began at
09:14:38 UTC. These are preparation results, not serving-performance results.

Current native log: `~/tl-logs/copy-default-screening/candidate-39-window-attempt-2.log`
on DGX1; setup uses `launch-attempt-2.json` and `exit-attempt-2.json`. The
controller receipt is `results/private/lab/candidate-39-round-attempt-2/receipt.json`.
The initial source/launch/reservation/image snapshots are retained in
`runs/2026-11-04-dgx/attempt-2`. Keep execution pinned at `b464e9ad`; later
publication of evidence is not a reason to synchronize an active window.

## Scheduled DGX4 sharing after the gate

The LSA work requested a 10–15 minute DGX4 slot through the regular request file
while candidate-39's release gate was running. The owned hold records a planned
serialized handoff after this gate ends and the existing service is verified.
A hardware-review hold blocks that handoff. No simultaneous DGX4 work is added
to the TP3 timed window. The request does not interrupt the current gate.

After the slot, a completed, passing image gate can be reused for the remaining
unmeasured work with a distinct controller attempt:

```sh
python3 experiments/2026-11-04-copy-default-screening/run_round.py --attempt 3 --resume-image-gate-from results/private/lab/candidate-39-round-attempt-2/receipt.json
```

This is a plan unless `--apply` is supplied. The resume guard requires a safely
closed scheduling handoff, all image/build/gate exits zero, a nonempty passing
test summary, no guard/restoration/hardware error, and the exact frozen image
`sha256:17550af8ae03a42919d72fac7d2771a18f0f30300b45f0b06edc1ac83625f885`.
It rechecks the unchanged native gate receipt and DGX4 image before transferring
it. It rejects any earlier functional or timed phase, and preserves prior
outputs. Nine offline safety cases pass. The current gate is still pending;
this control change neither qualifies it nor changes its running checkout.
Only after the foreign slot ends and the usual publish/sync/live/idle guards
pass may the new controller resume. No image build or test gate is repeated.

## Completed gate and safe shared-slot closure

Attempt 2 completed its image gate at 09:58:18 UTC. Image
`sha256:17550af8ae03a42919d72fac7d2771a18f0f30300b45f0b06edc1ac83625f885`
passed all seven preparation/build/gate steps. The 136 per-file receipts cover
exactly the gate manifest: 8,074 tests passed, 360 skipped, zero failed and zero
errors. The original JSONL, XML, manifest and logs are retained alongside hashes
in `runs/2026-11-04-dgx/attempt-2`. Skipped tests remain a qualification limit;
this gate does not establish model-output correctness or serving performance.

During build and tests, maximum GPU temperature was 64 C, minimum thermal
headroom 23 C, maximum system temperature 74.6 C, minimum available host memory
79.3 GiB, and swap growth zero. Thermal slowdown was zero; the recorded power-cap
counter was 407,500.5 ms. These are build/gate conditions, not serving metrics.

The controller honored the queued clustered-indexer request before seed,
functional or timed work. It ended at 09:58:37 UTC, actual exit 1, with the exact
scheduling closure error and `restored=true`. Its child image/gate exit was zero.
The release-test container stopped, the hold was removed normally, and an
independent candidate-36 live doctor confirmed every rank matches the named
restoration profile, retaining the two known recommendations. The shared
handoff receipt was marked ready only after these checks. No owner relay is
needed to schedule the foreign slot through the hold/request files.

The resume validator accepts these preserved receipts. After the other slot
ends, synchronize a single published control commit through the usual guarded
process, then use attempt 3 to reuse this identical image and completed gate.
Do not rebuild it or repeat any already completed measurement. The subsequent
functional and two-boot serving results decide whether a tuning round is sound.

## Third dispatch: remaining qualification resumed

The other task returned DGX4 at 10:15:21 UTC. Our own queued return request was
fulfilled without changing another request. Guarded staging window
`dgx1-1793787483` used the named candidate-36 restoration profile; the start plan
reported the same live and planned TP3 shape. Git-based cluster sync put DGX1–3
at published commit `2e6bc10ed8ae6a5f36459e9985011aca070179e3`, with the clean
owned DGX4 build checkout detached at that same commit. The frozen image and
original gate receipt retained their build lineage at `b464e9ad`. Staging closed
at 10:19:14 UTC after verifying the already-matching serving profile, with no
serving restart.

An operator launch command failed in shell/Python quoting before any dispatcher,
controller, hold or native attempt-3 directory was created. Its error and the
absence checks are preserved. Corrected argument-safe transport then launched
exactly one supervisor (PID 3000758) and controller (PID 3000759), at 10:20:40 UTC.
This is a launch error, not a measured candidate failure or replaced data point.

Attempt 3 opened window `dgx1-1793787679` at 10:21:19 UTC. The explicit resume
control validated the unchanged gate and actual DGX4 image, then began seeding
that image. It repeats neither build nor release-test gate. Functional and
two-boot serving measurements remain pending. The actual wait exit will be
retained separately when this controller ends. Initial launch, queue fulfillment,
staging and resume receipts are in `runs/2026-11-04-dgx/attempt-3`.

A fresh PR inventory still contains only the four distinct open changes #851,
#865, #875 and #877; no additional superseded PR was found.

## Independent correctness follow-up

PR #878 isolates a top-k merge scratch-read fix from the separate clustered
search experiment. On the clean prepared candidate-39 tree `8e33015b`, local
inspection confirms the same eager bitwise condition around `WorkEntries`:
the mask does not prevent reading a list whose scratch entries were unwritten.
The proposed outer lazy guard applies with `git apply --check`, without changing
the prepared tree. The PR supplies original/fixed sanitizer receipts; those GPU
checks have not been independently rerun by this experiment. This source review
is preserved in `runs/preflight/topk-review.json`.

Keep #878 as unique correctness work. Preserve the frozen candidate-39 matrix
and assess its results with this known source issue recorded. Qualify a separate
corrected candidate before copy-match tuning or promotion. Passing ordinary
output comparisons alone would not close the scratch-read issue; performance
measurements do not replace the sanitizer and correctness gates.

## Functional collector identity error

The functional phase exited zero at 11:14:03 UTC and the controller proceeded to
performance. That phase exit is not an all-step qualification pass: both runtime
collector extras exited 1. This experiment reused a candidate-38 collector whose
expected image ID is a hardcoded candidate-38 digest. Its original failed checks
remain preserved. No other nonzero functional exit line appeared in the compact
controller-log review; each raw result still needs the final local audit.

Read-only inspection of the six saved rank identities across both functional
boots confirms the actual container and image IDs all equal candidate-39's gated
`sha256:17550af8ae03a42919d72fac7d2771a18f0f30300b45f0b06edc1ac83625f885`.
All six recorded a running unchanged container before/after collection, with
only the stale expected-ID issue. The compact review is saved separately in
`runs/2026-11-04-dgx/attempt-3/collector-identity-review.json`. It does not rewrite
the native verdicts or establish a full repaired evidence pass.

Keep the active image, specs and workloads pinned at execution commit `2e6bc10e`.
After timing ends, collect and verify the original logs, hashes, source labels and
rank agreement locally against the intended immutable image, retaining the old
failures and any newly found issue. Future candidates need a collector with an
explicit expected-image input. Do not remeasure candidate-39 to repair a metadata
check or silently replace a native failed receipt.

## Completed candidate-39 measurements

Attempt 3 completed at 13:59:26 UTC on 2026-11-04, actual controller wait exit 0.
Both functional boots and all six performance suites completed. The 26 ordered
performance steps exited zero; the request tables, full output lengths, workload
identities, same-boot companion suites and fresh repeat boot passed local audit.
The 256-prompt corpus digest matches both saved boots of both candidate-38
recipes. No released build or earlier candidate image was remeasured.

| Output tokens/s, boot / repeat | C38 standard | C38 copy | C39 default |
|---|---:|---:|---:|
| 8 streams | 163.95 / 163.80 | 163.64 / 163.28 | 163.34 / 163.80 |
| 4 streams | 126.45 / 126.29 | 126.13 / 125.83 | 125.91 / 126.04 |
| 1 stream | 65.52 / 65.42 | 65.43 / 65.29 | 65.35 / 65.34 |
| Sampled 16 streams | 160.11 / 160.20 | 159.41 / 159.66 | 160.73 / 160.05 |

These are the original window-throughput means, not pooled recipes. Every boot
retains its ten-window confidence interval. `recipe-comparison.json` keeps all
four boot pairings against each earlier recipe, all adverse rows, and headlines
using both candidate and saved-reference A/A floors. C39 has no proven serving
throughput gain over either saved C38 recipe. The small C4/C1 losses against the
standard recipe are not robust across both saved-reference repeat floors. The
15-shape engine-grid cost is consistently worse against the standard recipe
(about 0.9% for its first saved boot); the smaller copy-recipe cost is less
conclusive after saved-reference variation. Do not attribute inherited gains
over 1030 to this default-enablement round.

C39's eight-stream median user decode is 20.67 / 20.83 tokens/s. TTFT p99 is
0.417 / 0.414 s; TPOT p99 is 65.58 / 65.03 ms. TPOT is not an inter-token-gap
percentile. The four-stream TTFT p99 is 0.322 s on both boots, versus the saved
1030 observation 0.311 s, without a percentile confidence interval. The short
decode wait before the first 131K prefill chunk is 0.527 / 0.514 s, versus
0.438 s in 1030; this remains an interactivity lever. Raw stall tails have wide
intervals and are retained. Actual 524K/1M reference token counts differ, so
small prefill comparisons retain that warning.

Accepted tokens per draft at eight streams are 1.305 / 1.308; prefix-cache hit
rate is zero on these timed inputs, and all timed preemption batches are zero.
This workload does not establish cache-rich agent behavior. The exports do not
retain timed KV occupancy, raw hit/miss totals or ITL-tail distributions; no
values are inferred for those fields. The copy probe reports about 16 output
tokens/step for dense copy, 14 for edits and 6.8 for tool-like prompts, with no
request errors on either boot. Those synthetic results do not score useful task
completion.

During timed suites, minimum host memory available was 6.11 GiB, swap growth was
zero, maximum GPU temperature 78 C, and minimum thermal headroom 9 C. Thermal
slowdown and power-cap counters were zero. Performance boot readiness was
106.6 / 103.3 s (see the native boot-time records for exact values). The two
native runtime collectors still have failed verdicts: explicit local revalidation
checks all 178 exported file hashes, exact source/profile/image identities,
container stability and untruncated fault-free logs, and accepts only the stale
C38 expected-ID exception. Three rejection checks prove a wrong actual image,
an additional native issue, or a different stale-ID origin is still rejected.

Candidate-36 was restored under the tracked profile, its actual image verified
independently on every rank, and its live doctor passed. The hold was removed
normally. The other task then acquired its separately authorized capture window;
this task performs no GPU or serving operations during that window.

`source-receipt.json` and `publication-receipt.json` preserve native hashes and
byte-for-byte published timing/conditions/identity JSON. The full private export,
model text and raw logs remain locally under `final-export`, following the saved
C38 publication convention. Original failed collector summaries are published.
No native verdict or unfavorable point was overwritten. The three analysis
scripts reproduce the audits from that private export.

CopyDrafter defaults are retained as source behavior, without a speedup claim.
The known inactive scratch-read issue in #878 keeps C39 unqualified for release.
Candidate-40 isolates that correctness fix and uses an explicit collector image
identity plus checks for every functional extra. Its serving qualification is
pending. Minimum match 8 is dropped at review; the compiler refresh remains a
separate candidate when its prerequisites allow.
AgentX does not score useful task completion or replace the outstanding TP4
correctness and release gates. No promotion is made.
