# Authorized serving qualification

The owner authorized hardware testing, including the coordinated sync, stop,
start/restart and restoration needed for this experiment. This does not authorize
promotion, merging, recabling or running under another holder's window.

At preparation time dgx1's hold belonged to the frontend lifetime qualification,
and `norm-projection-boundaries` already owned the request file. Neither is ours to
replace. The standard queue has no pending jobs, but its publication guard accepts
only main; this branch experiment will use the existing coordinated CLI in an
exclusively owned window after the prior holder/request hand off. Do not weaken the
main-only helper or let its queue runner race the branch checkout.

## Comparison identity

The control is unreleased candidate-38: vLLM head `1f33102547e4da5c64712390604695d1abd577c4`,
tree `c061e5f6931306fa4655ba531428f639c365bf10`, dependency image
`sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`.
Reverify its source labels and image identity on every serving rank; a matching
dependency image on Spaceport alone is not a distributed serving receipt.

The candidate adds exactly one patch: head `0aaebda5cc9fcc941973b43d62fd7b229508a371`,
tree `5328d6b7279c789e824dd5a3800f1165f27ff1f7`. All source inputs and export-context
hashes have been prepared and checked. Its image is not built yet. Build once with
the repository's guarded image command and smoke test under our own GPU window;
record the image ID and build receipt and distribute that same image to every rank.
DGX4 already has the identical parent image ID above; after its prior work hands off,
it is a suitable build/distribution host. The control uses a new immutable alias
`prefix-control-c061e5f6` for that exact ID, avoiding any overwrite of candidate-38.

No released build is rerun. The new native multi-turn workload has no matched
recorded before result, so the owner-authorized causal control uses the unreleased
parent. Existing release/candidate suites remain the references for the broader
deployment performance; do not replace their numbers with this experiment.

`lab-tp3-comparison.json` records the order: candidate A, control A, control B,
candidate B, three repeats on each boot. This is a specification to inspect with
`scripts/lab.py run ... --dry-run`, not permission to invoke its main-only live runner
on an unmerged branch. Each guarded probe cools all ranks, requires the configured
fan policy, verifies the live source/image/deployment identity, and watches memory
and temperatures. Parent and candidate use the same TP3 topology, model, runtime
arguments, native transcript and source files. Only endpoint retention differs.
The expected cache count is selected from the verified source tree; the control is
allowed its old hit but must pass all output, ownership and pressure checks.

## Exclusive branch window procedure

1. Wait for the current holder and prior request. Read hold/request state immediately
   before taking any GPU action. Use exclusive creation of our request only when
   the request slot is free; never replace someone else's file. Preserve ordering.
2. Capture the exact original deployment SHA on all nodes, their clean status,
   current live profile, image IDs, node map and shape. Verify that the profile to
   restore is actually live and that production is idle. The observed restore profile
   during preparation was `experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json`;
   re-read the current holder's record rather than assuming it stayed unchanged.
3. Take an exclusive hold with our unique holder/window ID and recorded restore
   SHA/profile. Keep its heartbeat current. Before changing checkouts, install and
   validate a watchdog that restores both the original commit and profile when our
   heartbeat expires; the ordinary watchdog alone only restores a profile, so it
   is insufficient for a branch checkout. It must never act on a different window.
4. Keep separate clean control checkouts for the published candidate commit and
   the captured original commit. Use the site's actual node map, not the offline
   fixtures used for `doctor`. Both controllers must resolve exactly the same nodes.
5. From the candidate controller inspect the complete plan for
   `bin/spark --cluster-config experiments/2026-11-05-prefix-endpoint/tp3.json cluster sync --revision <published-SHA>`.
   Read its `live:` and `shape:` lines, then use the same command with `--apply`.
   Every node must be clean and detached at that exact published SHA. Never copy a
   dirty tree or bypass a refusal with `--change-shape`.
6. Inspect each named arm's coordinated start/stop plan before applying it. Start
   each arm with `cluster start --replace --apply` only after the image/commit/node
   checks pass. Run the guarded probe arguments printed by the comparison spec.
   Keep the hold across all four boots; do not restore production between arms.
7. Before closing, stop the candidate with its named profile. From the original
   controller inspect and apply coordinated `cluster sync --revision <original-SHA>`
   using the captured restore profile, then inspect/apply its coordinated start.
   Verify original commit, image identity and `doctor --live` on all ranks before
   removing our hold. Keep the hold and error evidence if restoration fails.

This is a runbook, not a claim that the watchdog or handoff has been executed. Its
exact commands, plan outputs, heartbeat/watchdog tests and restore receipts must be
captured before and during the window. Do not start the branch run without them.

## Evidence required

Record each arm's boot SHA, source tree, image ID, shape, cache state, fan/cooling
conditions, request IDs, workload/token hashes and memory/temperature summaries.
Compare cold/warm and solo/concurrent token IDs and returned logprobs, plus identical
requests across parent/candidate boots. Confirm real server-side cancellation from
logs/metrics; a client stream close is not proof of a cancelled prefill. Require
observed eviction after the immediately preceding pressure baseline and preserve
every failure/retry.

Report before/after incremental TTFT and request/aggregate output throughput,
absolute/relative change, 95% intervals across the independent boot pairs, cached
and recomputed tokens, memory and preemptions, with receipts. The wider guard should
include candidate 32K/128K/256K/512K/1M cold prompts and c1/c8/c16 distinct traffic
against matching recorded reference workloads. Reuse existing benchmark tools and
references; these guards do not turn unrelated historical deltas into a causal
claim about this patch. The prefix-retention mechanism predicts no cold-prefill
arithmetic speedup and at most one additional 256-token tail page of reuse.

Hardware results are pending. Neutral, negative and inconclusive results remain
valid outcomes; neither PR becomes ready on CPU results alone.

## Watchdog and request lease

`branch_watchdog.py` restores through a separate clean controller pinned to the
captured original SHA. Its state file records `window`, exact `holder`, `hold_path`,
`restore_controller`, `restore_commit`, `restore_profile`, `restore_image_tag`,
`restore_image_id`, `operation_lock`, `restore_receipts` and `activity_markers`.
The restore controller runs on the head so its local Docker image is the captured
source for coordinated distribution. Preserve the sites selected node map there.nnStart the new watchdog and verify its ready receipt/PID before replacing the ordinarynwatchdog for this same window. Never stop another windows watchdog. Hold the named
operation lock around every coordinated mutation. Create an activity marker before
a build, smoke or probe and remove it only after confirming that job exited. The
watchdog restores service after a stale heartbeat, but retains the hold if any job
marker is unresolved; reconcile those jobs before releasing the GPU window. This
prevents a surviving build/probe child from overlapping the next holder.

The guarded probe requires `--window-state`; the same state goes to its child.
Both re-check exact ownership and heartbeat, and every HTTP request checks the lease
before sending. Loss of ownership or a stale heartbeat stops further requests.
The watchdog runs stop, sync to the captured SHA, start and `doctor --live`, retaining
all plans/results. A refusal, changed topology, failed sync or failed live check
keeps the hold. `--restore-now` uses the identical procedure at normal completion.

CPU tests cover foreign-window refusal, stale leases, refused/changed-shape plans,
restore ordering and failed-sync hold retention. They do not replace the required
actual pre-window state capture, plan inspection and final live restore check.
