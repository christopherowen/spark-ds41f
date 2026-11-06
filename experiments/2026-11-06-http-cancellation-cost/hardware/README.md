# Authorized hardware qualification

Status: prepared, not yet run on GPUs. The owner authorized coordinated hardware
validation and restoration on 2026-11-06 (Europe/Belgrade). Correctness with
preserved performance is acceptable; a speedup is not required. Do not merge or
promote as part of this procedure. Do not use another task's hold or overwrite
its pending request.

## Identity and scope

Both arms use the already built and gated candidate-38 image
`sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`,
verified present on dgx1, dgx2 and dgx3. Its release gate and TP3 qualification
are recorded under `experiments/2026-10-26-spark-ds41f-candidate/candidate-38/`
and `experiments/2026-11-03-candidate-38-tp3-qualification/`.
It is an unreleased control, not a re-benchmark of a released build.

`identity.json` records every source hash. The candidate's complete 574-patch
source differs from that image's source only in `api_utils.py` and its tests.
Thus model arithmetic, native libraries, batches, kernels, scheduler, capacity
and checkpoint are identical between the two hardware arms. These are explicit
source-overlay tests, not a claim that a new OCI image has been qualified.
The API copies are generated artifacts of the recorded source commits; the
ordered patch series remains the source authority.

Effective argument receipts assert that only the port and middleware differ
from the resolved parent arguments. The added middleware requires an explicit
argument list because `serve_options` only replaces existing flags.

Both profiles preserve all parent mounts, use the same TP3 triangle and node
map, enable the same development RPC for the standard screen, and expose only
the experiment on port 18001. Both mount identical deadline fault injection;
the sole arm difference is the native API helper. The fault middleware runs only
when the client sends its experiment header to a non-streaming inference route.
It is not a production timeout policy. Source hashes and the actual image ID are
checked inside every running rank before the guarded client starts.

## Fixed workload and interpretation

The guarded client tests six trials of each combination:

- Four distinct synthetic 32K or 131K token contexts, 64 output tokens each,
  cancelled after two seconds by closing their client requests.
- The same contexts with a two-second server task deadline injected identically
  in both arms. This exercises the demonstrated outer-cancellation defect while
  the real native engine computes. The response must be an explicit HTTP 504.
- Eight distinct short agent requests immediately following each abandonment,
  with temperature zero, thinking disabled, 32 output tokens, and fixed unique
  cache salts. Native running/waiting requests, generated tokens, cache occupancy,
  time to first token, useful completion time and time to idle are retained.
- Six normal-traffic trials each at offered concurrency 1, 8 and 16. Every trial
  uses all sixteen distinct agent prompts; c1 is sequential, c8 uses two groups,
  and c16 one group. Native token counts and text hashes support cross-arm and
  cross-batch comparison.

Cold prefix work is enforced by unique fixed cache salts; every arm starts with
a fresh engine. The client, model, contexts, seeds and request order are fixed.
Long contexts here are explicit synthetic stress, not a realistic conversation;
useful requests are sixteen distinct agent tasks. This task changes request
ownership, not prefill kernels: no new 1M prefill-rate or context-capacity claim
is made. The unchanged image's existing long-context qualification remains the
reference, alongside the new CPU 1M request-boundary and lifetime tests.

A server deadline is a deliberate fault injection. Its result must not be
presented as a speedup for ordinary clients. Ordinary disconnects may be neutral.
Generated native tokens beyond the useful responses, retained active requests,
and time to idle establish whether the real engine continues abandoned work.
Failures and an unobserved engine-active interval are retained, not converted
into a gain. The CPU transport test uses short simulated jobs and a 500 ms
deadline only to validate the client/protocol; it is not hardware evidence.

Each arm additionally runs the existing `screen-tp3` suite, unchanged, with
`--compare none`. Order: candidate-a, parent, candidate-b. The standard suite
comparison uses the repository's paired-request and repeat-boot method, including
its recorded A/A floors (c1 0.35%, c8 0.39%, c16 0.43%). Keep the per-request
interval, repeat drift and floor together. An interval including zero alone does
not establish preserved performance: if a material slowdown remains compatible
with the evidence, report inconclusive and collect a justified additional run.
Do not call the runtime preserved or faster until the data support it.

## Exclusive window and coordinated operations

The main-only lab queue is not used for this published experiment branch. No
publication guard is weakened and no PR is merged to enable testing. The owner
explicitly authorized an exclusively held window using the existing coordinated
cluster commands, with exact checkout and service restoration.

1. Wait for the current holder and earlier pending request. Submit a request with
   exclusive file creation only when the request slot is free. Remove only this
   task's matching request when its turn arrives.
2. From a clean, published checkout, capture the live restore profile, all node
   checkout commits and image IDs, node map, and `doctor --live`. All three must
   be coherent. Verify thirty seconds idle. The currently observed restore
   profile is candidate-36/tp3-dev.json; reverify it at handoff rather than treating
   another holder's record as authority.
3. Create an exclusive hold with holder `http-cancellation-hardware`, unique
   window ID, since, expected_end null, heartbeat and the rule it imposes. Store
   a private state file outside the checkout containing `checkout`, `hold_file`,
   `window`, `restore_commit`, `restore_profile`, `restore_images`, `container`,
   configured `nodes` (name and SSH target), `active_profile`, `stopped` and
   `restored`. Keep an immutable copy of `restore_window.py` beside that state.
4. Start that script detached with `--state STATE --watch --apply`. It restores
   after a heartbeat older than fifteen minutes, checks exact ownership, refuses
   changed shapes, restores the captured commit before starting the captured
   service, verifies the original image on every rank and runs doctor. It removes
   the hold only after success. A management failure retains the reservation;
   do not retry a launch without isolation and renewed authorization.
5. Inspect each cluster plan, especially its live and shape lines, before apply.
   Stop the captured service with its own profile; set `stopped=true` in state
   after success. Use `bin/spark --cluster-config <committed experiment profile>
   cluster sync --revision <exact published commit> --apply`. All node checkouts
   must be clean and detached at that same commit. `deployment.branch` names
   `feat/http-cancellation`; no main-syncing queue runner may operate this window.
6. Before each start, record `active_profile` and `stopped=false` in the private
   state. Start the arm through `bin/spark --cluster-config PROFILE cluster start
   --replace --apply`, with startup/steady memory guards. Never bypass a shape
   refusal. Call the guarded client below, then the unchanged standard screen.
7. Refresh the hold heartbeat during coordinated operations with the copied
   restore helper's `--beat --apply`; `guarded_run.py` refreshes it during its
   workload. Recheck exact ownership before every operation. When another request
   arrives, finish the active arm and restore; retain completed arms and resume
   only missing arms in a later window. Do not remeasure a completed successful
   arm without a new concern.
8. Restore once at the end of the window with the private restore helper
   `--state STATE --apply`. It stops the current arm, synchronizes the captured
   original commit, starts the captured profile, verifies every original image
   and doctor, then releases only its own hold. Keep every receipt and failure.

The root controller must not sync or start while the hold says `restoring` or
`management_loss_latched`. A managed benchmark failure is not permission to
bypass startup guards. Two negative tests verify that a foreign hold cannot be
changed/restored and a changed shape never reaches apply.

## Commands within an owned window

For each profile/label pair (`hardware-candidate-tp3.json` / `candidate-a`,
`hardware-parent-tp3.json` / `parent`, then candidate / `candidate-b`), run from
the actual synchronized head checkout:

```sh
python3 experiments/2026-11-06-http-cancellation-cost/hardware/guarded_run.py \
  --cluster-config experiments/2026-11-06-http-cancellation-cost/hardware-candidate-tp3.json \
  --window WINDOW --arm candidate --boot candidate-a \
  --output results/private/http-cancellation-hardware/candidate-a --apply

bin/spark --cluster-config experiments/2026-11-06-http-cancellation-cost/hardware-candidate-tp3.json \
  bench --suite screen-tp3 --compare none --label candidate-a \
  --output results/private/suites/http-cancellation-hardware/candidate-a
```

`guarded_run.py` checks the live identity and mounted source, cools all nodes below
55 C, requires the performance fan curves, and uses the existing `MemoryWatch`
for thermal and memory limits. The CPU-only client is limited to two CPUs on
cores 0-3, outside the serving cores, and 4 GiB. Preserve `conditions.json`,
identity, raw metrics, request hashes, every command and engine log on failure.
The standard screen has its own identical conditions machinery. During that
external command the root controller must keep the window heartbeat fresh.
