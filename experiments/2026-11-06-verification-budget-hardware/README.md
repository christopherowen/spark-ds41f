# Verification row-cost hardware qualification

Base: `6d152f4a15ed539f402fc264bc089cf77f4f94fb`. The owner authorized hardware
testing on 2026-11-06, including coordinated replacement and restoration. This
extends the offline `2026-11-05-verification-budget-replay` experiment. No
hardware result or speedup is claimed yet.

One causal variable: `adaptive_verification_row_cost`, 0.035 control versus
0.01 and 0.02. Every arm uses the same existing immutable C40 image, source
lock, native arithmetic, CopyDrafter settings, TP3 triangle, eight active slots,
KV budget and 6,144-token prefill chunks. The profiles name the published
experiment branch for deployment without changing the promoted recipe.
`base.json` records the image, source and reference identities. Read-only
preflight found the exact image on dgx1, dgx2 and dgx3.

## Shared window and deployment

The observed dgx1 hold belongs to the frontend lifetime qualification,
window `dgx1-1793917240`; `norm-projection-boundaries` has the existing request.
Neither file may be overwritten. The native lab queue had no pending/running
specs. This experiment waits behind that handoff; it must not run through the
generic `spark-lab` holder-prefix check while another task owns the window.
Prefill work can take priority while this decode experiment is waiting.

The main-only queue's sync step cannot deploy this PR. Use the existing branch
deployment workflow in a separately owned window, without modifying its guard:

1. After the current holder and pending request finish and production is
   restored, inspect the hold/request files again, check no queue runner can
   race us, and capture the original clean deployment SHA on every node.
   Require one identical SHA on the three nodes. Capture the live profile and
   image as well. The currently recorded restore profile is
   `experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json`.
   A different live restore profile requires reconciling this runner first.
2. From that original clean main checkout on dgx1, open a native lab window
   with note `verification budget hardware` and the named restore profile.
   The native publish/live/idle guards and restoration watchdog remain active.
3. From a clean isolated checkout at the exact published experiment SHA on
   dgx1, inspect the candidate's coordinated sync plan, then
   `bin/spark --cluster-config <experiment>/control.json cluster sync
   --revision <exact published SHA> --apply`. Every configured node must be
   clean, detached at that SHA, and hold the recorded image digest. No rsync,
   dirty-file deployment, merge, topology change or one-rank launch is used.
4. In the canonical deployment checkout run `run_hardware.py --stage screen
   --window <owned window ID> --restore-commit <captured original SHA>
   --apply --keep-open`, detached with all streams redirected. The runner
   verifies the exact holder, window and restore profile, refreshes heartbeat,
   checks live/shape plan evidence before every start/stop, records memory
   through startup, and uses the native thermal/memory/fan guards for traffic.
5. Review the causal screen before selecting `qualify010` or `qualify020`.
   Those stages run release-tp3, single-stream and sampled on two fresh boots;
   comparisons use the saved C40 parent boot/repeat and the recorded release,
   never a remeasurement of a released build. Qualified distinct-agent traffic
   remains a separate required stage before any gain claim is complete.
6. Keep the window held between stages; restore only once when finished or
   handing off. Without `--keep-open`, or on failure, the runner stops the
   current arm, syncs all nodes to the captured original SHA through the
   named restore profile, restores that profile, verifies it with the native
   close operation, and then removes its hold. `--stage restore` performs
   that cleanup if the screen yields no candidate. Do not leave an idle
   owned window without heartbeat; the native watchdog remains the fallback.

Run `run_hardware.py` without `--apply` for a read-only plan. The screen spec
also validates through `scripts/lab.py run <experiment>/screen.json --dry-run`,
but **do not execute that spec with the main-syncing queue**. The guarded
runner adds the functional gates and exact restoration sequence.

## Workload and decision

The first stage uses six fresh boots in order 0.01, 0.035, 0.02, 0.02, 0.035,
0.01, retaining both candidate alternatives and two control boots. Its custom
suite is a new workload, with 128 model prompts followed by distinct closed
loops at c1/c4/c8/c16, each with 60 seconds warmup and 300 seconds measured.
c16 means sixteen clients queueing into the unchanged eight active slots,
not a change to active capacity. The first arm's model corpus is scored by
later arms. The control is explicitly an unreleased C40 causal control on this
new workload; it is not a rerun of the released 1030 reference.

Before timing each arm, require the existing wide greedy token/logprob gate,
seeded temperature-1 batch/repeat gate, and controls/tool/grammar digests.
False summary verdicts fail even when an old script exits zero. Signatures
must agree across arms/boots; subsequent qualification also compares to the
screen's control signatures. Native logs, source hashes, image identities,
run commands, cooling/fan state, thermal and memory observations stay under
`results/private/lab/verification-budget-hardware/<window>/<stage>/`.
Preserve all unsuccessful arms and guard failures; never silently restart an
already-recorded stage.

Compare both boots with the two control boots using prompt-paired results and
boot variability. Report throughput, completion/errors, per-output-token
latency and TTFT (including tail latency), admitted/accepted drafts and memory.
A cost setting needs a material throughput improvement beyond its 95% interval
without worse correctness, c1/queued c16 latency, or error behavior. A neutral,
negative or inconclusive screen does not justify promotion or a claimed win.

Any surviving setting needs the standard release-tp3 prefill points at 32K,
128K, 256K, 512K and 1,000,000 cold tokens, stalls, sampled c1/c8/c16, and
realistic distinct agent traffic at c8/c16 across boot/repeat. The row-cost
change cannot accelerate pure prefill kernels; cold prefill is a regression
guard, not a route to the 10k prefill target. Agent replay needs an adequately
provisioned independent CPU client: do not squeeze a 12-GiB client into the
TP3 head's serving memory margin. Prepare and validate that stage only after
a promising screen, with the same corpus/revision/seed for before and after.

The final evidence table must include metric, before, after, absolute and
percentage delta, 95% interval, and native receipt links. CPU/offline results
remain separately labelled. Until actual hardware runs finish, the status is
preparation or pending shared-window access, never a proved improvement.
