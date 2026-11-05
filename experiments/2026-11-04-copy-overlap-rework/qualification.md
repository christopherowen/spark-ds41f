# Qualification packet

Current execution: attempt 2 is active from the unchanged completed image gate.
Do not launch another controller, synchronize active nodes, rebuild/re-gate or
repeat a serving point. Execution commit remains
`0da35cc9668ff491065446b2a842fe996a73e80f`; later evidence commits are not runtime
authority. Performance remains pending.

Purpose declared before measurement: remove optional overlap complexity while
preserving correctness and avoiding a material incremental performance
regression. The focused rework and its withdrawn unmeasured event-reuse proposal
are recorded in README.md and source-review.json. No serving-speedup claim is
made. C39/C40 source, images and measured evidence remain unchanged.

C41 source is the exact C40 series plus one standalone follow-on patch:

- vLLM head: `38ade7f947cb09c550c2bce077286e467c7fed8d`.
- Tree: `df6b6bc6841c1126b37be652c0e93c13debb8f42`.
- Ordered patchset: `3c6e010aab24a6de5c2c543e7c3fbc69abb491e552f3b15bd7b82c36137dd7ef`, 575 patches.
- Native qualified image digest: `sha256:38ff08bb1eb2495cb934e008b6f31a482aaa238c08b959aaf72cf2cb5658d29c`.
  All eight native image gates passed. Independent local audit verified all 30
  original builder artifacts, exact 136-file coverage, 14 XML reports and both
  changed GPU regressions. The initialization log binds the actual native
  source path/hash and zero errors, with no source overlay.

Fresh canonical replay and the repository's independent `build prepare --only
vllm` reproduce the head/tree. All 19 source-review artifact hashes verify.
Both actual C41 offline profile doctors, eight control rejection tests, Python
3.12 parsing and final source whitespace checks pass. The original preparation
log retains an intermediate historical blank-line-at-EOF warning. All four
rendered profiles match C40 except the declared image/source identity and notes;
model, draft policy, weights, topology, serving arguments and hardware policy
are unchanged. Full Linux preparation/build check remains mandatory in the
owned guarded builder.

Other work subsequently changed the promoted image-admission default to 67.
C41 explicitly pins the recorded C40 limit of four images per request, preserving
this experiment's original input surface. All four rendered profiles were
revalidated against the saved pre-refresh C40 renders. The global default and
the other owner's work remain untouched.

Publish one exact deployment commit before source staging. Respect actual
shared holds/requests; no GPU work or synchronization under another holder.
Use the named C36 restoration profile, read the live/shape and sync plans,
require clean identical serving checkouts and owned DGX4 build worktree at that
published commit. Source staging closes without restarting an already-matching
C36 service. Do not use rsync, a dirty source tree or a shape bypass.

Launch exactly one `python3 -u experiments/2026-11-04-copy-overlap-rework/run_round.py --apply`
on DGX1 after staging and fresh idle/reservation checks. It builds once on DGX4,
requires all eight image/release/actual-source initialization gates before
seeding/distribution, audits all 40 functional exits, runs unchanged full
release-tp3/single-stream/sampled families on a boot and repeat, and restores
C36 once. No concurrent DGX4 GPU work during TP3 timing. A completed prefunctional
image gate can be reused only through the tested safe scheduled-handoff resume
path, with a distinct attempt and conservative explicit review. No failed gate,
functional point or metric arm is repeated.

Native controller output will be `results/private/lab/candidate-41-round/receipt.json`
on DGX1, with an attempt suffix only for a later distinct safe handoff. Builder
output will be `results/private/candidate-rounds/candidate-41-image/receipt.json`
in the owned DGX4 checkout. Preserve true launch PID, detached supervisor, actual
wait exit and original logs under a fresh `tl-logs/copy-overlap-rework/setup/`.
Do not substitute C40's digest or overwrite a prior image/output tag.

After completion verify every native outcome, both complete performance boots,
actual supervisor exit, named live doctor and all-rank restoration to C36 image
`sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11`.
Collect and hash the complete portable snapshot, analyze locally, retain all
adverse rows and publish original failed verdicts if any. Show 1030, C36,
previous C40 and current C41, with both C41/C40 boot pairings and reference and
candidate repeat floors. C36's one compatible saved boot is a stated limit.
No old image is remeasured. Restore first, then release shared capacity.

The matching request was fulfilled and guarded source staging closed with all
four checkouts clean at the frozen published execution commit, without a C36
restart. The sole attempt-1 controller launched on 2026-11-05 at 06:29:27 UTC,
opening window `dgx1-1793860208` at 06:30:08 UTC. A fresh compact source recheck
confirmed all four checkouts still match. The eight image gates finished at
07:17:01 UTC: 8,075 tests passed, 360 skipped, and strict actual-source top-k
initialization checking reported zero errors. The controller then honored the
queued lossless-memory request before seeding or any serving phase. Its original
`RuntimeError: window changed or closure requested`, restored=true and actual
wait exit 1 at 07:17:20 UTC are preserved as scheduling history, not a measured
candidate failure. Both original processes are absent.

The independent named live doctor exited zero and all three original C36
containers remained running at the expected image/profile with restart count
zero at 07:18:53 UTC. Eleven portable head-side artifacts were downloaded and
hashed locally under `runs/2026-11-05-dgx/attempt-1/handoff`; original logs remain
private and their hashes are published. The tested resume validator accepts the
exact original handoff and image receipts. Full original builder evidence and
actual DGX4 image identity passed the separate local audit before resume.

The foreign short window was `dgx1-1793863306`. Its original request was left
untouched; after it acquired its own window and fulfilled that request, our
return request was atomically queued at 07:22:10 UTC. Do not stage or launch
under that foreign window. Once it releases, re-read actual holds/requests,
fulfill only the exact matching return request, verify clean source/restoration
and stage the frozen published execution commit if needed. Resume with attempt
2 and the original completed gate; never rebuild, re-gate or repeat a serving
point. At that scheduling boundary, no C41 functional or performance
measurement had started.

The foreign slot released normally; the exact return request was fulfilled and
short source-staging window `dgx1-1793864305` opened at 07:38:25 UTC. Named plans
showed existing C36 live with unchanged TP3 shape. Git-based cluster sync placed
all three serving checkouts at the frozen execution commit; the owned DGX4
worktree already matched. All four were clean, DGX4 was idle and the actual
image/original gate receipt were unchanged. The named C36 doctor passed, with
all running containers unchanged across this staging. It closed at 07:40:52 UTC
without a restart or clock change.

The sole resumed controller launched at 07:42:15 UTC: PID 190054, supervisor
190053. Window `dgx1-1793864574` opened at 07:42:54 UTC. At the compact 07:44:21
UTC check, both processes and heartbeat were healthy, no request or final exit
existed, and identical-image seeding was active. No gate was repeated. The native
receipt is `results/private/lab/candidate-41-round-attempt-2/receipt.json`; actual
wait exit will appear only at completion in
`/home/swank/tl-logs/copy-overlap-rework/setup/exit-attempt-2.json`. Twelve original
operator files and hashes are retained in
`runs/2026-11-05-dgx/attempt-2/operator-setup`. Direct waiting uses compact native
state only; serving metrics are collected and analyzed locally after both boots
and independent final C36 restoration verification.

An earlier operator command failed local Python parsing before any SSH/native
controller action; absence checks and the original error record are retained.
The corrected transport locally parsed both supervisor and remote program before
its single launch. This is operator setup history, not candidate data.

No promotion or bypass of outstanding TP4 release gates is implied. The recurring
schedule remains paused; continue with direct waiting here.
