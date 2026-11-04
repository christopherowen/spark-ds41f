# Qualification packet

Purpose declared before measurement: remove optional overlap complexity while
preserving correctness and avoiding a material incremental performance
regression. The focused rework and its withdrawn unmeasured event-reuse proposal
are recorded in README.md and source-review.json. No serving-speedup claim is
made. C39/C40 source, images and measured evidence remain unchanged.

C41 source is the exact C40 series plus one standalone follow-on patch:

- vLLM head: `38ade7f947cb09c550c2bce077286e467c7fed8d`.
- Tree: `df6b6bc6841c1126b37be652c0e93c13debb8f42`.
- Ordered patchset: `3c6e010aab24a6de5c2c543e7c3fbc69abb491e552f3b15bd7b82c36137dd7ef`, 575 patches.
- Actual image digest: unknown until the new image passes its gates.

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

No C41 image, GPU or serving work has started. The packet awaits normal shared
scheduling. No promotion or bypass of outstanding TP4 release gates is implied.
The recurring schedule remains paused; continue with direct waiting here.
