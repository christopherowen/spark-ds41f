# Qualification plan — not executed

Candidate source head `8dea43f62a149a3aef2be4b24fa1c68e9b33407d`, tree
`3f9b475dfcd86aabbc042299131cd38aa3612be0`; image tag
`spark-ds41f:verification-shapes-8dea43f62a14` must be built once and its immutable
image ID recorded. Use `tp3-dev.json` for engine measurements. No shape change
is proposed. Do not run these jobs under a foreign hold or jump the existing
norm-kernel request.

1. On a reserved idle remote build host, prepare the complete locked inputs,
   run `build check`. The CPU-only build phase below uses a separately recorded
   two-CPU/24-GiB allocation on SP8, after checking other reservations and host
   capacity. The controller remains capped at one CPU/4 GiB. A controller
   cgroup alone does not cap the Docker daemon's workers. The GPU smoke and
   image gates run only after exclusive GPU window handoff.
   Never run a heavy build on the workstation.
2. Gate that actual image, without source overlays. Run graph-manager, warmup,
   DSpark context, and MXFP8 linear tests. Test odd and boundary rows
   1/3/5/7/11/15/16/17/18/24/30/31/32/33/34/36/42/47/48/51/58/59/63/64/65/68/
   84/85/95/96/97/102/106/107/127/128, subject to the layer's existing bounds.
   Initcheck must see freshly allocated unwritten tails before poisoning them;
   memcheck must see zero invalid accesses. Replay captured graphs after changing
   live inputs and compare exact live results with independent eager references.
   If the separate MXFP8 investigation exposes a pre-existing tail read, stop
   qualification and record it; do not combine fixes without a new experiment.
3. The owner has authorized the coordinated three-node sync, stops, starts,
   repeat boot and restoration needed for these tests. After the image gates and
   exclusive window handoff, resolve this experiment's published deployment commit; all three nodes must be clean and
   detached at that exact commit. Distribute one verified image digest to all
   ranks. Use the normal shared window and watchdog. Inspect each plan's `live:`
   and `shape:` lines; never pass `--change-shape` to bypass a refusal.
4. Run `functional.json`, then `performance.json`, keeping the window open
   between jobs. The performance spec runs the candidate and repeat candidate
   boots only, against existing immutable reference suites. It does not boot a
   released comparison image. Preserve all functional exits, errors, clocks,
   thermal/cooling evidence, per-rank startup free memory, and graph capture-pool
   high-water marks. Confirm no request changes output when batched with other
   requests, including copy and non-copy requests together.
5. Separately screen the native-row-only and union alternatives in `alternatives.md`
   against this proposal, one capture policy per candidate. Include an every-row
   ladder if measured graph memory permits. Preserve each result. On both
   candidate boots, also measure the repaired full-depth request counts
   3/5/6/7 and the exact 16/32/64/96/128 kernel boundaries with the same workload
   and seed. These isolate the mechanism; aggregate c1/c8 results alone do not.
   Run c1/c8/c16 client load with the bench's default cooling guard and 95% CI:

   ```sh
   bin/spark --cluster-config experiments/2026-11-05-verification-shapes/tp3-dev.json bench --suites decode --concurrency 1,8,16 --min-samples 6 --max-samples 40 --precision 2 --output results/private/verification-shapes/boot-1/client-load
   ```

   Repeat into a new `boot-2` directory after the candidate's repeat boot.
   TP3 admits at most eight requests at once: c16 is queued client load, not
   sixteen simultaneously executing target requests. True TP4 c16 requires its
   own already-qualified hardware window and separate explicit topology
   authorization. Do not infer it from TP3.
6. Compare both candidate boots with the saved candidate-40 boots under
   `experiments/2026-11-04-topk-live-qualification` and the immutable release
   reference. Report each workload's two-boot results and confidence intervals,
   acceptance/verified rows, actual target/context padded rows, memory, and any
   regression. Do not average away a failing cell or infer speed from row counts.
   Record the original deployment commit and restore profile before moving any
   checkout. Restore both through coordinated sync/start commands, then verify
   every rank and live doctor before releasing the hold. The previous window recorded
   candidate-36 TP3 development; re-read its authority before any restoration.

The following commands print plans. The owner authorized their test operations;
apply only in this task's exclusively held window after checking `live:`/`shape:`:

```sh
bin/spark --cluster-config experiments/2026-11-05-verification-shapes/tp3-dev.json cluster sync --revision <published-experiment-commit>
bin/spark --cluster-config experiments/2026-11-05-verification-shapes/tp3-dev.json cluster stop
bin/spark --cluster-config experiments/2026-11-05-verification-shapes/tp3-dev.json cluster start
scripts/lab.py run experiments/2026-11-05-verification-shapes/functional.json --dry-run
scripts/lab.py run experiments/2026-11-05-verification-shapes/performance.json --dry-run
```

Acceptance requires exact outputs, no unwritten or invalid accesses, a fitting
memory budget on every rank, and preserved performance within the predeclared comparison bounds. A
speedup is optional for this capture-coverage repair. Otherwise retain the unfavorable evidence and leave the source
PR unadopted. No merge, baseline rewrite, or promotion is authorized by this plan.

## Published branch and exclusive ownership

The root experiment profile restricts `deployment.branch` to
`feat/verification-shapes`; both alternative profiles inherit it. The existing
coordinated CLI accepts `cluster sync --revision <exact published commit>` for
that branch. No merge is needed to test it. Open the window through the normal
published-main and idle guards, then execute the branch test from a separate
clean checkout. Do not use the continuous main-only queue during this window.
Check the exact holder and window ID before every mutation; the lab helper's
shared `spark-lab` prefix is insufficient to identify this task. Keep its
heartbeat and watchdog active. The watchdog does not restore a deployment
checkout revision, so successful closure additionally requires explicitly
restoring the recorded commit before releasing the hold.

## CPU build preparation and strict branch controller

`build_cpu_image.py` separates the CPU image stages from all GPU work. It uses
the exact locked contexts and Dockerfile command printed by `bin/spark build
image`, the normal host-memory start/reserve guards, and the normal build lock.
An isolated, digest-pinned BuildKit worker is limited to two CPUs and 24 GiB
including swap, with one build stage and one compile job at a time. CDI and
device entitlements are disabled; a worker probe must see no GPU device paths
before compilation begins. It runs only on SP8 under a recorded CPU reservation.
The existing 1-CPU/4-GiB controller budget remains separate. This CPU phase may
progress independently of TP3 or DGX4 GPU windows because it cannot access a
GPU and does not change any serving node. It cannot produce a passing gate.
The image is unqualified until the normal GPU smoke and complete applicable
image gates pass in an owned window. All failures remain in new attempt paths.

`branch_round.py` consumes that completed gate receipt; it cannot start with a
CPU-only build receipt. The original production controller must be a separate
clean checkout pinned to the original published main commit. Before opening,
its shape and node map must exactly match the candidate, and every node must
match its revision. Window opening retains the normal publish/live/idle guards.
Each cluster mutation first runs and records its plan, rejects any shape change,
and checks the exact holder/window again before applying. Functional script
failures are fatal. Restoration stops all configured ranks, syncs the saved
revision, starts the saved profile, checks live doctor and every clean checkout,
then uses normal window closure. A separate recovery process restores after a
controller exit; it refuses foreign ownership and leaves the hold on failure.
The existing watchdog remains active.

The CPU controller tests cover foreign/stale ownership, shape/refusal plans,
window loss between plan and apply, ordered coordinated restoration, and
retaining the hold after failed sync. The first two test attempts exposed a
mutable command-list reference in the test observation; constructing the apply
argv separately fixes the recorded plan identity. Their logs are retained.
No simulated restoration or CPU controller result is serving-hardware evidence.

The isolated worker uses Docker's documented [container driver limits](https://docs.docker.com/build/builders/drivers/docker-container/) and [CDI/parallelism settings](https://docs.docker.com/build/buildkit/toml-configuration/). No daemon-wide configuration or default builder is changed.

The first worker setup stopped before the worker probe or compilation because
Buildx 0.35's managed container driver automatically requested all GPU devices.
The resource/device guard rejected it, and that builder was stopped. Its log
and failed receipt are retained. The CPU-only path now uses the supported
[remote driver over a Docker container connection](https://docs.docker.com/build/builders/drivers/remote/)
with an explicitly created bounded `runc` container, no device requests, and a
direct `buildkitd` entrypoint. The same CDI/device and worker-probe gates remain.
The default builder and host daemon settings are unchanged.

The owner's clarified acceptance criterion permits a demonstrated coverage or
correctness repair with preserved performance; a speedup is not required.
Native exact-shape restoration remains distinct from the optional row-only and
union tuning trials. Report a correctness/coverage reproducer on the affected
path and repeatable latency/throughput/memory preservation. An interval that
merely contains zero while allowing a meaningful regression is inconclusive.


## Predeclared preservation checks

For the capture-coverage repair, use independent boot-level observations and
retain all reference/candidate boot pairings. The material regression bounds
are the existing comparison defaults: 3% for serving metrics, 0.5% for the
engine grid. Report repeat ranges separately; these bounds do not turn a
component gain into a serving claim. `preservation.py` requires the adverse
95% boot-level interval and every observed boot pairing to stay inside the
applicable bound. A wide interval, missing reference repeat, incompatible
workload identity, or absent affected cell is inconclusive. Do not count
within-boot replays as independent boots. Preserve memory/correctness failures
regardless of timing.

The GPU screen uses `test_graph_replay.py` from a read-only test mount against
the actual image's unchanged source. It first checks native context preparation
and MXFP8 boundary replay, then memcheck/initcheck with CUDA allocator caching
disabled. Unwritten source/activation tails precede poisoned-tail replay.
This is a kernel-path gate, not full-model graph proof. A pre-existing tail
read blocks this gate and is recorded separately; no prototype kernel is
silently substituted. After that screen survives, `gpu_gate.py` runs every
applicable release test against the image. It uses the normal GPU lock,
exact holder/window checks, cooling/fan/memory guards, and two-CPU/32-GiB
test containers. Every stage records exact commands and outcomes.

`boundary_grid.py` measures the real server's dummy model step at native
depth for the repaired request counts 3/5/6/7 at 4K and 64K, with standard cooling and
condition receipts. These measurements expose the repaired 3/5/6/7-request
cells; they contain no draft acceptance or scheduler effects. Real serving
suites and distinct requests remain separate mandatory evidence.

The CPU image build records its deployment commit. Qualification-only scripts,
docs and evidence may be committed afterwards without rebuilding: the gate
refuses any other post-build change, checks ancestry, original build inspection,
all source labels, ordered image layers and runtime configuration. Docker
stores may use different local image IDs for identical content. The serving
controller checks content identity and records its own local ID for runtime
collection.

The first hardware turn is a separate `gpu_gate.py --screen-only` run, expected
to take 20–30 minutes and bounded at roughly 45 minutes including cooling:
two minutes for smoke, ten each for replay/memcheck/initcheck, plus ten minutes
for cooling and bounded process cleanup. It cannot produce a passing full gate.
A later turn may use `--screen-receipt` to reuse that exact successful screen
without rerunning it; unchanged execution commit, image content, required checks
and all four log hashes are mandatory. The full release tests remain a separate
later job (the saved C40 gate took 36 minutes; this two-CPU run has a three-hour
hard cap). Restore/release at the agreed handoff after each job. No acquisition
waiter is installed; the recorded coordinator order takes precedence over
request-file races.


The repaired request-count cells have no matching saved C40 grid. If a causal
comparison needs a new control, use only that unreleased parent and the newly
specified cells under the owner's explicit control-measurement authorization.
Keep both control boots and their exact identities, and label them new
workloads; existing C40 and released workload measurements are never refreshed.
The candidate's separate client-load extra records c1/c8/c16 on both performance
boots, in addition to the fixed release, sampled and distinct-request checks.

`gate_window.py` runs one bounded remote gate unit from a clean published
checkout. Its separate guardian stops that exact unit/container before checking
the untouched production service and releasing the hold. The gate process
also monitors exact ownership and has a systemd runtime cap. A failed remote
cleanup or restoration retains the hold. This is a job runner, not an automatic
lease-acquisition waiter; invoke it only after the recorded handoff.

Legacy determinism and sampled-digest probes return process status zero even
when their JSON verdict reports differences. The branch runner explicitly
requires deterministic token/logprob invariance and the requested logprob mode.
For seeded sampling, it preserves the C40 qualification contract: the isolated
repeat must match; concurrent-batch trajectories are observations because RNG
inputs include the local speculative step. A sampled batch difference remains
a limitation, never a passing batch-invariance claim. Missing, ambiguous or
failed required evidence stops before performance and follows normal restoration.
This contract is declared before any hardware run; collector output is unchanged.
The focused shared-runner validation work belongs to the verification-budget
experiment; this check is confined to this experiment's branch controller.

The boundary grid rejects missing/truncated timing arrays and requires every
recorded step to use the full graph and reach the drafter at the requested live
dimensions. Its existing mode fields report live rows/requests, not padded
capture descriptors; those fields alone do not prove an exact capture size.
Keep capture-policy evidence and measured model timings distinct.
