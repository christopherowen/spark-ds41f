# Collective launch overlap

Status: source [PR #5](https://github.com/christopherowen/dgx-spark-networking/pull/5)
is implemented; CPU checks and 84 SM121 compilations pass. Three published
trigger placements are ready for comparison. GPU ordering, fabric qualification
and performance are pending. No serving change or promotion has been made.

## One variable

The all-reduce and all-gather kernels wait for their programmatic stream
dependency before accessing any tensor or protocol state, including the epoch,
then trigger dependent launches. This permits launch setup to overlap execution.
The following kernel still waits for completion before accessing memory.

The proxy, wire layout, two-slot protocol, resident-grid limit, both rails,
fail-stop timeout/trap paths and fixed rank-order arithmetic are unchanged.
Exchange and reduce-scatter kernels are unchanged. No new runtime buffer is
allocated. The risk to test is scheduling/resource contention and generated
memory ordering, not a different arithmetic formula.

The source starts at candidate-36/38's sparknet `1fd6a1f` / tree `8cc28a5c`,
and the entry candidate pins `fb027b4` / tree `1e1720f0`. The first
`1a1205e` revision was replaced only to correct generated-source inspection
of a forward declaration; the kernel implementation did not change. Candidate-38's vLLM
series was freshly applied: 572 patches reproduce head `1f331025` and tree
`c061e5f6`. [base.json](base.json) records the full identities, and
[candidate.json](candidate.json) the delta. The candidate locks and triangle/ring
profiles are unqualified preparation; neither shape may run this image before
its fabric gates pass. The image has not been built.

## Existing boundaries and the bound

Our SP implementation already exchanges WO output in slices under projection,
exchanges the MoE output under the combine, and passes unreduced rank parts to
mHC, which sums them as it loads them. It gathers the MoE's local MXFP8 values,
scales and routes together. Candidate-38 also contains the direct switched
four-rank exchange from #846; the ring still relays its opposite peer. These
mechanisms are not new changes in this experiment.

The recorded PDL study (`../2026-10-25-pdl/README.md`, remaining work) bounds
joining the one-shot kernels to the chain at 57 microseconds per c1 step,
0.17% on that older trace. This is a historical ceiling, not a candidate gain.
It does not establish a 10k-token/s bulk-prefill path. Large SP transfers keep
the existing NCCL exchange and overlap schedule; no switched-fabric throughput
is being attributed to the ring.

Communication measurements must keep three different quantities separate:

- rank arrival skew, using clock-adjusted doorbells;
- payload progress after posting, including NIC/memory and polling latency;
- exposed waiting on the critical path after accounting for concurrent compute.

The recorded triangle diagnostic in sparknet
`evidence/2026-10-20-wide-read/fabric/diag2/trace-summary.json`, arm `main`,
60 KiB, has rank-0 medians of 16.58 us total, 2.27 us staging, 11.78 us wait and
2.50 us reduction. Pair 0/1 has 0.22 us median doorbell skew and 11.04/10.56 us
post-to-first-observed-flag delays. The clocks are estimated (local calibration
halfwidth 0.37-0.38 us); these delays are not pure wire-transfer time, and
quantile components are not additive. These older diagnostics do not measure
candidate-38, PDL or ring performance. Current candidate traces are required
before attributing a long collective to either transport or a late rank.

## Validation so far

DGX4, CPU-only, `CPUQuota=200%`, `MemoryMax=8G`, `TasksMax=64`, nice 10:

- `make check`: 98 tests pass, including the real C proxy under the sanitized
  simulator; ruff and `git diff --check` pass.
- Deployment unit suite: 755 tests pass in 36.813 s; generated-doc checks pass.
  Its benchmark-client fixtures use local test servers, not a serving cluster.
- Fresh application of candidate-38's vLLM series reproduces its lock exactly.
- Deployment doctor passes for the base and both candidate profiles. Fresh
  sparknet preparation reproduces the exact candidate commit and tree.
- Source CI passes the Python 3.10/3.12/3.14, installed wheel, NCCL patch and
  Dockerfile checks. The receipt links the exact source commit and Actions runs.
- Complete image inputs now pass `build check` at the current source pin. The
  earlier incomplete-input warning remains in the original receipt. No candidate
  image has been built.
- Each of entry, staged and arrived passes 98 CPU checks. All-reduce BF16/F16/F32
  and all-gather compile for every TP3 and TP4 rank: 28 SM121 specializations per
  phase, 84 total. GPU access was explicitly disabled. Generated CUDA records
  the dependency wait before the epoch and the intended trigger placement.
- The synthetic benchmark's two copy kernels compile. Five CPU controller
  checks cover bounded all-rank commands, variant identity, gate-log hashes and
  rejecting a foreign or mismatched window before any hardware action, and
  requiring all lane/error/port counters. Missing counters cannot qualify a gate.
- New GPU tests are written but unrun: delayed PDL producers/consumers,
  cancellation-sensitive rank-order sums, mixed collective slot reuse, eager
  and graph execution, and generated CUDA's wait before the epoch read.

The simulator checks host protocol logic; it cannot qualify GPU memory
ordering or hardware topology. [results.json](results.json) includes every
attempt and limitation. The Mac's interrupted test/preparation attempts are
not counted as results. Runtime/GPU memory observations and confidence
intervals are absent because no candidate boot has run.

[commands.md](commands.md) records the offline commands and resource limits.
The archived scripts retain the executed setup and validation commands.

## Trigger-placement alternatives

[alternatives.json](alternatives.json) pins three published trees: entry triggers
immediately after the dependency wait; staged triggers after posting; arrived
triggers after waiting for peers. Only trigger placement differs between them.
The later placements test whether keeping waiting consumer CTAs out of the
peer-wait interval leaves more capacity for side work. The winning placement
must be measured; early triggering is not assumed to be best.

The three variants emit byte-identical CUDA after removing only the moved
trigger line: 56 alternative/entry comparisons across all 28 specializations.
This supports arithmetic/transport isolation, not a hardware correctness claim.

The bounded [run-source-gates.py](run-source-gates.py) runs the full source GPU
suite and fabric probe on every triangle rank before allowing that variant's
benchmark. It checks exact window ownership, clean source/harness checkouts,
image identity, every rank's verdict and prerequisite log hashes. Containers
have four CPUs, 16 GiB with no extra swap (probe: 12 GiB), process and time limits.
The controller requires 40 GiB free before launch, keeps the existing 24 GiB
memory floor and thermal limits, retains raw memory/thermal samples, cools
below 55 C and maintains the window
heartbeat. Management uncertainty stops further launches and retains the hold.

The synthetic matrix has 7 row counts, all-reduce/all-gather/mixed chains and
0/4/32 MiB of concurrent copy work. Each case has 40 samples of 16 graph chains. External CUDA event-record
nodes inside the captured graph exclude host enqueue gaps from the timer.
Each process must first distinguish short and long captured device delays, so
stale or ineffective timing-event handles cannot supply accepted samples.
That timer check still requires its first GPU execution.
Two fresh process runs reverse the variant order. All rank samples, exact-output
checks, graph/compile warmup, memory and thermal evidence are retained. Background
cases fork a side stream after the producer, so that event orders the producer;
the measured launch chain still includes the collective and its consumer. This
is controlled resource contention, not a model-layer or serving result. CPU bandwidth-throttled samples are rejected; four CPUs leave room for
the proxy, launcher and coordination, while compilation stays at two jobs. A
within-process interval cannot substitute for serving boot/repeat variability.
`analyze_launch_chain.py` retains all cells and reports pointwise Student-t
intervals on the two independent process-round means/differences (df=1), without
pooling replays or automatically choosing a winner. These exploratory intervals
are not adjusted for multiple comparisons.

## Acceptance and workload identity

A concrete correctness repair is also acceptable with demonstrated performance
preservation: an old-fails/new-passes reproducer on the affected path, and
intervals narrow enough to exclude a material regression. Merely including zero
is inconclusive. No pre-existing correctness failure has been established for
this patch: the waits protect the newly enabled overlap, while the earlier
ordinary launches serialize. The current change remains a performance
enhancement. Acceptance therefore requires an incremental gain
beyond the candidate and saved reference repeat margins, exact output identity
where the existing suite promises it, and no material latency, memory or failure
regression. An unchanged arithmetic formula is not a GPU correctness result.
Keep existing sampled batch-dependence as a limitation; do not relabel it as
sampling invariance. No change to the sampler or its scheduling is intended.

The triangle candidate inherits candidate-38 `tp3-dev.json`, including the dev
RPC required by the engine grid. Use the same `release-tp3`,
`single-stream` and `sampled` suite identities as
`../2026-11-03-candidate-38-tp3-qualification/release-default.json`.
The immediate reference is its saved standard boot/repeat under
`../2026-11-03-candidate-38-tp3-qualification/runs/2026-11-03-dgx1-3/performance-default/`.
Keep both candidate boots against both saved candidate-38 boots, with their
separate repeat margins. Also report compatible saved C36 records from that
qualification's `previous-candidates/` receipts and the immutable 1030 suite
references in `manifests/suites/2026-10-30-spark-ds41f-1030-tp3{,-single-stream,-sampled}`.
Verify workload/client identities before comparing; label missing or incompatible
rows. An improvement over 1030 alone cannot establish an effect from this patch.
Do not remeasure any existing image as a control.

## Coordinated hardware plan

The next requested window is **TP3 triangle only**, after the existing holder
releases it. No topology change is requested. At the read-only inspection,
DGX1-3 served candidate-36, image ID
`sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11`,
from `candidate-36/tp3-dev.json`; this is observed state, not a new configuration
authority. Re-read the live profile, hold and node map before any operation.

1. The owner authorized coordinated stop/start/replacement/sync and hardware
   testing. Recheck both the active hold and any waiting request. Use the existing
   published lab runner's idle/publish guards and heartbeat/watchdog; do not
   bypass a guard or take another holder's window. The experiment branch is
   not on main, so its own runner cannot open a window. The authorized runner
   must use its clean published deployment checkout; test source checkouts are
   separate, clean, detached at the exact published source commit on every rank.
2. Inspect `bin/spark --cluster-config <verified-live-profile> cluster stop`
   and its `live:`/`shape:` lines before applying. Run all
   triangle ranks together, one bounded container per rank, after cooling all
   nodes below 55 C. Do not launch a serving rank on its own.
3. From each pinned sparknet source on every rank, run its fleet GPU suite and
   collective probe with the triangle map, both rails, the same source commit,
   and candidate-38's verified existing image. The fleet command is:
   `scripts/gpu-test-fleet.sh MAP oneshot-direct tp3-triangle IMAGE`.
   Set `CHECKOUT` to the verified source checkout, retain per-rank logs and
   failure-injection results, and record the exact map/image hashes. The probe
   command is rendered with `sparknet probe render-command MAP NODE
   --transport oneshot-direct --profile tp3-triangle --image IMAGE --
   --benchmark --counter-samples` for each configured rank. Preserve the normal
   timeout and capture all rail counters. Stop if management is lost.
4. After each variant passes those gates, run the two synthetic process rounds
   before choosing a serving candidate. These runs may reject a variant but do
   not establish a serving gain. Only then prepare/check/build one immutable candidate
   image under the build memory guard; distribute its one OCI digest using
   the coordinated tooling. Review each sync/start plan and its live/shape
   lines. Run the unpromoted triangle candidate for a boot and a repeat boot,
   back to back in the same window. Use the saved candidate-38 repeats for
   attribution and the recorded C36/1030 triangle runs for context, as above.
   Report c1/c4/c8, the engine grid, 32K/131K/262K/524K/1M cold prefill,
   mixed stalls in seconds, greedy/teacher-forced identity, memory minimum,
   errors and 95% intervals. A small positive estimate within boot noise is
   not an accepted performance improvement.
   The committed candidate overlay names `deployment.branch=feat/collective-overlap`;
   `cluster sync --revision <exact published SHA>` permits testing it without a
   merge. Capture the original commit and restore profile, exclude any main-syncing
   queue runner, and validate restoration before changing canonical checkouts.
   Do not weaken the lab's main-publication guard.
5. Restore the pre-window service once through the authorized coordinated
   runner, restore the original deployment commit/profile, verify it, then release the hold. No merge or promotion is part of
   this task. Repeat the fabric and serving gates on ring4 only in a separately
   authorized window when that topology is available; triangle results do not
   qualify the ring.

Conclusion: [retain for hardware evidence](decision.md). No measured speed gain
is claimed.
