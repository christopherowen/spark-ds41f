# Pending hardware qualification

The owner explicitly authorized hardware testing and the necessary coordinated
sync/stop/start/restart operations. No merge, promotion, cabling, topology,
firmware, kernel or driver change is authorized. No operation has yet run for
this experiment. Foreign holds and existing requests retain priority.

## First gate: isolated kernel correctness

After the holder releases the hardware, re-read `~/spark-hold.json` on dgx1,
check the selected node's GPU jobs and memory, and acquire a coordinated window.
Do not stop a serving container to make room. Use an idle qualified Spark with
the existing candidate-38 image digest recorded in README. Publish the source
commit and deployment experiment first; use clean checkouts at those exact
commits. Expose the prepared source file and test file read-only to the test
container, leaving the installed image and all serving containers unchanged.

Run the changed source test and existing bit-equality cases in two independent
processes, retaining both logs and memory/thermal observations:

```sh
.venv/bin/python -m pytest tests/kernels/moe/test_deepseek_v41_tilelang_moe.py \
  -k 'emits_requested_prefill_swizzle or prefill_swizzle_panel_matches_unswizzled_bits' -v
```

Use the image's existing packages through a `uv` system-site-packages environment;
no replacement PyTorch, compiler, or checkpoint is needed. Cap host compilation
at two CPUs and 16 GiB, with `MAX_JOBS=2` and `OMP_NUM_THREADS=2`. Run no GPU job
while another holder is active. The cases cover gather, aligned TP3 down K,
TP4's exact 64-column down tail, both route-ID widths, and prefill sizes 171/500.
If any bit mismatch or native compilation failure occurs, retain it and stop
qualification. The CPU-generated-source test is not a substitute for this gate.

## Second gate: missing route evidence

The current saved captures inspected on dgx1 do not contain the consumed expert
IDs needed here. They contain earlier gate/fingerprint, shared-projection, or
transport data. They cannot establish current route overlap. A new collector
needs both post-mask route IDs at `RoutedExperts.run` and the scheduler's actual
row-to-request/anchor/draft mapping. This integration is not implemented yet;
there is no executable capture profile to approve at this stage.

The collector must preserve native IDs and weights, record actual allocated rows
including graph padding, identify target/drafter and layer, hash the source tree
and checkpoint identity, and record each projection's per-expert scale width.
Keep captures private/node-local and bounded to selected layer invocations.
Validate the row mapping against scheduler metadata, not timing or guessed
request order. Collect at a graph-compatible boundary or use a separately
identified eager capture arm; do not interpret instrumented timing as performance.

Required workload strata are code, prose, and independent agent requests, at
concurrency 1, 8, and 16. Retain distinct prompts and genuinely low-overlap calls;
do not replace them with duplicated code prefixes. Fix the exact workload files,
hashes, seeds, and sampling parameters before a capture run. Capture TP3 and TP4
independently; arithmetic projection of one trace is not topology qualification.
Analyze calls separately with `analyze_routes.py`, then aggregate by workload,
layer, concurrency, and boot. Report distributions of distinct/repeated experts,
occupancy, cross-request overlap, repeated payload, and scale-tail waste.

Before requesting a serving operation, finish and publish that collector, its
arm profiles, and a lab specification; inspect the plans' `live:` and `shape:`
lines. Then use the already authorized coordinated stop/start/restart/sync operations
covering every configured node. Do not use a one-rank launch,
dirty tree, rsync deployment, or `--change-shape` to bypass a refusal. TP4 must
use its qualified node map and hardware window.

## Third gate: exactness and useful performance

Proceed only if observed route occupancy and a kernel screen show an actionable
opportunity. The swizzle fix does not reduce logical byte counts; collect cache
or DRAM counters if claiming physical byte savings. Use a built-once OCI digest
on every rank. Run batch-invariance/trace checks and the native sampling gate,
then candidate boot A and repeat boot B against recorded reference runs. Include
code, prose, distinct concurrent requests, agent traffic, and low-overlap cases;
report confidence intervals, all failures, memory, and thermal observations.
Never rerun a released reference. Restore the authorized production profile once
when the window ends, then release only this task's hold.

Reject the patch as a performance change if gains do not survive these gates.
Any promotion requires the owner's acceptance and a separate immutable baseline.


## Published branch execution

Keep the promoted recipe untouched. `cluster sync --revision <published SHA>`
supports a clean experimental commit; the committed candidate profile can select
`deployment.branch: feat/expert-route-reuse`. Read its `live:` and `shape:` plan
lines, name that profile on every apply, and detach every configured node at the
same published commit. The main-only queue is a separate restriction: do not
merge an experiment or weaken the guard to satisfy it. Before taking a manual
window, ensure no queue runner can race it, record the original deployment
commit and restore profile, and arrange the existing heartbeat/watchdog path.
Restore the original serving profile and commit before closing the window.

The first benchmark uses unreleased same-commit panel zero as the paired causal
control. It is not a rerun of a released baseline. Kernel speedups must survive
the full path: representative cold 32K through 1M prefill, decode concurrency
1/8/16, and distinct agent traffic, with candidate serving boot and repeat boot,
95% intervals, exact workload/source/image identities and cooling/fan receipts.
Use immutable recorded release references for release comparisons. A full-path
regression, neutral result or remaining hardware block must remain explicit.
