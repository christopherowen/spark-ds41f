# Drafter metadata replay

Performance hypothesis: DS41's FULL DFlash query replay can omit the eager Python
metadata rebuild by capturing the existing integer metadata kernel immediately
before the query forward. The sole causal variable is the placement of that
metadata work. No new kernel, draft length, target verification, sampling,
collective, scheduling or copy-draft policy is introduced.

The deployment review is [PR 919](https://github.com/christopherowen/spark-ds41f/pull/919).
The source is [vLLM PR 3](https://github.com/christopherowen/vllm/pull/3), one commit
on candidate-38 (`1f33102547e4da5c64712390604695d1abd577c4`). `base.json` records the
source head, tree, patch fingerprint, deployment base and CPU test image. The
ordered 573-patch series was freshly applied from the locked upstream base and
reproduced the exact candidate head and tree. The
source-fork base branch exists only to keep this review to the single delta.

The DS41 builder exposes a fixed-shape device refresh over persistent buffers.
DFlash captures it before attention, then uses the existing FULL-graph fast path.
Context parallelism, model-state metadata overrides, unsupported builders and
non-FULL execution retain eager metadata construction. Other supported builders
keep their prior behavior. Graph count and compiler specialization keys do not
change by design; startup memory and compilation still need measurement.

Host regressions check refresh-before-query order, mutable buffer contents,
padded capture shapes, mode/opt-in boundaries and fallback guards. The added CUDA
regression compares every readable integer metadata row against eager execution
for paged, compressed and circular caches after changed lengths, padding and slot
reuse, with compilation frozen before capture. All three CUDA cases passed on
each of two fresh boots; 192 additional exact metadata/consumer comparisons
passed per boot. [micro-results.md](micro-results.md) records before/after
timings, alternative placements, the overlap challenge and memory observations.

Historical rank-0 traces in
`experiments/2026-10-26-core-metrics-plan/runs/trace-waits-ports-rank0.txt` show
0.72, 1.98 and 0.79 ms of all host gaps per 34.63, 29.45 and 58.20 ms step
(2.1%, 6.7%, 1.4%). These are an old upper bound, not candidate-38 measurements,
not same-window rank traces and not a speedup prediction for this smaller change.

All heavy preparation, environments, tests and publication after workstation
recovery ran on Spaceport with a two-CPU/8-GiB limit and the shared CPU lock.
`runs/` retains the unsuccessful setup and checks as well as the passing runs.
`commands.md` gives reproduction and limitations; `qualification.md` gives the
remaining gates. `results.json` distinguishes host checks, measured component
results and unmeasured serving outcomes. No source-complete image build or build check has run because
only vLLM inputs were prepared.

Acceptance requires a repeatable incremental serving improvement beyond both
candidate and compatible saved-reference repeat margins, with exact integer
metadata, preserved target sampling/verification, rank agreement and no startup,
cancellation, streaming, deadlock or memory regression. Keep the candidate a
performance experiment; neutral performance is not sufficient to accept it.
The candidate profile has launch disabled. Nothing here approves a cluster
operation, changes the promoted recipe, or updates an immutable baseline.
