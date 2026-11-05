# Verification budget replay

Base deployment: `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`.
Intended variable: the denominator of the expected-token objective, using
recorded step costs instead of the linear 0.035 row cost. This is an offline
screen; no runtime policy, configuration, arithmetic or source series changes.

The existing allocator already ranks survival across requests. Candidate 15
tried graph/tile/dead-row pricing and candidate 16 removed it after regressions.
A revisit requires a new measured reason. Aggregate acceptance counters cannot
stand in for per-request confidence vectors or counterfactual token matches.

`replay.py` compares the two objectives at the *same measured grid cells*,
without interpolating absent graph or tile boundaries. Its restricted linear
control is not the unrestricted serving policy. The historical five-draft
trace is from r4a, before current checkpoint arithmetic, dead rows and copy
drafts. Pairing it with later grids is cost sensitivity, not a speedup verdict.
The script records missing c16 coverage and the fraction of unrestricted
linear selections whose row count is actually measured.

## Execution record

- Read the required repository guides, current lock and immutable 1030 baseline.
- Recovered the existing r4a trace and its benchmark/cost receipts from dgx1
  with read-only copies. Raw token records remain in ignored storage.
- Initial local source preparation ran out of disk; it did not complete.
  Initial replay failed because NumPy was absent; its dependency setup also
  ran out of disk. Both replay error logs are retained under `runs/`.
- Initial local doctor used the example node map and failed the head-address
  check; `runs/doctor.txt` preserves this error.
- The owner then required resource-intensive work to run remotely. No local
  job survived the workstation restart. Further source preparation, environments,
  replay and tests run on Spaceport sp8, in `~/spark-lab/verification-budget-replay`,
  under a separate CPU reservation, two-core/4-GiB limits. Serving operations
  remain unauthorized and are not performed.

## Results and decision

**Retain the current policy. No supported serving improvement was found.**
The existing global survival allocation and ratio objective are present in the
freshly reconstructed candidate-40 source. Candidate 38 and 40 have byte-identical
`adaptive_verification.py` (SHA-256 `54c2e186cdc17aea950d3f8d6549b1f463cc780c836dac75f6c33eaf9d45c7d1`).
Candidate 38 removed candidate 36's rank-local proposal limit from host selection:
eligible slots now come from the scheduler's shared drafts. Preserve that fix.
`runs/source-inspection.json` records all three locks and the reconstructed head
and tree. No source fork or deployment patch is proposed.

The recovered trace exactly reproduces **19,654/19,654** logged acceptance
decisions. After excluding startup records it contains 5,618 steps; 5,402 can
be scored (216 prefill/terminal steps excluded), including 1,610 at c1, 1,191
at c4 and 1,219 at c8. It contains no c16 steps. Workload: r4a's quality gate
and prose/code/JSON, reasoning on/off, greedy, at c1/c2/c4/c8. No sampled or
agent trace is manufactured from these records.

The cell-restricted cost substitution below holds the same drafts, matches,
ranker and eligible row counts on both sides. These are **modeled ratios on
old observations**, not measured throughput changes, confidence intervals,
or comparisons with the actual unrestricted serving control.

| Concurrency | Measured rows at 4K | Unrestricted linear choices covered | 1030 TP4 grid | C40 TP3 grid | C40 repeat grid |
| --- | --- | ---: | ---: | ---: | ---: |
| c1 | 2, 3, 6 | 831/1,610 (51.6%) | -0.075% | -0.078% | -0.076% |
| c4 | 8, 12, 24 | 212/1,191 (17.8%) | -0.015% | +3.119% | +3.130% |
| c8 | 16, 24, 48 | 99/1,219 (8.1%) | +1.341% | +3.288% | +3.312% |
| c16 | 32, 48, 96 on 1030 only | no trace | unsupported | no grid/trace | no grid/trace |

Both C40 grid files are existing boot and repeat-boot receipts; no released
build was remeasured. The two cost inputs' agreement does not create a confidence
interval for the unobserved counterfactual policy. The trace also predates
current checkpoint arithmetic and does not record context lengths, so even
the 4K workload match is unknown. The sparse cost coverage prevents the
apparent c4/c8 gains from supporting a runtime change.

### Costs that must remain distinct

- **Graph padding:** the grid's `num_tokens` is the requested row count, not
  a measurement of every adjacent graph boundary. Snapping it to another
  bucket and charging those rows again can double-count padding. MoE skips
  padding; dense work does not necessarily do so.
- **Dense tiles:** split-K decode uses 16/32/64/128 buckets, with different
  wide-decode tile choices by projection. A universal per-16-row surcharge
  does not describe this dispatch. No adjacent-boundary cost is imputed.
- **Head tiles:** current head row tiles include 128. The source states that
  65–128 verification rows read the packed weight shard once in that tile.
  Reusing the former per-64-row traffic cliff would be stale.
- **Dead rows:** host selection uses stale/planned confidence; device marking
  uses the current proposal. The current grid has dummy full-live rows and
  cannot price changing the live-row count or expert union. The old trace
  predates dead rows. No stale host cutoff or estimated union saving is added.
- **Content and sampling:** a trace can score only the observed draft prefixes.
  It does not reproduce the new policy's future batches, current CopyDrafter,
  seeded sampling, grammar/tool behavior, or per-request fairness.

Candidate 15's pricing work was dropped in candidate 16 after TP3 regressions:
single-stream -2.84% ±0.12%, Poisson -1.49% ±0.30%, and sampled c16/c8/c1
-4.27%/-3.12%/-2.90% (the combined policy arm; not isolated causal attribution).
Its follow-up audit found artificial tile cliffs and stale host pruning.
The older static-table experiment also found no throughput advantage over
0.035. This screen provides no specific measured reason to override those
decisions. Relevant records: candidate README's candidate-15/16 sections,
PR #693's qualification comments, and
`experiments/2026-10-18-dspark-static-costs/README.md`.

Current agent evidence is materially different from this old trace. The saved
candidate-38 Spaceport c8 replay completed 765 requests in one hour, with
1.657 accepted and 2.742 verified drafts per draft event in `server-workload.json`.
Its c16 point stopped thermally during warmup and repeat boots did not run.
It has aggregate counters, not the per-step confidences needed by this screen.
See `experiments/2026-11-04-release-trace-replay/README.md`. No current
agent-traffic or c16 policy validation is claimed.

## Completion audit, 2026-11-06

The stronger completion requirement does **not** turn this into a qualified
performance implementation. It supports a narrower conclusion: the proposed
cross-request allocator mechanism already exists and is optimal for its stated
fixed-score, row-only-cost objective. Better cost or confidence estimates remain
promising and unqualified. Retaining the current policy is an evidence decision,
not proof that 0.035 is the fastest possible serving parameter.

For request `i`, let `s(i,j)` be the product of conditional confidences through
draft `j`. With confidences in [0,1], these scores are nonincreasing within a
request. At fixed total budget `K`, the globally largest `K` scores maximize
their sum. Ties can be ordered by request then position, preserving prefixes.
Any other feasible prefix allocation has no larger sum. When cost depends only
on total rows, maximizing this sum for every `K` and then taking the best
expected-token/cost ratio exhausts the choices in that model. Equal allocation,
independent request caps, or a different shared-pool implementation cannot add
expected tokens under these assumptions. This argument does not cover
miscalibrated/stale confidences, request-specific costs, fairness objectives,
future batching, or floating-point equivalence between CPU and GPU.

`audit.py` checks the exact pinned C38/C40 CPU `get_num_tokens` method (extracted
without importing the GPU runtime, with an enforced whole-file SHA-256).
It passes **160 exhaustive prefix comparisons**, each also checked with reversed
request order, and **5 boundary checks**: 17-slot copy proposals, scheduler
eligibility despite a zero rank-local limit, all-zero confidence, periodic
exploration, and a zero logits allowance. The exhaustive cases include ties,
variable capacities, logits clipping and mixed prefill. These are host-selector
checks; they do not claim native GPU arithmetic or generated-output invariance.

The additional screen varies only row cost: 0.01, 0.02, 0.035, 0.05 and 0.07,
plus the already screened measured-cell denominator. All choices retain the
same confidence ranker and measured row cells. For each frozen step an oracle
can also choose a budget knowing the eventual matches; a second oracle can
allocate those rows knowing which drafts match. A finite fractional optimization
maximizes **total tokens / total modeled time** over all frozen steps. Merely
maximizing each step's realized ratio would not give that bound; an independent
exhaustive test and a counterexample check this distinction.

| C40 4K cell screen | c1 | c4 | c8 |
| --- | ---: | ---: | ---: |
| Row cost 0.01, modeled change vs restricted 0.035 | -0.285% | +3.200% | +3.164% |
| Row cost 0.02 | -0.049% | +2.619% | +2.169% |
| Row cost 0.05 | -0.207% | -2.227% | -2.659% |
| Row cost 0.07 | -0.819% | -4.405% | -5.271% |
| Perfect future knowledge, budget only: model headroom bound | 3.027% | 5.368% | 5.027% |
| Perfect future knowledge, budget and allocation: model bound | 3.027% | 7.779% | 10.292% |

The saved repeat grid gives similar descriptive numbers; all outputs, including
the 1030 grid, are in `runs/alternatives.json`. These bounds apply **only to the
frozen old trace and three measured cells**, not actual serving throughput or
unmeasured budgets. The parameter sweep uses the same observations throughout;
it is not held-out policy selection. At 64K and 256K, recorded request/context
pairs have only one measured row count, so every denominator chooses the same
budget: zero difference there is absence of an alternative, not validation.
c16 has no trace at any context. No confidence interval is attached to these
model deltas; two existing cost grids cannot quantify the model's missing data.

### Alternatives and remaining bottleneck

| Alternative | Evidence and decision |
| --- | --- |
| Another global allocator, equal shares, or per-request caps | Already covered by global prefix ranking; zero fixed-score objective headroom at a fixed row-only cost. Preserve scheduler-shared eligibility and exploration. |
| Lower row cost or measured-cell pricing | Still promising on this incomplete model at c4/c8. Not disproved. Historical 0.02 serving runs gave no throughput gain and later first tokens; current 0.01/0.02 require current traces and actual qualification. |
| A fixed survival threshold | Recorded 0.2 serving test lost about 3.1% at c1 and 0.7% at c16; it ignores the batch's base cost (`2026-10-18-dspark-linear-budget`). |
| Full static cost shape, cost scales 1/2/4, depth 3/5, marginal running-rate objective, confidence calibration | The retained historical replay already enumerates these; calibration/marginal changes are small in that model and full static serving results were inconclusive within drift. No current-stack advantage is established. |
| Graph/head/dense boundary pricing | Candidate 15's artificial cliffs regressed. Current dispatch differs, including the 128-row head tile. Adjacent boundary measurements are missing; interpolating them would invent the deciding evidence. |
| Stale host cutoff, fresh dead-row pricing, expert-union costs | Stale pruning previously regressed. Current marking uses fresh confidence; no joint current confidence/live-row/routing cost trace supports new pricing. |
| Fresh host confidence or a learned long-run objective | Potentially changes synchronization, estimator feedback, future batches and fairness. The current trace cannot price those effects or prove seeded/tool behavior. No timing or implementation win is claimed. |

The remaining bottleneck in this investigation is identification: the traces
and costs do not jointly describe current serving work. A faster allocator
cannot fix that. Current per-step confidence/proposal/acceptance/context data,
boundary and live-row costs, and distinct-agent c1/c4/c8/c16 serving runs are
still required to select between the promising denominators. Native arithmetic,
batch invariance, seeded sampling, tools/grammar, fairness, throughput, latency,
GPU memory and failure behavior of such a candidate remain untested. A boot
and repeat boot with intervals against recorded references would be necessary;
serving operations have not been authorized. This is not a fully proved
performance result or an assertion of a global optimum.

The audit used published deployment commit `2a769af2` and the original locked
C40 prepared tree. **11 offline unit checks passed**; source checks and the
alternative screen completed in 2.52 s with 64,448 KiB maximum process RSS,
under one CPU core and a 2-GiB limit on sp8. Exact commands are in
`audit-commands.sh`; native logs and input/script hashes are retained. The
complete diff was reviewed; no runtime patch, arithmetic change, additional
source preparation, image build, GPU operation or serving operation was added.

## Validation and resources

- Remote `doctor`: configuration OK, exit 0. The isolated CPU checkout uses the
  existing DGX node map only to validate the candidate's source configuration;
  it is not a Spaceport launch profile and no live command was executed.
- Fresh full C40 vLLM series application: exit 0, head
  `5f926df6d3126266a9c88c82c06544060008c33c`, tree
  `ca1496c6481dc5bc4b4234b743e4ca023a6c7597`, matching its lock. A pre-existing
  patch blank-line warning is retained in the native log; final diff check passed.
- Offline checks: **7 passed**, including missing-grid refusal, strict
  request/context separation, incomplete terminal-prefix handling, unmeasured
  control coverage, and rejection of corrupt acceptance or invalid costs.
- `build check`: exit 1 because only vLLM sources were prepared; complete image
  inputs are absent. No image build or GPU tests were needed or run for this
  evidence-only change. This is not a successful full build check.
- Under two CPU cores and a 4-GiB scope: original replay 1.93 s / 59,596 KiB
  max process RSS; new replay 0.84 s / 63,592 KiB; fresh source preparation
  32.79 s / 65,840 KiB; tests 0.12 s / 32,880 KiB. No GPU memory was allocated.
  The CPU reservation was released after the checks.
- A receipt-copy attempt used unsupported brace expansion and failed without
  altering the remote receipts; ordinary globs copied the same files afterward.

Exact commands are in `commands.sh`. Raw trace and benchmark receipts stay in
`/home/chris/spark-lab/verification-budget-replay/inputs` on sp8; their identities
are recorded in `base.json`. The checked-in outputs are summaries, diagnostics
and source identities, not raw generated text. Future policy work needs current
trace inputs and boundary costs on the same source, then separate boot/repeat
serving tests at c1/c4/c8/c16 and scored agent traffic. A trace-only run must not
be used for timing, and any serving replacement still requires authorization.

The owner supplied that hardware-testing authorization on 2026-11-06. The
separate `experiments/2026-11-06-verification-budget-hardware/` directory holds
the one-variable runtime profiles and guarded execution plan. This offline
record remains unchanged evidence; hardware results are pending there.
