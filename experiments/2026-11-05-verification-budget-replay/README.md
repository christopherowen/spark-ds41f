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
