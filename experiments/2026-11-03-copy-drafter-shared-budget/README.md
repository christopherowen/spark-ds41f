# Copy drafter: one budget for every rank (#852)

Owner, 2026-11-03: "I don't like the performance loss, we should remove the
device-side fix and do this correctly." Then: "get to the point you are ready for
testing, stop and show me where it is."

**Status:** the patch is unit-tested in candidate-37's image on sp8, and the repro
has run on sp5–sp7 (window `spark-0013-1793667996`, run `shared-budget-repro-sp`).
- **#852 reproduced on candidate-37** at min match 8 with dead rows off.
- **The minimal arm ran the same sequence clean**, its ranks' layout CRCs equal in
  every window.

Neither of the two mechanisms considered so far explains the crash (*The repro on
sp5–sp7*). The cost run (`lab-cost.json`) and the parent session's diagnosis job
(`lab-diag.json`) are queued, with no results yet. Not promoted; no recipe change.

This supersedes both earlier fixes, #856's host wait
([experiments/2026-11-02-copy-drafter-limit-sync](../2026-11-02-copy-drafter-limit-sync/))
and #859's device clamp
([experiments/2026-11-02-copy-drafter-device-limit](../2026-11-02-copy-drafter-device-limit/)).

- **Base:**
  - deployment: `origin/main` at `da1cfbec`;
  - vLLM: candidate-37's patch head `5b7cfa37f` (tree `af0fb152`);
  - image: `spark-ds41f:candidate-37` (`sha256:e1ee7b1953b1…` on sp5, sp6, sp7 and
    sp8).
- **One variable:** the adaptive verification budget no longer reads the copy
  drafter's host draft limits (`slot_draft_limits=None` at the manager). Everything
  else is candidate-37's: placeholder drafts, rejection, sampling, dead rows, and the
  copy machinery.
- **Patch 1, the one measured:**
  [vllm/0001-fix-copy-drafts-budget-verification-from-what-every-.patch](vllm/0001-fix-copy-drafts-budget-verification-from-what-every-.patch).
  It is vLLM `72abb31ff` on `5b7cfa37f`, tree `58882ab9`. Its code change is one
  argument in `model_runner.py`; the rest is tests.
- **Patch 2, prepared only:**
  [vllm/0002-cleanup-copy-drafts-remove-the-per-slot-draft-limits.patch](vllm/0002-cleanup-copy-drafts-remove-the-per-slot-draft-limits.patch).
  It is vLLM `88982f34e` on patch 1, tree `f473423c`, and it is not part of this
  measurement. It removes the limits nothing reads any more:
  - the host mirror, its staging ring and event;
  - `sync_slot_draft_limits`, and `add_request`'s reset of the mirror;
  - the limits' copy in `attach_copy_draft_alloc`;
  - the device buffer and `copy_alloc`'s write of it;
  - the manager's `slot_draft_limits` argument.

  The asynchronous delivery of `alloc` to the scheduler stays.
- Both patches apply cleanly to a fresh `5b7cfa37f` tree, with the trees above, and
  `git diff --check` is clean. The branch is `copy-shared-budget` in
  `/tmp/vllm-shared-budget` on sp8.
- **Credit:** Jordi Posthumus (#852, #855) diagnosed the cause: patch 0612's
  non-blocking limit read lets the budget act on an older proposal's limits.

## Rationale

- Since patch 0612, `CopyDrafter.sync_slot_draft_limits` publishes the latest
  proposal's per-slot draft limits only once their copy has landed. Until then it
  keeps an older proposal's.
- The adaptive budget capped each request's scheduled drafts with that mirror.
- Each tensor-parallel rank reads its own mirror, at its own moment, and nothing
  makes the ranks agree on it. So the budget could depend on the rank and on when a
  copy landed: the step's token count, its drafts per request, and the device layout
  built from them.
- The budget's other inputs are rank-wide:
  - the scheduler's draft counts;
  - the confidence snapshot, which `record_confidences` waits for and broadcasts from
    rank 0 (or which is rank-invariant by construction).

  The tests already treat that snapshot this way
  (`test_confidence_snapshot_agrees_before_cpu_and_gpu_budgeting`).
- Without the host limits, every rank lays out the same step, and no host waits.
- #856's wait made the read current at a cost of −0.73% single-stream. #859's clamp
  made the rows past the device limit safe but kept the rank-local read. This keeps
  candidate-37's step and drops only the read.

**Which mechanism: the operator's report.** This is their report, not our
measurement
([#855, comment 6592146119](https://github.com/christopherowen/spark-ds41f/pull/855#issuecomment-6592146119),
2026-11-03 00:32Z). It points at the per-rank layout rather than placeholder rows
routed live.
- Their crashing ring2 candidate-37 arm ran with
  `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2`:
  - inherited from `releases/spark-ds41f-1030/cluster.json`;
  - confirmed in the live ring2 container's environment;
  - no ring2 candidate-37 arm had it unset.

  The crashed boot's own environment was not captured.
- At 0.2 the dead-row gate makes placeholder rows padding, which the router skips.
  So live placeholders should not have reached the gate's assert, yet the arm
  crashed.
- Their `-1` → `0` control, valid ids in those rows, still crashed.
- Both fit ranks that laid out different steps: rows summed across ranks that hold
  different tokens. Neither fits a live placeholder row.
- They withdraw their device-clamp speedup reading (159.90 against 126.05 tok/s:
  two boots, not paired). They call their patch 0004 a precursor.

The caveat: no per-rank diagnostic had run on an unfixed arm, on their ring or ours.
They have offered a paired run on ring2 in a window, which would add a second
topology.

**Our repro since (below) agrees on one point and not the other.**
- On placeholders it agrees: the minimal arm verifies more `-1` rows, routed live
  with dead rows off, and stays clean. So live placeholder rows are not the NaN
  producer here.
- On the layout it disagrees. On candidate-37's crashing arm the three ranks'
  CRCs of the consequential stale reads were equal in every window: the ranks read
  the same stale limits and laid out the same steps. So per-rank divergence did not
  cause this crash either.

The patch's rationale, one rank-wide budget, stands as a design property. The repro
shows it fixes #852 here, but not why.

## Quality implications

- **More placeholder rows.** The budget may verify more rows than the latest
  proposal has. When a proposal narrowed below the rows the scheduler laid out (a
  rejected or lost copy), the step verifies its `-1` placeholders wherever the
  planned confidences pay for them. Verification rejects them, so the committed
  tokens are the target's.
  - With dead rows on (the recipe's `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2`) they have
    confidence 0 and are padding, which the router skips.
  - **With dead rows off they are routed live**, more often than on candidate-37,
    which routed them only on a stale read. `minimal-m8-nodead` did exactly that
    through #852's sequence and 8 and 16 streams, with no non-finite gate row. Live
    placeholder rows are not the producer on sp5–sp7.
  - The extra rows cost compute on the steps after a copy shrinks. The cost spec
    measures it.
- **Stale narrow limits no longer drop drafts.** On sp5–sp7, #856's run had
  candidate-37 at 6.083 tool tokens per step against 6.232 with the wait.
- **Bit-identity.** Where candidate-37's host limit didn't bind (it was current and
  at least the scheduled drafts), every step is bit for bit candidate-37's. Where it
  bound, the step has more rows and the same temperature-0 text.
- **No learning from the extra rows.** DSpark's confidences come from the
  checkpoint's confidence head; `use_acceptance_estimator` is off when the head
  exists. The copy state counts at most a copy's own drafts (0612's min with
  `pending`).
- **Memory:** nothing new is allocated. Patch 2 frees the staging ring and both
  limit buffers, a few `max_num_reqs` int32 buffers.

## Evidence

### Unit tests (sp8, candidate-37's image)

Receipts are in [runs/unit-tests.txt](runs/unit-tests.txt). Each run mounted the
tree's changed files read-only, one container at a time, on 2026-11-03.

| Tests | Candidate-37 | Patch 1 | Patches 1 + 2 |
|---|---|---|---|
| `test_copy_drafts.py`, `test_copy_drafts_scheduler.py`, `test_adaptive_verification.py`, `test_prepare_kernels.py` | 77 passed, 1 skipped | **79 passed, 1 skipped** | **79 passed, 1 skipped** |
| `test_async_verified_draft_counts.py`, `test_gpu_model_runner_v2.py` (`AsyncOutput`) | 20 failed, 22 passed | – | 20 failed, 22 passed, the same 20 |

- The skip is the two-GPU test.
- The 20 runner failures exist on candidate-37 too: their stand-in runners lack
  `num_draft_steps` and `copy_drafter`. They are listed in the receipt.

Patch 1's tests, in `tests/v1/spec_decode/test_copy_drafts.py`:
- `test_ranks_budget_alike_whenever_their_limit_copies_land`: two "ranks" (request
  state, copy drafter and manager each) share histories, proposals and the
  confidence snapshot. One's limit copy has landed when it budgets; the other's is in
  flight.
  - Without host limits, they budget the same token count, drafts per request,
    query starts and logits offsets.
  - With them (candidate-37's wiring), the landed rank caps the copy at its 8 drafts
    and the other at the stale 16, and their layouts differ.
- `test_verification_through_copy_growth_shrinkage_and_slot_reuse`: two requests in
  one slot, decoded 400 tokens each the way the runner does. The pieces are:
  - the scheduler's layout from the previous proposal's allocation;
  - the budget;
  - `reallocate_drafts`;
  - the draft combine;
  - greedy verification.

  Each prompt makes the first copy wrong, and each target soon cycles. What is
  checked:
  - copies grow past the initial 8, one to the 16-draft lookahead;
  - after a rejection a step verifies more rows than its proposal has, those
    inputs are `-1`, and the step commits at most the proposal's drafts and the
    bonus;
  - the committed text is the target's greedy text;
  - the second request takes the first's slot.
- 0612's `test_stale_wide_allocation_rejects_placeholder_tail_and_doubles_cap_without_sync`
  is unchanged in patch 1, and passes.

Patch 2's test changes:
- the assertions on the removed limits go;
- 0612's test keeps its placeholder and doubling checks and loses the read that no
  longer exists;
- `test_slot_draft_limits_cap_stale_wide_scheduled_drafts` becomes
  `test_the_budget_takes_the_schedulers_drafts`;
- the two-rank test keeps its equal budgets.

The repro arms' diagnostics were checked in the same image
([runs/overlay-checks.txt](runs/overlay-checks.txt)):
- [layout_census_check.py](layout_census_check.py), with overlay/minimal-crc's
  `model_runner.py`: equal step sequences give equal CRCs, and one changed draft
  count changes the CRC;
- [limit_reads_smoke.py](limit_reads_smoke.py), with overlay/limit-reads'
  `copy_drafts.py`: stale reads, their resolution (wide and narrow), the unresolved
  path, and the periodic line.

### The repro on sp5–sp7 (`shared-budget-repro-sp`)

The window was `spark-0013-1793667996` on 2026-11-03, with the arms of
[lab-repro.json](lab-repro.json) at min match 8 and dead rows off. Each arm got one
boot and ran the extras in order. The receipts are on sp5:
- `results/private/determinism/shared-budget-repro-sp/`, which holds each extra's
  JSON lines and the engine logs kept with the 8- and 16-stream steps;
- `results/private/suites/shared-budget-repro-sp-single-stream/`;
- the table, `results/private/lab/shared-budget-repro-sp-table.txt`.

Read from those receipts:

| | `c37-m8-nodead` (counter) | `minimal-m8-nodead` (layout CRC) |
|---|---|---|
| Copy probe, tokens per step (copy / edit / tool / prose) | 16.367 / 15.745 / **6.636** / 2.79 | 16.367 / 15.745 / **6.905** / 2.79 |
| Single-stream suite | **died**: 50.61 tok/s, CI95 31%, 1,708,441 failed requests, suite exit 1 | 60.96 tok/s, CI95 5.1%, 2.383 tokens/step, 0 failed |
| 8 / 16 streams | engine already dead | 156.38 tok/s ±0.1% / 161.54 tok/s ±0.3% |
| Asserts in the engine logs | `route_partials: gate logits contain nan or inf`: 352 lines each on sp5 (TP0) and sp7 (TP2), from 01:15:39Z (sp5's log); sp6 lost its peers (TCPStore) | none |
| Per-rank CRCs | equal on sp5, sp6 and sp7 in all 8 windows | equal in all 10, 11 and 14 windows (`200f4afc` at 10,240 budgets, `7f4546ec` at 11,264, `669cab09` at 14,336) |

So #852 reproduced on tp3-switched, in the single-stream suite after the copy probe:
#852's own sequence, whose tool numbers match the operator's m=8 pair (6.636, 6.905).
The queue log showed the step as passing because `suite_single_stream.py` exits 0
whatever the bench returns; the bench's own status, 1, is in its JSON line.

Candidate-37's stale-read counter, on every rank alike:

| Budget reads (time, sp5) | stale | consequential | wider: slots / placeholder rows | narrower: slots / dropped drafts |
|---|---|---|---|---|
| 1,024 | 1,020 | 16 | 14 / 143 | 2 / 16 |
| 2,048 (01:11:07Z) | 2,043 | 31 | 25 / 245 | 6 / 48 |
| 4,096 | 4,091 | 32 | 26 / 256 | 6 / 48 |
| 6,144 | 6,139 | 37 | 31 / 311 | 6 / 48 |
| 8,192 (01:15:05Z, the last before the assert) | 8,187 | 44 | 38 / 375 | 6 / 48 |

What follows from it:
- **Almost every budget read is stale on sp5–sp7**: 8,187 of 8,192 found the
  latest limit copy still in flight.
- **Few reads change anything.** Only 44 changed a scheduled slot's eligible drafts.
- **Narrow reads stopped early.** Narrower reads (drafts dropped) stopped growing
  after 2,048 budgets, while wider ones (placeholder rows budgeted) kept growing up
  to the crash.
- **The ranks did not diverge.** Their CRCs of the consequential reads agree, so
  they read the same stale limits at the same steps.

What it rules out, and what is left:
- **Live placeholder rows** are not the producer: the minimal arm budgets more of
  them, live, and is clean.
- **Per-rank layout divergence** is not the cause here: candidate-37's ranks agreed.
- **The lead the parent session draws: the trimmed-budget path.** A stale limit
  between the current proposal's drafts and the scheduled ones makes eligible drafts
  fewer than scheduled. That sends the step through `compact_batch` and
  `assign_draft_budget`, with the budget below the scheduled drafts. The minimal arm
  admits the scheduled drafts and mostly takes the untrimmed path.

  The trimmed path also exists in the minimal arm, whenever the planned confidences
  don't pay for every row. So it would have to be the stale cap's particular layouts
  that break. This is not yet shown.
- **Queued on main** (`c7563669`, `b28365c3`) in [lab-diag.json](lab-diag.json), run
  `shared-budget-diag-sp`, to find the producer:
  - [arm-diag-c37-m8-nodead.json](arm-diag-c37-m8-nodead.json):
    candidate-37 with the non-finite census read every 32 steps rather than every
    step, so the race stays as in serving. The router zeroes a non-finite row
    instead of asserting, and every hit is logged with a ring of the last
    consequential stale reads.
  - [arm-no0676-c37-m8-nodead.json](arm-no0676-c37-m8-nodead.json): candidate-37
    with patch 0676 reverted and no census.

  Neither has results yet.

### Context: #856's and #859's runs on sp5–sp7

These are not this experiment's runs. The figures come from their receipts on sp5
(`results/private/suites/…`, `results/private/determinism/…`, and the queue log
`results/private/lab/queue-856.log`).

- **#856 (host wait), `limit-sync-cost-sp`, m=16, the recipe's dead rows.** All
  boots completed.
  - Single-stream suite (tok/s, CI95):
    - fix: 62.89 ±3.6%;
    - candidate-37: 63.44 ±2.9%;
    - fix repeat: 63.12 ±3.2%.

    The lab's paired verdict is −0.73% ±0.12% for the wait.
  - Copy probe, tool tokens per step: 6.232 on both fix boots, 6.083 on
    candidate-37. Copy, edit and prose are identical: 16.021, 15.322, 2.651.
  - 8 streams: 157.59 and 157.57 against 158.22 tok/s.
  - 16 streams: 163.34 and 163.76 against 162.34 tok/s.
- **#859 (device clamp), `device-limit-cost-sp`.** It was cancelled by the owner at
  00:20:40Z ("the device-side fix is dropped"), during the candidate-37 boot's
  suite.
  - The fix's first suite boot completed: 63.59 tok/s ±2.9%, 2.259 tokens per step.
  - Candidate-37's suite is **invalid**: the cancellation stopped its server
    mid-run (`BrokenPipeError`, 4,004,424 failed requests logged, 25.6 tok/s).
  - No repeat boot and no lean job ran.

  So the clamp's cost was not measured. The record keeps the run as cancelled.
- **`device-limit-repro-none-sp`** (the minimal fix's earlier arms): stopped at
  00:21:58Z before candidate-37's arm finished booting ("the shared-budget fix is
  shown for review first"). Nothing was measured.
- The window closed at 00:38:41Z with production restored and `doctor --live` OK.

## The specs

`lab-repro.json` has run (*The repro on sp5–sp7*); `lab-cost.json` is queued. Both
dry-run clean with `--production-config
config/candidate-37-tp3-switched-dev.local.json`. That is the profile sp5–sp7 serve
(sha `92b1712d…`, node map `config/nodes-switched3.local.json` sha `0c508533…`); both
files were copied from sp5 and not committed. `bin/spark --cluster-config <arm>
doctor` reports configuration OK for all four arms. The arms extend that profile,
mount at most one file, and carry no census.

#842's census `.item()`s after the drafter every step, so the limit copy always lands
before the next budget read and the race never happens. That is why the repro arms
use host-only diagnostics.

- [lab-repro.json](lab-repro.json), run `shared-budget-repro-sp`: #852's sequence at
  min match 8 with dead rows off, one boot per arm. The steps:
  - the copy probe;
  - the single-stream suite ([suite_single_stream.py](suite_single_stream.py));
  - the reads ([limit_reads_logs.py](limit_reads_logs.py));
  - 8 streams, then the reads;
  - 16 streams, then the reads.

  The arms:
  - [arm-c37-m8-nodead.json](arm-c37-m8-nodead.json) is candidate-37 with
    [overlay/limit-reads](overlay/limit-reads/)' `copy_drafts.py`. Each rank logs its
    stale budget reads, those that changed eligible drafts (wider or narrower), and a
    CRC of them.
  - [arm-minimal-m8-nodead.json](arm-minimal-m8-nodead.json) is patch 1 with
    [overlay/minimal-crc](overlay/minimal-crc/)'s `model_runner.py`. Each rank logs a
    CRC of its budgeted layouts (token count and eligible drafts per request) every
    1,024 budgets. It is host only: 1.2–1.7 µs a step and no device sync, measured
    on #859's identical method.

  Outcomes as written before the run:
  - Candidate-37 asserts, or its ranks' CRCs disagree, while the minimal arm stays
    clean with agreeing CRCs: the rank-dependent read was #852's mechanism, and this
    fixes it.
  - The minimal arm asserts: live placeholder rows produce non-finite logits. Then
    the fix needs dead rows on, or a clamp.

  What happened was neither. Candidate-37 asserted with its ranks' CRCs equal, and
  the minimal arm was clean. So the fix holds on this sequence, while the first
  outcome's explanation does not.
- [lab-cost.json](lab-cost.json), run `shared-budget-cost-sp`: the paired
  single-stream suite for minimal-m16, c37-m16, minimal-m16, then the copy probe and 8
  and 16 streams in the same order. It runs at m=16 with the recipe's dead rows.
  - [arm-minimal-m16.json](arm-minimal-m16.json) mounts
    [overlay/minimal](overlay/minimal/)'s `model_runner.py`, byte-identical to patch
    1's, and nothing else.
  - [arm-c37-m16.json](arm-c37-m16.json) mounts nothing.

**Acceptance criteria**, set before running, with where each stands:
- the minimal arm passes the repro with no assert or non-finite gate row, and its
  ranks' CRCs agree in every window: **met** (one boot);
- the paired single-stream verdict against candidate-37 is not worse beyond its CI:
  pending (`lab-cost.json`);
- the copy probe's tool tokens per step are at least candidate-37's: met at m=8
  (6.905 against 6.636); m=16 pending;
- 8 and 16 streams are within noise: pending. The repro's candidate-37 arm died
  before its streams, so there is nothing to pair them with.

Tables: [tables_arms.py](tables_arms.py) `RUN ARM...` prints each arm's probe, suite,
reads with the ranks' CRC agreement, and streams.

## Commands

The tree (any clone that has `5b7cfa37f`):

```bash
git init fresh && cd fresh
git fetch --depth 1 --no-tags /path/to/vllm 5b7cfa37fee42f96bfafbe09b7d30e8c0eda8575
git switch -c copy-shared-budget FETCH_HEAD                  # tree af0fb152
git am --committer-date-is-author-date \
  experiments/2026-11-03-copy-drafter-shared-budget/vllm/0001-*.patch   # tree 58882ab9
git am --committer-date-is-author-date \
  experiments/2026-11-03-copy-drafter-shared-budget/vllm/0002-*.patch   # tree f473423c
```

The unit tests use [run_image.sh](run_image.sh), which mounts the tree's files
changed since `5b7cfa37f`. Candidate-37's counts are the same command with nothing
mounted.

```bash
TREE=/tmp/vllm-shared-budget PLUGINS=/tmp/loop-test-plugins ./run_image.sh \
  "python3 -m pytest -q -p no:cacheprovider -p tblib_stub -rfEs \
   tests/v1/spec_decode/test_copy_drafts.py tests/v1/core/test_copy_drafts_scheduler.py \
   tests/v1/spec_decode/test_adaptive_verification.py tests/v1/worker/test_prepare_kernels.py"
```

Overlay checks run with the image's own `PYTHONPATH`, so the mounted tree, not the
installed copy, is imported:

```bash
docker run --rm --gpus all -v $PWD:/x:ro \
  -v $PWD/overlay/minimal-crc/vllm/v1/worker/gpu/model_runner.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/model_runner.py:ro \
  -w /opt/spark-ds41f/candidate/vllm --entrypoint python3 spark-ds41f:candidate-37 /x/layout_census_check.py
```

The dry runs:

```bash
scripts/lab.py run experiments/2026-11-03-copy-drafter-shared-budget/lab-repro.json \
  --dry-run --production-config config/candidate-37-tp3-switched-dev.local.json
scripts/lab.py run experiments/2026-11-03-copy-drafter-shared-budget/lab-cost.json \
  --dry-run --production-config config/candidate-37-tp3-switched-dev.local.json
```

## Not verified

- **The cost:** `lab-cost.json` is queued.
- **The repro's repeat:** one boot per arm so far. The minimal arm's clean run is a
  single boot, so its absence of a crash bounds a rate rather than proving zero.
- **The NaN producer:** neither live placeholder rows nor per-rank divergence
  (above). The trimmed-budget path is the lead; `lab-diag.json` is queued to find it.
  Until it is found, the minimal fix is known to avoid the crash on this sequence,
  not why.
- **The operator's ring:** their crash ran with dead rows on (their report); ours
  ran with them off. Their paired run on ring2 is offered, not run.
- **Patch 2** is unit-tested only. It is not measured and not in any arm.
- **#859's clamp cost:** never measured (cancelled, above).

## Conclusion

On sp5–sp7, #852 reproduces on candidate-37, and the minimal fix runs the same
sequence clean. That sequence is the copy probe, the single-stream suite, then 8 and
16 streams, at min match 8 with dead rows off. The fix also keeps the tool drafts a
stale narrow read used to drop (6.905 tokens per step against 6.636).

The fix is one argument, without a host wait or a clamp. Unit tests show the layouts
agree across simulated ranks, and the text is the target's through copy growth,
shrinkage and slot reuse.

What the repro does not support is the reason given for the fix:
- candidate-37's ranks agreed;
- the minimal arm's live placeholder rows were harmless.

The lead is that it works by keeping the budget off the stale cap's trimmed
layouts; the diagnosis job tests that. Its cost is the other open measurement.
