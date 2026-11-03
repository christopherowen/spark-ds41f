# Copy drafter: one budget for every rank (#852)

Owner, 2026-11-03: "I don't like the performance loss, we should remove the
device-side fix and do this correctly." Then: "get to the point you are ready for
testing, stop and show me where it is."

**Status:** the patch is unit-tested in candidate-37's image on sp8. Three jobs have
run on sp5–sp7, in window `spark-0013-1793667996`:
- **The repro** (`shared-budget-repro-sp`): #852 reproduced on candidate-37 at min
  match 8 with dead rows off. The minimal arm ran the same sequence clean, its ranks'
  layout CRCs equal in every window.
- **The cost** (`shared-budget-cost-sp`): single-stream speed is the same as
  candidate-37's (paired −0.08% ±0.09%, inside the 0.30% floor), and 8 and 16 streams
  are no slower.
- **The parent session's diagnosis** (`shared-budget-diag-sp`):
  - with patch 0676 reverted, candidate-37 failed the same way, sooner;
  - with a census read every 32 steps, it found no non-finite row.
- **The parent session's metadata census** (`shared-budget-diag2-sp`): with the
  census's per-step work, candidate-37's ranks budgeted differently from the first
  1,024 budgets on. Non-finite gate rows then appeared in both the target's and the
  drafter's forwards within 2,048 steps.
- **Two reruns** (`shared-budget-diag3-sp`, `shared-budget-diag4-sp`):
  - diag3 repeated diag2's picture, and its metadata census again kept only a
    warm-up batch;
  - diag4 died a minute into serving, with no census window completed.

- **The layout ring** (`shared-budget-ring-sp`): a per-step record of every rank's
  layout that outlives the engine. Candidate-37 crashed again, and the crash step was
  the one step of 5,340 whose layout differed across the ranks: a copy proposal of 16
  drafts that two ranks' stale mirrors admitted and the third's capped at 5, so two
  ranks ran the 17-token graph and one the 6-token graph. The repeat boot crashed the
  same way at its own last step, 6,373 steps in, with the same rank lagging. That is
  #852's mechanism, measured twice; the repro's "ranks agreed" held only through the
  counter's last window, which closed 7 s before each assert.

Not promoted; no recipe change.

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

**Our repro since (below) agreed on one point and seemed to disagree on the other.**
- On placeholders it agrees: the minimal arm verifies more `-1` rows, routed live
  with dead rows off, and stays clean. So live placeholder rows are not the NaN
  producer here.
- On the layout it seemed to disagree: on candidate-37's crashing arm the three
  ranks' CRCs of the consequential stale reads were equal in every window. But the
  last window closed 34 s before the assert, and the layout ring (*The layout ring*)
  then showed the crash step itself laid out differently on the ranks. The operator's
  report stands.

**The diagnosis job since** (*The diagnosis job*, below) narrows it.
- **Stale reads alone are not enough.** Stale reads of both kinds, mostly narrow
  ones, ran through 11,264 budgets on candidate-37 with no non-finite row, once a
  census read every 32 steps had changed their timing.
- **Without patch 0676 the assert comes sooner.** With 0676 reverted, candidate-37
  met the repro's stale reads: equal CRCs at 1,024, 2,048 and 3,072 budgets. Its TP2
  rank asserted in `route_partials` between 3,072 and 4,096 budgets, where the repro
  asserted only after 8,192.

The parent session's reading of the three runs: the producer is a live row whose
metadata leaves it nothing to attend to.
- That is a negative request index or position, whose lengths `window_row` zeroes, or
  a `query_start_loc` end short of the token count.
- 0676 writes such a row's output as 0 and skips its output projection's input, so
  the projection reads stale scratch. That is usually finite and eventually not.
  Without 0676 the row goes non-finite sooner.
- The census records only non-finite values, so a finite-garbage row leaves no trace.

None of this is shown yet. The minimal arm's clean run does not show either that the
shared budget never builds such a row: a finite-garbage row would leave no trace
there either.

**The metadata census since** (*The metadata census*, below) adds two things.
- **The ranks diverged.** With the census's work each step, candidate-37's ranks'
  CRCs of their consequential stale reads differed in every window from the first
  1,024 budgets on, with different counts (61, 41 and 42 after the suite).
  - This is the per-rank divergence of the operator's report and of the two-rank
    unit test, now on hardware. It was seen only under the census's timing, though.
    That ordinary differences between hosts can do the same is an inference, not
    measured.
- **Non-finite gate rows followed early.** The first census read, 2,048 steps in,
  found them on all three ranks:
  - in the drafter's three MoE gates (DSpark's block);
  - in the target's gates from layer 20 to 39;
  - in some of its first layers, where the flagged rows differ by rank.

  The copy probe's text had already changed: its copy and edit prompts gave 3,168
  and 2,713 output tokens, against 4,550 and 4,566 in every other m=8 arm.

The census cannot order those flags: it keeps them until it is read. So it does not
say which forward, or which step, produced the first non-finite value. Nor does the
divergence explain the plain repro, whose ranks agreed and which asserted only after
8,192 budgets.

The producer is not named. Two runs of the metadata census kept only a startup or
warm-up batch as their first hit: every row of those is invalid by design. So it says
nothing about real steps. A third run, with those batches ignored, died a minute in
(*The reruns*).

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
  - The extra rows cost compute on the steps after a copy shrinks. The cost run
    found none measurable at m=16 with dead rows on: rows per step are 4.07 on both
    builds, and single-stream speed is the same.
- **Stale narrow limits no longer drop drafts.** On sp5–sp7, #856's run had
  candidate-37 at 6.083 tool tokens per step against 6.232 with the wait. The shared
  budget gives 6.232 too (cost run, m=16), and 6.905 against 6.636 at m=8 (repro).
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
- **The lead the parent session drew then: the trimmed-budget path.** A stale limit
  between the current proposal's drafts and the scheduled ones makes eligible drafts
  fewer than scheduled. That sends the step through `compact_batch` and
  `assign_draft_budget`, with the budget below the scheduled drafts. The minimal arm
  admits the scheduled drafts and mostly takes the untrimmed path.

  The trimmed path also exists in the minimal arm, whenever the planned confidences
  don't pay for every row. So it would have to be the stale cap's particular layouts
  that break. This is not yet shown, and the diagnosis job's reading (below) has
  since replaced it as the lead.
- **The parent session's diagnosis job** has run since: *The diagnosis job*, below.

### The cost run on sp5–sp7 (`shared-budget-cost-sp`)

The run was in the same window, after the repro, with the arms of
[lab-cost.json](lab-cost.json) at m=16 and the recipe's dead rows (0.2). The arms ran
in the order minimal, candidate-37, minimal, each with one boot for the suite and
another for the extras. The receipts are on sp5:
- `results/private/suites/shared-budget-cost-sp/`;
- `results/private/determinism/shared-budget-cost-sp/`;
- the table, `results/private/lab/shared-budget-cost-sp-table.txt`;
- the verdicts, in `results/private/lab/queue-860.log`.

| | `minimal-m16` | `c37-m16` | `minimal-m16` repeat |
|---|---|---|---|
| Single-stream suite, tok/s (CI95) | 63.50 ±3.0% | 63.56 ±2.9% | 63.48 ±3.0% |
| Tokens / rows per step | 2.259 / 4.07 | 2.259 / 4.07 | 2.259 / 4.07 |
| Copy probe, tool tokens per step | **6.232** | 6.083 | **6.232** |
| 8 streams, tok/s | 157.50 ±0.2% | 157.36 ±0.3% | 157.89 ±0.2% |
| 16 streams, tok/s | 163.47 ±0.9% | 162.61 ±1.4% | 164.69 ±0.3% |

- **The lab's paired verdict is "same":** −0.08% ±0.09% decode tok/s against
  candidate-37. It is over both fix boots, with the recorded floor of 0.30%, a margin
  of 0.32%, and a repeat of +0.01%.
  - Alone, the first fix boot reads −0.11% ±0.05%, which its row calls worse on the
    request interval without the floor.
  - The repeat boot reads −0.06% ±0.14%, same.
  - #856's host wait was −0.73% ±0.12% on the same suite.
- **Draft acceptance is unchanged.** Rows per step, acceptance by position and the
  copy, edit and prose probes (16.021, 15.322, 2.651) are identical on all three
  boots.
- **The tool drafts that candidate-37's stale narrow reads dropped come back:**
  6.232 against 6.083, the same as with #856's wait.
- **8 and 16 streams are no slower:** both fix boots are above candidate-37's single
  boot.

### The diagnosis job (`shared-budget-diag-sp`)

This is the parent session's job, published on main (`c7563669`, `b28365c3`) and run at
`b28365c3` from 02:31 to 02:55Z in the same window. Its arms of
[lab-diag.json](lab-diag.json) are candidate-37 at m=8 with dead rows off, each with
the stale-read counter. Each had one boot, which ran the copy probe, the single-stream
suite, then 8 streams.
- [arm-diag-c37-m8-nodead.json](arm-diag-c37-m8-nodead.json) adds the non-finite
  census, read every 32 steps. The router zeroes a non-finite row instead of
  asserting.
- [arm-no0676-c37-m8-nodead.json](arm-no0676-c37-m8-nodead.json) reverts patch 0676
  and has no census.

The receipts are on sp5:
- `results/private/determinism/shared-budget-diag-sp/`, which holds the census and
  reads lines and the reverted arm's engine logs;
- `results/private/suites/shared-budget-diag-sp-single-stream/`;
- the table, `results/private/lab/shared-budget-diag-sp-table.txt`.

| | `diag-c37-m8-nodead` (census) | `no0676-c37-m8-nodead` (0676 reverted) | the repro's `c37-m8-nodead` |
|---|---|---|---|
| Copy probe (copy / edit / tool / prose) | 16.367 / **13.351** / 6.636 / 2.79 | 16.367 / 15.745 / 6.636 / 2.79 | 16.367 / 15.745 / 6.636 / 2.79 |
| Single-stream suite | 59.25 tok/s, CI95 5.7%, 2.377 tokens/step, 0 failed | **died**: 8.19 tok/s, 8,336,417 failed requests, suite exit 1 | **died**: 50.61 tok/s, 1,708,441 failed |
| 8 streams | 153.71 tok/s ±0.2% | engine already dead | engine already dead |
| Non-finite rows (census) | **0** on every node, through the suite and 8 streams | no census | no census |
| Asserts | none | `route_partials: gate logits contain nan or inf` on sp7 (TP2) | the same, on sp5 (TP0) and sp7 (TP2) |
| Budget reads | 11,264, no failure | failed between 3,072 (02:49:16Z) and 4,096 | failed after 8,192 (01:15:05Z) |
| Counter CRCs | equal on all ranks, and differ from the repro's from 1,024 on | equal on all ranks, and equal to the repro's at 1,024, 2,048 and 3,072 | equal on all ranks |

The census arm's counter, on every rank alike:

| Budget reads | stale | consequential | wider: slots / placeholder rows | narrower: slots / dropped drafts |
|---|---|---|---|---|
| 9,216 (after the suite) | 8,922 (96.8%) | 111 | 20 / 185 | 91 / 959 |
| 11,264 (after 8 streams) | 10,906 | 205 | 28 / 256 | 193 / 1,954 |

The reverted arm's failure, from its engine logs:
- **TP2 (sp7) asserted first.** The first 50 matching lines of its container log
  are all `route_partials: gate logits contain nan or inf`.
- **TP0's fault was a consequence.** On sp5 the `CUDA error: unspecified launch
  failure` at 02:49:53Z came with the collective's report: "RoCE collective on rank 0
  saw nothing from rank 2, HCA 0, for 10.000 s … the GPU was stopped there". So
  TP0's launch failure was the stop that timeout imposed, not a fault of its own.
- **sp6 (TP1)** lost the TCPStore and asserted nothing. That was also true in the
  repro's crash.

The parent session's first reading had the reverted arm faulting a kernel on TP0
without a router assert. The logs show the router assert, on TP2.

What follows from it:
- **Stale reads alone are not enough.** The census arm budgeted stale reads of both
  kinds, mostly narrow, with no non-finite row in 11,264 budgets.
- **The census changes the race it watches.** Its read every 32 steps is a blocking
  `.item()`, which lets limit copies land. Its stale sequence differs from the
  repro's from the first 1,024 budgets on, with 91 narrow slots against the repro's 6
  and a different edit probe (13.351 against 15.745). So its clean run does not test the
  repro's layouts. `lab-diag2.json`'s census then read every 2,048 steps instead
  (below).
- **Without 0676, the same stale sequence asserts sooner:** before 4,096 budgets,
  where with 0676 it asserted after 8,192. This fits a row that 0676 makes finite most
  of the time, which is the parent session's reading in the cause section.
- **Since run:** the metadata census, `shared-budget-diag2-sp`, below. Its census reads
  every 2,048 steps rather than 32 (main `760614dd`).

### The metadata census (`shared-budget-diag2-sp`)

This is the parent session's job, run at `760614dd` from 04:22 to 04:41Z in the same
window. It is one boot of
[arm-diag2-c37-m8-nodead.json](arm-diag2-c37-m8-nodead.json): candidate-37 at m=8
with dead rows off, carrying:
- the non-finite census, whose router zeroes a flagged row's partials instead of
  asserting;
- the stale-read counter, with a ring of the last 64 consequential reads;
- the metadata census in `sparse_mla.py`'s `_build_tilelang`, for the target's and
  the drafter's builders. It flags a live row with a negative request index or
  position, or a `query_start_loc` end that differs from the token count.

Both censuses are read every 2,048 steps. The job ran the copy probe, the
single-stream suite, then 8 and 16 streams, each followed by the census and the
reads. The receipts are on sp5:
- `results/private/determinism/shared-budget-diag2-sp/`;
- `results/private/suites/shared-budget-diag2-sp-single-stream/`;
- the table, `results/private/lab/shared-budget-diag2-sp-table.txt`.

| | `diag2-c37-m8-nodead` | the repro's `c37-m8-nodead` |
|---|---|---|
| Copy probe (copy / edit / tool / prose) | **15.529 / 14.354** / 6.636 / 2.79 | 16.367 / 15.745 / 6.636 / 2.79 |
| Copy / edit output tokens | **3,168 / 2,713** | 4,550 / 4,566 |
| Single-stream suite | 55.73 tok/s, CI95 3.1%, 2.361 tokens/step, 0 failed | **died** |
| 8 / 16 streams | 148.22 ±1.1% / 155.82 ±0.4% tok/s | engine already dead |
| Asserts | none: the census zeroes flagged rows | `route_partials` on TP0 and TP2 |
| Ranks' counter CRCs | **differ in every window**: 9 of 9 after the suite, 10 of 10 after 8 streams, 13 of 13 after 16 | equal in every window |

The counter by rank:

| | sp5 (TP0) | sp6 (TP1) | sp7 (TP2) |
|---|---|---|---|
| After the suite: 9,216 reads, 9,207 stale on each | 61 consequential; wider 51 slots / 407 rows; narrower 10 / 68 drafts | 41; 28 / 244; 13 / 98 | 42; 32 / 277; 10 / 73 |
| After 16 streams: 13,312 reads, 13,301 stale on each | 692; 98 / 854; 626 / 6,573 | 669; 67 / 604; 634 / 6,642 | 669; 70 / 634; 631 / 6,617 |

The first census read was at 04:28:18Z, census call 4,096, 2,048 steps in. On every
rank, it flagged gate rows in both forwards:
- **the drafter's** (`model.layers.40–42.ffn.gate`, DSpark's block): rows 0–4;
- **the target's** (`language_model.model.layers.N.ffn.gate`):
  - rows 0–16 of layers 20–39 on every rank;
  - rows of its first layers that differ by rank: rows 0, 1 and 6–16 of layers 0–2
    on sp5 and sp7 (sp7 without row 9), and rows 0–5 of layers 0–6 on sp6.

Several things limit what the receipt says:
- **No order in time.** A router's flags accumulate until the read, so the rows are a
  union over 2,048 steps. They say neither which step nor which forward came first.
- **The labels.** The lines read "NONFINITE drafter row" because the read came after
  the drafter's forward; the label is the reading call's phase. The routers are
  listed in registration order, the drafter's first. The parent session first read
  the hit as the drafter's gate alone, with the target's recording nothing. The
  layer names show the target's gates too.
- **Truncation.** The census step keeps each node's first 400 NONFINITE lines, and
  this read alone fills them. The list is cut within the target's layer 3 (sp5, sp7)
  or layer 6 (sp6), so later layers are unknown, not clean. Later reads appear only
  as counts: 1,084, 986 and 1,512 lines through the suite on sp5, sp6 and sp7.
- **The step logged.** The layout logged with the hit, one request in slot 7 with 2
  tokens, is the reading step's, not the hit's.

The ring at that read holds every consequential read until then: 46 on sp5, 31 on
sp6 and sp7. Each entry gives a slot, the eligible drafts the budget used, and those
the landed limits gave.
- **Slot 7's entries** have the budget at 16 where the landed limits gave 5, 5 where
  they gave 1, and 8 or 5 where they gave 16. The host's mirror was a proposal or two
  behind each time.
- **The parent session's reading** of 16, 5 and 1 is a copy proposal's end, DSpark's
  block, and a single draft.
- **The rings differ by rank.** sp5 logged nine reads at 5 against 1 (reads
  423–431); sp6 and sp7 logged one each.

The metadata census recorded one hit, at call 1. That was startup's dummy batch: 6,155
rows, all invalid by design. The census keeps only its first hit, so it says nothing
about real steps. Main `416393fe` changed it for `shared-budget-diag3-sp` (*The
reruns*): the first real step discards startup's hit.

What follows from it:
- **The census overlay shifts the race even at this cadence.** The ranks' CRCs
  differ from the repro's, and from each other, from the first 1,024 budgets on. The
  blocking read is now rare, so the shift comes from the per-step work: the device
  census and the host's layout ring.
- **Ranks that budget differently coincide with early non-finite rows.** With the
  ranks diverged, non-finite gate rows appeared within the first 2,048 steps, and the
  probe's temperature-0 text changed. With the ranks agreeing, the plain repro first
  asserted after 8,192 budgets.
- **It does not explain the plain repro.** There the ranks agreed by the counter: the
  same CRCs of the consequential reads, and the same totals of the drafts those reads
  changed.
- **The producer is not named.** Neither forward is singled out, and the metadata
  census saw no real step.

### The reruns (`shared-budget-diag3-sp`, `shared-budget-diag4-sp`)

These are the parent session's last two jobs in the window, each one boot of
[arm-diag2-c37-m8-nodead.json](arm-diag2-c37-m8-nodead.json) with the census overlay
at a later main. The receipts are on sp5:
- `results/private/determinism/shared-budget-diag3-sp/` and
  `…/shared-budget-diag4-sp/`, the latter with each step's engine logs;
- the tables, `results/private/lab/shared-budget-diag3-sp-table.txt` and
  `…-diag4-sp-table.txt`;
- the queue log, `results/private/lab/queue-860.log`.

The runs:
- **diag3** ran at `416393fe` from 04:41 to 05:00Z. The first real step discards
  startup's metadata hits.
- **diag4** ran at `c26ed0a4` from 05:00 to 05:14Z. The metadata census ignores
  batches whose rows are all invalid, and each rank logs a running CRC of its
  budgeted steps' layouts every 1,024 steps.

| | diag2 (above) | diag3 | diag4 |
|---|---|---|---|
| Copy probe, copy / edit tokens per step | 15.529 / 14.354 | 15.539 / 13.685 | 14.833 / **died** |
| Copy / edit output tokens (other m=8 arms: 4,550 / 4,566) | 3,168 / 2,713 | 3,372 / 2,736 | 3,188 / 1,251 from 1 of 3 prompts |
| Single-stream suite | 55.73 tok/s, CI95 3.1%, 0 failed | 55.73 tok/s, CI95 3.1%, 0 failed | engine dead |
| 8 / 16 streams, tok/s | 148.22 ±1.1% / 155.82 ±0.4% | 148.97 ±0.6% / 153.92 ±0.1% | engine dead |
| Ranks' counter CRCs | differ in 9/9, 10/10, 13/13 windows | differ in 9/9, 10/10, 14/14 windows | no window completed |
| Metadata census's hit | call 1, startup's batch | call 225, a warm-up batch | none logged |

**diag3.** It repeats diag2's picture.
- **The ranks diverged** from the first 1,024 budgets on.
  - After the suite: 62, 37 and 42 consequential reads on sp5, sp6 and sp7.
  - After 16 streams: 911, 886 and 892, with 853, 860 and 858 narrower slots and
    8,734, 8,797 and 8,780 dropped drafts.
- **The metadata census's first hit was a warm-up batch:** call 225, request
  `_warmup_greedy_`, builder 29, all 59 rows invalid. Again no real step was recorded.
- **The first non-finite census read** (04:47:20Z, 2,048 steps in) flagged the same
  routers on every rank as diag2's did:
  - DSpark's `model.layers.40–42`, rows 0–4;
  - the target's layers 20–39, rows 0–16;
  - the target's layers 0 and 1, rows 0–16, and layer 2 from row 0.

  The receipt's 400 lines end within layer 2 on all three ranks. The parent session
  read these lines as the drafter's layer 40 alone; the layer names say otherwise,
  as in diag2. The full container logs went with the boot.

**diag4.** The engine died at 05:03:58Z, a minute after the cluster was ready
(05:02:58Z).
- **When:** during the copy probe's second edit prompt. The copy prompts and the
  first edit prompt had completed; the remaining edit prompt and the three tool
  prompts got HTTP 500 from the dead engine.
- **The first silent rank was TP1 (sp6).** sp5 (TP0) and sp7 (TP2) both raised
  `CUDA error: unspecified launch failure`, each with the collective's report:
  "RoCE collective on rank 0 [rank 2] saw nothing from rank 1, HCA 0, for 10.000 s …
  at sequence 115676; the GPU was stopped there". So rank 1 stopped answering a
  collective about 05:03:48Z. sp6's log has no error at all, and its last line is
  05:03:18Z.
- **What the traceback shows.** It is sp5's and sp7's, in a target MoE layer of the
  sequence-parallel prefill path: `model.py:452` (`sp_moe_inputs`), `route_local`,
  `_census_rows` (`router.py:1092`). That is where those GPUs were when they were
  stopped. It says nothing about which kernel failed, or how, on rank 1.
- **Nothing else was logged:** no `nan or inf` assert or NONFINITE line on any rank,
  and no COPY-LIMITS or layout CRC line. No census window of 1,024 steps completed.

Whether diag4's instrumentation or the stale cap made rank 1 fail is undetermined.
The 0676-reverted arm also died with a launch failure that followed a peer's
collective timeout, but there the silent rank had asserted.

### The layout ring (`shared-budget-ring-sp`, `shared-budget-ring2-sp`)

Owner, 2026-11-03: "Keep digging." Then: "Open the window." This is the parent
session's job, run in window `spark-0013-1793692440` on sp5–sp7 from 07:54Z, at main
`74409a7e` (the first boot) and `c2bb2ebb` (the repeat).

**The gap it closes.** The repro's claim that candidate-37's ranks agreed rests on the
counter's windows, and the last window closed before the crash:
- the ranks' `COPY-LIMITS reads=8192` lines are at 01:15:05Z on sp5, sp6 and sp7, with
  equal CRCs;
- the asserts are at 01:15:39Z, 34 s and several hundred budgets later, and nothing of
  those budgets was logged;
- the 0676-reverted arm is the same: equal CRCs at 3,072 reads, the assert before
  4,096.

So a divergence in the last steps before each crash was unrecorded, not ruled out.

**What it records.** [overlay/layout-ring](overlay/layout-ring/)'s `model_runner.py`
is candidate-37's runner with one call in `prepare_inputs`, after the batch is
compacted and before the query starts are built. Each rank writes one 40-byte record a
prepared step to `/cache/spark-ds41f/layout-ring/rank<r>.bin` on its node (the
checkout's `cache/`, which the profile mounts read-write), with `os.pwrite`: no
buffering, no device sync, a few microseconds. The fields:
- the step, the token count and the padded count the dispatch chose, the request
  count;
- CRCs of the compacted per-request token counts, of the scheduler's drafts per
  request and of the budget's eligible drafts per request;
- the draft budget;
- the counter's cumulative budget reads and stale reads, which tie a record to the
  `COPY-LIMITS` windows.

The step index is rank-invariant: every rank prepares the same scheduler outputs.
The files outlive the engine, so [layout_ring_logs.py](layout_ring_logs.py) reads
them after the crash as after a clean workload: one line a node, and one line that
compares the ranks at equal steps, with the steps whose layouts differ (by field), the
first with its neighbours, each rank's last records, and the steps whose stale-read
counts differ.

**The arm and the spec.**
- [arm-ring-c37-m8-nodead.json](arm-ring-c37-m8-nodead.json) is the repro's
  `arm-c37-m8-nodead` (candidate-37, m=8, dead rows off, the stale-read counter) plus
  the ring's `model_runner.py`. No census, no device sync: the repro's timing, a
  `pwrite` a step apart.
- [lab-ring.json](lab-ring.json), run `shared-budget-ring-sp`: the repro's sequence,
  one boot, with the ring read after the suite and after each streams step. Dry-runs
  clean with the window's profile, and `bin/spark --cluster-config
  arm-ring-c37-m8-nodead.json doctor` reports configuration OK.
- [layout_ring_check.py](layout_ring_check.py), in candidate-37's image on sp8 with
  the overlay mounted: 9 records of 40 bytes, equal sequences give equal records, one
  changed eligible count changes only its CRC, a step without drafts records no budget,
  and the counter's reads accumulate (`LAYOUT-RING /tmp/.../rank2.bin | layout ring
  OK`).
- [tables_arms.py](tables_arms.py) prints the comparison beside the counter's.

**The arm and the spec.**
- [arm-ring-c37-m8-nodead.json](arm-ring-c37-m8-nodead.json) is the repro's
  `arm-c37-m8-nodead` (candidate-37, m=8, dead rows off, the stale-read counter) plus
  the ring's `model_runner.py`. No census, no device sync: the repro's timing, a
  `pwrite` a step apart.
- [lab-ring.json](lab-ring.json), run `shared-budget-ring-sp`, and
  [lab-ring-2.json](lab-ring-2.json), run `shared-budget-ring2-sp`, its repeat boot:
  the repro's sequence, one boot each, with the ring read after the suite and after
  each streams step.
- [layout_ring_check.py](layout_ring_check.py), in candidate-37's image on sp8 with
  the overlay mounted: 9 records of 40 bytes, equal sequences give equal records, one
  changed eligible count changes only its CRC, a step without drafts records no budget,
  and the counter's reads accumulate (`LAYOUT-RING /tmp/.../rank2.bin | layout ring
  OK`).
- [tables_arms.py](tables_arms.py) prints the comparison beside the counter's.

**Outcomes, written before the run.**
- The ranks' records differ at steps before the assert: the per-rank stale read laid
  out different steps, and the crash follows the divergence. Which field differs says
  how: `padded` means the ranks ran different graphs, so their collectives disagreed
  in size; `eligible_crc` or `tokens` alone means equal sizes and rows that hold
  different tokens, summed across the shards.
- The ranks' records are equal through the last common step, and the assert follows
  anyway: the producer is on one rank, and the metadata census is the next
  instrument.
- The engine does not crash: the pwrite shifted the timing. The ring still says
  whether the ranks diverged without a census's work.

**The first boot** (07:54 to 08:07Z). The receipts are on sp5:
`results/private/determinism/shared-budget-ring-sp/` (the ring's and the counter's JSON
lines, and the engine logs kept with the streams steps),
`results/private/suites/shared-budget-ring-sp-single-stream/`, and the table
`results/private/lab/shared-budget-ring-sp-table.txt`.

| | `ring-c37-m8-nodead` | the repro's `c37-m8-nodead` |
|---|---|---|
| Copy probe (copy / edit / tool / prose) | 16.367 / 15.745 / 6.636 / 2.79; 4,550 / 4,566 output tokens | the same |
| Single-stream suite | **died**: 20.13 tok/s, 6,663,222 failed, exit 1 | died |
| Asserts | `route_partials: gate logits contain nan or inf` on sp5 (TP0) and sp7 (TP2); none on sp6 (TP1), which lost the TCPStore at 08:02:45Z | the same pattern |
| Counter windows | 5 (5,120 reads), the last at 08:02:38Z; CRCs equal on all ranks; 34 consequential (28 wider / 278 rows, 6 narrower / 48 drafts) | 8, equal |
| Ring records | **5,340 on each rank; 5,339 steps equal; step 5,339, the last, differs** | – |
| Repeat boot (below) | died at step 6,372, the only divergent step of 6,373 | – |

The probe's text and timing are the repro's, so the pwrite changed nothing visible.
The crash is 7 s and 159 budgets after the last counter window, inside its blind spot
again. The ring's last five steps, one request in the batch throughout (the counter's
`reads` and `stale` are cumulative; CRCs are of one-element arrays):

| Step | TP0 (sp5) | TP1 (sp6) | TP2 (sp7) |
|---|---|---|---|
| 5,335 | 5 tokens, graph 5, scheduled 5 drafts, eligible 5, budget 4 | the same | the same |
| 5,336–5,338 | 6 tokens, graph 6, scheduled 5, eligible 5, budget 5 | the same | the same |
| **5,339** | **17 tokens, graph 17, scheduled 16, eligible 16, budget 16** | **6 tokens, graph 6, scheduled 16, eligible 5, budget 5** | **17 tokens, graph 17, scheduled 16, eligible 16, budget 16** |

Every rank's budget read was stale at every one of these steps (5,274 of 5,279 reads
were, on each rank alike), and the counter's stale counts agree on all three. What
differed is which older copy each mirror held. At step 5,339 the scheduler laid out a
copy proposal of 16 drafts; TP0's and TP2's mirrors gave 16, TP1's gave 5, the block
before. So TP0 and TP2 prepared a 17-row step and ran the 17-token graph, and TP1
prepared a 6-row step and ran the 6-token graph, for the same scheduler output.

What follows from it:
- **The crash step is a divergent step, and the only one.** The 5,339 steps before it
  laid out alike on all ranks, and the one step that did not is the one that
  asserted. The repeat boot repeats this at 6,372 of 6,373.
- **The ranks ran different graphs.** `padded` differs (17 against 6), so every
  collective of that forward disagreed in size across the ranks: rows 6–16 exist on
  TP0 and TP2 and not on TP1. The two 17-row ranks asserted in `route_partials`, where
  a row's gate logits are the sum of the shards' partials; the 6-row rank asserted
  nothing and lost its peers. This is the pattern of both earlier crashes.
- **It is the operator's mechanism** (#855, comment 6592146119): the per-rank layout,
  not a live placeholder row. The repro's "ranks agreed" was true of the counter's
  windows, which the crash never reached.
- **No kernel is at fault.** The first non-finite value is made where the shards' rows
  are combined for a step they did not lay out alike. Which collective delivers what
  to the 17-row ranks for rows 6–16 is not instrumented; the rows have no
  counterpart on TP1.
- **The counter could not see it.** Its stale flag was the same on every rank, and
  its CRC of consequential reads would have differed at read 5,279, but the next
  window, 6,144, never came.

**The repeat boot** (`shared-budget-ring2-sp`, 08:10 to 08:21Z, main `c2bb2ebb`; receipts
in `results/private/determinism/shared-budget-ring2-sp/`,
`results/private/suites/shared-budget-ring2-sp-single-stream/` and
`results/private/lab/shared-budget-ring2-sp-table.txt` on sp5) gave the same picture:
- the copy probe is identical (16.367 / 15.745 / 6.636 / 2.79; 4,550 / 4,566 tokens);
- the suite died: 29.10 tok/s, 5,271,210 failed, exit 1;
- asserts on sp5 (TP0, 352 lines) and sp7 (TP2, 226); none on sp6 (TP1), which lost
  the TCPStore at 08:18:08Z;
- the counter's last window, 6,144 reads, is at 08:18:01Z with equal CRCs; the crash
  is 7 s and 157 budgets later;
- **6,373 records on each rank; 6,372 steps equal; the last, 6,372, differs**, and
  in the same way:

| Step | TP0 (sp5) | TP1 (sp6) | TP2 (sp7) |
|---|---|---|---|
| 6,368 | 4 tokens, graph 4, scheduled 5, eligible 5, budget 3 | the same | the same |
| 6,369–6,371 | 6 tokens, graph 6, scheduled 5, eligible 5, budget 5 | the same | the same |
| **6,372** | **17 tokens, graph 17, scheduled 16, eligible 16, budget 16** | **6 tokens, graph 6, scheduled 16, eligible 5, budget 5** | **17 tokens, graph 17, scheduled 16, eligible 16, budget 16** |

Two boots, the same step shape (a 16-draft copy proposal after DSpark's block of 5),
the same lagging rank (TP1), the same two ranks asserting. The crash step is the only
divergent step in both, 5,340 and 6,373 steps in.

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

Both have run: `lab-repro.json` (*The repro on sp5–sp7*) and `lab-cost.json` (*The
cost run on sp5–sp7*). Both dry-run clean with `--production-config
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

- [lab-ring.json](lab-ring.json), run `shared-budget-ring-sp` (*The layout ring*): the
  repro's sequence on [arm-ring-c37-m8-nodead.json](arm-ring-c37-m8-nodead.json), one
  boot, with [layout_ring_logs.py](layout_ring_logs.py) after the suite and after each
  streams step. It dry-runs clean with the same profile. Not run.

**Acceptance criteria**, set before running, with where each stands:
- the minimal arm passes the repro with no assert or non-finite gate row, and its
  ranks' CRCs agree in every window: **met** (one boot);
- the paired single-stream verdict against candidate-37 is not worse beyond its CI:
  **met**, "same" at −0.08% ±0.09% over two fix boots, with the 0.30% floor (m=16);
- the copy probe's tool tokens per step are at least candidate-37's: **met** at m=8
  (6.905 against 6.636) and m=16 (6.232 on both boots against 6.083);
- 8 and 16 streams are within noise: **met** at m=16, with both fix boots at or above
  candidate-37's (157.50 and 157.89 against 157.36; 163.47 and 164.69 against
  162.61). At m=8 the repro's candidate-37 arm died before its streams, so there is
  nothing to pair them with.

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
docker run --rm --gpus all -v $PWD:/x:ro \
  -v $PWD/overlay/layout-ring/vllm/v1/worker/gpu/model_runner.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/model_runner.py:ro \
  -w /opt/spark-ds41f/candidate/vllm --entrypoint python3 spark-ds41f:candidate-37 /x/layout_ring_check.py
```

The dry runs:

```bash
scripts/lab.py run experiments/2026-11-03-copy-drafter-shared-budget/lab-repro.json \
  --dry-run --production-config config/candidate-37-tp3-switched-dev.local.json
scripts/lab.py run experiments/2026-11-03-copy-drafter-shared-budget/lab-cost.json \
  --dry-run --production-config config/candidate-37-tp3-switched-dev.local.json
scripts/lab.py run experiments/2026-11-03-copy-drafter-shared-budget/lab-ring.json \
  --dry-run --production-config config/candidate-37-tp3-switched-dev.local.json
```

## Not verified

- **The repro's repeat:** one boot per arm so far. The minimal arm's clean run is a
  single boot, so its absence of a crash bounds a rate rather than proving zero.
- **The cost's repeat:** candidate-37 had one boot of each kind in the cost run, so
  its side of the 8- and 16-stream comparison has no interval across boots.
- **The kernel that first produces the non-finite value inside a stale-cap step is
  not named.**
  - Live placeholder rows and a stale read alone are ruled out (above).
  - Per-rank divergence coincided with early non-finite rows under the census's
    timing (*The metadata census*), but the plain repro crashed without it.
  - The parent session's lead, a live row whose metadata leaves it nothing to attend
    to, is untested on real steps. Two metadata censuses kept only startup or warm-up
    batches, and the third died a minute in (*The reruns*).
  - A per-rank row census cannot see a cross-rank layout mismatch: each rank's rows
    can be valid on their own while the ranks lay out different steps.
  - The cross-rank comparison has since run (*The layout ring*): the crash step was
    laid out differently on the ranks, with different graph sizes, and no other step
    was. The first non-finite value is made where the shards' rows are combined for a
    step they did not lay out alike; which collective delivers what to the rows the
    small rank never computed is not instrumented. A run with synchronized launches
    would name that collective; it is a follow-up outside #860, and the fix does not
    depend on it.
  - The repro's "ranks agreed" holds through 8,192 budgets, the last counter window
    before the assert, and the reverted arm's through 3,072. The ring run's agreed
    through 5,338 and diverged at 5,339, the crash step; the repeat boot's through
    6,371 and diverged at 6,372, its crash step.
  - **Why TP1 lags** in both boots is not measured: its mirror held the older limit
    each time. A host difference between sp6 and the others (its copy lands later, or
    its budget runs earlier) is the reading.

  Until the kernel is named, the minimal fix is known to avoid the crash on this
  sequence, but not why.
- **diag4's death:** whether its instrumentation or the stale cap caused it is
  undetermined. Rank 1 went silent with no error logged.
- **Divergence without instrumentation:** not measured. The census's work split the
  ranks' reads; whether ordinary host timing can do the same is open.
- **The flagged layers past the receipt's 400-line cap,** and the later census
  reads: not in the receipts.
- **That the shared budget builds no such row:** not shown. A finite-garbage row
  would leave no trace in the minimal arm's run.
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

It costs nothing measurable. At m=16 with the recipe's dead rows, single-stream
speed is the same as candidate-37's (−0.08% ±0.09%, where #856's wait cost 0.73%), 8
and 16 streams are no slower, and the tool probe keeps 6.232 tokens per step against
6.083.

What the repro alone does not support is the reason given for the fix:
- candidate-37's ranks agreed there;
- the minimal arm's live placeholder rows were harmless.

The diagnosis jobs since add three things:
- a stale read alone does not produce a non-finite row;
- reverting 0676 makes the same stale sequence assert sooner;
- with the census's work each step, candidate-37's ranks did budget differently on
  hardware, and non-finite gate rows appeared in both forwards within 2,048 steps.

The shared budget rules that divergence out by design: its layouts do not read the
limits, and its ranks' layout CRCs were equal in every window.

The layout ring then measured the mechanism, twice: candidate-37 crashed again in both
boots, and each crash step was the only one of its run (5,340 and 6,373 steps) laid
out differently across the ranks. A copy proposal
of 16 drafts met mirrors that said 16 on two ranks and 5 on the third, so two ranks
ran the 17-token graph and one the 6-token graph for the same scheduler output, and
the two asserted in `route_partials`. There is no single-rank NaN producer to name:
the first non-finite value is made where the shards' rows are combined for a step
the ranks did not lay out alike. The shared budget removes that step by
construction, and its ranks' layouts were equal in every window of its runs.
