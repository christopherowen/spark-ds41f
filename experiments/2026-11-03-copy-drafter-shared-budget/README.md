# Copy drafter: one budget for every rank (#852)

Owner, 2026-11-03: "I don't like the performance loss, we should remove the
device-side fix and do this correctly." Then: "get to the point you are ready for
testing, stop and show me where it is."

**Status:** the patch is ready and unit-tested in candidate-37's image on sp8, and
the lab specs are prepared and dry-run clean. Nothing has run on sp5–sp7; that
waits for the owner. Not promoted; no recipe change.

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

## Quality implications

- **More placeholder rows.** The budget may verify more rows than the latest
  proposal has. When a proposal narrowed below the rows the scheduler laid out (a
  rejected or lost copy), the step verifies its `-1` placeholders wherever the
  planned confidences pay for them. Verification rejects them, so the committed
  tokens are the target's.
  - With dead rows on (the recipe's `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2`) they have
    confidence 0 and are padding, which the router skips.
  - **With dead rows off they are routed live**, more often than on candidate-37,
    which routed them only on a stale read. If live placeholder rows are what
    produce #852's non-finite gate logits, `minimal-m8-nodead` will show it.
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

## Prepared, not run

Both specs dry-run clean with `--production-config
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

  Outcomes:
  - Candidate-37 asserts, or its ranks' CRCs disagree, while the minimal arm stays
    clean with agreeing CRCs: the rank-dependent read was #852's mechanism, and this
    fixes it.
  - The minimal arm asserts: live placeholder rows produce non-finite logits. Then
    the fix needs dead rows on, or a clamp.
- [lab-cost.json](lab-cost.json), run `shared-budget-cost-sp`: the paired
  single-stream suite for minimal-m16, c37-m16, minimal-m16, then the copy probe and 8
  and 16 streams in the same order. It runs at m=16 with the recipe's dead rows.
  - [arm-minimal-m16.json](arm-minimal-m16.json) mounts
    [overlay/minimal](overlay/minimal/)'s `model_runner.py`, byte-identical to patch
    1's, and nothing else.
  - [arm-c37-m16.json](arm-c37-m16.json) mounts nothing.

**Acceptance criteria**, set before running:
- the minimal arm passes the repro with no assert or non-finite gate row, and its
  ranks' CRCs agree in every window;
- the paired single-stream verdict against candidate-37 is not worse beyond its CI;
- the copy probe's tool tokens per step are at least candidate-37's;
- 8 and 16 streams are within noise.

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

- **Anything on sp5–sp7:** the repro, the cost, real tensor-parallel ranks. Those are
  the two specs, waiting for the owner.
- **Rank agreement on real ranks:** the unit test simulates two ranks on one GPU;
  sp8 has one GPU, so the two-GPU test skips. The repro's per-rank CRCs are the
  first multi-rank evidence.
- **Whether live placeholder rows produce non-finite logits** on our cluster, with
  dead rows off. `minimal-m8-nodead` shows it.
- **Patch 2** is unit-tested only. It is not measured and not in any arm.
- **#859's clamp cost:** never measured (cancelled, above).

## Conclusion

Ready for testing. The patch removes the rank-dependent input from the budget
without a host wait or a clamp; unit tests show the layouts then agree across
simulated ranks, and the text is the target's through copy growth, shrinkage and slot
reuse. Whether it is safe with dead rows off and costs nothing on sp5–sp7 is what the
two specs measure.
