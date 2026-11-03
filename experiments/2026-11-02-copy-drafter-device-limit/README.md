# Copy drafter: rows past the device's draft limits are dead (#852)

Owner, 2026-11-02: implement the GPU-side alternative to #856's blocking sync, keeping
patch 0612's non-blocking host limits.

**Status:** patch ready and unit-tested on candidate-37's image on sp8. The lab specs
are prepared and dry-run clean but not run: they need an owner-authorized window on
sp5–sp7. Not promoted; no recipe change.

- **Base:** candidate-37, vLLM patch head `5b7cfa37f` (tree `af0fb152`). Image
  `spark-ds41f:candidate-37` (`sha256:e1ee7b1953b1…` on sp5, sp6, sp7 and sp8).
- **One variable:** with copy drafts on, the draft combine kills the verification
  rows past each slot's device-side draft limit (`CopyDrafter.slot_draft_limits.gpu`),
  with dead rows on or off. The host read stays 0612's: nothing waits.
- **Patch:** [vllm/0001-fix-copy-drafts-kill-verification-rows-past-the-devi.patch](vllm/0001-fix-copy-drafts-kill-verification-rows-past-the-devi.patch).
  It is vLLM `cc2b18de2` on `5b7cfa37f`, tree `afa120c4`, branch
  `copy-device-limit` in `/tmp/vllm-device-limit` on sp8. It applies cleanly to a
  fresh `5b7cfa37f` tree and would go next in the series after 0676, in place of
  #856's patch.
- **Alternative to** [#856](https://github.com/christopherowen/spark-ds41f/pull/856)
  (vLLM `8bb508091`): its host wait fixed the operator's crash but costs −0.73% ±0.12%
  in paired single-stream decode on sp5–sp7 (lab run `limit-sync-cost-sp`).
- **Design:** Jordi Posthumus, the operator of #852. They found the cause, measured
  it, and proposed this device-side clamp in #855 (commit `c30d6b6`): the device
  limits decide, the rows past them are dead, and those rows take the last sampled
  token as their input id. This experiment completes it (below, *The operator's
  patches*). The commit message credits them.

## The bug

From [experiments/2026-11-02-copy-drafter-limit-sync](../2026-11-02-copy-drafter-limit-sync/)
and #852:
- Patch 0612 made `CopyDrafter.sync_slot_draft_limits` publish the host's per-slot
  draft limits only once `_limits_event.query()` reports their copy landed. Until
  then the adaptive budget reads the previous proposal's limits.
- A stale limit wider than the current proposal admits rows the proposal doesn't
  have, the `-1` placeholders (`adaptive_verification.py`, `eligible_drafts =
  np.minimum(scheduled_drafts, slot_limits[slots])`).
- With dead rows off (`SPARK_DS41F_DSPARK_DEAD_ROWS_TAU` unset or 0, the code's
  default) they are routed live, and `route_partials` asserts on non-finite gate logits
  (`router.py:973`). With the recipe's 0.2 their confidence of 0 makes them dead rows,
  which the router skips.

## The change

The device's limits are always the current proposal's. `copy_alloc` writes
`slot_draft_limits.gpu` in `prepare_alloc`, before the drafter runs, on the main
stream, which every later step's input preparation runs on. The patch hands that
buffer to the draft combine:

- `DeadRows` gains `limits` (the device buffer), and `log_tau` may be None (no
  survival threshold). `AdaptiveVerificationManager.dead_row_args` returns dead rows
  whenever the manager has device limits, that is, whenever copy drafts are on.
- The combine kernel (`tilelang_inputs.combine_sampled_and_draft_tokens_kernel`) takes
  a `draft_limits` variant. A decoding request's cut is at most its slot's limit:
  - with dead rows, the survival cut clamped to the limit;
  - without, the drafts up to the limit, a thread a request as the plain combine.

  Rows from the cut on are padding (`is_padding`, which `route_partials` and the MoE
  dispatch skip). Rows past the limit take the request's last sampled token as their
  input id instead of `-1`. The logits indices and the batch shape are unchanged.
- The cut then clamps what the request commits, on the paths the cut already serves:
  - the token counts (greedy verification's commit, and every sampler's);
  - block verification's end of block;
  - the DSpark acceptance estimator's observed rejections;
  - the copy state's update (unchanged where it matters: a copy's limit is its pending
    drafts).

  Whatever the target makes of a row past the limit, its token is never accepted.
- Kernel declarations: `declare_step_kernels` declares the cut variants (counts,
  rejection, insertion) and the combine's `(dead rows, draft limits)` variant whenever
  copy drafts are on with adaptive verification.

Not changed: the host budget, the layout, the scheduler, the copy kernels, 0612's
non-blocking read and its test.

**CUDA graphs.** The combine runs eagerly before the graph replays. It writes the
persistent `input_buffers.is_padding` that the graphs captured, and
`padding_rows` resets it each step before the combine (`VLLM_MOE_SKIP_PADDING`, on by
default). No host sync, no allocation.

**Batch-sharded sampling.** Adaptive verification refuses it
(`VllmConfig._validate_batch_sharded_sampling`: the shard plan is built from host
logits counts that adaptive verification decides on the device), and the host and
device draft limits exist only with adaptive verification. So no step has both. The
runner's existing guard (no dead rows with a batch sharder) stays.

**A refusal.** A configuration with copy drafts whose block verification runs on its
Triton kernels (FP64 draws or a watermark) is now refused at kernel declaration, as
dead rows already were: those kernels can't end a block at a cut. Copy drafts already
require greedy drafting; neither option is in the recipe.

## Quality

- **Bit-identical where the limits are current.** If the host read the current
  proposal's limits, no row is past the device limit. Every output is then
  candidate-37's, bit for bit:
  - input ids, padding, logits rows and offsets;
  - the commit counts, since the cut equals the request's drafts and no clamp binds.

  The tests check this with dead rows off and on, over a copy and its doubling.
- **Stale and wider:** the rows the host budgeted past the proposal are padding, and
  the request commits at most the proposal's drafts and the bonus. Temperature-0 text
  is unchanged; those rows only cost their (MoE-free) compute.
- **Stale and narrower:** the device limit can only remove rows the host scheduled, so
  a stale narrow limit still leaves drafts out, as on candidate-37. A unit test shows
  it by construction; the repro arms' counters will measure it on the cluster. #856's
  wait fixes it; this change does not.

## Tensor-parallel invariance

The marking is a pure function, on each rank, of:
- the step's layout (`cu_num_logits`, `query_start_loc`);
- `seq_lens` and `prefill_len`;
- the request state's drafts and last sampled tokens;
- the broadcast confidences;
- `slot_draft_limits.gpu`.

`slot_draft_limits.gpu` is written only by `copy_alloc` (and reset by
`apply_staged_writes` for a new request). Its inputs:
- `idx_mapping`: slots are assigned alike on every rank;
- `total_len` and the history `copy_match` scans: committed tokens, alike;
- `num_sampled`;
- the admitted drafts (`verified`) and the cut;
- `int_state`.

The kernels have no atomics and no rank-dependent input. `verified` and the cut come
from earlier steps' layouts. So as long as the ranks' layouts have agreed, every rank
computes the same limits and marks the same rows. This is by reading only: sp8 has one
GPU, and the two-GPU tests skip there.

**Open, and not caused or cured by this change:** the layout itself.
- Each rank's host runs `get_num_tokens` with its own read of 0612's mirror, and
  nothing makes the ranks agree on it. Whether a copy has landed at that moment can
  differ by rank.
- If it differs at a step where a slot's limit changed (a copy starting, doubling or
  ending), the ranks' eligible drafts differ. Their draft budgets, token counts and
  per-request capacities can then differ too.
- The confidence snapshot was made TP-wide for exactly this reason:
  `test_confidence_snapshot_agrees_before_cpu_and_gpu_budgeting` and
  `test_tp_confidence_publication_keeps_graph_and_request_budgets_consistent`. The
  limits' read has no such guarantee since 0612.
- #856's wait removes the hazard. So would a host budget that reads only the
  scheduler's layout and the planned confidences, which every rank holds alike.
- The repro arms below log, per rank, a CRC of the budget reads whose stale limits
  changed eligible drafts. Equal CRCs across sp5, sp6 and sp7 mean the ranks read
  alike; unequal ones locate a divergence.

## The operator's patches (#855, reviewed, not run)

Their branch `JordiPosthumus:copy-limit-sync`, commit `c30d6b6`, holds
`experiments/2026-11-02-copy-drafter-limit-sync/repro/patches/`. Each patch was
read, none was run.

- **0004, the device clamp.** This experiment builds on its design. It has two parts:
  - `CopyDrafter.propose` zeroes `confidences` at columns from
    `slot_draft_limits.gpu[idx_mapping]` on. This changes nothing:
    - `copy_select` already writes confidence 0 past `offered` (a copy) and past the
      block (DSpark), and the device limit is exactly `offered` or the block.
    - The mask adds a gather, a compare and a masked fill to every step.
    - Its test sets the tail to 0.9 before `propose` and asserts 0 after. But
      `copy_select` overwrites every column below the lookahead, so the assertion
      holds on candidate-37 too.
  - The combine writes the input ids after the dead-row cut, the last sampled token
    for every row from the cut on, but only with dead rows.
    - Without dead rows (the threshold unset or 0, or a batch sharder) the
      combine is candidate-37's. The placeholder rows still route live with `-1`
      ids: #852's configuration.
    - With dead rows, candidate-37 already kills those rows (their confidence is
      0). The change is then the ids of dead rows, including rows dead by survival
      inside the proposal, whose bits now differ from candidate-37's.
    - It also moves the stores behind the survival scan, which the kernel had
      issued first so they wait on nothing.

  This version clamps the cut to the device limit in both kernels and rewrites
  only the ids past the limit. It masks no confidences. Its stale-wide test fails
  on candidate-37 (*Evidence*).

  Their validation ran one run per arm at m=8: 8 concurrent streams, a 10,456-token
  prompt, host sync 126.05 tok/s, device clamp 159.90. That is not #852's
  deterministic sequence (copy probe, then the single-stream suite), and their
  dead-row setting isn't recorded. With the threshold at 0, 0004 leaves the combine
  as candidate-37's, so a clean run there doesn't isolate the clamp.
- **0002, the stale-limit counter.** It prints on every `sync_slot_draft_limits` call
  that finds the copy in flight, from any caller (`add_request`, the budget,
  `prepare_alloc`), and on an alloc-ring reuse. Its 1,209 therefore counts calls,
  not budget reads. Nor does it say whether a read changed eligible drafts.
  [overlay/limit-reads](overlay/limit-reads/) counts the budget's reads only, and
  resolves each against the copy once it lands.
- **0003, the `-1` → `0` control.** It pads `copy_select`'s drafts with 0, and the m=8
  repro still crashed. Placeholder rows routed live don't explain that well: with an
  id of 0 at a valid position, a live row is an ordinary token.

  A rank-dependent layout (*Tensor-parallel invariance*) would explain it.
  - At one stream, two ranks' layouts agree up to the shorter one's rows and differ
    past them. The first rows past the shorter layout are the placeholder rows:
    #849's block [4] is rows 8–9, and draft 8 is the first placeholder of an
    8-draft copy.
  - The tensor-parallel reductions then sum different rows there.
  - If so, this clamp keeps those rows from the router at one stream, but not the
    misaligned rows of later requests at several streams. Only a layout every rank
    decides alike fixes that: #856's wait, or a budget without the racy read.

  This is a hypothesis, not a measurement. The repro arms' per-rank CRCs test it.

## The census hides the race

#842's census overlay reads its flag with `.item()` after sampling and after the
drafter, every step (`model_runner.py` `_log_nonfinite_rows`). That host sync waits
until the drafter has finished, by which time the limit copy, queued before the
drafter, has landed. The next budget read is therefore fresh, and the stale read
#852 needs does not happen.

This bears on two records:
- every #852 repro arm of the sp5–sp7 window (`spark-0013-1793639688`) ran with the
  census, besides dead rows at 0.2;
- #856's prepared `lab-repro.json` pairs the census with dead rows off.

So this experiment's repro arms carry no census. The device assert is the detector,
and a host-only counter (below) measures the race without perturbing it.

## Evidence

### Unit tests (sp8, candidate-37's image, changed files mounted)

sp8 was idle before the runs (no containers, GPU at 0%); they ran one at a time,
2026-11-02 23:00–23:55 UTC (sp8's clock). No timing was taken. The commands are under
*Commands*; the summaries are in [runs/](runs/).

| Tests | Candidate-37 | This change |
|---|---|---|
| `test_copy_drafts.py`, `test_copy_drafts_scheduler.py`, `test_adaptive_verification.py` (#856's set) | 69 passed, 1 skipped | **75 passed, 1 skipped** |
| `test_dead_rows.py`, `test_gpu_tilelang_inputs.py`, `test_prepare_kernels.py`, `test_gpu_input_batch_v2.py`, `test_gpu_verified_logprob_token_ids.py`, `test_rejection_sampler_utils.py`, `test_tilelang_rejection.py` | 1,572 passed, 1 skipped | **1,584 passed, 1 skipped** |

The skips are the two-GPU TP test (sp8 has one GPU) and a combine case without a
budget below its scheduled drafts. None fails. The new tests:
- `test_copy_drafts.py`, through a real `CopyDrafter` and `AdaptiveVerificationManager`
  on one rank, each verification step as the runner prepares it (host budget, device
  allocation, combine, commit count):
  - `test_stale_wide_host_limit_routes_no_placeholder_row[0.0, 0.2]`: the host holds 16
    (an in-flight copy, read without waiting) for an 8-draft copy. The step verifies 17
    rows. With the fix, rows 9–16 are padding with the last sampled token as input,
    the cut is 8, and a target that accepted every row still commits 9. The test
    also shows candidate-37's step with dead rows off: the same rows live, with `-1`
    ids. That is #852's path.

    The test fails on candidate-37. Run there with the device limits handed to the
    manager after construction (candidate-37's constructor has no such argument),
    it fails at row 9 (`-1 != 50300`), with dead rows off and on
    (`runs/unit-c37-stale-wide.txt`). Run unchanged there, it fails on the
    constructor argument.
  - `test_current_host_limits_leave_the_step_bit_identical[0.0, 0.2]`: over a copy
    and its doubling (8 then 16 drafts), the fix's rows, input ids, logits rows,
    offsets, padding and commit count equal candidate-37's (and the cut, with dead
    rows).
  - `test_stale_narrow_host_limit_still_under_admits`: the host holds 8 while the
    device's limit is 16 after the cap doubled. The step verifies 8 drafts and marks
    nothing: the under-admission stays.
  - 0612's `test_stale_wide_allocation_rejects_placeholder_tail_and_doubles_cap_without_sync`
    is unchanged and passes: `sync_slot_draft_limits` never blocks.
- `test_gpu_tilelang_inputs.py`:
  - `test_combine_draft_limits[num_reqs 1, 3, 8, 16 × dead rows off, on]`: random,
    zero and narrow limits against a host reference of the combine without them, at 16
    drafts. Limits no request reaches give the same ids, padding and logits indices,
    and the cut is the survival cut or the drafts.
  - `test_declared_combine_variants`: the combines a step launches are the declared
    ones, for each of the four (dead rows, draft limits) settings.
- `test_adaptive_verification.py`: `test_dead_row_args_carry_the_device_draft_limits`.
- `test_prepare_kernels.py`: the declaration test gains copy drafts with dead rows off.
  The cut variants (counts, rejection) and the combine's draft-limit variant are then
  declared.

The counters of the repro arms' overlay were checked the same way.
[limit_reads_smoke.py](limit_reads_smoke.py), with
`overlay/limit-reads/…/copy_drafts.py` mounted, drives the real budget through:
- a stale read that resolves one slot wide (8 placeholder rows) and one narrow (11
  dropped drafts);
- an unresolved read;
- the periodic line.

It printed `COPY-LIMITS reads=1024 queued=2 stale=2 resolved=1 unresolved=1
consequential=1 wide_slots=1 placeholder_rows=8 narrow_slots=1 dropped_drafts=11
crc=6522df69`. #856's three test files (candidate-37's own) pass with the overlay
mounted: 69 passed, 1 skipped.

`bin/spark --cluster-config <profile or arm> doctor` (sp5's node map): configuration OK
for the serving profile and all four arms. `bin/spark build check` was not run: no
build input changes.

### The interaction with #842

PR #842's CopyDrafter rework (`exp/dspark-default-copy-drafts` at `8299c8e0`, vLLM
`0c017c680`) is rebased on #856 (`8bb508091`). If this change replaces #856, #842 is
rebased onto this patch instead:
- `copy_drafts.py`: #842's hunks were written over #856's version (the waiting
  sync, no `_pending_limit_resets`). On this patch they meet 0612's version again, so
  #842's copy_drafts.py and its #856-era tests (`test_add_request_resets_its_slot_after_the_queued_copy`)
  need reworking to 0612's reads.
- `model_runner.py`: #842's hunk that drops the copy drafter without adaptive
  verification follows this patch's new `device_draft_limits=` argument directly; a
  context conflict, resolved by keeping both.
- #842 runs `prepare_alloc` on the output copy stream, and `propose` makes the main
  stream wait for it. The device limits are therefore still written before the next
  step's combine reads them.
- #842 declares the combine's variants before it may drop the copy drafter (a
  checkpoint without a confidence head). The `draft_limits` variant is then declared
  and never launched, which is harmless.
- With copy drafts the default (#842), every greedy adaptive DSpark deployment runs
  the `draft_limits` combine.

### The race on sp5–sp7 (#856's cost run)

These are not this experiment's runs; the numbers come from the receipts of #856's
census-free run `limit-sync-cost-sp` on sp5
(`results/private/determinism/limit-sync-cost-sp/copy-probe-*.jsonl`). The copy probe
at m=16, with the recipe's dead rows, in tokens per step:

| Arm | copy | edit | tool | prose |
|---|---|---|---|---|
| candidate-37 (`c37-m16-c`) | 16.021 | 15.322 | 6.083 (84 drafts) | 2.651 |
| #856's wait (`fix-m16-c`, `fix-m16-c-repeat`) | 16.021 | 15.322 | 6.232 (82 drafts) | 2.651 |

These are the operator's m=16 numbers. The host's limit read goes stale on sp5–sp7
too, and a stale narrow read drops tool drafts there. This change leaves that loss
in place; #856 removes it. Its census-free repro below measures how often each
happens.

## Prepared, not run

The specs dry-run clean with `--production-config
config/candidate-37-tp3-switched-dev.local.json`, the profile sp5–sp7 serve (sha
`92b1712d…`, node map `config/nodes-switched3.local.json` sha `0c508533…`, copied from
sp5 for the dry runs and not committed). They need an owner-authorized window.

- [lab-repro.json](lab-repro.json), run `device-limit-repro-sp`: #852's sequence on
  one boot of each arm, dead rows off, min match 8. The steps:
  - copy probe;
  - the single-stream suite ([suite_single_stream.py](suite_single_stream.py), #856's);
  - the limit reads ([limit_reads_logs.py](limit_reads_logs.py));
  - 8 streams, then the limit reads;
  - 16 streams, then the limit reads.

  Arms: [arm-c37-m8-nodead.json](arm-c37-m8-nodead.json), the expected reproduction,
  and [arm-fix-m8-nodead.json](arm-fix-m8-nodead.json), the fix. Both mount
  [overlay/limit-reads](overlay/limit-reads/)' `copy_drafts.py`, candidate-37's file
  plus host-only counters. It changes nothing the budget reads and adds no GPU sync.
  Every 1,024 budget reads, each rank logs a `COPY-LIMITS` line:
  - reads, and reads with a copy queued;
  - stale reads, resolved and unresolved;
  - consequential reads: those that, once the copy landed, had changed a scheduled
    slot's eligible drafts;
  - wider slots and their placeholder rows, narrower slots and their dropped drafts;
  - the CRC of the consequential reads' indices.

  Outcomes:
  - Candidate-37 asserting (or logging consequential wide reads) while the fix stays
    clean with the same counts confirms the mechanism on our cluster, and the fix.
  - Neither asserting, with zero stale reads, means the race doesn't happen here.
  - The fix's narrow slots and dropped drafts measure the under-admission it leaves.
- [lab-cost.json](lab-cost.json), run `device-limit-cost-sp`: the fix's cost at
  m=16 with the recipe's dead rows, no overlays besides the fix. It mirrors #856's
  `limit-sync-cost-sp`:
  - the single-stream suite (the lab's paired verdict);
  - then the copy probe and 8 and 16 distinct streams.

  The arm order is fix, candidate-37, fix again, so each comparison has a repeat boot:
  [arm-fix-m16.json](arm-fix-m16.json) against [arm-c37-m16.json](arm-c37-m16.json).
  The candidate-37 arm is #856's `arm-c37-m16.json` (same serve options).

- [lab-repro-rank.json](lab-repro-rank.json), run `device-limit-repro-rank-sp`
  (parent session's request, 2026-11-03; queued only if the owner says so). It tests
  the rank-dependent layout hypothesis (*The operator's patches*, 0003). It has one
  arm, [arm-rank-m8-nodead.json](arm-rank-m8-nodead.json): the fix with the host
  budget reading no host limits.
  - The overlay is [overlay/fix-rank](overlay/fix-rank/)'s `model_runner.py`: the
    fix's, with `slot_draft_limits=None` at the manager. Every rank then budgets from
    the scheduler's drafts and the planned confidences, which they all hold alike.
    The device limits clamp whatever the budget admits past a proposal.
  - The stale-read counter has nothing to count here, so the arm doesn't mount it.
    Instead each rank logs a CRC of its budgeted layouts every 1,024 budgets: the token
    count and every request's eligible drafts, as `COPY-LIMITS reads=N layout=1
    crc=…`. The same extras and table parse it.
  - It is a second variable on top of the fix, so it gets its own spec.
  - Mounts, environment, serve options and extras are `fix-m8-nodead`'s, apart from
    the model runner and the counter. So it compares directly with the two arms of
    `lab-repro.json`.

  Outcomes:
  - Candidate-37 crashes, the fix still asserts or its ranks' CRCs disagree at
    several streams, and this arm stays clean with agreeing CRCs: the layout is the
    mechanism, and the owner's choice is between #856's wait and this budget.
  - All three clean: either the race doesn't reach the router here, or this window's
    load didn't trigger it.
  - The arm also shows what a budget without the host's limit costs in rows. Its
    tokens per step and streams compare with the other arms', though cost needs
    `lab-cost.json`'s paired runs to decide.

  Checked on sp8 in the image, as for the fix:
  - [layout_census_check.py](layout_census_check.py): equal step sequences give equal
    CRCs, one changed draft count changes it, and the line comes every 1,024 budgets;
  - with the arm's five files mounted, `test_copy_drafts.py`,
    `test_adaptive_verification.py`, `test_prepare_kernels.py` and `test_dead_rows.py`
    give 76 passed, 1 skipped.

  `bin/spark doctor`: configuration OK.
- **The minimal fix** (parent session's request, 2026-11-03; prepared, not queued; the
  owner hasn't chosen). It is candidate-37 with only `slot_draft_limits=None` at the
  adaptive verification manager, and no device clamp:
  [overlay/none](overlay/none/)'s `model_runner.py`.
  - By diff, that file differs from candidate-37's only in that argument, `import
    zlib`, and the rank arm's host-only layout CRC (the method and its call after the
    budget).
  - The budget then comes from the scheduler's draft counts and the confidence
    snapshot. `record_confidences` waits for that snapshot and broadcasts it from rank
    0 (or it is rank-invariant by construction), so every rank lays out the same step
    without a wait.
  - It keeps candidate-37's placeholder handling: the `-1` drafts, their rejection, and
    dead rows.

  What changes, besides the layout:
  - Without the host cap, the step verifies the scheduled rows whenever the planned
    confidences pay for them. That includes the `-1` placeholders of a proposal that
    narrowed after a rejection or a lost copy, not only on a stale read.
  - With the recipe's dead rows those rows have confidence 0 and are dead (padding).
    With dead rows off they route live, more often than on candidate-37.
  - So `none-m8-nodead` separates the hypotheses sharply. If placeholder rows routed
    live are the mechanism, it should crash sooner than candidate-37; if the layout
    is, it stays clean with agreeing CRCs.
  - As a fix it relies on dead rows being on. A deployment that leaves the threshold
    unset would be more exposed than on candidate-37.
  - A stale narrow read can no longer drop drafts: there is no host cap.

  Its arms:
  - [arm-none-m8-nodead.json](arm-none-m8-nodead.json): the threshold at 0, m=8.
  - [arm-none-m16.json](arm-none-m16.json): the recipe's dead rows, m=16. Only the CRC
    separates it from candidate-37 plus the one argument. The CRC adds no device sync
    and costs 1.2–1.7 µs of host time a budgeted step at 1 to 16 requests
    ([layout_census_cost.py](layout_census_cost.py)), under 0.02% of a step, so it
    stays in.

  Both mount only that file. Their serve options are `c37-m8-nodead`'s and `c37-m16`'s.

  Its specs, both dry-run clean, with doctor OK for both arms:
  - [lab-repro-none.json](lab-repro-none.json), run `device-limit-repro-none-sp`:
    `c37-m8-nodead` (the counter arm, unchanged), then `none-m8-nodead`, with
    `lab-repro.json`'s extras.
  - [lab-cost-none.json](lab-cost-none.json), run `device-limit-cost-none-sp`:
    `lab-cost.json`'s jobs with none-m16, c37-m16, none-m16 (the paired single-stream
    suite), then the same order for the copy probe and 8 and 16 streams.

  Tests (sp8, candidate-37's image; [runs/unit-none.txt](runs/unit-none.txt)):
  - [test_none_rank_budgets.py](test_none_rank_budgets.py), on candidate-37's own vLLM:
    2 passed. It sets up two ranks with the same histories, proposals and
    confidences; rank 0's limit copy has landed and rank 1's is in flight.
    - Under candidate-37's wiring, rank 0 caps the copy at its 8 drafts and rank 1 at
      the stale 16, so their token counts and query starts differ.
    - Under the none wiring, the token count, budget, query starts and logits offsets
      are equal.
  - Candidate-37's `test_copy_drafts.py`, `test_copy_drafts_scheduler.py`,
    `test_adaptive_verification.py` and `test_prepare_kernels.py`: 77 passed, 1
    skipped with overlay/none mounted, the same as without it.

The fix arms mount [overlay/fix](overlay/fix/): the patch's five vLLM files,
byte-identical to the patched tree, over the shared image, so every rank keeps one
digest.

## Commands

The tree (sp8; any clone that has `5b7cfa37f`):

```bash
git init fresh && cd fresh
git fetch --depth 1 --no-tags /path/to/vllm 5b7cfa37fee42f96bfafbe09b7d30e8c0eda8575
git switch -c copy-device-limit FETCH_HEAD                    # tree af0fb152
git am --committer-date-is-author-date \
  experiments/2026-11-02-copy-drafter-device-limit/vllm/0001-fix-copy-drafts-kill-verification-rows-past-the-devi.patch
git rev-parse HEAD^{tree}                                      # afa120c4
git diff --check HEAD~1 HEAD
```

The unit tests: [run_image.sh](run_image.sh) runs a command in
`spark-ds41f:candidate-37` with the tree's files changed since `5b7cfa37f` mounted
read-only over `/opt/spark-ds41f/candidate/vllm`, and the tblib stand-in
(`scripts/release_tests/tblib_stub.py`) on `PYTHONPATH`. Candidate-37's numbers are the
same commands in the image with nothing mounted.

```bash
TREE=/tmp/vllm-device-limit PLUGINS=/tmp/loop-test-plugins ./run_image.sh \
  "python3 -m pytest -q -p no:cacheprovider -p tblib_stub -rfEs \
   tests/v1/spec_decode/test_copy_drafts.py tests/v1/core/test_copy_drafts_scheduler.py \
   tests/v1/spec_decode/test_adaptive_verification.py"
TREE=/tmp/vllm-device-limit PLUGINS=/tmp/loop-test-plugins ./run_image.sh \
  "python3 -m pytest -q -p no:cacheprovider -p tblib_stub -rfEs \
   tests/v1/worker/test_dead_rows.py tests/v1/worker/test_gpu_tilelang_inputs.py \
   tests/v1/worker/test_prepare_kernels.py tests/v1/worker/test_gpu_input_batch_v2.py \
   tests/v1/worker/test_gpu_verified_logprob_token_ids.py \
   tests/v1/spec_decode/test_rejection_sampler_utils.py tests/v1/spec_decode/test_tilelang_rejection.py"
```

The lab specs' dry runs (this checkout, with the two node-local files copied from sp5):

```bash
scripts/lab.py run experiments/2026-11-02-copy-drafter-device-limit/lab-repro.json \
  --dry-run --production-config config/candidate-37-tp3-switched-dev.local.json
scripts/lab.py run experiments/2026-11-02-copy-drafter-device-limit/lab-cost.json \
  --dry-run --production-config config/candidate-37-tp3-switched-dev.local.json
scripts/lab.py run experiments/2026-11-02-copy-drafter-device-limit/lab-repro-rank.json \
  --dry-run --production-config config/candidate-37-tp3-switched-dev.local.json
scripts/lab.py run experiments/2026-11-02-copy-drafter-device-limit/lab-repro-none.json \
  --dry-run --production-config config/candidate-37-tp3-switched-dev.local.json
scripts/lab.py run experiments/2026-11-02-copy-drafter-device-limit/lab-cost-none.json \
  --dry-run --production-config config/candidate-37-tp3-switched-dev.local.json
```

The two-rank test runs in the image with nothing else mounted:

```bash
docker run --rm --gpus all --ipc=host -v $PWD:/x:ro -w /opt/spark-ds41f/candidate/vllm \
  --entrypoint python3 spark-ds41f:candidate-37 -m pytest -q -p no:cacheprovider \
  /x/test_none_rank_budgets.py
```

The overlay checks run in the image with the image's own `PYTHONPATH`, which puts
`/opt/spark-ds41f/candidate/vllm` ahead of the installed copy. With `PYTHONPATH`
overridden, a script imports the installed copy; pytest, run from the tree, does not.
For example, with the rank arm's model runner over the fix's files:

```bash
docker run --rm --gpus all -v $PWD:/x:ro -v $RANK_MODEL_RUNNER:$V/vllm/v1/worker/gpu/model_runner.py:ro \
  <the fix's other four files mounted likewise> -w $V --entrypoint python3 \
  spark-ds41f:candidate-37 /x/layout_census_check.py
```

After a window: `tables_arms.py RUN ARM...` prints each arm's probe, suite, limit reads
(with the ranks' CRC agreement) and streams.

## Memory

The patch allocates nothing: the device limits already exist, and the cut and padding
buffers are the dead rows'. No memory run was made; the lab-cost runs record startup
memory per arm.

## Not done

- **Our cluster measurements:** the two specs above.
- **The layout's TP agreement** under 0612's read (above). The repro arms' CRCs are
  its first evidence. Making it deterministic is its own variable.
- **The producer of the non-finite logits:** which kernel turns a routed placeholder
  row non-finite. The operator reports that ids of 0 instead of `-1` still crash.
  The fix makes those rows padding whatever the cause.
- **A layout every rank decides alike:** prepared as `lab-repro-rank.json` (with the
  clamp), and as the minimal fix's `lab-repro-none.json` and `lab-cost-none.json`
  (without it); none of them run.
  Its cost in rows against the fix is unmeasured. It admits every scheduled draft the
  planned confidences pay for, and the clamp pads the ones a narrowed proposal lacks.

## Conclusion

On the unit evidence, the device-side fix makes #852's rows safe without a host wait,
with dead rows on or off:
- the rows past the current proposal's drafts are padding with valid ids;
- the request commits at most those drafts and the bonus;
- where the host's limits are current, every step is bit for bit candidate-37's.

It leaves two things as candidate-37 has them:
- a stale narrow limit still drops drafts (the operator's 6.636 → 6.905 tool tokens
  per step came from #856 fixing that);
- the layout is still decided from each rank's own non-blocking read.

#856's wait fixes both, at −0.73% ±0.12% single-stream on sp5–sp7. This change should
cost nothing measurable: it changes no layout, and adds a load and a min per request
to the combine. The prepared runs measure that. #856's cost run already shows the
race happens on sp5–sp7. The choice between this, #856, or this plus a host budget that every
rank computes alike, is the owner's once they have run.
