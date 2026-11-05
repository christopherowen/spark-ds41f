# Copy drafts as the one behaviour of DSpark greedy adaptive verification (#842)

Owner's rule: if it is better it is the default; if it does not work it goes. Copy
drafts qualified in candidate-36 (36/36 sessions better than spark-ds41f-1030, 0
worse than candidate-34). They still ran only as separate arms
(`candidate-37/tp3-copy.json`, `2026-10-29-copy-drafts-c12/copy-tp4.json`), and
`SpeculativeConfig.copy_drafts` still defaulted to `None`. This experiment makes
them the only behaviour, with no option, and finishes the stalled rework of their
step work.

- **Base:** candidate-37: vLLM `19f2c20ed` plus
  `../2026-10-26-spark-ds41f-candidate/candidate-37/vllm/series`, patch head
  `5b7cfa37f`, tree `af0fb1525fc6aa4149b18dee7d8c3ae5120b6f33`. Image
  `spark-ds41f:candidate-37` (label `local.spark-ds41f.vllm.patch-head`
  `5b7cfa37...`), on dgx4 and sp8 (GB10, 48 SMs).
- **Patch:** [`0001-perf-dspark-copy-drafts-as-the-one-behaviour-of-gree.patch`](0001-perf-dspark-copy-drafts-as-the-one-behaviour-of-gree.patch),
  one commit (`9b8f05a35`, tree `949f829d6eba853ea7f53238669fe5d439782eda`) on
  that head plus [#860](https://github.com/christopherowen/spark-ds41f/pull/860)'s
  two commits, in order (`experiments/2026-11-03-copy-drafter-shared-budget/vllm/`
  on #860's branch):
  - `72abb31ff` (tree `58882ab9`), #852's fix: the adaptive budget reads no host
    draft limits (`slot_draft_limits=None`), so every rank budgets from the
    scheduler's drafts and the confidence snapshot;
  - `88982f34e` (tree `f473423c`): the per-slot draft limits are removed (host
    mirror, staging ring, event, `sync_slot_draft_limits`, device buffer and
    `copy_alloc`'s write).

  Its message states base, rationale, quality, tests and upstream status. This is
  candidate-39's #842, staged on candidate-38 (#860, #848, #853, #854). It
  replaces three earlier versions:
  - the earlier `0001-dspark-default-copy-drafts.patch` of this PR, which was
    written on candidate-27 and kept the option;
  - this PR's first candidate-37 version (`bc71d8191`, tree `9f9c6d73`, directly
    on `5b7cfa37f`), which still read the host limits without waiting; it is this
    version's starting point;
  - the rebase on #856 (`0c017c680`, tree `458b91a2`), which kept #856's
    blocking limit wait. The owner dropped #856's wait and #859's device clamp
    for their cost ("I don't like the performance loss").
- **Variable:** copy drafts on wherever DSpark verifies greedy drafts adaptively.
  At the cluster this is candidate-37's copy arms run from the plain recipe. The
  rework of the step's copy work is bit-identical in its outputs and is measured
  here at kernel level.

> **Status: rebased on #860, unit-tested; not measured at the cluster on that
> base.** Copy drafts on candidate-37 crash an operator's engine under real
> traffic (#852; below, *Unfavourable: the crash*). This patch computes the same
> proposals as candidate-37's copy arm. Its kernel and c1 results cannot clear it
> of the crash. An sp5-sp7 lab window running the operator's exact sequence with
> dead rows on did not reproduce it (*Lab window on sp5-sp7*). #860's window
> reproduced it on candidate-37 with dead rows off and ran #860's fix clean
> (its README has the runs). This patch now sits on that fix; its own cluster run
> comes with candidate-39.

## Unfavourable: the crash (operator's TP3 ring, 2026-11-02)

The source is JordiPosthumus's comments on #842 ([6581345928](https://github.com/christopherowen/spark-ds41f/pull/842#issuecomment-6581345928),
15:42Z; [6581566292](https://github.com/christopherowen/spark-ds41f/pull/842#issuecomment-6581566292),
15:51Z) and their closed #849. The setup was unpatched candidate-37 on a 3-node
TP3 ring at 2200 MHz, with explicit `copy_drafts` and this PR's envelope (budget
6,155, capture to 59, sizes +56, 59).

- **m=16, synthetic suites: passed.**
  - `copy_probe` tokens per step: copy 5.082 → 16.014, edit 5.163 → 15.315,
    tool 5.268 → 6.083, prose level.
  - Single-stream closed loop: 64.897 → 64.683 tok/s (−0.33%, CI95 ±2.33%).
  - Token-identical to candidate-37 on greedy and copy-dense prompts.
- **m=16, real agent traffic: crashed.** A few minutes in the operator's gateway
  pool, then a device assert `route_partials: gate logits contain nan or inf` on
  worker 0, at 15:47:41Z. Ring2 was restored to plain candidate-37.
- **m=8, single-stream suite: crashed the same way.** Reproduced on two boots
  (#849: `cudaErrorAssert`, block [4]). The m=16 arm and the baseline passed that
  suite.

Not reproduced here: neither dev node (dgx4, sp8) has DS4.1 weights. What reading
candidate-37 establishes is under *Crash analysis* below. The rest needs the
operator's repro or a serving-cluster window.

The crash is tracked in [#852](https://github.com/christopherowen/spark-ds41f/issues/852)
(JordiPosthumus, 16:03Z), and the fix should close it. Their exact repro is
deterministic at m=8, on two boots:
- the `candidate-37-af0fb152` image, TP3, 2200 MHz;
- `copy_drafts={"min_match":8,"initial_drafts":8,"max_drafts":16}` at the #842
  envelope (budget 6,155, capture 59, sizes +56, 59);
- `copy_probe.py --prompts 3`, then servebench's `single-stream` suite. The suite
  dies in its warm-up with `route_partials: gate logits contain nan or inf`
  (`router.py:973`, worker 0, block [4]).

m=16 passes the probe and the suite with identical tokens, and dies only under
their real traffic. Plain candidate-37 (without copy drafts) passes. Nobody has
run candidate-37 with copy drafts at that envelope.

## The change

1. **No option.** `SpeculativeConfig.copy_drafts` becomes a derived field
   (`field(init=False)`): `CopyDraftsConfig()` (minimum match 16, 8 to 16 drafts)
   when `method == "dspark"`, `enable_adaptive_verification` and
   `draft_sample_method == "greedy"`. `--speculative-config` refuses
   `copy_drafts`, so the old arms' JSON fails loudly instead of being half-applied.
   Copy drafts are off only where the engine cannot carry them:
   - without async scheduling on the V2 model runner (`VllmConfig`, an info line);
   - with a drafter checkpoint that has no confidence head, hence no adaptive
     verification (the runner drops the copy drafter; it used to raise).

   The request-boundary-checkpoint refusal goes: `use_request_boundary_checkpoints`
   holds only for MTP/DFlash on hybrid models, never for DSpark. The
   pipeline-parallel refusal also goes: adaptive verification refuses PP itself.
2. **The recipe's sizing fits the copy envelope.** `scripts/recipe.py` sizes every
   shape for DSpark's uniform decode rows (`max_num_seqs × 6`). Copy drafts let one
   request verify 16 drafts, so `max_decode_rows` is 11 rows wider. `VllmConfig`
   now widens two things:
   - **Budget.** With aligned prefill chunks, a token budget equal to the
     threshold plus the uniform rows grows by the 11 rows: 6,144 → 6,155
     (`tp3-triangle`), 16,384 → 16,395 (`tp4-ring`). The encoder budgets grow
     with it. The threshold (6,096 / 16,288), and with it every chunk boundary,
     is unchanged.
   - **CUDA-graph range.** A capture range ending at the uniform rows extends on
     vLLM's default stride of 8, ending at the envelope: `[..., 48]` →
     `[..., 48, 56, 59]` and `[..., 96]` → `[..., 96, 104, 107]`.

   Both log the widening. This is the configuration of candidate-36's qualified
   copy arms, now reached from the plain recipe. Both steps are idempotent:
   `VllmConfig.__post_init__` may run again (e.g. `replace()`) without widening
   twice.
3. **One copy pass, beside DSpark.**
   - `copy_alloc` now updates the copy state from the step's verification and
     records the choice (source, offered drafts, next drafts, acceptance estimate)
     beside the allocation. (On candidate-37 it also wrote the slot's valid draft
     count; #860 removes those limits, and this patch writes none.)
   - `copy_apply` writes the proposals from that choice once DSpark has drafted.
   - `copy_select` is removed: it repeated the chunk reduction and the state
     update after DSpark (candidate-37 computed the state twice, discarding
     `alloc` the second time).
   - `prepare_alloc` runs `copy_match` and `copy_alloc` on the output copy stream
     after postprocess, beside DSpark's draft forward. The allocation's copy to
     the host follows on the same stream; `propose` makes the main stream wait.
     The allocation is the only result the host reads, through #860's alloc ring
     as before.
     Nothing DSpark runs reads or writes what they touch (history, lengths, copy
     state). Every input stays alive until `propose`'s wait, so no allocation can
     be reused under them.
   - A `copy_match` block whose share of the context is empty writes −1 without
     loading the key.

## Review of the stalled diff (`/tmp/vllm-842-work` on dgx4)

That tree (`1285a791f`, tree `378aa2db`) was candidate-37 plus one line in
`tests/models/test_deepseek_v4_1_l2_schedule.py`. Its uncommitted 838-line diff
was the starting point. I re-derived it on a freshly prepared candidate-37 tree:

| Item | Verdict |
|---|---|
| Default `copy_drafts` in `SpeculativeConfig`, `_copy_drafts_defaulted` flag | **Reworked.** It kept the option (a user value still went through) and needed a flag to tell defaulted from explicit. The field is now derived and not an init argument. |
| Clear for sync scheduling / V1 / PP / boundary checkpoints | **Reduced** to async + V2. PP is refused by adaptive verification; boundary checkpoints never hold for DSpark. |
| Budget widening in `_set_max_num_scheduled_tokens` | **Kept, tightened**: only with aligned chunks (where the scheduler requires threshold + envelope); widens the encoder budgets too; logs. |
| Capture-ladder widening | **Kept, re-ruled**: vLLM's default stride of 8, not a copy of the recipe's ladder rule. Also covers a max size without explicit sizes. Same result at TP3/TP4. |
| `copy_alloc` takes the state update, `copy_apply` added | **Kept.** The diff also kept `copy_select` behind `propose(matched=False)`: two implementations of the state update. Removed; `propose` now always follows `prepare_alloc`. |
| Side-stream `prepare_alloc` | **Kept**, with a guard: a `prepare_alloc` not followed by `propose` still orders the next one. |
| `wait_main_stream` flag in `async_utils.py` | **Dropped**: `attach_async_output` runs right after `prepare_alloc`, before DSpark is enqueued, so waiting on the main stream there costs nothing. |
| `WIDE_DECODE` `(5120, 3072/2048/1024)` → `{96: 96, 128: 64}` | **Dropped, a no-op.** Those shapes are WO-B (TP3/TP4/TP8). `wo.py` already pins WO-B's decode rows at 128 (`self.tilelang_decode_rows = 128`). `decode_tile_rows` gives 97 to 128 rows 64-row tiles with either table. TP4's 104/107-row steps already ran on the decode path; 0672 covered the shared experts' down projection. |
| "2-token register pre-filter" in `copy_match` | **Not in the diff; not added.** The run loop's first iteration already is the second-token compare, and only after a first-token hit. The kernel's cost is one coalesced load per window end. |
| `copy_match` empty-chunk exit | **Kept** (the barrier is block-uniform). Up to 44% faster at short contexts (below). |
| Tests | **Rewritten** for the above, plus the side-stream test and the not-an-option test. |

## Crash analysis (reading candidate-37; nothing here is reproduced)

**Where the NaN must be.** Rows are batch-invariant (rank-order reductions, aligned
chunks: the deployment's temperature-0 guarantee). A verified row whose drafts
were all accepted therefore computes what plain candidate-37 computes at that
position, and plain candidate-37 does not crash. So the non-finite gate logits
are in one of two places:
- a row past a rejected draft, an input plain candidate-37 never evaluates;
- state corrupted across steps. That would also break bit-identity, which held on
  the operator's copy-dense prompts.

`route_partials` asserts on every routed row, so a doomed speculative row is fatal
too. Each of its blocks routes two rows (`warps=2`), so #849's block [4] is rows
8-9. At c1 those exist only in a copy step: the 8th and 9th drafts, past DSpark's
6 rows.

**Ruled out:**

| Lead | Why it is not the cause |
|---|---|
| `-1` placeholders past `offered` | Their confidence is 0, so dead rows (`SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2`) make them padding, which the router skips. Even if routed, the TileLang embedding writes zeros for negative ids, Engram hashes them as the pad id, and every quantizer floors amax at 1e-4. |
| Layout lag under a binding budget (`_widen_copy_placeholders`) | The worker verifies `min(scheduled, slot limit)` drafts, inside the scheduler's allocation of `num_new_tokens` + 5 lookahead slots from its optimistic count. |
| SWA blocks freed under in-flight rows | Freeing uses `total_computed - num_in_flight_tokens`, which counts widened copy rows. |
| DSpark context rows | Sized by `max_decode_rows`; #853's poisoned-buffer soak covered the context ladder. |
| Sink rejection (one more spec row) | Off in the recipe. |
| Image placeholder ids copied from a prompt | The VL model's image mask is the runner's, by position, so a decode row with an image id gets that id's text embedding. |

**#852's suspects, against the reading above:**

| Suspect | Status |
|---|---|
| The copy verification row/placeholder path | Placeholders past `offered` are ruled out as a NaN source: they are dead rows, hence routing padding, and a negative id embeds as zeros and hashes as pad. The rest of the row path is the open question: copied tokens at rows 8-9 of a copy step. |
| The 0612-style async event sync racing the row layout under load | The host's `slot_draft_limits` mirror is read without waiting and can be a step stale. Staleness only changes how many scheduled rows are admitted: always inside the scheduler's layout and KV allocation, and any extra rows are zero-confidence dead rows. Not a NaN source on its own. |
| Draft ids or positions outliving their request or row (garbage embeddings, uninitialized KV) | Copy reads only its own slot's history below `total_len` (tested: stale tokens past a new request's length never propose). Positions come from the worker's own state. SWA blocks are freed on the processed-token basis. Still open: 0676's stale WO-A scratch for zero-length rows, uninitialized or stale input feeding a GEMM. |

**Open:**

- **0676** (in candidate-37, not candidate-36). For a row with zero SWA and
  indexed lengths, the extend and `block_H ≥ 32` decode combine write
  `Output = 0` but skip `wo_input`, so WO-A reads stale MXFP8 values and scales
  for that row. It is harmless only if every such row is also routing padding.
  Copy steps move c1 off the branch-free split route (≤ ~12 rows) onto the
  skipping decode route.
- Which copied inputs or row layouts produce a non-finite activation at all. That
  needs the model, so the operator's repro or a cluster window.

## Crash repro: the census and two arms

**Census** ([`census/0001-diag-dsv41-record-non-finite-MoE-gate-rows-instead-.patch`](census/0001-diag-dsv41-record-non-finite-MoE-gate-rows-instead-.patch)).
vLLM `cb46d6c82` on candidate-37's `5b7cfa37f` (tree `00d1e4ea`), its own commit,
never part of #842: the #842 patch keeps the assert.

- `TileLangGateRouter` checks the summed gate partials of every routed decode row
  before `route_partials`. Rows the kernel's mask skips are not looked at.
- A non-finite row is recorded in a per-router buffer and its partials are
  zeroed. The assert never fires, the step finishes, and the row routes to
  weight-0 experts with its hidden state still non-finite. These are device-only
  torch ops, so the decode graphs capture them.
- The V2 runner reads the census after sampling (target rows) and after the
  drafter (drafter rows). Each read is one host sync, so the arm runs slower than
  serving.
- It logs `NONFINITE` lines at ERROR:
  - the step's requests: rows, sampled, the verified copy proposal, total length;
  - per flagged row: layer, row, request, slot, draft index, position, input id,
    padding flag, origin (`anchor`, `copy`, `dspark`, `placeholder`) and
    `past_first_rejection`.
- Tests: `test_deepseek_v41_tilelang_router.py` 80/80, including the new test
  (NaN and inf rows recorded, padding not looked at, other rows routed as before,
  cleared on read, CUDA-graph replay). A stand-in smoke of the runner's logging
  printed all three origins.

**Arms.** Both start from candidate-37 as sp5-sp7 serve it, image
`sha256:e1ee7b1953b1131320746f87fe0cb11a673b53e9702bda63048ea3922768c15a`.
dgx4's own candidate-37 is a separate build (`c4c59de1…`).

| Arm | vLLM | Image (built on sp8; Dockerfile under `census/`) | Overlay arm, m=8 (`repro/`) |
|---|---|---|---|
| candidate-37 + census | `cb46d6c82` (tree `00d1e4ea`) | `spark-ds41f:candidate-37-census-842` = `sha256:a93654beeacb4a818fee1475d5ff6639653881c86131e05ab1d29996190877de` | `arm-c37-census-m8.json` |
| candidate-37 − 0676 + census | `cddb10983` = `5b7cfa37f~1` + census (tree `874cbfb2`) | `spark-ds41f:candidate-37-census-no0676-842` = `sha256:0224dca19023686dad5b754477fdb3447972c2563cde2a0071efa3a34e469f09` | `arm-c37-no0676-census-m8.json` (and `-m16` for a soak) |

- Each image is `FROM` that base. Its layers extend the base's (3 and 4 new
  layers), it holds no stale bytecode, and the router tests pass in it (80/80).
  The −0676 image also passes `test_deepseek_v41_tilelang.py` (54 passed, 1
  skipped, 1 xfailed).
- The overlay arms follow the precedent of `ed5ae38c` (the #848/#854 arms). They
  extend the profile sp5-sp7 serve (`config/candidate-37-tp3-switched-dev.local.json`,
  node-local, the same sha on all three) and mount the same files from
  `repro/overlay/` over the shared base image, so every rank keeps one digest.
  They add the operator's arm: `copy_drafts` m=8 (8 to 16 drafts) at
  candidate-37 `tp3-copy.json`'s envelope (6,155 tokens, capture to 59). The
  overlay files are byte-identical to the images' (sha256).
- `bin/spark --cluster-config <arm> doctor` (sp5's node map): configuration OK
  for the dev profile and all three arms.

The two images are the cross-check that the overlays equal them; the window uses
the overlays, so nothing is distributed and every rank keeps `e1ee7b19`.

**Lab specs.** Dry run with `--production-config
config/candidate-37-tp3-switched-dev.local.json`: no problems.

[`lab-repro.json`](lab-repro.json), 16 steps:
- `sync`;
- each m=8 arm on one boot, in #852's order:
  1. `copy_probe.py --prompts 3`;
  2. the `single-stream` suite (`repro/suite_single_stream.py`), where #852 dies in
     warm-up;
  3. `repro/census_logs.py`: every rank's NONFINITE and assert lines, collected
     before the next boot replaces the container;
  4. 8-stream `distinct_streams.py` (3 samples, 1,024 tokens);
  5. the census again;
- the table (`tables_arms.py`).

[`lab-soak16.json`](lab-soak16.json), 8 steps: queued only if the −0676 m=8 arm
is clean. The −0676 arm at m=16: the 12-prompt probe, the suite, 8- and 16-stream
loads, the census.

**The window** (owner: "Open a lab window", then "Add them" for the soaks), one
queue on sp5-sp7, restored once at close:
1. this repro;
2. #848 `experiments/2026-11-02-loop-handling/lab-soak.json`;
3. #854 `experiments/2026-11-02-dspark-markov-greedy-fusion/soak-tp3-switched.json`;
4. #853 `experiments/2026-11-02-dspark-context-capture-ladder/soak-spec.json`.

#853 and #854 both overlay `dspark.py`, so they stay separate boots. Before
`window open`, the node checkouts (sp5 `15cba190`, sp6/sp7 `7e33dcc6`) are synced
to the published main with `bin/spark --cluster-config
config/candidate-37-tp3-switched-dev.local.json cluster sync`.

## Commands

The tree (dgx4; any clone that has `19f2c20ed`):

```bash
# candidate-37's series, #860's two patches, then this patch, in a fresh repository
git init fresh && cd fresh
git fetch --depth 1 --no-tags /path/to/vllm 19f2c20ed4d64ae22bd009ceacadd85e2a81e354
git switch -c spark/build FETCH_HEAD
# each patch of ../2026-10-26-spark-ds41f-candidate/candidate-37/vllm/series in order
git am --committer-date-is-author-date <series patches...>            # tree af0fb152
# #860 (branch copy-drafter-shared-budget), vLLM 72abb31ff and 88982f34e
V=../2026-11-03-copy-drafter-shared-budget/vllm
git am --committer-date-is-author-date \
  $V/0001-fix-copy-drafts-budget-verification-from-what-every-.patch   # tree 58882ab9
git am --committer-date-is-author-date \
  $V/0002-cleanup-copy-drafts-remove-the-per-slot-draft-limits.patch  # tree f473423c
git am --committer-date-is-author-date 0001-perf-dspark-copy-drafts-as-the-one-behaviour-of-gree.patch
git rev-parse HEAD^{tree}                                              # 949f829d
git diff --check HEAD~1 HEAD
```

On dgx4 these trees are in `/tmp/842-final/vllm`:
- `copy-shared-budget`: #860's two commits, fetched from sp8's
  `/tmp/vllm-shared-budget` by bundle;
- `dspark-copy-drafts-default-on-860`: this patch, `9b8f05a35`.

`/tmp/842-final/fresh` checks the patch files, as above.

[`run_image.sh`](run_image.sh) runs a command in `spark-ds41f:candidate-37`. It
mounts the files that differ from `5b7cfa37f` (`git diff --name-only 5b7cfa37f` of
the tree, so #860's files too) read-only over `/opt/spark-ds41f/candidate/vllm`.
The tblib stand-in (`scripts/release_tests/tblib_stub.py`) is mounted on
`PYTHONPATH`:

```bash
TREE=/tmp/842-final/vllm ./run_image.sh "python3 -m pytest -q -rfEs -p no:cacheprovider -p tblib_stub \
  tests/v1/core/test_copy_drafts_scheduler.py tests/v1/spec_decode/test_copy_drafts.py \
  tests/v1/spec_decode/test_adaptive_verification.py tests/v1/worker/test_prepare_kernels.py"
```

The fourth file is the one #860's count includes. The runs on #856 and on
candidate-37 below ran the first three only.

The microbenchmark ([`bench_copy_kernels.py`](bench_copy_kernels.py)) puts
candidate-37's own `tilelang_copy.py` from the image beside it as
`tilelang_copy_base.py`, and imports vLLM from the candidate tree:

```bash
docker run --rm --entrypoint cat spark-ds41f:candidate-37 \
  /opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py > bench/tilelang_copy_base.py
./run_image.sh "PYTHONPATH=/c/w/bench:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x \
  python3 /c/w/bench/bench_copy_kernels.py /c/w/out/full-1.json"          # and full-2, separate processes
./run_image.sh "PYTHONPATH=... python3 /c/w/bench/bench_copy_kernels.py /c/w/out/c1-1.json --c1"   # and c1-2
```

The release gate: `scripts/release_tests.py check <tree> 5b7cfa37f HEAD` and
`gate <tree> HEAD gate-manifest.json`, then [`gate.sh`](gate.sh) (the bundle's
runner, environment, io_uring and O_DIRECT dirs, with the patch's files over the
image tree). `MANIFEST=` selects a subset.

## Runs

All runs are kept below, unfavourable and failed ones included. The raw JSON is
in [`runs/`](runs/).

### Unit tests (dgx4, 2026-11-02 and 2026-11-03)

**On #860 (this patch's base, rebased 2026-11-03; `9b8f05a35`, tree `949f829d`):**
- Started from the first version (`bc71d8191`), cherry-picked onto
  `88982f34e`. The conflicts were in `copy_drafts.py`, `tilelang_copy.py` and
  `test_copy_drafts.py`, where #860 removed the per-slot limits that version
  still wrote. They resolve to this patch's step work without any limit:
  - `copy_alloc` takes no `slot_limits` and writes none;
  - `prepare_alloc` reads no host limits;
  - nothing reads `slot_draft_limits.gpu`.

  `git grep` finds no `slot_draft_limits`, `slot_limits`, `sync_slot_draft_limits`
  or limit staging anywhere in `vllm/` or `tests/`.
- Tests adapted:
  - The kernel helpers and the CUDA-graph test no longer pass or check a
    limits buffer.
  - The prepare_alloc test becomes `test_prepare_alloc_writes_alloc_before_propose`.
  - The stale-wide test drops its limit check, as #860's cleanup does.
  - #860's two shared-budget tests run through this patch's `propose(idx_mapping,
    drafts, confidences)`.
  - The side-stream test keeps everything but the limits: the proposals,
    allocation and copy state equal the drafter-stream run's, with the side
    stream held busy.
  - The #856 rebase's `test_side_stream_limits_reach_the_host_as_the_latest_proposals`
    is gone with the mirror it tested.
- The first run failed: **3 failed, 81 passed, 1 skipped**. The kernel tests'
  helpers still passed `copy_alloc` a limits buffer (`TypeError: copy_alloc()
  takes 15 positional arguments but 16 were given`), so they were fixed and the
  commit amended.
- `test_copy_drafts_scheduler.py`, `test_copy_drafts.py`,
  `test_adaptive_verification.py` and `test_prepare_kernels.py` in the
  candidate-37 image: **84 passed, 1 skipped** (the two-GPU test). #860 alone,
  the same four files on dgx4: **79 passed, 1 skipped**, as #860 reports from
  sp8.
- Mutation check: `propose` without its wait for the side stream fails
  `test_side_stream_prepare_alloc_matches_the_drafters_stream` only
  (`test_copy_drafts.py`: 1 failed, 24 passed).
- Outputs: [`runs/unit-on-860.txt`](runs/unit-on-860.txt) (this patch, the
  failed first run, #860 alone, the mutation). dgx4 was otherwise idle (no
  compute process, no other container).
- Fresh tree: candidate-37's `af0fb152` with #860's patch files gives `58882ab9`
  and `f473423c`, and this patch file then gives `949f829d`. `git diff --check`
  is clean.
- vLLM `ruff` 0.14.0: no finding in this patch's hunks. The two E501s in
  `test_copy_drafts.py` are candidate-37's line and #860's line.

**On #856 (superseded; rebased 2026-11-02; tree `458b91a2`):**
- `test_copy_drafts.py`, `test_copy_drafts_scheduler.py` and
  `test_adaptive_verification.py` in the candidate-37 image: **77 passed, 1
  skipped** (the two-GPU test). #856 alone gives 71 passed, 1 skipped.
- The rebase kept #856's blocking `sync_slot_draft_limits`, without the reset
  bookkeeping it removed. It adds
  `test_side_stream_limits_reach_the_host_as_the_latest_proposals`: with the
  match and allocation on a side stream held busy before each step, the host
  limits after `attach_async_output` equal the latest proposal's device limits.
- Mutation check: putting candidate-37's non-blocking `query()` back fails that
  test and #856's `test_sync_slot_draft_limits_waits_for_the_queued_copy`.
- The kernels and the step's copy work are untouched by #856
  (`tilelang_copy.py` is the same), so the bit-identity and microbenchmark
  results below stand.
- The host's limit wait ends at the same point in the step as on
  candidate-37 + #856: the limit copy follows `copy_alloc` on the output copy
  stream, as it followed `copy_alloc` on the main stream there. Its serving
  cost is #856's to measure (`lab-cost.json` there).

**On candidate-37 (the first version, `bc71d8191`):**

- `test_copy_drafts_scheduler.py` + `test_copy_drafts.py`: **38 passed** (81 s).
  This run (17:04:41-17:06:10, dgx4 time) overlapped a sibling session's GPU
  benchmark on dgx4 (rework-739's `tp4-ladder-first-2`). It is functional, so the
  verdict stands; that session was told its run may be perturbed.
- Mutation check: `propose` without its wait for the side stream fails
  `test_side_stream_prepare_alloc_matches_the_drafters_stream` (1 failed), and
  passes with it.
- `scripts/release_tests.py check`: 127 selected test files (74 gate, 53
  excluded), **0 problems**. Gate manifest: 121 files.
- spark-ds41f, as CI runs it (`scripts/example_site.py`, then
  `unittest discover -s tests`): **615 tests OK** (5 skipped).
  `bin/spark --shape <shape> doctor`: configuration OK for all 9 shapes. A first
  pytest run without a site node map had 5 failures; `origin/main` has the same 5
  without one, and with the example site there are none.
- vLLM `ruff` 0.14.0 (the tree's pre-commit pin) on the changed hunks: clean. The
  remaining format and lint findings in those files are candidate-37's own lines,
  left untouched.

### Lab window on sp5-sp7 (tp3-switched), 2026-11-02

**Setup.** The owner authorized the window ("Open a lab window", then "Add them"
for the soaks); the user confirmed it in this session.
- 17:13Z: `cluster sync --apply` with `config/candidate-37-tp3-switched-dev.local.json`
  (plan read first). All three nodes clean and detached at main `2e0f01c2`.
- 17:14:48Z: `window open` with that production profile, window
  `spark-0013-1793639688`.
- Queue, runner detached on sp5 (`results/private/lab/queue-842.log`):
  1. `lab-repro.json`;
  2. `lab-clock2200.json` (main `7084d2bd`), inserted after the repro;
  3. #848, #854 and #853's soaks.
- The arms serve on `e1ee7b19` with the census overlays mounted (checked in the
  containers). The serve arguments carry `copy_drafts` m=8, 6,155 tokens,
  capture 59 and threshold 6,096.

**Arm 1, `c37-m8` (candidate-37 + census, m=8), boot 17:15:36Z:** #852 did not
reproduce.
- `copy_probe --prompts 3` matches the operator's m=8 numbers: copy 16.37, edit
  15.75, tool 6.91, prose 2.65 tokens per step.
- The `single-stream` suite (sha `7c59b27e8503`) ran its 60 s warm-up and all 10
  windows: 60.08 tok/s (CI ±5.4%), 2.27 tokens per step, 0 failed.
- 8 distinct streams × 1,024 tokens: 153.69 tok/s (CI ±0.1%), exit 0.
- **Census:**
  - After the suite, checked by hand on sp5, sp6 and sp7 (`docker logs |
    grep`): 0 NONFINITE lines, 0 assert or traceback lines. No gate row went
    non-finite in the probe or the suite.
  - The extra's own census records for this arm are **invalid**. `census_logs.py`
    as first published (`2e0f01c2`) resolved the site default's node map, so it
    read `docker logs` of sp1-sp4's serving containers. Those were read-only log
    reads, with no other effect.
  - The 8-stream load has **no valid census**: arm 2's boot replaced the
    container before it was read.
  - Fixed in `3ae1b273` (main `78acd0e5`): the census takes the window's
    production profile from the hold. Read-only followers on sp5-sp7 now keep
    every serving container's log by id (`/tmp/842-census/`).

**Arm 2, `no0676-m8` (candidate-37 with 0676 reverted + census, m=8), boot
17:28:35Z:** clean too.
- The copy probe matches arm 1 exactly (copy 16.367, edit 15.745, tool 6.905,
  prose 2.651).
- Suite: 59.93 tok/s (CI ±5.3%), 2.27 tokens per step, 0 failed.
- 8 streams: 153.44 tok/s (CI ±0.05%).
- Census: complete container logs from the followers (`6f69c3adcda6`,
  `067b4750b93a`, `8a18587eec25`): 0 NONFINITE and 0 assert or traceback lines
  on sp5, sp6 and sp7, through probe, suite and 8 streams. The extra's own
  records are again invalid (sp1-sp4): this job ran the first collector.
- The table step raised `KeyError: 'exit'`: the suite's output file carries
  servebench's own lines before the extra's record. The data is unaffected;
  `tables_arms.py` is fixed on this branch.

**So #852 does not reproduce here.** On sp5-sp7 (tp3-switched, unlocked clocks,
image `e1ee7b19`, 4K-page `7.0.0-1024-nvidia` kernels), the operator's exact m=8
sequence passes on candidate-37 with and without 0676. No gate logit went
non-finite. Untested differences from their ring:
- the fabric (ring there, switched here);
- a 2,200 MHz clock lock;
- their own build of tree `af0fb152`;
- kernel and page size;
- their real agent traffic.

**The 2,200 MHz pass** (`lab-clock2200.json`, job 17:41:04Z) failed before
booting. It locked the SM clocks at 2,200 MHz on all three nodes, then raised
`FileNotFoundError` on `~/spark-lab/gpu-clocks-locked.json.tmp`: `~/spark-lab/`
did not exist on sp5, and the lab does not create it (a `lab.py` follow-up). The
run ended without its reset, and the queue moved on to #848. I reset the clocks
by hand at about 17:42Z (`sudo -n nvidia-smi -rgc` on sp5-sp7), while #848's
first arm was still booting (17:41:43Z), so #848 measures unlocked.
`~/spark-lab/` was created on sp5, and the identical spec (sha `f7da0e41`) was
requeued after #848.

**The 2,200 MHz pass, rerun** (job 18:35:32Z, after #848): **clean.**
- The SM clocks were locked at 2,200 MHz on sp5-sp7 at 18:35:59Z and reset by the
  job at 18:50:32Z.
- Arm 1 (`c37-m8-2200`: candidate-37 + census, m=8): copy probe identical to the
  unlocked runs (copy 16.367, edit 15.745, tool 6.905, prose 2.651).
- `single-stream` suite: 59.31 tok/s, 2.26 tokens per step, 0 failed.
- 16 distinct streams × 1,024 tokens: 160.15 tok/s (CI ±0.9%).
- Census, from the window's nodes with the fixed collector: 0 NONFINITE and 0
  assert lines on sp5, sp6 and sp7, after the suite and after the 16 streams.

**Across the window.** The followers kept every serving container's log, seven
per node, from the repro, the 2,200 MHz pass and the three soaks. They hold 0
NONFINITE lines and 0 `nan or inf`, `cudaErrorAssert`, device-side assert or
`EngineDeadError` lines. Neither soak below logged a non-finite or an assert.

**#852 conclusion from this window.** On tp3-switched (sp5-sp7, image
`e1ee7b19`), candidate-37 with copy drafts at m=8 and the #842 envelope passes
#852's exact sequence. It passes with and without 0676, at unlocked clocks and
at 2,200 MHz, under 1, 8 and 16 streams. No routed gate row went non-finite.
What remains differs from the operator's ring: the fabric, their image build,
kernel and page size, and their real agent traffic. The operator's offer to
run the census on their ring is the next step; the owner decides that.

**A likely reason it stayed clean here** (from #855, the operator's claimed root
cause, opened during the window and not run here):
- #855's mechanism: `sync_slot_draft_limits` publishes the host limits only once
  `_limits_event.query()` is true. A stale, wider limit can then admit `-1`
  placeholder rows into verification, and they NaN the gate. Their control was
  `-1`→`0`; their fix blocks on the event, reversing 0612's non-blocking design.
- Every arm in this window ran with `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2` (the
  recipe's, seen in the containers). Such placeholder rows have confidence 0,
  so they are dead rows and marked routing padding, and `route_partials` skips
  them.
- A run without dead rows (tau unset or 0), or with a batch sharder (dead rows
  are skipped then), routes them live.
- A fix that keeps the non-blocking design: clamp each request's verified drafts
  to the device-side `slot_draft_limits.gpu`, which is always current on the
  GPU, inside `combine_sampled_and_draft_tokens`. Rows past it become padding
  with a valid input id, whatever the host's stale mirror admitted. Open; the
  owner's call.

**The rest of the queue** (other experiments' soaks, recorded here only as this
window's log; their results belong to their own PRs):

| Job | Time (Z) | Result |
|---|---|---|
| #848 `lab-soak.json` | 17:41-18:35 | Done; every measured step "clocks unlocked". Probes failed: loop 2/15, c37 3/15, loop-repeat 2/15. |
| #854 `soak-tp3-switched.json` | 18:50-19:29 | release-tp3 halted in prefill on the bench's thermal guard ("sp7: GPU at 82 C"). |
| #853 `soak-spec.json` | 19:29-20:16 | Halted in prefill repeat 2 ("sp6: GPU at 80 C"). Engaged: "Capturing DSpark context CUDA graphs (FULL): 23/23". |
| #848 `lab-probes-v2.json` (user-approved addition) | 20:36-20:53 | 15 probes, 0 failed. |

sp5-sp7 have no fan curve installed (the suites warn that `dgx-fan-control.service`
is not installed), which likely explains both thermal halts.

**Close.** At 20:56:24Z: `window close` booted the production profile
`config/candidate-37-tp3-switched-dev.local.json` (shape unchanged), `doctor --live`
OK, hold removed.
- Every node serves under that profile's label, `/health` returns 200, the clocks
  are unlocked, and no lab process remains.
- The nodes stay detached at main `700967cc`, the last sync.
- The followers' per-container counts, 27 containers with 0 NONFINITE and 0 fault
  lines, are in [`runs/window-census-followers.jsonl`](runs/window-census-followers.jsonl).
- The followers were then stopped and `/tmp/842-census` removed on sp5-sp7.
- `~/spark-lab/` stays on sp5 (the lab's directory).

Follow-up, for after the window (a small fix with a test, done separately; not
patched during the window, whose runner has the code loaded):
- `scripts/lab.py`'s `gpu_clocks` lock writes its marker
  (`GPU_CLOCKS_MARKER = LAB_HOME / "gpu-clocks-locked.json"`) without creating
  `LAB_HOME`;
- a lock job that then raises leaves the clocks locked. The run ends without its
  reset step and the queue continues, so the next job can run locked.

### Release gate

Run on the first version (`bc71d8191` on candidate-37), not rerun on #860.

`scripts/release_tests.py gate` at the patched tree: 121 files. The run used
[`gate.sh`](gate.sh) on dgx4, 17:43-18:44 (dgx4 time): the bundle's runner and
environment, with the patch's files over the candidate-37 image. Result:
**7,316 passed, 172 failed, 166 skipped, 0 errors**
([`runs/gate-full-results.jsonl`](runs/gate-full-results.jsonl)). No failure
comes from the patch.

| File | Failed | Cause | Control |
|---|---:|---|---|
| `tests/v1/worker/test_gpu_worker.py` | 62 | stand-in `SimpleNamespace` lacks `_roce_failure` | unpatched candidate-37: the same 62 |
| `tests/v1/worker/test_gpu_model_runner_v2.py` | 20 | stand-in runner lacks `num_draft_steps` | unpatched: 21 failed with the next file |
| `tests/v1/attention/test_b12x_v41_ced_metadata.py` | 1 | `DFlashSpeculator` lacks `_query_start_loc_cpu` | unpatched: the same |
| `tests/kernels/quantization/test_deepseek_v41_tilelang_wo.py` | 11 | `'Module' object has no attribute 'tilelang_key'` | unpatched: the same 11 |
| `tests/model_executor/model_loader/test_direct_loader_cuda.py` | 36 | io_uring: Docker's 8 MiB `memlock` | patched with `--ulimit memlock=-1`: pass |
| `tests/kernels/test_deepseek_v41_engram_rows.py` | 35 | the same | patched with `memlock=-1`: pass |
| `tests/models/test_deepseek_v4_1_engram.py` | 7 | 6 the same; 1 `TypeError` at line 568 | patched with `memlock=-1`: 6 pass; the `TypeError` test fails on unpatched candidate-37 too |

`gate.sh` passes `--security-opt seccomp=unconfined` but not `--ulimit
memlock=-1`, which the lab's io_uring bundles need. The three io_uring files
were rerun with it: 125 passed, 1 failed (the pre-existing `TypeError`).

### Microbenchmark (`bench_copy_kernels.py`)

Run on the first version (`bc71d8191` on candidate-37), not rerun on #860. In
both arms #860 drops the same thing: `copy_alloc`'s one store per request of the
slot's limit, and the limits' copy to the host. The other kernels and the
patch's copy work are unchanged by the rebase. The bench script still passes
the limits buffer, so it runs only against candidate-37's kernels.

Every case runs base (candidate-37's kernels) and patch alternated (ABBA) in one
process; each measurement was run twice, in separate processes, on an otherwise
idle GPU. Container events confirm no other container ran: dgx4
17:15:53-17:18:21, sp8 16:21:17-16:24:17 (local times).

**Bit-identity.** Two checks:
- every case's first step, from zero state: proposals, confidences, planned
  confidences, allocation and copy state;
- a 400-step walk of 8 requests (copied documents with edits, a repeated phrase,
  random text), with candidate-37's step and the patch's on separate copy state,
  the verified, accepted, sampled and cut counts varied every step: 1,495 copy
  proposals.

Result: **identical in every case and every step** (0 mismatched steps), in all
five runs.

**c1** (`--c1`, sp8, 1,000 alternated steps per case). The main stream's time from
postprocess to the proposals, with a ~4.8 ms DSpark stand-in: 40 bandwidth-bound
BF16 GEMMs, every SM busy, the worst case for kernels beside it. Δ = patch −
candidate-37, mean ± 95% CI:

| context | text | candidate-37 (µs) | patch (µs) | Δ run 1 (µs) | Δ run 2 (µs) |
|---|---|---:|---:|---:|---:|
| 2K | prose | 4799.6 | 4801.2 | -0.6 ± 5.6 | -1.1 ± 5.6 |
| 2K | zipf | 4807.7 | 4808.1 | -2.1 ± 5.7 | +1.4 ± 5.0 |
| 32K | prose | 4811.6 | 4812.9 | -0.5 ± 4.3 | -1.4 ± 5.4 |
| 32K | zipf | 4828.1 | 4829.2 | -0.8 ± 5.3 | +0.3 ± 4.9 |
| 128K | prose | 4832.9 | 4833.5 | -1.0 ± 5.4 | +0.5 ± 4.7 |
| 128K | zipf | 4833.9 | 4834.9 | +2.8 ± 5.4 | +1.9 ± 5.6 |
| 1M | prose | 4901.6 | 4867.5 | -34.5 ± 4.8 | -33.4 ± 5.5 |
| 1M | zipf | 4916.8 | 4879.0 | -35.6 ± 4.5 | -36.2 ± 6.3 |

At 2K to 128K the copy work is 4-9 µs, and moving it beside DSpark changes the
step by −2.1 to +2.8 µs (CI ±5 µs): **no harm at c1**. At 1M contexts it takes
`copy_match`'s ~35 µs off the critical path in both runs.

**Kernels** (dgx4, GPU µs per launch from CUDA-graph replays). Selected rows; all
36 cases are in `runs/bench-full-*.json`. Copy work per step: candidate-37 is match
+ alloc + select, the patch match + alloc + apply.

| requests | context | text | match | alloc | select → apply | step's copy work (run 1; run 2) |
|---:|---|---|---|---|---|---|
| 1 | 2K | prose | 1.64 → 1.60 | 1.23 → 1.27 | 1.24 → 0.92 | 4.11 → 3.79; 4.18 → 3.81 |
| 1 | 128K | zipf | 5.78 → 5.90 | 1.35 → 1.39 | 1.35 → 1.03 | 8.48 → 8.32; 8.73 → 8.24 |
| 1 | 1M | zipf | 36.77 → 36.57 | 1.37 → 1.40 | 1.37 → 1.03 | 39.51 → 39.00; 39.72 → 39.33 |
| 8 | 2K | prose | 2.62 → 2.07 | 1.35 → 1.39 | 1.48 → 1.07 | 5.45 → 4.53; 5.38 → 4.43 |
| 8 | 128K | zipf | 7.64 → 7.62 | 1.39 → 1.40 | 1.39 → 1.06 | 10.42 → 10.08; 10.43 → 10.14 |
| 8 | 1M | zipf | 200.28 → 196.27 | 1.35 → 1.39 | 1.39 → 1.06 | 203.02 → 198.72; 199.59 → 199.33 |
| 16 | 2K | prose | 4.10 → 2.29 | 1.42 → 1.50 | 1.40 → 1.07 | 6.92 → 4.86; 6.87 → 4.87 |
| 16 | 128K | zipf | 19.82 → 19.55 | 1.44 → 1.44 | 1.39 → 1.06 | 22.65 → 22.05; 22.95 → 22.03 |
| 16 | 1M | zipf | 389.55 → 385.33 | 1.35 → 1.39 | 1.45 → 1.10 | 392.35 → 387.82; 392.70 → 394.66 |
| 16 | 1M | loop | 4149.24 → 4037.01 | 1.50 → 1.64 | 1.59 → 1.19 | 4152.33 → 4039.84; 4157.39 → 4040.71 |

How the kernels moved:
- `copy_alloc` changes by −0.04 to +0.16 µs: it now writes the state and the
  choice.
- `copy_apply` costs 0.32-0.51 µs less than `copy_select`.
- `copy_match`'s empty-chunk exit saves up to 1.8 µs (−44%) at 16 × 2K. The match
  is 0.04-2.25 µs slower in 4 of 72 cells.

The step's copy work is lower in 71 of 72 cells. The exception, 16 × 1M zipf in
run 2, is +2.0 µs (+0.5%) against −4.5 µs in run 1.

**Step, 1 to 16 requests** (dgx4, 200 alternated steps per case, stand-in as
above; Δ mean ± 95% CI):

| requests | context | text | stand-in (µs) | candidate-37 (µs) | patch (µs) | Δ run 1 (µs) | Δ run 2 (µs) |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 2K | prose | 4991 | 5018.6 | 5019.8 | -9.4 ± 16.4 | -19.4 ± 15.0 |
| 1 | 2K | zipf | 5042 | 5019.3 | 5018.3 | +8.5 ± 15.8 | +2.0 ± 4.6 |
| 1 | 32K | prose | 4990 | 5019.8 | 5015.0 | -13.3 ± 14.1 | -11.8 ± 15.1 |
| 1 | 32K | zipf | 4992 | 5015.5 | 5015.7 | +3.6 ± 15.4 | +0.5 ± 11.2 |
| 1 | 128K | prose | 4997 | 5016.7 | 5023.2 | +18.1 ± 15.3 | -98.8 ± 133.5 |
| 1 | 128K | zipf | 4996 | 5020.0 | 5023.0 | +9.9 ± 10.6 | +1.9 ± 9.6 |
| 1 | 128K | loop | 4995 | 5179.8 | 5051.9 | -145.2 ± 14.8 | -123.4 ± 15.1 |
| 1 | 1M | prose | 5090 | 5078.9 | 5046.1 | +44.3 ± 127.1 | -35.7 ± 14.4 |
| 1 | 1M | zipf | 5004 | 5082.8 | 5047.6 | -36.7 ± 11.6 | -31.9 ± 16.9 |
| 1 | 1M | loop | 4998 | 6268.4 | 5252.8 | -1023.1 ± 14.5 | -1032.3 ± 13.7 |
| 8 | 2K | prose | 3534 | 3550.4 | 3551.3 | +2.0 ± 12.8 | +4.7 ± 10.7 |
| 8 | 2K | zipf | 3524 | 3551.6 | 3554.7 | +2.5 ± 12.1 | +3.5 ± 10.8 |
| 8 | 32K | prose | 3526 | 3549.7 | 3550.7 | -3.0 ± 12.3 | -6.4 ± 12.9 |
| 8 | 32K | zipf | 3527 | 3553.7 | 3553.6 | +1.5 ± 12.0 | +1.4 ± 15.1 |
| 8 | 128K | prose | 3531 | 3563.5 | 3559.7 | +3.8 ± 9.8 | +4.0 ± 7.5 |
| 8 | 128K | zipf | 3522 | 3561.2 | 3555.6 | -5.9 ± 15.3 | +3.2 ± 14.7 |
| 8 | 128K | loop | 3526 | 3871.1 | 3797.6 | -71.0 ± 9.1 | -79.1 ± 5.6 |
| 8 | 1M | prose | 3524 | 3721.7 | 3712.4 | -6.6 ± 8.2 | -3.5 ± 13.9 |
| 8 | 1M | zipf | 3523 | 3737.4 | 3703.5 | -22.9 ± 10.5 | -20.6 ± 12.1 |
| 8 | 1M | loop | 3625 | 6013.2 | 5531.7 | -476.6 ± 37.5 | -497.1 ± 34.5 |
| 16 | 2K | prose | 3591 | 3618.2 | 3618.3 | +5.2 ± 7.7 | -8.1 ± 9.8 |
| 16 | 2K | zipf | 3590 | 3617.1 | 3619.2 | +5.1 ± 5.7 | +2.5 ± 7.7 |
| 16 | 32K | prose | 3593 | 3622.2 | 3622.1 | +4.3 ± 6.6 | -0.2 ± 8.6 |
| 16 | 32K | zipf | 3593 | 3625.3 | 3624.5 | +12.2 ± 65.9 | -8.8 ± 10.0 |
| 16 | 128K | prose | 3592 | 3650.1 | 3641.3 | -8.0 ± 6.0 | -4.8 ± 7.2 |
| 16 | 128K | zipf | 3592 | 3653.9 | 3642.8 | -10.7 ± 8.6 | -4.6 ± 9.0 |
| 16 | 128K | loop | 3625 | 4145.3 | 4085.1 | -64.7 ± 7.7 | -43.5 ± 7.4 |
| 16 | 1M | prose | 3620 | 3985.0 | 3973.3 | -16.9 ± 8.6 | -16.9 ± 7.9 |
| 16 | 1M | zipf | 3625 | 4000.9 | 3970.4 | -32.2 ± 9.2 | -39.4 ± 8.3 |
| 16 | 1M | loop | 3703 | 7786.3 | 7385.0 | -412.5 ± 25.0 | -393.8 ± 24.0 |

**Natural text up to 128K** (24 cells): one positive Δ excludes zero, 1 × 128K
prose in run 1 (+18.1 ± 15.3 µs). Its run 2 is −98.8 ± 133.5 (outliers), and the
c1 runs above (1,000 steps) put the same cell at −1.0 ± 5.4 and +0.5 ± 4.7, so
I read it as noise. Three favourable cells exclude zero (−8.0, −10.7, −19.4 µs);
the rest are inside their CI.

**At 1M**, natural text saves 3.5-39.4 µs a step in 11 of 12 cells (8 × 1M prose
inside its CI). The twelfth, 1 × 1M prose run 1, is +44.3 ± 127.1 (outliers);
its run 2 is −35.7 ± 14.4.

**A repeated phrase** (`loop`, the match's worst case) saves 44-1,032 µs. At 8
and 16 requests the side stream's copy work competes with the GEMMs for the SMs,
so it saves less than its full length there.

**Failed and discarded attempts.** Each was fixed in the script; none produced
numbers:
1. vLLM was imported from the base image's site-packages, not the candidate tree
   (`ImportError: cannot import name 'CopyDraftsConfig'`). Fixed with `PYTHONPATH`.
2. The stand-in was timed by capturing a graph replay inside a graph
   (`cudaErrorStreamCaptureUnsupported`). It overlapped a sibling's functional
   pytest on dgx4.
3. A host tensor indexed a device tensor in the equivalence walk.

`runs/bench-quick.json` is the `--quick` validation run (60 steps per case) on
an idle dgx4. Its conclusions agree with the full runs.

### Memory

- `CopyDrafter`: `chosen` (`max_reqs × 3` int32) and `estimates` (`max_reqs` float32)
  added, `_select_alloc` (`max_reqs` int32) removed: under 200 bytes at 16
  requests.
- The engine's sizing is candidate-36's qualified copy arms': 11 more tokens of
  step budget and two more CUDA graphs per shape (56 and 59, or 104 and 107).
- Startup memory was not measured here (it needs the cluster). Expect it to equal
  candidate-37's `tp3-copy.json`, including the startup tokens line.

## Conclusion

**Not ready.** Copy drafts, as candidate-37 computes them and as this patch
computes them (bit-identically), crash an operator's TP3 engine:
- under real traffic at m=16;
- reproducibly at m=8 on single-stream.

#860's fix is now this patch's base (unit-tested here, 84 passed). Whether
copy drafts on that base are clear of the crash is candidate-38's and
candidate-39's to show at the cluster. The patch stays on this PR as the
finished mechanism:
- no option, and the old arms' `copy_drafts` JSON is refused;
- the recipe's plain shapes reach the copy envelope (6,155 / 16,395 tokens,
  capture to 59 / 107) with unchanged chunk boundaries;
- the step's copy work is computed once and beside DSpark.

On GB10 its step work is never slower than candidate-37's:
- level at c1 from 2K to 128K (|Δ| ≤ 2.8 µs, CI ±5 µs, two runs);
- 33-36 µs a step faster at c1 at 1M;
- up to 1 ms faster on degenerate repeats.

Like the operator's m=16 suite pass, these numbers cannot clear the crash. The
stalled diff's `WIDE_DECODE` change was a no-op, and its stream flag was
unnecessary; both are dropped.

**What it takes to be ready** (owner's call; the serving clusters and any GitHub
post to the operator are yours):
1. Localize the NaN and fix it.
   - The sp5-sp7 window could not reproduce it: #852's exact sequence is clean on
     tp3-switched with and without 0676, at unlocked clocks and at 2,200 MHz,
     under 1, 8 and 16 streams, with dead rows on.
   - The operator's claimed cause, #855, was first taken in as
     [#856](https://github.com/christopherowen/spark-ds41f/pull/856) (a blocking
     limit sync), then #859 (a device clamp). The owner dropped both for their
     cost.
   - [#860](https://github.com/christopherowen/spark-ds41f/pull/860) is the fix
     now, and this patch's base: the budget reads no host limits, and the limits
     are removed. Its window reproduced #852 on candidate-37 (m=8, dead rows off)
     and ran the fix clean; its README has the runs and the open lead.
2. #860 ships in candidate-38 (with #848, #853 and #854), without #842.
3. Then this patch as candidate-39 on candidate-38, with the plain `tp3.json` /
   `tp4.json`, and a concurrent soak where the token budget binds (mixed prompt
   lengths, copy-dense and tool-like prompts, several streams).

Expected startup lines: "Copy drafts widen max_num_batched_tokens from 6144 to
6155 ..." and "... capture range from 48 to 59 rows" (TP3); 16384 → 16395 and
96 → 107 (TP4). Determinism digests should equal candidate-37's copy arm's.

`min_match` and a copy gate (the replay's oracle reaches +9.0% at m=16 and
+14.3% at m=8) are later, separate variables, after the crash is fixed.
