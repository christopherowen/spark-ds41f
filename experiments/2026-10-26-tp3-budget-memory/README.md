# What limits TP3's long-prefill budget, and what would raise it

**Status (owner, 2026-10-25 evening CET): the cluster measurement is dropped.**
- The study was asked for on the belief that the triangle's 4,096 → 8,192 step bought only 1-3%
  prefill. It bought +10-13% (experiments/2026-10-19-prefill-budget-tp3), and candidate 6 at
  8,192 is about +26-29% over spark-ds41f-1022 on the triangle.
- So the TP3 budget sweep below ([tp3-budget-memory.json](tp3-budget-memory.json)) is not queued,
  and this record is not merged for it. It stays as a draft record for next week.
- **Still worth pursuing, whatever the budget:**
  - the 64-row MoE tiles for prefill forwards (lever 1): +3.5-5% prefill on both shapes, outputs
    bit-identical on dgx4;
  - the memory cuts (levers 2-4): layer 20's candidates as block ids, the drafter's workspace lane
    without the target's indexer scratch, the two unread buffers.

## On next-candidate with the kernel audit (2026-10-26, about 01:30 CET)

next-candidate moved to 16c1b8cae: e20c47020 (the replay-start reference fix) and the kernel
audit's seven commits. The levers land second.
- **vLLM branch:** `tp3-levers-next2` (local, not pushed), on 16c1b8cae. Its commits, in order:
  1. 0504 9ad4a1fdb;
  2. 0505 a0dc662f5;
  3. 0506 410338b04;
  4. 0507 577091997.
- **Patches:** [vllm-next2-candidate/](vllm-next2-candidate). They replay onto 16c1b8cae to the
  branch's tree.
- **The one conflict, `test_deepseek_v41_tilelang.py`:**
  - both sides' appended tests are kept;
  - the audit's `test_indexer_reads_only_live_pages[candidates=True]` now builds block ids as
    the source writes them (the last block partly visible, one masked block, `-1` padding),
    since 0507's re-index plan takes block ids.
- **dgx4 runs:**

  | Bundle | Result | Record |
  | --- | --- | --- |
  | `next2-candidate` (16c1b8cae itself) | 207 passed, 1 xfailed | [runs/next2-candidate.json](runs/next2-candidate.json) |
  | `next2-levers` | 232 passed, 1 xfailed; prefill tiles −11.4% at TP3's 8,192 and −15.1% at TP4's 16,384, bits identical; re-index 239.1 vs 247.0 µs; lane check; initcheck control | [runs/next2-levers.json](runs/next2-levers.json) |
  | `next2-audit-attention` (the audit's `audit-next-attention` steps on this branch) | 175 + 5 + 5 passed, 1 xfailed; both of the audit's gated benchmarks pass | [runs/next2-audit-attention.json](runs/next2-audit-attention.json) |

  The audit's own generator rewrites its bundles in place after deleting them, so its steps
  are mirrored in [make_bundles.py](make_bundles.py) (`AUDIT_ATTENTION_STEPS`) and run with
  this experiment's temp directories and dgx4's compute-sanitizer.

## On next-candidate (2026-10-26, about 01:00 CET)

Owner direction, through the coordinator: build a new candidate from next-candidate, and push
each lever once if it can improve first.
- **vLLM branch:** `tp3-levers-next` (local, not pushed), on next-candidate 4209a1903. The four
  levers are individual commits, in order:
  1. 0504 e836c5e71: 64-row MoE tiles for prefill calls;
  2. 0505 9895b5ea7: the two unread buffers dropped;
  3. 0506 bbfa2b3ac: the drafter's workspace lane;
  4. 0507 e0e89bb13: layer 20's candidates as block ids.
- **Patches:** [vllm-next-candidate/](vllm-next-candidate) (`git format-patch`). They replay onto
  4209a1903 to the branch's tree.
- **Overlaps resolved:**
  - **0504 and the mHC cast (a7ba079c7):** the routed experts take the `MXFP8Rows` the cast
    produces, and the benchmark casts once for both plans.
  - **0507 and the B12X retirement (0493):** `CANDIDATE_WIDTH` goes with B12X's indexer, and the
    staging is sized by `CANDIDATE_BLOCKS`.
  - **0507 and 0503:** 0503 makes the indexer read no key past a row's visible positions; see
    the push below.

**The one push.** 0507 is the only lever that changed.
- **The problem:** past a row's count, the last block's ids point past the row's end. On
  next-candidate the score kernel must not read those keys.
- **Every position is a written one now:**
  - a masked block reads block 0, which is whole in any row of more than one block;
  - past the count, a candidate takes offset 0. That is the last kept block's first position,
    or position 0 past the kept blocks, just as a position list's `-1` reads position 0.
- **Only the tile holding the count needs the bound.** The kernel chooses once a tile, so the
  tiles wholly below the count compute positions without it.
- **Timings** (dgx4, re-index call against next-candidate's position lists):

  | Form of the bound | Block ids vs positions (µs) | Record |
  | --- | --- | --- |
  | every load predicated on the count | 257.3 vs 247.1 (+4.1%, interval +7.9 to +10.9) | [runs/next-levers-predicated.json](runs/next-levers-predicated.json) |
  | every offset sent to position 0 through `if_then_else` | 262.9 vs 247.1 (+6.4%, interval +13.5 to +17.3) | [runs/next-levers-position0.json](runs/next-levers-position0.json) |
  | every offset selected (`T.Select`) | 260.4 vs 250.0 (+4.2%, interval spans 0) | [runs/next-levers-select.json](runs/next-levers-select.json) |
  | **the count's tile only, chosen once a tile** | **238.6 vs 246.3 (−3.1%, interval −27.6 to +14.3)** | [runs/next-levers.json](runs/next-levers.json) |

- **The other levers:**
  - **0504:** no change. The 2,048-token threshold is where the average expert fills a 32-row
    tile, and 128-row tiles exceed shared memory and registers. Its benchmark got a closure
    lint fix and ruff formatting.
  - **0505 and 0506:** no change.

**dgx4 runs** (`kernel-local`):
- **Setup:**
  - image `-next-6`, with next-candidate's tree mounted and its deleted modules failing import;
  - dgx4's compute-sanitizer mounted, so the read-bound test (`test_indexer_reads_only_visible_rows`)
    runs instead of skipping;
  - bundles `bundles/next-candidate` and `bundles/next-levers`.
- **next-candidate itself** ([runs/next-candidate.json](runs/next-candidate.json)): 203
  passed, 1 xfailed over every test file the levers touch, the initcheck included. The
  release's 12 failures pass there, so nothing is deselected.
- **The levers** ([runs/next-levers.json](runs/next-levers.json)):
  - **Tests:** 228 passed, 1 xfailed.
  - **The prefill-tiles gate:** bits identical. Routed MoE −12.2% at TP3's 8,192 (23.62 →
    20.74 ms), −16.9% at 12,288 and −15.4% at TP4's 16,384 (32.73 → 27.69 ms).
  - **The candidate-ids gate:** bit-identical and not slower (above).
  - **The lane check.**
  - **The positive control** ([initcheck_control.py](initcheck_control.py)): the indexer without
    the count's bound reports thousands of unwritten reads under initcheck.
- **The FAIL in kernel-local's own output:** its `exit` verdict reads the last output line as a
  pytest summary, and these bundles end with a check. run.py's exit status (0, recorded as
  `exit`) is the verdict.
- **Not run per commit:** the GPU checks ran on the branch head, not on each commit's own tree.

**With the kernel audit:**
- **The conflict:** `kernel-audit-next` (39fcba699) conflicts with this branch only in
  `test_deepseek_v41_tilelang.py`, where both sides append.
- **More than keeping both:** the audit's `test_indexer_reads_only_live_pages[candidates=True]`
  passes position lists to the re-index plan, which takes block ids after 0507. Whichever lands
  second rebuilds that case in block ids, as `deepseek_v41_indexer_reads.py` does.

**Needs the cluster:**
- [tp3-memory-cuts.json](tp3-memory-cuts.json) and
  [tp3-memory-cuts-gate.json](tp3-memory-cuts-gate.json) measure the cuts over candidate 6.
- For the new candidate, the same questions are answered by its own triangle qualification:
  - its start-up low against the 5 GiB guard;
  - its determinism gate;
  - the prefill suite for 0504's gain.

## Built and tested for next week (2026-10-25 evening, CET)

Owner direction: "queue up the tests that TP=3 recommends". Nothing else open covered these
levers, so each is now a vLLM patch with tests, checked on dgx4.
- **Branches:** local, off the release's 26382f5cf, one variable each, not pushed (the
  freeze).
- **Patches:** 0504-0507, after the 0491-0503 other drafts hold.
- **Bundles:** [make_bundles.py](make_bundles.py) writes them from the branches. They ran
  with `scripts/lab.py kernel-local` on dgx4, on the `-next-6` image, under this
  session's own checkout there; nothing ran on dgx1-dgx3.
- **Records:** [runs/](runs).

| # | Lever (patch, vLLM branch) | dgx4 result | Saving or gain | Still needs the cluster |
| --- | --- | --- | --- | --- |
| 1 | 64-row MoE tiles in prefill calls (0504, `prefill-moe-tiles` 8149aa4ad) | 153 tests pass; the gated benchmark passes (outputs identical at every size and routing, faster from 2,048 tokens, nowhere slower) | routed MoE −12.8% at 8,192 and −16% at 12,288 (TP3), −15% at 16,384 (TP4): prefill about +4% / +5% / +4% | determinism gate and prefill suite on both shapes (arms `prefill-tiles-tp3.json`, `prefill-tiles-tp4.json`) |
| 2 | Drop the two unread buffers (0505, `unread-buffers` 406886c8f) | 80 tests pass (the release's own 12 failures skipped) | exactly 10 KiB per budget token: 80 / 120 / 160 MiB at 8,192 / 12,288 / 16,384 | [tp3-memory-cuts.json](tp3-memory-cuts.json) |
| 3 | The drafter's workspace lane holds only the drafter's scratch (0506, `drafter-lane-scratch` b78a6a7ab) | 102 tests pass, the new lane test included; the lane check from the release's plan functions | about 0.80 GiB per rank on both shapes at every budget (837.25 → 12.84 MiB at TP3's 8,192): a ceiling, see below | the same spec (its slot log shows each lane's size) |
| 4 | Layer 20's candidates as block ids (0507, `candidate-block-ids` 5473283ef) | 111 tests pass; the gated check passes: the ids expand to the release's positions exactly, scores and selections are bit-identical, and the re-index call is not slower | 56 KiB per budget token: 448 / 672 / 896 MiB | the same spec |

### Lever 1: 64-row tiles for the routed experts in prefill calls

Each grouped-GEMM tile holds `block_M` routes of one expert and reads that expert's weight
slab again. At prefill sizes the 32-row decode tile makes the routed GEMMs L2-bound on
weight re-reads.
- **The rule:** a call takes 64-row tiles once the average expert has a full 32-row tile of
  routes. That is `rows × top-k ≥ 32 × experts`: 2,048 tokens for 384 experts at top-6.
- **What stays at 32:** decode steps and their CUDA graphs (at most 96 rows), the CED
  decoder's compacted rows and the drafter's MoE.
- **Declared at start-up:** both heights' GEMMs, fused SwiGLU and dispatch plans.
- **Workspace:** each call's scratch is laid out at its own height, about 0.16 GiB more at
  TP3's 8,192 tokens (12,288 more layout rows).

Bits are unchanged: a row's sum over K runs in the same order at either height.
- **New tests:**
  - the two heights through the whole plan, at the threshold and above;
  - the fused SwiGLU against TileKernels at 64-row tiles;
  - the tile-height rule and the scratch layout.
- **Existing tests:** the batch-composition checks now cross the two heights (one row alone
  runs at 32, the batch at 64).

The gated benchmark (`benchmark_deepseek_v41_moe_prefill_tiles.py`) times one layer's routed
experts as serving calls them (`plan.run`) against the release's own `moe.py`.
- **Shapes:** TP4's and TP3's widths, uniform and skewed routing, 1,024-24,576 tokens. Under
  sequence-parallel prefill the MoE runs on the chunk's gathered rows on every rank, so a
  call's rows are the chunk's tokens on either shape.
- **Fair memory:** both plans read one weight copy and share one workspace. Its first run
  gave each plan its own weight copy, and the rows below the threshold, which run the
  release's kernels in both plans, read 0.3-1.5% apart. That is GB10's memory-placement
  spread (experiments/2026-10-19-memory-placement); with one shared copy those rows read the
  same.

| Tokens | TP3 uniform | TP3 skewed | TP4 uniform | TP4 skewed |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 (32 rows: the release's kernels) | +0.5% (interval spans 0) | +0.2% (spans 0) | −0.3% (spans 0) | +0.4% (spans 0) |
| 2,048 | −2.6% | −2.1% | −1.5% | −2.9% |
| 4,096 | −6.3% | −9.0% | −4.3% | −6.6% |
| 8,192 | −12.8% (24.09 → 21.04 ms) | −11.9% | −11.0% | −10.4% |
| 12,288 | −16.1% | −15.2% | −13.3% | −14.2% |
| 16,384 | −17.4% | −15.7% | −15.2% (33.75 → 28.69 ms) | −14.0% |
| 24,576 | −18.1% | −18.5% | −16.2% | −16.4% |

What this means for a prefill chunk:
- 20 full-row layers a chunk turn this into −61 ms at TP3's 8,192 (about +4% prefill),
  −107 ms at 12,288 (+5%), and −101 ms at TP4's 16,384 (+4%).
- It shrinks the fixed cost of a chunk and its per-token cost alike, so it helps at every
  budget without lengthening any decode step.

### Levers 2-4: the memory cuts

**The release's own failures.** Twelve tests in `tests/models/test_deepseek_v4_1_attention.py`
fail on the release's tree with no overlay at all (7 failures, 5 errors in four test
functions; [runs/release.json](runs/release.json)). They build attention modules by hand
without attributes the release's geometry now sets. Every lever bundle skips them
(`RELEASE_FAILURES` in make_bundles.py), so its verdict reads its own change.

**Lever 2: the two unread buffers (0505).**
- `DeepseekV4Model.candidate_block_buffer` (budget × 2,048 int32) went to every attention
  layer. The layer takes the argument and never keeps it: the candidate source, layer 20,
  publishes its candidates in its own staging (`_candidates`).
- `DSparkDeepseekV4Model.topk_indices_buffer` (budget × 512 int32) went to the drafter's
  three layers. They are window-only (compress ratio 0): their attention is its own owner,
  has no compressed cache, and so reads no top-k buffer. DSpark's loader shares no indexer
  storage with the target.
- Both allocations go, and the drafter now refuses a layer that is not window-only.
- **Saving:** exactly 10 KiB per budget token per rank, 80 / 120 / 160 MiB at 8,192 /
  12,288 / 16,384.
- **Tests:** the attention, CED-model, SP-prefill and DSpark context-KV tests, 80 passing on
  dgx4 ([runs/unread-buffers.json](runs/unread-buffers.json)).

**Lever 3: the drafter's workspace lane (0506).**
- **Before:** `WorkspaceManager.reserve_all` sizes every slot of every lane to the largest
  reservation so far. Each TileLang attention layer called it when its plans were bound, so
  DSpark's lane held layer 20's indexing scratch.
- **After:** the layer records its reservations when the plans are built and reserves them
  in `reserve_profile_scratch`. The V2 runner calls that in the lane of each model holding
  the layer, before profiling, and then evens each lane's microbatch slots out. That is the
  per-lane design `tests/v1/worker/test_workspace.py` already describes for other models.
- **The new test** drives the runner's reservation with a target index layer and a
  window-only draft layer.
- **The bound** comes from the release's own plan functions on dgx4
  ([lane_check.py](lane_check.py), [runs/drafter-lane.json](runs/drafter-lane.json)):

| Largest reservation by layer role | TP3 8,192 | TP3 16,384 | TP4 16,384 |
| --- | ---: | ---: | ---: |
| candidate source (20): 256 rows × 1M BF16 scores, its top-k and block selection | 837.25 MiB | 837.25 MiB | 837.25 MiB |
| ratio-2 indexers (2, 8, 14) | 358.25 | 358.25 | 358.25 |
| compressed layers' metadata arena (budget × (128 + 2 + 4,096) int32) | 138.65 | 270.71 | 268.52 |
| drafter (40-42): window-only arena | 12.84 | 18.90 | 16.64 |
| **the drafter's lane: before → after** | **837.25 → 12.84 MiB: −0.80 GiB** | **−0.80 GiB** | **−0.80 GiB** |

  - The saving is about 0.80 GiB per rank on both shapes at every budget: a fixed cost, not
    one per token.
  - The lane also holds the drafter's other borrows, such as its MoE scratch for draft rows.
    DSpark's context-KV projection allocates its own tensors, outside the workspace.
  - So 0.80 GiB is the ceiling, and the cluster spec's slot log gives the lane's final size.
- **Tests:** the workspace tests, the new one included, and the model tests, 102 passing on
  dgx4.

**Lever 4: layer 20's candidates as block ids (0507).**
- **Before:** the candidate source (layer 20) published, for each row, the 16,384 positions
  of the 2,048 blocks of 8 it keeps, and the re-index layers (24, 28, 32, 36) read them back.
- **Why block ids are exact:**
  - Every kept block but the last is wholly visible.
  - The last block holds the row's last position, which is always kept and the highest.
  - So the list is the blocks' positions, with −1 only past its count: candidate `j <
    count` is `ids[j // 8] * 8 + j % 8`, entry for entry.
- **After:**
  - The source writes the 2,048 block ids (−1 padded) and the count of positions.
  - The re-index score kernel computes each candidate's position from them.
  - The re-index top-k maps the entries it keeps through them.
  - A selected entry never lies in a masked block or past the count: those score −inf or
    lie past the row's length.
- **B12X:** its indexer still exchanges positions. The width is the attention class's
  `CANDIDATE_WIDTH`: 16,384 in the base class, 2,048 in the TileLang class.
- **Tests:**
  - the grouped top-k and the source plan against the reference, in block ids;
  - re-index rows reading block-id lists: a partly visible last block, a masked block, an
    empty row;
  - a chain test: the source's ids expand to the release's position list exactly, and the
    re-index scores and selection equal the reference over it.
- **Gated check** (`benchmark_deepseek_v41_candidate_ids.py`): the release's `indexer.py` and
  this one on the same inputs at the serving shapes (1M positions, pages of 256, 32 rows,
  three seeds). Gated on bit equality and no slowdown.
- **dgx4, 111 tests pass** ([runs/candidate-ids.json](runs/candidate-ids.json)):
  - bit equality on every seed: the source's top-512, scores, counts, the expanded list, the
    re-index scores and top-512;
  - the re-index call 244.7 against 249.0 µs (interval −23.0 to +6.7);
  - its score kernel 228.5 against 232.7 µs (interval −4.6 to −2.8);
  - the top-k and the source step level.
- **Rejected along the way:** two variants were slower. Staging positions in shared memory
  doubled the re-index time, and a shift-and-OR position formula cost the score kernel
  6-8%.
- **Saving:** the TileLang candidate staging goes from 64 KiB to 8 KiB per budget token per
  rank: 448 / 672 / 896 MiB at 8,192 / 12,288 / 16,384. The workspace scratch is unchanged.
- **Start-up:** the changed kernels' compile keys differ, so the first boot compiles them
  again.
- **B12X tests:** `tests/v1/attention/test_b12x_v41_workspace.py` fails 45 of 50 on the
  release's own tree, and identically with 0507, so its bundle leaves that file out.

**Together** (levers 2-4, at TP3's 8,192): about 0.08 + 0.80 + 0.44 = 1.3 GiB a rank, and
1.6 GiB at 12,288.
- That is more than the 1.0 GiB the first record estimated: lever 3 measures 0.80 GiB
  against the 0.5 estimated.
- Lever 1 takes 0.16 GiB of it back.
- [tp3-memory-cuts.json](tp3-memory-cuts.json) measures it on the triangle (queue nothing
  until the coordinator schedules it):
  - candidate 6 at 8,192 with the allocator trace, then the same with the three cuts, then
    candidate 6 again, on the fit check's suite;
  - each arm's start-up and steady lows (memory/watch.sh with `SPEC=tp3-memory-cuts`);
  - the workspace slot log;
  - and, as a separate spec so its table does not overwrite the suite's
    ([tp3-memory-cuts-gate.json](tp3-memory-cuts-gate.json)), the determinism gate on the
    cuts arm against candidate 6's hashes (62313cb5 / 9742ac2c, 4d143977 / 12fd8257).

  Both plan cleanly with `--dry-run`. The merged bundle, `bundles/memory-cuts` (vLLM
  `memory-cuts` 63f5984f5: the three branches merged over the release), runs every cut's
  tests together on dgx4: 133 pass, 1 xfailed, the release's own 12 failures skipped
  ([runs/memory-cuts.json](runs/memory-cuts.json)).

  Its cuts arm, `cuts-8192-tp3.json`, comes from `make_arms.py` once the merged
  `memory-cuts` bundle exists.

**Overlap with the other drafts** (test merges of the vLLM branches):
- **#593 (`moe-exchange-direct`) and #585 (`kernel-audit`):** both merge with every lever
  without a textual conflict.
- **The one shared function:** #593 defers the combine at the end of
  `MoeExpertsV41Plan.run`, and lever 1 picks the dispatch plan at its top. The deferred
  combine then uses the chosen plan, which is correct.
- **Rule:** whichever lands second reruns the other's MoE tests.

**Display carve-out: no placement work planned.** The 1.27 GiB still free in the carve-out on each triangle node, and the whole 2.1 GiB reserve, are expected back as ordinary memory after NVIDIA's BIOS update, which lets the Spark skip the reservation (owner, 2026-10-25).

Owner, 2026-10-25: why can the ring's long-prefill budget (`max_num_batched_tokens`) go so
much higher than the triangle's, and should the implementation improve?

Base: candidate 6 (vLLM `tilelang-ports` 26382f5cf, image `spark-ds41f:19f2c20ed4d6-next-6`).
- **Sources:**
  - existing runs: the 10-19 TP3 sweep, the 10-23 TP4 budget runs, the 10-23 memory window's
    allocator traces and samplers, the 2026-10-25 TP3 fit check (`candidate-6-prefill-tp3`) and
    request-inventory-5's 15,395-token prefill trace;
  - two read-only code audits of the tree;
  - one single-node kernel check on dgx4 ([bundles/moe-rows](bundles/moe-rows)).
- **Not done:** no cluster time, nothing changed in serving, and no vLLM branch yet. Every lever
  below is bounded, none is built.

## Answer

- **Headroom.** The ring has about three times the triangle's. At its budget, dgx1 keeps
  18.7-20.1 GiB on the ring (16,384) and 6.67 GiB on the triangle (8,192).
  - A TP3 rank holds 94.1 GiB of weights against 73.0 GiB on TP4.
  - The smaller KV pool (4.9 GiB against 10.5) gives back only a quarter of those 21 GiB.
- **Cost per budget token.** It is about the same on both shapes: 0.33-0.48 GiB a rank per
  1,024 tokens on the ring, 0.36-0.41 GiB on the triangle.
  - Sequence-parallel prefill splits only the residual and mHC work.
  - Attention, the indexer, the compressors and the MoE run on all T gathered rows on every rank,
    and their buffers are sized by T.
- **Why 3.3 GiB against 0.7.** The ring's 8,192 → 16,384 step added twice the tokens of the
  triangle's 4,096 → 8,192.
  - It cost 3.4-3.9 GiB on every ring node.
  - The triangle step looked cheap on dgx1 (−0.5 GiB) because:
    - single-run lows on the triangle move by up to 0.55 GiB between identical arms, and dgx2
      lost 2.0 GiB in the same arms;
    - below about 6.7K tokens the MoE's workspace sits under a fixed 0.88 GB floor.
- **12,288 on the triangle.** Expect about 1.4 GiB less at the lows, for +3.3-5.3% prefill:
  - dgx1 6.67 → about 5.2 GiB during 1M prefills;
  - the start-up low falls by the same amount; it is not recorded today, and must be above
    about 6.4 GiB now for 12,288 to clear the 5 GiB guard (the sweep below measures it);
  - the median decode step beside a long prompt 1.44 → about 2.0 s.
- **16,384 on the triangle.** It needs about 2.9 GiB for +5.0-7.2% prefill and a median step
  of about 2.6 s. It does not fit today.
- **The implementation can improve on both axes.**
  - **Memory:** three low-risk cuts free about 1.0 GiB a rank at 8,192 and 1.3 GiB at 12,288,
    about what 12,288 needs. NVIDIA's BIOS update is expected to return the 2.1 GiB display
    reserve as ordinary memory on every node (owner, 2026-10-25).
  - **Speed:**
    - Each chunk pays a fixed cost of about 196 ms (TP3) and 155 ms (TP4). About 70% of it is
      the routed MoE's size-independent time: about 70% of one read of every expert in the 20
      full-row layers, which compute does not hide.
    - A 64-row MoE tile for prefill is bit-identical on dgx4 and makes the routed MoE 11-19%
      faster. That is about +3.5-5% prefill on both shapes, with no stall trade and no guard
      change.

## 1. Headroom per shape

| | TP4 ring at 16,384 | TP3 triangle at 8,192 |
| --- | --- | --- |
| weights a rank (`Model loading took`) | 72.79 GiB | 94.12 GiB |
| KV pool | 10.5 GiB | 4.91 GiB |
| display carve-out | unused (the head lost its L2 reuse there, 2026-10-12) | 737.2 MiB of about 2,040 MiB: embedding + packed head |
| graph capture + tail-shape warm-up | reserve +2.84 GiB, MemAvailable −3.3 GiB | "took 1.93 GiB" (engine log) |
| lowest MemAvailable dgx1 / others (budget suite) | 18.66 / 19.93-20.15 GiB (candidate 1) | 6.67 / 7.53, 7.94 GiB (candidate 6) |
| guards | 5 GiB start-up, 3 GiB running, 4 GiB bench floor | same |

On the triangle, dgx1 is the floor by 0.86 GiB (dgx2) and 1.27 GiB (dgx3) at the fit check.
Candidate 6 at 8,192 sits 1.4-2.1 GiB above r6x at the same budget on all three nodes
(10-19: 4.57 / 5.49 / 6.50 GiB). Existing data cannot attribute this.
- glibc pinning's measured effect was dgx1's API server (the 10-23 soak), not the other nodes.
- B12X's removal from serving is the likely source, unverified.
The sweep below measures candidate 6's own per-token cost.

## 2. What grows with the budget

### Measured, per rank (budget suite: 32K-500K prefill, the 131K stall, cold 1M prefills)

| Step | dgx1 | dgx2 | dgx3 | dgx4 | GiB per 1,024 tokens |
| --- | ---: | ---: | ---: | ---: | --- |
| TP4 candidate 1, 8,192 → 16,384 (`next-budget-tp4`, two 8,192 arms) | −3.40 | −3.38 | −3.67 | −3.86 | 0.42-0.48 |
| TP4 candidate 1, 16,384 → 24,576 (`next-budget24-tp4`) | −2.60 | −2.71 | −2.24 | −2.62 | 0.28-0.34 |
| TP3 r6x + skip, 4,096 → 8,192 (`prefill-budget-tp3`, two 4,096 arms) | −0.54 | −2.02 | −0.93 | | 0.13-0.50 |
| TP3 r6x, 4,096 → 8,192 (experiments/2026-10-19-r6x-tp3-1m: serving / 1M low) | −1.63 / −1.44 | | | | 0.36-0.41 |

The two 4,096 arms of the TP3 sweep differ by 0.30 / 0.55 / 0.37 GiB on dgx1 / dgx2 / dgx3, so
single-run TP3 lows resolve no better than about half a GiB.

### The ring's boot, traced (dgx2, 16,384, 2026-10-23 18:33 CET)

From [memory/analyze_sweep.py](memory/analyze_sweep.py) on the memory window's allocator trace
and sampler:

| after | torch allocated / reserved | MemAvailable | driver (GPU) | anon |
| --- | --- | ---: | ---: | ---: |
| weights | 72.99 / 73.27 GiB | 39.30 | 78.24 | 3.36 |
| kernel preparation, the profile run and its release | 77.41 / 77.72 | 33.89 | 82.40 | 4.29 |
| KV pool | 87.94 / 88.14 | 23.35 | 92.86 | 4.29 |
| graph capture and tail-shape warm-up | 88.00 / 90.98 | 20.03 | 95.98 | 4.71 |
| serving: lowest; torch peak | allocated 89.59, reserved 91.04 | 19.95 | | |

What each phase means:
- **Released memory returns.** The profile run's peak (reserved 79.8 GiB) went back to the OS at
  its `empty_cache` (driver −2.0 GiB).
- **The budget-sized forward that stays reserved is the tail-shape warm-up.**
  - Inside `capture_model`, after capture's own `empty_cache`, adaptive verification runs eager
    dummy forwards that double up to the budget
    (`preparation_tail_sizes` / `warm_tail_shapes`, `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:96-108, 189-196`,
    called at `gpu/model_runner.py:1430`).
  - Their activation peak stays cached: reserved +2.84 GiB with allocated flat.
  - Serving's long prefills then reuse it: reserved +0.06 GiB more over 75 minutes of prompts up
    to 966K tokens.
- **The GPU's memory outside torch is about 1 GiB a rank:** nvidia-smi 92.07 GiB against
  91.04 GiB reserved.

### From the code (per budget token, per rank, TP3 / TP4)

Two audits (model directory; runner, workspace and collectives), spot-checked; file:line at
26382f5cf.

| Allocation | Bytes a token | Kind |
| --- | --- | --- |
| MoE plan in workspace lane 0 (`tilelang/moe.py:1257-1269`): per-pick down outputs 6 × 5,120 BF16 (60 KiB), gate/up output (18 / 13.5 KiB), MXFP8 inputs (about 10 KiB), fold output (10 KiB) | about 100 / 95 KiB, above a fixed 0.88 GB floor (layer 20's indexer chunk scratch, sized by max_model_len) that it exceeds from about 6.7K / 7.2K tokens | persistent; grows in the profile run |
| layer 20's `_candidates` (budget × 16,384 int32) + lengths (`attention.py:721`) | 64 KiB | persistent; written only for rows that reach the CED decoder, except for requests that keep every row (prompt logprobs) |
| compressor scratch, layers 2, 8, 14, 20 (`compressor.py:207-215`) | 17 KiB | persistent |
| DSpark speculator context, encoder `inputs_embeds` (budget × 5,120 BF16 each) | 10 + 10 KiB | persistent |
| Engram staged rows, two tables (`common/engram.py:900`) | 8 / 6 KiB | persistent |
| Engram pinned host buffers (`engram_rows/rows.py:129-131`), every column although a rank writes only its 24/N (`_row_plan.h:257`) | 12.75 KiB, of which 8.5 / 9.6 KiB never written | pinned host |
| `candidate_block_buffer` (budget × 2,048 int32, `nvidia/model.py:456`): the attention takes the argument and never stores it | 8 KiB | persistent, unread |
| DSpark's top-k buffer (budget × 512 int32, `nvidia/dspark.py:405`): the drafter's layers are window-only | 2 KiB | persistent, unread |
| shared top-k, metadata builders, CED and mHC odds | about 3 KiB | persistent |
| **persistent total** | **about 230 / 220 KiB** | |
| activation peak a budget-sized forward leaves cached (aux hidden states, residual carries, MoE partials, gathered rows, q and attention outputs, WO send buffers) | up to 155 / 131 KiB (ring at 16,384: +2.84 GiB reserve at capture = 182 KiB with graph pools) | held after the warm-up |
| **total** | **about 385 / 350 KiB = 0.37 / 0.34 GiB per 1,024 tokens** | matches the measured rates |

**Sized for the full chunk where SP needs a share?**
- Only the residual side runs on local rows, and its buffers are local (the mHC constant mixes,
  32 B a token, are the exception).
- Everything large runs on the gathered rows by design.
- The waste is elsewhere:
  - **CED rows:** budget-sized buffers that only the CED decoder's compacted rows use (layer 20's
    candidates: at most max_num_seqs × 128 rows unless a request keeps every row);
  - two unread buffers;
  - Engram's unwritten columns;
  - **the drafter's workspace lane (a fixed cost):**
    - `WorkspaceManager.reserve_all` (`vllm/v1/worker/workspace.py:224-251`) sizes every slot of
      both lanes to the largest reservation.
    - Every target attention layer reserves its indexer scratch through it
      (`tilelang/attention.py:205-214`).
    - So the DSpark lane holds at least 0.88 GB although its window-only layers borrow about
      0.27 GB at 8,192.
    - The sweep's slot log confirms the size.

## 3. The rank-0 tax

The ring at 16,384 with glibc pinned (soak, 2026-10-23 18:33-19:50 CET; process memory from
[process_memory.py](process_memory.py)):

| Process | dgx1 anon PSS (start → after 75 min) | other nodes |
| --- | --- | --- |
| API server | 1.06 → 1.43 GiB | none |
| engine core | 0.71 → 0.97 GiB | none |
| headless launcher (`vllm serve --headless`, idle) | none | 0.98 GiB, flat |
| worker | 3.30-3.34 GiB | 3.30-3.34 GiB |

The head node's extra:
- 0.79 GiB at the start, 1.42 GiB after 75 minutes of million-token agent traffic;
- on the triangle at the fit check, dgx1 is 0.86 / 1.27 GiB below dgx2 / dgx3.
- humming's nvrtc and ops probes spawn at start-up, but each lives a second or two, before the
  weights load.

What could move:
- **The API server, off the triangle** (vLLM's API-only frontend against a headless engine).
  It pays only until dgx2 becomes the floor: +0.86 GiB at the fit check.
- **The engine core** stays with the mp executor.
- **The display carve-out:** the triangle uses 737.2 MiB of its about 2,040 MiB (the embedding and the packed head). The 1.27 GiB still free, and the whole reserve, are expected back as ordinary memory after NVIDIA's BIOS update (owner, 2026-10-25), so no placement work is planned.

## 4. What every chunk pays regardless of its size

**TTFT fits** ([fit_chunks.py](fit_chunks.py)): TTFT = K_N + chunks × b, per prompt size, across
a run's budgets.

| Run | b, ms a chunk (32K / 131K / 262-524K / 1M) | worst residual |
| --- | --- | --- |
| TP3 r6x + skip, 4,096 / 6,144 / 8,192 | 192 / 198 / 196-200 / 203 | 0.06-0.45 s |
| TP4 candidate 1, 8,192 / 16,384 | 151 / 157 / 158 / 168 | 0.01-0.36 s |
| TP4 candidate 1, 16,384 / 24,576 | none (two chunks either way) / 139 / 162 / 154 | 0.04-0.37 s |

The fixed cost does not depend on context. It is 13% of a TP3 chunk at 8,192 (1.48 s at 32K)
and 6-7% of a TP4 chunk at 16,384 (2.43 s).

**One forward** ([prefill_trace.py](prefill_trace.py) on request-inventory-5's 15,395-token
prompt, TP4 rank 0): 2,140 ms, GPU busy 2,134 ms (no host gaps).
- **Layers 0-19 run on all rows:** 91-101 ms each; layer 0 is 174 ms with the embedding
  exchange, and the Engram layers are longer.
- **Layers 20-39 run on the CED decoder's 128 compacted rows:** 3.6-5.9 ms each, 87 ms in all,
  52 ms of it routed MoE.
  - A prompt's final chunk pays these.
  - Nonfinal chunks keep one decoder row (the skip): a few milliseconds of launches a layer.

**One layer's routed experts against the chunk size** (dgx4, the release kernels, uniform
routing, [bundles/moe-rows](bundles/moe-rows), [moe-rows.json](bundles/moe-rows/moe-rows.json);
2026-10-25 20:16 CET):

| | TP3 (768 columns) | TP4 (576) |
| --- | --- | --- |
| all 384 experts once (512 tokens) | 9.71 ms (241 GB/s) | 7.52 ms (234 GB/s) |
| fit over 2,048-24,576 tokens, 32-row tiles | 6.93 ms + 2.06 µs a token | 5.39 ms + 1.62 µs a token |
| the fixed part × 20 full-row layers | **139 ms a chunk (71% of b)** | **108 ms (70% of b)** |
| 64-row tiles | 7.69 ms + 1.58 µs a token | 5.80 ms + 1.32 µs a token |
| 64 against 32 rows at 4,096 / 8,192 / 12,288 / 16,384 tokens | −7.3 / −11.0 / −16.2 / −19.1% | −5.6 / −11.4 / −12.4 / −14.8% |
| outputs, 64 against 32 rows | bit-identical at every size | bit-identical |

So about 70% of one full read of each full-row layer's experts is not hidden under compute.
That one cost is most of what a chunk pays regardless of its size.

The remaining 55-65 ms a chunk is not attributed. The candidates:
- the latency of about 170 NCCL exchanges and gathers;
- the decoder's one-row pass;
- small kernels and partial waves.

The first chunk's Engram wait (35-40 ms) and the final chunk's 128-row decoder pass are per
prompt (in K), not per chunk. Per-token communication is the fusion survey's subject
(experiments/2026-10-25-fusion-survey, #583, #593).

The kernel lab marks this bundle FAIL only because its `exit` verdict expects pytest output.
The step exited 0 in 43 s.

## 5. Ceilings: 12,288 and 16,384 on the triangle

The forecast takes candidate 6's fit-check TTFTs at 8,192 and moves the chunk count with
b = 196 ms. It assumes per-token cost does not depend on the chunk size: the TP4 fits held
within 0.5% at a third budget.

| Prompt | 8,192 (measured) | 12,288 | 16,384 |
| --- | ---: | ---: | ---: |
| 32,761 tokens | 5,542 tok/s | 5,733 (+3.4%) | 5,936 (+7.1%) |
| 131,900 | 5,618 | 5,914 (+5.3%) | 6,020 (+7.2%) |
| 262,554 | 5,343 | 5,589 (+4.6%) | 5,708 (+6.8%) |
| 500,373 | 4,828 | 5,028 (+4.1%) | 5,129 (+6.2%) |
| 999,979 | 3,949 | 4,078 (+3.3%) | 4,145 (+5.0%) |
| median decode step beside a 131K prompt | 1.44 s | about 2.0 s (+41%) | about 2.6 s (+81%) |
| memory at the lows, at 0.35-0.40 GiB per 1,024 tokens | 6.67 GiB (dgx1) | about 5.1-5.3 | about 3.5-3.9: below the bench floor |

On the ring's curve at equal rows per rank, 12,288 on the triangle is the ring's 16,384
(4,096 rows a rank). The ring's 2,048 → 4,096 rows gave +5.4-6.3%, the same order as the
forecast. Rows per rank is the wrong axis, though: the fixed cost follows the experts' bytes per
rank and the number of chunks, not SP's row split.

The 64-row MoE tile adds about +3.6% at 8,192 and +5% at 12,288 (−51 / −103 ms a chunk) on top
of these figures.

## Ranked levers

Gains are per triangle rank unless marked. Nothing is built; each lever is one variable for
its own branch.

| # | Lever | Gain | Evidence | Effort | Risk |
| --- | --- | --- | --- | --- | --- |
| 1 | 64-row tiles for the routed MoE in prefill forwards (2,048 rows or more; decode keeps 32) | MoE −11% (8,192) / −16% (12,288) / −19% (16,384); prefill about +3.6% at 8,192 and +5% at 12,288 on TP3, about +4% at 16,384 on TP4; no decode change; costs about 0.16 GiB fixed (12,288 more layout rows) | dgx4 sweep, bit-identical outputs at every size; built as 0504, gated benchmark passes (above) | S-M: a second plan by row count, both declared at start-up | low |
| 2 | Layer 20's candidates as 2,048 block ids a row instead of 16,384 positions | 0.44 / 0.66 / 0.88 GiB at 8,192 / 12,288 / 16,384 | code (`attention.py:721`; the four reindex layers read positions); built as 0507, bit-identical on dgx4 (above) | M: the producer's top-k writes ids, the consumers expand ×8 on load | low: same candidates in the same order; check outputs bit for bit |
| 3 | The drafter's workspace lane without the target's indexer scratch (reserve by lane) | about 0.80 GiB fixed, every rank, both shapes (first estimated at 0.5) | code (`workspace.py:224-251`, `tilelang/attention.py:205-214`); built as 0506, lane check on dgx4 (above); the final size comes from the triangle's slot log | S | low: the locked workspace refuses an undersized lane at start-up |
| 4 | Drop the two unread buffers | 0.08 / 0.12 / 0.16 GiB | code; built as 0505 (above) | S | none |
| 5 | The display carve-out | the 1.27 GiB still free (and the whole 2.1 GiB reserve) expected back as ordinary memory after NVIDIA's BIOS update (owner, 2026-10-25) | owner | none: no placement work planned | none |
| 6 | Hide the routed experts' weight stream under compute (prefetch the next experts' slabs, persistent CTAs) | ceiling 139 ms a chunk on TP3 (about 9% at 8,192), 108 ms on TP4 (about 4.5% at 16,384) | dgx4 sweep: the fixed part is 71% of one full weight read | L | medium: new kernel structure; same arithmetic per row |
| 7 | API server off the triangle | up to 0.86 GiB on dgx1, until dgx2 is the floor | soak process table | L: deployment topology | medium |
| 8 | SwiGLU in the gate/up GEMM's epilogue (no BF16 gate/up buffer) | 0.14 / 0.21 / 0.28 GiB, plus about 1% prefill | code; fusion survey row 12 | M-L | medium: the BF16 rounding point must be kept |
| 9 | Engram's pinned buffers for the rank's own columns | 0.07 / 0.10 / 0.13 GiB of pinned host memory | code | S-M | low |
| 10 | One compressor scratch set for the four compressor layers | about 0.09 / 0.13 / 0.18 GiB | code | M | low |

**Not levers:**
- **The MoE's per-pick down outputs** (60 KiB a token): splitting the routed MoE into row
  slices pays the fixed 6.9 ms a layer again for every extra slice.
- **The held activation peak:** every long prefill needs it, and releasing it between prefills
  does not raise the minimum the guard sees.
- **The guards:** never loosened.

**Together:**
- Levers 2-4 free about 1.0 GiB at 8,192 and 1.3 GiB at 12,288: enough for 12,288 at the
  lows the fit check measured.
- With the 2.1 GiB reserve back after the BIOS update, 16,384 (about 2.9 GiB) comes within
  reach. The embedding and head (0.72 GiB) then return to ordinary memory, so the net gain is
  about 1.4 GiB.
- Levers 1 and 6 take part of what a larger budget buys without its decode stall.

## Measurements

### Queued: nothing

The cluster sweep (item 1) was dropped by the owner on 2026-10-25 (see the status at the top). It
is kept here, ready, in case candidate 6's own cost per budget token is wanted later.

1. **[tp3-budget-memory.json](tp3-budget-memory.json): the TP3 budget sweep with the allocator
   trace.**
   - Arms: candidate 6 at 8,192, 12,288, 6,144, then 8,192 again, on the fit check's suite
     (`experiments/2026-10-19-prefill-budget-tp3/suite.json`). About 2 hours.
   - [make_arms.py](make_arms.py) generates the arms from `rc6/rc-tp3.json`. They differ only in
     the budget and threshold, a diagnostic env, and the overlay mount:
     - `budget-6144-tp3.json` resolves byte-identical to `rc6/rc-tp3-6144.json`'s serve_args;
     - the overlay is [memory/overlays](memory/overlays): the release's `gpu_worker.py` plus an
       allocator sampler thread, phase marks and per-pool summaries; serving behaviour is
       unchanged;
     - the env adds `VLLM_DEBUG_WORKSPACE=1`, which logs only slot resizes.
   - **Order:**
     - 12,288 runs second: if it trips the 5 GiB start-up guard, the boot fails closed and the
       other arms still run.
     - The repeat bounds the session's drift.
   - **Before queueing:**
     1. Merge this PR: the lab syncs the node checkouts to origin/main.
     2. Copy [memory/sampler.py](memory/sampler.py) to `~/spark-lab/tp3-budget-memory/` on dgx1-3.
     3. Start [memory/watch.sh](memory/watch.sh) on dgx1. It arms the samplers and an engine-log
        follower when the job starts, and collects the traces after the restore.
     4. Pass `--production-config experiments/2026-10-23-next-release/1022/cluster.json`
        (production on the triangle is 1022).
   - **Analysis:** [memory/analyze_sweep.py](memory/analyze_sweep.py) on the watcher's output. It
     gives, per arm and node:
     - the phase table above;
     - the start-up low against the guard;
     - the serving low;
     - the workspace slots;
     - the pools;
     - the change per 1,024 tokens per phase.
   - **What it decides:**
     - candidate 6's real cost of a budget token on the triangle;
     - whether 12,288 clears the start-up guard;
     - lever 3's size.
2. **Single-node, dgx4:**
   - **Done:**
     - [bundles/moe-rows](bundles/moe-rows) (the MoE sweep above);
     - the levers' bundles ("Built and tested for next week", [runs/](runs)).

## Files

- [README.md](README.md): this record.
- [fit_chunks.py](fit_chunks.py): the per-chunk fixed cost from a budget run's TTFTs.
- [prefill_trace.py](prefill_trace.py): one prefill forward split into full-row and compact
  layers.
- [process_memory.py](process_memory.py): a node's memory by component and process across an arm.
- [make_arms.py](make_arms.py): writes `budget-{6144,8192,12288}-tp3.json` and, once the merged
  memory-cuts bundle exists, `cuts-8192-tp3.json`.
- [tp3-budget-memory.json](tp3-budget-memory.json): the budget sweep spec (dropped, kept ready).
- [tp3-memory-cuts.json](tp3-memory-cuts.json) and [tp3-memory-cuts-gate.json](tp3-memory-cuts-gate.json):
  the memory cuts' triangle check and its determinism gate (not queued).
- [memory/](memory): the trace overlay, sampler (unchanged from 2026-10-23), watcher and
  analysis.
- [bundles/moe-rows](bundles/moe-rows): the first dgx4 kernel bundle and its record.
- [make_bundles.py](make_bundles.py): writes the levers' bundles (`bundles/<lever>`) and their
  serving arms (`<lever>-tp3.json`, `<lever>-tp4.json`) from the vLLM branches. It writes in
  place and stops on stale files; it never deletes. `bundles/next-candidate` and
  `bundles/next-levers` are next-candidate and `tp3-levers-next`, and `bundles/next2-*` the
  same after the kernel audit plus its attention steps; none has serving arms.
- [initcheck_control.py](initcheck_control.py): the read-bound test's positive control in
  candidate mode.
- [lane_check.py](lane_check.py): each attention layer role's workspace reservation (lever 3's
  bound).
- [record_run.py](record_run.py): a kernel-lab output directory as a run record.
- [runs/](runs): the dgx4 run records.
- [vllm/](vllm): patches 0504-0507 (`git format-patch` of the four local branches, each on
  the release's 26382f5cf).
- [vllm-next-candidate/](vllm-next-candidate): the same four on next-candidate 4209a1903, as
  `tp3-levers-next` holds them.
- [vllm-next2-candidate/](vllm-next2-candidate): the same four on next-candidate 16c1b8cae
  (with the kernel audit), as `tp3-levers-next2` holds them.
