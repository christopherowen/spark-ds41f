# candidate-54: attributing the small-batch grid loss at tp3-switched

`../tp3-switched/README.md` finds candidate-54's engine grid at `tp3-switched` 2.87% slower than 1107, all of it at
small batches. This folder is the attribution's first part: an analysis of the recorded grids and the controls for
the testbed run. The sub-agent's record is [result.json](result.json); its commands are in
[commands.txt](commands.txt), and its evidence in `raw/`.

**From the recorded grids** (`raw/grid-split-from-receipts.txt`):
- The loss is all in the target forward: +0.95 to +1.20 ms a step at 1 request, at every context from 4K to 1016K.
  The drafter is unchanged at 1-2 requests and 0.14-0.19 ms faster at 4-8.
- It shrinks with rows (+0.8 ms at 8 rows, +0.3 at 16, -0.12 at 48): a fixed cost a step, hidden by longer steps.
- It mirrors candidate-51's `tp4-ring` gain cell by cell (1x3: -1.24 ms at TP4, +1.20 at TP3).

**Suspects, from reading the code:**
- Ruled out for the grid: #987 (its 128-row WO-B tile acts only at 97-128 rows), #1001 r1 (65-128-row vocab
  tiles), #968 (the scheduler) and #969 (the Engram disk reader).
- Leading, unmeasured: **#961**. Its prefetch windows run on one side stream that rejoins at the model's end, in
  64 KiB chunks with a fixed 200 ns pause. TP3's windows sit at their budget caps (20/16/20 MB, 56 MB, about 857
  chunks a layer, against TP4's 18.9/14.8/14.8 MB and 741).
- Low: #998 r1, #960 r1, #963 r1 and #1030.

**The testbed run:** [../../attribute-tp3-grid-tp3-switched.json](../../attribute-tp3-grid-tp3-switched.json), the grid
step alone ([../../grid-tp3-suite.json](../../grid-tp3-suite.json): screen-tp3's grid step), six boots in one
session: candidate-54, without #961, without #998, without #960, without #963 and #1030, then candidate-54 again.
Each `no<PR>` overlay holds the non-test files that differ when that commit is reverted from candidate-54's prepared
tree; all reverts applied cleanly. Overlays for #963, #987, #1001, #1030 and all of candidate-51 (`noC51`) are kept
for a second pass.

**The one-GPU harness did not run:** the coordinator's release-test script on sp8 wrapped `kernel-local` in `flock`
on the GPU lock, which `kernel-local` takes itself, and deadlocked the lock from 03:10:50Z to 04:30:49Z
(`raw/gpu-lock-deadlock.txt`).

## Result on sp5 (2026-11-10, 05:19-05:52Z)

`lab/candidate-54-attribute-tp3-grid-tp3-switched-table.txt` (the table's base is its first-sorted arm,
`candidate-54-no961`), the window log in `lab/candidate-54-attribute-tp3-grid-window.log`, the runs in `suites/`.
Against candidate-54 without #961:

| Arm | Grid geometric mean | 1x2 rows at 4K | 1x3 rows at 4K | 8x48 rows at 4K |
| --- | --- | --- | --- | --- |
| without #961 | (base) 18.70 ms | 18.70 ms | 18.87 ms | 28.84 ms |
| candidate-54 | +2.90% | +5.26% | +7.12% | -0.51% |
| candidate-54 again | +2.84% | +5.71% | +7.41% | -0.44% |
| without #998 | +2.45% | +4.30% | +6.40% | -0.51% |
| without #960 | +2.34% | +4.54% | +6.46% | -0.67% |
| without #963 and #1030 | +3.28% | +5.21% | +7.12% | -0.01% |

- **#961 causes the TP3 small-batch loss.** Without it the grid matches 1107's (`../tp3-switched/`). Its candidate-54
  repeat agrees within 0.06%.
- Smaller effects, one boot each: without #998 or #960 the grid is 0.45% or 0.56% faster than candidate-54; without
  #963 and #1030 it is 0.38% slower.
- **Next:** [../../attribute-tp3-prefetch-tp3-switched.json](../../attribute-tp3-prefetch-tp3-switched.json) tests
  recipe-only fixes on candidate-54's image: the prefetch windows capped at TP4's planned sizes
  (`tp3-switched-l2-tp4-budget-dev.json`, 19/15/15 MB) and the prefetch off (`tp3-switched-l2-off-dev.json`), against
  candidate-54 and candidate-54 without #961.

## Recipe-only fix (sp5, 2026-11-10, 06:08-06:28Z)

[../../attribute-tp3-prefetch-tp3-switched.json](../../attribute-tp3-prefetch-tp3-switched.json), the grid step alone,
six boots in one session (`lab/candidate-54-tp3-prefetch-tp3-switched-table.txt`; the table's base is
`candidate-54-l2-tp4-budget`, the windows capped at TP4's planned sizes, 19/15/15 MB):

| Arm | Grid geometric mean | 1x2 rows at 4K | 1x3 rows at 4K | 4x8 rows at 4K | 8x48 rows at 4K |
| --- | --- | --- | --- | --- | --- |
| windows 19/15/15 MB | (base) | 18.66 ms | 18.90 ms | 20.79 ms | 28.51 ms |
| windows 19/15/15 MB again | -0.28% | -0.13% | -0.39% | +0.11% | -0.51% |
| candidate-54 (20/16/20) | +2.20% | +6.58% | +7.75% | +1.31% | +0.47% |
| candidate-54 again | +1.56% | +4.48% | +6.03% | +1.14% | +0.29% |
| without #961 | -1.08% | +0.55% | -0.56% | -2.25% | +0.92% |
| prefetch off | +11.64% | +13.60% | +13.47% | +10.50% | +10.33% |

- **Capping TP3's windows at TP4's sizes removes the 1-request loss** (1x3: 18.90 against 20.37/20.05 ms, and 18.80
  without #961). The grid stays 1.1% behind candidate-54 without #961, mostly at 4 requests (4x8: -2.25% without #961).
- The prefetch is worth about 11% at TP3: the loss is #961's paced windows at TP3's cap-sized windows, not the
  prefetch itself.
- **Next:** [../../attribute-tp3-prefetch-sweep-tp3-switched.json](../../attribute-tp3-prefetch-sweep-tp3-switched.json)
  tries smaller caps (17/13/13 and 15/12/12 MB) against 19/15/15 and against candidate-54 without #961.

## Cap sweep (sp5, 2026-11-10, 06:32-06:53Z)

[../../attribute-tp3-prefetch-sweep-tp3-switched.json](../../attribute-tp3-prefetch-sweep-tp3-switched.json), six boots
in one session (`lab/candidate-54-tp3-prefetch-sweep-tp3-switched-table.txt`; the table's base is 17/13/13 MB):

| Arm | Grid geometric mean | 1x2 rows at 4K | 1x3 rows at 4K | 4x8 rows at 4K | 8x48 rows at 4K |
| --- | --- | --- | --- | --- | --- |
| 17/13/13 MB | (base) | 18.61 ms | 18.82 ms | 20.83 ms | 28.66 ms |
| 17/13/13 MB again | -0.25% | -0.04% | -0.10% | -0.47% | -0.51% |
| 19/15/15 MB | -0.42% | +0.24% | +0.15% | -0.27% | -0.66% |
| 19/15/15 MB again | -0.41% | -0.01% | -0.20% | -0.45% | +0.91% |
| **15/12/12 MB** | **-0.83%** | **-0.62%** | **-0.80%** | **-1.69%** | +0.20% |
| without #961 | -1.29% | +0.49% | -0.05% | -2.34% | +0.38% |

- **15/12/12 MB is the best cap measured**: as fast as candidate-54 without #961 at 1 request (1x2 18.49 against
  18.70 ms), within about 0.5% of it in the grid geometric mean, the gap mostly at 4 requests.
- **Recommendation:** carry `VLLM_DS41_L2_PREFETCH_WO_MB=15`, `_FFN_MB=12`, `_NEXT_MB=12` in the TP3 shapes'
  environment (`tp3-triangle`, `tp3-triangle-4k`, `tp3-switched`) in the next merged base, as its own variable, with
  TP4's recipe unchanged so #961's TP4 gain stays. The remaining 0.5% belongs to a #961 rework (pacing derived from
  each window's size) if one is wanted.

