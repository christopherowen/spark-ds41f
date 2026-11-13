# L2 prefetch: the WO window sized by row band

Base commit: `543e52288`. Image: `spark-ds41f:candidate-55`, fourth build (vLLM tree `8aaae3df`), with its TP4
recipe (L2 prefetch windows at 15/12/12 MB; at `tp4-ring`, collectives above the 2 MiB slot on PyNCCL). The change is
mounted over the image; nothing is built.

## Why

candidate-55's 24- and 32-row grid shapes at `tp4-ring` are 1.3% to 1.9% slower than candidate-54's, and the cause
is the WO prefetch window at 15 MB in the 32-row decode tile tier
([rows24-attribution](../2026-10-26-spark-ds41f-candidate/candidate-55/rows24-attribution/README.md)):

- with the prefetch off candidate-55 is faster than candidate-54 at those shapes;
- with WO at 10 MB for every row count they are 2.4% to 2.7% and 1.5% faster, but 1x2 rows are 1.9% slower and 48
  to 72 rows 0.5% to 1.6% slower, so the grid is the same.

One budget for every row count is wrong somewhere. The prefetcher's `issue()` already receives the forward's row
count, and under capture it runs once for each captured row count, so each graph can hold its own plan.

## The one variable

[The patch](vllm/0001-perf-dsv41-size-the-WO-and-COMP-L2-prefetch-windows-by-row-band.patch) (on candidate-55's vLLM
head `d6d4319e`; its tree `7d05b12b`) adds `VLLM_DS41_L2_PREFETCH_WO_BANDS`: budgets for the WO window, and for COMP,
which takes WO's budget in the four layers that own a compressor, by row count. `17-32=10` means 10 MB for forwards
of 17 to 32 rows and `VLLM_DS41_L2_PREFETCH_WO_MB` elsewhere.

- `build_l2_prefetch` makes one plan a band beside the default; `issue()` launches the plan the forward's row count
  selects. Nothing is decided at replay, and no kernel changes.
- Unset, the plan object is the one `make_plan` returns today.
- The drafter's layers build their windows through the same function, so the band applies to the drafter's forward
  by its own row count.

**Arm:** `VLLM_DS41_L2_PREFETCH_WO_BANDS=17-32=10`. The band is the 32-row decode tile tier (`DECODE_TILE_ROWS` is
16, 32 and wider); the 10 MB is the value the attribution measured. 20 and 28 rows are in the band and not in the
grid.

**Control:** the overlay with no band set (`candidate-55-overlay-unset`), which must measure as `candidate-55`.

[overlay/](overlay) holds the two files the patch changes outside `tests/`, from the fourth build's prepared tree
with the patch applied.

## Marks, declared before the runs

Against `candidate-55` in the same job, both boots of each, at each TP4 testbed:

1. **The band:** the four 24-row shapes at least 2% faster as a group, and 16x32 at least 1% faster.
2. **Outside the band:** every other shape the same within its interval, in particular 1x2, 1x3, 48, 72 and 96 rows.
3. **The grid:** its geometric mean faster beyond the margin.
4. **The drafter:** its time at every shape the same within `candidate-55`'s own repeat boots.
5. **The control:** `candidate-55-overlay-unset` the same as `candidate-55` on the grid.
6. **Outputs:** the temperature-0 gate on 2 boots gives candidate-54's TP4 hashes 18d61961/d6568649.

The marks at `tp4-switched` are the same, though its 24-row loss was smaller (0.8% with both builds capped), so mark 1
may not be met there; that would be a result, not a failure of the change.

## Workload

The engine grid step alone ([candidate-48/grid-suite.json](../2026-10-26-spark-ds41f-candidate/candidate-48/grid-suite.json),
20 decode shapes, 100 replays) and the determinism gate's workloads, as candidate-55's
[gate-caps-tp4.json](../2026-10-26-spark-ds41f-candidate/candidate-55/gate-caps-tp4.json) runs them.

## Commands

```bash
bin/spark candidate queue tp4-ring experiments/2026-11-13-l2-prefetch-row-bands/grid-tp4.json experiments/2026-11-13-l2-prefetch-row-bands/gate-tp4.json --apply
bin/spark candidate queue tp4-switched experiments/2026-11-13-l2-prefetch-row-bands/grid-tp4-switched.json experiments/2026-11-13-l2-prefetch-row-bands/gate-tp4-switched.json --apply
bin/spark candidate verdicts tp4-ring l2-prefetch-row-bands-grid-tp4-r2
bin/spark candidate gate tp4-ring l2-prefetch-row-bands-gate-tp4-r2 --expect 18d61961/d6568649
```

## Checks before the runs

- The patch's tests (`tests/models/test_deepseek_v4_1_l2_schedule.py`, 17 cases, 10 of them new) pass in the
  candidate-55 image on sp8, run with `pytest --noconftest` because the image has no `tblib`.
- `bin/spark candidate lint` is clean, `bin/spark doctor` passes on each profile, and the dry runs plan 6 boots a
  grid job and 2 a gate job.

## Results

**First run void (2026-11-13 07:59Z to 08:30Z).** The profiles named the overlay under `container.extra_mounts`, which
no command reads, so both grid jobs (`l2-prefetch-row-bands-grid-tp4`, `…-grid-tp4-switched`) measured candidate-55
six times: the "band" arm's 24-row shapes were within 0.1% of `candidate-55` at both testbeds. The profiles now use
`container.mounts`, and the grid runs again as `…-grid-tp4-r2` and `…-grid-tp4-switched-r2`. Both gate jobs
(`l2-prefetch-row-bands-gate-tp4`, `…-gate-tp4-switched`) started at `37630b335`, before the fix, so they gated
candidate-55 itself (both pass with candidate-54's hashes, which says nothing about the band); they run again as
`…-gate-tp4-r2` and `…-gate-tp4-switched-r2`.

On one GPU of sp8 the overlay was mounted with `docker -v` and the band acts: with `17-32=10` the forward is 2.051 and
2.061 ms at 8x24 against 2.018 and 2.025 ms unset, and within 1% of unset at 8x16 and 16x48. (One GPU has no all-reduce for the
fill to hide behind, so the sign there says nothing about TP4.)

### The grid with the overlay mounted (2026-11-13 09:37Z to 10:13Z)

Runs `l2-prefetch-row-bands-grid-tp4-r2` (`tp4-ring`, commit `50748ee6b`) and `l2-prefetch-row-bands-grid-tp4-switched-r2`
(`tp4-switched`, commit `19ff11ab2`), three arms, two boots each, every boot warm. The live container of a band boot was
inspected at each testbed's first run: it mounts the two overlay files and carries `VLLM_DS41_L2_PREFETCH_WO_BANDS=17-32=10`.

Step time, mean of both boots of the arm against the mean of both boots of `candidate-55` in the same job; negative is
faster. Forward is the target's forward alone.

| shape | `tp4-ring` step | forward (ms) | `tp4-switched` step | forward (ms) |
| --- | ---: | ---: | ---: | ---: |
| 4x24 @4K | -2.72% | -0.551 | -2.31% | -0.456 |
| 8x24 @4K | -2.90% | -0.575 | -2.13% | -0.462 |
| 8x24 @64K | -2.71% | -0.543 | -1.87% | -0.391 |
| 8x24 @256K | -2.49% | -0.544 | -2.61% | -0.518 |
| 16x32 @4K | -1.70% | -0.385 | -2.10% | -0.474 |
| 1x2 | -0.38% | -0.047 | +0.23% | +0.037 |
| 1x3 @4K, @64K, @256K, @1016K | -0.75%, +0.11%, +1.08%, -0.83% | | +0.67%, -0.11%, -0.20%, +0.28% | |
| 6 to 16 rows | +0.10% | | +0.19% | |
| 16x48 @4K, 8x48 @4K, 16x48 @64K | -0.16%, +0.29%, +0.07% | | -0.10%, +0.02%, -0.31% | |
| 12x72 | -0.17% | -0.060 | +0.09% | +0.066 |
| 16x96 | +0.10% | +0.073 | +0.10% | +0.046 |
| **grid, geometric mean (20)** | **-0.45%** ±0.15%, margin 0.40%: better | | **-0.52%** ±0.17%, margin 0.34%: better | |

The grid rows are the lab's verdicts (`bin/spark candidate verdicts`), the mean over both boots of each arm.

Against the marks:

1. **The band: met at both testbeds.** The four 24-row shapes are 2.70% faster as a group at `tp4-ring` and 2.23% at
   `tp4-switched`; 16x32 is 1.70% and 2.10% faster. All of it is the target's forward (0.39 to 0.58 ms a step).
2. **Outside the band: met.** The largest change at a shape outside the band is 1.07% (1x3 @256K, slower) at `tp4-ring`,
   where `candidate-55`'s own two boots differ by up to 2.35% at one shape, and 0.67% at `tp4-switched`, where they differ
   by up to 0.88%. 1x2, 1x3, 48, 72 and 96 rows are inside that spread at both.
3. **The grid: met.** Faster beyond the margin at both testbeds. At `tp4-ring` the two band boots differ by 0.61% and the
   first measured the same as `candidate-55`'s first (+0.01%), so the margin is beaten narrowly there (0.45% against 0.40%).
4. **The drafter: met.** Its time moves by at most 0.032 ms at a shape at `tp4-ring` and 0.053 ms at `tp4-switched`;
   `candidate-55`'s own boots differ by up to 0.079 and 0.064 ms.
5. **The control: met.** `candidate-55-overlay-unset` is +0.08% on the grid at `tp4-ring` and -0.06% at `tp4-switched`, and
   its 24-row shapes are within 0.3% of `candidate-55`.
6. **Outputs: not measured yet.** See the note on the first gate jobs above.

`tp4-switched` gains as much as the ring, though its 24-row shapes were only 0.8% behind candidate-54 with both builds
capped: the 15 MB WO window is too large for the 32-row tier at both fabrics, not only where it showed as a loss.

Receipts: [candidate-55/receipts/l2-prefetch-row-bands](../2026-10-26-spark-ds41f-candidate/candidate-55/receipts/l2-prefetch-row-bands).
