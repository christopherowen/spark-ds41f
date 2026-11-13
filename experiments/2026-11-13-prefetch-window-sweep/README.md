# L2 prefetch windows, swept on candidate-55's own kernels

Base commit: `3922346fa`. Image: `spark-ds41f:candidate-55`, fourth build, booted on a kernel cache its own CUDA
13.5.37 compiled (`TILELANG_CACHE_DIR` on a directory of its own, as
[the kernel cache experiment](../2026-11-13-kernel-cache-toolchain/README.md) does). Nothing is built.

## Why

- The recipe's windows (WO 15, FFN 12, NEXT 12 MB) and the row band `17-32=10`
  ([the row-band experiment](../2026-11-13-l2-prefetch-row-bands/README.md)) were all chosen while the image ran
  kernels an earlier CUDA 13.0.2 build had compiled. On its own kernels the grid at `tp4-ring` is 0.68% slower, at
  1x2, 1x3 and 48 rows.
- Each window was set at one point. The prefetch is worth 4.5% to 21% of a step at `tp4-ring`, and one WO size for
  every row count is wrong somewhere.
- `tp3-switched` has the same caps and has never had its own sweep.

## Arms

One variable an arm, one boot an arm, against the recipe on the same kernel cache (`own-ref`, booted first and last).
The suite is the engine grid step alone (`candidate-48/grid-suite.json` at TP4, `candidate-54/grid-tp3-suite.json` at
TP3).

| Arm | Variable | TP4 (both testbeds) | `tp3-switched` |
| --- | --- | --- | --- |
| `own-ref`, `own-ref-again` | none: WO 15, FFN 12, NEXT 12 | yes | yes |
| `own-wo8`, `own-wo10`, `own-wo12`, `own-wo18`, `own-wo20` | `VLLM_DS41_L2_PREFETCH_WO_MB` | all five | 10, 12, 20 |
| `own-ffn9`, `own-ffn15` | `VLLM_DS41_L2_PREFETCH_FFN_MB` | yes | yes |
| `own-next9`, `own-next15` | `VLLM_DS41_L2_PREFETCH_NEXT_MB` | yes | yes |

One boot an arm, because the question is a choice among arms and the confirmation below boots the choice twice. Over
identical candidate-55 boots today (24 at `tp4-ring`, 10 at `tp4-switched`) the grid's geometric mean has a boot SD of
0.15%, and a shape's 0.3% to 1.2% at the ring and 0.14% to 0.72% switched.

At `tp3-switched` the first boot compiles every kernel into the new cache directory (about 18 minutes a node at
`tp4-ring`).

## How the recipe is chosen from the sweep, declared before the runs

The WO budget is global in each arm, so every arm gives each shape's target forward and drafter time at that budget.
The row-band patch then lets a band take the budget that suits it.

1. **Bands:** 1-3, 4-16, 17-32, 33-64 and 65-96 rows. The target's forward of a shape is in the band of its row
   count; the drafter's forward is in the band of its request count (1 to 16).
2. **A band's budget** is the WO value with the smallest summed time over what the band covers (the target forwards
   of its shapes, and the drafter times of the shapes whose request count is in it). It replaces 15 MB only when it
   beats `own-ref` and `own-ref-again` both, by more than those two differ on the same sum. Otherwise 15 MB stays.
3. **FFN and NEXT** have no bands. One changes only when its arm's grid geometric mean beats both reference boots by
   more than 0.42% (twice the 0.15% boot SD, for a difference of two boots) and no band's sum is slower than both.
4. The composed recipe is a prediction. It is adopted only if the confirmation meets its marks.

## Confirmation, marks declared before the runs

The composed recipe (the row-band overlay mounted, `VLLM_DS41_L2_PREFETCH_WO_BANDS` and any FFN or NEXT change) on the
same kernel cache, two boots at each testbed, against the two `own-ref` boots of the sweep:

1. **Grid:** the geometric mean faster than the reference beyond the margin (0.30%: two boots against two, 0.15% a boot).
2. **No band worse:** each band's group of shapes the same or faster within the reference's own spread.
3. **Outputs:** the temperature-0 gate on both boots gives candidate-54's hashes for the shape.
4. **Serving, `tp4-ring`:** in the `screen` suite, single-stream and c16 decode not slower than the recorded candidate-55
   screen with the caps, beyond the tool's margin. (Narrowed before any result, 2026-11-13 11:35Z: a lab job runs a suite
   or the gate's workloads on a boot, not both, so the screen's two boots are run at `tp4-ring`, where the targets are
   set; `tp4-switched` and `tp3-switched` confirm on the grid suite and the gate. Their screens run on the fifth build.)

If mark 1 or 2 fails, the recipe falls back to the bands that individually held, and that is recorded.

## Commands

```bash
bin/spark candidate queue tp4-ring experiments/2026-11-13-prefetch-window-sweep/sweep-tp4.json --apply
bin/spark candidate queue tp4-switched experiments/2026-11-13-prefetch-window-sweep/sweep-tp4-switched.json --apply
bin/spark candidate queue tp3-switched experiments/2026-11-13-prefetch-window-sweep/sweep-tp3-switched.json --apply
```

## Results

None yet.
