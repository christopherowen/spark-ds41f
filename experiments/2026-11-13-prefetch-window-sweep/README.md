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

Added 2026-11-13 12:52Z, before `tp4-switched`'s confirmation ran: its sweep picks `65-96=12` on one boot's 0.38%,
because its two reference boots happen to agree to 0.13% there. That band is kept only if its shapes (12x72, 16x96) are
faster than both reference boots on both confirmation boots; otherwise the recipe is `17-32=10` alone, as at `tp4-ring`.

### Added after the ring's confirmation screen (2026-11-13 12:55Z)

- The `screen` suite has no single-stream step, so mark 4's single-stream half runs as its own job:
  [confirm-c1-tp4.json](confirm-c1-tp4.json), two boots of the composed recipe with candidate-55's c1 workload
  (`c1-decisions-caps-tp4.json`), against its recorded 86.07 to 86.33 tok/s over four boots.
- The composed recipe's screen has 256K prefill 1.13% slower than candidate-54's recorded screen (8,761 and 8,748
  tok/s), where candidate-55 with the caps on the earlier kernels had 8,854 and 8,847. The row band cannot act on a
  prefill forward, so [screen-own-kernels-tp4.json](screen-own-kernels-tp4.json) runs the same screen on the own
  kernels without the overlay, two boots, to say whether the kernels are the cause.

## Commands

```bash
bin/spark candidate queue tp4-ring experiments/2026-11-13-prefetch-window-sweep/sweep-tp4.json --apply
bin/spark candidate queue tp4-switched experiments/2026-11-13-prefetch-window-sweep/sweep-tp4-switched.json --apply
bin/spark candidate queue tp3-switched experiments/2026-11-13-prefetch-window-sweep/sweep-tp3-switched.json --apply
```

## Results

Sweeps: `tp4-ring` 11:32Z to 12:15Z (deployment `c49275ac0`), `tp4-switched` 12:03Z to 12:50Z, `tp3-switched` 12:24Z to
13:19Z (its first boot compiled the kernels: ready after 1,369 s). Every boot reached ready. The two TP4 testbeds
waited 35 and 55 minutes for another session's jobs; `tp3-switched`'s confirmation was queued at 16:15Z, after this
session lost its connection to the testbeds from 13:20Z to 16:12Z.

### The sweeps: the recipe's windows hold on the own kernels

`compose.py` on each run's receipts. A cell is the band's summed time against the mean of the two reference boots.

`tp4-ring`:

```
prefetch-window-sweep-tp4: 11 arms, 20 shapes; reference boots differ by -0.03% on the grid

WO: band sum (target forwards of the band's shapes + drafter times of the shapes whose requests are in it), ms
band           ref ref-again |      wo8     wo10     wo12     wo18     wo20 | chosen
1-3         68.702    68.749 |   +3.07%   +3.01%   +0.87%   +0.37%   +2.23% | 15 MB (no value beats both reference boots by their spread)
4-16       112.260   111.878 |   +2.27%   +1.97%   +0.84%   -0.25%   +0.18% | 15 MB (no value beats both reference boots by their spread)
17-32       81.526    81.796 |   -1.68%   -2.44%   -1.25%   +0.47%   +1.00% | 10 MB
33-64       65.094    64.916 |   +0.82%   +0.05%   +0.60%   +2.13%   +2.07% | 15 MB (no value beats both reference boots by their spread)
65-96       62.881    63.234 |   +1.51%   +0.89%   +0.38%   +0.33%   +1.05% | 15 MB (no value beats both reference boots by their spread)

FFN and NEXT: grid geometric mean against the reference boots
  ffn9   -0.12% against the mean reference; bands slower than both reference boots: 1-3, 65-96
  ffn15  -0.09% against the mean reference; bands slower than both reference boots: 17-32
  next9   +0.87% against the mean reference; bands slower than both reference boots: 1-3, 4-16, 17-32, 33-64
  next15  -0.26% against the mean reference; bands slower than both reference boots: none

VLLM_DS41_L2_PREFETCH_WO_BANDS=17-32=10
predicted grid change from the WO bands alone: -0.47%
```

`tp4-switched`:

```
prefetch-window-sweep-tp4-switched: 11 arms, 20 shapes; reference boots differ by +0.16% on the grid

WO: band sum (target forwards of the band's shapes + drafter times of the shapes whose requests are in it), ms
band           ref ref-again |      wo8     wo10     wo12     wo18     wo20 | chosen
1-3         67.268    67.334 |   +3.25%   +2.20%   +0.64%   +1.16%   +2.84% | 15 MB (no value beats both reference boots by their spread)
4-16       110.428   110.406 |   +2.11%   +1.14%   +0.71%   +0.09%   +0.50% | 15 MB (no value beats both reference boots by their spread)
17-32       79.951    80.603 |   -2.35%   -2.68%   -2.46%   -0.08%   +0.04% | 10 MB
33-64       63.348    63.206 |   +1.23%   +1.00%   -0.11%   +1.82%   +1.80% | 15 MB (no value beats both reference boots by their spread)
65-96       62.712    62.795 |   +1.88%   +0.74%   -0.38%   +0.62%   +0.83% | 12 MB

FFN and NEXT: grid geometric mean against the reference boots
  ffn9   -0.19% against the mean reference; bands slower than both reference boots: 4-16, 65-96
  ffn15  -0.21% against the mean reference; bands slower than both reference boots: 17-32
  next9   +0.67% against the mean reference; bands slower than both reference boots: 1-3, 4-16, 17-32, 33-64
  next15  -0.49% against the mean reference; bands slower than both reference boots: 65-96

VLLM_DS41_L2_PREFETCH_WO_BANDS=17-32=10,65-96=12
predicted grid change from the WO bands alone: -0.54%
```

`tp3-switched` (its grid stops at 8 requests, so it has no 65-96 band):

```
prefetch-window-sweep-tp3-switched: 9 arms, 15 shapes; reference boots differ by +0.05% on the grid

WO: band sum (target forwards of the band's shapes + drafter times of the shapes whose requests are in it), ms
band           ref ref-again |     wo10     wo12     wo20 | chosen
1-3         90.331    90.132 |   +0.88%   +0.38%   +1.84% | 15 MB (no value beats both reference boots by their spread)
4-16       105.419   106.214 |   -0.17%   +0.49%   +2.05% | 15 MB (no value beats both reference boots by their spread)
17-32       72.913    72.580 |   -0.65%   -1.18%   +3.26% | 12 MB
33-64       22.952    23.004 |   +0.69%   +1.75%   +1.44% | 15 MB (no value beats both reference boots by their spread)

FFN and NEXT: grid geometric mean against the reference boots
  ffn9   +0.29% against the mean reference; bands slower than both reference boots: 4-16, 17-32, 33-64
  ffn15  -0.27% against the mean reference; bands slower than both reference boots: none
  next9   +0.83% against the mean reference; bands slower than both reference boots: 1-3, 4-16, 17-32, 33-64
  next15  -0.01% against the mean reference; bands slower than both reference boots: 1-3, 33-64

VLLM_DS41_L2_PREFETCH_WO_BANDS=17-32=12
predicted grid change from the WO bands alone: -0.25%
```

- **WO 15 MB is right outside one band** at all three testbeds: at 1 to 16 rows every other size is slower (0.4% to
  3.3% at TP4), and at 33 to 96 rows no size beats the reference boots by their spread, but for `tp4-switched`'s
  65-96 at 12 MB on one boot (below).
- **17 to 32 rows want a smaller window on the own kernels too:** 10 MB at TP4 (2.4% and 2.7% faster in the band), 12 MB
  at `tp3-switched` (1.2%; 10 MB gives 0.65%).
- **FFN and NEXT stay.** No arm beats both reference boots by 0.42% with no band slower. NEXT 9 is 0.7% to 0.9%
  slower at all three. NEXT 15 reads 0.26% and 0.49% faster at the two TP4 testbeds and the same at TP3: under the
  rule's margin, and a lead for a later look, not a change.

### Confirmation at `tp4-ring`: `17-32=10`

Jobs `confirm-screen-tp4` (12:16Z to 12:45Z), `confirm-gate-tp4` (to 13:11Z), `confirm-c1-tp4` (to 13:23Z) and
`screen-own-kernels-tp4` (to 13:52Z), two boots each; the live container of the first composed boot was inspected (the
two overlay files, the band, the own kernel cache).

| | Own kernels, no band | Composed (`17-32=10`) | Recorded, candidate-55 with the caps on the earlier kernels |
| --- | --- | --- | --- |
| Grid against candidate-54's recorded screen | -1.78%, better | -1.96%, better | -1.55%, better |
| Grid geometric mean, each boot (ms) | 19.873, 19.875 | 19.818, 19.761 | |
| c16 decode against candidate-54 | +0.35%, same | +0.22%, same | +0.47%, same |
| Prefill 32K against candidate-54 | -0.16%, same | +0.05%, same | +0.78%, same |
| Prefill 256K against candidate-54 | -0.72%, same (margin 1.12%) | **-1.13%, worse** (margin 0.42%) | -0.04%, same |
| Prefill 256K, each boot (tok/s) | 8,801, 8,780 | 8,761, 8,748 | 8,854, 8,847 |
| Single stream, servebench output tok/s, each boot | | 82.33, 82.78 | 82.68 to 82.92 over four boots |

1. **Grid: met.** 0.43% faster than the own-kernels screen's two boots and 0.39% faster than the sweep's two reference
   boots; the margin is 0.30%.
2. **No band worse: not cleanly met.** 17 to 32 rows are 1.7% to 2.3% faster. But 12x72 is 0.98% slower and 16x96
   0.68% slower than the own-kernels screen (0.81% and 0.38% against the sweep's reference boots), about twice the
   boot SD of those shapes for two boots against two. Both comparisons use the same two composed boots. It did not show
   at `tp4-switched` (-0.23%, +0.02%), nor at `tp4-ring` on the earlier kernels (-0.17%, +0.10%). Open: the fifth
   build's screen gives two more boots.
3. **Outputs: met.** Both boots identical, 18d61961/d6568649.
4. **Serving: met as declared, with a finding outside the mark.** c16 is the same and single stream is 82.33 and 82.78
   tok/s against 82.68 to 82.92 (each boot's figure is a mean of ten windows with a 5.7% interval, so this says "not
   visibly slower", no more). **256K prefill is lower on the own kernels:** 8,801 and 8,780 tok/s without the band and
   8,761 and 8,748 with it, against 8,854 and 8,847 on the earlier kernels. The first step (0.7%) is inside the screen's
   margin and the second (0.4%) is no more than the boots' spread, but together the composed recipe is 1.13% behind
   candidate-54, beyond the margin. A row band cannot select a prefill forward's plan, so the band's share is not
   explained; the kernels' share is the CUDA 13.5 compile.

The own-kernels screen also says the grid costs nothing on the own kernels: 1.78% ahead of candidate-54, where the
earlier kernels' recorded screen has 1.55%. The 0.68% the kernel cache experiment measured in one job does not show
across these sessions.

### Confirmation at `tp4-switched`: `17-32=10,65-96=12`, of which `17-32=10` is kept

Jobs `confirm-grid-tp4-switched` (12:52Z to 13:00Z) and `confirm-gate-tp4-switched` (to 13:27Z), two boots each, the
live container inspected.

1. **Grid: met.** 19.476 and 19.459 ms against the reference boots' 19.573 and 19.605: 0.62% faster.
2. **No band worse: met.** 17 to 32 rows 1.8% to 2.7% faster; nothing else beyond the reference's spread.
3. **Outputs: met.** Both boots identical, 18d61961/d6568649, with both bands set.
- **`65-96=12` is dropped,** by the condition written before the run: 12x72 is faster than both reference boots on both
  boots (34.723 and 34.711 ms against 34.749 and 34.846), 16x96 is not (38.787 and 38.782 against 38.704 and 38.852).
  The recipe is `17-32=10`, as at `tp4-ring`. That exact recipe on the own kernels has no gate of its own at
  `tp4-switched`; the fifth build's gate is it.

### `tp3-switched`: `17-32=12`, confirmation running

Queued 16:15Z (`confirm-grid-tp3-switched`, `confirm-gate-tp3-switched`). Results to follow here.

## Conclusion so far

- The TP4 recipe is unchanged by the sweep: WO 15, FFN 12, NEXT 12 MB and `VLLM_DS41_L2_PREFETCH_WO_BANDS=17-32=10`,
  now measured on the kernels the build compiles. The windows have no more to give at TP4.
- A one-boot pick under about 0.5% does not survive two boots (`65-96=12`); the 2% ones do.
- Two things to settle on the fifth build: 256K prefill at `tp4-ring` on CUDA 13.5 kernels, and 72 to 96 rows with
  the band there.

Receipts: [candidate-55/receipts/prefetch-window-sweep](../2026-10-26-spark-ds41f-candidate/candidate-55/receipts/prefetch-window-sweep).
