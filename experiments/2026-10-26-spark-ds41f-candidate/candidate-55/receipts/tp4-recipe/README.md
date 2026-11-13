# candidate-55 with the TP4 recipe, fourth build, 2026-11-12

The fourth build's image (vLLM tree `8aaae3df`) with the two recipe variables of
[TP4's recipe](../../README.md#tp4s-recipe-l2-prefetch-caps-and-the-rings-collective-limit): the L2 prefetch caps at
both TP4 testbeds, and the collective limit at `tp4-ring`. The image and the locks are unchanged. The fourth build's
own gates and screens, taken before the recipe, are not in this folder.

## Jobs

| Testbed | Job | Spec | Window (UTC) | Deployment |
| --- | --- | --- | --- | --- |
| `tp4-switched` | gate, 2 boots | [gate-caps-tp4-switched.json](../../gate-caps-tp4-switched.json) | 20:42-21:09Z | `d8a24bf9` |
| | screen, 2 boots | [screen-caps-tp4-switched.json](../../screen-caps-tp4-switched.json) | 21:09-21:38Z | `d8a24bf9` |
| | c1 decision check, 4 boots | [c1-decisions-caps-tp4-switched.json](../../c1-decisions-caps-tp4-switched.json) | 21:41-22:03Z | `b1426138` |
| `tp4-ring` | gate, 2 boots | [gate-caps-tp4.json](../../gate-caps-tp4.json) | 21:38-22:04Z | `b0fce00e` |
| | c1 decision check, 4 boots | [c1-decisions-caps-tp4.json](../../c1-decisions-caps-tp4.json) | 22:04-22:28Z | `b1426138` |
| | screen, 2 boots | [screen-caps-tp4.json](../../screen-caps-tp4.json) | 22:27-22:56Z | `b1426138` |

All 16 boots reached ready and no memory guard fired; no boot compiled a kernel. One `tp4-ring` boot (c1 check, boot
3) was ready at +171 s where the others took 112 to 119 s, the slow pre-launch the third build's record describes.

## Gates: candidate-54's hashes

| Testbed | Boots | Runs identical | Hashes (logprobs; text) | c16 / c8 / c4 distinct streams, tok/s |
| --- | --- | --- | --- | --- |
| `tp4-switched` | 2 | 8 of 8 | 18d61961/d6568649; 0a2dacdd/12fd8257 | 269.69, 269.58 / 199.17, 198.96 / 154.58, 154.97 |
| `tp4-ring` | 2 | 8 of 8 | 18d61961/d6568649; 0a2dacdd/12fd8257 | 272.50, 270.84 / 201.33, 201.38 / 156.40, 155.45 |

The caps and the collective limit change no temperature-0 output.

## c1 decision checks

[c1_decisions.py](../../../../2026-11-10-tp4-ring-determinism/boot-differences/c1_decisions.py) and
[c1_paired_decode.py](../tp4-ring/boot-differences/c1_paired_decode.py). The fourth build's c1 boots without the
recipe are the same day's (`candidate-55-c1-decisions-tp4-b4`, `…-tp4-switched-b4`, on the heads); candidate-54's are
its [final](../../../candidate-54/receipts/final) four, cross-session.

| Testbed | Decisions | Decode tok/s, per boot | Against the fourth build without the recipe | Against candidate-54 | servebench output tok/s |
| --- | --- | --- | --- | --- | --- |
| `tp4-ring` | 4 of 4 boots agree (59 requests); all 28 pairs with candidate-54's four agree on the 58 both sent | 86.07-86.33 | **-0.13% ± 0.14%** decode time (+0.16% tok/s): the same | **-1.62% ± 0.15%** (+1.69% tok/s) | 82.68-82.92 |
| `tp4-switched` | 4 of 4 agree (59 requests); all 28 pairs with candidate-54's agree on the 57 both sent | 86.17-86.29 | **-0.32% ± 0.17%** (+0.35% tok/s) | **-2.84% ± 0.13%** (+2.95% tok/s) | 82.67-82.82 |

**The caps make the 1x2 and 1x3 grid rows 5% to 18% faster and a single stream's decode 0.2% to 0.4% faster.** The
grid replays one shape back to back; a single stream's serving steps gain far less from the same change.

## Screens against candidate-54's recorded screens

`tp4-switched`, the mean over both boots of each
([table](tp4-switched/lab/candidate-55-screen-caps-tp4-switched-b4-table.txt)):

| Metric | With the recipe | Verdict | The fourth build without it |
| --- | --- | --- | --- |
| engine grid, geometric mean | -3.11% (faster), margin 0.33% | better | +1.01%, worse |
| c16 paired decode | +1.45%, margin 0.46% | better | +1.59%, better |
| c16 output | +1.47% (268.12 / 268.93 tok/s), margin 0.98% | better | +1.72% (269.03 / 269.33), better |
| prefill 32K | +0.51% (8,649 / 8,764 tok/s), margin 2.27% | same | +0.88%, same |
| prefill 256K | -0.01% (8,719 / 8,782 tok/s), margin 1.25% | same | +0.56%, same |

- **No grid shape is slower than candidate-54's.** 1x2 rows are 9.9% faster, 1x3 rows 7.2%, the 4- to 16-row shapes
  4% to 5%; the fourth build without the caps had 1x2 rows 11.4% slower and 1x3 rows 6.6% slower.
- c16 and prefill are the fourth build's within the intervals: the caps do not reach them.

`tp4-ring`, the mean over both boots of each ([table](tp4-ring/lab/candidate-55-screen-caps-tp4-b4-table.txt)):

| Metric | With the recipe | Verdict | The fourth build without it |
| --- | --- | --- | --- |
| engine grid, geometric mean | -1.55% (faster), margin 0.30% | better | +0.25%, same |
| c16 paired decode | +0.47%, margin 0.66% | same | +0.78%, better |
| c16 output | +0.45% (271.00 / 271.18 tok/s), margin 0.93% | same | +0.83% (272.09 / 272.15), same |
| prefill 32K | +0.78% (8,784 / 8,752 tok/s), margin 2.61% | same | -1.81% (8,530 / 8,557), same |
| prefill 256K | -0.04% (8,854 / 8,847 tok/s), margin 0.70% | same | -3.05% (8,577 / 8,591), **worse** |

- **Prefill at 262,144 tokens is candidate-54's again**, with the collective limit: 3.1% above the fourth build
  without it, the same day.
- **1x2 rows are 5.1% faster than candidate-54's** (13.74 ms against 14.47); without the caps they were 11.6% slower.
- **Four shapes of 24 and 32 rows are still slower than candidate-54's on both boots:** 16x32 by 1.6% and 2.0%, 8x24
  by 1.3% and 0.7%, 4x24 by 0.8% and 2.1%, and 8x24 at 256K context by 1.2% and 2.4%. Without the caps they were 1.3%
  to 3.3% slower. The cause is not attributed: at `tp4-switched` the arms without #982, without #989's proxy
  pinning and without the thread pinning left these rows unchanged, and the third build's control had them too.
  c16, whose steps average 44 rows, is not slower.
- c16 is 0.4% below the fourth build without the recipe, inside the interval of either.

## Marks

| Mark | `tp4-ring` | `tp4-switched` |
| --- | --- | --- |
| The gate on 2 boots gives candidate-54's TP4 hashes | **Met** | **Met** |
| The screen's grid is not slower than candidate-54's recorded screen | **Met:** 1.55% faster; four shapes of 24 and 32 rows 1% to 2% slower | **Met:** 3.11% faster |
| c16 and prefill not worse than the fourth build's screen without the recipe | **Met:** c16 the same within the intervals, prefill 2.6% and 3.1% faster | **Met:** the same within the intervals |
| Prefill at 262,144 tokens not slower than candidate-54's recorded screen | **Met:** the same (-0.04%, margin 0.70%) | not declared (it never regressed there); the same |
| The c1 check's boots agree with each other and with candidate-54's | **Met** | **Met** |

## Files

- `<testbed>/lab/`: each job's table, window-log segment and overlay list (none).
- `<testbed>/determinism/<run>/`: each gate run's summary and telemetry, each distinct-stream run, and the c1 runs.
- `<testbed>/suites/<run>/`: the screens' suite runs. Each `model.json.gz` stays on the head, listed in
  `<testbed>/large-files-on-nodes.tsv`.
- `<testbed>/boot-differences/<run>/`: each c1 boot's `c1.json`.
