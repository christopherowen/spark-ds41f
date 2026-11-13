# candidate-55: attributing and fixing the TP4 engine-grid regression

candidate-55's screens are slower than candidate-54 on the engine grid at both TP4 testbeds (third build: 0.67% at
`tp4-ring`, 1.11% at `tp4-switched`; fourth build 1.01% at `tp4-switched`) and faster at `tp3-switched`. This folder
finds which change causes it and what removes it.

- **Attribution run:** base commit `aa31800e5`, third build (`spark-ds41f:candidate-55`, `ad905a6d`, vLLM tree
  `e8c20f54`), against `spark-ds41f:candidate-54` (`d502217e`).
- **Confirmation run:** base commit `6c3e98b56`, fourth build (vLLM tree `8aaae3df`; the sparknet tree is the same,
  `e06f6a06`).
- **Workload:** the release suite's engine grid step alone
  ([candidate-48/grid-suite.json](../../candidate-48/grid-suite.json), 20 decode shapes, 100 replays, suite
  `8aadf78fff4d`), at `tp4-switched` (sp1..sp4). One boot an arm in the attribution run, two in the confirmation run.

## Conclusion

1. **The regression is in the 1-request rows.** At `tp4-switched`, 1x2 rows are 9% slower and 1x3 rows 8% slower
   than candidate-54, at every context length. Rows of 6 to 16 are 3% faster. Rows of 24 are 2% slower and wider
   rows 1% slower.
2. **#982's trigger position causes the 1-request loss.** Its `T.pdl_trigger()` sits after the doorbell and before
   `roce_wait_peers`. Without sparknet `0003` (`no982`), or with the trigger moved after the wait (`pdlpostwait`),
   1x3 returns to candidate-54's time and 1x2 to within 3% of it. Neither changes any other shape.
3. **TP4's L2 prefetch windows are too large, in candidate-54 too.** With the windows at 15/12/12 MB (TP3's caps;
   the defaults are 20/16/20), candidate-54's 1x2 rows are 9% faster than without, and candidate-55's are 18%
   faster. Under the caps the trigger position no longer matters: the caps with and without the moved trigger give
   the same grid (-4.60% and -4.66% against candidate-55 as built, two boots each).
4. **With the caps, candidate-55 is faster than candidate-54 with the same caps** by 1.7% on the grid: 3% to 4% at
   1x2 and 1x3, 3% at 6 to 16 rows, 0.7% at 32 rows and more. At 24 rows it is 0.8% slower, at the edge of the
   interval.
5. **Not causes:** the unpinned RoCE proxy (`no989`, 0.65% slower than candidate-55) and the stream priorities
   (`streamprio`, +0.34%) are within or beyond the repeat boot (+0.26%) in the wrong direction. Hypotheses 2 and 3
   below, as written for 24 rows and more, are not supported: the moved trigger and the priorities leave those rows
   unchanged.
   - **Corrected 2026-11-13:** two arms did not isolate what their names say. `no989` sets
     `SPARKNET_ROCE_PROXY_CPU=none`, which unpins the proxy (sparknet `0001`, #1134); #989's kernels stay.
     `noaffinity` sets `VLLM_SPARK_THREAD_AFFINITY=0`, which the image does not read (the core partition's switch is
     `VLLM_ROCE_CPU_AFFINITY`), so it is a third boot of candidate-55 (+0.42%). #989's kernels and #1134's core
     partition are isolated in [rows24-attribution](../rows24-attribution/README.md).

6. **`tp4-ring` agrees.** There the caps make candidate-55's grid 2.51% faster (margin 0.34%, two boots each): 1x2
   rows 16.2 to 13.5 ms and 1x3 rows 14.8 to 14.0 ms, against candidate-54's recorded 14.5 ms for both.

**The fix is a recipe variable:** candidate-55's TP4 profiles cap the windows at 15/12/12 MB, as candidate-54's TP3
profiles do. The image and the locks are unchanged, and sparknet `0003` stays as #982 placed it. The moved trigger is
not taken: it gains nothing measurable once the windows are capped.

**Not covered:** TP5 to TP8 keep the default windows and were not measured. The single-GPU screen below is slower at
24 rows and more for candidate-55 with both builds capped, which the TP4 grid shows only at 24 rows.

## Hypotheses, as written before the attribution run

1. **1x2 and 1x3 at TP4 (`l2cap`):** TP4 keeps the default 20/16/20 MB L2 prefetch budgets (about 741 chunks of
   64 KiB a layer, paced at 200 ns on `L2Prefetcher.side`, about 250 us a layer), whereas TP3 caps them at
   15/12/12 MB. candidate-55's main stream is faster at 2 and 3 rows (227-235 us a layer), so the side stream falls
   behind across the 40 layers, evicts live L2 lines with late prefetches and stalls `l2_prefetch.join()`.
   **Supported**, and the windows are already too large for candidate-54.
2. **24 rows and more, SM contention during the RoCE wait (`pdlpostwait`, `no982`):** #982 triggers dependents right
   after the doorbell, so their CTAs are resident during `roce_wait_peers` and delay the prefetch kernel.
   **Not supported for those rows.** The trigger position acts on 1x2 and 1x3 instead.
3. **24 rows and more, equal-priority stream inversion under CUDA 13.5 (`streamprio`):** the L2 side stream at
   priority -2, the main stream at -1, the aux stream at 0. **Not supported:** no shape changes.
4. **Combined (`all3`):** the caps, the moved trigger and the priorities. The same as the caps alone.
5. **Controls (`no989`, `noaffinity`, repeat boots):** no effect, and two of them not the controls intended; see
   the correction above.

## Arms

Each arm is the build's `tp4-switched-dev.json` with one change. The overlays mount files over the image; their
kernels compile into their own `TILELANG_CACHE_DIR`.

| Arm | Change |
| --- | --- |
| `l2cap` | `VLLM_DS41_L2_PREFETCH_WO_MB=15`, `_FFN_MB=12`, `_NEXT_MB=12` |
| `pdlpostwait` | [pdlpostwait/overlay](pdlpostwait/overlay): `T.pdl_trigger()` after `roce_wait_peers` in the four one-shot collectives |
| `no982` | [no982/overlay](no982/overlay): the four collectives without `0003`'s `pdl_sync` and `pdl_trigger` |
| `streamprio` | [streamprio/overlay](streamprio/overlay): stream priorities in `gpu_worker.py` and `l2_prefetch.py` |
| `all3` | [all3/overlay](all3/overlay): `l2cap`, `pdlpostwait` and `streamprio` together |
| `no989` | `SPARKNET_ROCE_PROXY_CPU=none`: the RoCE proxy unpinned, not #989's kernels |
| `noaffinity` | `VLLM_SPARK_THREAD_AFFINITY=0`: read by nothing in the image, so candidate-55 unchanged |
| `l2cap-pdlpostwait` (confirmation) | `l2cap` and `pdlpostwait` together |
| `candidate-54-l2cap` (confirmation) | candidate-54 with the same caps: do the caps act on candidate-55's changes or on any build? |

The third build's arm profiles are `tp4-switched-<arm>-dev.json`, on `tp4-switched-b3-dev.json` and this folder's
lock of that build. The fourth build's are `tp4-switched-b4-<arm>-dev.json`, on
[tp4-switched-b4-dev.json](tp4-switched-b4-dev.json): candidate-55's `tp4-switched-dev.json` with the default
windows, as it stood at `6c3e98b56` when the run was taken, before the caps entered it. The `tp4-ring` arms are on
[tp4-b4-dev.json](tp4-b4-dev.json), the same for `tp4-dev.json`.

## Commands

```bash
bin/spark candidate queue tp4-switched experiments/2026-10-26-spark-ds41f-candidate/candidate-55/grid-attribution/attribute-grid-tp4-switched.json --apply
bin/spark candidate queue tp4-switched experiments/2026-10-26-spark-ds41f-candidate/candidate-55/grid-attribution/confirm-fix-grid-tp4-switched.json --apply
bin/spark candidate verdicts tp4-switched candidate-55-confirm-fix-grid-tp4-switched
bin/spark candidate queue tp4-ring experiments/2026-10-26-spark-ds41f-candidate/candidate-55/grid-attribution/confirm-fix-grid-tp4.json --apply
```

## Attribution run: third build, one boot an arm

Job `20261112T145553-attribute-grid-tp4-switched.json`, 2026-11-12 14:56Z to 16:21Z, deployment `998b21a0`. Step
time in ms, the mean of 100 replays; the last column is the tool's geometric mean over the 20 shapes against
`candidate-55`
([table](../receipts/grid-attribution/tp4-switched/lab/candidate-55-attribute-grid-tp4-switched-table.txt)).

| Arm | 1x2 | 1x3 | 1x6 | 4x8 | 8x16 | 8x24 | 16x32 | 12x72 | 16x96 | 1x3 at 1016K | Grid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `candidate-54` | 15.16 | 14.41 | 15.50 | 16.55 | 18.94 | 20.59 | 22.72 | 34.72 | 38.49 | 14.46 | -2.09% |
| `candidate-55` | 16.64 | 15.86 | 14.86 | 15.92 | 18.55 | 21.32 | 22.92 | 35.63 | 38.96 | 15.85 | |
| `candidate-55-l2cap` | 13.42 | 13.55 | 14.96 | 15.72 | 18.11 | 20.59 | 22.72 | 34.68 | 38.67 | 13.70 | -4.78% |
| `candidate-55-pdlpostwait` | 15.67 | 14.27 | 14.91 | 16.11 | 18.57 | 21.33 | 23.14 | 35.40 | 38.83 | 14.29 | -1.96% |
| `candidate-55-streamprio` | 16.59 | 15.98 | 14.95 | 15.88 | 18.50 | 21.40 | 22.97 | 35.62 | 38.98 | 15.94 | +0.34% |
| `candidate-55-all3` | 13.52 | 13.60 | 14.99 | 15.86 | 18.38 | 20.75 | 22.81 | 34.72 | 38.81 | 13.71 | -4.61% |
| `candidate-55-no982` | 15.63 | 14.19 | 14.82 | 15.82 | 18.47 | 21.18 | 22.92 | 35.56 | 39.07 | 14.44 | -2.26% |
| `candidate-55-no989` | 16.62 | 16.18 | 14.97 | 15.99 | 18.51 | 21.46 | 23.11 | 35.55 | 38.90 | 16.22 | +0.65% |
| `candidate-55-noaffinity` | 16.59 | 15.94 | 15.03 | 15.99 | 18.55 | 21.33 | 23.04 | 35.55 | 39.01 | 15.97 | +0.42% |
| `candidate-55-again` | 16.57 | 15.67 | 15.01 | 15.96 | 18.49 | 21.43 | 23.06 | 35.58 | 38.93 | 15.67 | +0.26% |
| `candidate-54-again` | 15.30 | 14.48 | 15.51 | 16.35 | 18.69 | 20.96 | 22.69 | 34.89 | 38.87 | 14.46 | -1.38% |

- candidate-54's two boots differ by 0.72% on the grid, and by 1.8% and 2.4% at 8x24 and 4x24.
- Every boot reached ready and no memory guard fired. The `pdlpostwait` and `no982` arms compiled their kernels on
  the first boot (24 minutes each); the other boots compiled none.
- The job restarted once at its start, when its sync moved the checkout.

## Confirmation run: fourth build, two boots an arm

Job `20261112T194852-confirm-fix-grid-tp4-switched.json`, 2026-11-12 19:49Z to 20:30Z, deployment `6c3e98b5`
([table](../receipts/grid-attribution/tp4-switched/lab/candidate-55-confirm-fix-grid-tp4-switched-table.txt)).

| Arm | 1x2 | 1x3 | 1x6 | 4x8 | 8x16 | 8x24 | 16x32 | 12x72 | 16x96 | 1x3 at 1016K | Grid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `candidate-55` | 16.59 | 15.83 | 14.90 | 15.94 | 18.53 | 21.33 | 23.06 | 35.70 | 39.06 | 15.91 | |
| `candidate-55-again` | 16.57 | 15.81 | 14.93 | 15.97 | 18.54 | 21.43 | 22.94 | 35.48 | 38.91 | 15.65 | -0.43% |
| `candidate-55-pdlpostwait` | 15.68 | 14.02 | 14.78 | 15.77 | 18.35 | 21.14 | 22.75 | 35.55 | 39.05 | 14.29 | -2.78% |
| `candidate-55-pdlpostwait-again` | 15.53 | 14.00 | 14.85 | 16.15 | 18.57 | 21.30 | 22.99 | 35.36 | 38.84 | 13.93 | -2.73% |
| `candidate-55-l2cap` | 13.37 | 13.66 | 15.01 | 16.06 | 18.39 | 20.79 | 22.89 | 34.71 | 38.66 | 13.89 | -4.58% |
| `candidate-55-l2cap-again` | 13.47 | 13.63 | 14.98 | 16.07 | 18.44 | 21.00 | 22.91 | 34.63 | 38.55 | 13.74 | -4.74% |
| `candidate-55-l2cap-pdlpostwait` | 13.35 | 13.63 | 15.02 | 15.97 | 18.36 | 21.02 | 22.90 | 34.68 | 38.78 | 13.68 | -4.73% |
| `candidate-55-l2cap-pdlpostwait-again` | 13.55 | 13.70 | 15.13 | 15.96 | 18.35 | 20.95 | 22.96 | 34.71 | 38.72 | 13.73 | -4.48% |
| `candidate-54-l2cap` | 13.83 | 14.24 | 15.60 | 16.58 | 18.79 | 20.86 | 22.69 | 34.68 | 38.85 | 14.21 | -2.82% |
| `candidate-54-l2cap-again` | 13.98 | 14.24 | 15.45 | 16.62 | 18.78 | 20.79 | 22.58 | 34.48 | 38.60 | 14.21 | -3.14% |

- **Verdict of the run** (`l2cap-pdlpostwait` against `candidate-55`, both boots of each): grid -4.40%, margin
  0.34%, better.
- **Repeat boots agree:** each arm's two boots are within 0.3% of each other on the grid.
- **The fourth build repeats the third:** `candidate-55`, `l2cap` and `pdlpostwait` give the same step times on
  both builds.
- **By shape group**, the mean of both boots against `candidate-55` (geometric means of the step times):

  | Arm | 1x2 | 1x3 | 1x3, long contexts | 6 to 16 rows | 24 rows | 32 rows and more | All 20 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | `candidate-55-pdlpostwait` | -5.9% | -10.8% | -8.9% | -0.1% | -0.4% | -0.5% | -2.5% |
  | `candidate-55-l2cap` | -17.9% | -12.7% | -11.5% | +0.0% | -1.8% | -1.7% | -4.3% |
  | `candidate-55-l2cap-pdlpostwait` | -18.1% | -12.3% | -11.5% | +0.2% | -2.0% | -1.4% | -4.2% |
  | `candidate-54-l2cap` | -15.2% | -9.1% | -7.7% | +2.8% | -2.6% | -1.0% | -2.6% |

- candidate-54 without the caps was not booted again; its two boots in the attribution run are the comparison
  (1x2 15.16 and 15.30 ms; with the caps 13.83 and 13.98 ms).
- Every boot reached ready and no memory guard fired.

## Check at tp4-ring: fourth build, two boots an arm

Job `20261112T204239-confirm-fix-grid-tp4.json`, 2026-11-12 21:17Z to 21:33Z, deployment `d8a24bf9`
([table](../receipts/grid-attribution/tp4-ring/lab/candidate-55-confirm-fix-grid-tp4-table.txt)). `tp4-ring` is the
testbed the targets are measured at, and its 1x3 rows had not regressed in the screens as `tp4-switched`'s did.

| Arm | 1x2 | 1x3 | 1x6 | 4x8 | 8x16 | 8x24 | 16x32 | 12x72 | 16x96 | 1x3 at 1016K | Grid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `candidate-55` | 16.23 | 14.71 | 15.11 | 15.87 | 18.29 | 21.41 | 22.96 | 35.64 | 38.76 | 14.69 | |
| `candidate-55-again` | 16.17 | 14.96 | 15.10 | 15.78 | 18.39 | 21.37 | 23.01 | 35.53 | 38.85 | 14.89 | +0.24% |
| `candidate-55-l2cap` | 13.57 | 13.94 | 15.13 | 16.03 | 18.11 | 21.04 | 22.93 | 35.02 | 38.31 | 13.87 | -2.54% |
| `candidate-55-l2cap-again` | 13.48 | 14.05 | 15.15 | 16.08 | 18.42 | 21.15 | 23.09 | 34.89 | 38.38 | 13.89 | -2.24% |

- **Verdict of the run:** grid -2.51%, margin 0.34%, better (both boots of each).
- **By shape group**, against `candidate-55`: 1x2 -15.6%, 1x3 -5.1% (long contexts the same), 6 to 16 rows +0.1%,
  24 rows -0.9%, 32 rows and more -1.0%.
- The gain is smaller than at `tp4-switched` (-4.66%) because the ring's 1x3 rows lost less to begin with.
- candidate-54 and the moved trigger were not measured at `tp4-ring`.

## Single-GPU screen on sp8, one boot an arm

A second line of work ran both images on one GPU of sp8 with dummy weights at a TP4 rank's shapes
(`tensor_parallel_size=1`, so no sparknet collective runs), 60 replays a shape, both builds with the 15/12/12 MB
caps ([receipts](../receipts/grid-attribution/sp8-single-gpu)). Forward time in ms:

| Arm | 1x2 | 1x3 | 8x16 | 8x24 | 16x32 | 16x48 | 12x72 | 16x96 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `c54_l2cap` | 1.684 | 1.696 | 1.792 | 1.936 | 2.031 | 2.281 | 2.887 | 3.091 |
| `c55_l2cap` | 1.688 | 1.684 | 1.802 | 2.035 | 2.136 | 2.351 | 3.026 | 3.187 |
| `c55_l2_141517` (14/15/17 MB) | 1.696 | 1.702 | 1.807 | 2.063 | 2.160 | 2.366 | 3.001 | 3.167 |

- **candidate-55's forward is 3% to 5% slower at 24 rows and more on one GPU**, with no collective in the step, so
  #982 and #989 cannot cause that part. In the CUPTI traces of the same boots the L2 prefetch kernel takes longer
  from 24 rows up (1,751 to 2,110 us a step at 8x24) and `gemm_mxfp8_kernel` runs alone for less of its time: the
  two overlap more.
- **It does not carry to TP4 as measured:** at `tp4-switched` with both builds capped, candidate-55 is 0.8% slower
  at 24 rows and faster at 32 and more. A TP4 step has the RoCE waits between the kernels, which one GPU does not.
- **The other arms are not verdicts.** The folder also holds one boot each of seven PRs outside this candidate
  (#994, #992, #1078, #1086, #1005, #957 parts 2 and 4) mounted over candidate-55. Their differences from
  `c55_l2cap` are 0.2% to 0.6% of forward time with no repeat boot (`c55_l2cap_r2` did not run), which is no
  interval. #1081's arm did not boot (reported as a `dead_rows_enabled()` mismatch between its overlay and the image; no
  log was kept). Those PRs enter later candidates with their own validation, not this one.
