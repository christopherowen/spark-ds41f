# candidate-55, third build, 2026-11-12

vLLM tree `e8c20f54` (#1134 at `6757c7a1`, #1145's `0804`), after "Yes, rebuild" (owner, 2026-11-12). The specs' runs
are named `<run>-b3`, so the second build's result folders on the nodes are kept.

## Build and release tests

| Image | Image ID | Layers | Built |
| --- | --- | --- | --- |
| `spark-ds41f:candidate-55` | `ad905a6d` | 56 | dgx4, job `20261112T070706-build.json` (deployment `5c50cf14`), 07:07-07:19Z |
| `spark-ds41f:candidate-55-no974` | `1d02f07b` | 53 | dgx4, job `20261112T073136-build.json` (deployment `da2f49dd`), 07:38-07:48Z |

- **Job `20261112T070706` failed** after building and distributing candidate-55: "stale spark-ds41f:candidate-55-no974
  on dgx1 (tree ed91c2734897) could not be removed". Exited `spark-ds41f` containers still referenced the second
  build's control image. They were removed, and job `20261112T073136` reused candidate-55's image (its tree label
  matched), built the control and ran the release tests.
- Each image has the same layers on all of dgx1..dgx4. `ad905a6d` was then shipped to sp1..sp4 and sp5..sp7.
- **Release tests on `ad905a6d`:** pass, 10,243 passed, 357 skipped, 0 failed, 2,628 s on dgx4
  ([release-tests-ad905a6d](../release-tests-ad905a6d)). That is one more pass than `4aff1368` with the same skips.
  `0804`'s `test_external_dmabuf_carveout_loads_and_rejects_out_of_bounds` is the only test function the series
  adds between the two trees (`0800`'s new hunks in `test_direct_loader_cuda.py` define none), and dgx4 has
  `/dev/dri/card0`, so #1145's test ran and passed.

## Jobs

Every job ran at deployment commit `bbfe2618`, and every boot ran the pre-launch defragmentation.

| Testbed | Job | Spec | Window (UTC) |
| --- | --- | --- | --- |
| `tp4-switched` (sp1..sp4) | gate, 4 boots | [gate-tp4-switched.json](../../gate-tp4-switched.json) | 08:02-08:56Z |
| | c1 decision check, 4 boots | [c1-decisions-tp4-switched.json](../../c1-decisions-tp4-switched.json) | 08:56-09:18Z |
| | screen, 2 boots | [screen-tp4-switched.json](../../screen-tp4-switched.json) | 09:18-09:47Z |
| `tp3-switched` (sp5..sp7) | gate, 4 boots | [gate-tp3-switched.json](../../gate-tp3-switched.json) | 08:08-09:10Z |
| | screen, 2 boots | [screen-tp3-switched.json](../../screen-tp3-switched.json) | 09:10-11:05Z |
| `tp4-ring` (dgx1..dgx4) | gate, 12 boots | [gate-tp4.json](../../gate-tp4.json) | 08:40-11:16Z |
| | control gate, 2 boots | [gate-no974-tp4.json](../../gate-no974-tp4.json) | 11:16-11:42Z |
| | c1 decision check, 4 boots | [c1-decisions-tp4.json](../../c1-decisions-tp4.json) | 11:42-12:04Z |
| | screen, 2 boots | [screen-tp4.json](../../screen-tp4.json) | 12:04-12:33Z |
| | control screen, 2 boots | [screen-no974-tp4.json](../../screen-no974-tp4.json) | 12:33-13:02Z |

Every boot reached ready; no memory guard fired. No image's first boot on a node compiled a kernel (kernel caches +0
files).

## Gates: every run gives candidate-54's hashes

| Testbed | Image | Boots | Runs identical | Hashes (logprobs; text) |
| --- | --- | --- | --- | --- |
| `tp4-ring` | candidate-55 | 12 | 48 of 48 | 18d61961/d6568649; 0a2dacdd/12fd8257 |
| `tp4-ring` | control | 2 | 8 of 8 | 18d61961/d6568649; 0a2dacdd/12fd8257 |
| `tp4-switched` | candidate-55 | 4 | 16 of 16 | 18d61961/d6568649; 0a2dacdd/12fd8257 |
| `tp3-switched` | candidate-55 | 4 | 16 of 16 | 5c22ce41/bce6092e; 26a739a9/12cfee9b |

- Every scenario (`c8`, `staggered`, `mixed`, `cached`, `again`, `mixed15`) has 0 token and 0 logprob differences.
- These are candidate-54's TP4 hashes ([gate-0799](../../../candidate-54/receipts/gate-0799/README.md)) and TP3
  hashes ([candidate-54's README](../../../candidate-54/README.md)). #974's toolchain leaves temperature-0 output
  unchanged at TP3 as well as TP4.
- Read with `bin/spark candidate gate <testbed> <run> --expect <hashes>`.

Distinct-stream checks, mean tok/s over the boots (range):

| Testbed | Image | c16 | c8 | c4 |
| --- | --- | --- | --- | --- |
| `tp4-ring` | candidate-55, 12 boots | 272.92 (271.38-274.36) | 201.88 (200.60-202.94) | 156.61 (155.95-157.99) |
| `tp4-ring` | control, 2 boots | 272.57 (272.53-272.62) | 200.47 (200.22-200.72) | 156.39 (155.88-156.91) |
| `tp4-ring` | candidate-54's `0799` gate, 12 boots (another session) | 270.60 (269.92-271.45) | 200.81 (199.84-201.29) | 155.07 (154.19-156.61) |
| `tp4-switched` | candidate-55, 4 boots | 269.64 (269.31-270.14) | 198.25 (197.60-199.00) | 154.28 (154.01-154.52) |
| `tp3-switched` | candidate-55, 4 boots | - | 153.83 (153.23-154.36) | 118.22 (117.71-118.93) |

## c1 decision checks

[c1_decisions.py](../../../../2026-11-10-tp4-ring-determinism/boot-differences/c1_decisions.py) and
[c1_paired_decode.py](../tp4-ring/boot-differences/c1_paired_decode.py) against candidate-54's four c1 boots
([final/](../../../candidate-54/receipts/final)), both cross-session:

| Testbed | Decisions | Paired decode time per request | Decode tok/s, per boot | servebench output tok/s |
| --- | --- | --- | --- | --- |
| `tp4-ring` | 4 of 4 boots agree (59 requests); all 28 pairs of the eight boots agree on the 58 both sent | **-1.59% ± 0.17%** (+1.63% tok/s) | 85.76-85.91, candidate-54 84.42-84.49 | 82.57-82.83, candidate-54 80.55-80.65 |
| `tp4-switched` | 4 of 4 boots agree (59 requests); all 28 pairs agree on the 57 both sent | **-2.28% ± 0.08%** (+2.35% tok/s) | 84.85-84.98, candidate-54 82.92-83.01 | 82.05-82.13 |

At `tp4-ring` the third build's c1 boots decode as fast as the second build's (+0.00% ± 0.17%, paired over 59
requests).

## Screens against candidate-54's recorded screens

The mean over both boots of each, against `candidate-54-0799-<testbed>-screen` and `-again` (taken without the
pre-launch defragmentation). Read with `bin/spark candidate verdicts <testbed> <queue entry>`.

`tp4-ring`:

| Metric | candidate-55 | Verdict | Control | Verdict |
| --- | --- | --- | --- | --- |
| engine grid, geometric mean | +0.67% (slower), margin 0.30% | worse | +0.22%, margin 0.42% | same |
| c16 paired decode | +0.75%, margin 0.48% | better | +0.72%, margin 0.55% | better |
| c16 output | +0.86% (272.51 / 271.86 tok/s), margin 1.03% | same | +0.81% (272.21 / 271.93), margin 0.98% | same |
| prefill 32K | -2.31% (8,491 / 8,508 tok/s), margin 2.90% | same | -2.09% (8,491 / 8,547), margin 6.41% | same |
| prefill 256K | -3.37% (8,544 / 8,567 tok/s), margin 0.55% | **worse** | -3.08% (8,580 / 8,583), margin 0.30% | **worse** |

`tp4-switched`:

| Metric | Change | Margin | Verdict |
| --- | --- | --- | --- |
| engine grid | +1.11% (slower) | 0.30% | **worse** |
| c16 paired decode | +1.49% | 0.50% | better |
| c16 output | +1.47% (268.39 / 268.67 tok/s) | 1.03% | better |
| prefill 32K | -0.01% (8,672 / 8,652 tok/s) | 2.19% | same |
| prefill 256K | +0.11% (8,754 / 8,770 tok/s) | 0.95% | same |

`tp3-switched`:

| Metric | Change | Margin | Verdict |
| --- | --- | --- | --- |
| engine grid (15 shapes) | -2.29% (faster) | 0.33% | better |
| c8 paired decode / output | +0.43% / +0.57% | 0.65% / 1.45% | same |
| `single-stream` c1 paired decode / output | +1.42% / +1.29% | 0.29% / 4.28% | better / same |
| `sampled` c16, c8 paired decode | +0.53%, +0.55% | 1.45%, 1.76% | same |
| `sampled` c1 paired decode | +1.60% | 1.56% | better |
| `stall-3`: 131K prefill under decode | -1.32% | 0.94% | **worse** |
| `stall-6`, `stall-7` | -0.79%, -0.86% | 2.09%, 2.52% | same |

- **TP3 prefill measured for the first time:** 32K 6,415 / 6,387 tok/s, 256K 6,344 / 6,314. candidate-54's TP3 screen
  has none to compare (sp5's 4 GiB bench floor).
- Model step: acceptance length 2.301-2.304 at `tp4-ring` (both images), 2.304-2.313 at `tp4-switched`, 2.307 at
  `tp3-switched`; perplexity 1.31888 at TP4, 1.31608 at TP3.

### candidate-55 against the control, the same window at `tp4-ring`

[candidate-55-vs-no974-screen-tp4-b3-table.txt](tp4-ring/lab/candidate-55-vs-no974-screen-tp4-b3-table.txt) (its
first line is the commit the table tool ran at):

| Metric | Change | Margin | Verdict |
| --- | --- | --- | --- |
| engine grid | +0.11% | 0.36% | same |
| c16 paired decode | +0.03% | 0.52% | same |
| c16 output | +0.11% | 1.12% | same |
| prefill 32K | +0.01% | 6.48% | same |
| prefill 256K | -0.42% | 0.52% | same |
| model step | greedy 128/128 identical, teacher-forced distributions 100% identical (KL 0) | | |

```
python3 scripts/servebench_suite_table.py --suites-dir /tmp/c55cmp-b3 cmp candidate-55 candidate-55-no974 candidate-55-again
```

over a folder on dgx1 linking the four suite runs.

- **#974's serving mark at `tp4-ring` passes again:** grid, c16 and prefill not worse than the control.
- **The 256K regression at `tp4-ring` is in the control too**, as in the second build: it comes from #1134, #1096,
  #989, #982 or the pre-launch defragmentation, not CUDA 13.5. It does not appear at `tp4-switched`.

## Time to ready

`ready` splits into the time to launch (stopping the previous engine, the pre-launch defragmentation, arming the
guards) and the engine's own time from launch to API ready. Each image's first boot on a testbed is left out
(none compiled a kernel); candidate-54's `0799` gate had none:

| Testbed | Image, job | Boots | Launch | Engine | Ready |
| --- | --- | --- | --- | --- | --- |
| `tp4-ring` | candidate-54, `0799` gate (2026-11-11) | 12 | +10 s | 98-103 s, mean 101.0 | 109-114 s |
| `tp4-ring` | candidate-55, gate | 11 | +11 s | 102-107 s, mean 105.0 | 113-118 s |
| `tp4-ring` | candidate-55, c1 check and screen | 6 | +11 s, one +68 s | 103-105 s | 115-116 s, one 171 s |
| `tp4-ring` | control, gate and screen | 3 | +11 s | 101-109 s | 112-120 s |
| `tp4-switched` | candidate-55, all jobs | 9 | +12-13 s | 117-127 s | 130-140 s |
| `tp3-switched` | candidate-55, all jobs | 11 | +12-13 s, one +70 s | 112-126 s | 125-139 s, one 184 s |

- **The engine's startup is about 4 s (4%) longer than candidate-54's** at `tp4-ring`, in candidate-55 and the
  control alike, so not from #974. Which of #1134 (the sponge held through post-load processing) and #1096 is not
  measured; #1096's serving mark (time to ready not worse) is not met as measured, cross-session.
- **The slow boots are slow before launch, not in the engine:** at `tp4-ring` (c1 check, boot 3) and `tp3-switched`
  (screen, boot 6) the launch came at +68 s and +70 s, and the engine's time was normal (103 s, 114 s). The second
  build's 176.3 s boot (gate, boot 11) was the same, launch at +68 s. That is 3 of 66 boots over both builds. The log
  does not time the pre-launch steps separately.

## Files

- `<testbed>/lab/`: each job's table, window-log segment (`*-window.log`), overlay list (none) and, for the gates,
  `<run>-records.sha256`. The record files (`record-*.json`) stay on each testbed's head (dgx1, sp1, sp5) under
  `results/private/determinism/<run>/`; each list was checked against the head's copies. No run differed, so they
  hold no differing request.
- `<testbed>/determinism/<run>/`: each run's summary and telemetry, and each distinct-stream run.
- `<testbed>/suites/<run>/`: the screens' suite runs. Each `model.json.gz` stays on the head, listed in
  `<testbed>/large-files-on-nodes.tsv`.
- `<testbed>/boot-differences/`: each c1 boot's `c1.json`, with every request's per-step token counts.
