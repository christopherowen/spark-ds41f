# candidate-55 at `tp4-ring`, 2026-11-11/12

The five `tp4-ring` jobs of dgx1's window, queued back to back on deployment commit `1d5065b7`:

| Job | Spec | Image | Window (UTC) |
| --- | --- | --- | --- |
| Gate, 12 boots | [gate-tp4.json](../../gate-tp4.json) | `spark-ds41f:candidate-55`, `4aff1368` | 23:56-02:34Z |
| Control gate, 2 boots | [gate-no974-tp4.json](../../gate-no974-tp4.json) | `spark-ds41f:candidate-55-no974`, `6c9d7c02` | 02:34-03:00Z |
| c1 decision check, 4 boots | [c1-decisions-tp4.json](../../c1-decisions-tp4.json) | `4aff1368` | 03:00-03:21Z |
| Screen, 2 boots | [screen-tp4.json](../../screen-tp4.json) | `4aff1368` | 03:21-03:49Z |
| Control screen, 2 boots | [screen-no974-tp4.json](../../screen-no974-tp4.json) | `6c9d7c02` | 03:49-04:17Z |

Both images come from build job `20261111T223443-build.json` (vLLM tree `ed91c273`). Every boot ran the
pre-launch defragmentation (`host.defragment_before_start`). candidate-54's recorded runs were taken without it.

## Gates: every run gives candidate-54's TP4 hashes

| Image | Boots | Runs identical | Hashes (logprobs; text) |
| --- | --- | --- | --- |
| candidate-55 | 12 | 48 of 48 | 18d61961/d6568649; 0a2dacdd/12fd8257 |
| control (`no974`) | 2 | 8 of 8 | 18d61961/d6568649; 0a2dacdd/12fd8257 |

- Every scenario (`c8`, `staggered`, `mixed`, `cached`, `again`, `mixed15`) has 0 token and 0 logprob differences.
- **These are candidate-54's hashes** ([gate-0799](../../../candidate-54/receipts/gate-0799/README.md)). #974's
  toolchain (ptxas 13.5, cuBLASLt 13.10) leaves temperature-0 output unchanged at TP4, and so do the other four PRs.
  The README's expectation that #974 changes output does not hold at TP4.
- Read with `bin/spark candidate gate tp4-ring <run> --expect 18d61961/d6568649`.

Distinct-stream checks, mean tok/s over the boots (range):

| | c16 | c8 | c4 |
| --- | --- | --- | --- |
| candidate-55, 12 boots | 273.41 (272.53-274.59) | 201.66 (200.48-202.81) | 157.06 (155.36-157.98) |
| control, 2 boots | 273.02 (272.17-273.86) | 201.90 (201.36-202.44) | 157.10 (156.78-157.42) |
| candidate-54's `0799` gate, 12 boots (another session) | 270.60 (269.92-271.45) | 200.81 (199.84-201.29) | 155.07 (154.19-156.61) |

## Time to ready

Warm boots only (an image's first boot on a node compiles kernels):

| Image | Job | Boots | Ready |
| --- | --- | --- | --- |
| candidate-54 | `0799` gate | 11 | 110.3-113.8 s, mean 112.0 s |
| candidate-55 | gate | 11 | 113.9-124.7 s, median 117.1 s, and one at 176.3 s (boot 11) |
| candidate-55 | c1 check | 4 | 111.7-116.2 s |
| candidate-55 | screen | 2 | 117.2 s, 121.3 s |
| control | gate, screen | 4 | 111.5, 114.0, 119.2, 114.0 s |

The control has the pre-launch defragmentation and #1134's sponge too. Its boots are no further from candidate-54's than
candidate-55's c1 boots are, so neither adds a measurable startup cost here. Boot 11's 176.3 s is not explained.

## c1 decision check

`c1_decisions.py` ([source](../../../../2026-11-10-tp4-ring-determinism/boot-differences/c1_decisions.py)): the four
boots agree request by request on all 59 requests. With candidate-54's four c1 boots
([final/tp4-ring](../../../candidate-54/receipts/final/tp4-ring/boot-differences/candidate-54-c1-decisions-tp4)) all
28 pairs of the eight boots agree on the 58 requests both sent.

The accept decisions are the same, so the requests pair one to one
([c1_paired_decode.py](boot-differences/c1_paired_decode.py)):

| | candidate-54 | candidate-55 |
| --- | --- | --- |
| c1 decode, 58 requests, tok/s per boot | 84.47, 84.42, 84.43, 84.49 | 85.83, 85.67, 85.88, 85.98 |
| servebench output tok/s | 80.55-80.65 | 82.60-82.86 (59 requests in the 150 s window) |
| mean TTFT | 114.9 ms | 113.8 ms |

Paired decode time per request: **-1.59% ± 0.13%** (decode tok/s +1.64%). candidate-54's check ran on 2026-11-11 at
13:37Z, so this is cross-session. The c1 check has no control arm, so it does not say which PR the gain comes from.

## Screens

Against candidate-54's recorded screen (`candidate-54-0799-tp4-ring-screen` and `-again`, without the
defragmentation), the mean over both boots of each:

| Metric | candidate-55 | Verdict | control | Verdict |
| --- | --- | --- | --- | --- |
| engine grid, geometric mean | +0.77% (slower), margin 0.36% | worse | +0.40%, margin 0.30% | worse |
| c16 paired decode | +0.91%, margin 0.58% | better | +0.56%, margin 0.45% | better |
| c16 output | +0.94% (272.52 / 272.28 tok/s), margin 0.93% | better | +0.68% (271.87 / 271.52), margin 0.94% | same |
| prefill 32K | -1.52% (8,572 / 8,564 tok/s), margin 2.03% | same | -2.50% (8,466 / 8,501), margin 2.33% | worse |
| prefill 256K | -3.25% (8,574 / 8,560 tok/s), margin 0.66% | **worse** | -3.80% (8,506 / 8,530), margin 0.92% | **worse** |

candidate-55 against the control, the same window and testbed
([candidate-55-vs-no974-screen-tp4-table.txt](lab/candidate-55-vs-no974-screen-tp4-table.txt); its first line is the
commit the table tool ran at):

| Metric | Change | Margin | Verdict |
| --- | --- | --- | --- |
| engine grid | +0.24% | 0.50% | same |
| c16 paired decode | +0.19% | 0.50% | same |
| c16 output | +0.24% | 0.88% | same |
| prefill 32K | +1.25% | 1.35% | same |
| prefill 256K | +0.80% | 0.68% | better |
| model step | greedy 128/128 identical, teacher-forced distributions 100% identical (KL 0) | | |

- **#974's serving mark at `tp4-ring` passes:** grid, c16 and prefill not worse than the control, and 256K is 0.80%
  faster.
- **The 256K regression is in the control too,** so it comes from the other four PRs (#1134, #1096, #989, #982) or
  from the pre-launch defragmentation, not from CUDA 13.5. The screens can't say which. The series misses #989's
  serving mark (prefill at 32K and 256K not worse), but whether #989 is the cause is not known. Attributing it needs
  controls that leave out one change at a time.
- Model step: acceptance length 2.301-2.302 on all four boots; perplexity 1.31888 on both first boots.

The suite table compared with the table tool on dgx1, over a folder linking the four suite runs:

```
python3 scripts/servebench_suite_table.py --suites-dir /tmp/c55cmp cmp candidate-55 candidate-55-no974 candidate-55-again
```

## Files

- `lab/`: each job's table, window-log segment (`*-window.log`), overlay list (none) and, for the gates,
  `<run>-records.sha256`. The record files (`record-*.json`, 560 KB each) stay on dgx1 under
  `results/private/determinism/<run>/`. No run differed, so they hold no differing request.
- `determinism/<run>/`: each run's summary and telemetry, and each distinct-stream run.
- `suites/<run>/`: the screens' suite runs. Each `model.json.gz` stays on dgx1, listed in
  [large-files-on-nodes.tsv](large-files-on-nodes.tsv).
- `boot-differences/candidate-55-c1-decisions-tp4/`: each boot's `c1.json`, with every request's per-step token counts.
