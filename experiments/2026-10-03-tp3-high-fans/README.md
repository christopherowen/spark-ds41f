# TP3 with sustained maximum fans

Owner request: maximum fans for 20 minutes, then repeat the test after the
[ordinary-fan TP3 screen](../2026-10-03-tp3-revalidation/README.md) aborted on
dgx2 during long prefill.

The intended variable is the cooling policy: fan state 12 on all three nodes
for at least 1,200 seconds while serving is stopped, retained through startup
and the entire benchmark. The r5o image, triangle, serving configuration,
three-sample decode matrix and source/prefix prompts remain unchanged. Startup
will profile its normal TP3 speculative-verification costs again, so draft
acceptance and verification work must be reported alongside TPS.

The runner uses the existing published `6f71c3c` qualification checkout and
`experiments/2026-10-03-tp3-revalidation/control.json`. It sends no source or
configuration overlays to the nodes. It records fan states and RPM readings,
refreshes the exclusive hold, checks the actual 20-minute interval, and brings
every node below 55 C again after startup if necessary. The benchmark uses
`--cool-below 0` only because its normal precooling routine can release the
manual fan override. The runner performs that initial temperature check;
runtime thermal, memory and request guards remain unchanged.

Live doctor normally requires the fan-curve service to be active. Before using
the benchmark's `--allow-mismatch` for this deliberate manual override, the
runner asserts that its only three live errors are the paused fan services.
The result summarizer checks the benchmark's own live findings against the same
exact three messages. Image, source, memory, network and runtime requirements
remain subject to the usual checks.

Quality, decode, prefill and prefix run consecutively, without separately
cooling between suites. This tests continuous operation over this workload,
not just independently cooled bursts. On completion or error, the runner stops
the cluster, restores the original fan controllers, and releases the hold only
when restoration succeeds. The physical triangle and its addressing remain.

The cooling interval began at 12:45:09 UTC (14:45 Prague) on 2026-10-03;
startup cannot begin before 13:05:09 UTC (15:05 Prague). An initial setup attempt
failed on the Mac SSH control-socket path before changing any fans; it was
retried with the repository's required `TMPDIR=/tmp`. Both facts are recorded.

## Supervision incident

The first supervisor needed that manual-fan preflight adjustment. Replacing it
at the cooldown deadline left its coordinated-start child alive. The replacement
refused an existing log, then its cleanup briefly restored automatic fans while
the original start was still launching. The full 1,201.5-second idle cooldown
had completed, and no model measurements had begun. The original start completed
normally with every memory guard active. The hold and maximum fans were restored,
and `--resume-ready` supervised that existing boot after checking readiness and
the original cooldown receipt. It did not launch another boot. The incident,
original logs, and brief startup fan-policy interruption are retained; maximum
fans throughout startup cannot be claimed. Maximum fans throughout measurement
are verified separately in the sampled fan/RPM receipts.

## Result

The combined quality/decode/prefill/prefix screen **passed**, including both
nominal 256K requests that the earlier run could not finish. The normal runtime
thermal guard remained enabled. No thermal slowdown, power-cap time or swap
growth was recorded. Minimum available host memory was 6.88 GiB.

Recorded peak GPU temperatures were **69 / 67 / 65 C** on dgx1/dgx2/dgx3;
minimum thermal headroom was **22 / 21 / 27 C**. The earlier ordinary-fan
prefill run stopped on dgx2 at 83 C with 3 C headroom. Maximum board readings
were 77.7 / 75.7 / 71.8 C. The benchmark started with hottest-zone readings
39.8 / 38.8 / 39.7 C after the completed 20-minute idle cooldown and startup.

### Performance

Aggregate decode tokens/s, three samples per point:

| Workload | Published main | Previous ordinary fans | High-fan rerun |
| --- | ---: | ---: | ---: |
| prose-c1 | 52.8 | 50.9 | 51.4 |
| code-c1 | 64.8 | 60.0 | 65.0 |
| prose-c8 | 171.2 | 170.4 | 170.1 |
| code-c8 | 195.8 | 193.9 | 193.4 |

Single-stream step times were **40.744 ms prose / 45.765 ms code**, versus
40.845/45.356 ms on published main and 41.108/45.822 ms in the preceding
ordinary-fan screen. There is no statistically resolved decode regression
against main. The apparent code-c1 TPS recovery from 60.0 to 65.0 comes with
higher draft acceptance (1.874 to 2.132), while step time is nearly unchanged;
it is not evidence of an 8% kernel speedup from cooling.

| Nominal prefill length | Published main tok/s | High-fan tok/s |
| --- | ---: | ---: |
| 1024 | 2181.3 | 2182.9 |
| 32768 | 3768.1 | 3794.6 |
| 65536 | 3796.4 | 3819.2 |
| 262144 | 3684.1 | 3703.2 |

Both repetitions completed at every length. Actual input lengths match the
published baseline, as checked by the summarizer. Prefill changes are +0.1% to
+0.7%, with all comparison intervals including zero. Prefix reuse passed:
6.524 s cold, 0.268 s warm and 99% hit rate. Quality passed 5/5.

This demonstrates successful continuous operation for this combined workload
under the requested cooling policy. It does not establish indefinite sustained
capacity or qualify the unbuilt explicit-policy image. The experiment bundles
a longer idle cooldown with maximum fans throughout measurement; it cannot
separate their effects. The previous daemon log already reported state 12 during
its failing prefill, so the effective fan behavior and accumulated heat remain
worth distinguishing before changing the normal fan policy. No kernel/transport
speedup or permanent cooling change is claimed.

### Evidence and restoration

The 20-minute interval was 1,201.5 seconds. Every sampled fan state was 12/12.
Available paired RPM readings stayed near maximum; see `results.json` for each
node's minimum. One dgx2 sample at 13:09:50 UTC lacked both RPM reads; the raw
sample is retained and the summarizer counts it as incomplete rather than
filling it in. The pre-benchmark supervision incident above remains part of the
record and prevents claiming uninterrupted maximum fans throughout startup.

The benchmark client and all serving checkouts stayed on `6f71c3c`, with the
same configuration hash and image as the preceding TP3 screen. Only the three
intentional paused fan-service findings were allowed by live validation; the
preflight and report summarizer both check that exact error set. Native thermal,
memory and request checks were unchanged. The corrected local supervisor and
reporting sources are retained here; no remote source edits or overlays were used.

`hardware/runs.tar.gz` preserves the reports, command receipts, cooldown and
fan/RPM samples, startup and supervision failure, host journals, and restoration.
`hardware/sha256.json` records every file hash and the archive hash. Extract and
reproduce the comparison with:

```sh
TMPDIR=/tmp python3 experiments/2026-10-03-tp3-high-fans/summarize.py \
  RAW_DIRECTORY experiments/2026-10-03-tp3-high-fans/results.json
```

The cluster is stopped again, all three normal fan controllers are active, GPUs
are idle, and the exclusive hold is released. Triangle addressing remains.
