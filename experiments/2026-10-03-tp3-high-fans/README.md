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

Status: benchmark running; evidence and results pending.
