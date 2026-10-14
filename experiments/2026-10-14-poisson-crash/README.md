# Engine crash under Poisson load

## What happened

Both overnight goodput runs on r6k (pinned, production config, lab-booted) ended with
every request failing: `server: EngineCore encountered an issue` on the second run. The
first run showed it as "no output", because the error arrived inside the stream.
- `overnight-goodput`, 11:02 UTC: 158/159 failed at 0.4/s.
- `overnight-goodput2`, 13:18 UTC: 159/159 failed.

Closed-loop, long-context and grid runs on the same build in the same queue were clean.
Poisson at 0.6/s ran clean on r6h the day before, which was before lab telemetry existed.
The nodes logged no memory-guard kill, no OOM kill and no Xid. The engine's stack trace
was lost when the next boot replaced the container.

Direct reproduction failed. Both runs went straight from dgx1's shell, with no lab and no
telemetry:
- 18:24 UTC, Poisson 0.4/s against the running server: 74/74 succeeded;
- 18:26 UTC, a fresh boot, then 0.4/s with the overnight settings: 159/159 succeeded.

## Leading suspect: lab telemetry

Since 2026-10-14, every lab script step polls each node during the run:
- `nvidia-smi` every second;
- `docker top` on the serving container, plus a read of every serving thread's
  `/proc/<pid>/task/*/stat`, every 2 s.

The direct runs had none of this. Poisson load leaves the engine idle between arrivals,
a pattern the closed loops never produce.

## A/B ([ab.json](ab.json))

Four fresh boots of `config/cluster-tp4.json`, telemetry on, off, on, off (the new
per-job `"telemetry"` switch). Each runs Poisson 0.4 and 0.6/s, 60 s warm-up and 120 s
window. Every failed step now saves every node's serving log
(`<output>.engine-<node>.log`), so a crash keeps its stack trace.

### Result: telemetry is not the cause

| Boot | Telemetry | 0.4/s | 0.6/s |
| --- | --- | --- | --- |
| on1 (first boot after a dgx1 host reboot) | on | 78/78 failed | engine already down |
| off2 | off | 0/78 failed | 0/109 failed |
| on3 | on | 0/78 failed | 0/109 failed |
| off4 | off | 0/78 failed | 0/109 failed |

on1's logs (saved beside `results/private/determinism/poisson-crash-ab/poisson-0.4-on1.jsonl`):
- Rank 0 (dgx1) only: the TileLang router's check `route_partials: gate logits contain
  nan or inf` fired in block 0, threads 0-31. That is one warp, so one row (row 0) of a
  six-row step, non-finite across all 384 experts. Ranks 1-3 logged nothing until rank 0
  closed the TCP store.
- The scheduler dump: one running request, 25 tokens computed, 7 output tokens, six
  scheduled rows (one plus five drafts). The engine died within seconds of its first
  requests.
- Rank 0's sparknet health check: `RoCE proxy failed: relay stopped or GPU failed at seq
  4855`. On the ring, a rank whose collective wait timed out reports its relay's proxy
  failure (sparknet 2716866). The worker was already shutting down when the check ran,
  which gives the same message, and `check_health` reports the proxy failure before the
  kernel's own timeout record, so the waiting rank, HCA and sequence were not printed.

## Leading suspect: a collective wait past the spin limit

When a one-shot wait exceeds `SPARKNET_ROCE_SPIN_LIMIT` (5,000,000 polls, documented as
about 5 s, unmeasured on GB10), that collective and every later one become no-ops, and
the step keeps computing on unwritten buffers until the host's health check after the
step. A NaN check inside the step fires first, as the router's did here. A one-rank
stall at the first decode steps after boot would do it. The prime candidate is the
first asynchronous disk Engram reads (`SPARK3_ENGRAM_ASYNC=1` in production), which our
2026-10-07 trace already showed as the worst source of rank skew.

The same failure class appears in a forum report of r6h on three non-FE GB10s
(2026-10-14): a wait timed out at startup after one rank finished its cold-cache JIT
warmup 83 s before the others.

## Traced repro ([traced-repro.json](traced-repro.json))

Eight fresh boots of [traced.json](traced.json): the production configuration with
`SPARKNET_ROCE_TRACE=1` (each collective's per-peer arrival times on the GPU clock) and
`SPARK3_ENGRAM_ASYNC_TRACE=1` (every Engram stage, read and wait). Each boot opens with
a lone short request (Poisson 0.4/s, no warm-up, 20 s, so the trace ring keeps the first
steps). [scripts/collect_node_traces.py](../../scripts/collect_node_traces.py) then
saves every node's serving log and trace files to
`results/private/poisson-crash/traced/<label>/`.
