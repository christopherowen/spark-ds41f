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
