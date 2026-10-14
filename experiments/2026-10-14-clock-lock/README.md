# GPU clock locking for measurements

Owner, 2026-10-14: try locking GPU clocks during lab windows.

## Why

servebench's engine grid (settle runs, 100 replays, heat soak;
[2026-10-14-servebench](../2026-10-14-servebench/README.md)) has a ±0.15% statistical
interval on its decode geometric mean, but repeated grids moved by up to 0.5%: within a
boot -0.43 / +0.04 / -0.05%, boot to boot +0.14 / +0.52%. Systematic variation, not
sampling noise, sets its floor.

- The CPU governor is already `performance` (cores fixed at 2.8 and 3.9 GHz).
- The release run recorded no thermal slowdown or power cap (GPUs 68-72 °C, about 70 W).
- So if the GPU clock moves, it moves through ordinary frequency scaling.
- The GB10 reports a 2418 MHz default application clock and a 3003 MHz maximum, and
  lists no supported clocks.

## Method

`scripts/lab.py` gains a `gpu_clocks` job ([docs/lab.md](../../docs/lab.md)):
- `sudo -n nvidia-smi -lgc min,max` on every node, all or none;
- any production restore resets the clocks first, so production never serves locked.

1. [trace.json](trace.json): r6k unlocked, servebench's sixteen-stream closed loop (60 s
   warm-up, 300 s window), then one engine grid. The lab's telemetry samples every node's
   SM and memory clocks, power, temperature and clock-event reasons for each step
   (`<output>.telemetry.json`). This shows the clocks serving runs at, which sets the
   lock.
   - The first attempt (05:28 UTC) used a one-off sampling script whose arguments did not
     parse on the cluster. Its grid ran; the sampling moved into `lab.py`, so experiments
     need no script for it.
2. Locked validation, the same shape as the second grid validation:
   - r6k, r6h, r6k again;
   - each boot runs the closed loop with clock samples, then two grids;
   - all at one locked clock.
   If the half-percent drift is the clock, the grids' repeat differences fall to their
   statistical interval.

A lock below what serving reaches costs speed in that window only. Production keeps the
default clocks.
