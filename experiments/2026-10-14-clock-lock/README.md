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

## Unlocked trace (2026-10-14, 05:42-05:52 UTC)

Lab telemetry while r6k served sixteen streams in a closed loop (372 samples a node):

| node | SM MHz p10 / p50 / p90 (min, max) | power p50 | GPU °C p50 (max) |
| --- | --- | ---: | ---: |
| dgx1 | 2489 / 2489 / 2496 (2411, 2509) | 38.9 W | 60 (62) |
| dgx2 | 2489 / 2489 / 2496 (2411, 2515) | 41.0 W | 59 (61) |
| dgx3 | 2496 / 2496 / 2502 (2411, 2515) | 39.7 W | 57 (58) |
| dgx4 | 2489 / 2489 / 2489 (2405, 2515) | 37.7 W | 57 (58) |

- **Clocks under load:** nearly fixed, 2489-2502 MHz. The minimum (about 2410) is the
  step from idle to load. There were no clock-event reasons, and the memory clock is not
  reported on GB10.
- **During the grid:** the same pattern (dgx3 2496-2522).
- **Why it can still matter:** nodes differ by up to about 13 MHz (0.5%), and the slowest
  rank paces every all-reduce. That is the size of the grid's unexplained shifts.

## Locked validation ([locked.json](locked.json))

All four nodes are locked at 2489 MHz, the clock every node sustains under load:
- the `screen` suite runs on r6k, r6h and r6k again (servebench#217);
- then the clocks reset.

Against the unlocked second grid validation, the question is whether r6k against r6k
again moves less than +0.14 / +0.52% (geometric mean of the grid) and +0.13% (sixteen
streams). [sync.json](sync.json) moves the runner to the suite code first.

## Locked results (2026-10-14, 05:54-06:40 UTC)

The lock was accepted on every node, but GB10 ran under it. Telemetry showed 2463-2483 MHz
(p50 2470 / 2483 / 2476 / 2476), about 0.6% below the unlocked clocks, still differing by
node, with no clock-event reason. Sixteen streams fell from 225.3-225.7 to 224.7 tok/s.

| comparison | grid geometric mean | 16 streams tok/s | prefill 32K / 256K |
| --- | ---: | ---: | ---: |
| r6h -> r6k | -0.36% ±0.15 | -0.26% ±0.89 | -0.80% / -0.30% (level) |
| r6k -> r6k again (A/A, another boot) | **-0.44% ±0.16** | +0.10% ±0.93 | +0.42% / -0.00% |
| unlocked A/A (second grid validation) | +0.14 / +0.52% | +0.13% ±1.04 | |

- **Locking does not remove the boot-to-boot shift.** The same build still moved 0.44%
  between boots, beyond the grid's within-run interval. The shift belongs to the boot,
  not to the clocks. Within a boot, grids repeat to about ±0.05%.
- **Locking costs speed and buys nothing**, so measurements stay unlocked. The
  `gpu_clocks` job remains for experiments.
- **Model behaviour reproduced bit for bit** across boots and sessions: r6h on r6k's text
  had KL 0.0207 nats and top-1 agreement 95.15% on 128 prompts, against 0.0207 and 95.13%
  on 256 earlier.
- **Consequence:** a single comparison cannot resolve differences smaller than about half
  a percent on the grid. Suite tables therefore judge reference -> candidate against the
  run's own repeat: a change counts only beyond its interval plus the candidate's own
  boot-to-boot change. Here r6h -> r6k is "same" on every headline measure.
