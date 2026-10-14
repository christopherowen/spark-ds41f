# Boot-to-boot spread: CPU placement and memory layout

Owner, 2026-10-14:
- "we should either reduce the amount of work necessary for serving or pin the processes
  to high clocks (assuming we do not have so many processes to require the slower cores)";
- "if there is memory layout optimizations to take into account, we should".

## Why

Within a boot, engine grids repeat to about ±0.05%. Between boots of the same build they
move 0.14-0.52% unlocked and 0.44% with clocks locked
([clock-lock](../2026-10-14-clock-lock/README.md)). Something set once per boot differs.
Two suspects:

1. **CPU placement.** GB10's Grace has ten fast cores (5-9, 15-19, 3.9 GHz, capacity
   997-1024) and ten slow ones (0-4, 10-14, 2.8 GHz, capacity 718-731), the same on every
   node. The serving container may run on all twenty. Only sparknet's proxy is confined to
   the fast cores (`SPARKNET_ROCE_PROXY_CPU=big`), which sparknet's probe found tightened
   one-shot latency. Host work in each step runs about 40% slower on a slow core, and the
   slowest rank paces every all-reduce.
2. **Physical memory layout.** GB10's GPU maps system memory. It can only use large
   pages over physically contiguous blocks, and the free-block mix at boot depends on
   what ran before. A fragmented boot could get smaller GPU pages and slower steps.

## Method ([spread.json](spread.json))

1. **Inventory under real load.** The sixteen-stream closed loop, first on the r6k
   configuration and then pinned ([r6k-big.json](r6k-big.json): the same plus
   `--cpuset-cpus=5-9,15-19`). The lab's telemetry now records the container's busy
   threads, their share of a core and the share of their CPU time on slow cores. This
   shows whether ten fast cores are enough, and whether pinning costs real serving
   anything.
2. **Twelve boots, interleaved:** unpinned, pinned, pinned with memory compacted before
   boot, four times over. Each boot stops serving, records every node's free memory by
   block size (`memory_snapshot`; `compact_memory` also runs
   `echo 1 > /proc/sys/vm/compact_memory`), boots, and runs one engine grid (100 replays,
   settle runs).
3. **Analysis:** `servebench spread` over each configuration's four grids, and between
   configurations with boots as the samples.

The questions:
- Does pinning shrink the boot-to-boot spread, or make steps faster?
- Does compaction add to that?
- Do slow boots line up with threads on slow cores, or with little free memory in
  blocks of 2 MiB or more?

## Results (2026-10-14, 07:10-08:08 UTC)

### Threads under real load (sixteen streams, closed loop)

| node | container cores | busy threads (share of a core) | CPU time on slow cores |
| --- | ---: | --- | ---: |
| dgx1 | 4.17 | sparknet-proxy 1.00, EngineCore 1.00, Worker_TP 0.94, Worker 0.94, Worker_TP 0.20 | 2% unpinned, 0 pinned |
| dgx2-4 | 3.12-3.13 | sparknet-proxy 1.00, Worker_TP 0.94, Worker 0.94, Worker_TP 0.19 | 1-2% unpinned, 0 pinned |

- **Busy threads:** three or four per node, against ten fast cores, so pinning fits.
- **Spinning:** the near-full threads mostly spin. The proxy busy-polls, and the CUDA
  synchronization spins while waiting on the GPU. Sleeping instead would free cores but
  add wake-up latency to every step.
- **Throughput:** pinned 226.2 tok/s ±0.73% against 225.2 ±0.81% unpinned (+0.4%,
  within the intervals).

### Twelve boots, interleaved (one grid each, 100 replays, settle runs)

| boots | decode step time vs the unpinned mean | boot-to-boot SD |
| --- | --- | ---: |
| unpinned u1-u4 | +0.85 / -0.04 / -0.57 / -0.24% | 0.61% |
| pinned p1-p4 | +0.11 / -1.21 / -0.78 / -0.85% | 0.56% |
| pinned, compacted c1-c4 | -1.06 / -0.75 / -1.31 / -1.06% | 0.23% |

- **Pinned against unpinned (all eight pinned boots):** -0.86% ±0.9% step time,
  borderline significant. Pinning never hurt anywhere.
- **Clocks and free memory were the same at every boot:** SM 2496-2502 MHz, 80-93 GiB free
  in blocks of 2 MiB or more after serving stopped. Compaction added only 0.1-1.5 GiB, so
  fragmentation does not explain the spread.
- **Unpinned boots spent only about 1% of their CPU time on slow cores**, yet ran slower.
  The cost is more likely migration (a cold private 2 MB L2, or a move between the two
  clusters' 8 MB L3s) than slow-core time itself.
- **Compaction was confounded:** the compacted boots always ran third in each round, right
  after another pinned boot. [compact.json](compact.json) repeats pinned against
  pinned-and-compacted in a balanced order (ABBA BAAB).

### Decision

The TP4 profile pins serving to the fast cores (`--cpuset-cpus=5-9,15-19`, owner
2026-10-14), and `doctor --live` checks it on every node. The TP3 profiles take the pin
with their move to r6k at the next re-cable.
