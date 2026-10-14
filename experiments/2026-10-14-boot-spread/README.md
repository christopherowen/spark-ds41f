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
