# Relay streaming and GPUNetIO qualification

The existing balanced policy remains the recommendation. Chunk-level forwarding
works, but neither of the two streaming designs tested here improves bulk
collective latency. GPUNetIO's CPU-assisted path works on every neighbour link
after adapting its sample to shared host memory and system-scope publication.
The direct GPU doorbell experiment had a host-level failure and is excluded
from the runner. Nothing here is promoted or installed into serving.

Base deployment: `441c97f`, branch `switchless-3-4`. Frozen measured image:
`vllm-ds41f-kkref:04c30fa98e79-r5o-roce-balanced-dispatch-v1`, digest
`sha256:6e03995e36aac90c578fe2d00796068a32ecb8316ed0bae201352d1486272d4d`.
The model-free probes mount only their source-qualified proxy C file and Python
ABI binding over that image. They compile the native proxy through its existing
source-hashed loader. The `*-source-only.json` profiles are unbuilt and cannot
launch serving. Production configuration, weights and host packages are unchanged.

## What blocks the existing relay

RDMA posting is already asynchronous. The CPU proxy observes its local GPU's
ready epoch, posts its local sends, waits for incoming neighbour fragments and
forwards them. The two directions already progress independently. Each lane's
readiness flag covers its entire payload, however, so forwarding cannot begin
at the first chunk. CQ polling is nonblocking; the explicit queue-capacity loop
only spins when a per-QP outstanding limit is reached.

Patch 0012 records bounded host timestamps outside the measured logging path.
Records are printed after the proxy thread joins. `B12X_ROCE_TRACE_EARLY=1`
additionally checks incoming flags every 64 idle polls, observing data available
before the local GPU's doorbell. It adds observer cost and is not a speed arm.
The trace uses one CPU clock domain; it does not infer GPU timestamps.

Representative medians from `early-a`, all four ranks, BF16 all-reduce:

| Input bytes per rank | Local posting phase | Relay phase | Inside posting calls, both phases | Final CQ drain |
| --- | ---: | ---: | ---: | ---: |
| 10 KiB | 0.67 us | 6.42 us | 0.99 us | 0.08 us |
| 60 KiB | 0.66 us | 11.22 us | 0.96 us | 0.08 us |
| 480 KiB | 0.67 us | 45.52 us | 0.98 us | 0.08 us |
| 1 MiB | 0.66 us | 86.80 us | 0.98 us | 0.08 us |

Posting-call time overlaps the phase columns and must not be added to them.
The relay phase includes incoming transfer, readiness polling and forwarding;
it is not all recoverable CPU overhead. Median queue-capacity wait is zero.

Early readiness appears on 0.2–1.4% of measured rank-calls across the tested
custom cases. Of 148 observations delayed over 20 us, 141 occur at the first
call of a measurement sample, where host barriers and event scheduling separate
samples. This is weak evidence for removing the local-GPU gate in steady,
synchronized traffic, and does not establish its cost under mixed serving load.
The summary retains non-boundary observations separately. An early timestamp is
a lower bound on observed availability, not a promised end-to-end saving.

`trace-a` mixed large stdout JSON records and stderr trace records in Docker's
output stream. It passed, but its summary cannot be reconstructed reliably;
its raw output is retained. Subsequent runs separate stderr. `early-a` has
complete reconstructed JSON and exactly 1,280 attributed traces per timed
custom case per rank. `trace-b` repeats the trace-only arm with separated logs.

## Streaming experiments

Patch 0013 adds opt-in `B12X_ROCE_STREAM_CHUNK_BYTES`. Word zero of each existing
128-byte flag line retains the GPU's whole-fragment completion contract. Words
1–31 carry independent 32-bit epoch flags for chunks. RC QP ordering publishes
each flag after its payload; the relay uses the existing device-to-CPU barrier
before forwarding. It preserves disjoint clockwise/counterclockwise halves,
both HCAs per neighbour, odd-pack rotation, reduction arithmetic and two-slot
buffer ownership. The neighbour's relay-relevant half is submitted first.

Chunks are multiples of 16 bytes. A stripe at or below the requested chunk size
uses its existing whole-fragment operation. For larger stripes, the effective
chunk size is at least `align16(ceil(stripe_bytes / 31))`, so progress fits the
reserved words. Requested 32 KiB therefore becomes 33,840 bytes for a 1 MiB
stripe. The new ABI and peer blob reject mismatched chunk settings before work.

This first design still submits all local chunks before entering the relay
loop. Patch 0014 adds a shared progress loop, with
`B12X_ROCE_STREAM_WINDOW=2` or `4` outstanding operations per outgoing path.
Ready relay chunks get a turn before more local chunks are queued. Paths skip
unavailable credit and keep progressing other work; they do not block inside a
credit wait. This tests the combined interaction of streaming and short NIC
queues. The local-GPU epoch gate remains in both designs.

### Measured outcome

Post-recovery controls bracket the candidate runs. Values are microseconds per
collective, median of the slowest rank in each of five samples. Each sample is
16 CUDA-graph replays with 16 calls per graph. All ranks pass eager, graph and
numerical checks; all six post-recovery screens record zero RDMA error deltas
and exactly 25% of RoCEnante payload bytes on each interface. Sizes below are input bytes per rank.

| Operation | Whole-fragment controls, before / after | 64 KiB chunks | 32 KiB chunks | 64 KiB, window 2 | 64 KiB, window 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 all-reduce, 480 KiB | 95.6 / 95.5 | 94.1 | 97.3 | 103.8 | 100.5 |
| BF16 all-gather, 480 KiB | 105.8 / 105.9 | 105.1 | 107.6 | 112.3 | 106.7 |
| BF16 all-reduce, 1 MiB | 167.2 / 164.7 | 174.8 | 178.1 | 206.6 | 185.8 |
| BF16 all-gather, 1 MiB | 186.6 / 186.1 | 192.7 | 196.7 | 223.9 | 204.3 |
| FP32 all-gather, 2 MiB | 352.4 / 347.9 | 368.9 | 428.6 | 429.3 | 387.6 |

At 10 KiB, all-reduce controls themselves range from 16.6 to 20.5 us. The
unaffected NCCL cases also vary. Do not attribute apparent tiny-message wins
to streaming. Bulk controls are much tighter and show a consistent penalty.
This is a finite model-free screen, not an exhaustive tuning search or a model
TPS/TTFT measurement. No claim about serving speed or model determinism follows.

Streaming overlaps dependencies but does not remove payload bytes. This
one-shot algorithm sends two local payloads plus one payload's worth of relayed
fragments per rank: three input payloads total. Local and transit traffic share
the outgoing links. At 1 MiB with 64 KiB chunks, the protocol submits 52 signalled
fragment operations instead of eight, including four flag-only end markers.
The extra notifications and credit recycling are costs; limiting the queue
further can leave the link without work. These are explanations consistent with
the measurements, not a hardware-counter proof of the precise bottleneck.

## GPUNetIO on the current Sparks

Pinned source: [NVIDIA-DOCA/gpunetio](https://github.com/NVIDIA-DOCA/gpunetio/tree/586453728bcab2d4c50574924dc6cf43543c9ed4),
revision `586453728bcab2d4c50574924dc6cf43543c9ed4` (library build 4.0.1).
Built locally on all four hosts for SM121 with CUDA 13.0, driver 580.178.04,
64 KiB kernel `7.0.0-1019-nvidia-64k` and existing rdma-core 50 packages.
No DOCA SDK, GDRCopy, driver replacement, network reconfiguration or root package
installation was needed for the CPU-assisted sample.

[NVIDIA's Spark guidance](https://networking-docs.nvidia.com/doca/archive/3-5-0/doca-gpunetio#general-performance-and-best-practices)
specifies CPU/GPU shared memory and CPU-proxy transmissions. Our sample changes:

1. Put SQ/CQ backing memory and data buffers in explicit CPU/GPU shared host
   memory. Register data through the CPU pointer; reject unequal CPU/GPU aliases
   in this narrowly scoped sample adaptation.
2. Publish GPU-written host work queues with a system-scope release fence before
   signalling the CPU proxy. The original sample's GPU-scope fence is insufficient
   for this host-memory adaptation.
3. Keep the CPU proxy explicitly selected. There is no automatic direct-mode
   fallback and no implication that successful shared-memory allocation enables
   CPU-free transmission.

Before system-scope publication, both bounded pair attempts timed out. Diagnostic
producer indices show real traffic followed by a request-error CQE, rather than
failure to launch: the server stalled at producer 1,761 after completing the
1-byte test. With system scope, all ten sizes (1 byte through 16 KiB) completed
on both lanes of every cable: 1–2, 2–3, 3–4 and 4–1. Six link runs used 4,096
iterations per size; two used 512. Both endpoints returned success in all eight.

Reported half-round-trip estimates are 4.13–4.48 us at one byte and 6.71–6.96 us
at 16 KiB. These are diagnostic ping-pong samples, not four-rank collectives or
measurements directly comparable with the RoCEnante table. The sample synchronizes
on the last payload byte; it does not independently validate every payload byte.
A full correctness and contention test is still required for an adapter.

### Direct-mode incident and recovery

`gpunetio-direct-pair12` explicitly selected GPU doorbells (handler 2). Setup
succeeded, but no latency size completed. dgx1 rebooted during the run; dgx2's
process timed out yet the GPU continued reporting about 96% utilization with no
compute process listed. dgx1's previous journal ends abruptly and pstore is empty,
so the exact driver/firmware failure is unclassified. Successful UAR mapping was
not sufficient evidence that direct transmission was usable.

GPU reset on dgx2 failed, including with the persistence service temporarily
stopped and restored. An explicit reboot restored idle GPU activity. The pending
streaming control was interrupted before launching any GPU probe. Subsequent
controls and all streaming measurements were taken after recovery. Both hosts
retain their original kernel, driver and network configuration. No serving
containers were running before or during the incident.

The runner now rejects handler 2. A follow-up direct-mode fence patch was authored
(commit `b331472`) but never built or executed and was removed from active inputs; it is not a recovery or support claim. Further direct
mode work requires a separately planned host-level investigation. CPU-assisted
results and direct-mode failure remain separate findings.

## Decision and next work

- Keep the balanced whole-fragment policy. Neither streaming arm is a promotion.
- Do not equate nonblocking software with lower latency: some observed waiting
  corresponds to useful transfers on already shared links. The measurements do
  not prove that every possible streaming design is slower.
- Prefer an algorithmic reduction in bytes for bulk all-reduce. A standard
  four-rank ring reduce-scatter/all-gather sends 1.5 input payloads per rank,
  versus three here, with more sequential phases. NCCL already handles the large
  side of our measured crossover; retest that crossover before adding a custom
  bulk algorithm.
- A GPUNetIO adapter is now technically worth a bounded CPU-assisted experiment:
  it moves WQE construction to the GPU while the CPU still rings the NIC. It
  must beat the frozen policy with equivalent correctness, bidirectional routing,
  graph execution and mixed-load tests. These ping-pong results do not justify
  replacing RoCEnante.
- Event notifications are a separate power/idle strategy. A receive-side event
  needs an arrival notification protocol, not merely outgoing send completions.
  No interrupt-per-collective speed benefit was measured here.

## Reproduction and evidence

Obtain the shared cluster window first. Use published, identical qualification
checkouts and the profile-selected `nodes.json`. Every measured run cools all
four hosts below 55 C, restores normal fan control, records exact commands,
RDMA/port counters and memory snapshots, and bounds container lifetime.

```sh
TMPDIR=/tmp bin/spark3 --cluster-config experiments/2026-10-03-relay-progress/window-source-only.json build prepare --only b12x
TMPDIR=/tmp python3 experiments/2026-10-03-relay-progress/run-trace.py relay new-control \
  --source-profile experiments/2026-10-03-relay-progress/window-source-only.json \
  --benchmark --counter-samples --lengths 5120 30720 245760 524288
# Add --chunk-bytes 65536 for chunking; add --window 2 for shared progress.
# Prepare the same source on each node before using the runner.

# On each node, builds into .work only:
bash experiments/2026-10-03-relay-progress/build-gpunetio.sh system
# From the coordinator, CPU-assisted adjacent pair:
TMPDIR=/tmp python3 experiments/2026-10-03-relay-progress/run-gpunetio.py new-pair \
  --source system --server 0 --lane 0 --iterations 4096
```

The source branches are `relay-progress-metrics`, `relay-chunk-stream` and
`spark-shared-memory`. Patch series, exact trees and reproducible assembly heads
are pinned beside this file. The final native proxy passes 36,000 ASAN/UBSAN
simulated collectives, including sequence wrap, delayed DMA, partially ready
chunks, independent directions, mixed-configuration rejection and bounded stop.
The repository's 176 tests pass. Fresh B12X source preparation and configuration
doctor pass. Full image build-check is not available for these source-only
profiles: complete image inputs and a rebuilt serving image were not prepared.

All attempted runs and failures are retained in [hardware](hardware/README.md),
including stock memory-registration errors, dgx4's original parallel build-order
failure, the generated-header build guard, the local temporary-directory failure,
source-head guard failures, mixed trace output and the direct-mode incident.
The build helper now builds the library before examples and resets only its
known generated configuration header, retaining that diff in the build log.

The window was released at **10:45 UTC**, with all four GPUs at 0% utilization,
no compute processes or containers, and all fan-control services active.
Qualification checkouts were clean at published commit `9571973`.
