# Bidirectional opposite-rank relay

Candidate, not promoted. Base deployment commit: `5cd9d26`. Source identity
and the full series are in `source.json` and `upstreams.lock.json`.

## Change

Direct neighbours exchange their complete payloads over their direct QPs.
For the opposite rank, each physical stripe is split into disjoint 16-byte
packs: one fragment travels clockwise, the other counterclockwise. Both
directions are used within a collective. Odd packs alternate by epoch and
lane; a 16-byte payload necessarily has one empty fragment.

The relay progresses each incoming lane independently. A missing clockwise
input does not prevent forwarding a ready counterclockwise input. Every
fragment, including an empty fragment, has its own ordered completion flag.
The GPU waits for all opposite-rank fragments before reading the receive
slot. Direct peers still require only their direct-lane flags. This works
with one or two physical lanes per neighbour.

The two-slot lifetime, original-source receive slots and fixed-rank sum
order are preserved. Wire ABI 9 rejects mixed old/new peers. Both ordinary
and prepared GPU launcher keys include the relay layout; the preparation
query's `roce_rdma` capability name is distinct from the runtime routing mode.

## Why

The previous ring relay forwarded opposite-rank data clockwise only. For a
payload of S bytes per rank it sent 2S clockwise and S counterclockwise.
The new relay sends approximately 1.5S each way, still 3S total. For normal
evenly divisible message sizes the directional split is exact.

Saved hardware counters confirm the old imbalance (`control-counters.json`,
extracted from collective-policy run15): roughly 32.04 GB clockwise and
16.02 GB counterclockwise per node. All four nodes had the same aggregate
payload work. These counters establish a routing imbalance, not the cause
of the observed temperature difference between nodes.

The new relay posts four rather than three signalled transfers per physical
lane per collective (two direct, two relay fragments). Extra completions and
GPU flag waits can matter for small messages. Hardware qualification must
measure latency and power rather than assume balanced traffic is faster or
cooler. NCCL's separate bulk path is unchanged by this patch.

## Local verification

```sh
bin/spark3 --cluster-config experiments/2026-10-03-ring4-bidirectional/cluster.json doctor
bin/spark3 --cluster-config experiments/2026-10-03-ring4-bidirectional/cluster.json build prepare --only b12x
python3 experiments/2026-10-03-ring4-bidirectional/test-protocol.py
python3 -m unittest discover -s tests
git diff --check
```

The native C simulator uses delayed DMA reads and independent QP progress,
with address and undefined-behaviour sanitizers. It exercises 26,000
collectives across direct TP3, ring4 and mesh4, including wraparound,
unequal rank progress, tiny/uneven messages, explicit direction stalls,
stop/error handling and mixed-ABI rejection. It verifies complete payloads,
no duplicate packs, direct-neighbour routing and directional byte balance.
CPU tests also execute the GPU launchers' flag-selection expressions and
check layout-specific cache keys and preparation metadata.

During test development, the explicit direction-stall fixture initially
rang its doorbell before starting the proxy, which correctly treated that
epoch as its initialization state. The fixture now starts before publishing.
The first launcher-inspection test used the wrong method name (`_kernel`
instead of `kernel`); that test was corrected. These were fixture failures,
not evidence of successful hardware execution.

## Hardware acceptance

Compare `control.json` and `cluster.json` with the same node map, message
sizes, NCCL settings, pinned verification table and thermal starting point.
Use the existing collective-policy runner for an A/B/A screen of all-reduce,
all-gather and NCCL fallback boundaries. Require exact data, graph replay,
no transport errors and per-port counters confirming the directional split.

Then compare decode at one/eight streams and prefill with temperature,
actual fan RPM, proxy CPU time, GPU clocks/power and NIC bytes recorded on
every node. Directional balance alone is insufficient thermal evidence.
Keep all failed runs. No production changes or promotion are implied.

## Hardware result, 2026-10-03

Three launches (control / candidate / control) passed on all four ranks:
exact-data checks, changing CUDA-graph inputs, dispatch boundaries and zero
tracked RDMA errors. Candidate image:
`sha256:08b001f19414fbcebb4bcdd8b4d449569977108e70daa5a7adfb8990ebc923ec`,
built from deployment `abcfda0`, with the same digest verified on all nodes.
The GPU preparation suite passed all four tests. The native protocol suite
also passed on a Spark's ARM CPU. CI passes after initializing its example
ring map; the first CI job lacked that intentionally ignored site file.

Each node's proxy counters changed from approximately 32.04/16.02 GB
clockwise/counterclockwise to 24.03/24.03 GB, with equal aggregate payload
work. Both HCA lanes and both directions contribute; payloads are split,
not duplicated.

BF16 all-reduce microseconds per call (median of the slowest rank in each of
five samples; each sample contains 256 calls):

| Input per rank | Control before | Bidirectional | Control after |
|---|---:|---:|---:|
| 10 KiB | 17.42 | 16.90 | 17.05 |
| 60 KiB | 28.07 | 28.04 | 28.84 |
| 480 KiB | 95.07 | 93.99 | 95.06 |
| 2 MiB | 314.81 | 307.74 | 315.31 |

This establishes working bidirectional delivery at near-baseline latency,
not a serving TPS improvement. The broader all-gather results are retained
in `results.json`; some are slightly slower. The short probes did not
reproduce the serving thermal failure. Their GPU temperature/power pattern
does not establish a thermal benefit.

The first telemetry wrapper matched the host filename rather than its
container name `/probe.py`, so these three launches have GPU/fan/system CPU
samples but no per-thread samples. The corrected wrapper records threads in
the subsequent NCCL experiment. Do not infer proxy CPU savings from this arm.

Native receipts are in `hardware/runs.tar.gz`, with every member hash in
`hardware/sha256.json`. Extract into a directory and reproduce the summary:

```sh
python3 experiments/2026-10-03-ring4-bidirectional/summarize.py EXTRACTED_DIR control1 candidate1 control2
```

NCCL bulk rings were independently found to use only clockwise channels;
their separate candidate and results are in `../2026-10-03-nccl-bidirectional`.
Neither candidate is promoted. The window ended at 06:56:41 UTC with all
nodes idle, fan controllers active and shared production checkouts unchanged.
