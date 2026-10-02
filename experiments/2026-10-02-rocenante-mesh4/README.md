# RoCEnante over a NIC-forwarded four-node fabric

Status: hardware collective correctness passes on all four Sparks for NCCL ring,
CPU relay and NIC forwarding. One transport-only candidate image has been built
and distributed with the same digest to every rank. Latency screening is recorded
below. Model serving and persistent fabric lifecycle remain unqualified; no
promotion has occurred.

The current site has four connected nodes. Real addresses, MACs and HCA mappings
live in the git-ignored `config/nodes-ring4.local.json`, selected by the three
`qualification-*.json` profiles. The tracked example stays synthetic. The live
inventory supersedes the earlier provisional dgx4 address. Qualification uses
separate clean checkouts at `{home}/projects/spark3-ring4-qualification` so host
maintenance cannot replace a running test's code.

Base deployment: `a5e66d1`. The intended variable is the path taken by
opposite-peer RoCEnante traffic: intermediate NIC forwarding instead of the
host CPU relay. The ring4 candidate remains available as a control. Native
weights, GPU collective kernels, buffer layout, reductions, and NCCL's
neighbour-ring policy are unchanged.

## Implementation

`rocenante-mesh4` derives a full logical peer map from the physical two-neighbour
map. Each rank opens two stripes to each of its three peers. The opposite pair's
first stripe crosses one intermediate rank on the first PCI root; its second
crosses the other intermediate on the second PCI root. Reverse QPs use the same
intermediate and lane, including RC acknowledgements.

Only opposite-peer QPs set flow label 16383 (`B12X_ROCE_TOPOLOGY=mesh4`). On mlx5
this selects the reserved UDP source 65535. A host-side RDMA-TX marker matches
that port, UDP destination 4791, IPv4 EtherType, and the exact source/destination
IPv4 pair. It changes Ethernet EtherType to 0x88b5. The intermediate NIC's TC
rule restores 0x0800, sets the final endpoint's MAC, and forwards to its other
port on the same PCI device. IP, UDP, BTH and payload bytes are preserved.

The endpoint's RC QP remains responsible for delivery. There is no intermediate
host buffer or CPU relay operation. Existing small/large collective dispatch
thresholds stay in place; this does not add bandwidth to a physical cable.

Source authority:

- `upstreams.lock.json`, `source.json`, and `b12x/series` select the reproducible
  candidate source. The series retains the CPU-relay control and adds a separate
  mesh mode; connection geometry and ABI checks reject mixed modes.
- `scripts/topology.py` derives logical peer routes; the node map describes
  physical cables only.
- `scripts/mesh_fabric.py` plans routes and exact TC rules and inventories NICs.
- `scripts/mesh_probe.py` owns forwarding for one bounded collective-probe
  container per host. It waits for marker readiness, requires hardware TC
  offload, stops the probe before removing paths, and rolls back its own entries
  after partial setup failures. Existing entries are never replaced.
- `native/marker.c` is the narrow host helper; source/license provenance is in
  `native/NOTICE`. It is built separately from the serving image.

Serving launch is blocked for this candidate until the hardware and persistent
serving lifecycle are qualified. The bounded runner is for the collective probe,
not a background fabric service. A force-killed runner may leave routes/filters;
its markers close on parent death. Stop the named probe container on all ranks
before clearing those exact rules. Do not run this fabric over live serving.

## Prepare and inspect

These local steps use no GPUs and do not change NIC state:

```sh
bin/spark3 --cluster-config experiments/2026-10-02-rocenante-mesh4/cluster.json doctor
bin/spark3 --cluster-config experiments/2026-10-02-rocenante-mesh4/cluster.json build prepare --only b12x
python3 experiments/2026-10-02-rocenante-mesh4/test-protocol.py
python3 scripts/mesh_fabric.py plan --nodes experiments/2026-10-02-rocenante-mesh4/nodes.example.json
```

Prepare a real site map after cabling ranks 0–1–2–3–0. Every HCA needs its
actual netdev, PCI address, MAC, IPv4 /24 and selected IPv4 RoCE-v2 GID. Use two
stripes per cable, in matching order at both ends. Update the candidate's
`nodes_config` and rendezvous address, publish the profile commit, and
synchronize clean checkouts with `bin/spark3 cluster sync`. The selected site map
stays git-ignored and is copied by sync under the new nodes standard. Keep
`deployment.launch_enabled` false while qualifying.

On each host, the following is read-only (substitute the published real map and
local rank):

```sh
python3 scripts/mesh_fabric.py doctor --nodes path/to/real-nodes.json --rank 0
```

The mesh profile's `bin/spark3 doctor --live` includes these NIC checks alongside
its ordinary live image and command checks. Before a mesh service exists, its
live-container comparison will naturally report a mismatch. Use the standalone
NIC doctor for network readiness alone.

Required settings are legacy eswitch / inline none / encap basic, hmfs steering,
hardware TC offload, MTU 9000, active RDMA MTU 4096, four hairpin queues and queue
size selected by `mesh_hairpin_queue_size` (1024 or 8192; default 8192). This
qualification explicitly selects the existing 1024 setting and changes no NIC
parameters. Doctor reports deviations and separate correction commands; it does
not execute them. Driver-init changes require stopping all RDMA users and a
coordinated reload or reboot. A readback of driver-init values alone does not
prove that the running driver has applied them; the hardware probe is the gate.

Build the host helper from the published checkout on each node using its installed
RDMA development headers/libraries (`libibverbs-dev`, mlx5 provider development
headers and a C compiler):

```sh
cc -std=c11 -O2 -Wall -Wextra -Werror \
  experiments/2026-10-02-rocenante-mesh4/native/marker.c \
  -o /tmp/spark3-roce-marker -libverbs -lmlx5
sudo install -D -m 0755 /tmp/spark3-roce-marker /usr/local/libexec/spark3-roce-marker
```

Record the helper's source/binary hash and compiler version in hardware results.
Build and distribute one candidate image digest through the normal build flow.
The candidate image is now built; its exact digest and source identity are
recorded with the hardware results. `build-candidate.py` creates the transport-only
overlay after checking the installed base image ID and all source-tree labels.

## Hardware qualification

Take an explicit four-host maintenance window, check the shared hold, stop and
remove the serving containers through the coordinated cluster workflow, then
render a probe command for each rank:

```sh
bin/spark3 --cluster-config experiments/2026-10-02-rocenante-mesh4/cluster.json topology probe dgx1
```

Render dgx2–4 as well and execute the generated commands together on their
respective hosts. Each command wraps the existing CUDA/NCCL/RoCEnante probe in
its local fabric lifetime. The runner checks the actual site inventory and
existing container/rule ownership before any mutation. It creates routes with
`add`, not `replace`, and checks `skip_sw` rules report `in_hw`. It never reloads
a driver or applies netplan. Requests have a 600-second container timeout and a
660-second host bound; marker failure stops the dependent probe.

Collect all four rank logs and TC counters. Qualify both directions on both
lanes: capture the reserved UDP source/mark at source, require the intended TC
counters to advance, and verify the final destination receives the original
RoCE frame. Require exact all-reduce, all-gather and reduce-scatter checks for
BF16/FP32 around both dispatch thresholds, changing graph-replay payloads, and
real RoCEnante operation counts. The combined probe also exercises NCCL; a
fallback-only result fails. Repeat after reboot and inject one marker failure
to verify container teardown and cleanup on every host. The current automated
runner handles local failures; other ranks terminate by their bounded probe
limits. This is not yet a cluster-wide serving supervisor.

After correctness, compare NCCL ring, CPU relay and NIC forwarding with the same
payloads/stripes. Then add and qualify the persistent serving lifecycle before
model benchmarks: c1/c8 decode, 4K/16K/64K prefill, TTFT, memory, and repeatability
on the same pinned cost table. Keep failed runs. Restore the agreed entry state before releasing the window. This qualification
started with all four nodes idle; it does not restore an obsolete three-node
physical topology or start an unqualified four-node serving profile.

## Evidence so far

- Local topology/doctor tests cover reciprocal opposite-peer paths, same-PCI
  forwarding, IP/GID matching, wrong mode rejection, hardware-only TC checks,
  partial-setup cleanup and unchanged promoted TP3 command/environment behavior.
- Fresh B12X preparation reproduces the recorded head/tree. Its real C proxy
  passes 18,000 simulated rounds with ASan/UBSan: direct3, relay4, mesh4,
  one/two stripes, changing sizes, delayed DMA, sequence wrap and error paths.
  The fake verbs layer checks opposite-only flow labels. This does not emulate
  NIC packet rewriting, reliable hardware delivery or CUDA kernels.
- The native marker compiles and links with strict warnings on all four hosts;
  its exact RDMA flow attaches successfully. Hardware TC rules attach with the
  existing 1024 queue size. No driver reload or host configuration change was needed.
- Hardware results and remaining limits follow below.

## Four-node hardware session, 2026-10-02

The session ran 20:23–20:55 UTC. Entry and exit state: four idle nodes, no model
service running. No driver, kernel, netplan, NIC queue, NVMe or sysctl setting was
changed. The shared deployment checkouts finish clean at `4449d6e`; isolated
qualification checkouts finish at `38cfe04`. The owned hold was removed at 20:55.
Unrelated stopped containers were preserved.

Identity:

- Linux `7.0.0-1019-nvidia-64k`, driver `580.178.04` on all four hosts.
- Same physical two-lane neighbour ring, MTU 9000 / RDMA MTU 4096, IPv4 RoCE v2
  GID 3, hmfs steering, four hairpin queues of size **1024** on all 16 functions.
- Base r5o image ID:
  `sha256:aad8a74089ff379f5bc7905e86f9c7e2c053396d0a4027039e869c505ca7621b`.
- Candidate image `vllm-ds41f-kkref:04c30fa98e79-r5o-roce-mesh4-v1`, same ID on all
  four hosts:
  `sha256:488fed96fecec12e60a32a75f387e057bbf56da384098ca6efb6757869734c2d`.
- Candidate B12X head `f7639617b6ba677cc2996b7351b99baa35f45b4f`, tree
  `f5b9429596f789ed88952860b43c97823bc44679`. vLLM and NCCL trees match r5o.
- Marker binary SHA-256 on every host:
  `239ed684d3ab4fc3bf02d5a5b44c7a3aaed03e4788c675b950ce99d8401fc563`.
  Built and run from `~/spark3-lab/mesh4/marker`; no system-wide installation.

### Run ledger

[Raw evidence](hardware/runs.tar.gz) contains every run's exact command arrays,
rank logs, result codes, inventory and cleanup evidence. The JSON
[summary](hardware/summary.json) includes every successful correctness result,
per-rank samples and forwarding counters. [Checksums](hardware/sha256.json)
cover the archive and session orchestration sources, retained as `.py.txt`
provenance rather than a supported deployment API.

| Run | Result |
| --- | --- |
| 00 inventory | Four nodes reachable by SSH aliases; every physical HCA/GID verified. The default 8192 queue contract differed from the installed 1024, so the explicit site contract selected 1024. |
| 01 direct links | Jumbo ping and bidirectional RDMA write passed on all eight neighbour lanes; 212.39–213.26 Gbit/s aggregate bidirectional throughput. Serial short screens, not a simultaneous saturation test. |
| 02 NCCL | Exact GPU collective results on all four ranks, eager and graph replay. |
| 03–06 build, attachment, distribution | Built one candidate image, distributed the same ID, compiled identical markers, attached hardware rules and cleaned them up on all four nodes. |
| 07 mesh | Stopped before GPU work: another session moved the shared checkout to newer main during the hold, removing the runner. No transport failure. |
| 08 mesh | Repeated from isolated checkouts; all four ranks passed, 114 RoCEnante operations each. |
| 09 CPU relay | Same correctness matrix passed, 114 RoCEnante operations per rank. |
| 10–12 latency | Mesh, NCCL, CPU relay respectively; all correctness checks passed. |
| 13–15 repeat latency | Mesh, CPU relay, NCCL respectively, with before/after RDMA counters; all correctness checks passed. |
| 16 marker failure | Terminated one dgx1 marker during active GPU benchmarking. Local runner detected its death and stopped the probe; coordinator stopped the other ranks. Expected rank exits 1/137/137/137. All processes finished about 6.4 seconds after injection. |
| 17 cleanup | No test containers, GPU processes, markers, owned routes or TC filters remain; all four NIC inventories exactly match run 00. NIC doctor passes. |

Two harness issues are retained in the interpretation: the first local sync
attempt hit macOS's SSH control-path length limit and changed no checkout;
subsequent syncs used `TMPDIR=/tmp`. The initial cleanup assertion incorrectly
required *all* containers to be absent; unrelated containers were in Created
state. The corrected audit requires no running or owned probe container and
preserves those unrelated containers.

The correctness matrix uses BF16/FP32, tiny and large payloads, both sides of
RoCEnante's 2 MiB all-reduce and 4 MiB all-gather dispatch limits, and four graph
replays with changing input on each size. Small uniform all-reduce/all-gather
use RoCEnante; **reduce-scatter uses NCCL in every arm**, as do over-limit calls.
The 19 size/dtype cases in each custom arm each check three collectives in eager
execution and four graph replays: 285 output checks per rank. This qualifies
these collectives, not arbitrary point-to-point traffic, expert all-to-all,
model quality, or batch-invariant inference.

### Latency screen

Identical inputs on four ranks; BF16 and FP32; 5,120 / 30,720 / 245,760 / 1,048,576
input elements per rank. For reduce-scatter this is the output shard length; the
input has four shards. Each graph contains 16 calls; each sample replays it 16
times, with a CPU-group barrier before the timer. Five CUDA-event samples per
case, two fresh container launches per arm. Take the slowest rank per sample,
then the median. The table gives the range of the two launch medians, not a
confidence interval. Compilation, connection setup and graph capture are outside
the timer. Final output checks also pass after the timing loops.

BF16 all-reduce, microseconds per call (lower is better):

| Per-rank input | NCCL ring | CPU relay | NIC forwarding |
| --- | ---: | ---: | ---: |
| 10 KiB | 90.6–92.4 | 18.2 | **16.2–16.9** |
| 60 KiB | 111.5–114.2 | 29.6–30.0 | **28.1–28.5** |
| 480 KiB | 168.9–174.1 | **96.3–98.4** | 286.3–326.6 |
| 2 MiB | 359.5–362.1 | **316.3–319.1** | 796.2–961.2 |

The first three lengths correspond to 1/6/48 rows of 5,120 BF16 elements, but
this is a collective microbenchmark without model computation or contention.
It does not measure end-to-end TPS, TTFT or prefill. No performance promotion or
new dispatch threshold follows from two screening launches alone. Container
memory was bounded at 12 GiB; peak memory was not measured, and no model weights
or KV cache were allocated.

### Forwarding works; large-payload reliability needs investigation

All eight owned forwarding rules report hardware offload and advancing packet
counters. Run 10 records **22,217,536 hardware-forwarded packets**, zero software
packets and zero TC-action drops; run 13 records 23,962,190 hardware packets.
These delayed counters prove both forwarding lanes carry traffic. They do not
measure every NIC queue or prove the absence of packet loss.

Run 13's RDMA counter deltas across the 16 HCAs are:

- `roce_adp_retrans`: **18,316**;
- `packet_seq_err`: **70,450**;
- `out_of_sequence`: **83,905**;
- `np_cnp_sent`: **167,810**.

The retransmission and sequence-error increments occur on the HCAs selected for
opposite-peer traffic. Identical screens through the CPU relay (14) and NCCL
(15) add **zero** to those counters. No unrecovered transport error or incorrect
result was observed. The retransmissions are consistent with the larger-payload
slowdown, but this session does not distinguish loss from reordering or identify
the specific NIC queue mechanism. A zero TC drop count is insufficient to make
that diagnosis. At the end of this first session, queue size 8192 was untested.
The subsequent
[queue-capacity investigation](../2026-10-02-mesh4-queue-depth/README.md)
reproduced buffer overflows, demonstrated the partial benefit of 8192, and
rejected a direct-first send-wave variant. Its evidence supersedes this open
loss/reordering diagnosis.

Next work is bounded: separate the loss/reordering mechanism using per-port and
queue counters, then compare one forwarding/pacing change at a time against the
same CPU-relay control. Retain the measured CPU relay as a candidate alternative.
Before any serving promotion, qualify restart/reboot and persistent fabric
supervision, select thresholds from a denser size sweep, then run full TP4 model
startup and c1/c8 decode plus 4K/16K/64K prefill. None of those serving tests ran
in this session.

Repository validation: 165 tests pass; promoted and qualification doctor checks
pass; `git diff --check` is clean. A full local `build check` reports that the
complete promoted tree is not prepared. The transport-only image build and
its source-label verification succeeded on dgx1; this is distinct from a full
base-image rebuild. Both logs are retained in `hardware/`.
