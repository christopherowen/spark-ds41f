# RoCEnante over a NIC-forwarded four-node fabric

Status: implemented for bounded collective qualification; no image built, no
hardware forwarding enabled, no throughput measurement, no promotion. dgx4 is
installed but its ring cable and netplan configuration are not ready. The checked-in
site map is an example (documentation management IPs and synthetic MACs/netdevs).

Site input: dgx4's management address is `10.0.1.79` (provided by the owner).
Use that address when preparing the real four-node map; keep the example map
synthetic. SSH from the development machine returned `No route to host` on
2026-10-02, so its hostname and NIC inventory have not been verified.

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
`nodes_config` and rendezvous address, publish the site/config commit, and
synchronize clean checkouts with `bin/spark3 cluster sync`. Keep
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
size 8192. Doctor reports deviations and separate correction commands; it does
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
The image tag in `cluster.json` is reserved, not an existing qualified image.

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
on the same pinned cost table. Keep failed runs. Restore the promoted topology
and service before releasing the window.

## Evidence so far

- Local topology/doctor tests cover reciprocal opposite-peer paths, same-PCI
  forwarding, IP/GID matching, wrong mode rejection, hardware-only TC checks,
  partial-setup cleanup and unchanged promoted TP3 command/environment behavior.
- Fresh B12X preparation reproduces the recorded head/tree. Its real C proxy
  passes 18,000 simulated rounds with ASan/UBSan: direct3, relay4, mesh4,
  one/two stripes, changing sizes, delayed DMA, sequence wrap and error paths.
  The fake verbs layer checks opposite-only flow labels. This does not emulate
  NIC packet rewriting, reliable hardware delivery or CUDA kernels.
- The native marker passes strict C syntax/type checking against dgx1's installed RDMA
  headers; it has not attached a flow. A subsequent link-build attempt could
  not complete because SSH to dgx1 timed out; linking remains to be verified.
- Read-only inventory on dgx1, dgx2 and dgx3 succeeds for all twelve functions.
  The only candidate-contract differences found were queue size 1024 instead
  of 8192. No host setting changed. dgx4 and the recabled fabric are untested.
- No serving image was built or deployed. TPS, TTFT and hardware memory impact
  remain unmeasured; no performance gain is claimed.
