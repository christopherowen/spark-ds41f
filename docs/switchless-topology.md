# Three- and four-node switchless deployments

The deployment tools accept three or four Sparks, with one GPU and one TP rank
per node. The promoted configuration remains the measured three-node deployment.
Four-node support is a candidate configuration path: its topology checks and
launch rendering are tested locally; its communication probe and full DS4.1
serving qualification still need four physically connected Sparks.

| Fabric | TP transport | Peer map |
| --- | --- | --- |
| Three-node triangle | Existing RoCEnante small collectives plus NCCL | Both other ranks |
| Four-node ring | NCCL ring for TP collectives | Previous and next ranks only |
| Four-node ring, experimental relay image | RoCEnante ring4 small collectives plus NCCL ring | Previous and next ranks only |

Ranks follow cable order: `0—1—2—3—0`. The three-node triangle already connects
every pair. A four-node ring has no direct link to the opposite rank. The B12X
peer-HCA map chooses an interface; it does not relay through another host.
The generated NCCL profile therefore disables RoCEnante. The separate
[RoCEnante ring4 candidate](../experiments/2026-10-02-rocenante-ring4/README.md)
adds host forwarding and requires its own image. It preserves the GPU allocator
and kernels. This candidate selects its isolated build lock with
`upstreams_config`; omitting that key preserves `upstreams.lock.json`.

## Configuration

Each cluster profile can select a repository-relative `nodes_config` file.
Omitting it preserves `config/nodes.json`. All cluster operations, rendering,
live checks and benchmarks resolve the same selected node map. No separate
node-map command-line option needs forwarding to remote helpers.

Start with `config/examples/nodes-ring4.json`. Its management addresses are
documentation addresses; replace them, the SSH user and interface names with
the site's actual values. The private fabric subnets are examples too.

For each node:

- `rank` is its position in the cable loop; the API head is rank 0.
- `roce_peer_hcas` lists the local HCAs reaching each immediate neighbour.
  Each link has one or two stripes, with matching counts at both endpoints.
  A local HCA belongs to only one neighbour.
- `roce_subnets` records the IPv4 `/24` network for each local HCA. Each
  cable/stripe has its own subnet, present at exactly its two endpoints.
  Two PCIe-root interfaces on one cable use two separate subnets.
- `roce_gid_index`, when supplied, selects that node's IPv4 RoCEv2 GID slot
  for both B12X and NCCL. It may differ between nodes.

Create a complete candidate from a base profile:

```sh
mkdir -p experiments/2026-10-02-switchless-ring4
cp config/examples/nodes-ring4.json experiments/2026-10-02-switchless-ring4/nodes.json
# Edit nodes.json for the actual hosts and cable subnets before proceeding.
bin/spark3 topology create \
  --nodes-config experiments/2026-10-02-switchless-ring4/nodes.json \
  --output experiments/2026-10-02-switchless-ring4/cluster.json
bin/spark3 --cluster-config experiments/2026-10-02-switchless-ring4/cluster.json doctor
bin/spark3 --cluster-config experiments/2026-10-02-switchless-ring4/cluster.json render dgx4
```

To use the 4 KiB base profile, put `--cluster-config config/cluster-4k.json`
before `topology create`. Creating a three-node candidate preserves the base
three-node transport settings. Creation never overwrites a file and always
sets `deployment.launch_enabled=false`; the image, model, context, KV budget
and promoted baseline remain inherited. The baseline identifies the source
configuration, not four-node acceptance. Additional memory is not automatically
allocated to KV before measurement.

The generator updates target TP, draft TP and `--nnodes` together. For four
nodes it selects `fabric.transport=nccl-ring` and the required transport policy
from `scripts/topology.py`. `doctor` rejects contradictory settings, non-neighbour
routes, duplicate ranks, asymmetric links and ambiguous subnets. Live doctor
and launch preflight also compare each selected GID with its declared subnet.
These checks report configuration/address mismatches; they do not prove that a
cable actually delivers RDMA packets. The collective probe checks communication.

## Why the NCCL settings matter

The pinned NCCL source is
[`73cf1122`](https://github.com/NVIDIA/nccl/tree/73cf112295c33aee2b895f329f592f2a9b4b0f97).

- `NCCL_ALGO=Ring`, one channel and ranks in cable order keep collective
  communication on neighbour edges. See `src/graph/connect.cc:connectRings`.
- `NCCL_RUNTIME_CONNECT=1` postpones connections until their algorithm is used.
  In `src/init.cc`, runtime connection requires **cuMem support**, so this profile
  sets `NCCL_CUMEM_ENABLE=1`. Keeping the three-node profile's `0` would eagerly
  connect trees and PAT, including unreachable opposite peers. This setting
  must be verified on-device; setting the variable cannot create missing CUDA
  virtual-memory capability.
- PAT, NVLS, CollNet, MNNVL, RMA and GIN are disabled. Custom vLLM all-reduce is
  disabled in the NCCL-only profile. The RoCEnante relay profile enables it
  with `B12X_ROCE_TOPOLOGY=ring4` and requires the patched image.
- NIC merging and subnet-aware routing are enabled. NCCL can select the local
  interface matching the neighbour's advertised subnet. The renderer supplies
  an exact per-node HCA list from the cable map.

This supports a single tensor-parallel group, with uniform all-reduce,
all-gather, reduce-scatter and broadcast. Expert parallelism, arbitrary
non-neighbour send/receive, additional data/context/pipeline-parallel groups
and alternative topology/algorithm plugins are outside this profile. Merely
setting `NCCL_ALGO=Ring` does not make those operations ring-aware.

## Qualification on the four-node fabric

Reserve a cluster window before recabling, GPU work or serving changes. Publish
the candidate and use coordinated `cluster sync` so every participating node
has the same revision and image. Physical recabling requires an explicit plan
for restoring the original triangle; a watchdog cannot restore cables.
The existing `scripts/lab.py` runner rejects a candidate whose node map differs
from the promoted map, because its automatic restoration targets that map.
Once a four-node topology is promoted, lab reads its configured node map.

With serving stopped and the fabric addressed, render one probe command per
node and run each command on the node it names, concurrently:

```sh
bin/spark3 --cluster-config experiments/2026-10-02-switchless-ring4/cluster.json topology probe dgx1
bin/spark3 --cluster-config experiments/2026-10-02-switchless-ring4/cluster.json topology probe dgx2
bin/spark3 --cluster-config experiments/2026-10-02-switchless-ring4/cluster.json topology probe dgx3
bin/spark3 --cluster-config experiments/2026-10-02-switchless-ring4/cluster.json topology probe dgx4
```

These commands **print plans** and do not start containers. Each printed command
uses the selected serving image, a separate container and IPC namespace, a
12 GiB memory limit, a different rendezvous port and a 600-second timeout.
The limit covers both NCCL communicators and cold RoCEnante compilation. It
mounts only the probe script, loads no weights and does not use serving caches.

The probe initializes vLLM's distributed groups and tests torch NCCL plus the
actual vLLM TP communicator: BF16/FP32, tiny/unaligned/large messages,
all-reduce, all-gather, reduce-scatter and repeated CUDA graph replay with
changing inputs. Every rank must print a passing JSON result and exit zero.
Keep the NCCL logs; check the rank ring and IB transport on each node. Run again
after a cold restart. A passing probe is a prerequisite, not serving acceptance.

Next enable launch in the reviewed candidate and start it through the existing
coordinated cluster command. Measure startup/steady memory, the quality gate,
one/eight-stream decode, short/32K/64K/long prefill, prefix replay and admission.
Run the model's determinism checks independently: a different TP size changes
floating-point reduction order, and neither this profile nor NCCL promises
bitwise equality with TP3 or batch-invariant output.

The fourth GPU reduces some per-rank weight work and adds memory, while the ring
adds communication hops and uses NCCL for small messages. Performance is not
assumed equal to the three-node RoCEnante baseline. Record a new baseline only
after four-node serving and measurements pass and the owner accepts promotion.

## NIC-forwarded RoCEnante candidate

The isolated [mesh4 experiment](../experiments/2026-10-02-rocenante-mesh4/README.md)
adds endpoint QPs between opposite ranks through ConnectX-7 hardware forwarding.
It retains the physical two-neighbour map and derives the logical peer map.
Its bounded collective runner owns temporary NIC markers, routes and TC rules;
NIC doctor reports settings and correction commands separately. The CPU relay
remains a comparison candidate. Cabling, the real four-node site map, hardware
qualification and the persistent serving lifecycle are still required before
launch or promotion.
