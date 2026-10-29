# Fabric conventions

Status: proposal (2026-10-29). Step 1 of the migration (one renderer) is implemented; the
migration at the end says in what order the rest lands.

A site describes its cluster with as few settings as possible, and everything else is a fixed
convention. A convention is not a default: there is nothing to override. A host that does not
follow it is an error with the fix in the message, never a fallback to something else.

## What a site writes

| Setting | Example |
|---|---|
| Node names, in rank order (the first is the head) | `dgx1`, `dgx2`, `dgx3`, `dgx4` |
| Each node's management address | `10.0.1.71` |
| The fabric: `cabled` or `switched` | `cabled` |

Everything below is derived from those three, the cables and the hosts' addresses.

## The conventions

**Names.** A node's name ends in its node number (`dgx3` is node 3). The number, not the rank,
picks its addresses.

**Ports.** A GB10's ConnectX-7 has two QSFP ports, and each port has two PCIe functions (device
paths). They have the same names on every GB10:

| Port | Path 1 | Path 2 |
|---|---|---|
| 0 | `enp1s0f0np0` / `rocep1s0f0` | `enP2p1s0f0np0` / `roceP2p1s0f0` |
| 1 | `enp1s0f1np1` / `rocep1s0f1` | `enP2p1s0f1np1` / `roceP2p1s0f1` |

Collectives always use both paths of a port.

**Cabled fabric.** Port 0 cables to the next node, port 1 to the previous one. Three nodes form a
triangle and four a ring; the cables are read over LLDP, not written down. The cable between
nodes `a < b` has one /24 for each path, and node `N` takes host `N` in it:
`10.<a><b>.<path>.<N>/24`. For example, dgx1–dgx3 path 2 is `10.13.2.0/24`, with dgx3 at
`10.13.2.3`. This is sparknet's addressing rule (`sparknet/topology/discover.py`) and the one
the current node maps already follow.

**Switched fabric.** Each node uses port 0, both paths, into the switch. Each path has its own
/24, which is the same on every node; the subnets are read from the hosts' addresses.

**RoCE.** The RoCE v2 GID that holds the path's IPv4 address, and MTU 9000. The traffic class depends on the fabric:

- **Cabled:** 0. There is no switch to sort packets.
- **Switched:** DSCP 26 with ECN-capable transport, byte 106. The switch has to map DSCP 26 to a
  priority with PFC on it, and the hosts have to map it to the same priority (`dcb`, applied at
  boot). That is the one requirement this recipe places on a switch.

**Sockets.** Bootstrap and Gloo use the interface that routes to the head's management address.

**Transport.** It follows from the fabric and the node count: cabled with three nodes is
`oneshot-direct`, cabled with four is `oneshot-ring4`, and switched is `oneshot-switched`.

## One renderer

sparknet's `topology` package already discovers cables (`collect_lldp`, `resolve_links`), writes
node maps under the addressing rule (`generate_nodes`, `generate_switched_nodes`) and renders
each rank's environment (`render.node_environment`). It is the only renderer and validator of
the fabric: spark-ds41f calls it and adds only what serving needs (`VLLM_HOST_IP`,
`SPARK_DEPLOYMENT_TRANSPORT`, the node counts the kernels support, and the parallelism a
transport allows).

spark-ds41f runs the sparknet revision the cluster's upstream lock pins, the one its image
builds. `scripts/sparknet_topology.py` fetches that revision's `sparknet/topology` once into
`~/.cache/spark-ds41f/sparknet/<revision>` and imports it from there, so the recipe and the
runtime read one version, and a candidate that pins a newer sparknet gets that release's rules
with it.

The node map becomes generated: `bin/spark fabric discover` runs sparknet's discovery and
writes it. A person never edits it.

## What doctor shows

`bin/spark doctor` reads each node (read-only, over SSH) and prints what it derived: for each
node, the devices, which peer each cable reaches, the subnets, and the environment it will
render. A host that departs from a convention is an error saying what to change. For example:

```
dgx3 enp1s0f1np1: has 10.23.1.3/24; its cable reaches dgx1, so the convention is 10.13.1.3/24
```

## Levers this removes

| Today | After |
|---|---|
| `NCCL_IB_HCA` written into `config/cluster.json` (`tp4-ring`, `tp3-triangle`, as a prefix list) | Rendered, always exact (`=`) |
| `roce_peer_hcas`, `roce_subnets`, `nccl_hcas` in a hand-edited node map | Generated from LLDP and the hosts' addresses |
| `roce_gid_index` | Found by address: the GID holding the path's IPv4 address (sparknet #1) |
| `roce_traffic_class` (#669) | Fixed by the fabric: 0 cabled, 106 switched |
| `socket_interface`, `management_interface` | Derived from the route to the head |
| `fabric.transport` | Derived from the fabric and the node count |
| Hand-set `NCCL_IB_HCA`, `SPARKNET_ROCE_*` or `NCCL_IB_TC` in the recipe environment | Rejected by doctor: the renderer owns them |

**Changes in sparknet.** `SPARKNET_ROCE_HCA` takes only a comma-separated list of device names
present on the node, at most two. `=`, `^`, `:port` or an unknown name is an error. sparknet
stops reading NCCL's variables (`NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_IB_TC`): each library
reads its own, and the renderer writes both from the same values. That removes the grammar
mismatch, where `^` exclusions became inclusions and prefixes were taken as names, because one
library no longer parses the other's syntax.

## Migration

Each step leaves serving unchanged, and is checked as such before the next.

1. **Render from the node map.** Done.
   - spark-ds41f renders every shape's environment through sparknet's renderer at the lock's
     revision and drops the hand-written `NCCL_IB_HCA` and the inert `SPARKNET_ROCE_GID_INDEX`
     from `config/cluster.json`.
   - Check: on the production maps, every rank of `tp4-ring` renders the same values as before
     (the same `docker run` command). `tp3-triangle` differs only in `NCCL_IB_HCA`, from the
     prefix list to the exact list of the same four devices; confirm on the nodes that NCCL
     selects the same devices either way before the triangle next serves.
2. **Generate the node maps.**
   - `bin/spark fabric discover` writes `config/nodes.json` and the ring map.
   - Check: the generated files match the committed ones.
3. **Remove the levers.** Doctor rejects the fields above with a message naming the convention.
   Our maps carry none of them by then, so serving is unchanged.
4. **sparknet release.**
   - Strict `SPARKNET_ROCE_HCA`, no NCCL fallbacks, and a traffic class of 106 when switched.
   - It goes into the next candidate's upstream lock. Collective outputs must be bit-identical
     (`probe collectives`) and the release tests pass, as for any upstream change.
5. **Switched sites.** `docs/switched-fabric.md` states the switch requirement (DSCP 26 to a PFC
   priority). tsw2k's switch (#669) already maps DSCP 26 to priority 3. vedcsolution's (#662
   thread) is lossless on TC0, so it needs DSCP 26 mapped to that queue.

## Open questions

- **NCCL-only transports** (`nccl-switched`, `nccl-direct`): keep them as diagnostic modes
  outside the recipe, or remove them.
- **A second cable between the same pair** (`nccl_hcas`'s extra lanes): sparknet handles it with
  paths 3 and 4. Keep it as part of the convention, or reject it.
- **`ssh_host`, `transfer_host` and `mesh_*` in the local ring map:** these concern SSH, bulk
  copies and the one-shot proxy, not addressing. Each needs classifying as a convention, a
  derived value or a genuine site setting.
