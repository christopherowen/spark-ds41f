# Switched fabric deployments

The recipe describes two physical fabrics. A **switchless** deployment cables
each node to its neighbours (`oneshot-direct` for three nodes, `oneshot-ring4`
for four). A **switched** deployment cables every node to one switch port and
lets each node reach every peer directly.

A ring gives each node two cables; a switch gives it one port and independent
paths to all peers, so switched is not the obvious bandwidth choice. Measure it
rather than assume either way. On the reference four-node deployment a c1-c6
concurrency sweep (four fixed prompts, aggregate tokens per second, thinking
off, three repeats) averaged **195.7 tok/s across its 24 cells — a 1.031x
geometric mean (+3.1%) against the ring's 190.5**, with 23 of 24 cells inside
+-6%.

Read that as *no worse, possibly a few percent better*, not as a transport win.
It is not a controlled ring-versus-switched A/B: the two rows ran different
runtime builds, and 3% is inside the build-to-build spread this cluster shows
on the same method (one repeated cell disagreed with its own single-pass run by
12%).

Network-only numbers, for reference, are larger and easier to interpret: about
196 Gb/s combined all-pairs through the switch, against 112 Gb/s over one HCA
path, with every rank completing 420 writes on both HCAs. Those are fabric
figures and say nothing about model throughput.

Choose switched when the cabling is already in a switch, when peers must not
depend on rank order, or when the node count must change without re-cabling.

## Declaring one

Both parts are required: a transport alone leaves the fabric ambiguous, and a
topology alone has no runtime.

```json
{"fabric": {"transport": "oneshot-switched", "topology": "switched"}}
```

The node map follows sparknet's switched schema (sparknet's `docs/topology.md`), which
validates and renders it at the revision the upstream lock pins. Each node needs three
fields the switchless maps do not use:

| Field | Meaning |
|---|---|
| `management_interface` | The one wired interface that carries rendezvous (GLOO, `TP_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`) |
| `roce_hcas` | The node's rails in order: both device paths of its switch port |
| `roce_subnets` | One IPv4 `/24` per rail, the same rail subnet on every node |

One optional field says how the hosts mark RoCE packets for the switch:

| Field | Meaning |
|---|---|
| `roce_traffic_class` | The IP traffic class byte of every RoCE packet, `(DSCP << 2) \| ECN`, the same on every node; left out, both backends send 0 ([marking the lossless class](#marking-the-lossless-class)) |

`management_ip` may sit on any reachable network. It is not required to be on a
fabric lane: an API head on a management network is normal, and the bootstrap
path is qualified at run time, not statically.

## What the renderer does

- `NCCL_IB_HCA` becomes `=<rails>`, and `SPARKNET_ROCE_HCA` the rails. The
  leading `=` makes NCCL match device names exactly rather than by prefix, so a
  node selects only its own devices.
- Ring traffic controls are rejected, not ignored: `NCCL_ALGO`,
  `NCCL_MIN_NCHANNELS`, `NCCL_MIN_TRAFFIC_PER_CHANNEL`, `NCCL_THREAD_THRESHOLDS`
  and `NCCL_SWITCHLESS_BIDIRECTIONAL` cannot be set. A switchless setting left
  in a switched environment changes dispatch silently, so it is an error.
- `SPARK_DEPLOYMENT_TRANSPORT` names the deployment's transport. The collective
  probe reads it to keep a physical switched fabric distinct from sparknet's
  `direct` kernel mode, which is the same mode on both fabrics.
- The traffic class is sparknet's: at sparknet 0.5.0 the map's
  `roce_traffic_class` becomes `NCCL_IB_TC` (and `SPARKNET_ROCE_TRAFFIC_CLASS`
  under `oneshot-switched`); from the release after it, every switched fabric
  marks DSCP 26 with ECN, byte 106 ([fabric conventions](fabric-conventions.md)),
  and the field is retired.

## Validation

`bin/spark doctor` runs sparknet's map checks: rails named once, at most two,
with one subnet a rail across every node, and the same rail counts everywhere.
From the release after sparknet 0.5.0 it also rejects a rendezvous interface
that is a loopback, container bridge, overlay or wireless device, and rails
that share a subnet.

`nccl-switched` names plain NCCL with per-node HCA selection. An image that
replaces NCCL with sparknet cannot run it, and it requires
`--disable-custom-all-reduce`.

## Qualification scope

**Four nodes at TP4 is the only switched shape that has run on hardware.** A
three-node switched map is accepted by the tooling — the validation is written
for a clique of any size, and a triangle of peers is a valid one — but it is
**unqualified**: no three-node switched deployment has been run, so run the
collective probe before relying on one. Treat the four-node result above as the
only measured one.

## Qualifying one

Run the collective probe on every rank in one coordinated window with serving
stopped (`bin/spark topology probe`). It must agree on the deployment identity
before it touches a device, so a mismatched profile fails immediately instead
of at the end of a run. Pass `--port-samples` to record per-HCA payload and
per-port counters; the probe then fails if any HCA in the pair never carried
payload, which is how a dead path shows up before it stalls a real all-reduce.

A network-only aggregate is not a model-performance claim. Report switch
throughput and end-to-end serving numbers separately.

## Switch configuration

The fabric assumes a lossless RoCEv2 path: RoCEv2 with GID index 3, one `/24`
per lane, and jumbo frames end to end. The reference deployment ran **MTU 9000**
on both lanes and revalidated GID, MTU and routes on every host before launch.

The reference deployment used a four-port 200 Gb/s switch with one port per
node. **Its switch-side configuration is not recorded here**: no PFC or ECN
setting was captured for that run, so it is unknown whether a lossless class was
configured or whether the fabric worked on default drop-and-retry behaviour.
Qualify your own switch by setting a lossless class (PFC) or ECN marking, then
watching pause and discard counters while a collective runs. If you measure the
defaults, record that — "worked without PFC at four nodes" is itself useful.

### Marking the lossless class

A switch sorts RoCE packets into its classes by the DSCP in their IP header, and
the sender chooses that DSCP: every queue pair carries one traffic class byte,
DSCP in the upper six bits and ECN in the lower two. NCCL takes it from
`NCCL_IB_TC`, sparknet from `SPARKNET_ROCE_TRAFFIC_CLASS` (falling back to
`NCCL_IB_TC`), and both default to 0, the best-effort class. A lossless class on
the switch therefore protects nothing until the hosts mark their packets with its
DSCP. Unmarked, the collectives ride the default class, which drops under incast,
and an all-reduce is incast by construction: every peer sends to one port at once.

The hosts mark DSCP 26 with ECN-capable transport, ECT(0), `(26 << 2) | 2 = 106`
([fabric conventions](fabric-conventions.md)); at sparknet 0.5.0, set
`roce_traffic_class` to 106 on every node. The host has to
agree as well: the NIC maps that DSCP to a priority with PFC on it, on both
interfaces of the pair, for example

```sh
sudo dcb app add dev enp1s0f1np1 dscp-prio 26:3
sudo dcb pfc set dev enp1s0f1np1 prio-pfc 3:on
```

and the switch maps the same DSCP to its lossless queue. `dcb` state does not
survive a reboot by itself; apply it from a boot-time unit.

Losses in the wrong class need not show on the switch, so check the hosts. While
the collective probe runs, `roce_adp_retrans` and `out_of_sequence` under
`/sys/class/infiniband/<hca>/ports/1/hw_counters/` should stay put, and the
marked priority's counters (`ethtool -S <interface> | grep prio3` for priority
3) should carry the traffic.

**Externally reported, not qualified here:** a switched GB10 site (MikroTik
CRS804-4DDQ, PFC on DSCP 26 to priority 3, MTU 9000) measured `all_reduce_perf`
over one PCIe path at the default traffic class with about 1,000
`out_of_sequence` packets per run and bus bandwidth falling from 95 Gb/s at
64 MB to 78 Gb/s at 512 MB, while the switch showed no drop, overflow or FCS
error. With `NCCL_IB_TC=106` the same runs had no retransmissions and held
88-92 Gb/s; both PCIe paths of the port, marked, reached 174-182 Gb/s
(2026-10-17).

### One switch port, two device paths

A host reaches its switch port through two HCA device nodes on the same physical
port — for example `rocep1s0f1` and `roceP2p1s0f1`, which are two PCIe paths to
one cable, not two rails. **Both must appear in `roce_hcas`**, and so in
`NCCL_IB_HCA`, or the node selects one and halves its fabric bandwidth. Measured
on the reference deployment, all-pairs host-memory traffic reached about
**196 Gb/s with both devices listed and about 112 Gb/s with one**.

This is the trap the pair validation exists for: a map that lists one device, or
two devices that belong to different ports, either runs at half rate with no
error anywhere. Confirm both paths
carried writes during the probe rather than inferring it from a healthy link —
`--port-samples` reports per-HCA payload for exactly this reason.
