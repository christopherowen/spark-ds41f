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

Each node then needs three fields the switchless maps do not use:

| Field | Meaning |
|---|---|
| `socket_interface` | The one wired interface that carries rendezvous (GLOO, `TP_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`) |
| `roce_peer_hcas` | Every peer rank, each mapped to the **same ordered pair** of HCAs |
| `roce_subnets` | One IPv4 `/24` per HCA in that pair, matching the peer's entry for the same lane |

`management_ip` may sit on any reachable network. It is not required to be on a
fabric lane: an API head on a management network is normal, and the bootstrap
path is qualified at run time, not statically.

## What the renderer does

- `NCCL_IB_HCA` becomes `=<pair>`. The leading `=` makes NCCL match device
  names exactly rather than by prefix, so a node selects only its own pair.
- Ring traffic controls are rejected, not ignored: `NCCL_ALGO`,
  `NCCL_MIN_NCHANNELS`, `NCCL_MIN_TRAFFIC_PER_CHANNEL`, `NCCL_THREAD_THRESHOLDS`
  and `NCCL_SWITCHLESS_BIDIRECTIONAL` cannot be set. A switchless setting left
  in a switched environment changes dispatch silently, so it is an error.
- `SPARK_DEPLOYMENT_TRANSPORT` names the deployment's transport. The collective
  probe reads it to keep a physical switched fabric distinct from sparknet's
  `direct` kernel mode, which is the same mode on both fabrics.

## Validation

`bin/spark doctor` rejects a switched map whose peers do not reuse one ordered
HCA pair, whose lanes disagree with the peer's subnets, whose lanes share a
subnet, or whose rendezvous interface is a loopback, container bridge, overlay
or wireless device. Two peers on the same HCA are rejected: a switched node
needs both paths of its pair.

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

### One switch port, two device paths

A host reaches its switch port through two HCA device nodes on the same physical
port — for example `rocep1s0f1` and `roceP2p1s0f1`, which are two PCIe paths to
one cable, not two rails. **Both must appear in the ordered pair**, and so in
`NCCL_IB_HCA`, or the node selects one and halves its fabric bandwidth. Measured
on the reference deployment, all-pairs host-memory traffic reached about
**196 Gb/s with both devices listed and about 112 Gb/s with one**.

This is the trap the pair validation exists for: a map that lists one device, or
two devices that belong to different ports, either runs at half rate with no
error anywhere or fails the "same ordered pair" check. Confirm both paths
carried writes during the probe rather than inferring it from a healthy link —
`--port-samples` reports per-HCA payload for exactly this reason.
