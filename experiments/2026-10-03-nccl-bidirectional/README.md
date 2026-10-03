# Bidirectional NCCL bulk rings

Candidate, not promoted. This isolates NCCL channel direction after the
RoCEnante change in `../2026-10-03-ring4-bidirectional`.

The saved four-channel NCCL logs show four copies of `0 1 2 3`. In the pinned
NCCL source, `connectRings` connects node n to n+1 for every channel.
Increasing the channel count retains that direction. The NCCL graph-file
input describes local GPU/NIC paths; it does not replace this inter-node loop.

With `NCCL_SWITCHLESS_BIDIRECTIONAL=1`, the second half of the final channels
is reversed after channel duplication. At four channels, channels 0/1 run
clockwise and 2/3 run counterclockwise. This retains both NIC roots in each
direction when NIC selection alternates by channel; reversing odd channels
would instead pair one direction with each root. All links remain between
direct neighbours in cable/rank order.

The switch defaults off. It rejects unsupported configurations before
altering any ring: require three/four nodes, one rank per node, an even
channel count of at least two, unshared communication resources and existing
rings in ascending cable/rank order. Receiver and sender maps are reversed
together and then validated by NCCL's ordinary ring builder.

`cluster.json` and `control.json` use the same image, RoCEnante patch and
runtime settings. Their only runtime difference is this NCCL switch.
NCCL may use fewer channels for some message sizes: per-operation NIC
counters must demonstrate actual balance, not just the initialization log.

This changes floating-point reduction order in reversed channels. It does
not change precision or weights, but bitwise equivalence with the old rings
is not promised. Serving quality and determinism need their own validation
before promotion.

## Verification

```sh
bin/spark3 --cluster-config experiments/2026-10-03-nccl-bidirectional/cluster.json doctor
bin/spark3 --cluster-config experiments/2026-10-03-nccl-bidirectional/cluster.json build prepare --only nccl
python3 experiments/2026-10-03-nccl-bidirectional/test-policy.py
```

The sanitizer-backed CPU test executes the policy extracted from the actual
prepared C++ source, across every TP3/TP4 rank and 2/4/8/16 channels. It
checks direct adjacency, reciprocal links, complete rings and rejection
before mutation for malformed/unsupported configurations.

Hardware screen: compare control/candidate/control on the same four nodes,
covering all-reduce, all-gather and reduce-scatter, eager and CUDA graph
replay, both sides of custom/NCCL dispatch and large prefill payloads. Save
the ring logs, per-operation physical NIC counters, exact-data results,
numerical diagnostics and host telemetry. Both directions and both NIC
roots must carry bulk data; initialization messages alone are insufficient.
