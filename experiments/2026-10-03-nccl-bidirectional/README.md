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

## Hardware results, 2026-10-03

Image `sha256:0fa46803915c3ee6c3c292cd58804c2257492d987838e14027fb56e985a89ac6`
was built from deployment `24ce29d` and verified identical on every node.
Both arms use this image: control disables the new policy, candidate enables
it. Control / candidate / control passed on all four ranks, followed by a
fourth launch retaining more channels for small messages. All exact-data,
CUDA-graph and boundary checks passed with zero tracked RDMA errors.

The logs show channels 0/1 clockwise and 2/3 counterclockwise. Physical-port
counter ratios confirm approximately 50/50 bulk transmission, replacing
the control's approximately 99.5/0.5 (the minority includes acknowledgements).
Small RoCEnante messages remain bidirectional in both arms.

NCCL's smallest reduce-scatter still selected too few channels and remained
approximately 95/5. `cluster-full-channels.json` adds the existing NCCL
setting `NCCL_THREAD_THRESHOLDS=-2 -2 -2 1 1 1`: leave Tree defaults intact,
lower Ring's channel-retention threshold. The smallest tested reduce-scatter
then uses both directions (approximately 76/24); larger cases approach 50/50.
NCCL chunk granularity prevents assuming exactly equal bytes for every size.
This setting is a separate fourth screen, not part of the original A/B/A.

Representative BF16 latencies in microseconds, using the slowest rank per
sample and then the median of five samples of 256 calls:

| Collective | Elements per rank | Control before / after | Reversed half | Retain channels |
|---|---:|---:|---:|---:|
| Reduce-scatter | 5,120 | 59.6 / 55.2 | 59.9 | 58.1 |
| Reduce-scatter | 30,720 | 96.3 / 92.5 | 85.5 | 95.9 |
| Reduce-scatter | 245,760 | 141.5 / 130.3 | 154.5 | 149.0 |
| Reduce-scatter | 1,048,576 | 346.7 / 357.9 | 372.2 | 350.9 |
| All-reduce | 2,097,152 | 337.0 / 344.9 | 347.6 | 360.0 |
| All-reduce | 5,242,880 | 802.8 / 797.4 | 828.6 | 833.4 |
| All-gather | 5,242,880 | 1,544.5 / 1,553.0 | 1,566.7 | 1,569.8 |

For all-reduce/all-gather the listed count is each rank's input; for
reduce-scatter it is each rank's output, with an input four times larger.
BF16 uses two bytes per element; the full results also include FP32.

**Decision: retain as an experiment.** Routing is now bidirectional, but
there is no general latency gain. Against the mean of the two controls, the
reversed-half screen ranges from approximately 9% faster to 14% slower on
NCCL cases; retaining channels ranges from 8% faster to 10% slower. In
particular, the 480 KiB-per-rank BF16 reduce-scatter remains slower in both
variants. These are screens, not confidence intervals or serving results.
Do not promote this as a performance improvement.

The GPU temperature peaks were roughly 56–57 C on dgx1/2 and 49–50 C on
dgx3/4 in both control and candidate. Peak GPU power also stayed close
(dgx2 approximately 21.3–21.6 W). The traffic rebalance did not remove that
pattern in these short probes, but they do not reproduce sustained serving
heat. A thermal cause has not been established.

Native receipts, per-thread/system CPU samples, GPU clocks/power and fan
RPM are in `hardware/runs.tar.gz`, checked by `hardware/sha256.json`.
The interface counters are retained by interface in the native reports.
`directional_tx_counter_sums` is for comparing direction shares: separate
PCIe function views may report the same physical-port counter, so do not
treat their sum as unique wire bytes or use it for a bandwidth claim.

```sh
python3 experiments/2026-10-03-ring4-bidirectional/summarize.py EXTRACTED_DIR nccl-control1 nccl-candidate1 nccl-control2 nccl-full-channels
```

No serving promotion occurred. The shared cluster window was released at
06:56:41 UTC, all four nodes idle and fan controllers active. Shared
production checkouts remain at `cfa6e5c`.
