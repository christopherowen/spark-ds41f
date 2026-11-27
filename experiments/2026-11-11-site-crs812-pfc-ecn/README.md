# External site receipt: lossless RoCE on a MikroTik CRS812 (RouterOS 7.23.5)

A four-Spark switched site made the switch's lossless class actually work, then replayed the same
spark-ds41f requests before and after. It is a receipt from another site, not a promotion request:
nothing in `config/` changes.

## Site

| | |
|---|---|
| Nodes | 4 x DGX Spark (GB10), one ConnectX-7 port per node into the switch, both PCIe device paths of that port in `roce_hcas`, one /24 per path, RoCE v2 GID index 3, MTU 9000 |
| Switch | MikroTik CRS812-8DS-2DQ-2DDQ (ARM64), one 200 Gb/s breakout port per node (`200G-baseCR4`, auto-negotiation off), L2MTU 9216 / MTU 9000 on the fabric ports; fabric ports isolated from management |
| Firmware | before: `routeros` and `switch-marvell` 7.20.8; after: both 7.23.5 (long-term). RouterBOOT left at 7.20.8 |
| Runtime | spark-ds41f at upstream `c460d1b4` with the frozen 1107 lock, `oneshot-switched`, four nodes, 262,144 context, 2 sequences, KV 3 GiB, started with `--unqualified-shape`; the renderer marks `NCCL_IB_TC` / `SPARKNET_ROCE_TRAFFIC_CLASS` = 106 (DSCP 26, ECT(0)) |
| Weights | DeepSeek-V4.1-Flash-1017 `1bae0e10` |

## What did not work on 7.20.8

1. **The PFC menus accepted a configuration, but no pause frames were sent.** Readback flags alone were
   not proof; counters were. MikroTik's [7.23 release notes](https://forum.mikrotik.com/t/v7-23-stable-is-released/270721/1)
   add PFC and ECN for CRS8xx.
2. **Receive PFC needs an explicit rate on the lossless queue, and 7.20.8 refused anything above 100G.**
   The site did not cap its 200G ports at 100G to get around it.
3. **802.3x global pause** (rx and tx on the four fabric ports, hosts honouring pause) was configured and read
   back at 200G, but a steady fan-in probe still dropped and the hosts received no pause frames. Rejected.
4. **Transmit-only PFC** (`rx=no tx=yes`, which avoids the queue-rate requirement): a short screen still
   showed about 4.2 million switch drops. Rejected.

All of these were reverted before the upgrade.

## Configuration

Hand-written from the site's scripts, with the site's port names replaced by placeholders:

- [`switch/routeros-7.23.5-roce-lossless.rsc`](switch/routeros-7.23.5-roce-lossless.rsc): jumbo frames, the
  lossless class, DSCP map, ECN, PFC and queue rate. Apply in Safe Mode with serving stopped.
- [`switch/routeros-7.23.5-roce-lossless-revert.rsc`](switch/routeros-7.23.5-roce-lossless-revert.rsc):
  removes the QoS changes, keeping 7.23.5.
- [`host/roce-lossless-qos.sh`](host/roce-lossless-qos.sh): per-host DSCP trust, `dscp-prio 26:3 48:6`,
  PFC on priority 3, global pause off, on `enp1s0f0np0` and `enP2p1s0f0np0` (port 0, both paths, per
  [fabric conventions](../../docs/fabric-conventions.md)); installs a oneshot systemd unit and a
  NetworkManager dispatcher hook so it survives reboots and link flaps.
- [`host/roce-lossless-qos-revert.sh`](host/roce-lossless-qos-revert.sh): undoes the host side.

In short:

| | Switch (7.23.5) | Each host |
|---|---|---|
| Lossless class | traffic class 3, `lossless-buffers=auto`, shared and multicast buffers `auto` | priority 3 |
| Classification | `trust-l3=keep`; DSCP 26 -> TC3, DSCP 48 (CNP) -> TC6; profile `automap=no` | `mlnx_qos --trust=dscp`; `dcb app ... dscp-prio 26:3 48:6` |
| PFC | rx and tx on TC3, fabric ports only | `mlnx_qos --pfc=0,0,0,1,0,0,0,0` |
| ECN | `ecn=yes`, `wred=no` on the default tx-manager's queue 3 | (marking comes from traffic class 106) |
| Queue rate | `egress-rate-queue3=200G` per fabric port (required for receive PFC) | - |
| Global pause | off | `ethtool -A <if> rx off tx off` |

**DSCP 0.** DSCP 0 is only needed for engines that send traffic class 0. spark-ds41f marks DSCP 26,
so it does not need that mapping, and putting DSCP 0 in the lossless class makes all unmarked traffic
on those ports lossless too. The scripts leave it out and show where it would go.

## Results

### spark-ds41f, before and after (2026-11-11)

The original request bodies, replayed after a fresh model boot: five repeats per cell, C1, temperature 0,
thinking off, 512 output tokens for prose and code, 16 for prefill, real text with unique salts, 0 cached
tokens on every request. Prefill is the server's prefill timing. This
site's own client, not `bin/spark bench`.

| Workload | Before, 7.20.8 (tok/s) | After, 7.23.5 + QoS (tok/s) | Change |
|---|---:|---:|---:|
| Prose decode | 77.9 | 77.3 | -0.7% |
| Code decode | 94.6 | 93.6 | -1.1% |
| Cold prefill, 4K | 5,366.5 | 6,116.7 | +14.0% |
| Cold prefill, 32K | 8,446.6 | 8,711.0 | +3.1% |
| Cold prefill, 128K | 8,528.7 | 8,570.6 | +0.5% |

Decode ranges overlap (prose 74.6-80.0 before / 76.3-79.4 after; code 85.1-95.2 / 92.7-100.5). The 4K and
32K prefill ranges do not overlap in these five samples; the 128K ranges do.

| Counter, whole run | Before | After |
|---|---:|---:|
| `roce_adp_retrans`, summed over 8 HCAs | 1,425 | 0 |
| Packet-sequence / `out_of_sequence` increments, 8 HCAs | not compared | 0 |
| Switch queue drops | not recorded | 0 |
| Pause frames, switch tx = host rx | - | 744 |
| CNPs sent (counted once per physical port) | 0 | 12,125 |

### Fan-in probe

A steady six-stream fan-in probe (`perftest`, host memory, serving stopped) was unchanged in bandwidth
(195.4 Gb/s summed into one 200G port) and latency (2.50 us idle both times; 9.95 us before and 9.97 us
after under load), and its drops fell from 1,064 to 0. Steady fan-in loses far less than bursty prefill,
so a bandwidth test alone would not have found this.

## Limits

- Firmware, buffers, classification, PFC and ECN changed together; no factorial attribution.
- One boot per side, sequential before/after, no randomized switchback; boots and thermal history differ.
- Five repeats at C1 only; no concurrency sweep, soak or agent workload.
- The site's own client, not `bin/spark bench`; no fan-floor control.
- Measured on a site branch over `c460d1b4` (1107 lock), not on 1120 or 1126. The site has run 1120 on this
  configuration since, but has no before/after for it.

## Files

- `base.json`, `candidate.json`, `results.json`, `decision.md`
- `switch/`, `host/`: the configuration above
- `runs/README.md`: where the raw receipts are
