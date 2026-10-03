# Four-node forwarding queue capacity and send order

The default 1024-entry hairpin queues are a demonstrated cause of the large
forwarded-collective slowdown. Applying 8192 and restoring 1024 reproduced the
improvement and regression, respectively. The larger queues reduce packet loss
substantially, but do not eliminate it or match the CPU relay at large sizes.
A direct-first send-wave candidate was slower and is rejected for promotion.

This investigation left production configuration and shared deployment checkouts untouched.
All work used isolated qualification checkouts and bounded collective probes;
no model-serving container was started. Host settings were restored to their
entry values after testing. No model TPS, TTFT or prefill result is claimed.

## Identity and method

Base: `bc67e13`. The queue helper and initial profile were published as `f50502a`;
the send-wave candidate and per-case counter sampler as `e834d00`.

| Image | ID | B12X tree |
| --- | --- | --- |
| Original mesh/relay | `488fed96fecec12e60a32a75f387e057bbf56da384098ca6efb6757869734c2d` | `f5b9429596f789ed88952860b43c97823bc44679` |
| Send-wave experiment | `8a70ccd331cb0e751e8e6183676de64860ccbd5dbfc10e94a7a0a3fc66f3f520` | `5553fe9cd59efad1b52ac879d08d65eade1c6a2a` |

IDs are SHA-256 Docker image IDs. Each image was identical on all four nodes.
The second image overlays only verified B12X source on the installed r5o base
`aad8a74089ff379f5bc7905e86f9c7e2c053396d0a4027039e869c505ca7621b`.
vLLM and NCCL source labels match the base. This is a transport-only image build,
not a rebuild of the full runtime foundation.

Four connected Sparks, kernel `7.0.0-1019-nvidia-64k`, driver `580.178.04`.
The selected site maps follow the ignored `nodes.json` standard and retain the
actual physical rank order, addresses, MACs, PCI functions and GID index 3.
Qualification checkouts were clean at the same published revision on all nodes.
Shared deployment checkouts entered clean at `4449d6e`. Another session advanced
all four to `cfa6e5c` during the hold (benchmark thermal-baseline tooling). They
were clean at exit and were left intact. Our isolated checkouts remained pinned
at the intended published revision throughout each comparison.

The queue experiment changes only `hairpin_queue_size` on all 16 data functions,
from 1024 to 8192; queue count stays four. Each change is followed by successful
`driver_reinit`, with an increasing reload counter and a post-transition check
of addresses, GIDs, MTU, steering and hardware-offload settings. A configured
`driverinit` value by itself is not proof of the active value. Management
networking is outside this procedure.

The helper requires idle GPUs, containers and RDMA user contexts, no TC filters,
and an exact before-inventory match. The coordinator checks the owned hold and
all hosts before the first mutation. The bounded forwarding runner adds and
removes only its own routes, hardware TC rules and marker processes.

The send-wave experiment changes only `B12X_ROCE_MESH_WAVE_BYTES` between zero
and 131072 in the same new image, with queues fixed at 8192. Above the threshold,
it posts direct-neighbor writes, waits for their acknowledged completions,
then posts the opposite-peer writes. Below the threshold it keeps the original
order. No GPU arithmetic, reduction order, buffer layout or kernel changes.
ABI 7 includes the threshold in the rank handshake and rejects mixed settings.
The proxy reports 9,485 wave activations on each rank in both enabled runs,
versus zero in both disabled runs.

Each launch checks exact BF16 and FP32 all-reduce, all-gather and reduce-scatter
results, eager calls, four changing-input graph replays, and the custom/NCCL
boundaries at 2 MiB all-reduce and 4 MiB all-gather. There are 19 dtype/length
cases per rank. Reduce-scatter always uses NCCL; the 4 MiB FP32 all-reduce also
uses NCCL. All nine launches passed on all four ranks.

The latency screen has four lengths (5,120, 30,720, 245,760 and 1,048,576 elements)
for each dtype and operation. A graph has 16 calls, replayed 16 times per sample;
five samples follow warmup. Report the slowest rank for each sample, then the
median. Compilation, counter reads and output verification are outside the
CUDA-event timing. Ranges below are launch medians, not confidence intervals.
Starting with run 05, RDMA counters are additionally sampled around each timed
case. Whole-launch port and RDMA counters cover setup, correctness and warmup too.
Per-case counter windows are approximate attribution windows, not packet traces.

## Results

BF16 all-reduce, microseconds per call; lower is better:

| Payload per rank | 1024 queues, runs 01/04 | 8192 queues, runs 02/03 | Same new image, waves off, 05/08 | Waves on, 06/07 | CPU relay, 09 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 KiB | 16.2 | 17.4–17.9 | 16.0–16.6 | 16.5–16.9 | 17.9 |
| 60 KiB | 25.5 | 26.5–26.6 | 27.8–28.0 | 26.0–26.7 | 31.4 |
| 480 KiB | 312.3–325.1 | 127.0–129.1 | 126.7–128.3 | 137.2–137.9 | 93.4 |
| 2 MiB | 964.1–982.0 | 540.4–569.7 | 540.2–543.6 | 624.7–649.1 | 313.2 |

Whole-launch deltas, summed over all 16 functions:

| Run | Arm | `rx_out_of_buffer` | `roce_adp_retrans` | `packet_seq_err` |
| --- | --- | ---: | ---: | ---: |
| 01 | 1024 | 3,367,984 | 20,905 | 64,172 |
| 02 | 8192 | 111,995 | 352 | 4,025 |
| 03 | 8192 repeat | 124,640 | 374 | 4,173 |
| 04 | 1024 restored | 3,014,925 | 20,297 | 65,774 |
| 05 | 8192, new image, waves off | 141,587 | 448 | 4,387 |
| 06 | 8192, waves on | 378,010 | 552 | 4,999 |
| 07 | 8192, waves on repeat | 342,671 | 463 | 5,018 |
| 08 | 8192, waves off restored | 106,099 | 256 | 3,842 |
| 09 | 8192, CPU relay | 0 | 0 | 0 |

The buffer-overflow increments occur on forwarding ingress functions. Queue
size changes their rate and the latency reproducibly; this identifies buffer
pressure and packet loss rather than relying only on sequence-error counters.
Successful reliable-RDMA recovery explains why exact output checks still pass.
TC-action drop counters alone missed this loss in the earlier qualification.

At 8192 with waves off, most retransmissions occur in the 4 MiB FP32 all-gather:
435 of 448 in run 05, and the corresponding per-case details are preserved for
run 08. That case takes 1,172–1,241 us with waves off and 1,477–1,519 us with
waves on. The 480 KiB BF16 cases have no sampled retransmission or sequence-error
increments in runs 05–08, despite remaining slower than relay. Packet loss
therefore does not explain the whole residual latency gap.

The route map distributes work unevenly across PCIe-root interface paths.
`link-load.py` derives payload work from the forwarding plan: for an equal
all-to-all payload of N bytes, the 16 directed interface paths carry between
0.5N and 1.5N, averaging N. Four carry two forwarded half-payloads plus a direct
half-payload; four carry only the direct half. This is a 3:1 **interface-path**
imbalance, not a measurement of utilization.

Correction from the subsequent [four-path experiment](../2026-10-02-mesh4-fourpaths/):
the two PCIe-root interfaces share a physical cable. Summing both interfaces
on each directed cable gives 2N for every cable already in this control.
Physical-port byte counters confirm balanced cable traffic. The model therefore
identifies uneven interface/forwarding-queue work, not unused cable bandwidth.
Local direct-send completion does not ensure that every other rank has stopped
direct traffic. The measured send-wave regression rejects that specific remedy;
it does not identify the exact latency share due to burst timing, interface
imbalance or NIC scheduling.

Decision: retain 8192 as the queue-capacity candidate; do not enable this
send-wave patch in production. The next transport experiment should balance
opposite-peer traffic across the available interface paths, with physical-port byte
and drop counters, before another model-serving benchmark. Keep the CPU relay
as the measured large-payload control. A hybrid dispatcher, new thresholds,
persistent NIC setup, reboot qualification and full TP4 serving remain separate
unmeasured changes.

## Run ledger, memory and cleanup

- Entry: all four hosts idle; no serving changes. The owned investigation hold
  began at 21:04 UTC on 2026-10-02.
- Run 01: original image and 1024 queues.
- Change 01: all 16 functions to 8192; post-change NIC doctor passed on all nodes.
- Runs 02–03: unchanged workload and image, 8192.
- Change 02: preflight refused while the previous benchmark container was still
  exiting. **No mutation occurred.** Change 02b retried after completion and
  restored all 16 functions to 1024.
- Run 04: restored control reproduced the loss and slowdown.
- Change 03: all 16 functions to 8192; NIC doctor passed on all nodes.
- Candidate build/distribution: one source-verified image, identical ID on all
  nodes. Runs 05/06/07/08 are off/on/on/off in that image. The enabled variant
  regressed; all results are retained.
- Run 09: CPU-relay control at the same applied queue size. Its profile's site
  file describes the entry mesh setting; relay does not install hairpins or
  validate that unused field. Captured devlink values record the actual 8192
  state. No loss or retransmission increments occurred.
- Change 04: restore all 16 functions to entry 1024, including driver reloads.
  Final audit requires exact entry NIC inventory, no owned routes/filters/marker
  processes/test containers and idle GPUs. All checks passed. Shared checkouts
  were clean but had advanced externally as described above; no rollback of that
  work was attempted. The owned hold was released at 21:37:29 UTC.

Each probe container is bounded at 12 GiB. Before/after `MemAvailable` stays
between 117.70 and 118.51 GiB per host across these launches. These snapshots
are not peak-memory measurements or an isolated estimate of queue-memory cost;
no model weights or KV cache were allocated.

Repository tests: 165 pass. The new B12X patch series applies in a fresh prepared
tree with its expected head and tree. The real C proxy passes 26,000 simulated
collective rounds under ASan/UBSan, including delayed delivery, one/two stripes,
sequence wrap, threshold mismatch rejection and stop/error handling. The new
same-image off/on GPU comparisons validate activation and exact output.

`hardware/runs.tar.gz` preserves every run, transition, refused transition,
command, image build/distribution log, raw port/RDMA counter, memory snapshot,
NIC doctor result and cleanup receipt. Unrelated VPN-interface lines are omitted
from the entry inventory text. `hardware/summary.json` is generated by
`hardware/summarize.py.txt`; per-case counters are null for runs before the
sampler was added. The orchestration scripts are archived as provenance, with
explicit site paths; they are not general-purpose deployment commands.
