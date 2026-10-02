# Four-path opposite-peer forwarding

Four paths improve large forwarded collectives reproducibly, but do not beat the
CPU relay at the largest sizes. Rotating submission order adds no consistent
benefit. Retain fixed-order four-path forwarding as a candidate, not a promotion.
No model-serving container was started; these results do not establish TPS,
TTFT or model-output determinism.

## Identity and implementation

Base: `0fbd7b7`, with main's thermal-baseline tooling merged before testing.
All seven launches use clean isolated checkouts at published `27a85c1` and one
identical image on all four nodes:

- Tag: `vllm-ds41f-kkref:04c30fa98e79-r5o-roce-mesh4-fourpaths-v1`
- Image ID: `sha256:9671c96903dc7cc5a9bbae4249e7b7aa9efa550980f1dab5fa142dac3ebe842d`
- B12X patch head: `6056b0a181a5388a7a77e4192121d89a0a42d426`
- B12X tree: `2077a53bfb2a993b385f1a1bd9799b0a89694976`
- Base image: `sha256:aad8a74089ff379f5bc7905e86f9c7e2c053396d0a4027039e869c505ca7621b`
- Kernel: `7.0.0-1019-nvidia-64k`; NVIDIA driver: `580.178.04`.

The source-verified image overlays B12X on the installed r5o base; vLLM and NCCL
source labels match the base. The rejected direct-first wave patch is excluded.
The native source branch is `mesh4-fourpaths`; `b12x/0008-four-path-mesh.patch`
is its exported change and the source manifest verifies its exact head/tree.

Split the opposite peer's payload equally over both PCIe-root interface stripes
through both neighbors. Direct-neighbor payloads retain two paths. A separate
option rotates the starting interface and direct/opposite posting priority,
offset by rank and sequence, without a completion barrier or changing the
payload split. ABI 8 includes this geometry and setting in rank negotiation.
The GPU skips unused neighbor flag slots and waits for all four opposite-peer
flags. Numerical operations and reduction order are unchanged.

| Profile | Opposite paths | Posting order |
| --- | ---: | --- |
| `cluster-control.json` | 2 | original |
| `cluster-four.json` | 4 | original |
| `cluster-rotate.json` | 4 | rotating |
| `cluster-relay.json` | CPU relay | original |

`cluster.json` uses the public example site for source preparation and CI.
Actual profiles select the ignored four-node site map through `nodes_config`.
All arms use the same image and applied 8192-entry hairpin queue setting;
relay installs no forwarding rules. Queue count stays four.

## Method and qualification

Tests before hardware: 171 repository tests and 26,000 native C simulated
collectives under ASan/UBSan, covering mixed two/four-path geometry, explicit
rotation-order assertions, delayed DMA, uneven small chunks, sequence wrap,
malformed geometry and stop/error handling. Fresh patch preparation and CI pass.

The hardware order is control, four, rotating, rotating repeat, four repeat,
control repeat, then CPU relay. Every launch starts with all four nodes' hottest
thermal zones below 55 C. Run 01 needed five seconds of pre-cooling; each node's
fan service was restored before measurement. Other cooling receipts are retained.
Each container is bounded at 12 GiB and the coordinated runner has a timeout.

Each launch checks exact BF16/FP32 all-reduce, all-gather and reduce-scatter:
19 dtype/length cases per rank, eager execution and four changing-input graph
replays, uneven small inputs, and both sides of the custom/NCCL dispatch
boundaries. All seven launches pass on all four ranks. Runtime statistics
confirm two/four path slots and rotation off/on as intended.

The timing matrix is four lengths (5,120, 30,720, 245,760 and 1,048,576 elements)
for both dtypes and all three operations. A graph contains 16 calls, replayed
16 times per sample; five samples follow warmup. Take the slowest rank per
sample, then the median. Compilation, verification, barriers and counter reads
are outside CUDA-event timing. Ranges below are the two launch medians, not
confidence intervals. Reduce-scatter always uses NCCL; FP32 4 MiB all-reduce
also uses NCCL, so those cells are controls rather than four-path measurements.

RDMA error and physical-port byte/drop counters bracket every timed case, with
CPU barriers separating the windows. Whole-launch counters include startup,
correctness and warmup. These are attribution windows, not packet traces.
The base image lacks ethtool/libmnl: the runner mounts the host binaries
read-only in every arm. Their hashes match on all four hosts and are recorded.

## Latency results

BF16 all-reduce, microseconds per call; lower is better:

| Payload per rank | Two paths, 01/06 | Four paths, 02/05 | Rotating, 03/04 | CPU relay, 07 |
| --- | ---: | ---: | ---: | ---: |
| 10 KiB | 15.34–16.28 | 16.19–16.54 | 16.42–17.43 | 17.97 |
| 60 KiB | 24.35–26.01 | 24.34–24.76 | 23.99–24.87 | 29.33 |
| 480 KiB | 123.12–129.54 | 98.00–100.01 | 95.33–96.02 | 92.99 |
| 2 MiB | 554.69–554.74 | 416.86–418.89 | 421.34–424.12 | 314.98 |

BF16 all-gather, microseconds per call:

| Input per rank | Two paths, 01/06 | Four paths, 02/05 | Rotating, 03/04 | CPU relay, 07 |
| --- | ---: | ---: | ---: | ---: |
| 10 KiB | 15.60–16.35 | 16.82–17.59 | 15.83–17.23 | 20.82 |
| 60 KiB | 28.44–29.31 | 26.33–26.74 | 25.99–27.39 | 29.75 |
| 480 KiB | 133.84–137.11 | 106.34–107.48 | 106.83–110.40 | 102.69 |
| 2 MiB | 562.59–562.78 | 481.80–486.06 | 487.82–488.72 | 353.76 |

Four fixed paths reduce 480 KiB all-reduce latency by 19–24% and 2 MiB by
24–25% against this session's controls. FP32 960 KiB all-reduce is also faster
(171–179 us versus the controls recorded in `hardware/summary.json`). Small
messages show little gain and some cost: for example 10 KiB all-gather grows
from 15.60–16.35 to 16.82–17.59 us. Rotation helps one size but loses elsewhere;
there is no basis to enable it globally. CPU relay remains the large-payload
control to beat. The full 24-cell matrix for every launch is in the summary.

## Counters and what was balanced

Whole-launch deltas summed over all 16 functions:

| Run | Arm | `rx_out_of_buffer` | `roce_adp_retrans` | `packet_seq_err` |
| --- | --- | ---: | ---: | ---: |
| 01 | two paths | 129,818 | 381 | 4,013 |
| 02 | four paths | 51 | 0 | 8 |
| 03 | rotating | 42 | 0 | 8 |
| 04 | rotating repeat | 40 | 0 | 8 |
| 05 | four paths repeat | 51 | 0 | 5 |
| 06 | two paths restored | 115,789 | 274 | 3,467 |
| 07 | CPU relay | 0 | 0 | 0 |

The few remaining timed drops/sequence errors in four-path arms occur at the
FP32 4 MiB all-gather. None of their timed BF16 cases increments these counters.
Zero retransmission counter increments are not a claim of zero packet loss.
Latency still trails relay even in cases without any observed errors.

**Correction to the original link-load explanation:** the two PCIe-root
interface stripes share a physical cable. The original 3:1 modeled imbalance
is over 16 directed **interface paths**, not 16 independent physical links.
Aggregating the two stripes gives equal payload work on every directed cable
in both the old and new maps. Physical TX/RX counters are duplicated between
the two functions sharing each port; do not sum those duplicates as independent
wire bytes. Per-function buffer counters remain separate.

The physical-port measurements confirm that the control's cable traffic was
already balanced. Four paths spread forwarding over 16 hardware rules instead
of eight, with approximately half as many bytes per rule in the same workload.
The interface-path work model becomes uniform, and the measured drop reduction
supports relief of forwarding-queue pressure. It does not demonstrate additional
physical-link bandwidth. Exact attribution of the remaining latency to PCIe,
NIC scheduling or burst timing needs a separate experiment.

## Decision and remaining scope

Retain four fixed paths as an experimental improvement; leave rotation off.
Do not promote it as a universal replacement: small-message overhead, the
remaining largest-gather drops and the CPU-relay advantage remain. The next
useful comparison is a bounded outstanding-byte scheme or size-based choice
between these already measured transports, with the same correctness gates.
A separately tuned NCCL baseline would also be useful: the current profile
forces Ring and one channel for switchless qualification.

Serving TPS/TTFT, full TP4 startup, persistent forwarding setup, reboot behavior,
long-duration stress and model numerical equivalence are not qualified here.

## Run ledger and restoration

- The initial hold acquisition refused another session's reservation; no host
  changes occurred. Our owned window began at 22:05:02 UTC on 2026-10-02.
- Local validation initially caught a missing `paths` field in a test fixture
  and a real head address inherited by the public example. Both were corrected;
  receipts and passing reruns are in `local-validation/`.
- Isolated checkouts were synchronized to published `27a85c1`. All four hosts
  were idle before mutation; shared deployment checkouts were clean at `cfa6e5c`.
- Applied 8192 to all 16 data functions, including successful driver reloads
  (reload count 4 to 5) and four clean post-change NIC doctors. Management
  networking was not changed.
- The first image transfer began before export finished and failed with a
  truncated-stream error. No GPU probe used that attempt. After export exited
  successfully, the archive passed integrity testing and a repeated transfer
  verified the identical image ID on every node. Both attempts are preserved.
- Runs 01–07 completed with all ranks successful, in the order listed above.
- Restored all 16 functions to entry 1024 with confirmed reload counts advancing
  from 5 to 6. Final NIC inventory equals the entry inventory; four NIC doctors
  pass; no owned routes, filters, markers, probe containers or GPU clients remain.
  All fan services are active. Shared checkouts remain clean at `cfa6e5c` and
  unrelated stopped/Created containers are preserved exactly. The entry state
  was idle and was restored without starting serving. Released at 22:24:27 UTC.
  The isolated qualification checkouts remain at `27a85c1`.

Before/after available memory is roughly 117.6–118.3 GiB per host; this is not
peak-memory measurement or an estimate of the isolated queue-memory cost.
No model weights or KV cache were allocated. `hardware/runs.tar.gz` retains
all attempts, commands, raw logs, image receipts, port/RDMA counters, memory
snapshots, cooling records, NIC changes and cleanup. The runner and summary
scripts are archived as provenance with site-specific paths, not general
production deployment commands.
