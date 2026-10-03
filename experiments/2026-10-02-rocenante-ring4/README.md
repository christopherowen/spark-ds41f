# RoCEnante on a four-node switchless ring

Status: CPU relay now passes exact hardware GPU collective and CUDA graph checks
on all four Sparks, with two latency-screen launches. The built mesh candidate
includes this relay mode unchanged and supplies the common image for both arms.
See the [hardware session](../2026-10-02-rocenante-mesh4/README.md#four-node-hardware-session-2026-10-02)
for image identity, raw evidence and remaining serving gates. The promoted
configuration remains TP3; full TP4 model serving is still unqualified.

Deployment base: `b643f1654916eb2ad69892b5f4e75cdd45227b4d`.
B12X base: `f8069b2c0be1311df3b112591c6b8876a843f8be` plus the production
0001–0005 series, ending at `bb40849faeede0330ed0ee4f52188a58177121cb`.
Candidate head and tree are recorded in `source.json`. The independent patch is
`b12x/0006-rocenante-ring4.patch`; upstream submission is pending.

## What changes

The intended variable is the small-collective transport on an otherwise matched
four-node configuration: NCCL-only versus RoCEnante relay plus NCCL. Comparing
against the production three-node cluster also changes TP size and hardware,
so it cannot isolate relay performance.

| Mode | Connected peers | Payload delivery | GPU arithmetic |
| --- | --- | --- | --- |
| Existing `direct` | Every other rank | Direct RDMA writes | Existing kernels |
| New `ring4` | Previous and next rank | Direct to neighbours; one host relay to opposite | Same kernels and source-rank ordering |

For physical cable order `0—1—2—3—0`, rank R writes its staged payload to both
neighbours. It then waits for **every stripe** of R−1's payload and forwards
that receive slot to R+1. The destination indexes it as source R−1, not source R.
Thus rank 0 reaches rank 2 through rank 1, and each receiver still sees one slot
for every source. There is no QP to the opposite rank and no host reduction.

This preserves all-reduce and all-gather buffer layouts and their GPU reduction
order. It does not imply identical model outputs between TP3 and TP4. Existing
size/dtype eligibility and NCCL fallbacks remain; reduce-scatter uses NCCL.
The relay adds a second network hop for the opposite rank and CPU polling.
Its collective latency is now screened; model throughput remains unmeasured.
Clockwise traffic carries
two payloads per operation, counterclockwise traffic one; balancing that load
is a possible later optimization, not part of this candidate.

## Ordering and buffer lifetime

Each stripe is an RC RDMA data write followed by an inline sequence flag on the
same QP. The MR does not enable relaxed ordering. Both the GPU consumer and the
relay wait for all stripe flags. After the CPU observes those flags it uses an
AArch64 outer-shareable load barrier (`dmb oshld`), following
[rdma-core's device-observation convention](https://github.com/linux-rdma/rdma-core/blob/master/util/udma_barrier.h),
before posting the forwarding DMA. Real NIC/GPU coherence remains a hardware
qualification requirement; the CPU simulator cannot prove it.

No third staging buffer is allocated. For source A, relay B and opposite C:

1. B forwards A's receive slot for operation N to C.
2. A cannot reuse that slot for N+2 until A completes N+1.
3. A's completion of N+1 requires C's N+1 contribution.
4. C cannot publish N+1 until consuming N, including the relayed A payload.
5. C's receipt means B's forwarding DMA has already read the source bytes.

This relies on the existing serialized collective protocol and its two slots.
It does not support independent unsequenced operations sharing one runtime.
The existing pending-doorbell catch-up handles up to two operations when a
proxy is descheduled. Relay waits drain completions, honour shutdown and GPU
poison, reject a newer sequence in the same slot, and time out after five
seconds. Payload sizes exceeding their slot are rejected before posting.

Proxy ABI 5 exchanges topology, rank, world size, stripe count and slot size.
Every record is validated before connection, including the opposite rank.
Mixed direct/relay modes and incompatible layouts fail initialization.
The source-content cache key forces a new proxy build.

## Buildable candidate

`cluster.json` selects its own `upstreams.lock.json` and `source.json` through
`upstreams_config`. Production locks and patch series remain unchanged. The
candidate tag is `vllm-ds41f-kkref:04c30fa98e79-r5o-roce-ring4-v1`; it has **not**
been built. Doctor requires the relay capability in the selected source
manifest and a matching expected B12X image tree. Image preflight checks that
label against the actual image.

The candidate uses the documentation-address node map in
`config/examples/nodes-ring4.json`. Copy that map into this experiment, fill in
the four actual hosts/interfaces/subnets, and change `nodes_config` to the copy.
Keep ranks in cable order and use the same one or two stripes on every edge.
The candidate starts with `deployment.launch_enabled=false`.

Source-only validation (no Docker or GPU):

```sh
bin/spark3 --cluster-config experiments/2026-10-02-rocenante-ring4/cluster.json doctor
bin/spark3 --cluster-config experiments/2026-10-02-rocenante-ring4/cluster.json build prepare --only b12x
python3 experiments/2026-10-02-rocenante-ring4/test-protocol.py
python3 -m unittest discover -s tests
```

The build follows the existing deployment workflow on an idle build host:

```sh
bin/spark3 --cluster-config experiments/2026-10-02-rocenante-ring4/cluster.json build prepare
bin/spark3 --cluster-config experiments/2026-10-02-rocenante-ring4/cluster.json build check
bin/spark3 --cluster-config experiments/2026-10-02-rocenante-ring4/cluster.json build image
# In an authorized cluster window, --apply builds and runs the GPU import smoke.
bin/spark3 --cluster-config experiments/2026-10-02-rocenante-ring4/cluster.json build image --apply
```

Publish one deployment revision and distribute the same built image digest to
all four nodes. The relay profile keeps the NCCL ring routing constraints for
large messages and reduce-scatter. Do not run it with three-node NCCL settings.
See [topology setup and coordination](../../docs/switchless-topology.md).

## Validation evidence and remaining gates

Local development runs of the C simulator passed before and after adding the
forced two-doorbell catch-up test. The final recorded run in
`validation-local.txt` executes the freshly applied candidate, verified against
its recorded head and tree. It covers 10,000 logical collective rounds:

- Three-node direct and four-node relay, one and two stripes.
- Payloads of 16, 32, 48, 64, 128, 4,000 and 4,096 bytes, including an empty
  second stripe and unequal stripe lengths.
- Random per-QP network progress with data and flag delivered separately.
  DMA reads source bytes at delivery time, exposing premature buffer reuse.
- Exact payload checking for every source, receiver and sequence; opposite-rank
  network sends are forbidden by the simulator.
- Deliberately paused proxy with two pending doorbells; 32-bit sequence wrap.
- Mixed-mode rejection, oversized payload rejection, GPU poison, newer-slot
  rejection and prompt shutdown while a predecessor is absent.
- Production C compiled with AddressSanitizer and UndefinedBehaviorSanitizer.
- Python neighbour resolver at all ranks/stripe widths and invalid geometry.

The fake verbs layer validates host protocol logic. It does not emulate NIC
coherence, GPU memory ordering, actual registration, GID addressing or compiler
code generation. The later hardware session supplies collective correctness and
latency evidence; serving quality is still unmeasured. A complete model image
rebuild remains separate from the verified transport-only overlay used there.

Hardware qualification sequence (the linked session completes the address,
image and collective screens; sustained stress and model serving remain):

1. Verify addresses/GIDs and build/image identity on all four nodes.
2. Render `topology probe dgx1` through `dgx4` with this cluster config, then
   run those plans concurrently in a coordinated window with serving stopped.
   The probe prepares the actual vLLM RoCEnante adapter, rejects disabled or
   old direct-only backends, checks proxy activity, and tests eager execution
   and CUDA graph replay alongside NCCL fallback collectives. It changes data
   between replays so stale buffers cannot pass.
3. Run sustained mixed-size collectives, unequal rank progress, stop/failure
   handling, and repeated initialization on real hardware; keep per-rank logs.
4. Boot DS4.1 TP4 with unchanged weights/precision, check model loading, short
   and long prompts, draft/target collectives, graph replay, errors and memory.
5. Compare the same TP4 image in NCCL-only and relay modes: small-collective
   latency, decode at one/eight streams, and prefill at 4K/16K/64K. Pin the
   verification-cost table, retain all runs, and record image/config hashes,
   node clocks, temperatures, memory and raw results. No speed claim before
   those measurements.

These gates qualify the candidate; promotion remains a separate measured
configuration and baseline update. The existing lab watchdog cannot restore
physical cabling, so it refuses a candidate topology different from production.
