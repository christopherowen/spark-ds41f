# Prepared collective policy

Status: design and offline evidence normalization, based on published `240bca1`.
No runtime dispatch, GPU experiment or network configuration was changed. Another
session owns dgx4's GPU window. The completed four-path screen is the evidence
source; `evidence.py` reproduces `evidence.json` without accessing the cluster.
External research comparisons are kept out of this repository.

## Objective

Use one prepared communication policy to select a small set of qualified plans.
Optimize serving step time and prefill completion, with numerical requirements
and bounded resources enforced before speed. Prefer fewer backend families and
fewer boundaries when measured end-to-end performance is equivalent.

The design separates:

1. Physical fabric: nodes, cables, PCIe-root interfaces, reachable neighbors,
   NIC queue/offload settings and supervision. Doctor reports mismatches and
   separate correction commands; inference does not change host settings.
2. Collective plan: operation, data volume, representation, synchronization,
   reduction arithmetic, transport, chunking and maximum in-flight bytes.
3. Measured policy: a versioned set of supported plans selected before graph
   capture. Each selection carries its evidence and source/fabric identity.

The two PCIe-root interfaces on each edge share a cable. They are separate
injection/forwarding paths, not independent physical-link bandwidth budgets.

## Traffic distinctions

| Traffic | Main objective | Policy inputs |
| --- | --- | --- |
| Small all-reduce/all-gather on the next-token dependency path | Low startup and synchronization latency | Operation, per-rank input/output bytes, dtype, layout and supported numerical contract |
| Large all-reduce | Throughput with bounded GPU/CPU/queue occupancy | Same inputs plus chunking and reduction order; evaluate reducing while forwarding |
| Large all-gather | Move distinct shards efficiently | Output expansion, pipelining, available paths and in-flight-byte limit |
| Reduce-scatter | Reduce and partition, avoiding unnecessary replication | Full per-rank input bytes and output shard bytes; presently NCCL only |
| Concurrent collectives/compute | Shorten the critical path without starvation | Measured overlap and resource budget; do not infer this from isolated latency |
| Bootstrap, model transfer, health checks | Reliable control or bulk transfer | Existing management path and ordinary host-network policy |

Decode and prefill help weight the benchmark workload. The dispatcher should
use actual operation descriptors: a mixed batch can contain both phases, and
prompt length alone does not determine collective bytes. Tensor-parallel MoE
traffic must not be mislabeled as expert-parallel all-to-all.

## Current evidence and simplification candidates

The current screen supports small-message forwarding and large-message relay
as useful regimes. At 2 MiB BF16 all-reduce the launch medians are about 555 us
(two-path forwarding), 418 us (four-path forwarding), 423 us (rotating), and
315 us (CPU relay). At 10 KiB they are about 16, 16, 17 and 18 us respectively.
These are isolated communication measurements, not serving savings.

Four paths are not automatically a third production family. They improve the
old large-message forwarding path but relay is faster there. There is a modest
four-path advantage at the sampled 60 KiB BF16 all-gather (26.3–26.7 versus
28.4–29.3 us for two paths). That one point does not establish a stable interval
or justify permanent fabric/dispatch complexity. Rotation and the previously
rejected direct-first waves stay out of the initial policy search.

Compare these designs, in this order:

- CPU relay plus NCCL for unsupported operations/sizes. This removes the need
  for hardware forwarding rules, markers, hairpin setup and their supervisor.
- Small-message forwarding plus CPU relay and NCCL, only if serving gains repay
  that additional lifecycle and prepared-state cost.
- Four-path forwarding only if a repeated interval wins against both simpler
  candidates in a relevant serving workload.

A tuned NCCL baseline is required before committing to these boundaries. Current
NCCL is forced to one-channel Ring. Vary channels within legal neighbor rings
first; prove the actual edges in the graph logs and correctness probes. Do not
remove reachability constraints or enable non-neighbor trees as a tuning shortcut.

## Runtime shape

Resolve a plan from a descriptor such as:

    (fabric/source identity, world size, collective, dtype, layout,
     per-rank input bytes, per-rank output bytes, numerical contract)

A plan fixes the backend, algorithm/protocol, channel count, chunk size and
in-flight-byte budget where supported. These are conceptually separate:
paths choose where bytes go; credits/pacing control how much enters at once;
collective algorithms determine which bytes must move and be reduced.

All ranks validate one policy hash and the applicable descriptor before use.
Initialize only the selected backend families in a canonical order and prepare
all buffers and graph variants before capture. Do not retune inside requests or
mutate environment variables between calls. Expose plan ID and aggregate
call/byte/time counters, with disabled-by-default detailed tracing.

Unknown/unsupported descriptors go to an explicitly prepared compatible plan;
strict numerical mode refuses an incompatible fallback. Transport failure is a
coordinated failure, never a local mid-collective backend switch.

The current B12X runtime reads one topology at construction, owns registered
buffers/QPs/sequence state, and uses size eligibility bounds. A hybrid is therefore
not an environment toggle or one threshold edit. Either prepare independent
caller-owned runtimes with separate buffers/sequence state and canonical
initialization, or extend one native descriptor protocol. Start with the former
only if the relay-only serving screen shows a worthwhile small-message deficit;
measure its memory and polling-thread cost before adopting it.

Numerical constraints are part of eligibility. Changing transport can preserve
an existing fixed reduction, but changing collective algorithm, chunking,
precision or channel layout can change summation order. A fixed policy alone
does not prove batch-invariant output. Validate fixed inputs across supported
sizes, batch layouts, graph/eager execution and restarts against the selected
numerical reference before admitting a plan to strict mode.

## Bounded tuning sequence

1. Obtain a serving collective histogram and representative dependency/overlap
   trace, with tracing overhead measured. Record call counts, operation, dtype,
   bytes, backend and row shape for target and draft paths. Include decode,
   speculative verification, prefill and chunk transitions. Weight improvements
   by frequency and critical-path contribution; avoid a synthetic average score.
2. Reuse correctness gates and screen only eligible candidates. Repeat relay and
   test NCCL 1/2/4 channels on verified neighbor rings. Keep graph identity,
   source/image, thermal baseline and native workloads fixed. Reset the process
   between NCCL environment changes; collect selected algorithms and protocols.
3. Densify sizes only near observed crossovers. Current four sizes cannot locate
   thresholds. Normalize all-gather input versus output and reduce-scatter's
   whole input; do not pool same-byte operations as equivalent traffic.
4. Prefer a small number of contiguous regions. Require reproducible benefit
   beyond run variability and a material workload gain before adding a region
   or backend. Freeze the table with source, driver, kernel, firmware and fabric
   fingerprints. Do not silently reuse it after those identities change.
5. Replay observed compute overlap, then the existing serving c1/c8 and prefill
   matrix. Include tail latency, memory, proxy CPU cost, errors/retries and
   numerical gates. Accept a winner on serving behavior, not just isolated us.
6. Remove dominated runtime variants from the selected policy; retain their
   experiment receipts. Promote only through the normal owner-accepted baseline.

If forwarding remains worthwhile, the next congestion experiment is a sliding
bound on outstanding bytes, with fairly interleaved independent chunks and final
completion only after every chunk is visible. It must preserve RC ordering and
completion semantics. It is different from the rejected global direct-first
completion barrier. Query NIC capabilities before considering hardware pacing or
traffic classes; verify where drops and congestion signals originate before
changing flow control. Ordinary software packet schedulers do not govern these
offloaded payload writes.

## Evidence limits

`evidence.json` contains the 24 sampled operation/dtype/size cells and all seven
launch identities, with per-case counters. It marks every cell as not yet
qualified for dispatch. Reduce-scatter and over-limit FP32 all-reduce are pooled
as NCCL, regardless of the enclosing arm label. The relay has only one launch
in this matched session. No thresholds, mixed-backend correctness, end-to-end
speedup, strict determinism, or four-path superiority between samples are inferred.
