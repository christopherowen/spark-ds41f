# Prepared collective policy

Status: four-node NCCL/relay hardware screen; no production promotion or runtime
dispatch changes. `evidence.py` reproduces the earlier forwarding comparison.
`summarize.py` normalizes the new raw launches, including failed/confounded runs.
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

## NCCL and relay screen

The next published harness uses `run.py ARM RUN_ID --benchmark --counter-samples`.
It selects the frozen image from `cluster.json` and the bounded, explicit overrides
in `arms.json`. Those overrides apply only to model-free probe containers; the
production ring policy is not relaxed. Fresh processes test Ring with 1/2/4
channels and automatic (existing LL128 exclusion), Simple or LL protocol. Every launch records the complete
command. It changes no NIC parameters. Actual initialized channels, neighbor
ring edges and GDR status must be verified in every rank's logs.

The shared collective probe now accepts optional benchmark lengths and an
optional cancellation-sensitive numerical screen. The same first 64 elements
are reduced at every sampled capacity, with output hashes and differences from
a fixed rank-order FP32 accumulation. These differences characterize arithmetic;
they are not automatically classified as transport errors or model-quality loss.
The existing exact transport checks remain mandatory. No strict determinism
claim follows from these synthetic inputs alone.


The follow-up bounded arms change one setting from an existing four-channel
control: `NCCL_NTHREADS=128` with LL; `NCCL_BUFFSIZE=262144` with Simple;
and `NCCL_BUFFSIZE=4194304` with automatic protocol selection. The original
buffer is 1 MiB. These are experimental process-local overrides, not promoted
settings. A faster isolated result still requires a serving/overlap screen.

The per-call plan log revealed that the 10 KiB all-reduce still uses one LL
channel with four channels initialized. Two targeted arms lower only the Ring/LL
thread threshold from 32 to 1 (`NCCL_THREAD_THRESHOLDS="8 8 64 1 8 64"`),
with two/four initialized channels and automatic protocol selection. The pinned
NCCL source reads the six values as Tree then Ring, each LL/LL128/Simple. Confirm
the resulting per-call channel ranges in the logs; initialization count alone
is not evidence of the channels used by a call. This internal tuning control is
version-specific and experimental.


## Hardware findings, 2026-10-02 UTC

The four-node window began at 22:42 UTC with idle GPUs and no running containers.
All nodes use the same `vllm-ds41f-kkref:04c30fa98e79-r5o-roce-mesh4-fourpaths-v1`
image, ID `sha256:9671c96903dc7cc5a9bbae4249e7b7aa9efa550980f1dab5fa142dac3ebe842d`,
NCCL 2.30.7 with the pinned ARM send fence, kernel `7.0.0-1019-nvidia-64k`,
and driver `580.178.04`. The NIC hairpin queue size remains 1024; no forwarding
rules or host NIC changes are used by these arms. Qualification checkouts were
clean published `f6ed9fc` for runs 01–10, `9df271f` for 11–15 and `a2ee86f` for 16–21.
Those harness updates change only experiment controls/analysis, not the image.

Each launch starts below 55 C on all four nodes. Containers are capped at 12 GiB.
The exact command, cooling receipt, port/error counters, memory snapshots and
all rank logs are retained. BF16 and FP32 correctness is checked in eager and
CUDA graph execution. The 24 timing cells contain three operations, two dtypes
and four sizes. Each of five CUDA-event samples covers 256 graph calls; the
reported value is the median of the slowest rank in each sample. Compilation,
CPU barriers and counter reads are outside timing. These are isolated collective
latencies, not TPS/TTFT results. Small differences require repeated clean launches.

All-reduce BF16, microseconds per call. Payload is the input bytes per rank.
Automatic protocol means the existing LL128 exclusion remains in force.

| Run | Arm (initialized channels; actual plans in logs) | 10 KiB | 60 KiB | 480 KiB | 2 MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 01 | nccl-c1-auto | 85.82 | 108.76 | 159.02 | 360.19 |
| 02 | nccl-c2-auto | 93.70 | 96.90 | 148.56 | 250.55 |
| 03 | nccl-c4-auto | 82.56 | 105.61 | 157.38 | 238.92 |
| 04 | nccl-c1-simple | 111.69 | 116.73 | 171.63 | 368.85 |
| 05 | nccl-c2-simple | 104.61 | 115.97 | 143.13 | 271.32 |
| 06 | nccl-c4-simple | 99.83 | 121.74 | 156.37 | 234.51 |
| 07 | nccl-c1-ll | 87.38 | 111.23 | 374.60 | 1418.14 |
| 08 | nccl-c2-ll | 84.92 | 86.08 | 222.62 | 756.54 |
| 09 | nccl-c4-ll | 80.12 | 90.51 | 191.61 | 520.77 |
| 10 | relay | 17.25 | 29.07 | 94.40 | 314.41 |
| 11 | nccl-c4-ll-threads128 | 97.86 | 101.23 | 249.24 | 813.78 |
| 12 | nccl-c4-simple-buffer256k | 94.46 | 117.01 | 147.50 | 230.45 |
| 13 | nccl-c4-auto-buffer4m (confounded) | 82.85 | 94.33 | 162.12 | 221.11 |
| 14 | nccl-c4-ll-repeat (confounded) | 72.74 | 78.73 | 167.62 | 521.48 |
| 15 | relay-repeat (confounded) | 23.15 | 34.07 | 94.86 | 318.17 |
| 16 | nccl-c2-auto-llthreshold1 | failed | — | — | — |
| 17 | nccl-c2-auto-llthreshold1-retry | 76.90 | 87.00 | 143.02 | 255.99 |
| 18 | nccl-c4-auto-llthreshold1 | 85.25 | 101.11 | 143.25 | 219.84 |
| 19 | relay-clean-repeat | 18.76 | 28.79 | 91.42 | 315.35 |
| 20 | nccl-c4-auto-buffer4m-clean-repeat | 88.35 | 97.59 | 156.57 | 226.91 |
| 21 | nccl-c1-auto-clean-repeat | 74.59 | 100.82 | 170.14 | 348.93 |

Runs 13–15 overlap independent checkpoint copies and are **confounded**; their
numbers are retained but excluded from tuning decisions. Run 16 failed before
collective initialization on dgx2 and has no timing result. It does not test the
small-message threshold. See the environment incident below.

The first 12 clean launches and the clean post-recovery launches 17–21 support
these conclusions:

- Increasing NCCL channels materially improves bulk transfers. At 2 MiB BF16,
  all-reduce changes from 360 us with one channel to 239 us with four (34% lower
  latency); all-gather from 704 to 381 us; reduce-scatter from 729 to 368 us.
  Four-channel Simple is similar for these large cells. This is useful tuning.
- Small all-reduce remains much faster through the relay: 17–19 us at 10 KiB and
  about 29 us at 60 KiB, versus the best clean NCCL screen values of 75 and 86 us.
  These minima come from different NCCL settings, not one deployed policy.
- At 480 KiB BF16 the relay is also faster (91–94 us versus NCCL's best 143 us).
  At 2 MiB NCCL all-reduce is faster (220–239 us versus relay 314–315 us), but
  all-gather is still slightly faster through relay (351–352 versus 357–381 us
  across the clean four-channel automatic screens). Do not reuse an all-reduce crossover for
  all-gather; its output is four times its per-rank input.
- Reduce-scatter always uses NCCL, even in an arm labeled relay. Its benchmark
  `elements_per_rank` is the output shard size; its full input is four times
  that. FP32 4 MiB all-reduce also exceeds the custom 2 MiB limit and uses NCCL.
- The 128-thread LL arm regressed. The 256 KiB Simple buffer has mixed effects:
  it does not close the small-message gap and worsens some large gathers.
  These controls are not candidates for promotion.

### What the NCCL logs establish

Every completed rank uses legal neighbor rings; no opposite-peer connection is
introduced. Two/four initialized channels use both PCIe-root virtual NICs
(`/4` and `/5`) instead of only `/4`. The subnet-aware resolver selects the
physical interface reaching the same neighbor. This increases injection
parallelism without adding physical cable capacity.

Initialization channel count is not per-call channel count. The logged plan
for 10 KiB all-reduce is Ring/LL with channel 0, even with four initialized
channels. At 60 KiB that arm uses channels 0–2, and at 480 KiB/2 MiB automatic
selection uses Ring/Simple with channels 0–3. Plan logs are emitted on rank 0;
ring topology is checked on every rank. This explains why forcing LL alone is
not a new small-message plan. The clean threshold tests use two actual channels with two initialized (run 17),
and three actual channels with four initialized (run 18), for the 10 KiB call.
They take 76.9 and 85.3 us respectively. A clean one-channel automatic repeat
takes 74.6 us (run 21), versus 85.8 us originally. Thus the tested small-message
tuning has no established benefit beyond launch variability. It does not close
the gap to the relay.

All completed launches report `GDR 0`. This alone does not explain the custom
backend's advantage: NCCL's non-GDR network path also gives the GPU access to
pinned host buffers. The relevant algorithm difference is NCCL's four-rank
Ring all-reduce with six logical transfer steps versus exchanging contributions
and reducing locally in the custom path. Protocol/credit handling and proxy
scheduling can also contribute. The current timings do not decompose those
costs; no claim is made that NCCL cannot be improved further.

### Numerical contract

All exact transport correctness checks pass on the completed launches. The
cancellation-sensitive first-64-element diagnostic distinguishes the fixed
rank-order FP32 reference from NCCL's reduction. In the relay's custom range,
it matches that reference; the NCCL path differs on all 64 stress elements
(maximum absolute difference 0.25). This is floating-point ordering/precision
behavior, not evidence of a transport bug or measured model-quality loss.
The over-limit FP32 relay arm switches to NCCL and changes its prefix hash.
This existing fallback already crosses numerical contracts.

The same first 64 elements match across the tested capacities within each
backend's range, but that prefix alone is not a proof of batch invariance:
other positions can change reduction ownership or arithmetic as sizes change.
Any strict deterministic policy needs explicit numerical qualification rather
than admitting NCCL solely because it is faster.

### Environment incident and remaining work

Six independent rsync transfers of the model checkpoint from dgx2/dgx3 to dgx4
started at 23:02:24 UTC, partway through run 13. They consume CPU, memory and
network/storage resources. Their process receipts and the launch timestamps
are retained. Runs 13–15 are confounded, even though their correctness and NIC
error checks pass. In run 16, dgx2 fails in `torch.cuda.set_device(0)` before
NCCL starts; peers are aborted by the runner. Driver messages report memory
descriptor/context allocation failure. dgx2 has only 1.05 GiB free but 118 GiB
available, with most RAM in file cache. After all rsync processes finished, all
four idle nodes reclaimed clean file cache with `sudo sysctl -w vm.drop_caches=1`; checkpoint files were preserved.
Free memory returned to about 118 GiB and the unchanged run-16 arm passed as
run 17. This is consistent with cache-related driver allocation pressure; the
copy completion and reclamation were not isolated as separate variables. Do not
interpret this failure as a property of the NCCL threshold setting.

The serving preflight also found the configured checkpoint mount absent on
dgx4 before these transfers began. An initial Mac-side preflight attempt used
a too-long SSH ControlPath; the successful retry used `TMPDIR=/tmp`. Neither
attempt started serving. After the transfer, a third read-only preflight reports
no problems on any rank; this verifies mount availability and runtime identity,
not full checkpoint integrity or TP4 model startup. No TPS/TTFT or compute-overlap
claim follows here.

Next densify the all-reduce crossover before selecting any threshold, then
measure actual serving collectives and compute overlap. The tested channel
threshold change does not merit another runtime policy knob. The serving
checkpoint and model consistency checks must pass before measuring the operation histogram and model-level c1/c8/prefill matrix.
Prefer the existing relay plus NCCL adapter to another transport framework.
Do not retain hairpin forwarding solely for an isolated few-microsecond saving.

Changing `VLLM_ROCE_ALLREDUCE_MAX_SIZE` is not an isolated transport policy edit:
vLLM patch 0018 derives its sequence-parallel prefill boundary from the same
reported limit. A clean future policy must distinguish the registered buffer's
capacity, the preferred transport boundary, the prefill execution choice, and
the numerical contract. Record that coupling when comparing serving candidates.


### Completion and receipts

There were 21 launches: 20 passed on all four ranks; three of those passed
while checkpoint transfers were active and are excluded from performance
comparisons. The remaining launch failed before NCCL and was successfully
repeated after the host-memory recovery. All completed launches have zero
observed NIC buffer drops, retransmissions and packet-sequence error increments.
Before/after memory snapshots are retained but do not establish peak runtime
memory usage. The 171 repository tests passed before the screen; the experiment
config passes doctor, and the analysis reproduces all retained timing cells.

Cleanup verifies idle GPUs, no probe/running containers, no owned forwarding
rules/routes/markers, all NIC settings matching entry, and active fan control.
The shared checkouts remain clean at `cfa6e5c`; only the isolated qualification
checkouts changed. The hold was released at 23:17:31 UTC. Entry was idle and was
restored idle. The cache reclamation is recorded as a temporary host-state
change; no persistent kernel/NIC policy or production configuration changed.

`hardware/runs.tar.gz` contains raw logs, commands, snapshots, launch timestamps,
run annotations, failed attempts, cache-pressure/recovery receipts, three source
syncs, local checks and cleanup/release receipts. `hardware/summary.json` is
reproduced by extracting that archive and running `summarize.py` on its root.
`hardware/sha256.json` records artifact checksums. Selected NCCL plans retain
the library's logged byte convention; use the explicit benchmark operation and
shape when comparing all-gather output or reduce-scatter input sizes.
