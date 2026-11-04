# Streaming IKET collection

Owner, 2026-11-04: implement a reusable collector whose total capture size is
limited by disk, complete bias/slot-mapping instrumentation, and check sparknet
coverage. Authorized hardware: idle Spaceport sp8 with candidate-38. Both serving
clusters remain running. No production configuration or baseline changes.

Base identities are in base.json. The NVIDIA collector is binary-only here and
has no supported reclamation API. This experiment tests an opt-in collector
using the existing TileLang phase annotation, a fixed CPU/GPU-visible ring,
system-scope publication, and a native background binary writer. Ring slots are
reused after copying into a bounded write batch. Slow or failed storage applies
backpressure or fails collection explicitly; it never silently overwrites data.
This alternate collector does not instrument arbitrary ATen/NCCL SASS.

Variables are screened separately: ring transport correctness/backpressure;
phase collection; previously unsampled runtime loops; sparknet coverage. Every
accepted hardware point boots twice, retaining failures, exact commands, memory and
identity receipts. Full-fabric tests require an idle authorized multi-node window
and cannot use the serving ranks. A one-node test is not fabric qualification.

Acceptance: output bytes exceed ring capacity many times while resident trace
memory stays bounded; no missing/repeated sequence IDs; slow/disk-error cases
behave explicitly; CUDA-graph and multi-stream captures retain identity; sampled
runtime loops have finite event bounds; networking phases appear in manifests
and exported traces without changing ordinary kernels when profiling is off.

## Implementation and use

`stream/device.cuh` emits 64-byte records into a sequence-numbered multi-producer
ring. A native reader copies published records into a fixed 64 KiB batch before
releasing their slots. `stream/writer.cpp` appends that batch to rotating binary
files (64 MiB by default), recording counts and CRC32 checksums. Completed data
files are synced before rename. Neither the GPU allocation nor the write batch
grows with the number of events. The operating system may cache disk writes.

The default 8 MiB ring is a working-memory choice, not a total capture limit.
No maximum event count or run-length estimate is required. A full ring stalls
producers until storage catches up. Storage failure sets a shared error and
aborts the traced workload; the command fails and preserves completed chunks
and the unfinished file. This is lossless backpressure, not a promise of
undisturbed application timing. Process or power failure is outside the
completeness guarantee; this is not a transactional logging service.

The shared allocation uses managed memory and checks concurrent managed access
and host native atomics. Mapped pinned memory is retained only as a diagnostic
option: mapped-memory atomicity is insufficient for this protocol. Do not use
that option for collection.

Enable before importing/compiling TileLang kernels, in the exact candidate-38
image recorded in `base.json`, with the candidate's normal environment:

```sh
python3 experiments/2026-11-04-iket-streaming/run.py /out/capture \
  --ring-mib 8 --chunk-mib 64 --ctas 0 --iters 0 --warps 0 \
  -- workload.py workload-arguments
python3 experiments/2026-11-04-iket-streaming/stream/export.py \
  /out/capture /out/capture/trace.json
```

Zero sampling settings record every reachable annotated phase iteration. The
defaults sample one CTA, one warp and one iteration for phase events. Warp
lifetimes are collected across the grid. Lane zero represents each warp;
branches in which lane zero does not participate are outside this collector's
scope. `%gridid` distinguishes actual launches, including graph replays.
Export streams native chunks into Chrome Trace JSON, which Perfetto can open;
`--grid ID` selects a launch. Sequence and checksum validation precede successful
completion. Incomplete captures are retained for diagnosis and refused by the
export command.

One collector is allowed per process. Do not combine it with NVIDIA injection or
a persistent cache of instrumented kernels: generated code embeds the ring's
process-local address. Closing drains the writer, retires recording, and keeps
the bounded allocation alive until process exit so already compiled kernels
remain safe to call. Restart the process to begin another capture.

The five source patches in `vllm/series` repair runtime-loop sampling, extend
network phase recognition, and rendezvous profiled vision TMA producers after
empty-buffer waits and before reuse. Their
base and generated overlay hashes are checked
at load time. A fresh application of the complete 572-patch candidate-38 stack
reproduced head `1f33102547e4da5c64712390604695d1abd577c4` and tree
`c061e5f6931306fa4655ba531428f639c365bf10`. Applying our five patches in another
fresh prepared tree produced head `6305f8aa742b3c13b8db74f12d61c9676d965b53`, tree
`ccdef550a36d542b91277ee200114ae3f46f4bcd`, and the overlay hash in
`vllm/identity.json`. All 20 CPU phase-pass checks passed.

## Findings and retained failures

The final producer ordering (`full-after-wait`, deployment commit `efabe7dc`)
passes both full boots of the unchanged 43-layer dummy-weight, TP4-shaped
single-GPU harness. Each request has a 32-token prompt and eight generated
tokens; sequence capacity is 16, context limit 8,192 and KV allocation 2 GiB.
This is collector qualification on sp8, not distributed model qualification.

| Boot | Records | Native event bytes | Ring | Wall seconds | Minimum host available GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 51,860,314 | 3,319,063,296 | 8 MiB | 305.78 | 35.931 |
| 2 | 51,860,314 | 3,319,063,296 | 8 MiB | 303.44 | 35.193 |

Both captures have 50 completed chunks and pass sequence, CRC and per-warp
lifetime/phase pairing checks. Each contains 25,791,230 lifetime pairs and
138,927 phase pairs across 106 executed kernel names; the streaming integrity
check needs at most 1,920 open warp lifetimes in memory. All 442 compiled
specializations (126 names) have finite sampled manifests, with none skipped.
Both captures contain bias and slot-mapping phase events. Tokens and every
reported log-probability are identical to the recorded uninstrumented
candidate-38 control; that control was not rerun. Its additional replay grid is
excluded from this request-output comparison. Wall times and host-wide memory
observations are diagnostic receipts, not uninstrumented performance results.

The harness reports native IKET mode `off` because collection is enabled by the
outer streaming wrapper, rather than its NVIDIA-injection path. Per-kernel
streaming manifests record the actual phase sampling (one CTA/warp/iteration);
warp lifetimes cover the entire grid. The all-events bounds probes below cover
the mode that needs no static event limit.

The writer's five CPU tests cover concurrent wraparound, slow storage, injected
disk-full errors, missing chunks and corrupted completed data. The GPU stress
matrix repeats ordinary collection, delayed writes, CUDA graphs, simultaneous
streams and disk-full failure. A 4 KiB ring holds only 63 records; successful
steady captures wrote 380,000 records (1,491 deliberately small chunks), with
every sequence and checksum validated and unchanged numerical output. Expected
disk-full cases return nonzero, status 28 and an incomplete summary.

The real bias and slot-mapping kernels match independent CPU references in two
plain, two streaming and two NVIDIA-native sampled boots (`bounds-v3`). Their
sampled event bounds are now 34 and 10 respectively. Two all-events streaming
boots (`final`) also pass with those bounds explicitly unknown: 5,920 records
through 63 reusable slots. Streaming does not require a static bound.

Sparknet's four kernel families compile for triangle, ring and switched layouts
(12 operation/topology cases). `network-all` executes allreduce, allgather,
reduce-scatter and exchange three times each with a local fake peer supplying
delayed payloads/flags. Two plain and two streaming boots match exact references.
Staging, proxy doorbell/tail, flag waits, reduction and gather phases are present.
The global TileLang callback covers sparknet before its prepare/freeze step;
no sparknet source or pin change is required. This checks the actual GPU kernels,
but uses no NIC traffic, proxy process or serving rank. Multi-node transport,
timeout behavior and clock alignment still need a separate fabric qualification.
All 40 installed sparknet Python files match a fresh checkout of the pinned
`1fd6a1fbd8dc4884057046d1ebd93d585c9e4bcf` commit, whose tree also matches the
image and lock (`runs/network-identity.json`). The candidate snapshot's human
`revision_url` points at an older commit; `base.json` records the corrected link.
The raw input snapshots remain unchanged.

All unsuccessful points remain in the receipt:

- `probe`: electing an arbitrary currently active lane caused duplicate events;
  fixed by using stable lane zero. `probe-v2` passes twice with exact counts.
- `stress`: a helper named `kernels.py` shadowed an installed dependency; every
  job failed before launch. Renamed to `replay_controls.py`.
- `bounds`: source-overlay setup assumed Git was installed in the image;
  native captures also lacked preflight manifests. Packaging a hash-checked
  overlay and generating preflight manifests fixed both problems.
- `network-v2`: the exchange builder's signature differs from reduction
  builders; corrected the probe's argument list before successful repeats.
- `full`, `vision-large`, `vision-sync`: the first mapped-memory transport
  stalled on large vision grids. Managed memory supplies the atomic semantics
  this protocol requires; isolated managed-memory repeats pass. This alone did
  not fix the full workload.
- `full-final`, `full-uniform`, `swiglu`, `swiglu-interaction`: producer stalls
  persisted with warp rendezvous removed or register redistribution disabled,
  separately and together. A successful first boot was followed by a failed
  repeat. Neither timing change is a sufficient fix.
- In `swiglu-interaction/cuda-debug.log`, the debugger inside the container found
  only a non-copy producer warp still active, in `mbarrier_wait`, after consumer
  warps had exited. The inferred cause is a producer falling multiple generations
  behind a reused parity barrier. PTX parity waits cover only the current or
  immediately preceding phase ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)).
  The third patch rendezvous all 128 producers at each iteration, in the two
  recognized vision producer loops only, and refuses barrier-ID conflicts or
  unfamiliar layouts. Two `swiglu-barrier` boots each pass 20 launches and
  1,436,440 records, preserving normal register redistribution and output bytes.
- `full-fixed`: with linear and SwiGLU repaired, the full encoder exposed the
  same late producer wait in `vision_attention_kernel`. The in-container CUDA
  debugger again found only producer warp 1 active, in `mbarrier_wait`. The fourth
  patch extends the rendezvous to that recognized CTA-uniform dynamic loop.
  Two `attention` boots each pass five real 8,704-row, 16-head attention launches
  against an exact uniform-softmax reference, with 87,110 balanced records.
- `full-complete`: its first boot passed, writing 51,860,314 records through the
  8 MiB ring. Per-warp pairing across 106 executed kernel names, every sequence,
  every checksum, and exact tokens/log-probabilities against the recorded plain
  control all passed. The repeat deadlocked in SwiGLU: one producer waited on
  the empty barrier while the other producers were at the next rendezvous and
  consumers awaited new copies. A loop-entry rendezvous was insufficient.
  The fifth patch moves it immediately after each producer empty wait and before
  buffer reuse, covering linear's one wait, SwiGLU's three activation/gate/up
  waits, and attention's two K/V waits.
- `producer-after-wait`: the strict layout guard correctly refused SwiGLU before
  launch because it has three independent waits. Corrected the expected layout;
  `producer-waits-v2` then passes both 20-launch boots with unchanged output.

The final emitter adds no warp barrier. `uniform_registers=True`, mapped memory
and tracing-time warp synchronization remain diagnostic options only. Profiling
off uses the original image and kernels. The producer fix changes synchronization
while profiling; its timings cannot establish ordinary runtime performance.

`runs/local-validation.json` records local checks: writer tests, experiment lint,
doctor and diff checks pass. Applying the affected upstream series in a fresh
tree reproduces the overlay. Upstream lint has five inherited E501 line-length
violations, with each exact line confirmed present before these patches and no
new violations. The image build check reports the promoted build directory is
not prepared locally, so no new OCI build is claimed. Hardware tests compile the
native writer and instrumented kernels inside the frozen candidate image.

## Evidence

`runs/hardware.json` records every attempted hardware point, source commit,
exact container command, image identity, output checks, memory observations,
failure tails and retained artifact locations. Native event files and full logs
remain on sp8 under `/home/chris/spark-lab/iket-streaming-20261104/`. The capture
files are too large for this repository. `collect_receipts.py` regenerates the
portable receipt from that directory. No published baseline was rerun or edited.

Final BF16 networking and regression checks are in progress. No performance claim
or promotion.
