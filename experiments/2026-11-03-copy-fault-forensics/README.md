# CopyDrafter fault investigation

Base: `b28365c38a74c966032dcf558d8d1bb0a282d4b6`.
This is diagnostic work, not a candidate promotion or a performance comparison.
A repeated rank-layout fault is now established: rank-local asynchronous host
data determines a shape that must agree across tensor-parallel ranks. The
original failing NaN's
first producing instruction and input bytes have not been captured. The
evidence below keeps those two claims separate.

## Repeated last-prepared layout disagreement

The crash-surviving layout recorder in deployment `c2bb2ebb` provides stronger
evidence than the earlier periodic counters. Two candidate-37 boots leave the
same disagreement in their last host preparation record:

| Boot | Last recorded step (zero based) | TP0 rows / drafts | TP1 rows / drafts | TP2 rows / drafts |
| --- | --- | --- | --- | --- |
| `shared-budget-ring-sp` | 5339 | 17 / 16 | 6 / 5 | 17 / 16 |
| `shared-budget-ring2-sp` | 6372 | 17 / 16 | 6 / 5 | 17 / 16 |

All ranks received 16 scheduled drafts. TP0 and TP2 report the router assertion;
TP1 does not in the saved logs. The recorder runs during host preparation,
before the forward, and preserves the compacted token count, padded count,
eligible-draft CRC and budget. It does not capture GPU metadata or tensor data.
It also precedes the graph/eager launch: it does not prove that this prepared
forward was enqueued or that its GPU work produced the assertion. An earlier
asynchronous failure could surface before that launch. References to the
"crash step" in other experiment reports are stronger than this recorder proves.

The first boot's retained report says its preceding 5,339 layouts agreed; its
complete binary files were overwritten by the repeat boot before collection.
For the repeat, all three complete binary files are preserved: 6,373 records
each, consecutive step IDs, and exactly one divergence, at step 6372. The
preceding 6,372 layouts agree. Two independent local reads reproduced that
result. [The analysis receipt](layout-repeat-evidence.json) contains file
hashes and the exact differing records. It was generated with:

```sh
python3 experiments/2026-11-03-copy-fault-forensics/analyze_layout_ring.py \
  results/private/copy-fault-forensics/tool-runner-ring-20261103T082405Z/sp5/cache/spark-ds41f/layout-ring/rank0.bin \
  results/private/copy-fault-forensics/tool-runner-ring-20261103T082405Z/sp6/cache/spark-ds41f/layout-ring/rank1.bin \
  results/private/copy-fault-forensics/tool-runner-ring-20261103T082405Z/sp7/cache/spark-ds41f/layout-ring/rank2.bin
```

The exact candidate-37 code lets the adaptive budget cap scheduler drafts with
a rank-local, asynchronously refreshed host mirror. Limits 16 and 5 therefore
produce 17 and 6 rows before graph dispatch. There is no TP agreement at that
boundary. On the reviewed target embedding path, these host layouts request
different all-reduce sizes: 174,080 versus 61,440 bytes for BF16 rows of width
5,120. The [source review](source-cause-review.md) records exact pins, file
hashes and the complete path.

The introducing change is local patch
[0612](../2026-10-26-spark-ds41f-candidate/rc20/vllm/0612-Eliminate-CopyDrafter-async-scheduling-event-sync-st.patch):
`sync_slot_draft_limits` stopped waiting for its event and instead returns an
older mirror when `event.query()` is false. The adaptive planner continued
using that locally selected value to size a TP-wide step. Removing the wait
required another way to keep those planning inputs consistent across ranks;
that part was missing.

Candidate-37 pins Sparknet `1fd6a1fbd8dc4884057046d1ebd93d585c9e4bcf`, tree
`8cc28a5cd3e683c1f114ed843453e3ce34f323e0`. Its lock's `revision_url` still
names the older promoted revision; this investigation uses `revision` and
checks the actual source hashes. The transport publishes each sender's local
payload followed by a sequence flag. Receivers check the flag and reduce their
own local number of rows; no peer length is checked. A 17-row receiver thus
consumes 11 rows of old receive-slot bytes after a six-row peer's new payload.
Those addresses remain inside allocated memory, so address sanitizers need
not report an error.

`collective_short_sender.py` tests the unchanged, hash-checked receive/reduction
kernel on one GPU with a simulated completed-delivery snapshot. Its finite
old-tail and integer old-tail cases are explicit controls, not a claim that
those particular bytes occurred in the serving failure. It opens no NIC or
process group. [The actual GPU receipt](sp8-tool-short-sender-20261103T082616Z.json)
records four passing mechanism checks:

| Current peer rows | Previous peer-slot bytes | Rows 0–5 | Rows 6–16 |
| --- | --- | --- | --- |
| 17 / 17 / 17 | BF16 value 5 | correct sum 6 | correct sum 6 |
| 17 / 6 / 17 | BF16 value 5 | correct sum 6 | incorrect sum 9 |
| 17 / 17 / 17 | integer −1 bytes | correct sum 6 | correct sum 6 |
| 17 / 6 / 17 | integer −1 bytes | correct sum 6 | NaN |

All current inputs were finite. All five protocol counters reached 1, and the
timeout/error word remained zero. Memcheck repeated the short integer-tail
case with the same eleven NaN rows and **zero reported memory errors**.

The complementary host fixture, `native_proxy_payload.c`, includes the exact
production proxy through its existing fake-verbs simulator. It performs three
actual `post_op` calls per rank; sequences 1 and 3 reuse slot 1. In the short
case, all sequence flags
advance to 3 while 56,320 BF16 elements (11 rows) retain sequence 1's bytes.
The equal-size controls overwrite them. Both ASan+UBSan and TSan runs reproduce
this with no reports ([host receipts](host-native-results.json)). This validates
the actual posting and slot-reuse logic
with simulated NIC delivery; the GPU fixture validates the actual reduction.
Neither captures the failing production slot's previous contents.

The reviewed shared-budget patch addresses this source: it removes the
`slot_draft_limits` callback from adaptive planning and uses the existing
shared scheduled-draft path. It adds no host wait or device clamp. The
[fix review](source-cause-review.md#review-of-the-shared-budget-fix) records
the exact patch hashes, sentinel rejection checks and remaining assumptions.
Removing unused limit buffers/traffic is a separate cleanup with a separate
source tree. This investigation makes no new performance claim or promotion.

## Completed isolated checks

All runs below used the same immutable candidate-37 image on idle sp8. The
serving clusters were not changed. Eight `sp8-*.json` receipts retain the
initial collection failure, four tool controls and six completed test runs,
including exact commands and raw artifact hashes.

| Existing tests | Memcheck | Initcheck |
| --- | --- | --- |
| Dead-row input combine | 2 passed, 0 errors | 2 passed, 0 errors |
| CopyDrafter state, stale allocation and graph replay | 6 passed, 0 errors | 6 passed, 0 errors |
| Gate/router, padding and programmatic graph edges | 15 passed, 0 errors | 15 passed, 0 errors |

These are 23 distinct cases executed under each tool, with zero skipped tests.
Memcheck caught the deliberate invalid global read, and initcheck caught the
deliberate uninitialized global read; both returned the expected error code 86.
Both corresponding clean controls returned 0 with no errors. Test runs took
14–47 seconds each under instrumentation; these are diagnostic durations, not
performance measurements. There were no timeouts or OOM events. The run receipts
retain memory observations; they do not claim a measured peak GPU footprint.
All diagnostic containers were removed, and sp8's final GPU process list was
empty.

The additional tool attempts use the same candidate image unless explicitly
identified as native host checks. Each receipt retains exact commands, failed
attempts, source identities and raw artifact hashes.

| Tool | Actual result | What remains outside its coverage |
| --- | --- | --- |
| Compute Sanitizer racecheck | Deliberate shared-memory WAW hazard detected; clean control passed; 21 candidate copy/gate cases passed with zero hazards | Global cross-stream ownership and CPU mirror epochs |
| Compute Sanitizer synccheck | Deliberate invalid warp mask detected; clean control passed; 21 candidate copy/gate cases passed with zero errors | Collective payload identity/length agreement |
| ASan + UBSan; TSan | Every deliberate host fault detected; clean controls passed; unchanged proxy simulator completed 31,000 collectives under each configuration | Actual NIC/GPU behavior; typed freshness of valid receive bytes |
| PyTorch CSAN | Torch stream-race control detected; ordered control passed; both real TileLang producer cases explicitly returned coverage-gap code 77 | The actual `copy_alloc` write is invisible to CSAN; CPU publication epochs are also outside its scope |
| Nsight Systems | 21 candidate cases passed; trace contains 1,604 kernel records, two streams and graph/NVTX activity | The deliberate assertion lost buffered GPU activity, despite retaining API/NVTX records; a trace is not an automatic race proof |
| Nsight Compute | One actual `router_partials_kernel` basic profile captured; selected router case passed | Replay/instrumentation changes execution; no production speed or race-freedom conclusion |
| CUDA-GDB and GPU core dump | Live debugger and fresh offline core reload preserve original control inputs 0/0, NaN bits `0x7fffffff`, consumer line and stage | A known deliberate-fault fixture, not the original failing model tensor |
| NVBit 1.8.1 official instruction counters | Direct and extended-launch controls instrumented; graph-aware counter also instruments stream capture | Explicit graph-node controls execute but report zero instructions; ordinary counter fails stream capture because it synchronizes |
| GPU-FPX detector and analyzer | Both compile; both fail direct clean control before output with `CUDA_ERROR_INVALID_SOURCE` | Neither numerical checker is qualified on this tested build; this is not a clean fault check |
| Nixnan | Exact source fails compilation because it copies `ChannelHost`, which NVBit 1.8.1 makes noncopyable | No runtime check occurred; this is a specific API mismatch, not proof that ARM64 or Blackwell is unsupported |

Receipts: [racecheck](sp8-tool-racecheck-c37-20261103T083451Z.json),
[synccheck](sp8-tool-synccheck-c37-20261103T090926Z.json),
[race/barrier controls](sp8-tool-sync-race-controls-20261103T083051Z.json),
[native host checks](host-native-results.json),
[CSAN](sp8-tool-csan-controls-20261103T082836Z.json),
[Nsight candidate trace](sp8-tool-nsys-c37-20261103T082316Z.json),
[Nsight controls](sp8-tool-nsys-controls-20261103T082016Z.json),
[Nsight Compute](sp8-tool-ncu-router-20261103T091103Z.json),
[debugger/core](sp8-tool-cuda-gdb-controls-20261103T082442Z.json).
The initial Nsight control build failed for a missing NVTX include path;
[that attempt](sp8-tool-nsys-controls-20261103T081914Z.json) is retained.
The first Nsight Compute invocation could not locate its separate installation;
[that attempt](sp8-tool-ncu-router-20261103T083215Z.json) is retained. The retry
mounted the existing installation read-only; nothing was installed on the host.

Numerical-tool receipts: [offline builds](sp8-tool-nvbit-build-20261103T091232Z.json),
[first controls](sp8-tool-nvbit-controls-20261103T091429Z.json),
[retry with the installed disassembler](sp8-tool-nvbit-controls-disassembler-20261103T091655Z.json).
All eight uninstrumented clean/NaN launch-form controls passed. Adding the
existing `nvdisasm` path fixed official-counter setup; it did not resolve
GPU-FPX's invalid-kernel-image error. Nixnan and GPU-FPX source was not patched
to manufacture a successful check. Any numerical checker could first observe
arithmetic consuming an already-NaN operand; that alone would not identify the
writer of stale bytes or prove a NaN was newly generated by arithmetic.

This narrows the tested memory-access hypotheses only. The gate fixtures use
finite fabricated operands and private fresh scratch. None of these runs
exercises a full target/DSpark forward, shared-buffer reuse beside the auxiliary
stream, distributed rank layout, or the failing serving step. Capturing the
original first nonfinite tensor remains separate work:
see the [proposed first-fault recorder](first-fault-design.md). That recorder is
not implemented. The existing census's attribution problems are documented in
the [source review](census-attribution-review.md).

## Preserved failure

`evidence.json` records source identities, hashes and paths for 29 existing
receipts collected from the switched TP3 run on sp5. Raw receipts remain under
the ignored `results/private/copy-fault-forensics/` directory. The collection
did not launch a workload or change serving state.

The recorded candidate-37 arm used copy limits 8/8/16 and dead-row threshold 0.
After the copy probe, the single-stream suite failed its c1 case. The saved
sp5 and sp7 engine logs contain `gate logits contain nan or inf`; the Python
trace observes the failure at an event synchronization and does not identify
the producer. The c8/c16 logs are duplicate snapshots of that failed engine,
not independent reproductions. The minimal shared-budget arm completed those
workloads. That counterfactual warrants investigation but does not locate the
first invalid value or prove the causal mechanism.

Matching baseline counters do not establish equal layouts across ranks: their
checksum incorporates consequential-read indices rather than the actual
per-request layout. Neither the first bad tensor nor target-versus-drafter
attribution is present in these saved crash logs.

## Diagnostic variable and workload

The first isolated diagnostic runs the existing candidate-37 dead-row combine
tests under Compute Sanitizer memcheck. Diagnostic mode combines sanitizer
instrumentation with disabled PyTorch allocation caching so initcheck can
observe fresh allocations. This is an explicit diagnostic interaction, not a
causal performance comparison. The source is vLLM
`5b7cfa37fee42f96bfafbe09b7d30e8c0eda8575`, tree
`af0fb1525fc6aa4149b18dee7d8c3ae5120b6f33`. The existing immutable sp8 image is
`sha256:e1ee7b1953b1131320746f87fe0cb11a673b53e9702bda63048ea3922768c15a`.
The host tool reports Compute Sanitizer 2025.3.1.0 / CUDA 13.0.

The isolated job must recheck holds and serving/GPU activity, acquire the local
GPU lock, and leave the serving clusters untouched. All output, errors, command
arguments, elapsed time and resource observations belong in its run receipt.
A collection failure, skip or timeout is not a successful GPU check. No
performance inference is valid under instrumentation. The experiment does not
request a serving restart or authorize bypassing another cluster holder.

Generate the broader command plan without running it:

```sh
python3 experiments/2026-11-03-copy-fault-forensics/plan_sanitizers.py \
  --sanitizer-path /opt/compute-sanitizer/compute-sanitizer \
  --source-root /opt/spark-ds41f/candidate/vllm \
  --report-dir /reports
```

The plan names existing copy-state, input-combine, gate/router and indexer tests.
It runs memcheck before initcheck and records each test's limited coverage.
It does not execute commands or contain a full TP3 serving reproducer. A clean
microtest cannot clear the asynchronous serving failure.

The first pytest attempt could not collect tests because the image lacks
`tblib`, imported by the repository-wide test configuration. Its failed receipt
is retained. The two selected dead-row tests take no fixtures; the retry uses
`--noconftest` to execute their unchanged assertions without importing that
configuration. This omits suite-wide cleanup and logging/HTTP reset fixtures;
it does not install or stub a dependency. The exact invocation and source
hashes belong in each run receipt.

`sanitizer_control.cu` provides separate clean, out-of-bounds and uninitialized
read controls. It uses two four-byte allocations and one GPU thread. It is a
standalone diagnostic fixture, never part of the inference runtime. Compile
and invocation commands, source/binary hashes and actual reports must be
retained together; a failing process alone is insufficient to validate a tool.

## What each check can establish

- Memcheck: invalid addresses and misaligned memory accesses.
- Initcheck: reads of memory the tool has not observed being initialized.
- Racecheck: shared-memory hazards inside kernels. This is not a general check
  for races between CUDA streams accessing persistent global tensors.
- Synccheck: invalid kernel synchronization operations. It does not establish
  correct ownership/versioning of CopyDrafter buffers.
- A first-fault recorder: the earliest nonfinite input or result, with rank,
  model, layer, row identity, masks and shapes. Save original operands before
  any clamp or replacement, then replay that operation in isolation.

Use positive controls on the same GPU/toolchain before treating a clean
sanitizer report as evidence. Instrumentation changes timing; keep graph/PDL
settings and avoid making `CUDA_LAUNCH_BLOCKING` the primary diagnostic. A
blocking run can be a separate comparison, not a substitute for the original
execution order. Full distributed coverage requires instrumentation on every
rank, not only the head process and its local children.

Tool definitions: [NVIDIA Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).
Version-specific options must be checked against the installed tool.

## Remaining fault-attribution limits

The per-step records and isolated mechanisms establish the unequal-layout
fault. A device clamp applied after budget selection cannot make those earlier
layouts identical. The first failing GPU forward, and the precise bytes that
first became nonfinite in the original serving crash, remain unknown. The
observed limit 16 can also be an
initial/reset default; it does not prove that a particular 16-token proposal
was copied to that rank's host mirror.

The singleton limits buffer is also read on an output-copy stream while later
steps can write it on the main stream. The reviewed source has no direct
reader-to-next-writer dependency; whether the surrounding scheduling prevents
overlap in the failing run remains unproved. An attempted deterministic test
using CUDA stream memory gates was withdrawn before GPU execution because it
could introduce an artificial deadlock. It provides no evidence of a
production race.

The router assertion follows reduction of gate projection partials. Without
the original gate input and each partial, it cannot distinguish upstream NaNs,
bad projection output, corruption after projection, or finite-value overflow
in the reduction. A diagnostic that replaces bad partials with zeros loses
this evidence and changes inference.

## Local validation

```sh
python3 -m unittest discover \
  -s experiments/2026-11-03-copy-fault-forensics -p 'test_plan_sanitizers.py'
bin/spark doctor
git diff --check
```

The five planner tests passed and doctor reported `configuration OK`.
`bin/spark build check` was also attempted: this worktree's promoted build
directory `.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-31ac0765` is not
prepared, so that check did not pass. No build was prepared or launched as part
of the diagnostic work; GPU runs used the existing immutable candidate image.

No runtime patch or build input is changed by this experiment. Results and
limitations of actual GPU attempts are recorded separately; a planned command
is not a completed test.
