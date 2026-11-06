# CED row-local global-KV projection

Status: experimental alternative; first native GPU boot passed 367 checks with
one skipped. Repeat, schedule screen and serving results are pending. Source
commit `51c1a32cd78cfb0a14087e34dfcc06be30928a0e` is published on
`https://github.com/christopherowen/vllm/tree/feat/ced-local-projection`.
The focused compact-carry source review remains
https://github.com/christopherowen/vllm/pull/4; deployment review is
https://github.com/christopherowen/spark-ds41f/pull/922.

## Identity and intended variable

Deployment base: `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`.
The immediate source base is the compact-carry candidate
`2b320049a5df8e62e1b3ee70f53304d6d55a15e3`. This experiment changes only where
ratio-one global-KV projection runs and what it gathers. It applies after the
compact-carry experiment; its separate source series has 560 patches. See
`identity.json`, `source.json`, `upstreams.lock.json` and `vllm/series`.

Instead of gathering every 5120-column normalized input, each rank projects its
local rows with the native BF16 compressor weight and gathers 512-column values.
The full-row normalization, global KV writes and index-key construction retain
all encoder rows and original metadata. Query construction gathers only selected
normalized replay rows with the residual carry.

TP3 already has the full weight. TP4 retains one additional 512-by-5120 BF16
weight before memory profiling: 5 MiB, while keeping the original shard for
decode. Checkpoint 1017 has one ratio-one owner (layer 20). The projection uses
exactly the original shard's five blocked K parts, including when the local row
count crosses 128. Decode keeps its original projection and workspace. The new
kernel is declared before the kernel cache is frozen.

The dense-path check counts every compact slot, SP padding and the new projected
value gather. TP3 previously did not gather compressor columns, so its additional
row gather must earn back that cost; for 128 replay rows, 385-390 total rows keep
the full path. At 391 rows the padded communication estimate favors the local
path. This is a byte bound, not a prediction of latency.

## Mechanism bounds and alternatives

For hidden width 5120 and four residual streams, the full carry has 51296 bytes
per row (BF16 hidden/residual and FP32 mixes). The local path gathers
`padded_rows * 512 * 2 + world_size * replay_rows * 51296` bytes per rank.
The old path gathers `padded_rows * 51296`, plus the existing full-row compressor
column gather on aligned TP. These are gathered tensor bytes, not wire traffic.
At TP4, 16384 rows and 128 replay rows, this is about 41.05 MiB versus 817.5 MiB
for the original boundary including its compressor gather, or 196.05 MiB for
the compact-carry candidate. At one replay row it is about 16.20 MiB.
None of those ratios predicts whole-model throughput.

`screen_projection.py` compares the native column projection with eighteen local
projection schedules: M tiles 32/64/128, N tiles 64/128, and two to four pipeline
stages, retaining K tiles of 64 and five blocked K parts. Twelve schedules use
128 threads; six larger-tile schedules also test 256 threads. Every
schedule must first match native finite and nonfinite output bits. The 128-by-128
128-thread combination is omitted (its 256-thread version is tested) because three FP32 tile accumulators alone
would require 384 registers per thread before operand/pipeline state. The screen
retains compiler failures, failed equality checks and all timing samples. It
measures warm reuse and a 128 MiB cache wash, separately at the TP3 6144-row and
TP4 16384-row chunk shapes. These are isolated one-rank compute timings, not
collective or serving measurements. A repeat process boot must recheck the
winning schedule and default; a schedule is not selected from compilation alone.

Owner-directed replay exchange could remove the remaining world-size multiplier
on compact slots. At TP4 and 128 replay rows the largest further tensor-payload
saving is 18.79 MiB; with one replay row it is 150.28 KiB. That requires a
GPU-resident ownership-aware exchange that handles dynamic multi-request replay
and graph padding without a host synchronization. The current grouped-gather ABI
has fixed equal rank capacities. This alternative remains open if hardware timing
shows the residual exchange is material. No global optimum is claimed.

## Completed checks and retained attempts

All substantial work ran on Spaceport, with one CPU and a 4 GiB limit. The Mac
performed lightweight coordination only. `receipts/` retains successful and failed
attempts; a large failed-copy log is compressed without dropping its contents.

- CPU runs: 140 passed/1 skipped, then 141/1 after the actual attention route test,
  then 151/1 after padded payload dispatch tests. The final run took 10.39 s with
  peak RSS 1263232 KiB. Host MemAvailable was 120500180 KiB before and 119834504 KiB
  after; those are endpoints, not a continuous minimum. These stand-ins prove
  orchestration and ownership, not native arithmetic or distributed execution.
- Both 128- and 512-column native BF16 projection kernels compile offline for
  SM121a. The first attempt used plain SM121, which rejects the existing
  warp-specialized register-reconfiguration instructions. The corrected target
  passed; peak process RSS was 1248972 KiB. No GPU device was opened.
- New lines pass Ruff 0.14.0 and whitespace checks. Inherited diagnostics remain
  unchanged. Native GPU tests were added for full-versus-local TP3/TP4 projection
  and rank-order mHC entry, including padding, row-dispatch boundaries and
  nonfinite payloads. They have not run yet.
- A fresh upstream checkout replays all 560 patches and reproduces candidate head
  `51c1a32cd78cfb0a14087e34dfcc06be30928a0e`, tree
  `5d8d1d1778c83a41d6dfb7364d1a18dad07df021`. The full series passes whitespace
  checks. The inherited patch warning is retained in the receipt.
- Default and candidate doctors pass. Complete build inputs pass `build check`.
  A first cached-input copy requested mandatory reflinks on a filesystem without
  support and failed; an ordinary bounded copy recovered it before preparation.
  No image has been built or GPU-smoked for this source.
- The six-arm TP3 plan now resolves. An initial compact-carry arm used a relative
  path interpreted from repository root and was rejected; the full repository
  path corrects it. TP4 planning still refuses without the site's qualified ring
  map. No map has been invented and no topology change is authorized.

## Hardware comparison and restoration

The owner authorized hardware testing and the coordinated operations needed for
candidate testing and restoration. Existing holds and pending requests still
apply. The first requested job is at most 30 minutes of kernel-only qualification
on an idle dgx4. It does not replace serving containers. Two clean process boots
must retain the exact source, dependency image, compiler/cache state, GPU identity,
clocks, temperatures, memory and commands. Kernel tests may mount the six changed
Python files from the published clean source over the frozen parent dependency
image; this is explicitly a kernel test environment, not a built serving image.

`control-tp3.json`/`control-tp4.json` name the frozen unreleased candidate-36 parent
(head `6d4e49d1cdbb6513819e47145e0e52fdab327b6b`), which can be measured for causal
comparison. The 1030 released reference is read from immutable recorded suites
and is never booted again. `qualify-tp3.json` and `qualify-tp4.json` keep separate
boot/repeat arms for that control, compact carry, and local projection. Their dry
runs are plans only: the main-only lab runner cannot execute an experiment branch.

Each candidate profile explicitly names the published deployment experiment
branch. In an exclusively owned window use only coordinated `bin/spark cluster`
plans and applications, with the profile named and the plan's live/shape lines
reviewed. All nodes must be clean and detached at the same published commit and
use one verified image digest. Do not merge to satisfy the queue, weaken the main
publication guard, or allow a main-syncing queue runner to race the experiment.

Capture the original serving commit and restore profile before any replacement.
Currently that profile is candidate-36/tp3-dev.json; it must be revalidated live
at handoff. The ordinary lab watchdog records the profile but not a restore
revision, so a branch test needs a restoration controller pinned to the original
commit that first coordinates sync back, then restarts and verifies that profile.
Do not begin branch serving tests until this fail-safe is validated. For the first
kernel-only window the serving checkout and profile remain untouched, so the
ordinary watchdog can verify and retain the already-running service.

The full serving gates remain outstanding: native correctness/batch invariance,
32K/131K/262K/524K/1M prefill, fixed-corpus long-needle retrieval, mixed short-request
TTFT and startup/steady memory. Preserve exact future prompt bytes/token IDs and
hashes with raw runs. Historical 1030 prompts did not retain those bytes; their
identity limitation cannot be repaired retroactively. Report absolute and percent
changes with 95% intervals and receipt links; distinguish within-boot repeat
variation from between-boot variation. No throughput benefit or promotion is
claimed before those measurements.

The acceptance criterion also permits a demonstrated correctness repair with
preserved performance. Such a repair needs an old-fails/new-passes reproducer on
the actual path and appropriate before/after evidence excluding a material loss.
These layout candidates do not claim an existing correctness defect. The recorded
prefill comparator uses a 0.5% tolerance and Welch intervals; an interval that
still admits a material regression is inconclusive, not proof of no loss.

## Native GPU boot 1

On dgx4 in exclusively owned window `dgx1-1793922629`, the published source
passed 367 native CED, mHC and compressor tests with one skipped in 286.98 s.
This includes full-versus-local native projection bits at TP3/TP4 shapes and
rank-order mHC entry across real chunk sizes, padding and nonfinite inputs.
The test process peaked at 3633984 KiB RSS. Full commands, mounted-file hashes,
dependency image identity and five-second memory/clock/temperature samples are
in `receipts/native-boot-1/`. This is one kernel-test process boot, not a serving
benchmark, and it does not certify actual multi-node collective behavior.
