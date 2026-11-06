# CED row-local global-KV projection

Status: experimental alternative. The complete image passed standard GPU smoke
on SP5-SP7, and two actual TP3 collective boots each passed 20 checks per rank.
Six standalone native runs each passed 368 checks without skips. Full-model
quality, batch invariance and qualified performance remain pending. Source commit `51c1a32cd78cfb0a14087e34dfcc06be30928a0e` is published on
`https://github.com/christopherowen/vllm/tree/feat/ced-local-projection`.
The source review now includes both compact carry and local projection:
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
  nonfinite payloads. They passed in both native boots documented below.
- A fresh upstream checkout replays all 560 patches and reproduces candidate head
  `51c1a32cd78cfb0a14087e34dfcc06be30928a0e`, tree
  `5d8d1d1778c83a41d6dfb7364d1a18dad07df021`. The full series passes whitespace
  checks. The inherited patch warning is retained in the receipt.
- Default and candidate doctors pass. Complete build inputs pass `build check`.
  A first cached-input copy requested mandatory reflinks on a filesystem without
  support and failed; an ordinary bounded copy recovered it before preparation.
  No image had been built at that stage; complete-image evidence is recorded below.
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

## Native GPU boot 2 and timing refinement

The second clean test-container boot again passed 367 checks, with one skipped,
in 284.90 s. The skipped existing partial-state offset test requires just over
8 GiB of additional GPU allocation and did not fit the bounded 16 GiB container;
none of the new CED/local-projection cases skipped. Peak PyTorch allocation was
4115485696 bytes, reserved storage 5802819584 bytes, and process RSS 3681024 KiB.
See `receipts/native-boot-2/` for exact identities and telemetry.

The discovery screen passed finite/nonfinite bit equality for all eighteen
layouts at both chunk shapes. Its ordinary event brackets also include gaps while
Python enqueues the kernel; the raw timings contain substantial outliers. The
parent callable additionally formed a weight view during that bracket. Those
results remain discovery data, not the basis of a throughput claim. The corrected
screen pre-binds the parent view and captures external CUDA timing-event nodes
inside graph replay, with any cache wash before the timed kernel. Both corrected process boots completed, and every schedule passed its bit gates.
They still do not qualify performance: the runner recorded GPU temperature but
did not enforce or record the required hottest-zone baseline and installed
performance fan curve. All three screens remain exploratory. The gap does not
invalidate native equality results. The service journal showed the fan service
starting at 00:20:06 UTC, after these screens had finished; the policy cannot be
established retroactively.

The raw discovery and two corrected runs are retained in
`receipts/screen-discovery-1/`, `receipts/screen-corrected-1/` and
`receipts/screen-corrected-2/`. `screen-comparisons.json` uses independent process
means with Welch 95% intervals, retaining every raw sample and outlier. It does
not pool within-process samples as independent boots. The 64-by-128 tile with
256 threads and three stages is a provisional follow-up candidate, not a selected
winner. In particular, the corrected TP4 comparison with the default still
admits a material regression, even without the fan-policy gap.

The next runner uses the unchanged production cooling helpers through a private
local command transport, touches only the owned idle kernel host, requires its
hottest zone below 55 C before launch, and verifies the installed performance
curve remains active. It records service actions, thermal zones, graphics clocks,
performance/throttle state and memory. A curve it starts is stopped on exit;
restoration failures fail the run. Every local policy action rechecks the owned
window. No timing run bypass is exposed. Five CPU failure-path tests pass, covering a
missing/failed curve, hot-host timeout restoration, preexisting active service,
ordinary start/stop restoration, and stop failure; see
`receipts/cooling-tests-1.log`.

The first window closed at 00:19:47 UTC with candidate-36/tp3-dev verified still
serving and the hold removed; see `receipts/window-close.txt`. No serving container
was replaced. The next requested job is a 25-minute DGX4-only timing screen,
including four fresh test-container processes, all eighteen schedules at both
chunk shapes, finite/nonfinite gates, and cooling before each process. It will
follow the explicit shared-hardware reservations; no request waiter is active.


## Python-only image preparation

`image_native_reuse.py` is an experimental CPU-only image recipe for these six
Python changes. It uses the already available immutable Spaceport C36 image
`sha256:3795255f485ea5c1a9db54c742921dd9b2917daad0799361e5c6f56a70119713`.
That image has the same locked source identities as the DGX4 native-test parent,
but a different image digest. Its control must therefore be this exact parent,
not the DGX4 image or a historical timing run.

The recipe first runs the normal complete build-input check, verifies every
tracked parent source file against the frozen control Git tree, and refuses any
source change outside Python implementation/tests. It copies the existing exact
candidate Git export with required license files, verifies every tracked source
file again, compares both compiled vLLM extension hashes, and checks that the
candidate inherits the parent's entire layer chain. Other dependency tree labels
must remain equal. The unique tag is never overwritten. This is deliberate reuse
of unchanged native binaries, not a native rebuild.

Preparation and verification use one CPU and 4 GiB on Spaceport, with a 48 GiB host
reserve and the normal image-build lock. The Dockerfile has no RUN step and no
package installation. CPU verification explicitly uses runtime runc with no GPU
exposure. The existing checked build context is reused instead of copying another
source tree or base-image archive. `--apply` refuses an unpublished or dirty
recipe. A successful CPU build still requires the normal `bin/spark build smoke`
in an exclusively owned hardware window before any serving qualification.

`native-reuse-tp3.json`/`native-reuse-tp4.json` select the resulting candidate tag;
`native-control-tp3.json`/`native-control-tp4.json` select an immutable alias of that
exact parent digest. Before distribution, the alias must be created only if absent
and verified against the parent ID; an existing mismatched tag is an error.
Build once, then distribute the same verified OCI identity to all configured
ranks through the coordinated tools. These profiles do not authorize a shape
change or substitute for the original A/B source pins and evidence.


The first apply attempt on Spaceport stopped at the normal build lock because
another image build held it. No image was assembled and the lock was not bypassed;
see `receipts/native-reuse-image-1.txt`. An independent bounded CPU check then
verified all 7810 parent source files against the frozen Git tree and retained the
two compiled-extension hashes in `receipts/parent-source-check-1.txt`. The candidate
image and its GPU smoke remain pending.


## Fixed long-context retrieval companion

`prepare_needle.py` prepares the immutable corpus on CPU with the exact checkpoint
1017 native `deepseek_v41` tokenizer. It archives UTF-8 prompt bytes, little-endian
32-bit chat token IDs, source/tokenizer hashes, sizing attempts, expected values,
and actual prefix token depths. Three independently seeded prompts per size each
contain three unique registry values, at 10%, 50% and 90% of document characters.
The requested lengths are 32768, 131072, 262144, 524288 and 1048576. Actual token
counts are recorded; the largest reserves 256 context positions for generation.
The generator refuses an existing output directory instead of changing fixtures.
Bulk corpus files stay in remote results storage, outside Git.

`needle-suite.json` uses the existing servebench model step, concurrency one,
128 output tokens, greedy decoding and thinking disabled, through
`bin/spark bench --suite`. This retains the ordinary live-identity, cooling, fan
curve and memory gates. It is a quality companion, not a substitute for the
release suite's streamed prefill rates or mixed-request TTFT. Fresh cache salts
come from the existing model runner. `check_needle.py` decodes returned token IDs
on CPU and requires both exact prompt-ID identity and the exact expected JSON
mapping; it retains raw output text and individual value-presence diagnostics.
Fixture creation and grader checks are preparation only, not model-quality proof.


## Independent SP5-SP7 correctness cohort

The ready bounded payload is two fresh native-test boots on each of SP5, SP6 and
SP7. Each uses source `51c1a32`, the already-present immutable C36 dependency image
`3795255f`, six read-only Python mounts and the same native test set used on DGX4.
This proves native bits on each rank's hardware; it does not prove a distributed
collective or whole-model prefill performance. There is no timing mode. Containers
are limited to one CPU/16 GiB, host memory must remain at least 48 GiB, every boot
starts with the hottest zone below 55 C, and GPU or thermal-zone readings above
70 C stop only the owned test container. A process is bounded at 600 seconds;
passive cooling waits are separately bounded at 600 seconds. Every stop and all
observations are retained. A new batch starts only with sufficient time for its
bounds and 15 minutes reserved for coordinated recovery, within a 45-minute window.

The existing published recovery authority is
`sp5:/home/chris/spark-lab/checkouts/clustered-serving` at
`e9eab4604cf4e46996d28c0bfba8c9fe75f7d282`, clean and identical on all three nodes.
`experiments/2026-11-04-token-lsa/paged/restore.json` extends the tracked
`experiments/2026-11-04-clustered-indexer/serving/restore.json`. Its node map selects
SP5/rank0 (10.30.0.13), SP6/rank1 (10.30.0.14), SP7/rank2 (10.30.0.15) on the existing
TP3 switched group. Restore image is
`sha256:e1ee7b1953b1131320746f87fe0cb11a673b53e9702bda63048ea3922768c15a`,
vLLM tree `af0fb1525fc6aa4149b18dee7d8c3ae5120b6f33`, sparknet tree
`8cc28a5cd3e683c1f114ed843453e3ce34f323e0`. The unrelated default checkout at
`806c5797` is not this deployment's authority. No serving checkout is changed by
the kernel cohort and no new topology is introduced.

`sp567_cohort.py` reuses that exact published controller's coordinated recovery
functions, selecting the exact live restore-profile name. It adds exact-token
holds on every participant, fresh scope and idle checks, an independent watchdog
with a verified startup handshake, native-container cleanup, a restoration lock,
and independent all-rank identity verification before heartbeat join and release.
Nine CPU guard tests cover foreign/legacy holds, mismatched scope, incorrect global
holder tokens, recovery targets, verification-before-release and retained recovery
failure. Restoration failure retains ownership and prevents an automatic restart
loop. Both start-plan live/shape lines are reviewed by the published recovery
function; no change-shape bypass is used. The ordinary startup memory guard remains
in effect. Requests are honored after the current native batch.

Ownership reconciliation and fresh live checks remain mandatory before acquisition.
The DGX global hold must be absent or explicitly scoped, by its holder, to DGX1-3
with matching recovery/watchdog targets; another scope blocks this controller.
All four local hold/request names are checked on every Spaceport, and 30 seconds
must pass with no running/waiting requests or changed completion counters. SP1-4
is a separate active cohort and is never touched. The prior stock-cooled SP7 native
run that stopped after 32/45 requests remains negative evidence, not a completed
quality reference. Fan-control is absent on these Spaceports, so there is no
compatible qualified timing reference for this job; immutable 1030 runs remain
recorded only. Further model/prefill qualification is still outstanding.

The SP5 controller has no route to DGX1. `observe_global.py` is a lightweight
management relay running on the coordinator Mac: every three seconds it reads
DGX1's live hold over authenticated SSH and atomically writes a timestamped
observation on SP5. Acquisition, stop and each batch refuse failed observations,
reads taking over ten seconds, observations older than fifteen seconds, or a
foreign unscoped/overlapping hold. The independent restoration watchdog runs on
SP5 and does not depend on the relay. Relay loss cannot authorize a new batch.

The first SP5-SP7 attempt acquired token `fe8cfe50be3540da97479c63784138f5`
and stopped C37, then failed before GPU execution because the generated remote
phase-update string contained an unescaped newline. That same phase update also
blocked its automatic restoration. The retained failure prompted immediate
recovery through the unchanged published C37 recovery functions, retaining the
owned leases until independent verification. The fix uses an unambiguous newline
expression, compiles every embedded remote program, and executes the actual phase
program against a temporary hold in CPU tests. Recovery now records and stops its
heartbeat on failures anywhere in the recovery path. No native result is claimed
for this failed attempt.

The failed attempt restored C37 at 2026-11-06 01:54:06 UTC. The live doctor and
independent checks verified all three clean `e9eab460` checkouts, exact C37 digest,
exact paged restore profile, and no extra running containers before releasing the
leases. Complete receipts are retained under `receipts/sp567-window-1/`. The phase
writer is now exercised before the stop operation as well as covered by CPU tests.
Six CPU-only grader controls also pass; these validate grading behavior, not model
quality, and are retained in `receipts/needle-grader-controls-1.txt`.

The second image attempt built immutable image `539193702d10`, but its complete
source check rejected the generated build-export `.dockerignore` (the only tracked
file mismatch). The image has not been distributed or used for GPU work. Both the
full log and failure receipt remain in `receipts/native-reuse-image-2.*`. The
corrected recipe uses `git archive` at the pinned source head, streaming its exact
tracked files beneath a `source/` context prefix plus the published Dockerfile.
That preserves the original `.dockerignore` without treating it as a context
filter, requires no additional source tree or image archive on disk, and retains
the same source/native checks and resource bounds. The corrected unique tag adds
`-git`; the rejected tag is never overwritten.

The third image attempt stopped before Docker build when the stream guard did
not accept Git archive's `source` root-directory entry. No corrected-tag image
was produced; its failure receipt is retained. The guard now explicitly accepts
that directory and a CPU-only consumer independently verifies every streamed
file's Git blob, file mode, symlink, file count and Dockerfile before another build.

The fourth image attempt was rejected by Docker's stdin archive detector before
image construction. Git's PAX comment preceded the first ordinary member, beyond
Docker CLI 29.6.2's 1024-byte detection peek. The stream now begins with the plain
published Dockerfile entry; all Git entries and content remain unchanged. The
independent stream check also requires that first header. This behavior was checked
against Docker CLI's `detectArchiveReader`/`isArchive` implementation at v29.6.2.
The failed attempt remains in `receipts/native-reuse-image-4.*`.

The second SP5-SP7 native window completed: **368 tests passed with zero skips on
each of two fresh boots on each of the three nodes**. Source remained `51c1a32`,
and the controller was published `ac777ffa`. The maximum recorded thermal zone
was 61.0 C, maximum GPU temperature 52 C, and minimum host memory available
103.38 GiB. Every run reported 8,889,463,808 peak CUDA-allocated bytes. All commands,
raw test output, telemetry and failures are retained under `receipts/sp567-window-2/`.
C37 was independently verified at clean `e9eab460` on all nodes and the leases
released at 2026-11-06 02:10:24 UTC. These are native correctness results only;
distributed collectives, whole-model serving and qualified performance remain
unmeasured for this candidate.

The fifth image attempt passed complete CPU identity verification. Immutable
image `sha256:b1dac2d1ecf9bde9a2412befa40b081745d69d1bc86f76f66bc99e53eb0765bf`
contains all 7,811 tracked files at source `51c1a32`/tree `5d8d1d17`, preserves
both compiled-extension SHA-256 values and all 57 parent layers, and adds one source
layer. It records published deployment `4e399935`; its labels were independently
checked against that commit and source head. The CPU build kept at least 115.11
GiB host memory available. The control alias `spark-ds41f:ced-control-3795255f485e`
was created only if absent and verified against immutable parent `3795255f`.
Exact identities and receipts are in `image-native-reuse.json` and
`receipts/native-reuse-image-5-*`. The image has not been distributed, GPU-smoked
or served; these gates remain required.

## Image smoke and actual TP3 collective follow-on

`sp567_cohort.py --job collective` selects a separately recorded 90-minute maximum
window. It uses the same exact all-node C37 recovery authority, lease handshake,
fresh global scope relay, idle guard, watchdog and verification before release.
After a coordinated stop, SP8 streams the one verified image to every SP5-SP7 node
with `bin/spark --cluster-config .../sp567-image.json build distribute --apply`.
The profile selects the existing TP3 switched node map and disables serving
launches. The transfer checks portable ordered-layer/runtime-Config identity,
refuses a conflicting existing immutable tag, checks every owned lease and the
48 GiB memory reserve while copying, and stops at 780 seconds. SP8 uses one CPU
and 4 GiB behind the shared CPU lock and never opens a GPU. The CLI's text plan
omits the head, so the guarded helper also records the actual missing targets
from all three image inspections; the normal apply verifies all three copies.

Each participant then runs the standard `bin/spark build smoke --image ...`
command. A published Docker wrapper adds only the exact token/name, one CPU,
network isolation and a process limit to the standard 16 GiB GPU-smoke container.
The normal import/native checks are unchanged. Two fresh three-rank containers
then run `collective_gate.py`, using native PyNCCL grouped gathers and the actual
compressor projection factory at 385, 390, 391, 1025 and 16384 rows. Every rank
checks compact carry bytes against independent indexed copies, including arbitrary
NaN/signed-zero bit patterns, padding, invalid/duplicate selections and empty
selection. The projection compares local rows plus real all-gather against the
full-row native prefill projection, finite and nonfinite, while checking the TP3
byte-cost crossover. This is synthetic collective correctness, not whole-model
serving or a timing benchmark. No source or precision parameter changes.

Workers retain the 55 C start, 70 C stop, 48 GiB memory floor, 16 GiB container
limit and bounded cooling/process waits. No new batch starts without 35 minutes
remaining for its bounds and restoration; pending requests yield after the current
complete batch. Every command, output, error and thermal/memory sample is retained.
The follow-on has been prepared but has not yet run; its actual outcomes must be
recorded before claiming image smoke or cross-rank qualification.


## Complete image smoke and first TP3 collective setup

Window `124def42f3f7495ab9c768f98ab9452a` distributed and verified the complete
immutable candidate image on SP5-SP7. The unchanged standard GPU smoke passed
on every node. All three first collective processes failed during initialization
because the standalone harness lacked a current VllmConfig; no candidate byte
check ran, and the repeat boot was not attempted. The original C37 service was
independently verified restored on all three nodes at 03:04:08 UTC, with leases
removed. All raw attempts and recovery evidence are in `receipts/sp567-window-3/`.

The repaired harness enters an explicit TP3 external-launcher configuration
context and resolves transport settings from the published SP5-SP7 profile and
node map. It retains native PyNCCL, byte equality and thermal/memory guards.
The repair has not yet passed a GPU collective run.


## Fixed cold-prefill inputs

`prepare_prefill.py` freezes three prompts at each requested size, using the
standard source-text cold-prefill recipe and seeded order. It archives complete
UTF-8 prompts, native completion token IDs, payloads, tokenizer and source hashes.
The top requested size is 1,048,000 so it fits the standard 1M-context size filter;
actual token counts are authoritative. Cache salts must be fresh for every replay.
This is CPU-only fixture preparation, not a benchmark; it does not reconstruct
the unavailable historical prompt bytes. Preparation results are recorded below.

The first preparation attempt stopped before writing prompts because the frozen
CPU dependency image does not include Git. Its log and capped-run receipt are
retained. The script now receives the exact revision from the host runner, which
verifies a clean published checkout before mounting it read-only.


## Two successful actual TP3 collective boots

Window `1c7f2a1738db457ea5ffabfaf2b2373a`, controller `a0524276`, passed
the complete-image smoke on all three nodes, followed by 20 synthetic checks
per rank in each of two fresh containers. The gate used native grouped PyNCCL
and the ordinary vLLM TP3 communicator with the published transport settings.
It checked five carry tensors with finite and arbitrary BF16/FP32 bit patterns,
invalid/duplicate/empty selections, padding and the 385/390/391-row byte guard,
plus 1025 and 16384 rows. Actual compressor projection factories matched full-row
results byte for byte, including nonfinite values and five blocked K parts.

All nine process receipts, raw output and recovery records are retained in
`receipts/sp567-window-4/`. C37 was independently verified restored at
03:14:20 UTC on SP5-SP7 and all leases were removed. These are synthetic native
correctness results; they establish neither full-model invariance nor throughput.

The standard fixture preparation completed with 15 prompts; all archived file
hashes, payload text and token-file lengths verified. Actual token lengths are
recorded in `receipts/prefill-standard-fixtures-summary.json`; the standard text
recipe can fall materially below its nominal length. `--sizing exact` therefore
adds an explicitly separate companion, adjusting only document length to land
within 64 tokens below each requested size. It records every sizing attempt and
keeps the standard corpus unchanged. Exact-size preparation results are recorded below.

Exact-length preparation completed with 15 prompts. Every input is within 64
tokens below its requested length; hashes, payload text and token-file lengths
verify. Three actual top-size inputs are 1047975, 1047974, 1047998 tokens.
The corpus remains remote at `results/private/ced-local-projection/prefill-exact-v1/`;
its manifest, complete sizing attempts and CPU run receipt are retained under
`receipts/prefill-exact-*`. No model request has used these inputs yet.


## Prepared full-model correctness cohort

The serving mode runs C36 twice and the combined CED candidate twice
on the same existing SP5-SP7 group, with identical C36 serving arguments and
separate fresh containers. It reuses the published CED cross-shape, temperature-0
wide determinism (with and without logprobs), and seeded temperature-1 clients.
Every HTTP request body is archived before sending. Strict summaries and raw
scenario checks treat negative functional results as failures even when a client
exits zero. Negative results may be characterized by subsequent correctness-only
arms; any failure blocks performance and selection. No timing path exists.

The 120-minute controller retains 35 minutes before beginning another arm, uses
the same independent C37 recovery watchdog, polls every node for thermal and
memory limits, cools below 55 C before each command, stops above 70 C, and retains
the published 5 GiB startup / 3 GiB steady host reserves. Clients are limited to
one CPU and 1 GiB address space. A failed monitor terminates the active command
before coordinated restoration. Raw per-node startup/steady observations, image
identities, complete bounded engine logs, and cross-boot digests are retained.

The serving profiles use the same existing node-local `/cache` source as the
published C37 restore profile (`{home}/projects/spark-ds41f/cache`), while all
serving source authority remains in the clean private deployment checkout. No
cache data or ignored files are deployed. The first dry plans retained their
expected publication/synchronization and missing-cache-path refusals.

Window `f6fcd3eec45847d698778927a493167c`, controller `0b71390d`, synchronized
SP5-SP7 and booted the first C36 control arm (`control-1`, image `3795255f`)
with startup memory guards active (`5,488,353` KV tokens; minimum host memory
available `8.77 GiB` on SP5, `10.59 GiB` on SP6 and `10.34 GiB` on SP7; hottest
zone during startup `60.7 C`). Pre-probe passive cooling started below `55 C`
(`50.7 C` on SP5, `51.3 C` on SP6 and `49.3 C` on SP7). During `control-1`'s
first CED probe (`ced_cross_shape.py`), `/tokenize` and the first three
completion requests (`8208` repeat 0/1 and `16352` repeat 0; the two `8208`
repeats matched token and logprob hashes `2b5ef3a8d5f79900` /
`d75204d9af786ff7`) completed before SP5's `thermal_zone0` and `thermal_zone4`
reached `73.2 C` on the fourth completion request (`16352` repeat 1; GPU
`63.0 C`, SP6 hottest `66.3 C`, SP7 hottest `63.1 C`, minimum host memory
available `7.27 GiB` on SP5).

The `70 C` thermal guard terminated the active client (`-15`) and triggered
coordinated restoration. C37 (`e9eab460`,
`experiments/2026-11-04-token-lsa/paged/restore.json`, image `e1ee7b19`) was
restored, independently verified across all three nodes, and all holds were
removed at `2026-11-06 03:59:22 UTC`. Neither `control-2` nor the candidate
arms ran; this records a stock-cooling thermal limit on SP5 under sustained 16K
prefill rather than a candidate correctness result. All three `control-1`
engine logs were recovered and archived with the five recorded HTTP request
bodies, startup/probe telemetry and restoration receipts in
`receipts/sp567-window-5/` (no pre-shutdown engine fault markers; only
post-failure `SIGTERM` shutdown traces when restoration began at `03:57:18 UTC`).

`serving_gate.py` now wraps probe execution in `retained_engine_logs` so every
rank's bounded engine log and `engine-logs.json` are collected before recovery
replaces the containers, without masking the original probe exception or
stopping on the first rank with an issue (`receipts/serving-guard-tests-4.txt`,
22 guard tests passing).

To prevent back-to-back 8K/16K cold prefills and concurrent determinism batches
from stacking thermal rise past `70 C` on stock-cooled SP5, `serving_gate.py`
and `record_requests.py` now require two consecutive `< 50 C` thermal-zone and
GPU readings across SP5-SP7 before each probe, after each `ced_cross_shape.py`
completion request (outside the request's timed bracket), after each
`together(...)` concurrent batch join, and after long or warm sequential
requests (`receipts/serving-guard-tests-5.txt`, 25 guard tests passing).
