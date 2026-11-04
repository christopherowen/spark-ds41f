# Native token-id inputs for scoring

**GPU performance gate passed for all 120 cases.** The final source retains
native integer widths and layouts and selects only GPU programs qualified on
both timing boots. Two fresh final boots verify exact code/input identity for
every case, pass 673 correctness checks each, and decode 56 IKET launches each,
including the shared-staging path. Full-model and host enqueue qualification
remain before promotion. See `performance-gate.json` and `paired-results-final.json`.

One variable: selected-token score kernels consume the token-id dtype and
strides supplied by their producers, replacing repeated int64 casts and
contiguous materializations. This builds on the selected-logit output patch
in PR #893; both remain unpromoted source candidates after candidate-38.

The log-softmax consumes int32 or int64 matrices and the producer's contiguous validity masks.
Top-k reads either sampled-id type and writes the required int64 result in its
existing final store. Ranks read the sampled vector's stride directly. Prompt
scoring no longer widens its ids before those operations. With no top-k columns,
reported ids retain their original view and integer width; the engine's
serialization converts either width to the same Python integers. FP32 scores,
floating-point arithmetic, ordering and token ranks are unchanged.

The source patch and `vllm/series` are authoritative; `overlay/` is an exact Git
export. `base.json` identifies the frozen image, prerequisite patch and every
source hash before/after that prerequisite. The harness verifies these stages
before importing the candidate, whose complete file hashes are in
`candidate.json`. No promoted configuration, default mode or source lock changes.

Correctness checks cover native dtypes, strides, masks, prompt boundaries,
speculative scores, wire ids, graph replay, FP64 score references and startup
preparation with a cold kernel cache. The benchmark loads the exact prerequisite
source's two kernel modules as independent references: normalization and
materialization costs are included there, with no remeasurement of a released
model build. Already-int64 contiguous inputs remain comparison cases, since
their old normalization did not launch extra GPU work.

Run from a clean published checkout on owner-authorized idle sp8:

```sh
python3 experiments/2026-11-04-native-score-inputs/run.py \
  --experiments experiments/2026-11-04-native-score-inputs --run native-v1
```

The runner uses the candidate-38 image by immutable digest, holds an exclusive
sp8 GPU lock and reservation, checks idle/memory conditions, cools below 55 C,
and monitors host memory. Tests, paired graph timings, and actual decoded IKET
captures each boot twice. Every run and failure is retained. No serving operation
is authorized by this experiment; full-image/full-model qualification remains
before promotion.

## Current qualification

Final source: `bce3375da1b478deabd64fe92a1ac4ac449131b7`, tree `499aea14913f209c2981da0db973c2cad924c660`.
Published test checkout: `2837a3251d0caea2861431cd56c5b93b8fedc4e6`. Native evidence:
`/home/chris/spark-lab/native-score-inputs-20261104/native-v9`; portable receipt: `runs/native-final.json`.

The final dispatch uses the existing native-stride kernel for larger selections
and a block-sized shared tile for a transposed selection that fits one warp.
The tile retains selected FP32 logits on chip, with no global conversion buffer
or extra launch. Already-compact int64 ids preserve the prerequisite's program.
The reduction, final arithmetic, ranks and serialized token values are unchanged.

All 120 cases pass on each final boot by exact program/input identity with
per-case programs qualified on both earlier timing boots: 112 from v6 and eight
small transposed selections from v7. Those eight shared-staging cases were timed
for 480 mirrored rounds on each boot. Every accepted measured case uses a
five-standard-error upper time ratio at most 1, with no regression tolerance;
reference-identical cases are proven by code and aliased preparation. The
unqualified v6 case is changed, and the rejected larger v7 kernels are never
selected by the final dispatch. No new timing measurements are claimed for
identical programs. Their source, inputs, strides and reference program are
verified again and their outputs checked independently.

Both final boots pass **673 tests**, without skips or failures. Each final IKET
capture decodes **56 launches** with valid phase ranges and warp lifetimes.
The probe explicitly runs sliced and transposed ids, both native widths,
masked/unmasked scores, eager/graph execution and aligned vector loads. Four
cases per boot dispatch shared staging; their numerical outputs are checked.
Minimum sampled host memory available: **111.08 GiB**. No memory guard stopped a
job, both reservations closed, and the containers were removed. Serving was
not changed.

The full candidate-38 series, prerequisite and all nine native-input patches
apply to a fresh tree matching the final export. Composition with the earlier
sampler, rank-order and gather candidates also applies cleanly (source check
only). Doctor, parsing, export hashes and diff checks pass; there are no new
lint diagnostics. Full build inputs are not prepared locally.

Reproduce the final audit without GPU access:

```sh
python3 experiments/2026-11-04-native-score-inputs/analyze_final.py \
  experiments/2026-11-04-native-score-inputs/runs/native-final.json
```

All unsuccessful variants and raw samples below remain unchanged. The results
qualify these GPU operations; they do not qualify full-model serving or host
enqueue performance. No promoted configuration, source lock or baseline changes.

## Initial variant retained

The initial variant (candidate-v1.json, published source 7b7a619d) passed 625 tests
per boot, 120 timing cases per boot, and two IKET captures. Its full evidence and
analysis are retained as runs/native-v1.json and results-v1.json. One already-int64,
contiguous-input FP32 masked case read 0.93 us (+4.78%) higher on boot means, with
substantial reference variation; no serving slowdown is inferred from it.

That variant also generalized mask strides even though the old path never
normalized masks and the producer already writes them contiguously. Patch 0002
keeps that existing contract, trimming unused stride parameters. This narrows
the candidate; it does not change the native token-id support or score arithmetic.
The final variant has its own repeated qualification below. Initial receipts are
not relabelled as evidence for the new tree.

The second variant (candidate-v2.json, published source 4cddbfc4) also passed its
625 tests, 120 timing cases and IKET captures on both boots. Its FP32 one-row,
masked, already-compact int64 cases still measured slower (up to 10.05% on boot
means; results-v2.json). Patch 0003 therefore preserves the original compact
addressing as a specialization and uses dynamic id strides only when needed.
Both are prepared at startup and read the original input buffer: the layout
selection never materializes, converts or repacks ids. The second variant's
source identities and measurements remain separate from the final variant.

## v3 historical qualification

The v3 source is `da925efbc83245de4cbd6824fda9a877b7968fd2`, tree
`032385a555e2c9f3ca7cc69ad0c355a7c7025279`. It was tested from published
deployment commit `1ed48ff2bb1568b33f8302fe2d4607ae53d48e15`, with native
evidence on sp8 at `/home/chris/spark-lab/native-score-inputs-20261104/native-v3`.

Both v3 test boots passed **625 tests**, without failures or skips. Both
benchmark boots passed **120 cases**, retaining 21 cold-L2 graph samples per
method and case. Each boot's 24 launch inventories show exactly one TileLang
candidate kernel, without an ATen conversion/copy kernel. References that already
have compact int64 inputs also launch one kernel: no removed copy is claimed
for those cases. The exact prerequisite source's modules are retained and hashed
with the native benchmark outputs.

The compact specialization is selected from existing layout metadata. A compact
tensor is already strided; specialization makes its strides compile-time
constants, while other layouts retain dynamic addressing. Neither branch
reformats its input. Both layouts and both integer widths are prepared before
cache freeze.

The v3 paired timing ratios range from **-12.71% to +5.95%** on boot means.
All cases, boot medians, raw samples and 95% Student t intervals (df=1) are retained.
No blanket speedup or statistical equivalence is claimed from two boots.

| Case (129,280-token vocabulary) | Reference boot medians, us | Final boot medians, us |
| --- | --- | --- |
| FP32 logits, 1 row, compact int64 ids, 129 masked scores (the earlier regression) | 20.160 / 18.592 | 20.160 / 18.528 |
| BF16 logits, 1 row, transposed int64 ids, 21 unmasked scores (largest final mean increase) | 16.096 / 16.192 | 16.096 / 18.112 |
| BF16 logits, 96 rows, dense int32 ids, 21 unmasked scores | 149.216 / 132.832 | 147.264 / 147.168 |

The first case returns to the reference timings. The largest mean increase
comes from a 2.016 us shift in one candidate boot; the 96-row case also has
substantial reference variation. Full-model qualification must judge practical
serving effects, including host enqueue costs, which these graph timings do not
measure.

Both IKET captures passed, each with **40 decoded launches** (20 eager, 20 graph
replays), covering score, top-k selection and rank kernels with numerical checks.
The minimum sampled host MemAvailable was **110.85 GiB**, with no memory guard
stop. Both test reservations closed, their containers were removed, and sp8 was
idle afterwards. No serving operation was performed.

`runs/local.json`, `runs/fresh-source.json` and `runs/composition.json` record
matching exports, Python parsing, no new lint diagnostics (one inherited test
line-length diagnostic), passing doctor/diff checks, fresh application of all
572 candidate-38 patches followed by PR #893 and this series, and composition
with the three prior ATen replacement candidates. The composition check is a
source check; those three other replacements are not part of these GPU runs.
The full build check reports that its build directory is not prepared locally.

Reproduce the v3 analysis without GPU access:

```sh
python3 experiments/2026-11-04-native-score-inputs/analyze.py \
  experiments/2026-11-04-native-score-inputs/runs/native-v3.json
```

Use `--candidate candidate-v1.json` or `--candidate candidate-v2.json` (with their
experiment-relative paths) to analyze an earlier receipt. Their analyses are
retained separately as `results-v1.json` and `results-v2.json`.

## Regression fix and controlled measurement

Patch 0004 retires native-id inputs before register-heavy computation. Noncompact
score ids are gathered into the existing output before normalization; top-k's
native sampled column is written before selection. Compact int64 cases retain
the original kernel form, chosen using metadata without a conversion or copy.

The previous timer ran each method's samples in one block, which did not control
short-term GPU drift well enough for the new per-case gate. The `pair` operation
instead uses 40 deterministic, randomized mirrored rounds per case, with two
samples per role in each round. The roles are the exact prerequisite reference,
the candidate, the recorded v3 implementation, and an unchanged reference
control. All share the same output address, and graph allocation is outside the
GPU event interval. Every captured role's output is checked against an independent
reference result. Input seeds and hashes are recorded.

The zero-margin GPU gate is evaluated separately for every case on both boots:
either the candidate has identical generated CUDA with the same input/output
bindings and no reference-preparation copy, or the paired log-time ratio's
five-standard-error upper bound is at most 1. Inconclusive cases do not pass.
This is a predeclared conservative measurement rule, not an allowance for a
percentage slowdown. Full-model and host enqueue qualification remain separate.

The old v3 results are retained as `results-v3.json`, `candidate-v3.json` and
`runs/native-v3.json`; their code is also retained under `control/v3/` with verified
hashes for simultaneous comparisons. No old result is relabelled as evidence for
the new source. The prototype's numerical, paired-performance and IKET checks
are still required.

### v4 timing diagnostic and correction

Source `d9f9a007` passed 625 checks and 40 IKET launches on each boot. Its first
paired timing run did not pass the gate: 62 and 67 cases were inconclusive,
respectively. The unchanged null control itself varied from -6.1% to +8.2% in
mean time across cases. Those samples are retained in `runs/native-v4.json` and
`paired-results-v4.json`; they do not establish performance acceptance.

The event markers had been submitted from Python around each graph replay.
They could therefore include host submission gaps. The revised timer captures
explicit external CUDA event nodes with the operation, with a cold-cache flush
before each of eight samples per replay, excluded from every event interval.
Each round still has two mirrored replays per role, and the gate is unchanged.
Reference and candidate run the same inputs and share the output address.
Only the measurement mechanism changes in this follow-up; the v4 source is
unchanged.

### v5 address simplification

The v4 follow-up with GPU-resident timing exposed a real penalty in large BF16
score cases from the early gather's additional output store/load. Patch 0005
removes that round trip and reduces address parameters from the existing input
metadata: contiguous, row-padded, column-sliced and transposed views have simple
expressions; other views keep general strides. It never repacks the data. Every
variant is declared before startup freeze and native-input checks now cover
padded and fully general views as well. The top-k lifetime change is unchanged.

The v5 qualification predeclares 120 mirrored rounds per case (three times the
earlier sample count), with eight cold-cache samples in each replay. Both boots
run the full 120-case matrix. Increases and unresolved intervals still fail.
This additional precision is needed to assess the small conversion savings in
large batches; it does not change the zero regression margin.

The GPU-resident v4 timing run is retained as `runs/native-v4-events.json` and
`paired-results-v4-events.json`. Both boots passed numerical and IKET checks,
but 35 cases failed to establish equal-or-better performance across the boots.
Four score cases were slower on both boots even at the five-SE lower bound.
That source is rejected; the additional sampling does not excuse it.

The first v5 run (`runs/native-v5.json`) passed 648 checks and failed one new
startup-freeze fixture: an elementwise operation preserved a transposed id
layout in the mask, although the production mask producer allocates a packed
mask. No performance jobs ran after that failure. Patch 0006 corrects the test
fixture only; the v5 runtime kernels and 120-round protocol are unchanged.
The corrected fixture is requalified as native-v6.

### Fixed precision extension rule

Before the second native-v6 timing boot completes, the follow-up is declared:
collect exactly 480 additional mirrored rounds on two fresh processes for
every case unresolved on either primary boot. An observed primary mean increase
is ineligible and requires a code fix. Every extension boot must independently
meet the same five-SE upper ratio at most 1, and pooling it with its primary
observations must also meet that criterion. Primary observations are never
discarded or reclassified. This is one fixed extension, not repeated sampling
until a case passes. The kernel source, input hashes, output sharing, cold-cache
protocol and zero regression margin remain identical. The filter is published
from the full primary result before any extension runs.

### v7 shared staging

The v6 extension retained 119 qualified cases and one unresolved transposed
BF16 96-row/21-column case (`paired-results-v6.json`). Its first extension
boot's upper bound remained 1.00153, so it is not accepted. No further
precision reruns of that source are allowed by this experiment's fixed rule.

Patch 0007 stages transposed selected logits in a 4 KiB shared array while the
row reduction runs. This avoids a late dependent gather and the rejected global
output-buffer round trip. Selections above 1024 columns use the existing native
strided kernel; correctness checks cover columns 1024 and 1025. All variants
are declared before startup freeze. There is still one GPU launch, and no
global normalization or conversion buffer.

V7 recompiles and numerically checks every one of the 120 cases on each boot.
For unchanged GPU programs, qualification transfers only when both generated
CUDA hashes (reference and candidate), input hashes and layouts match a
previously qualified v6 case. The single unqualified case cannot transfer.
The 16 changed transposed programs each receive 480 mirrored timing rounds on
both boots, with the same zero-margin five-SE gate. The frozen v6 source is
retained under `control/v6/` and runs beside the new kernels as the causal
control. This avoids remeasuring programs that are provably unchanged.

### v8 one-warp staging

V7 fixes the small transposed selection, but its first complete timing boot
establishes a regression in the larger BF16/int64 masked selection. It is
rejected, and its repeated timing boot is retained. Further correctness and
IKET work on that rejected source is unnecessary.

Patch 0008 bounds staging to one warp (32 selections, 128 bytes), a hardware
granularity rather than a dtype/request-shape exception. Larger selections
retain the v6 native-stride GPU program and its qualification. Boundary checks
cover 31, 32 and 33 columns. There are eight changed cases to time for 480 rounds
on each boot; every other case must prove exact code and input identity with
a qualified v6 case. The rejected source's single unqualified v6 case is
changed and must earn its own performance qualification.

### Final program selection

The v8 128-byte stage did not qualify the repeat BF16 96-row/21-column case.
Patch 0009 retains the exact block-sized shared stage whose eight small-selection
cases passed both v7 timing boots. Dispatch remains bounded to one warp. Larger
selections retain the v6 native-stride program; their two-boot qualification is
unchanged. This combines qualified programs, without a dtype/layout conversion.
The shared tile follows the normalization thread-block size (4 KiB on chip);
shrinking that tile changed generated code and did not qualify.

The final run must recompile and numerically execute all 120 cases twice. Every
reference and candidate CUDA hash, input hash and stride must match a previously
qualified per-case program: 112 from v6 and eight from v7. No unmatched or
unqualified program is accepted, and no new timing evidence is invented. The
factory's AST and constants for the shared stage are identical to v7; the host
metadata predicate selects it only for a one-warp selection. Full correctness
(673 checks per boot) still runs on the final source.

IKET now explicitly probes sliced and transposed ids, both native integer widths,
masked/unmasked scores, eager/graph execution, and aligned vector loads over a
16,385-token vocabulary. Its expected count is 56 launches per boot; four probe
cases per boot explicitly dispatch the shared stage. The timestamp capacity is
256 per sampled warp, within the same 0.25 GiB pool. These are bounded smoke
checks of the new path, not a full-model serving-speed measurement.
