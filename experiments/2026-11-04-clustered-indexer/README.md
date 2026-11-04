# Exact clustered indexer search

Status: **experimental; current grouping rejected for serving integration**.
[Native-input evidence](native-input-results.json) contains 12 actual C36 operands
from three ranks and layers2/8/14/source20. Two fresh sp8 processes preserve
selected keys, evaluated BF16 score bits, source block IDs/counts and the following
reindex result. Every sample retains every visible key: **0% pruning**. A dramatic
production speedup has not been established. Do not accept changed selections
without independent evidence of superior model quality.

The target is the checkpoint's existing attention rule and weights. This plan
only changes the search for its selected keys. It introduces no trained maps,
reduced precision or approximate attention. The original snapshot prototype has a separate bounded-workspace adapter
for owner-authorized Spaceport request measurements; see `serving/README.md`.
No promoted runtime selects it. No baseline or runtime image was
promoted.

## Matched request result on Spaceport

The [four-boot request comparison](runs/serving-window-02/comparison.md) now
measures the integrated candidate against the same C36 image with search off.
The only runtime variable is `VLLM_DS41_CLUSTERED_SEARCH=0/1`; both arms include
the same independent scratch-read guard. C36 is unreleased, so this does not
remeasure a released build. No release-relative speedup is claimed.

All141 request inputs per boot match exactly after excluding the deliberately
unique prefix-cache salt. All564 request outputs, token counts and finish reasons
match across the four boots. Five quality checks pass on every boot. Raw native
HTTP request elapsed time gives these point estimates across two boot medians:

| Workload | Actual prompt tokens | LSA request latency change |
| --- | --- | --- |
| 256-token prose/code, concurrency1/4/8 | Fixed native decode cases | +1.8% to +3.8% |
| Nominal32K cold source prefill | 28095–30739 | +14.5% |
| Nominal131K cold source prefill | 116664–121877 | +34.4% |

Positive change means slower. Both prefill ratio intervals are below1; several
small decode differences have intervals spanning1. Intervals use only two boot
medians (df1), with an unverified normality assumption; complete values and raw
samples remain in `comparison.json`. These are results on the selected existing
Spaceport4KiB kernel and stock fan control, not qualification of64KiB release
hardware or mixed production traffic. No memory or thermal guard aborted a run;
all four native reports retain memory, swap and temperature observations.

The first integrated pilot is retained separately in `runs/serving-window-01`.
It completed one LSA arm, then yielded for a method correction: adaptive3–4
sample counts could advance the shared RNG differently before source-prefill
sampling. The final matrix fixes every decode point at4 samples and records
exact model-input hashes and native elapsed times without changing payloads.
`serving/check_client.py` checks that client contract. No unfavorable result was
discarded or presented as a matched comparison.

The C37 service was restored on sp5–sp7 at16:27:33UTC (17:27:33CET). Independent
live doctor and per-node checks confirm the original image/settings, no search
mounts or environment, clean published checkouts, unchanged original806c5797
checkouts, and absent native reservations. The tracked restore profile now names
the isolated checkout, so its deployment/profile identity is explicit. No
promoted configuration, image tag or immutable baseline changed.

This candidate preserves tested outputs but fails the requested speed goal.
Keep the current native search; do not promote this grouping. The benchmark
adapter, exact commands, client identities, failures, repeated runs and closure
receipts remain reviewable under `serving/` and `runs/`.

## Algorithm and rounding argument

1. Build per-coordinate minimum and maximum values of every cluster's **visible**
   MXFP4 keys. Metadata is rebuilt for every snapshot in this first implementation.
2. For each head, choose the interval endpoint with the query coordinate's sign.
   Its dot product bounds that head's real dot product for every member key.
3. Inflate the FP32 dot estimate by `sum(abs(q) * max(abs(lo), abs(hi))) / 256`
   plus `2^-50`, then apply the scorer's BF16 rounding, ReLU, positive head
   weight and BF16 rounding. Negative-weight contributions are bounded by zero.
   Add head bounds in the same FP32 head order and round to BF16.
4. Score keys from the 32 clusters with the highest bounds with the unchanged
   MXFP4 scorer. Their kth selected value is a lower bound on the global kth value.
5. Keep every cluster whose bound is **greater than or equal to** that threshold.
   Compact its members in increasing position order, then run the unchanged
   candidate scorer and deterministic top-k. Equality must survive because a
   skipped earlier position could win a threshold tie.

For `u=2^-24`, a 128-term normal FP32 reduction has absolute error at most
`gamma(127) * A`, where `gamma(n)=n*u/(1-n*u)` and `A=sum(abs(products))`.
Endpoint selection makes the real interval dot no smaller than any member's
real dot. Allowing errors in both reductions requires at most `2*gamma(127)*A`.
The computed absolute envelope is at least `(1-gamma(127))*A`, and the `1/256`
allowance exceeds that requirement by over 250 times, including FP32 rounding
of the allowance and its addition. The smallest positive product in the
supported range is `2^-64`; neither products nor nonzero dot sums underflow.
Products of two decoded FP4 values are exact in FP32. At the largest supported
scales and head weights, dots and contributions cannot overflow BF16 or FP32.
The extra slack also covers zero and cancellation. BF16 round-to-nearest and
FP32 addition are monotone. Dropping negative weighted contributions can only
raise the bound, including when the true ordered head sum is negative.

Only UE8M0 scale bytes 96 through 144 and finite BF16 head weights of magnitude
at most `2^20` are certified here. A cluster or query outside that range gets an
infinite bound and is fully scored. It never drops a key based on an unsupported
encoding. Hardware execution must still establish that the compiler and MXFP4
MMA obey the assumed arithmetic; CPU checks alone do not qualify this proof.

This preserves the final scorer and tie rule, but exact output is a **gate**,
not an assertion inferred from an algorithm description. Every GPU case compares
all selected positions and checks each bound against every visible full score.
The gate includes signed weights, empty/short rows, page tails, noncontiguous
physical pages, allocator padding, unsupported scales, ties, and a mutated cache.

## Source handoff and shared owners

At the source layer, matching immediate top512 keys is insufficient. Native code
also publishes top2048 blocks of eight positions, in original order, with the
last visible block forced into the list and an exact position count. Later layers
reindex through that list. The source plan seeds32768 positions, derives a cutoff
from ordinary seed blocks excluding the forced-last block, and preserves every
possible winner plus the global last cluster. Unsupported or short seed domains
disable pruning.

The final cutoff packs native BF16 score order and complemented original block
position in a uint64 reduction. This preserves threshold ties, signed-zero order
and native position precedence. Stable whole64-cluster compaction preserves
8-position boundaries. The unchanged native block selector operates on compact
scores; a mapping kernel restores original block IDs. Native scoring and final
selection arithmetic are unchanged. Planned widths at most32768 use native
fallback for correctness only and are never timed.

Grouped plans share one freshly built metadata snapshot across1–256 queries with
one **explicit** cache owner. They do not infer cache ownership from allocation
order. A one-query source call skips unnecessary owner-table expansion. All plans
rebuild metadata per invocation, including graph replay; no persistent metadata
or invalidation protocol is claimed. Prefix reuse, speculative rollback across
calls and serving workspace ownership would need separate integration gates.

## Actual model operands and decision

The authorized TP3 window captured one query per layer/interface and rank from a
seeded public-source request of124426 actual prompt tokens. The native C36
attention/scorer ran during capture; this prototype did not run in serving.
The window restored C36 on all three ranks and closed at14:25:30UTC. Exact
commands, the failed initial checkout sync, restoration checks, input hashes and
transfer correction are retained in [the capture receipts](runs/capture-01/receipt.json).
Raw tensors, prompt text and response stay outside Git.

Replay reconstructs original physical page IDs, stride and alignment rather than
substituting a compact cache. Its address span is approximately5.27GB. Samples use
native planned widths524288/1048576, key pages128/256 and visible extents32896–36576.
The original query-row count is saved, but execution mode was not recorded and
is not inferred. This sample does not establish full-attention equality, decode
coverage, prompt-corpus quality or serving memory/admission behavior.

Two fresh pinned C36 sp8 containers pass all12 samples, including graph visibility
rollback/empty/restore and source reindex. Candidate-only process medians are
approximately0.182–0.185ms for dense samples and0.250–0.251ms for source samples.
Every sample keeps all visible keys. Native scoring is a correctness oracle only;
no released build was timed again. The empty-input run `sp8-real-01` is a retained
operator setup failure before any candidate case, followed by verified transfer
and two successful runs.

Untimed diagnostics on rank0's four layer/interface samples also find no pruning
with an ideal native cutoff: signed-head lower bounds, groups of8/64 keys,
semantic clustering of indivisible native eight-key blocks, and affine bounds
that combine signed heads before taking a support bound. The geometry diagnostic
is an idea screen with observed native-bound checks, not a qualified GPU kernel
or a substitute for a machine-arithmetic proof. These results reject the tested
bounds and groupings; they do not prove that every exact sparse search is impossible.
Adding persistent metadata or tuning the seed is not justified by these operands.

## Synthetic evidence and unfavorable runs

Every native receipt, exact command, mounted-input hash, error and memory
observation is retained in `runs/`. [Initial dgx4 results](results.json),
[sp8 results](sp8-results.json), [visible-domain results](visible-domain-results.json)
and [candidate-only profiles](source-profiling.json) index the stages. Timings
compare unreleased prototypes, never a remeasured released baseline.

| Candidate;262143 synthetic keys | Separable process medians | Random process medians |
| --- | --- | --- |
| Initial archived sp8 prototype | 3.591 /3.583ms | 5.678 /5.694ms |
| Streamed scalar envelope | 0.769 /0.791ms | 1.070 /1.092ms |
| Tensor-core bounds | 0.298 /0.306ms | 0.551 /0.553ms |
| Visible-domain metadata | 0.368 /0.377ms | 0.625 /0.622ms |

The latest visible-domain producer avoids writing invisible envelopes; its tensor
consumer avoids reading them. It improves1M-planned/short-visible source medians
from1.488/1.287ms to0.395/0.396ms, with a wide two-process interval. It also
regresses full-domain and grouped cases, as shown above and in the complete report.
It is not a universal replacement. A large synthetic prototype gain is not a
production speedup, especially when real operands do not prune.

Other isolated variables and their receipts remain available:

- Streamed endpoint construction and tensor-core interval products replace an
  expensive per-cluster register reduction. Visible endpoints remain bitwise equal.
- Explicit owner sharing reduces repeated metadata work. At64K planning and128
  queries, the earlier dense shared snapshot changes random medians13.383/13.466ms
  to3.498/3.530ms and separable7.046/7.047ms to0.767/0.767ms. Harness allocation
  peaks are not serving memory evidence.
- The first source tie cutoff is approximately three times slower. Its packed
  uint64 replacement recovers that loss while preserving16511/65535 separable
  keys. Both versions remain recorded.
- Mapping compact native block winners avoids restoring a dense score grid while
  retaining exact source handoff and following reindex results.
- The initial compiler failure and initialization failures (4608 reads in the
  full search;1536 in the isolated top-k regression) are retained. The independent
  lazy scratch-read guard fixes those reads without changing live arithmetic.

The latest42-job batch passes two fresh graph processes for scalar/matrix dense,
source, grouped dense and grouped source plans; new page128 cases; both-page
native-layout capture/replay fixtures; and24 dedicated initcheck/memcheck/racecheck
jobs. Bounded race/initialization screens are identified per receipt and do not
claim every large shape. Student-t intervals use two process medians (df1), with
an unverified normality assumption; they are descriptive prototype uncertainty.
Fourteen CPU reference checks pass. See each report for exact shapes, retained
counts, intervals and cumulative host memory/thermal observations.

## Reproduction and adoption boundary

The experimental vLLM series applies after promoted tree
`ea1281f880e7535e3d87725be0156a8e38fc2fb2`, upstream base
`19f2c20ed4d64ae22bd009ceacadd85e2a81e354`. It is not submitted upstream.
`bundle/overlay` is a byte-identical executable copy verified by `check_sources.py`.
The capture series independently applies after C36 tree
`cc0fdf582d0bddff71c6a853e237bc9e87587832`; its558-patch base is reproduced and
original source hashes recovered by reversing capture patches.

```sh
bin/spark doctor
python3 experiments/2026-11-04-clustered-indexer/check_sources.py
python3 -m unittest discover -s experiments/2026-11-04-clustered-indexer -p test_reference.py -v
python3 experiments/2026-11-04-clustered-indexer/summarize_native.py
git diff --check
```

Use fresh prepared trees to apply the complete affected source series. Complete
build inputs are absent locally; `bin/spark build check` reports an unprepared
build tree. No new runtime image was built. The independent scratch-read fix and
its regression belong to [PR878](https://github.com/christopherowen/spark-ds41f/pull/878).
That fix has merged; this experiment retains the same tested dependency.

Only run GPU jobs from one clean, published commit on an authorized idle node.
`sp8.json` pins both exact existing image IDs; C36 real replay uses its own pinned
runtime. `sp8_run.py` acquires the native hold and GPU lock, refreshes a heartbeat,
uses a16GiB container cap, observes a24GiB host memory floor and temperatures,
yields between jobs on request, and releases the hold. Receipts record exact
commands, commit and mounted hashes. Never substitute a dirty source copy.

```sh
python3 experiments/2026-11-04-clustered-indexer/sp8_run.py sp8-replay:real-a sp8-replay:real-b
```

The raw operand files must already exist at the recorded private node-local
path with matching SHA256 values. [Capture instructions](capture/README.md)
describe the guarded collection procedure; reusing the existing saved operands
requires no serving window. Rerun reports from committed receipts without a GPU.

No production adoption is proposed for this grouping. The paired experimental
Spaceport request measurement is complete and shows higher latency. A different exact search
must demonstrate useful real-input pruning and complete candidate cost first.
Only then would full candidate two-boot, attention/greedy/distribution, mixed
traffic and admission gates against immutable recorded release suites be useful.
Only the owner may accept promotion. Current production configuration and
immutable baseline evidence remain unchanged.
