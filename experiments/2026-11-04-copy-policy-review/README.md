# Copy policy performance review

C39 did not establish a serving-throughput improvement over either saved C38
recipe after both candidate and reference repeat floors. The proposed minimum
match 8 round is dropped at review: the available evidence does not justify an
improvement claim or another GPU round. This is not a second measured failure.
No patch was manufactured, and no speedup is established by this review.

This drops an optional performance hypothesis, not the owner's required
CopyDrafter default behavior, the safe copy envelope, or PR878's correctness
fix. C40's frozen source, specs, image and running qualification are unchanged.
No runtime/source/control change, node operation or GPU remeasurement occurred.

## Measured evidence

C39's first/repeat output rates were 163.34/163.80 tokens/s at eight streams,
125.91/126.04 at four, and 65.35/65.34 at one. Its engine-grid step cost was
0.86–1.32% higher than the two saved C38 standard boots across all four
pairings. Keep that adverse result visible.

The recorded explicit-copy C38 recipe already cost 0.39–1.34% against the
standard recipe across its four pairings. C39 versus those two C38 copy boots
ranges from -0.068% to +0.521% in engine-grid cost, while the saved copy boots'
own grid drift is +0.538%. These observations do not identify the new overlap
as the cause of the cost. [All original pairings and floors](../2026-11-04-copy-default-screening/runs/2026-11-04-dgx/attempt-3/recipe-comparison.json)
remain authoritative; this review does not replace their verdicts.

There is also a direct source distinction: `engine_grid` invokes `_dummy_run`,
which times the target and DSpark draft forward without invoking
`CopyDrafter.prepare_alloc` or `CopyDrafter.propose`. Copy overlap runs in normal
postprocess. Disabling that overlap is therefore not a direct source fix for
the observed grid cost. The [engine-grid source patch](../2026-10-16-r6r/vllm/0139-Engine-cost-grid-time-the-cost-profile-s-dummy-steps.patch)
and [C39 source patch](../2026-10-26-spark-ds41f-candidate/candidate-39/vllm/0688-perf-dspark-enable-copy-drafts-by-default.patch)
record the separate paths. The [C38 explicit-copy profile](../2026-10-26-spark-ds41f-candidate/candidate-38/tp3-copy.json)
already supplies the 6,155-token budget and 59-row capture envelope which C39
automatically derives. Removing that fitting is not a harmless performance
cleanup: it would alter copy capacity and aligned-prefill invariants.

## Minimum match 8 review

The original copy probes are byte-identical across both C39 boots, with their
hash verified against the published source receipt. Copy is active: the native
counter observations are 16.029 tokens/step on copy, 14.249 on edit and 6.773 on
tool inputs. The probe's fixed-m suffix simulation gives:

| Category | m=16 hybrid tokens/step | m=8 hybrid tokens/step | m=8 change |
|---|---:|---:|---:|
| Copy | 16.696 | 16.075 | -3.72% |
| Edit | 16.320 | 15.661 | -4.04% |
| Tool | 8.737 | 8.114 | -7.13% |

The oracle tool bound rises from 8.741 to 9.281 tokens/step, but an oracle is
not the implementation. Native matching chooses the longest matching tier;
minimum 8 preserves the 16-and-longer tiers and adds a sixth search/reduction
level. The simulation neither represents that exact adaptive policy nor
accounts for its kernel, verification or mixed-scheduling costs. It supplies
no measured wall-clock speedup. The serial probe also does not establish a
budget-binding mixed copy/edit/tool throughput baseline. Do not retrofit one
into C40, remeasure C39/C40, or treat a new workload observation as a compatible
old-baseline comparison.

## Acceptance and drop rule

Any future optimization needs a specific, defensible causal mechanism and a
compatible recorded baseline for its intended performance metric. Keep required
default behavior and correctness fixes. Use a clean published source, its own
image/gates, all native functional outcomes and two fresh metric boots.

A better verdict requires a real gain beyond paired uncertainty and both the
candidate and saved-reference repeat floors, without material adverse core
rows or correctness failures. Show 1030, C36, previous and current candidate;
label missing or incompatible evidence explicitly. Report throughput, per-user
decode, TTFT/TPOT and recorded ITL separately, real prefill/first-chunk waits,
engine-step cost, acceptance, memory/swap/startup and thermals. Keep every
unfavorable row. An engine-grid gain alone is not an end-to-end serving gain.
Targeted correctness/workload observations do not establish targeted throughput
gain without a recorded compatible timing baseline.

After a measured miss, allow at most one justified corrected attempt. If that
attempt still misses the mark, highlight the failed optimization as a problem
and drop its optional patch or performance claim, preserving all original and
corrected evidence. Do not keep producing speculative retries. This review
rejects minimum match 8 before GPU testing rather than presenting an unsupported
patch as an improvement candidate.

`review.json` records exact source identities, evidence hashes, all eight C39
reference grid pairings and the original probe values. The read-only C40
source inspected was head `5f926df6d3126266a9c88c82c06544060008c33c`, tree
`ca1496c6481dc5bc4b4234b743e4ca023a6c7597`. The canonical tracked patch
series/manifests remain authority; the prepared tree is only an inspection
cache. The local C40 offline doctor and whitespace checks passed.
