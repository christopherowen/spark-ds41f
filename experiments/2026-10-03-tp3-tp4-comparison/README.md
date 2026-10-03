# Matched three- versus four-Spark serving comparison

The four-node candidate improves measured decode throughput by 14–27% against
the published three-node baseline. Single-stream step times improve by about
19%. These are matched workloads against a historical control, not a fresh
measurement of both topologies. Sustained thermal qualification remains open.

The measured implementation also removes TP3's extra attention heads, Engram
columns and vocabulary rows. It retains ordinary graph/sequence padding and a
compact MoE tail path; see [the exact padding audit](padding.md).

## Decode

Aggregate throughput includes request startup and the full 256-token output
budget. Each point has three samples, with one discarded warm-up per point.
`prose` and `code` name prompts; their measured outputs are reasoning tokens
(reasoning share 1.0 in both runs), not a thinking-disabled answer benchmark.

| Prompt | Streams | TP3 tok/s | TP4 tok/s | Change |
|---|---:|---:|---:|---:|
| Prose | 1 | 52.804 | 65.360 | +23.8% |
| Prose | 2 | 79.368 | 90.480 | +14.0% |
| Prose | 4 | 117.869 | 144.657 | +22.7% |
| Prose | 8 | 171.178 | 216.625 | +26.5% |
| Code | 1 | 64.779 | 78.544 | +21.2% |
| Code | 2 | 90.678 | 104.742 | +15.5% |
| Code | 4 | 135.644 | 170.594 | +25.8% |
| Code | 8 | 195.824 | 243.548 | +24.4% |

Single-stream step time: prose **40.845 → 33.057 ms (−19.1%)**, code
**45.356 → 36.898 ms (−18.6%)**. These estimates use decode-window throughput
and accepted-draft counters. Acceptance is similar: prose 1.240 → 1.256,
code 2.089 → 2.069 accepted drafts per step. Verification work is not identical:
prose 3.006 → 3.278 and code 4.071 → 4.050 verified drafts per step. Therefore
step time is not a fixed-operation-count microbenchmark.

The single-stream code TPS difference has a wide Welch interval crossing zero
(approximately −8.9% to +51.4%); its step-time interval favors TP4. The other
seven TPS difference intervals favor TP4 in this small screen. These intervals
describe within-run sampling, not between-boot, cooling or historical-control
uncertainty. The CLI's “all points converged” means the configured three-sample
cap was reached here; it does not mean every point reached ±2% precision.

## Prefill

The clean prefill-only run passes with zero thermal slowdown on every rank.
It uses two uncached source-text requests per nominal size and one output token.
The token estimate is deliberately approximate; use the actual lengths below.

| Nominal size | Actual input tokens, repeat 1 / 2 | TP3 tok/s | TP4 tok/s | Change | Mean TTFT, TP3 → TP4 |
|---|---|---:|---:|---:|---:|
| 1K | 977 / 961 | 2,181.311 | 2,641.136 | +21.1% | 0.444 → 0.367 s |
| 32K | 28,806 / 29,364 | 3,768.065 | 4,602.855 | +22.2% | 7.719 → 6.319 s |
| 64K | 59,653 / 57,766 | 3,796.383 | 4,637.567 | +22.2% | 15.464 → 12.659 s |
| 256K | 249,199 / 237,985 | 3,684.052 | 4,491.272 | +21.9% | 66.117 → 54.233 s |

All eight actual input lengths match the historical reference. The 1K
difference interval crosses zero; the three longer sizes favor TP4 in this
two-repeat screen. These are source-text prefill results, not the repeated-word
filler results from the earlier channel-count experiment.

The separately cooled prefix-cache screen also passes: mean cold TTFT improves
from **6.553 to 5.737 s (−12.5%)**, warm TTFT from **268 to 233 ms (−13.1%)**,
and the reported warm hit rate remains 99.0%. This uses the original nominal
32K **filler** prefix workload, not the source-text prefill inputs above.

## Workload and configuration controls

Run the published TP3 baseline's `prose`/`code` prompts, reasoning mode,
256-token decode at concurrency 1/2/4/8, seed 0, three samples, source-text
prefill at nominal 1024/32768/65536/262144 tokens with two repeats, and 32K
prefix replay on the selected TP4 relay/four-channel NCCL profile. Quality
passed 5/5 before measurement. The client host is dgx1 in both runs. The earlier
portable/filler TP4 screen is not used for this comparison.

The six workload functions (quality, decode, prefill, prefix, source generation
and source-file selection), plus eight request/timing/statistics helpers, are
AST-identical to the reference client's
`5b9e49b802d350370e1fa4a08077d3279a20f853`. The prompt-definition hash matches.
Every decode request's actual input/output length matches the reference.
The Python standard-library corpus is generated on dgx1, not the Mac client.
Its current file hashes are recorded; the old report did not save corpus hashes,
so byte-identical historical source text cannot be independently proven.

Both configurations use the same model snapshot, vLLM tree, NCCL tree,
64 KiB host kernel and NVIDIA driver version, 524,288 context limit, 3.5 GiB
KV allocation per rank, eight maximum sequences, 4,096-token prefill chunks,
graph capacities through 48 and five DSpark draft tokens. Both retain the r5o
model arithmetic; neither enables experimental batch invariance.

Differences include TP3 triangle versus TP4 neighbor relay, three versus four
draft ranks, exact per-rank kernel dimensions, and the tuned TP4 NCCL Ring/four
channel policy (TP3 capped channels at eight without forcing Ring). TP4 also
enables NCCL cuMem and uses its own pinned speculative cost table. The original
TP3 costs were boot-profiled. Every configuration difference is archived rather
than treating this as a one-variable hardware experiment.

The physical ring has no dgx1–dgx3 cable. A fresh TP3 triangle needs recabling;
using a fourth forwarding host would change the transport and is not a fair
reproduction of the old triangle. TP4 is therefore reported against the stored
TP3 baseline explicitly as a historical-control comparison.
This compares useful deployment configurations, including their transport and
TP-dependent verification plans; it does not isolate GPU count alone. Do not
reuse a TP3 speculative cost table on TP4.

The serving image and existing TP4 cost table remain unchanged throughout:
image `sha256:9671c96903dc7cc5a9bbae4249e7b7aa9efa550980f1dab5fa142dac3ebe842d`,
cost-table SHA-256
`1c66058404523da5323fe46c0a467b1448a39a2eaa5d7cfb1f39e5dffa8d3a0e`.
Rank 0 confirms reuse of the pinned table. Native command receipts record
published isolated checkout revisions and the normal coordinated operations.

## Thermal failures and follow-up design

The first continuous run passed quality and decode, then stopped during prefill
when dgx2 reached 83°C with only 5°C thermal headroom. It recorded zero thermal
slowdown. Its completed decode measurements remain useful as short-run results;
the combined suite failed.

A second, separately cooled prefill-plus-prefix run completed all eight prefill
requests at approximately +22% TPS. It then stopped during prefix testing at
84°C/5°C headroom on dgx2 and recorded 53.8 ms of thermal slowdown. Those
prefill values are retained as thermally compromised evidence, not the clean
comparison. The aggregate telemetry cannot attribute the slowdown to a specific
request, so it must not be silently excluded from that run.

The follow-up separates prefill and prefix, cooling before each section and
restoring the normal fan controller before measuring. The recovery wrapper
advances the benchmark's RNG past skipped sections; offline replay against the
actual suite functions verifies the same state. This preserves the original
prefill prompts/order and prefix inputs without repeating decode. All memory
and thermal guards remain enabled. Section-level success establishes burst
performance only; cooling between sections does not resolve sustained load.

| Run | Result | Thermal interpretation |
|---|---|---|
| Main quality/decode/prefill/prefix | Quality 5/5 and decode complete; abort during prefill | No recorded slowdown; combined run fails. |
| Cooled prefill + prefix | Prefill complete; abort during prefix | 53.8 ms slowdown on dgx2; not the clean prefill comparison. |
| Cooled prefill only | All eight requests pass | Zero slowdown; minimum headroom 7°C. |
| Cooled prefix only | Three cold and six warm requests pass | Zero slowdown; minimum headroom 22°C. |

Across these runs minimum available host memory is 30.50 GiB, with no swap
growth. The TP3 reference's lower minimum includes its four simultaneous
approximately 485K-token admission requests, which were not repeated here;
those memory minima are not a matched capacity comparison. Startup and final
kernel journals show no allocation/OOM/Xid errors, and serving logs show no
NCCL warnings, CUDA errors or tracebacks.

## Evidence and restoration

`results.json` contains native summary statistics, Welch difference intervals,
workload assertions, identities and the full four-run outcome inventory.
`hardware/runs.tar.gz` preserves native reports, command/exit receipts, both
aborts, client corpus hashes, source fingerprints, configuration differences,
all-rank startup/final logs, and entry/restoration state. `hardware/sha256.json`
records every archived file's hash. Reproduce the summary from an extracted
archive with:

```sh
mkdir /tmp/tp3-tp4-evidence
tar -xzf experiments/2026-10-03-tp3-tp4-comparison/hardware/runs.tar.gz -C /tmp/tp3-tp4-evidence
TMPDIR=/tmp python3 experiments/2026-10-03-tp3-tp4-comparison/summarize.py /tmp/tp3-tp4-evidence /tmp/tp3-tp4-results.json
```

The cluster was returned to its idle entry state on all four nodes and the
window released at **2026-10-03 05:41:36 UTC**. Original containers and shared
deployment checkouts are preserved; every fan controller is active. No padding
implementation or promoted configuration is changed by this experiment.
