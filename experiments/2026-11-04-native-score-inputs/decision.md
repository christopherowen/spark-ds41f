# Decision

Not ready: the equal-or-better performance gate is not yet passed. PR #895
is a draft, stacked on PR #893. The owner requires fixing slower cases while
retaining the conversion removals.

The current source (`d20a95fd`, tree `ca7edc30`) consumes the producer's native
integer width and existing strides. Score addressing uses the compact, padded,
sliced or transposed layout's simple expression; other views use general
strides. It reads each selected logit once at the final store, without the
rejected v4 output-buffer round trip. Top-k writes native sampled ids before
selection. Compact int64 cases preserve the prerequisite's generated CUDA.
All variants are prepared before startup freeze; no new conversion or GPU
launch is added.

Both current test boots pass 649 checks. Both full 120-case timing boots and
IKET captures remain required. Timings use GPU-resident event nodes, eight
cold-cache samples per replay and 120 mirrored rounds per case. Every case
on both boots must have verified identical GPU code and bindings or a five-SE
upper time ratio at most 1. Inconclusive is not a pass.

Earlier variants and all unfavorable runs are preserved. V3's means increased
in 13 of 120 cases, up to 5.95%. V4's more controlled timing left 35 cases
unproved and established four score regressions on both boots. That source
is rejected. The first v5 test run passed 648 checks and rejected a new fixture
that derived a transposed mask instead of the production producer's packed
mask; the fixture alone is corrected in patch 0006.

The complete series applies cleanly after candidate-38 and composes with the
three prior ATen replacement candidates. Full-image/full-model qualification,
including host enqueue effects, remains before promotion. No promoted
configuration, default mode, source lock or immutable baseline is changed.
No serving operation was performed.
