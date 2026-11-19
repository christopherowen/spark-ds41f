# Decode each query once

Base `ed811312`, C36 image, block8 means, candidate budgets and native fine
arithmetic unchanged. One intended variable: decode the native packed query to
BF16 once per row instead of once per coarse scoring CTA. The preceding page-copy
screen preserves all actual score bits and wins larger row-count components but
still loses single-query shapes. It is the component comparator, not a final
answer or request-speed result.

The decoded-query workspace is explicitly admitted: 8192 bytes per query row,
bounded at 256 rows (2 MiB). Each chunk writes it before scoring. Covered domains
skip its writes and reads. Source and dense kernels use the exact same decoding
and BF16 values as before. Query, summary and scoring tensors stay on the GPU.
There are no reference calculations, output comparisons or grading in the runtime.

First check all 12 actual input batches against independent coarse/fine score
bits, ordered candidates and native covered-domain outputs. Then screen two
fresh performance processes, with Native68 and original physical page IDs.
If promising, graph/cache/lifecycle and strict uninitialized-suffix checks are
required before model startup. All commands, failures, allocator observations
and releases remain recorded; raw inputs stay node-local. Full-model no-loss
answer gates and request timing remain mandatory. No speed or quality win is
assumed from removing duplicate calculations.


The first query-reuse screen at `90154e84` passes all 12 actual batches against
independent coarse/fine score bits. Two fresh processes show native/candidate
ratios of 1.35–1.37 for dense8, 2.30–2.31 for dense64, approximately 1.50 for
source8 and 1.81 for source64. Single-query shapes remain slower (dense0.64–0.68,
source0.87). Samples and small-n t intervals are in `runs/screen-01`. These are
component results, not a request-speed or final-answer pass. Allocator peaks
are 281,548,288 allocated and 289,406,976 reserved bytes in the actual check;
component timings include their own peaks. Processes start at 51/52 C, end at
52/53 C, with no active hardware thermal slowdown at the end. Release is
15:09:15 UTC, with no remaining containers; an independent later reader sees
no LSA lease or pending request.

The prepared serving class keeps every scratch tensor local to its invocation
instead of retaining a workspace view in the plan. Its kernel sequence and math
match the screen. Before any model startup, rerun actual-input, strict
initialization and cache/graph/lifecycle gates on this exact source, then timing.
Both runtime sources equal a fresh application of the ordered seven-patch stack
against C36. Named-profile/default doctor and 19 answer-harness checks pass.
The full build tree is unavailable locally; no image build is claimed. The model
comparison uses actual Native68 with fresh controls. `native-source.json` is a
predeclared quality fallback that keeps dense selection and restores native
source/reindex handoff, with the same 100-byte layout and dense budget.


The production-shaped version at `fd5958df` passes all 12 actual batches, strict
initialization with zero errors and all six cache/graph/lifecycle cases. Two
fresh processes reproduce dense8 1.35–1.39x, dense64 2.30x, source8 1.50x and
source64 1.81x native component speed. Single-query cases remain slower. All
samples, t intervals, commands and allocator observations remain in
`runs/runtime-gates-01`. Release is 15:17:08 UTC with no containers; an independent
reader subsequently sees no LSA lease or request.

The first model window (`runs/pilot-window-01`, published source `ed3c1051`)
boots Native68 successfully but the benchmark client stops before any request:
its private output name collides with the historical 8192-token cohort's
`pilot-window-01/native-a`. The previous data is untouched. All error/boot/restore
receipts are retained. C37 is restored at 15:25:42 UTC and independently verified
on all three nodes. The controller now derives the private path from the complete
result namespace and checks absence during preflight, before opening a window.
The retry uses a new result path. This is a client-path failure, not a quality
or runtime result; no request-speed conclusion is drawn from the failed window.

The completed retry at source `e9eab460` passes the observed final-answer no-loss
gate on both boots: Native68 scores 8/12 twice and LSA 9/12 twice, with no
incomplete or unstable answer decisions. Total workload time is 11.6% lower,
but median individual latency is about 32.4% higher. The improvement is uneven
and largely comes from shorter reasoning on one long table question. This is
pilot evidence, not a general quality-superiority claim. Full results, uncertainty,
memory observations and the next frozen confirmation command are in
[results.md](results.md). Every run remains in `runs/pilot-window-02`.
