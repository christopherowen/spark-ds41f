# Direct coarse merging

Base `5728eabe`, same compact80 scorer, summaries, budgets and fine arithmetic.
One intended variable: source coarse top4096 selection over4096-entry segments
no longer selects every entry in each first segment before merging. A raw8192
entry kernel directly performs the first pair merge, using the same BF16 keys,
radix selection, stable tie order and final writer. Subsequent native merge
levels are unchanged. This removes one launch and redundant all-entry selection
and carry buffers, without approximating any additional score or candidate.

This is an out-of-band screen, not a serving recipe. Require all12 actual score
and candidate checks, strict sanitizer fixtures, six graph/lifecycle cases and
two fresh component timing processes. A standalone exact comparison must also
cover empty/partial segments, active caps, repeated scores, infinities, NaNs and
uninitialized tails against the existing selector. Record every error, sample,
memory observation and lease release; retain raw operands privately on sp8.
No final-answer or request-speed pass follows from component timing alone.

The first screen at `a250c47d` passes all20 adversarial top-k fixtures with
zero sanitizer errors, all12 captured coarse/fine checks (including exact
preceding-selector winners and value bits), four strict initialization fixtures
and all six graph/cache/recycling cases. Two fresh processes measure source
Native68 / candidate ratios of1.070x at one row, 1.702x at eight and2.040x at64.
The conditional two-process joint95 intervals are[1.012,1.129], [1.641,1.764] and
[1.964,2.120]; their normal-process-median assumption is unverified. Dense plans
remain unchanged. Full receipts and maintenance/memory observations remain in
`runs/screen-01`. The lease releases at19:59:20 UTC with no remaining containers.

The integrated source avoids constructing the discarded native coarse stage.
Its declared-call list now matches the actual direct route; other budgets and
short covered shapes keep the original selector. The merge module is byte-equal
to the screened prototype. `vllm/series` adds it after the eight preceding
patches; the fresh nine-patch application equals both overlays. Before any model
start, requalify this exact integrated class, including bind after frozen kernel
declarations, actual/sanitizer/lifecycle checks and two fresh timing processes.

The integrated retry at `cecb7aab` passes binding all70 declared calls across
eight configurations after freezing the compiler cache, all12 actual batches,
strict initialization with zero sanitizer errors and all six lifecycle cases.
Two fresh timing processes retain source ratios1.075x,1.701x and2.042x for
rows1/8/64. The single-row conditional interval[0.944,1.242] remains wide;
the eight/64-row intervals are[1.643,1.762] and[2.010,2.074]. These are component
results with small-sample assumptions, not request or quality guarantees.
Every command/sample and memory observation remains in `runs/runtime-gates-01`.
The lease releases at20:11:20 UTC with no remaining containers. The profile
remains unpromoted and must face independent model quality/performance tests.
