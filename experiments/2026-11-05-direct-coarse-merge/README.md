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
