# Decision

Not ready: the equal-or-better performance gate is not passed. PR #895 is a
draft, stacked on PR #893. The owner requires fixing the slower cases while
retaining the conversion removals. It removes repeated token-id widening and layout
materializations while preserving FP32 scores, token values, ranks and Python
wire ids. Compact and noncompact inputs each use their existing storage.

The final source passes 625 tests, 120 paired operation cases, and actual IKET
captures on each of two boots. GPU launch inventories confirm the eliminated
conversion/copy kernels. The earlier generic-addressing variants and their
unfavorable performance cases are retained, without relabelling their evidence.

The final compact FP32 case returns to reference timings. Other final cases have
timing variation and some increases; this is not a blanket serving-speedup
verdict. Full-image/full-model qualification, including host enqueue effects,
remains before promotion. No promoted configuration, default mode, source lock
or immutable baseline is changed, and no serving operation was performed.

The recorded v3 means increased in 13 of 120 cases, up to 5.95%. Timing
variation does not make those cases passes. `performance-gate.json` records
the unmet criterion; all earlier measurements remain unchanged.
