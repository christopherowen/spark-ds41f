# Decision

Retain the final variant for review, stacked on the selected-logit output
candidate in PR #893. It removes repeated token-id widening and layout
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
