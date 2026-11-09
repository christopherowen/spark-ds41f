# candidate-53 own marks (sp8, 2026-11-09)

The benches ran on the integrated candidate-53 tree, since all five PRs edit `model_runner.py`. Each was measured in one session, ABBA, against the base: candidate-50 without #1009, on the candidate-50 image.

| PR | Verdict | Evidence |
| --- | --- | --- |
| #960 | pass, on what one GPU shows | per-step metadata and CED staging fall from about 470 to 31–40 µs; byte-identical |
| #971 | miss; round 1 meets its mark (`../rework/pr-971-r1/`) | the tile cap gains 5% on the copy step. Its dead-row change does nothing at the recipe's `DEAD_ROWS_TAU=0.2`, so round 1 drops it |
| #977 | pass | `copy_match` is 16–90% faster from 64K to 1M, and none slower in a 201-replay re-run |
| #984 | cannot evaluate on one GPU | startup memory and TTFT during a long prefill need three ranks; the tp3-switched screen and its stall job decide. Risk: the post-prefill flush blocks the host for 32–35 ms |
| #998 | miss: in rework | candidate-53's release tests fail: the vocab tests segfault, and a `_Plan` stub has no `max_rows`. Its own mark was stopped part-way |
