# PR #1009 rework, round 1: index weights gathered in SP decode steps

**Sub-agent verdict:** `corrected-meets-mark`. The corrected patch is `0778-perf-dsv41-sp-prefill-local-attention-front.patch`; `0778-r1-vs-candidate-50.diff` is its change against candidate-50's. All figures are in `result.json`.

**The miss.** candidate-50 did not boot at `tp4-ring`. While capturing graphs, `indexer_score` got index-query rows padded to 64 and only 54 weight rows (`../../tp4-boot-failure/`).

**Root cause.** `_sp_front` keeps the indexer head weights rank-local in every sequence-parallel forward (`attention.py:1276`, `index_weights[: sp_rows.real_rows]`). But `_forward` splits the indexer by rank only in a step with prefill rows: at `attention.py:1514-1515`, `split` is None when `swa.is_decode`. So in an SP decode step every rank scores all T rows from its L local weight rows.

**Why tp4-ring and not TP3.**
- At tp4-ring the first eager warm-up shape past the capture sizes is T=214: 16 requests of 13-14 rows. That is a decode step (within the copy-draft decode threshold of 17), and sequence-parallel (from 205 rows). Its local L is 54.
- TP3's first SP-sized tail step has prefill rows, so TP3 booted. The rework derived this; it wasn't measured.
- tp4-switched and tp8-switched use tp4-ring's sizing.

**Fix** (`attention.py`, `_sp_front` only, +16/-5 lines):
- It keeps the weights local only where `_forward` splits the indexer.
- In a decode step, and under `_INDEX_SPLIT_CHECK`, it gathers them across ranks with `sp_rows.gather`.
- Steps with prefill rows launch the same kernels and collectives as before.
- The added cost is one 64 B/row gather in SP decode steps.

**Evidence** (one GPU, through vLLM's own TileLang attention, at TP3, TP4 and TP8 rank shapes):
- **Repro:** candidate-50 as built fails both decode steps. At TP4 it raises the boot's exact error.
- **Fix:** the corrected tree passes all 5 steps at all three sizes.
- **Bit-exact:** layer 24's outputs, top-k indices and SWA pages are byte-identical to candidate-50 without #1009, on every rank, over two boots.
- **Own mark:** same-session ABBA against no1009, two boots. Every cell is faster beyond the interval, with 0 slower cells:

  | Shape | Graph replay | Speedup |
  | --- | --- | --- |
  | TP3 T=4096 | 234 → 10.8 µs | 21.7x |
  | TP4 T=4096 | 229 → 9.3 µs | 24.5x |
  | TP3 T=16384 | 1146 → 218 µs | 5.2x |
  | TP4 T=16384 | 1097 → 177 µs | 6.2x |

  Kernel time agrees.
- **Tests:** 443 passed, 1 skipped and 1 xfailed on both boots, against 437 passed on candidate-50. The 6 new `is_decode` cases fail on candidate-50's code.

## Coordinator's verdict: meets the mark, pending a TP4 boot

Round 1 fixes the cause the boot failure points to, is bit-exact and keeps #1009's gain. What remains is a boot at TP4 sizing.

It is validated on the accumulated base, on the same candidate-50 image with `1009r1/overlay/.../attention.py` mounted:
- `tp4-switched`, which uses tp4-ring's sizing: `candidate-tp4-switched-1009r1.json` (gate) and `screen-tp4-switched-1009r1.json` (two boots against candidate-50 without #1009's screen in the same window);
- `tp3-switched`: the same pair of specs.

If both pass, #1009 round 1 joins the base of the round after 51–53. Round 51–53's base stays candidate-50 without #1009, so its references are unconfounded.

The analysis is the rework sub-agent's; the coordinator wrote the verdict.
