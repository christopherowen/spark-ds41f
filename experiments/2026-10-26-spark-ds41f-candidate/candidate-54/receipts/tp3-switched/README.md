# candidate-54 at tp3-switched (sp5..sp7), 2026-11-10

These are the lab's outputs from sp5's window (`spark-0013`, open since 2026-11-09), copied from sp5
`results/private/`. `lab/candidate-54-window.log` is the window log's part from the first gate attempt to the
screen's end.

| Path | What |
| --- | --- |
| `determinism/candidate-54-tp3-switched/`, `lab/candidate-54-tp3-switched-table.txt` | The gate: temperature-0 outputs identical in every scenario, new TP3 hashes 5c22ce41/bce6092e (logprobs) and 26a739a9/12cfee9b (text): #1030's declared change, first recorded at TP3 here. Distinct prompts give 154.0 / 116.7 tok/s at c8 / c4 |
| `determinism/candidate-54-screen-tp3-switched/` | The screen's four jobs, two boots each: `screen-tp3` against 1107's recorded screen, `single-stream` and `sampled` against candidate-50's, the stall steps against candidate-50's. The suite runs are in `../references/` |
| `lab/candidate-54-screen-tp3-switched-table.txt` | The last job's table (the stall steps): the four jobs share one table file. Every table is in the window log |

**First attempts failed:** the gate and screen first started at 01:50Z while `build distribute` was still loading the
image on sp7 ("image is unavailable"). The coordinator queued them too early; both ran again from 01:52Z.

**The bench memory floor cut the screen short (added 2026-11-10).** sp5's MemAvailable fell to 3.90 GiB in the first
boot's prefill step and to 3.99 GiB in the second boot's grid step, under the 4 GiB bench floor, and each halt ends
that suite. So the `screen-tp3` figures below are the **first boot's alone** (grid and c8), with no repeat, and
TP3's prefill is not measured. `single-stream`, `sampled` and the stall steps ran on both boots.

**Screen against 1107 (`screen-tp3`):**
- **engine grid geometric mean 2.87% slower (±0.12%), worse.** The loss is at small batches: 1-request shapes are
  5.5-7.2% slower (1x3 rows at 4K: 18.91 -> 20.23 ms), 4x8 3.2%, 2x12 2.1%, 4x12 0.8%; the 8-request shapes are the
  same or 1.1% faster (8x48). At `tp4-ring` and `tp4-switched` the same shapes are 8-9% faster.
- c8 output +1.2% and paired decode +0.9%, the same (intervals 1.5%); paired TTFT 2.2% better.
- candidate-50 and candidate-53 left TP3's grid unchanged (+0.13% and -0.21%). So the loss comes from patches first
  measured at TP3 here: candidate-51's (#961, #963 r1, #987, #1001 r1), candidate-52's (#1030, #968, #969), #960 r1
  and #998 r1. Its attribution is a follow-up on this testbed.

**Companions against candidate-50:** `single-stream` c1 the same (+0.55%, margin 3.2%); `sampled` the same at c16, c8
and c1; the stall steps the same (stall-3 +0.95%, margin 1.05%).

**Time to ready:** 180.9 s for the image's first boot (kernel compile), then 124-132 s, except one 185.5 s boot
(02:27Z, the `sampled` job's first).
