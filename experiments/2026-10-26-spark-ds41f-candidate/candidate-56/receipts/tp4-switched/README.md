# candidate-56 at tp4-switched (sp1..sp4), 2026-11-10

These are the lab's outputs from sp1's window, copied from sp1 `results/private/`. `lab/candidate-56-window.log` is
the window log's part from the gate's start to the grid attribution's end. The image was built on sp8 (ID
`0f65de86`), copied to sp1 over the fabric in 144 s and distributed to sp2..sp4 in 340 s.

| Run | Result |
| --- | --- |
| Gate (`candidate-56-tp4-switched`) | temperature-0 outputs identical in every scenario, hashes 18d61961/d6568649 (logprobs): candidate-54's, so 957 part 1 and #956 change no greedy output |
| Screen against candidate-54's recorded screen | grid geometric mean **+2.85% (worse)**, all at small batches (1x2 rows at 4K: 14.47 -> 16.35 ms); c16 decode and output the same; prefill 32K the same, **256K +3.31% (better)** |
| #956's mark (`attribute-956`: prefill 32K-256K, same session, ABBA against `tp4-switched-no956`) | 32K the same, **64K +2.33%, 256K +2.40% (better)**, 128K +2.52% (margin 2.63%, the same) |
| Grid attribution (`attribute-grid`: the grid step alone, six boots in one session) | see below |

**Grid attribution.** Against candidate-56's image running candidate-54's code (`tp4-switched-no957p1-no956`, the
control):
- candidate-54's own image (dgx4-built): grid +0.24%, the same; 1x2 rows at 4K 14.48 against 14.51 ms. **The sp8
  build host is not the cause.**
- with 957 part 1 alone (`tp4-switched-no956`): grid **+3.06%**; 1x2 rows at 4K **+12.3%** (16.29 ms), 1x3 +10.8%.
- with 957 part 1 and #956 (candidate-56): grid +3.08%; 1x2 +12.5%.
So **#957 part 1 adds about 1.6-1.8 ms to a 1-request decode step at TP4**, and #956 adds nothing to decode.

**Verdicts:** #956 meets its mark (prefill faster at 64K and 256K, decode unchanged). #957 part 1 misses: its own
mark is a faster decode step, and it makes small-batch decode 10-12% slower. Its cause is being found on sp8
(`receipts/own-marks/`), and it goes to rework.
