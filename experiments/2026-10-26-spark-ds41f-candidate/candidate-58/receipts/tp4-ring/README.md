# candidate-58 at tp4-ring (dgx1..dgx4), 2026-11-10

Copied with `bin/spark candidate receipts` from dgx1 `results/private/` (the windows of 12:23-14:39Z and 14:46-15:59Z).
Each run's window-log segment is in `lab/`.

| Run | Result |
| --- | --- |
| Build and release tests (`candidate-58-image`) | image `spark-ds41f:candidate-58`, ID `6f67e5e4`, the same on dgx1..dgx4 and sp1..sp7; release tests 10,200 passed, 311 skipped, 0 failed (dgx4, 2,652 s) |
| Gate (`candidate-58-tp4`) | identical in every scenario, `again` included; hashes 18d61961/d6568649 (logprobs), 0a2dacdd/12fd8257 (text): candidate-54's. Distinct prompts 271.5 / 201.3 / 155.6 tok/s at c16 / c8 / c4 |
| Screen against candidate-54's recorded `tp4-ring` screen | prefill 32K +1.83% (8,954/8,964 tok/s) and 256K +3.02% (9,141/9,138), better; c16 decode and output the same (270.2/270.4 tok/s); grid -2.36%, better, mostly at 4- and 8-request shapes (#957 part 4's dummy-run fix, a measurement change), one cell worse (1x3 rows at 1016K, +0.65%, interval 0.58%) |
| Live checks | #955, #1013, #972 and #1008's thread placement pass |
| `single-stream`, one session: candidate-58, candidate-54, again | c1 paired decode -0.35% (margin 0.31%), worse: 81.93/81.90 against 82.24 tok/s |
| c1 attribution (`candidate-58-attribute-c1-tp4`), one session | candidate-58 82.14/82.53, without #957 part 4 82.24, without part 2 82.49, without #956 82.28, candidate-54 82.20 tok/s: all the same within the interval. The earlier -0.35% rested on one candidate-54 boot and does not reproduce |

**Verdict at tp4-ring:** passes. The gate is identical; prefill is 1.8-3.0% faster; c16 and c1 are the same as
candidate-54's.

**Open (2026-11-10):** the determinism gate's `again` scenario at tp4-ring has failed in 2 of 5 boots since
candidate-51 (0 of 147 before), though not in candidate-58's boot. candidate-54 is being revised with the fixes
(`spark-ds41f:candidate-54-r1`), and candidate-58 is rebuilt on it before any freeze.

**Gate repeat (2026-11-10, `gate-repeat/`).** Twelve boots, interleaved: candidate-58, spark-ds41f-1107 and
candidate-58-noC51, four boots each, with the gate's records kept per call.

| Arm | Boots identical | Hashes (logprobs) |
| --- | ---: | --- |
| candidate-58 | 3 of 4 | 18d61961 / d6568649; boot 3's reference f06c3f3c |
| candidate-58-noC51 | 4 of 4 | 18d61961 / d6568649 |
| spark-ds41f-1107 | 4 of 4 | 8e46794a / 11ec073c |

candidate-58's boot 3 failed in its first c1 run (`reference`), not in `again`; every later scenario and the boot's
later greedy run read the canonical hashes. One request diverged: prompt 7 (17 prompt tokens with thinking off, as
the gate sends it). Its logprobs are equal to token 113, differ from token 114 (-0.00038 against -0.00046) and
widely from token 115 (-0.907 against -0.635); its text differs from token 155. The first differing forward is at
absolute position 130. The size of the change does not by itself say whether the first difference was large: a
difference far below one rounding step can flip a near-tied discrete choice (the MoE router's experts, the indexer's
top-k) and give a jump of this size. The cause is under investigation (`determinism-investigation/`).
