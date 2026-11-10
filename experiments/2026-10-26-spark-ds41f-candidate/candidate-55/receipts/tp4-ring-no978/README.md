# candidate-55 without #978 at tp4-ring (dgx1..dgx4), 2026-11-10

These are the lab's outputs from dgx1's window (07:40-09:17Z), copied from dgx1 `results/private/`. The arms run the
candidate-55 image with #978's files reverted (`../../tp4-no978.json`), so they measure #1008 (0792) and #988's
dead-row part (0794) on candidate-54.

The first gate and screen attempts were refused before booting: `tp4-no978.json`'s notes cited the reverted tree's
hash, and the lab accepts only the image's. The notes were fixed and both re-queued.

| Run | Result |
| --- | --- |
| Gate (`candidate-55-tp4-no978`) | temperature-0 outputs identical in every scenario, hashes 18d61961/d6568649 (logprobs): candidate-54's. Distinct prompts 270.1 / 201.9 / 154.3 tok/s at c16 / c8 / c4 |
| Screen against candidate-54's recorded `tp4-ring` screen | everything the same: grid -0.10%, c16 paired decode -0.36% (margin 0.47%), c16 output -0.31%, prefill 32K -0.17% and 256K +0.04%. Acceptance length 2.304, perplexity 1.31888 |
| `single-stream`, same session: candidate-55 without #978, candidate-54, again | c1 paired decode +0.00% (margin 0.31%): 82.24 / 82.03 / 82.21 tok/s |
| #1008's affinity check (`check_1008_affinity.py`, one boot each of candidate-54 and candidate-55 without #978) | passes: under a 16-stream load the threads are partitioned on candidate-55 and not on candidate-54. Client chunk gaps: p99 148.39 -> 147.57 ms, mean 92.21 -> 92.13 ms, one run each, so no interval |

**Verdicts:**
- **#988's dead-row part passes:** its own mark is met on sp8 (`../own-marks/`), and the screen and `single-stream`
  show no harm.
- **#1008 is not yet decided.** Its placement works, and nothing is worse. But its mark is a lower step p99 under
  rank-0 API load beyond the interval, with the mean not worse, and a single run each cannot show that. Its own bench
  (`bench_roce_core_affinity.py`), ABBA against candidate-54 on the TP4 ring, decides it.
