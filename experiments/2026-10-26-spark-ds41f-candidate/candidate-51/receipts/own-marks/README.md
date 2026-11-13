# candidate-51 own marks (sp8, 2026-11-09)

All four PRs are bit-exact (0 ULP). Each was measured in one session, ABBA, judged by CUDA-graph replay, against the base: candidate-50 without #1009, on the candidate-50 image.

| PR | Verdict | Evidence |
| --- | --- | --- |
| #961 | pass | consumer read 1.06x; decode chain faster or the same, never slower; pytest passes |
| #963 | miss: in rework | the certified greedy screen is 0.6–2.6% slower at TP4 M=97..120; `compressor_down` is 2.5–7.4% slower and `indexer_wk` 2–3% slower |
| #987 (WO-B part) | pass | 18/18 L2-hot cells faster at M=97..128; the PR's bench was never committed, so `h987.py` keeps its workload |
| #1001 | miss: in rework | the projection gains, but the greedy half cannot move: #997's screen serves greedy from `SCREEN_TILE_CONFIGS`, which #1001 leaves alone |

The tp4-ring screen of the whole candidate is in `../tp4-ring/`. Against the base, the engine grid is 3.7% faster and c16 decode 0.7% faster, and the gate is identical.
