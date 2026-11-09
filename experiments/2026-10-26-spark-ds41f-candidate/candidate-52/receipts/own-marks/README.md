# candidate-52 own marks (sp8, 2026-11-09)

Each PR was measured in one session, ABBA, against the base: candidate-50 without #1009, on the candidate-50 image.

| PR | Verdict | Evidence |
| --- | --- | --- |
| #1030 (fix) | pass | 0 FP8 and 0 scale differences from DeepSeek `Expert.forward` in all six cases, where 1107's kernel differs; kernel time 0.74–0.84x at TP4 |
| #968 | pass, on what one GPU shows | the verify-row cap acts as designed; outputs are bit-identical. Paired decode and tail steps are judged by the tp4-switched screen |
| #969 | pass | 48/60 cells faster and none slower; rows identical. The PR never committed `bench_and_verify`; main's module of that name reproduces the PR's recorded counters exactly |

A one-second debug check ran without the GPU lock while nothing was being measured; it is recorded in `common/setup.txt`.
