# candidate-57 own marks (sp8, 2026-11-10)

#957's parts 2 (`0797`) and 4 (`0798`) were measured in one session, ABBA, each against the base: candidate-54's tree on candidate-54's image. Each candidate arm is that image with one part's files mounted, so the two parts are isolated from each other. Both are judged by CUDA-graph replay at TP3 rank shapes, with TP4 beside them.

| PR | Verdict | Evidence |
| --- | --- | --- |
| #957 (part 2) | miss | Lower on the mean in all 16 TP3 cells. Faster beyond the interval only at 6 rows: the mark's 12- and 16-row cells are the same, because both arms' first boots ran slower and the interval widens. At TP4, 6 rows (mixed lengths) is 0.12 us slower and 32 rows 0.15 us slower; 12-24 rows are 0.5-0.7 us faster. Outputs are bit-identical. Serving share about 0.05% either way; a 4+4-boot confirmation would settle it |
| #957 (part 4) | pass | The fused confidence calibration is 0.65-1.35 us faster by graph replay in 5/5 cells (eager 19.3 -> 9.6 us, the PR's number), within 1-2 FP32 ULP of the two-step path; the test passes |

**Part 4's dummy-run fix is a measurement change.** At TP3 rank shapes, candidate-54's 4-request dummy steps draft for 8 requests eagerly. `0798`'s draft for the 4 active requests through the (4, 20) FULL draft graph: 0.83 ms less drafter time on one GPU. That explains the tp3-switched screen's 4-request grid cells being 1.7-1.9% faster. Served steps dispatch on their real request count, so it is not a served-decode gain.

Details are in `pr-957/result.json` (`parts.2`, `parts.4`). `common/` holds the setup shared with candidates 55 and 56.
