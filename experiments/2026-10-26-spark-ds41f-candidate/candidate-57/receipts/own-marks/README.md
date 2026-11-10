# candidate-57 own marks (sp8, 2026-11-10)

#957's parts 2 (`0797`) and 4 (`0798`) were measured in one session, ABBA, each against the base: candidate-54's tree on candidate-54's image. Each candidate arm is that image with one part's files mounted, so the two parts are isolated from each other. Both are judged by CUDA-graph replay at TP3 rank shapes, with TP4 beside them.

| PR | Verdict | Evidence |
| --- | --- | --- |
| #957 (part 2) | pass, settled by a 4+4-boot confirmation | The mark's TP3 cells at 12 and 16 rows are 0.60 and 0.70 us faster per call (-3.8%, -2.7%; all 4 adjacent pairs lower). No TP4 cell is slower: 6 rows (c1's shape) +0.02 us and 32 rows +0.03 us, both the same; 12-16 rows 0.57-0.96 us faster. Outputs are bit-identical. Serving share about 0.05% at most. The first 2+2-boot run read a miss (first-boot drift, and a +0.12 us TP4 cell that did not reproduce); it is kept as `first_run` |
| #957 (part 4) | pass | The fused confidence calibration is 0.65-1.35 us faster by graph replay in 5/5 cells (eager 19.3 -> 9.6 us, the PR's number), within 1-2 FP32 ULP of the two-step path; the test passes |

**Part 4's dummy-run fix is a measurement change.** At TP3 rank shapes, candidate-54's 4-request dummy steps draft for 8 requests eagerly. `0798`'s draft for the 4 active requests through the (4, 20) FULL draft graph: 0.83 ms less drafter time on one GPU. That explains the tp3-switched screen's 4-request grid cells being 1.7-1.9% faster. Served steps dispatch on their real request count, so it is not a served-decode gain.

Details are in `pr-957/result.json` (`parts.2`, `parts.4`). `common/` holds the setup shared with candidates 55 and 56.
