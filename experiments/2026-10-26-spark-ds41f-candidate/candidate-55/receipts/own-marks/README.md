# candidate-55 own marks (sp8, 2026-11-10)

#988's dead-row part (`0794`) was measured in one session, ABBA, against the base: candidate-54's tree on candidate-54's image. The candidate arm is that image with `0794`'s nine files mounted. The sampler runs outside the CUDA graphs in serving, so the eager call's stream time decides; CUPTI kernel time is recorded beside it.

| PR | Verdict | Evidence |
| --- | --- | --- |
| #988 (dead-row part) | pass | `apply_sampling_params` as `_verify` calls it is faster in 12/12 cells: 1.06-1.14x at 1 request, 1.36-1.53x at 4, 2.25-2.83x at 8, 2.26-3.68x at 16. Live rows are bit-exact with NaN dead rows, `rejection_sample` outputs are equal, and the PR's tests pass (131 passed, both boots) |

Predicted serving share: 1.2-1.8% of a TP4 c16 step, and 0.5-0.6% at c8, for requests that carry top_k, top_p < 1, min_p, penalties or the like. It is 0 for the recipe's defaults and every screen suite, which send none.

Not mine:
- #1008's own mark needs the TP4 ring; it is in candidate-55's screen window extras.
- #978 is in rework.

Details are in `pr-988/result.json`. `common/` holds the setup shared with candidates 56 and 57.
