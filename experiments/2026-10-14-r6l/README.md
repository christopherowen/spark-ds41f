# r6l: the kernel contracts and two NaN fixes, the TileKernels clamp fix, the engine grid

Owner, 2026-10-14: "begin the promotion of good work".

## The candidate

r6k's series ([vllm/series](vllm/series)) plus vLLM branch `r6l`. The branch is
`kernel-contracts` 0001-0005 on r6k's patch head, the engine-grid commits, and one test
commit:

| patch | change | evidence |
| --- | --- | --- |
| 0134-0136 | non-finite contracts for ten kernel families, DeepSeek-derived compression and Engram hash oracles, the calibrated MoE check (tests only) | [kernel-contracts](../2026-10-14-kernel-contracts/README.md) w1: 187 pass with the fixes |
| 0137 | shared SwiGLU clamps keep a NaN | bit-identical on finite inputs; level with r6k (round 2: warm medians 0.98-1.01; the cold flags are timer levels) |
| 0138 | Engram gate floor keeps a NaN | bit-identical; level (round 1 repeats) |
| 0139-0140 | the engine cost grid for the dev-mode RPC (`Worker.engine_grid`, settle runs) | used live through the overlay since 2026-10-14 |
| 0141 | tests: the routed experts' NaN contract is required (TileKernels patch below); the indexer's becomes an expected failure until its fix keeps r6k's speed | |

TileKernels: [patches/tile_kernels/0001](../../patches/tile_kernels/README.md), where the
routed SwiGLU clamp keeps a NaN: bit-identical, 0.99-1.01x
([moe-swiglu bench](../2026-10-14-moe-swiglu/README.md)).

Not in r6l: the indexer NaN fix (vLLM kernel-contracts 0006). Its forms either serialize
the loop (+2-3.5% on that kernel at prefill) or get folded away by TileLang's simplifier.
It waits for a vectorizable NaN-keeping max.

Source identity:
- vLLM tree `bc089a5c` (patch head `37e56d34`, as the build replays it with the fixed committer), checked by replaying the series onto
  `19f2c20e`;
- TileKernels tree `cb2fa41c` (patch head `edfa64d0`), checked by applying the patch onto
  `66258df`.

## Specs

- [image.json](image.json): build `-r6l` on dgx4, copy it, then r6k's image gate:
  determinism with and without logprobs, sixteen and eight distinct streams.
- [release.json](release.json): the `release` suite on r6l, r6k, r6l again, all pinned
  ([r6l-dev.json](r6l-dev.json) is the release configuration with the dev-mode RPC for
  the engine grid). Read it with
  `python3 scripts/servebench_suite_table.py r6l-release r6l r6k r6l-again`.
- [r6l-release.json](r6l-release.json): what production would run (pinned, no dev mode).

The temperature-0 references should stay r6k's (70b3379cc289 / 708ec5b17482): every change
is bit-identical on finite values. Promotion is the owner's call.
