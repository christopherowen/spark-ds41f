# B12X patch stack

Base: `local-inference-lab/b12x@e39b437bf7d5c6784fbee8dc7e87073dde753445`
(`integration/karmic-kraken-beta`), which contains upstream correctness commit
`02407f65`, concurrent Engram table reads (`run_lookups`, `6e2090bc`) and the
CuTe compile-cache integrity check (#418, `2fca4df8`).

- `0001-switchless-rocenante.patch` adds per-peer HCA routing for the
  direct-cabled three-node ring; Local Inference Lab's launcher targets a
  switched fabric. Applying it to the base yields tree `f77d175f` (on the r3
  base `0f846212`: `d661de31`, the promoted image's `local.spark3.b12x.tree`
  label). Qualified on
  the three Sparks by the 2026-09-23 karmic-kraken experiment (serving matrix and
  LRU gate).

- `0002-cutlass-dsl-4.7.1.patch` moves B12X's exact CuTe DSL pin from 4.6.2 to 4.7.1, the version
  the vLLM nightly base installs and vLLM and quack-kernels require (4.7.0 and
  4.7.1 add features and fix bugs without API removals). Holding 4.6.2 had
  broken quack-kernels, which disabled vLLM's DS4.1 L2 weight prefetch at
  startup. 0001-0002 on the base yield patch head `bbd69d16` and tree
  `ec4cced9`.

Not in the series: the W4A8 tiny-decode `swiglu_limit` fix
(`experiments/2026-09-23-karmic-kraken-reference/patches/b12x/0002-tiny-decode-swiglu-limit.patch`)
is an upstream contribution. The promoted runtime disables tiny decode instead
(`B12X_W4A8_TINY_DECODE=0`).
