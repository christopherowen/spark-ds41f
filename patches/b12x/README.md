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

Not in the series: the W4A8 tiny-decode `swiglu_limit` fix
(`experiments/2026-09-23-karmic-kraken-reference/patches/b12x/0002-tiny-decode-swiglu-limit.patch`)
is an upstream contribution. The promoted runtime disables tiny decode instead
(`B12X_W4A8_TINY_DECODE=0`).
