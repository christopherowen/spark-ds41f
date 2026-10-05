# B12X patch stack

Base: `local-inference-lab/b12x@f8069b2c0be1311df3b112591c6b8876a843f8be`
(`integration/karmic-kraken-beta`, #435: the V4.1 FP4 KV writer divides like
DeepSeek's reference quantizer, `aef9df3c`), which also contains upstream
correctness commit
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
  startup. 0001-0002 on the base yield patch head `7409da7a` and tree
  `640c8544` (on the r5j base `e39b437b`: `bbd69d16`, `ec4cced9`).

- `0003-dsa-topk-position-ties.patch` breaks exact top-k score ties by lowest
  logical position in the DSA radix top-k. The buffered arm's last round and
  the exact overflow fallback used to hand the remaining slots to whichever
  tied candidates reached a shared-memory counter first, so identical runs
  selected different positions (78% of DS4.1 layer 2's prefill rows, about 5 of
  512). An exact radix over the position key now chooses them; the FP8 fused
  indexer is unchanged. DeepSeek's reference (`torch.topk`) repeats for the same
  scores and dgpp pins the same lower-index rule. Measured in
  `experiments/2026-09-29-topk-ties`: zero selection differences, no prefill or
  decode cost, acceptance unchanged. 0001-0003 on the base yield patch head
  `47c70835` and tree `35299956`.

- `0004-gemm-fence-stage-reads-before-tma-refill.patch` fences the async proxy
  before each dense GEMM mainloop stage release on the TMA load path. The MMA
  warps read a stage through the generic proxy (ldmatrix, scale-factor
  copies) and released it right after issuing the last k block's copies; the
  refill that permits is a TMA write, which the release's ordering does not
  cover. When another kernel's CTAs share the SM, a delayed copy read the
  next k tile: DS4.1's shared-expert down projection returned wrong columns
  in about 5% of decode calls beside the routed MoE (compute-sanitizer's
  racecheck does not track TMA writes). Found and measured in
  `experiments/2026-09-29-determinism` (0007 there) and
  `experiments/2026-09-30-r5n`: 20/12000 wrong to 0/12000 in the standalone
  stress, no serving cost. 0001-0004 on the base yield patch head `8d08c583`
  and tree `693aaed5`.

- `0005-fence-stage-reads-three-kernels.patch` adds the same fence to the
  TMA-refilled stage releases of the BF16 prefill projection, the mHC TF32
  and BF16 TMA prefill projections and the contiguous attention forward. Their
  source consumes the stage's loads before the release, but the compiled code
  issues the arrive with four or five shared loads still pending (a scoreboard
  dataflow over the SASS). Beside co-resident kernels the mHC TF32 projection
  returned wrong outputs in 609/12000 calls and the BF16 prefill projection in
  up to 45/12000, none fenced; the attention fence is preventive. From
  `experiments/2026-09-30-proxy-fence-audit`, qualified in
  `experiments/2026-09-30-r5o`. 0001-0005 on the base yield patch head
  `bb40849f` and tree `1a8b9401`.

Not in the series: the W4A8 tiny-decode `swiglu_limit` fix
(`experiments/2026-09-23-karmic-kraken-reference/patches/b12x/0002-tiny-decode-swiglu-limit.patch`)
is an upstream contribution. The promoted runtime disables tiny decode instead
(`B12X_W4A8_TINY_DECODE=0`).
