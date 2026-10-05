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

- `0006-rocenante-ring4.patch` through `0011-loader-abi.patch` are the
  four-node transports, inactive on the triangle's direct peers: 0006 relays
  opposite-rank payloads over a switchless ring (neighbour QPs only), 0007
  opens NIC-forwarded mesh QPs, 0008 stripes opposite-peer payloads over four
  mesh paths, 0009 splits the ring relay across both directions, 0010
  separates the collective dispatch ceiling from the registered RoCE capacity
  (`B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES`) so NCCL takes intermediate
  payloads, and 0011 keeps the Python loader's ABI check on the version-10
  proxy. GPU kernels and reduction order are unchanged; each was tested with
  the real proxy under simulated verbs and sanitizers. From
  `experiments/2026-10-02-rocenante-ring4`, `-rocenante-mesh4`,
  `-mesh4-fourpaths`, `experiments/2026-10-03-ring4-bidirectional` and
  `-balanced-policy`; the TP4 recipe serves on the ring relay. Upstream
  status: local experiments, not submitted.

- `0012-packed-bf16-vocab-projection.patch` adds a vocabulary projection over
  an exact 12-bit packed form of BF16 weights: a sign-and-mantissa byte and a
  4-bit exponent code over a 15-exponent window per tensor, with out-of-window
  values kept exactly in a per-row list added after the main dot product.
  Packing refuses unless every value decodes back bit for bit. From
  `experiments/2026-10-03-packed-bf16-head`. 0001-0012 on the base yield patch
  head `97dc180b` and tree `7f666380`, the r5p image.

`series-r5o` keeps the r5o series (0001-0005) for the records that pinned it.

Not in the series: the W4A8 tiny-decode `swiglu_limit` fix
(`experiments/2026-09-23-karmic-kraken-reference/patches/b12x/0002-tiny-decode-swiglu-limit.patch`)
is an upstream contribution. The promoted runtime disables tiny decode instead
(`B12X_W4A8_TINY_DECODE=0`).
