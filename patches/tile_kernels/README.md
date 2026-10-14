# TileKernels patches

Applied after the pinned deepseek-ai/TileKernels revision (`66258df`, release 2.0.0) when a
lock sets `tile_kernels.patch_series` to `patches/tile_kernels/series`.

- **0001, routed SwiGLU clamp keeps a NaN.** In `swiglu_forward_and_per_token_cast`, the
  clamps `T.min`/`T.max` lower to `fminf`/`fmaxf`, which return the limit for a NaN.
  DeepSeek's reference clamps with `torch.clamp`, which keeps it. Compare-and-select keeps
  it too, as the kernel's own `count_clamp` branch already did.
  - Measured 2026-10-14 (spark-ds41f `experiments/2026-10-14-moe-swiglu`, bundle
    `tk-swiglu`) on GB10 at 1-8,192 rows: identical bytes and scales on finite inputs,
    0.99-1.01x time, and a NaN gate value now reaches the output.
  - Worth offering upstream.
