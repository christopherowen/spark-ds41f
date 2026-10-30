# Indexer Query `q_cast` Reuse & Fused `RoPE + MXFP4` Quantize (2026-10-30)

## Summary

On each of the 11 index-source layers, `DeepseekV4Attention._forward` already produces `q_cast` (`MXFP8Rows` of `qr`) for `self.wq_b(qr)` (either inside `attention_front` at decode sizes or via `kernels.cast_mxfp8(qr)` at prefill sizes). Previously:

1. `self.indexer.wq_b` (`(4096, 1280)` block-32 FP8 projection) was not marked `tilelang_mxfp8_input = True`, so `_index_query` re-ran `pdl.per_token_cast(qr)` on the exact same normalized `qr` tensor on every index-source layer.
2. `_index_query` launched `rope_(iq, positions, ...)` to rotate the trailing 64 columns of `[rows, 32, 128]` in global memory, and immediately launched `mxfp4_quantize_v41` (`quantize_mxfp4`), which read `[rows, 32, 128]` back from global memory twice (once for `amax` and once for E2M1 packing).

This change:
- Marks `self.indexer.wq_b.tilelang_mxfp8_input = True` and feeds `q_cast` (or `slice_mxfp8(q_cast, first, last)` under sequence-parallel prefill) into `self.indexer.wq_b` in `_index_query` (including under `SPARK_DS41F_INDEXER_SPLIT_CHECK=1`).
- Fuses GPT-J interleaved RoPE (`T.ieee_fmaf` + BF16 rounding) of the trailing 64 columns (`k = 2, 3` 32-element groups) into `mxfp4_quantize_v41` in registers, and caches each thread's 32-element group in registers so `X` is loaded once instead of twice.
- Drops the now-unreached `(32, 128, 1, False)` `rope_call` from `DeepseekV4Attention.declare_kernels()`.

## Verification & Benchmark (`dgx4`, `spark-ds41f:1030`)

- **Unit & PDL tests:** `test_deepseek_v4_1_attention.py` (`7 passed`), `test_deepseek_v41_tilelang_linear.py` (`3 passed`), `test_deepseek_v41_tilelang.py` (`6 passed`), `0` PDL violations (`-p tests.kernels.deepseek_v41_pdl_sources`).
- **Bit-for-bit equality:** `torch.equal` on both `iq_data` (`[rows, 32, 64]`) and `iq_scale` (`[rows, 32, 4]`) across all tested row counts (`1..4096`) and slices.

| Rows | Baseline (µs/layer) | Fused (µs/layer) | Saved (µs/layer) | Saved across 11 layers (µs/step) | Speedup | Launches/layer |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 61.51 | 31.87 | 29.63 | 325.97 | 1.93× | 4 → 2 |
| 8 | 61.24 | 31.73 | 29.51 | 324.59 | 1.93× | 4 → 2 |
| 16 | 61.52 | 31.87 | 29.65 | 326.18 | 1.93× | 4 → 2 |
| 32 | 62.09 | 31.86 | 30.23 | 332.57 | 1.95× | 4 → 2 |
| 64 | 62.13 | 31.80 | 30.33 | 333.62 | 1.95× | 4 → 2 |
| 512 | 78.45 | 71.27 | 7.18 | 78.94 | 1.10× | 4 → 2 |
| 2048 | 327.78 | 247.92 | 79.86 | 878.49 | 1.32× | 4 → 2 |
| 4096 | 682.15 | 591.37 | 90.78 | 998.60 | 1.15× | 4 → 2 |
