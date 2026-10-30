# 2026-10-30: DSpark Context KV Fused `context_kv_front` & Shared `main_x` MXFP8 Cast

## Summary
- Enables split-K block-32 FP8 projections whose decode row tiles divide `DECODE_ROWS` (`(512, 5120)` in `TileLangBlock32Rows`) to consume a producer's `MXFP8Rows` cast (`_block32_split_mxfp8`), so `DSparkDeepseekV4Model.precompute_and_store_context_kv` casts `main_x` to MXFP8 once (`kernels.cast_mxfp8(main_x)`) and feeds all 3 draft layers' `_context_kv_projections`.
- Adds `context_kv_front` (`context_kv_front_kernel` in `tilelang/front.py` and `DeepseekV4Attention.insert_raw_context_kv`) which fuses `kv_norm` + trailing 64-column `rope_` + 528-byte SWA `cache_writer` in one launch without writing or reading the intermediate `[T, 512]` BF16 `kv` tensor.
- Cuts DSpark context KV preparation in `precompute_and_store_context_kv` from **18 kernel launches to 10 launches** at decode capacities (saving **8 launches per target step** inside `DSparkContextCudaGraphs`) and from **12 launches to 4 launches** at prefill sizes.

## Verification on `dgx4` (`spark-ds41f:1030`)
- `pytest -q -p tblib_stub -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/test_deepseek_v41_tilelang_front.py tests/kernels/test_deepseek_v41_tilelang_dspark_context.py tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py -k "context or split_reads or reads_its_producers"`: `30 passed`, `0` PDL violations (`23 kernels launched, 19 with PDL, 0 violations`).
- Bit-for-bit identical (`torch.equal`) across all 3 draft layers' SWA cache pages for `rows = 1..2048`.
- CUDA-event benchmark (`results.json`):
  - Decode context graph sizes (`rows = 1..96`): `249.0–252.8 µs` → `126.3–127.9 µs` (**1.97×–1.98× faster**, saving **123–125 µs/step**, 18 → 10 launches).
  - Prefill context preparation (`rows = 256, 2048`): `235.8 µs` → `127.8 µs` (**1.85× faster** at 256 rows) and `780.7 µs` → `457.2 µs` (**1.71× faster**, saving **323.5 µs** at 2048 rows, 12 → 4 launches).
