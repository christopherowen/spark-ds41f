# DSV4.1 TileLang `bf16_gemm_partials` PDL Chaining & Non-Warp-Specialized Decode Tiles

## Summary

In `vllm/models/deepseek_v4_1/tilelang/gemm.py`, decode split-K projections with BF16 weights (`moe.gate` `384 x 5120`, DSpark `moe.gate` `128 x 5120`, and `indexer.weights_proj` `32 x 5120`) launch `bf16_gemm_partials` (`gemm_bf16_partials_kernel`) followed by either `route_partials` or `splitk_reduce`:

1. **Missing `TL_DISABLE_WARP_SPECIALIZED: True` on Decode Split-K Tiles:**
   - Unlike every other decode split-K kernel in `gemm.py` (`mxfp8_gemm_partials`, `fp8w_bf16_gemm_partials`, `bf16_gemm_splitk`), `bf16_gemm_partials` omitted `pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}`.
   - Leaving warp specialization enabled on small decode tiles (`block_M = 16, 32, 64`) doubled the CTA thread count from 128 to 256 (dedicating 4 producer warps to TMA weight loads while unpadded dynamic-`M` activation rows used predicated `cp.async`), inflating `block_M = 16` kernel latency by ~40% (`11.17 µs -> 6.69 µs`) and preventing PDL from being enabled.
2. **Missing `pdl.chain()`:**
   - In the 43-layer forward-pass IKET trace (`table_43l.md`), `gemm_bf16_partials_kernel` (`45` launches/step, `2,049.6 µs/step`) had zero `wait.pdl` ranges and a `1,236.2 µs` cumulative launch gap after `mhc_finalize_kernel` because its CUDA-graph edges from `mhc_finalize_kernel` / `indexer_mxfp4_quantize_kernel` and into `router_partials_kernel` / `gemm_splitk_reduce_kernel` were non-programmatic.
   - Adding `TL_DISABLE_WARP_SPECIALIZED: True` and `pdl.chain()` to `bf16_gemm_partials` (and moving `gemm_bf16_partials` from `WARP_SPECIALIZED` to `CHAIN` in `tests/kernels/deepseek_v41_pdl_sources.py`) makes the entire `mhc_finalize -> bf16_gemm_partials -> route_partials` and `indexer_mxfp4_quantize -> bf16_gemm_partials -> splitk_reduce` chains programmatic while preserving bit-identical FP32 partials.

## Verification & Benchmark (`dgx4`, `spark-ds41f:19f2c20ed4d6-next-20`)

### Correctness & PDL Verification
```bash
pytest -q -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/test_deepseek_v41_tilelang_pdl.py tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py
# 45 passed in 169.98s (0:02:49)
# PDL sources: 130 kernels launched, 95 with PDL, 0 violations
```

### CUDA-Graph Replay Latency (500 graph replays)

| Operation & Shape | Batch (`rows`, `block_M`) | `rc20` Baseline Median (`p10`) | Candidate Median (`p10`) | Delta |
| :--- | :---: | :---: | :---: | :---: |
| **`bf16_gemm_partials` (`384 x 5120`, 8 shards)** | `rows = 1`, `bm = 16` | `11.168 µs` (`10.880 µs`) | **`6.688 µs` (`6.432 µs`)** | **`-4.480 µs` (`-40.1%`)** |
| **`bf16_gemm_partials` (`384 x 5120`, 8 shards)** | `rows = 6`, `bm = 16` | `11.072 µs` (`10.688 µs`) | **`6.720 µs` (`6.528 µs`)** | **`-4.352 µs` (`-39.3%`)** |
| **`bf16_gemm_partials` (`384 x 5120`, 8 shards)** | `rows = 7`, `bm = 16` | `7.552 µs` (`6.624 µs`) | **`6.816 µs` (`5.408 µs`)** | **`-0.736 µs` (`-18.4%` p10)** |
| **`bf16_gemm_partials` (`384 x 5120`, 8 shards)** | `rows = 48`, `bm = 64` | `9.088 µs` (`8.640 µs`) | **`8.256 µs` (`7.584 µs`)** | **`-0.832 µs` (`-9.2%`)** |
| **Chained `mhc_entry + partials + route_partials`** | `rows = 1`, `bm = 16` | `19.712 µs` (`19.616 µs`) | **`17.888 µs` (`17.696 µs`)** | **`-1.824 µs` (`-9.3%`)** |
| **Chained `mhc_entry + partials + route_partials`** | `rows = 6`, `bm = 16` | `19.680 µs` (`19.616 µs`) | **`19.392 µs` (`18.048 µs`)** | **`-1.568 µs` p10 (`-8.0%`)** |
| **`indexer.weights_proj` (`32 x 5120`, 40 shards + reduce)** | `rows = 1`, `bm = 16` | `10.432 µs` (`10.176 µs`) | **`6.432 µs` (`6.176 µs`)** | **`-4.000 µs` (`-38.3%`)** |
| **`indexer.weights_proj` (`32 x 5120`, 40 shards + reduce)** | `rows = 6`, `bm = 16` | `10.400 µs` (`9.824 µs`) | **`6.528 µs` (`6.176 µs`)** | **`-3.872 µs` (`-37.2%`)** |
