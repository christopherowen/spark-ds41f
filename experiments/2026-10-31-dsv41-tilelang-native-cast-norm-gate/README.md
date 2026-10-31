# Native TileLang `per_token_cast`, `norm_forward`, and `moe_topk_gate_forward` (Eliminating `tile_kernels` Runtime Imports)

## Summary

Ports the remaining 3 external `tile_kernels` primitives used by the DeepSeek-V4.1 TileLang backend directly into `vllm/models/deepseek_v4_1/tilelang/` as first-party `@tilelang.jit` kernels, removing all `import tile_kernels` dependencies across `vllm/`:

1. **[`vllm/models/deepseek_v4_1/tilelang/quant_common.py`](../../vllm/models/deepseek_v4_1/tilelang/quant_common.py)**:
   - Native TileLang quantization configuration and macros (`BaseCastConfig`, `CastInputConfig`, `CastOutputConfig`, `get_cast_output_config`, `get_sf_and_inv`, `get_sf_index`, `get_sf_inv_dtype`, `get_sf_shape`, `get_best_vectorize_size`, `load_sf`, `transform_sf`, `store_sf`).
   - Native `@tilelang.jit` `get_per_token_cast_kernel_cuda` (`per_token_cast_kernel`), `per_token_cast_impl`, `per_token_cast`, `per_token_cast_with_sf_only`, and `per_token_cast_with_precomputed_sf` with explicit `use_pdl: bool = False`.
2. **[`vllm/models/deepseek_v4_1/tilelang/norm.py`](../../vllm/models/deepseek_v4_1/tilelang/norm.py)**:
   - Native `@tilelang.jit` `get_norm_forward_and_per_token_cast_kernel_cuda` (`norm_forward_and_per_token_cast_kernel`), `norm_forward_and_per_token_cast_impl`, `norm_forward_and_per_token_cast`, and `norm_forward` with explicit `use_pdl: bool = False`.
3. **[`vllm/models/deepseek_v4_1/tilelang/router.py`](../../vllm/models/deepseek_v4_1/tilelang/router.py)**:
   - Native `@tilelang.jit` `get_moe_topk_gate_forward_kernel_cuda` (`moe_topk_gate_forward_kernel`), `sqrt_softplus_forward_cuda`, `warp_reduce_sum`, and `moe_topk_gate_forward` with explicit `use_pdl: bool = False`.
4. **[`vllm/models/deepseek_v4_1/tilelang/pdl.py`](../../vllm/models/deepseek_v4_1/tilelang/pdl.py)**, **[`front.py`](../../vllm/models/deepseek_v4_1/tilelang/front.py)**, and **[`moe.py`](../../vllm/models/deepseek_v4_1/tilelang/moe.py)**:
   - Replaces `tile_kernels` imports and global PDL toggle state with direct `use_pdl=True` calls into the native TileLang modules and removes dead `_tile_kernels(block_M)` scaffolding from `moe.py`.

## Verification

Verified on `dgx4` (`spark-ds41f:19f2c20ed4d6-next-21`) with `-p tests.kernels.deepseek_v41_pdl_sources`:
- `tests/kernels/test_deepseek_v41_tilelang_pdl.py` (including bit-exact parity against `tile_kernels` and full `LayerChain` CUDA-graph replay & programmatic edge checks)
- `tests/kernels/test_deepseek_v41_tilelang_front.py`
- `tests/kernels/moe/test_deepseek_v41_tilelang_router.py`
- `tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py`
