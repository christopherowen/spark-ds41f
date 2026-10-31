# Fuse DSpark Context `main_norm` and MXFP8 Per-Token Cast

## Summary

Eliminates the standalone `kernels.cast_mxfp8(main_x)` launch in `DSparkDeepseekV4Model.precompute_and_store_context_kv` by fusing the MXFP8 per-token quantization into `main_norm` (`TileLangRMSNorm.forward_with_cast`):

1. **PDL-enabled `norm_forward_and_per_token_cast` in `TileLangRMSNorm`**:
   - Adds `pdl.norm_forward_and_per_token_cast` in `vllm/models/deepseek_v4_1/tilelang/pdl.py` and `TileLangRMSNorm.forward_with_cast` (plus `_warm_with_cast` when `with_cast = True`) in `vllm/models/deepseek_v4_1/tilelang/norm.py`.
   - `tile_kernels.quant.norm_forward_and_per_token_cast` rounds the normalized rows to BF16 before computing the 32-column block `amax` and quantizing to E4M3 + packed UE8M0, producing bit-for-bit identical BF16 `rows`, E4M3 `values`, and UE8M0 `scales` (`MXFP8Rows`) to running `norm_forward` followed by `per_token_cast`.

2. **Single-Launch Context Norm + Cast in `DSparkDeepseekV4Model`**:
   - When `_context_kv_projections` consume a producer's MXFP8 cast (`tilelang_mxfp8_input = True`), `combine_hidden_states` (and `combine_aux_hidden_states_into`) calls `self.main_norm.forward_with_cast(projected, out=out)` and retains the resulting `MXFP8Rows` on `self._context_main_cast`.
   - `precompute_and_store_context_kv` consumes `self._context_main_cast` when its `rows` buffer matches `main_x`, feeding all 3 draft layers' `_context_kv_projections` via `kernels.mxfp8_inputs(main_cast)` without launching `cast_mxfp8`.

## Verification

Verified on `dgx4` inside `spark-ds41f:19f2c20ed4d6-next-20` with `-p tests.kernels.deepseek_v41_pdl_sources`:

- `tests/kernels/test_deepseek_v41_tilelang_dspark_context.py`: 18 passed, 0 failed (`PDL sources: 4 kernels launched, 3 with PDL, 0 violations`), including bit-for-bit parity checks across `rows in {1, 6, 32, 96}` and end-to-end verification that `combine_hidden_states` + `precompute_and_store_context_kv` invokes zero standalone `cast_mxfp8` calls.

## Artifacts

- Patch (against `rc20` `1f18be14470339c1227ed0432b896a8ca09811b5`): [`0001-dspark-main-norm-mxfp8-cast.patch`](./0001-dspark-main-norm-mxfp8-cast.patch)
