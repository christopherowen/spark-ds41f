# Fuse Shared-Expert SwiGLU MXFP8 Cast and Reuse `q_cast` in `indexer.wq_b`

## Summary

Eliminates two remaining standalone `per_token_cast` kernel launches per layer (61 shared-expert casts + 4 indexer query casts = 65 kernel launches per forward step across both decode and prefill) by producing or reusing the consumer projection's MXFP8 activation cast in registers:

1. **Shared-Expert SwiGLU -> `down_proj` (`SiluAndMulClampFP32` with `cast=True`)**:
   - `swiglu_clamp_kernel` in `vllm/models/deepseek_v4_1/tilelang/activation.py` already computes the BF16 clamped SwiGLU output in registers (`_VEC = 8` adjacent columns per thread, 4 adjacent threads per 32-column scale block).
   - With `cast=True`, `swiglu_clamp_kernel` reduces the 32-column block `amax` across the 4 lanes via two warp-uniform `__shfl_xor_sync` steps and writes the E4M3 values and UE8M0 scale byte via `cast_block_mxfp8` in the same launch.
   - `shared_experts.down_proj` is marked `tilelang_mxfp8_input = True`, and `DeepseekV4MLP.forward` hands `x_q` from `act_fn.forward_with_cast(gate_up)` to `down_proj` via `provide_mxfp8`.

2. **Attention `q_cast` -> `indexer.wq_b` (`slice_mxfp8`)**:
   - `DeepseekV4Attention._forward` already produces `q_cast` (`MXFP8Rows` for `qr`) before `self.wq_b(qr)` (either fused in `attention_front` at decode sizes or via `kernels.cast_mxfp8(qr)` at prefill sizes).
   - Previously, `self.indexer.wq_b` (`ReplicatedLinear(1280, 4096)`) was not marked `tilelang_mxfp8_input = True`, causing `_index_query` to run a duplicate `per_token_cast` on `qr[first:last]` on every indexer layer.
   - Marks `self.indexer.wq_b.tilelang_mxfp8_input = True` and adds `slice_mxfp8(q_cast, first, last)` in `vllm/models/deepseek_v4_1/tilelang/linear.py` so `_index_query` reuses `q_cast` (including sequence-parallel prefill sub-slices `[first:last]`) without re-casting.

## Verification

Verified on `dgx4` inside `spark-ds41f:1030` with the PDL source-inspection plugin enabled (`-p tests.kernels.deepseek_v41_pdl_sources`):

- `tests/kernels/test_deepseek_v41_tilelang_activation.py`: 18 passed (including `test_forward_with_cast_matches_separate_cast` across `rows in {1, 6, 96, 2024}` and `n in {512, 576}`).
- `tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py`: 32 passed (including `test_slice_mxfp8_feeds_row_range`).
- PDL source inspection: `97 kernels launched, 54 with PDL, 0 violations`.

## Artifacts

- Candidate vLLM branch: `dsv41-producer-mxfp8-casts` (`7d420a660`)
- Patch: [`dsv41-producer-mxfp8-casts.patch`](./dsv41-producer-mxfp8-casts.patch)
