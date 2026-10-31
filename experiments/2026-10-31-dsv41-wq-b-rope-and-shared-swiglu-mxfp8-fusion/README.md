# DeepSeek V4.1 TileLang: Fused `wq_b` RoPE Epilogue & Shared-Expert SwiGLU + MXFP8 Cast (Part 2 Items 2 & 3 + Best of PR #788)

## Motivation
Across every decode step (61 main layers + 1 DSpark draft layer = 62 layers):
1. **Attention Query Projection (`wq_b`) + RoPE (`attention.py` / `gemm.py`)**:
   - `wq_b` projects normalized 1536-dim latent queries `q_c` (`[T, 1536]`) to 16384-dim `q` (`[T, 32 * 512]`).
   - Immediately after `wq_b`, a separate `rope_` kernel reads and writes the trailing `qk_rope_head_dim = 64` columns of each 512-dim query head (`32 * 64 = 2048` columns total).
   - Because `block_N = 64` in `_mxfp8_gemm` and `_mxfp8_gemm_decode` divides `head_dim = 512` (`512 = 8 * 64`), the trailing 64-dim RoPE slice of every query head maps to a single tile (`(bx * 64) % 512 == 448`), allowing interleaved RoPE to execute in shared memory before writing the tile to global memory.
2. **Shared-Expert SwiGLU + Clamped FP32 + MXFP8 Quantization (`activation.py` / `nvidia/model.py`)**:
   - In all 61 MoE layers (58 routed MoE + 3 dense MLP layers), the shared expert computes `gate_up_proj` (`[T, 2 * 576]` or `[T, 2 * 18432]`), launches `swiglu_clamp_kernel` to write BF16 intermediate activations to global memory, and then `down_proj` launches a separate `per_token_cast` kernel to re-read that BF16 tensor and quantize 32-element blocks to `float8_e4m3fn` + packed `UE8M0` scales.
   - Fusing `SiluAndMulClampFP32` and 32-element MXFP8 block quantization into `swiglu_clamp_cast_kernel` (`activation_swiglu_clamp_cast`) eliminates the intermediate BF16 global memory round-trip and saves 61 `per_token_cast` kernel launches per step.
3. **PDL Chain Completion on `bf16_gemm_partials` (Best of PR #788)**:
   - Places `bf16_gemm_partials` on the PDL chain (`TL_DISABLE_WARP_SPECIALIZED: True` + `pdl.chain()`).

## Changes
- `vllm/models/deepseek_v4_1/tilelang/gemm.py`:
  - Added `rope_head_dim: int = 0, rope_dim: int = 64` parameters to `_mxfp8_gemm` and `_mxfp8_gemm_decode`. When `rope_head_dim > 0`, tiles with `(bx * block_N) % rope_head_dim == rope_head_dim - block_N` rotate interleaved pairs `(2*d, 2*d+1)` in shared memory using `Positions[row]` and `CosSin[pos, :]` before copying to global memory.
  - Added `pdl.chain()` and `TL_DISABLE_WARP_SPECIALIZED: True` to `_bf16_gemm_partials`.
- `vllm/models/deepseek_v4_1/tilelang/linear.py`:
  - Extended `provide_mxfp8` with `positions: torch.Tensor | None = None` (`provided_positions(x)`) and wired `tilelang_rope_head_dim` / `tilelang_cos_sin` into `_block32_kernels`, `_declare_block32`, `_block32_gemm`, and `_block32_linear`.
- `vllm/models/deepseek_v4_1/attention.py`:
  - Set `self.wq_b.tilelang_rope_head_dim = self.head_dim` and `self.wq_b.tilelang_cos_sin = self.rotary_emb.cos_sin_cache`, passed `positions=positions` to `mxfp8_inputs(q_cast, positions=positions)`, and removed the separate query `rope_` launch when `wq_b` fuses RoPE.
- `vllm/models/deepseek_v4_1/tilelang/activation.py`, `vllm/models/deepseek_v4_1/kernels.py`, `vllm/models/deepseek_v4_1/nvidia/model.py`, `vllm/models/deepseek_v4/nvidia/model.py`:
  - Added `swiglu_clamp_cast_kernel` (`activation_swiglu_clamp_cast`, on the PDL chain) and `SiluAndMulClampFP32(..., mxfp8_output=True).forward_mxfp8(x)`, feeding pre-quantized `QuantizedTensor` directly into `down_proj` via `provide_mxfp8`.

## Verification
Verified on `dgx4` (`spark-ds41f:19f2c20ed4d6-next-21`):
- `pytest --noconftest tests/kernels/test_deepseek_v41_tilelang_activation.py tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py tests/kernels/test_deepseek_v41_tilelang_pdl.py` -> **86 passed, 0 PDL violations** (`154 kernels launched, 112 with PDL`).
