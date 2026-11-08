# Producer MXFP8 `bfloat162` Exponent Scaling and Paired E4M3 Conversion

## Hypothesis

Replacing FP32 unpacking, FP32 block-amax reduction, and FP32 reciprocal multiplication with `bfloat162` operations (`__habs2`, `__hmax2`, `__hmul2` by the power-of-two UE8M0 reciprocal `2^(127 - exp_sf)`) plus paired E4M3 conversion (`cvt.rn.satfinite.e4m3x2.bf16x2` on CUDA >= 13.1, or shift/mask `float2` + `__nv_cvt_float2_to_fp8x2` on CUDA 13.0) in DeepSeek-V4.1 TileLang producer MXFP8 cast kernels reduces SASS instruction count and register pressure while remaining 100% bit-identical (`torch.equal`) to the FP32 reference path.

## Intended delta

Base commit: `cc20ed18b98916797236f32c361a7316420d822c` (vLLM upstream `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` with `patches/vllm/0001..0044`).

Single causal variable: local patch `patches/vllm/0045-tilelang-producer-mxfp8-bf16x2-cast.patch`:

1. `vllm/models/deepseek_v4_1/tilelang/columns.py` (`vllm_columns_join_and_cast_8`):
   - Compute per-thread 8-element `bfloat16` absolute maximum via `__habs2` and `__hmax2` before warp-4 butterfly reduction.
   - Construct the power-of-two reciprocal scale directly as a duplicated `__nv_bfloat162` (`(254u - ue8m0_byte) << 7`), scale via `__hmul2`, and convert each pair via `cvt.rn.satfinite.e4m3x2.bf16x2` on CUDA >= 13.1 (or packed shift/mask `float2` + `__nv_cvt_float2_to_fp8x2` on CUDA 13.0).
2. `vllm/models/deepseek_v4_1/tilelang/gemm.py` (`cast_block_mxfp8` / `vllm_cast_block_mxfp8_8`), `activation.py` (`swiglu_clamp_cast_kernel`), and `mhc.py` (`finalize`):
   - Replace the per-element FP32 cast/amax loop and scalar `T.Cast(T.float8_e4m3fn, values[v] * inv)` in `cast_block_mxfp8` with `vllm_cast_block_mxfp8_8` operating directly on the 8-element `bfloat16` register array (`__habs2`/`__hmax2` -> warp-4 `__shfl_xor_sync` -> `__hmul2` -> paired E4M3 conversion).
3. `vllm/models/deepseek_v4_1/tilelang/quant_common.py` (`IMPORT_BF16X2_FP8_CVT`, `get_per_token_cast_kernel_cuda`), `norm.py` (`get_norm_forward_and_per_token_cast_kernel_cuda`), and `front.py` (`attention_front_kernel`):
   - Under CUDA 13.0 (`spark-ds41f:1107`), `<cuda_fp8.hpp>` implements `__nv_cvt_bfloat16raw2_to_fp8x2` as two scalar `__nv_cvt_float_to_fp8` calls (`2x F2FP` per `bfloat162`). Overriding `__tl_cvt_bfloat162_to_fp8x2` on CUDA < 13.1 to unpack via shift/mask `make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u))` and call `__nv_cvt_float2_to_fp8x2` emits a single paired `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` instruction per `bfloat162` pair. On CUDA >= 13.1 (`13.5.37`), `<cuda_fp8.hpp>` natively emits `cvt.rn.satfinite.e4m3x2.bf16x2` (`F2FP.SATFINITE.E4M3.BF16.UNPACK_B_MERGE_C`).
4. `moe_swiglu_cast` (`moe.py`):
   - Verified that `moe_swiglu_cast` computes SwiGLU in FP32 and already scales in FP32 with `sf_inv_dtype=T.float32`, which TileLang vectorizes into `__nv_cvt_float2_to_fp8x2` (`16` paired `F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C` instructions for 32 elements); left unchanged.

## Quality and safety gates

- **Bit-exactness (`torch.equal`)**: Because the UE8M0 reciprocal scale `2^(127 - exp_sf)` is an exact power of two and `exp_sf = ceil_log2(max(amax, 1e-4) / 448)` satisfies `110 <= exp_sf <= 254` (`2 <= 254 - exp_sf <= 144`), any normal `bfloat16` value scaled into the E4M3 normal/subnormal range (`>= 2^-10`) has a scaled exponent `>= 117`, well above the `bfloat16` subnormal threshold. Thus `__hmul2(bf2, inv_bf2)` only adds `(127 - exp_sf)` to the 8-bit exponent without touching the 7-bit mantissa, producing bit-identical E4M3 outputs to FP32 scaling.
- Verified with `runs/verify_bitexact.py` across 102 output tensors covering `join_columns_and_cast_mxfp8`, `cast_mxfp8`, `per_token_cast`, `activation_swiglu_clamp_cast`, `moe_swiglu_cast`, `mhc_finalize`, `norm_forward_and_per_token_cast`, and `attention_front` (including zeros, `-0.0`, subnormal/floor `1e-6`, and saturation `500.0` edge cases):
  - `cand_130` (`nvcc 13.0.88`): `102/102` tensors bit-identical (`0` mismatches).
  - `cand_135` (`nvcc 13.5.37`): `102/102` tensors bit-identical (`0` mismatches).
- Verified with `pytest tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py`: `97 passed, 0 failed`.

## SASS and Register Analysis (`sm_121a` on `sp6`)

Instruction counts exclude trailing `NOP` alignment padding:

| Kernel | Toolchain | Base Insts | Cand Insts | Inst Delta | Base Regs | Cand Regs | Reg Delta | Key Opcode Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `join_columns_and_cast_mxfp8` | `13.0.88` | 286 | 277 | -9 (-3.1%) | 38 | 25 | -13 (-34.2%) | `FMUL` 27->3 (`HMUL2` 0->6), `FMNMX` 30->9 (`HMNMX2` 0->12) |
| `join_columns_and_cast_mxfp8` | `13.5.37` | 277 | 244 | -33 (-11.9%) | 38 | 23 | -15 (-39.5%) | `FMUL` 27->3 (`HMUL2` 0->6), `FMNMX` 30->9 (`HMNMX2` 0->12), `F2FP.BF16` 12 |
| `per_token_cast_kernel` (`cast_mxfp8`) | `13.0.88` | 283 | 202 | -81 (-28.6%) | 32 | 32 | 0 | `F2FP` 32->16 (-50.0%) |
| `per_token_cast_kernel` (`cast_mxfp8`) | `13.5.37` | 172 | 172 | 0 (-39.2% vs 13.0 base) | 33 | 33 | 0 | `F2FP.BF16` 16 |
| `activation_swiglu_clamp_cast` | `13.0.88` | 450 | 445 | -5 (-1.1%) | 29 | 27 | -2 (-6.9%) | `F2F` 8->0, `FMUL` 18->9 (`HMUL2` 0->3), `FMNMX` 36->27 (`HMNMX2` 0->4) |
| `activation_swiglu_clamp_cast` | `13.5.37` | 446 | 432 | -14 (-3.1%) | 27 | 25 | -2 (-7.4%) | `F2F` 8->0, `FMUL` 18->9 (`HMUL2` 0->2), `FMNMX` 36->27 (`HMNMX2` 0->4), `F2FP.BF16` 4 |
| `mhc_finalize` | `13.0.88` | 2763 | 2697 | -66 (-2.4%) | 168 | 168 | 0 | `F2F` 104->0, `FMUL` 334->230 (`HMUL2` 0->27), `FMNMX` 148->44 (`HMNMX2` 0->52) |
| `mhc_finalize` | `13.5.37` | 2757 | 2598 | -159 (-5.8%) | 168 | 168 | 0 | `F2F` 104->0, `FMUL` 334->230 (`HMUL2` 0->27), `FMNMX` 148->44 (`HMNMX2` 0->52), `F2FP.BF16` 52 |
| `norm_forward_and_per_token_cast` | `13.0.88` | 821 | 725 | -96 (-11.7%) | 74 | 72 | -2 (-2.7%) | `F2FP` 60->40 (-33.3%) |
| `norm_forward_and_per_token_cast` | `13.5.37` | 690 | 690 | 0 (-16.0% vs 13.0 base) | 72 | 72 | 0 | `F2FP.BF16` 20 |
| `attention_front_kernel` | `13.0.88` | 1316 | 1265 | -51 (-3.9%) | 80 | 80 | 0 | `F2FP` 72->60 (-16.7%) |
| `attention_front_kernel` | `13.5.37` | 1220 | 1220 | 0 (-7.3% vs 13.0 base) | 80 | 80 | 0 | `F2FP` 60 (`F2FP.BF16` 12) |

## Workloads and 2-Boot Hardware Benchmark (`sp6`, GB10 SM121)

Measured via CUDA-graph replay (`warmup=50`, `rep=500` x 10 launches per graph = 5,000 kernel launches per point) across 2 independent container boots per toolchain (`spark-ds41f:1107` with `nvcc 13.0.88` and `nvcc 13.5.37`) under `flock /tmp/gpu.lock`. Full per-boot receipts are stored in `runs/bench_n3_graph_results.json`.

### 2-Boot Mean Latency (us)

| Kernel / Shape | 13.0 Base (us) | 13.0 Cand (us) | 13.0 Delta | 13.5 Base (us) | 13.5 Cand (us) | 13.5 Delta | 13.5 Cand vs 13.0 Base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `join_columns_m4` | 1.035 | 1.038 | +0.19% | 0.980 | 0.923 | -5.86% | -10.86% |
| `join_columns_m16` | 1.612 | 1.554 | -3.60% | 1.559 | 1.515 | -2.82% | -6.05% |
| `join_columns_m64` | 4.128 | 4.124 | -0.10% | 4.083 | 4.167 | +2.05% | +0.93% |
| `join_columns_m256` | 50.170 | 52.026 | +3.70% | 48.014 | 48.068 | +0.11% | -4.19% |
| `join_columns_m1024` | 365.217 | 366.329 | +0.30% | 369.078 | 365.701 | -0.92% | +0.13% |
| `cast_mxfp8_m1` | 0.886 | 1.004 | +13.37% | 0.904 | 0.857 | -5.14% | -3.22% |
| `cast_mxfp8_m4` | 1.039 | 0.941 | -9.52% | 0.847 | 0.820 | -3.24% | -21.12% |
| `cast_mxfp8_m16` | 1.050 | 1.035 | -1.43% | 0.988 | 0.947 | -4.20% | -9.86% |
| `cast_mxfp8_m64` | 1.443 | 1.314 | -8.94% | 1.288 | 1.307 | +1.51% | -9.42% |
| `cast_mxfp8_m256` | 2.684 | 2.558 | -4.69% | 2.567 | 2.574 | +0.23% | -4.13% |
| `cast_mxfp8_m1024` | 11.741 | 9.059 | -22.84% | 8.558 | 10.146 | +18.55% | -13.59% |
| `act_swiglu_m1` | 1.244 | 1.236 | -0.60% | 1.302 | 1.286 | -1.27% | +3.42% |
| `act_swiglu_m4` | 1.303 | 1.300 | -0.27% | 1.287 | 1.288 | +0.08% | -1.19% |
| `act_swiglu_m16` | 1.308 | 1.300 | -0.69% | 1.309 | 1.373 | +4.93% | +4.97% |
| `act_swiglu_m64` | 1.692 | 1.672 | -1.21% | 1.655 | 1.642 | -0.82% | -3.01% |
| `act_swiglu_m256` | 2.969 | 2.915 | -1.82% | 3.015 | 2.945 | -2.32% | -0.81% |
| `act_swiglu_m1024` | 6.482 | 6.326 | -2.40% | 6.581 | 6.272 | -4.69% | -3.22% |
| `mhc_finalize_m1` | 3.848 | 3.817 | -0.81% | 3.886 | 3.875 | -0.31% | +0.69% |
| `mhc_finalize_m4` | 3.931 | 3.887 | -1.12% | 3.963 | 3.838 | -3.15% | -2.35% |
| `mhc_finalize_m16` | 3.962 | 3.920 | -1.07% | 4.120 | 3.995 | -3.02% | +0.82% |
| `mhc_finalize_m64` | 5.240 | 5.095 | -2.79% | 5.287 | 5.175 | -2.12% | -1.25% |
| `mhc_finalize_m256` | 10.964 | 11.009 | +0.41% | 10.927 | 11.094 | +1.53% | +1.18% |
| `norm_cast_m1` | 1.998 | 1.954 | -2.18% | 1.983 | 1.902 | -4.06% | -4.78% |
| `norm_cast_m4` | 2.291 | 2.349 | +2.55% | 2.466 | 2.355 | -4.52% | +2.79% |
| `norm_cast_m16` | 2.365 | 2.428 | +2.71% | 2.348 | 2.418 | +3.00% | +2.28% |
| `norm_cast_m64` | 3.370 | 3.491 | +3.59% | 3.619 | 3.434 | -5.10% | +1.90% |
| `norm_cast_m256` | 9.466 | 9.659 | +2.04% | 9.492 | 9.619 | +1.34% | +1.61% |
| `norm_cast_m1024` | 213.137 | 215.834 | +1.27% | 214.969 | 214.320 | -0.30% | +0.55% |
| `attention_front_m1` | 1.381 | 1.364 | -1.20% | 1.336 | 1.325 | -0.82% | -4.02% |
| `attention_front_m4` | 1.438 | 1.431 | -0.52% | 1.472 | 1.446 | -1.83% | +0.49% |
| `attention_front_m16` | 1.659 | 1.655 | -0.24% | 1.679 | 1.672 | -0.42% | +0.84% |
| `attention_front_m64` | 1.685 | 1.683 | -0.15% | 1.810 | 1.740 | -3.89% | +3.26% |
| `attention_front_m256` | 3.162 | 3.169 | +0.24% | 3.266 | 3.233 | -1.01% | +2.26% |

## Conclusion

- **SASS & Register Reduction**: All 6 producer MXFP8 kernels achieve substantial instruction and/or register reductions:
  - `join_columns_and_cast_mxfp8`: `-33` SASS instructions (`-11.9%`) and `-15` registers (`38 -> 23`, `-39.5%`) on CUDA `13.5.37`, and `-13` registers (`38 -> 25`, `-34.2%`) on CUDA `13.0.88`.
  - `per_token_cast_kernel` (`cast_mxfp8`): `-81` SASS instructions (`-28.6%`, `F2FP` `32 -> 16`) on CUDA `13.0.88`, and `-111` SASS instructions (`-39.2%`) on CUDA `13.5.37` vs `13.0.88` base.
  - `activation_swiglu_clamp_cast`: `-14` SASS instructions (`-3.1%`) and `-2` registers (`27 -> 25`) on CUDA `13.5.37`.
  - `mhc_finalize`: `-159` SASS instructions (`-5.8%`, eliminating `104` `F2F`, `104` `FMUL`, and `104` `FMNMX` instructions) on CUDA `13.5.37`, and `-66` SASS instructions (`-2.4%`) on CUDA `13.0.88`.
  - `norm_forward_and_per_token_cast`: `-96` SASS instructions (`-11.7%`, `F2FP` `60 -> 40`) on CUDA `13.0.88`, and `-131` SASS instructions (`-16.0%`) on CUDA `13.5.37` vs `13.0.88` base.
  - `attention_front`: `-51` SASS instructions (`-3.9%`, `F2FP` `72 -> 60`) on CUDA `13.0.88`, and `-96` SASS instructions (`-7.3%`) on CUDA `13.5.37` vs `13.0.88` base.
- **Latency**: Decode and verification batch shapes (`M = 1..64`) improve consistently where compute/instruction overhead is exposed:
  - `cast_mxfp8` on CUDA `13.0.88`: `-9.52%` at `M=4`, `-8.94%` at `M=64`, `-4.69%` at `M=256`, `-22.84%` at `M=1024`; on CUDA `13.5.37` vs `13.0.88` base: `-21.12%` at `M=4`, `-9.86%` at `M=16`, `-9.42%` at `M=64`.
  - `join_columns_and_cast_mxfp8`: `-5.86%` at `M=4` and `-2.82%` at `M=16` on CUDA `13.5.37` (`-10.86%` and `-6.05%` vs `13.0.88` base).
  - `activation_swiglu_clamp_cast`: `-1.21%` to `-2.40%` at `M=64..1024` on CUDA `13.0.88` and `-2.32%` to `-4.69%` at `M=256..1024` on CUDA `13.5.37`.
  - `mhc_finalize`: `-1.12%` to `-2.79%` at `M=4..64` on CUDA `13.0.88` and `-2.12%` to `-3.15%` at `M=4..64` on CUDA `13.5.37`.
- **Recommendation**: Package as `patches/vllm/0045-tilelang-producer-mxfp8-bf16x2-cast.patch`.
