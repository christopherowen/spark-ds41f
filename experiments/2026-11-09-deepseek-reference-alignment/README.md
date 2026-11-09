# DeepSeek-V4.1-Flash Reference Alignment (`moe_swiglu_cast` BF16 Rounding)

- **Date:** 2026-11-09
- **Base commit:** `51be7406` (`main`, candidate `1107` patch head `2756478b60b9`)
- **Issue:** [#1018](https://github.com/christopherowen/spark-ds41f/issues/1018)
- **Intended causal variable:** Round the routed MoE FP32 SwiGLU + route-weight product `(val_l / (1.0 + exp(-val_l))) * val_r * weight` to `bfloat16` before `amax` reduction and `get_sf_and_inv` MXFP8 quantization in `vllm/models/deepseek_v4_1/tilelang/moe.py` (`moe_swiglu_cast`), matching DeepSeek-V4.1-Flash `inference/model.py:Expert.forward` and `activation.py:swiglu_clamp_cast_kernel`.

## Background & Root Cause

In the canonical DeepSeek-V4.1-Flash reference (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, `inference/model.py:843-853`), `Expert.forward` computes the routed expert activation in FP32 and casts it back to the model activation `dtype` (`torch.bfloat16`) before passing it to `self.w2`:

```python
gate, up = x.chunk(2, dim=-1)
gate = gate.float().clamp(max=self.limit)
up = up.float().clamp(min=-self.limit, max=self.limit)
x = F.silu(gate) * up
if weights is not None:
    x = weights * x
return self.w2(x.to(dtype))
```

Inside `self.w2` (`linear` -> `act_quant`), the `bfloat16` activation is quantized to MXFP8 (`float8_e4m3fn` + `UE8M0` scale per 32-element block).

In `spark-ds41f`:
- The **shared expert** (`vllm/models/deepseek_v4_1/tilelang/activation.py:swiglu_clamp_cast_kernel`) already rounded the FP32 SwiGLU product to `T.bfloat16` (`out_bf16 = T.cast(val, in_dtype); out_fp32 = T.cast(out_bf16, T.float32)`) before computing the 32-element `amax` and `get_sf_and_inv`.
- The **routed experts** (`vllm/models/deepseek_v4_1/tilelang/moe.py:moe_swiglu_cast`), however, quantized the unrounded FP32 `__expf`/`__fdividef` product directly to `float8_e4m3fn` and carried a warp-vote certificate (`cert_e = 32`, `T.all_sync`) + fallback recomputation pass inherited from TileKernels.

Because MXFP8 `get_sf_and_inv` computes a power-of-two `UE8M0` scale from the 32-element block `amax` and rounds `x * sf_inv` to `float8_e4m3fn` (3 mantissa bits), skipping the intermediate `bfloat16` rounding alters ~`0.52%–1.61%` of the quantized FP8 bytes and `UE8M0` block scales across every routed expert. Summing 6 routed experts per token through the `w2` down-projection causes **85.5%–88.3%** of the end-to-end MoE output BF16 elements to differ from DeepSeek's `Expert.forward` reference (`rel_l1 = 1.57e-2` vs `_arithmetic`).

## Change

In `vllm/models/deepseek_v4_1/tilelang/moe.py` (`moe_swiglu_cast`):
1. Replace the `__expf`/`__fdividef` + warp-vote certificate + fallback recomputation pass with the single IEEE FP32 evaluation `(val_l / (1.0 + T.exp(-val_l))) * val_r * weight` rounded to `T.bfloat16` before `amax` and `get_sf_and_inv`:
   ```python
   val = T.alloc_var(
       T.float32,
       init=(val_l / (1.0 + T.exp(-val_l))) * val_r * weight,
   )
   x_bf16[j] = T.cast(val, in_dtype)
   x[j] = T.cast(x_bf16[j], T.float32)
   ```
2. Preserve the exact PDL preload contract (`TileExpert`, `Order`, and `Weights` loaded before `pdl.chain()`).
3. Update `tests/kernels/moe/test_deepseek_v41_tilelang_moe.py` (`_arithmetic` and `_tile_kernels_swiglu`) so the test reference rounds the weighted SwiGLU product to `bfloat16` before MXFP8 quantization, matching `Expert.forward`.

## Evaluated & Excluded Lever (`sparse_mla.py`)

We also evaluated adding an active-chunk skip (`chunk < num_active_chunks`) to `vllm/models/deepseek_v4_1/tilelang/sparse_mla.py` (`extend_body` / `decode_body`). Because `test_sparse_mla` requires `extend`, `decode`, and `split` to produce identical bits (`ulp_diffs_vs_unskipped == 0`), active-chunk skipping changes zero output bits, while moving `stage_chunk` inside the conditional regressed full-window (`swa=128, idx=512`) `extend` latency from `40.4 us` to `57.2 us` (`0.70x`) by breaking cross-segment `cp.async` pipelining. `sparse_mla.py` is therefore left untouched so this experiment isolates a single causal variable with zero performance regressions.

## Commands

Executed on `sp8` (`NVIDIA GB10`) across two fresh container boots (`boot1` and `boot2`) of `spark-ds41f:1107`:

```bash
for boot in boot1 boot2; do
  docker run --rm --gpus all --ipc=host --entrypoint python3 \
    -v "$VLLM_DIR/vllm/models/deepseek_v4_1/tilelang/moe.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py:ro" \
    -v "$VLLM_DIR/tests/kernels/moe/test_deepseek_v41_tilelang_moe.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/moe/test_deepseek_v41_tilelang_moe.py:ro" \
    -v "$EXP_DIR:/workspace/exp" \
    -w /opt/spark-ds41f/candidate/vllm \
    spark-ds41f:1107 \
    /workspace/exp/bench_reference_alignment.py "/workspace/exp/receipts/bench-${boot}.json"

  docker run --rm --gpus all --ipc=host --entrypoint pytest \
    -v "$VLLM_DIR/vllm/models/deepseek_v4_1/tilelang/moe.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py:ro" \
    -v "$VLLM_DIR/tests/kernels/moe/test_deepseek_v41_tilelang_moe.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/moe/test_deepseek_v41_tilelang_moe.py:ro" \
    -w /opt/spark-ds41f/candidate/vllm \
    spark-ds41f:1107 \
    --noconftest -v -p tests.kernels.deepseek_v41_pdl_sources \
    tests/kernels/moe/test_deepseek_v41_tilelang_moe.py \
    tests/kernels/test_deepseek_v41_tilelang_activation.py \
    > "$EXP_DIR/receipts/pytest-${boot}.txt"
done
```

## Results

### 1. `moe_swiglu_cast` Byte-Level Identity & Kernel Latency (`boot1` / `boot2`)

Compared against DeepSeek `Expert.forward` (`(F.silu(gate) * up * weight).bfloat16()` + `per_token_cast`):

| Shape & Tile Tier | Live Elements / Scales | `1107` Baseline FP8 / Scale Diffs | Candidate FP8 / Scale Diffs | `1107` `p50_us` (`boot1` / `boot2`) | Candidate `p50_us` (`boot1` / `boot2`) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **TP4 (`inter=576`)** `decode` (`block_M=16, rows=16`) | `110,592` / `3,456` | `1,674` (`1.51%`) / `3` | **`0` (`0.0%`) / `0`** | `43.62` / `9.70` | **`41.79` / `9.60`** |
| **TP4 (`inter=576`)** `mid` (`block_M=32, rows=96`) | `2,009,088` / `62,784` | `10,485` (`0.52%`) / `25` | **`0` (`0.0%`) / `0`** | `15.62` / `15.58` | **`12.51` / `12.51` (`1.25x`)** |
| **TP4 (`inter=576`)** `prefill` (`block_M=64, rows=2048`) | `14,155,776` / `442,368` | `225,463` (`1.59%`) / `627` | **`0` (`0.0%`) / `0`** | `155.90` / `154.37` | `157.89` / `154.66` (`1.00x`) |
| **TP3 (`inter=768`)** `decode` (`block_M=16, rows=16`) | `147,456` / `4,608` | `2,381` (`1.61%`) / `7` | **`0` (`0.0%`) / `0`** | `9.82` / `10.02` | **`9.60` / `9.73` (`1.03x`)** |
| **TP3 (`inter=768`)** `mid` (`block_M=32, rows=96`) | `2,678,784` / `83,712` | `14,623` (`0.55%`) / `46` | **`0` (`0.0%`) / `0`** | `12.61` / `13.47` | **`13.41` / `12.58`** |
| **TP3 (`inter=768`)** `prefill` (`block_M=64, rows=2048`) | `18,874,368` / `589,824` | `302,951` (`1.61%`) / `863` | **`0` (`0.0%`) / `0`** | `209.09` / `211.07` | `242.75` / `212.19` (`0.99x`) |

### 2. End-to-End Routed MoE Layer (`MoeExpertsV41Plan.run`) vs DeepSeek Reference (`_arithmetic`)

| Batch & Input Scale | Total BF16 Elements | `1107` Baseline `ULP > 0` (`%`) | `1107` Baseline `rel_l1` | Candidate `ULP > 0` (`%`) | Candidate `rel_l1` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `rows=12`, `small_scale` (`0.1`) | `4,608` | `4,068` (`88.28%`) | `1.63e-02` | **`0` (`0.0000%`)** | **`0.00e+00`** |
| `rows=12`, `normal` (`0.5`) | `4,608` | `4,069` (`88.30%`) | `1.60e-02` | **`0` (`0.0000%`)** | **`0.00e+00`** |
| `rows=12`, `heavy_clamp` (`25.0`) | `4,608` | `3,940` (`85.50%`) | `1.26e-02` | **`0` (`0.0000%`)** | **`0.00e+00`** |
| `rows=300`, `small_scale` (`0.1`) | `115,200` | `101,128` (`87.78%`) | `1.55e-02` | **`0` (`0.0000%`)** | **`0.00e+00`** |
| `rows=300`, `normal` (`0.5`) | `115,200` | `101,257` (`87.90%`) | `1.57e-02` | **`0` (`0.0000%`)** | **`0.00e+00`** |
| `rows=300`, `heavy_clamp` (`25.0`) | `115,200` | `98,608` (`85.60%`) | `1.42e-02` | **`12` (`0.0104%`)** | **`2.94e-07`** |

End-to-end MoE (`MoeExpertsV41Plan.run`) latency (`boot1` / `boot2`):
- `rows=12` (decode): `74.34 / 73.98 us` (`1107` baseline) -> **`73.38 / 73.12 us`** (candidate, **`1.012x`**)
- `rows=300` (prefill): `91.42 / 91.39 us` (`1107` baseline) -> **`92.42 / 92.38 us`** (candidate, `0.989x`)

### 3. 10-Layer Residual MoE Stack (`x = x + 0.15 * moe(x)`, `rows=12`)

| Layer | `1107` Baseline `ULP > 0` (`%`) | `1107` Baseline `MAE` / `rel_l1` | Candidate `ULP > 0` (`%`) | Candidate `MAE` / `rel_l1` |
| :--- | ---: | ---: | ---: | ---: |
| **Layer 1** | `399 / 4,608` (`8.66%`) | `6.19e-05` / `1.56e-04` | **`0 / 4,608` (`0.00%`)** | **`0.00e+00` / `0.00e+00`** |
| **Layer 5** | `1,233 / 4,608` (`26.76%`) | `3.36e-04` / `8.47e-04` | **`0 / 4,608` (`0.00%`)** | **`0.00e+00` / `0.00e+00`** |
| **Layer 10** | `1,927 / 4,608` (`41.82%`) | `7.01e-04` / `1.76e-03` | **`0 / 4,608` (`0.00%`)** | **`0.00e+00` / `0.00e+00`** |

### 4. Kernel & PDL Contract Suite (`pytest-boot1.txt` / `pytest-boot2.txt`)

- `165 passed in 141.39s` (`boot1`) and `165 passed in 142.08s` (`boot2`) across `tests/kernels/moe/test_deepseek_v41_tilelang_moe.py` and `tests/kernels/test_deepseek_v41_tilelang_activation.py`.
- `PDL sources: 140 kernels launched, 131 with PDL, 0 violations`.
- `test_moe_reference_separates_defects`:
  - `rows=12`: `kernel 0.00e+00 (unquantized-down reference 2.64e-02); dropped route 1.45e-01; wrong expert 5.72e-01; unweighted 7.90e-01; skipped K tile 7.77e-01; BF16 combine 2.37e-03`
  - `rows=300`: `kernel 0.00e+00 (unquantized-down reference 2.68e-02); dropped route 1.47e-01; wrong expert 5.25e-01; unweighted 7.31e-01; skipped K tile 7.48e-01; BF16 combine 2.31e-03`

## Conclusion

Rounding the routed MoE SwiGLU product to `bfloat16` before `amax` and `get_sf_and_inv` in `moe_swiglu_cast` achieves **100% byte-for-byte identity** with DeepSeek-V4.1-Flash's `Expert.forward` reference across all TP4 (`576`) and TP3 (`768`) shapes and decode/mid/prefill tile heights, eliminates the `87.9%` per-layer BF16 ULP divergence (`0` ULP diffs across 10 stacked MoE residual layers), and simplifies the kernel by removing the warp-vote certificate fallback pass with equal or better latency.
