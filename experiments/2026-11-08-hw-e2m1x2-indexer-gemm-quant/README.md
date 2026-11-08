# Experiment: Hardware `cvt.rn.satfinite.e2m1x2.f32` (`F2FP.SATFINITE.E2M1.F32.PACK_AB`) and Vectorized Global Stores in TileLang Indexer and Fused GEMM+RoPE+MXFP4 Quantizers

- **Date**: 2026-11-08
- **Base commit (`deployment`)**: `cc20ed18b` (`upstreams.lock.json` vLLM patch head `320021e3c4dbb202d911ab70e2cd924c75420488`, patch `0761`)
- **Candidate patch**: `experiments/2026-11-08-hw-e2m1x2-indexer-gemm-quant/vllm/0762-perf-dsv41-hw-e2m1x2-indexer-and-gemm-mxfp4-quantize.patch` (commit `575ab53b9f25b2f630410e32258d758ee482bc72`, tree `f85de82ea26ab857867bee09762bcd85a541e7ac`)
- **Causal variable**: Replace the 7-comparison software `_e2m1_code` / `_mxfp4_byte` emulation across `vllm/models/deepseek_v4_1/tilelang/gemm.py` and `vllm/models/deepseek_v4_1/tilelang/indexer.py` with inline PTX `cvt.rn.satfinite.e2m1x2.f32` (`F2FP.SATFINITE.E2M1.F32.PACK_AB`) helpers (`dsv41_e2m1x2`, `dsv41_e2m1x8`, `dsv41_store_e2m1x8`, `dsv41_store_e2m1x32`), vectorizing the per-lane 4-byte stores in `mxfp4_norm_rope_quantize_v41` (`STG.E`) and per-group 16-byte stores in `mxfp4_quantize_v41` (`STG.E.128`) while keeping the `mxfp8_gemm_rope_mxfp4_decode` epilogue loop rolled (`#pragma unroll 1`).
- **Hardware**: `sp8` (NVIDIA GB10 `sm_121a`, CUDA 13.0, `spark-ds41f:1107`)

---

## 1. Motivation & Root Cause

While `vllm/models/deepseek_v4_1/tilelang/cache_writer.py` already lowers nearest-even E2M1 pair packing to the native SM120/SM121 PTX instruction `cvt.rn.satfinite.e2m1x2.f32` (`F2FP.SATFINITE.E2M1.F32.PACK_AB`), three hot MXFP4 quantization paths in `gemm.py` and `indexer.py` still used a 7-comparison software emulation (`_e2m1_code` / `_mxfp4_byte`):

1. **`mxfp8_gemm_rope_mxfp4_decode`** (`vllm/models/deepseek_v4_1/tilelang/gemm.py`): Fused `indexer.wq_b` MXFP8 GEMM + GPT-J interleaved RoPE + MXFP4 quantization epilogue (`N=4096, K=1280`). Each thread computes one pair of FP32 values per epilogue row and writes 1 byte (`Q[m, col0 // 2 + p]`), coalesced across the 32 lanes of a warp.
2. **`mxfp4_quantize_v41`** (`vllm/models/deepseek_v4_1/tilelang/indexer.py`): Used for standalone index-query RoPE + MXFP4 quantization (`[rows, 128]` with `rows = M * 32`) and unrotated paged key writes (`paged=True`). Each thread owns an entire 32-element group (`16` consecutive packed bytes). Because `_mxfp4_byte(vals, 2 * i, scale)` passed `T.fmul(T.cast(vals[...], T.float32), scale)` directly into `_e2m1_code(x)` without binding a local TIR variable, the expression was expanded 8 times per value (7 threshold comparisons + sign bit extraction), causing the generated C++ to recompute `__fmul_rn((float)vals[...], scale)` across every comparison and emit 16 separate 1-byte `STG.E.U8` stores strided by 16 bytes across warp lanes.
3. **`mxfp4_norm_rope_quantize_v41`** (`vllm/models/deepseek_v4_1/tilelang/indexer.py`): Fused split-K partial reduction + RMSNorm + RoPE + paged MXFP4 key writer (`norm_rope_write_index_keys`). Four threads (`group_lane = 0..3`) cooperate on each 32-element group; each thread holds `_VEC = 8` BF16 values and wrote `pairs = 4` consecutive bytes via 4 separate 1-byte `STG.E.U8` instructions.

---

## 2. Implementation

1. **`_E2M1_HEADER` in `vllm/models/deepseek_v4_1/tilelang/gemm.py`**:
   - `dsv41_e2m1x2(float v0, float v1)` (`uint8_t`): packs 2 FP32 values into 1 E2M1x2 byte via `cvt.rn.satfinite.e2m1x2.f32`.
   - `dsv41_e2m1x8(float v0, ..., float v7)` (`uint32_t`): packs 8 FP32 values into a 32-bit word via four `cvt.rn.satfinite.e2m1x2.f32` instructions and `mov.b16` / `mov.b32`.
   - `dsv41_store_e2m1x8(uint8_t* ptr, float v0, ..., float v7)`: stores 4 packed E2M1x2 bytes via a single 32-bit `STG.E` store (`*reinterpret_cast<uint32_t*>(ptr)`).
   - `dsv41_store_e2m1x32(uint8_t* ptr, uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3)`: stores 16 packed E2M1x2 bytes (one full 32-element MXFP4 group) via a single 128-bit `STG.E.128` store (`*reinterpret_cast<uint4*>(ptr) = make_uint4(w0, w1, w2, w3)`).
2. **`mxfp8_gemm_rope_mxfp4_decode` (`gemm.py`)**:
   - Replaces `c_lo = _e2m1_code(sv0); c_hi = _e2m1_code(sv1); Q[...] = c_lo | (c_hi << 4)` with `Q[m, col0 // 2 + p] = _e2m1x2_byte(sv0, sv1)`.
   - Annotates `for it in T.unroll(iters, explicit=False, annotations={"pragma_unroll_factor": 1}):` so `nvcc` keeps the epilogue loop rolled (`#pragma unroll 1`) rather than auto-unrolling `iters = 4` once the loop body shrinks.
3. **`mxfp4_quantize_v41` (`indexer.py`)**:
   - Replaces the 16-iteration `for i in T.serial(half):` 1-byte store loops for both `Pool` (`paged=True`) and `Q` (`paged=False`) with `_mxfp4_store_group(T.address_of(...), vals, scale)` (`dsv41_e2m1x8` x 4 + `dsv41_store_e2m1x32`).
4. **`mxfp4_norm_rope_quantize_v41` (`indexer.py`)**:
   - Replaces the 4-iteration `for i in T.unroll(pairs):` 1-byte store loop with `dsv41_store_e2m1x8(T.address_of(Pool[page, tok * 64 + group * 16 + group_lane * 4]), ...)`.

---

## 3. Static SASS & Register Analysis (`sm_121a`, CUDA 13.0)

Disassembled from `.text.<kernel_name>:` via `nvdisasm` (`receipts/bench-boot1.json`):

| Kernel Variant | Baseline Static SASS Insns (`sw`) | Candidate Static SASS Insns (`hw`) | Static Insn Delta | Registers (`sw` -> `hw`) | Global Store Insns (`sw` -> `hw`) | `F2FP.PACK_AB` Insns (`hw`) |
|---|---:|---:|---:|---:|---:|---:|
| `gemm_mxfp8_rope_mxfp4` (`block_M=16`) | 488 | **440** | **-48 (-9.8%)** | 60 -> 60 | 2 -> 2 (`STG.E.U8`) | 1 / iter |
| `gemm_mxfp8_rope_mxfp4` (`block_M=32`) | 536 | **496** | **-40 (-7.5%)** | 72 -> 72 | 2 -> 2 (`STG.E.U8`) | 1 / iter |
| `gemm_mxfp8_rope_mxfp4` (`block_M=64`) | 656 | **608** | **-48 (-7.3%)** | 110 -> 110 | 2 -> 2 (`STG.E.U8`) | 1 / iter |
| `mxfp4_quantize_v41` (`rope_heads=32, rope_dim=64`) | 864 | **416** | **-448 (-51.9%)** | 69 -> 69 | 7 static / 17 dyn (`16 STG.E.U8 + 1 STG.E.U8`) -> **2 (`1 STG.E.128 + 1 STG.E.U8`)** | 16 |
| `mxfp4_quantize_v41` (`paged=True, page_size=64`) | 656 | **224** | **-432 (-65.9%)** | **52 -> 43 (-9)** | 7 static / 17 dyn (`16 STG.E.U8 + 1 STG.E.U8`) -> **2 (`1 STG.E.128 + 1 STG.E.U8`)** | 16 |
| `mxfp4_norm_rope_quantize_v41` (`shards=1`) | 760 | **376** | **-384 (-50.5%)** | 45 -> 45 | 10 (`8 STG.E.U8 + 2 STG.E.U8`) -> **4 (`2 STG.E + 2 STG.E.U8`)** | 8 |
| `mxfp4_norm_rope_quantize_v41` (`shards=6`) | 880 | **480** | **-400 (-45.5%)** | 87 -> 87 | 10 (`8 STG.E.U8 + 2 STG.E.U8`) -> **4 (`2 STG.E + 2 STG.E.U8`)** | 8 |

---

## 4. Bit-Exactness & Unit Test Verification (2 Boots on `sp8`)

Executed across 2 independent container boots under `flock /tmp/gpu.lock` (`receipts/pytest-boot1.txt`, `receipts/pytest-boot2.txt`):

```bash
flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host --entrypoint pytest \
  -v /tmp/vllm-hw-e2m1x2-v2/vllm/models/deepseek_v4_1/tilelang/gemm.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/gemm.py:ro \
  -v /tmp/vllm-hw-e2m1x2-v2/vllm/models/deepseek_v4_1/tilelang/indexer.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/indexer.py:ro \
  -v /tmp/vllm-hw-e2m1x2-v2/tests/models/test_deepseek_v4_1_lsa.py:/opt/spark-ds41f/candidate/vllm/tests/models/test_deepseek_v4_1_lsa.py:ro \
  -v /tmp/vllm-hw-e2m1x2-v2/tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py:ro \
  -v /usr/local/cuda-13.0/bin/nvdisasm:/usr/local/bin/nvdisasm:ro \
  -w /opt/spark-ds41f/candidate/vllm \
  spark-ds41f:1107 \
  --noconftest -o addopts='' -o cache_dir=/tmp/pytest_cache -v \
  -k 'test_lsa or test_block_scale_words or test_mxfp8_gemm_rope_mxfp4_decode_matches_reference or test_hw_e2m1x2_codegen_in_gemm_and_indexer_quantizers or test_mxfp4_quantize or test_write_index_keys or test_norm_rope_write_index_keys or test_indexer_dense or test_indexer_candidates' \
  tests/models/test_deepseek_v4_1_lsa.py \
  tests/kernels/quantization/test_deepseek_v41_block_scale_words.py \
  tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py \
  tests/kernels/attention/test_deepseek_v41_tilelang.py
```

- **Boot 1 (`receipts/pytest-boot1.txt`)**: `28 passed in 128.82s`, `PDL sources: 55 kernels launched, 55 with PDL, 0 violations`.
- **Boot 2 (`receipts/pytest-boot2.txt`)**: `28 passed in 129.01s`, `PDL sources: 55 kernels launched, 55 with PDL, 0 violations`.
- **Bitwise Equivalence (`bench_hw_e2m1x2_quant.py`)**: `torch.equal` passed (`bit_exact: true`) across all 45 tested `(kernel, M)` combinations on both boots, including exact E2M1 tie midpoints (`0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0`).

---

## 5. Paired 2-Boot Hardware Benchmark Results (`sp8`, NVIDIA GB10)

Measured via interleaved AB/BA rounds (`80` rounds x `25` repeats per round for `M <= 512`, `80` rounds x `5` repeats for `M >= 2048`) in `experiments/2026-11-08-hw-e2m1x2-indexer-gemm-quant/bench_hw_e2m1x2_quant.py` (`receipts/bench-boot1.json`, `receipts/bench-boot2.json`).

### 5.1 `mxfp8_gemm_rope_mxfp4_decode` (`indexer.wq_b`: `N=4096, K=1280, head_dim=128, rope_dim=64`)

| `M` | `block_M` | Boot 1 `sw` (us) | Boot 1 `hw` (us) | Boot 1 Delta (%) | Boot 1 61-Layer Delta (us) | Boot 2 `sw` (us) | Boot 2 `hw` (us) | Boot 2 Delta (%) | Boot 2 61-Layer Delta (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 6.52 | **6.44** | **-1.24%** | -4.92 | 6.35 | **6.23** | **-1.93%** | -7.46 |
| 2 | 16 | 6.07 | 6.21 | +2.17% | +8.04 | 6.10 | **6.10** | **-0.01%** | -0.04 |
| 4 | 16 | 6.13 | **6.07** | **-0.98%** | -3.67 | 6.16 | 6.23 | +1.00% | +3.75 |
| 6 | 16 | 5.82 | 5.96 | +2.32% | +8.23 | 5.82 | **5.78** | **-0.63%** | -2.23 |
| 8 | 16 | 5.92 | **5.83** | **-1.51%** | -5.47 | 5.94 | **5.94** | **-0.12%** | -0.43 |
| 16 | 16 | 7.26 | **7.03** | **-3.11%** | **-13.78** | 7.11 | **6.91** | **-2.80%** | **-12.14** |
| 32 | 32 | 9.71 | **9.35** | **-3.73%** | **-22.10** | 9.65 | **9.29** | **-3.72%** | **-21.86** |
| 64 | 64 | 16.25 | **15.34** | **-5.59%** | **-55.36** | 16.11 | **15.29** | **-5.07%** | **-49.85** |

### 5.2 `mxfp4_quantize_v41` (Query RoPE + MXFP4 Quantize: `rows = M * 32, rope_heads=32, rope_dim=64`)

| `M` | `rows` | Boot 1 `sw` (us) | Boot 1 `hw` (us) | Boot 1 Delta (%) | Boot 1 61-Layer Delta (us) | Boot 2 `sw` (us) | Boot 2 `hw` (us) | Boot 2 Delta (%) | Boot 2 61-Layer Delta (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 5.02 | 5.07 | +1.05% | +3.20 | 4.52 | 4.55 | +0.78% | +2.15 |
| 2 | 64 | 4.99 | 5.02 | +0.62% | +1.87 | 5.11 | **4.92** | **-3.79%** | -11.83 |
| 4 | 128 | 4.97 | 5.02 | +0.89% | +2.70 | 5.06 | **4.92** | **-2.91%** | -8.98 |
| 6 | 192 | 4.98 | 4.98 | +0.14% | +0.43 | 5.06 | **4.98** | **-1.58%** | -4.88 |
| 8 | 256 | 4.97 | **4.96** | **-0.27%** | -0.82 | 5.15 | **5.03** | **-2.30%** | -7.22 |
| 16 | 512 | 5.03 | **4.97** | **-1.26%** | -3.86 | 5.03 | **4.91** | **-2.37%** | -7.26 |
| 32 | 1024 | 4.97 | **4.91** | **-1.27%** | -3.86 | 5.08 | **4.90** | **-3.52%** | -10.89 |
| 512 | 16384 | 19.45 | **12.56** | **-35.41% (1.55x)** | **-420.15** | 19.44 | **12.56** | **-35.37% (1.55x)** | **-419.45** |
| 2048 | 65536 | 80.32 | **59.03** | **-26.51% (1.36x)** | **-1298.86** | 67.02 | **47.35** | **-29.35% (1.42x)** | **-1199.89** |
| 8192 | 262144 | 622.55 | **619.34** | **-0.52%** | **-196.18** | 573.51 | **571.35** | **-0.38%** | **-131.96** |

### 5.3 `mxfp4_quantize_v41` (Paged Key Write: `paged=True, page_size=64`)

| `M` | Boot 1 `sw` (us) | Boot 1 `hw` (us) | Boot 1 Delta (%) | Boot 1 61-Layer Delta (us) | Boot 2 `sw` (us) | Boot 2 `hw` (us) | Boot 2 Delta (%) | Boot 2 61-Layer Delta (us) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.29 | 4.44 | +3.68% | +9.61 | 4.08 | 4.14 | +1.36% | +3.40 |
| 2 | 4.32 | 4.53 | +4.94% | +13.00 | 4.28 | 4.39 | +2.63% | +6.87 |
| 4 | 4.26 | 4.48 | +5.15% | +13.39 | 4.29 | 4.41 | +2.76% | +7.22 |
| 6 | 4.26 | 4.52 | +6.08% | +15.81 | 4.25 | 4.40 | +3.71% | +9.61 |
| 8 | 4.25 | 4.46 | +4.95% | +12.85 | 4.26 | 4.44 | +4.13% | +10.74 |
| 16 | 4.27 | 4.29 | +0.56% | +1.45 | 4.24 | 4.25 | +0.35% | +0.90 |
| 32 | 4.30 | **4.27** | **-0.68%** | -1.79 | 4.71 | **4.24** | **-10.02%** | -28.81 |
| 512 | 4.75 | **4.27** | **-10.02%** | **-29.01** | 4.73 | **4.24** | **-10.31%** | **-29.75** |
| 2048 | 5.36 | 5.64 | +5.38% | +17.57 | 5.55 | **5.55** | **-0.06%** | -0.20 |
| 8192 | 15.58 | **7.80** | **-49.93% (2.00x)** | **-474.53** | 15.61 | **7.78** | **-50.19% (2.01x)** | **-478.04** |

### 5.4 `mxfp4_norm_rope_quantize_v41` (Fused Split-K Reduce + RMSNorm + RoPE + Paged Key Write)

| Variant | `M` | Boot 1 `sw` (us) | Boot 1 `hw` (us) | Boot 1 Delta (%) | Boot 1 61-Layer Delta (us) | Boot 2 `sw` (us) | Boot 2 `hw` (us) | Boot 2 Delta (%) | Boot 2 61-Layer Delta (us) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `shards=1` | 1 | 4.74 | **4.73** | **-0.18%** | -0.51 | 5.42 | 5.44 | +0.26% | +0.86 |
| `shards=1` | 4 | 4.66 | **4.58** | **-1.65%** | -4.68 | 5.55 | **5.55** | **-0.14%** | -0.47 |
| `shards=1` | 8 | 4.71 | 4.79 | +1.64% | +4.72 | 5.52 | 5.53 | +0.17% | +0.59 |
| `shards=1` | 16 | 5.48 | 5.56 | +1.44% | +4.80 | 5.40 | 5.48 | +1.65% | +5.43 |
| `shards=1` | 32 | 4.88 | 5.14 | +5.25% | +15.62 | 5.50 | 5.53 | +0.43% | +1.45 |
| `shards=1` | 512 | 4.60 | **4.58** | **-0.43%** | -1.21 | 5.06 | 5.09 | +0.67% | +2.07 |
| `shards=1` | 2048 | 5.38 | **5.38** | **-0.12%** | -0.39 | 5.40 | 5.42 | +0.41% | +1.37 |
| `shards=1` | 8192 | 6.91 | **5.38** | **-22.06% (1.28x)** | **-92.92** | 6.95 | **5.40** | **-22.41% (1.29x)** | **-95.06** |
| `shards=6` | 1 | 4.63 | 4.63 | +0.03% | +0.08 | 4.74 | 4.75 | +0.27% | +0.78 |
| `shards=6` | 2 | 4.64 | 4.64 | 0.00% | 0.00 | 4.59 | **4.59** | **-0.06%** | -0.16 |
| `shards=6` | 4 | 4.64 | **4.62** | **-0.29%** | -0.82 | 4.58 | **4.57** | **-0.13%** | -0.35 |
| `shards=6` | 6 | 4.63 | **4.61** | **-0.32%** | -0.90 | 4.58 | 4.60 | +0.26% | +0.74 |
| `shards=6` | 8 | 4.62 | **4.61** | **-0.08%** | -0.23 | 4.58 | 4.59 | +0.24% | +0.66 |
| `shards=6` | 16 | 4.64 | 4.64 | +0.10% | +0.27 | 4.58 | 4.59 | +0.22% | +0.62 |
| `shards=6` | 32 | 5.75 | **5.71** | **-0.84%** | -2.96 | 4.56 | **4.56** | **-0.20%** | -0.55 |

---

## 6. Conclusion

Replacing the 7-comparison software `_e2m1_code` / `_mxfp4_byte` emulation with hardware `cvt.rn.satfinite.e2m1x2.f32` (`F2FP.SATFINITE.E2M1.F32.PACK_AB`) and vectorized `uint32_t` / `uint4` global stores across `gemm.py` and `indexer.py`:
- Is **100% bit-exact (`torch.equal`)** across all decode and prefill shapes (`28 passed, 0 PDL violations` on both boots).
- Cuts static SASS instructions by **`-45.5%` to `-65.9%`** across all `indexer.py` quantizer kernels (`864 -> 416` in `mxfp4_quantize_v41` query RoPE, `656 -> 224` and `52 -> 43` registers in `mxfp4_quantize_v41` paged keys, `760 -> 376` in `mxfp4_norm_rope_quantize_v41` `shards=1`, and `880 -> 480` in `shards=6`) and by **`-7.3%` to `-9.8%`** (`-40` to `-48` static SASS instructions) in `mxfp8_gemm_rope_mxfp4_decode`.
- Speeds up `mxfp8_gemm_rope_mxfp4_decode` by **`-2.8%` to `-5.6%`** at `M = 16..64` (`-12` to `-55 us / 61 layers`) and **`-1.2%` to `-1.9%`** at `M = 1`.
- Speeds up prefill `mxfp4_quantize_v41` query RoPE by **`-35.4%` (`1.55x`, `-420 us / 61 layers`)** at `M = 512` and **`-26.5%` to `-29.4%` (`1.36x–1.42x`, `-1.20` to `-1.30 ms / 61 layers`)** at `M = 2048`.
- Speeds up prefill `mxfp4_quantize_v41` paged key writes by **`-49.9%` to `-50.2%` (`2.00x`, `-475 us / 61 layers`)** at `M = 8192` and `mxfp4_norm_rope_quantize_v41` by **`-22.1%` to `-22.4%` (`1.29x`, `-94 us / 61 layers`)** at `M = 8192`.
