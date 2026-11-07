# 2026-11-06 — Warp-Shuffle `indexer.wq_b` Decode Epilogue Fusion

- **Deployment base commit**: `239fad997eac7f42bd9a3f001a6958b29080a2cf`
- **vLLM base commit**: `a75357bba81e31e4f3085700f94b6a0b29aed44a`
- **vLLM candidate commit**: `880d9b843c91d3396affe60c3154d2abf26e919d` (`feat/indexer-wqb-epilogue`)
- **Single intended causal variable**: Fuse GPT-J interleaved RoPE (trailing 64 columns of each 128-dim head) and block-32 MXFP4 E2M1/UE8M0 quantization into the decode GEMM epilogue of `indexer.wq_b` (`mxfp8_gemm_rope_mxfp4_decode`, `block_M=16, block_N=128, block_K=128, num_stages=2, threads=256`), eliminating 42 `quantize_mxfp4` launches per decode step and the `[rows, 32, 128]` BF16 intermediate buffer.

## Why the earlier `c_shared` epilogue (`8a1465c4e` Component 3) was reworked

1. In `8a1465c4e`, the epilogue assigned 1 thread per 32-column scale group (`block_M * (block_N // 32) = 32` active threads out of 128, leaving 3 of 4 warps idle) and allocated a 32-element local array `vals = T.alloc_local((32,), T.bfloat16)` indexed in `T.serial(32)`, which compiled (`nvcc -arch=sm_120a --ptxas-options=-v`) with a **64-byte local-memory stack frame (`STL`/`LDL`)** per thread and uncoalesced 1-byte global writes.
2. In the reworked 16-lane warp-shuffle epilogue (`_mxfp8_gemm_rope_mxfp4_decode`):
   - `C_shared` (`16 x 128` BF16 = 4,096 B) is placed at `buf_dyn_shmem + 0` aliasing `A_shared` after a `T.sync_threads()` barrier (**0 bytes** of additional dynamic shared memory: 38,016 B total, allowing **2 CTAs/SM = 96 CTAs/wave** across the 48 SMs of GB10).
   - Each thread in a 16-lane half-warp (`p = tx % 16`) holds one adjacent column pair `(2*p, 2*p+1)` in scalar registers (`v0, v1`), rotates the pair in registers when `in_head >= 64`, and reduces `amax` across the 16 lanes via four `__shfl_xor_sync` butterfly steps (`delta in (8, 4, 2, 1)`).
   - `nvcc -arch=sm_120a --ptxas-options=-v` confirms **64 registers/thread, 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads**.
   - Epilogue row iterations are guarded by warp-uniform `if m < M:` checks (`pairs_per_row = 64` is a multiple of the 32-thread warp size).

## Exact Verification & Benchmark Commands

Executed across 2 fresh container boots (`spark-ds41f:candidate-38`) on `sp8` (NVIDIA GB10, SM121):

```bash
# Boot 1: Unit tests + PDL source verification + bit-exactness + latency sweep
pytest --noconftest -p conftest_min -q \
  tests/models/test_deepseek_v4_1_attention.py \
  tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py \
  -k "index_query or mxfp8_gemm_rope_mxfp4 or slice_mxfp8" \
  -p tests.kernels.deepseek_v41_pdl_sources
python3 -u bench_wqb_epilogue.py --boot 1 --out boot1.json

# Boot 2: Repeat fresh container boot bit-exactness + latency sweep
python3 -u bench_wqb_epilogue.py --boot 2 --out boot2.json
```

## Bit-Exactness Verification (`boot1.json`, `boot2.json`)

- **100% bit-identical (`torch.equal` on `iq_data`, `iq_scale`, and `weights`)** against the unfused `indexer.wq_b` + `quantize_mxfp4` reference across all 19 tested batch sizes (`rows in {1, 2, 3, 4, 6, 8, 12, 16, 18, 20, 24, 28, 30, 32, 36, 40, 42, 48, 64}`) in both **eager execution** and **captured `torch.cuda.CUDAGraph` replay**, including `ratio=2` compressed positions and NaN/0xFF-poisoned padding rows.
- **PDL chain verification**: `4 passed, 0 violations` in `tests.kernels.deepseek_v41_pdl_sources`.

## 2-Boot Latency Summary (`sp8`, NVIDIA GB10, median us)

| Rows | Standalone Warm UF (B1/B2) | Standalone Warm `#832` (B1/B2) | Standalone Warm `shfl16` (B1/B2) | E2E `_index_query` Warm UF -> Fused (B1 / B2) | E2E `_index_query` Cold 128MiB UF -> Fused (B1 / B2) | E2E `_index_query` CUDAGraph UF -> Fused (B1 / B2) |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 99.94 / 102.46 | 57.66 / 57.15 | **55.71 / 14.94** | 51.52 -> **39.10** / 51.07 -> **39.33** | 42.08 -> **38.91** / 42.08 -> **39.94** | 15.65 -> **9.44** / 15.55 -> **10.72** |
| 2 | 20.70 / 18.62 | 15.68 / 15.68 | **13.76 / 14.69** | 52.00 -> **39.42** / 47.87 -> **37.18** | 42.05 -> **38.94** / 42.98 -> **39.94** | 15.62 -> **10.62** / 15.58 -> **9.50** |
| 3 | 18.27 / 18.62 | 15.58 / 13.47 | **13.44 / 13.70** | 50.56 -> **38.72** / 47.49 -> **37.12** | 42.08 -> **39.87** / 42.18 -> **39.94** | 15.65 -> **10.62** / 15.62 -> **9.47** |
| 4 | 18.18 / 18.59 | 14.78 / 16.58 | **14.75 / 13.66** | 50.37 -> **39.23** / 50.27 -> **39.23** | 42.05 -> **39.78** / 42.24 -> **40.03** | 13.63 -> **9.79** / 13.54 -> **10.69** |
| 6 | 18.14 / 18.53 | 15.68 / 15.71 | **15.10 / 15.14** | 49.82 -> **38.91** / 49.50 -> **38.94** | 42.11 -> **39.94** / 42.98 -> **40.96** | 13.63 -> **10.34** / 13.54 -> **10.50** |
| 8 | 18.14 / 18.56 | 14.62 / 15.97 | **13.76 / 15.30** | 50.30 -> **39.49** / 49.89 -> **39.10** | 43.07 -> **40.90** / 43.07 -> **40.96** | 13.98 -> **10.56** / 13.54 -> **10.50** |
| 12 | 18.59 / 19.97 | 14.14 / 15.07 | **15.68 / 14.37** | 49.47 -> **38.50** / 50.05 -> **39.39** | 43.10 -> **40.96** / 43.10 -> **42.02** | 11.55 -> **11.46** / 11.52 -> **10.14** |
| 16 | 18.27 / 21.12 | 14.50 / 15.74 | **13.60 / 16.16** | 47.23 -> **36.80** / 50.85 -> **39.78** | 44.00 -> **41.98** / 43.04 -> **43.14** | 11.49 -> **11.49** / 11.49 -> **11.52** |
| 18 | 18.08 / 18.69 | 14.59 / 14.75 | **11.55 / 15.42** | 50.18 -> **39.39** / 50.21 -> **39.62** | 45.15 -> **42.98** / 45.06 -> **44.03** | 16.22 -> **13.57** / 15.58 -> **15.62** |
| 20 | 18.21 / 21.09 | 11.55 / 15.71 | **14.56 / 14.05** | 47.68 -> **37.41** / 50.34 -> **39.97** | 46.08 -> **43.01** / 46.08 -> **44.99** | 15.58 -> **13.54** / 15.58 -> **15.52** |
| 24 | 18.21 / 18.46 | 14.59 / 14.75 | **14.59 / 11.90** | 51.39 -> **38.30** / 50.11 -> **40.16** | 46.08 -> **43.01** / 47.04 -> **45.89** | 15.52 -> **15.65** / 15.20 -> **13.66** |
| 28 | 18.34 / 18.40 | 14.62 / 14.53 | **14.59 / 14.62** | 50.30 -> **39.97** / 47.78 -> **37.34** | 47.04 -> **43.90** / 47.10 -> **46.88** | 15.10 -> **16.00** / 15.52 -> **16.10** |
| 30 | 18.11 / 18.43 | 14.59 / 14.56 | **13.60 / 14.59** | 50.37 -> **39.81** / 48.03 -> **37.28** | 47.23 -> **44.03** / 48.03 -> **46.08** | 16.16 -> **15.71** / 15.20 -> **15.55** |
| 32 | 18.11 / 18.40 | 14.78 / 11.52 | **13.57 / 14.62** | 49.98 -> **39.74** / 48.06 -> **37.63** | 46.18 -> **45.06** / 47.97 -> **46.08** | 15.30 -> **15.55** / 15.39 -> **15.55** |
| 36 | 21.82 / 20.32 | 21.76 / 21.86 | **16.67 / 15.78** | 50.21 -> **39.33** / 47.78 -> **37.25** | 47.20 -> **45.12** / 47.23 -> **47.04** | 21.82 -> **17.63** / 21.70 -> **17.63** |
| 40 | 21.73 / 21.76 | 23.42 / 21.73 | **15.62 / 15.62** | 50.18 -> **39.52** / 48.90 -> **38.66** | 47.10 -> **46.08** / 47.23 -> **47.10** | 21.89 -> **17.70** / 21.70 -> **17.60** |
| 42 | 21.76 / 21.73 | 21.02 / 19.71 | **15.58 / 15.58** | 50.62 -> **39.46** / 47.78 -> **37.31** | 48.10 -> **46.08** / 47.20 -> **47.01** | 20.77 -> **17.60** / 19.71 -> **17.63** |
| 48 | 20.26 / 19.52 | 19.71 / 19.68 | **15.65 / 15.58** | 50.75 -> **39.62** / 47.90 -> **37.44** | 47.23 -> **46.08** / 48.16 -> **47.10** | 20.13 -> **19.36** / 19.68 -> **17.63** |

## Conclusion

Reworking `#832` Component 3 into the 16-lane warp-shuffle epilogue (`_mxfp8_gemm_rope_mxfp4_decode`, `16x128`, `num_stages=2`, `threads=256`, `C_shared` aliasing `A_shared` after `T.sync_threads()`) eliminates 42 `quantize_mxfp4` launches and the `[rows, 32, 128]` BF16 intermediate buffer per decode step, uses 0 bytes of stack frame and 0 bytes of additional dynamic shared memory, compiles only 1 kernel per layer instead of 3, and achieves 100% bit-identity (`torch.equal`) with consistent latency reductions across warm L2, cold 128 MiB L2 wash, and CUDA graph replay across both container boots.
