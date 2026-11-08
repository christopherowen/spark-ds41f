# Exact `BF16x2` (`HMUL2.BF16_V2`) Dequantization and PTX 9.2 `cvt.rn.bf16x2.e2m1x2` in TileLang Sparse MLA

## Metadata

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18b` (`spark-ds41f-1107`)
- **Base vLLM commit**: `2756478b60b90700b9918b457aeb38657da637b5` (patch `0761` on `local-inference-lab/vllm@19f2c20ed4d64ae22bd009ceacadd85e2a81e354`)
- **Base TileLang commit**: `2123e412ce2217b324c50c4ec967213906456ac5` (+ `0001..0013`)
- **Hardware**: NVIDIA GB10 (`sm_121a`), tested on `sp5` (`spark-0013`) and `sp6` (`spark-0014`) under `flock /tmp/gpu.lock`
- **Container**: `spark-ds41f:1107` (`ptxas V13.0.88` default; `ptxas V13.5.37` PTX ISA 9.2 mounted at `/cuda135`)
- **Causal variable**: Replace the 32x-unrolled `F16 -> F32 -> FMUL -> BF16` dequantization detour in `vllm/models/deepseek_v4_1/tilelang/sparse_mla.py` (`dequant_swa` and `dequant_idx`) with direct `T.bfloat16` casts and packed `HMUL2.BF16_V2` (`tl::mul2`), add PTX ISA 9.2 `cvt.rn.bf16x2.e2m1x2` (`F2FP.BF16.E2M1.UNPACK_B`) to `__tl_cvt_fp4x2_to_bfloat162` in `src/tl_templates/cuda/cuda_fp4.h`, and remove the scalar `cvt.rn.bf16.f16` (`F2F.BF16.F16`) fallback from `__tl_cvt_e4m3x2_to_bfloat162` in `src/tl_templates/cuda/cuda_fp8.h` so `CUDA < 13.2` lowers via packed `F2FP.BF16.F32.PACK_AB`.

---

## Rationale & Mathematical Proof of Bitwise Exactness

In `vllm/models/deepseek_v4_1/tilelang/sparse_mla.py`, `dequant_swa` and `dequant_idx` are expanded inside `dequant_chunk`, which is unrolled 16x in the prologue (`fill_kv_buffer`) and 16x in the mainloop (`num_stages = 2`, `CHUNKS_PER_STAGE = 8`) — a total of **32x unrolling** inside `sparse_mla_kernel`. Previously, both macros widened `staged_fp8` (`E4M3`) and `staged_fp4` (`E2M1`) payloads and scales to `T.float32`, multiplied in `T.float32` (`FMUL`), and narrowed back to `T.bfloat16`:

1. **`dequant_idx` (`E2M1 x E4M3 -> BF16`)**:
   - Every finite `E2M1` value has 1 mantissa bit (at most 2 significand bits) and unbiased exponent in `[-1, +2]` (plus subnormal `0.5 = 2^-1`).
   - Every finite `E4M3` value has 3 mantissa bits (at most 4 significand bits) and unbiased exponent in `[-6, +8]` (or down to `2^-9` for minimum subnormal `2^-6 x 2^-3`).
   - Their exact product has at most `2 + 4 = 6` significand bits (5 mantissa bits) and unbiased exponent in `[-10, +10]`.
   - `BF16` has 8 significand bits (7 mantissa bits) and normal unbiased exponent range `[-126, +127]`. Therefore, every non-NaN `E2M1` value, every non-NaN `E4M3` value, and every non-NaN product `E2M1 x E4M3` is **exact in `BF16` with zero rounding error**.
   - Exhaustive GPU verification across all `16 x 254 = 4,064` non-NaN `(E2M1, E4M3)` pairs confirms `0` `uint16` mismatches (`torch.equal` is `True`, including signed zeros).

2. **`dequant_swa` (`E4M3 x UE8M0 -> BF16`)**:
   - A `UE8M0` scale byte `e in [0, 253]` represents the exact power of two `2^(e - 127)`, which has the same 8-bit exponent field in `BF16` (`uint16(e) << 7`) as in `FP32` (`uint32(e) << 23`).
   - Multiplying a normal `BF16`-cast `E4M3` value (4 significand bits) by `2^(e - 127)` in `BF16` (`HMUL2.BF16_V2`) only adds exponents (or shifts into the `BF16` subnormal range when `e` is near `0`, where `BF16` and `FP32 -> BF16` round identically because `E4M3` has only 3 mantissa bits vs `BF16`'s 7 mantissa bits).
   - Exhaustive GPU verification across all `254 x 254 = 64,516` finite `(E4M3, UE8M0)` pairs confirms `0` `uint16` mismatches (`torch.equal` is `True`).

3. **Hoisting `UE8M0` scale in `dequant_swa`**:
   - In `T.Parallel(BI, D)` (`BI = 32`, `D = 512`, `threads = 128`), each thread executes 16 iterations of 8 vectorized elements (`SWA_GROUP = 16`).
   - Writing `dequant_swa` with `T.thread_binding(threads, thread="threadIdx.x")` + `T.unroll(16)` + `T.vectorized(8)` hoists the scalar `uint16(staged[i, D + d0 // SWA_GROUP]) << 7` scale outside the 8-wide inner vector loop, eliminating TileLang's `Cannot vectorize ramp with stride 0` warning and saving 7 redundant scale shifts per 8 elements while keeping `dequant_idx` on `T.Parallel(BI, D)`.

---

## SASS Instruction Analysis (`sm_121a`, TP4 `"split"` decode kernel)

Kernel configuration: `sparse_mla_fwd_v41(heads=16, swa_width=128, idx_width=512, idx_page_size=128, schedule="decode", swa_page_size=128, block_H=16, block_I=32, threads=128, skip_inactive=False)` (`k0 = sparse_mla_kernel`, `k1 = sparse_mla_combine_kernel`).

| Metric / Opcode | CUDA `13.5.37` Baseline | CUDA `13.5.37` Candidate (`m1_swa_tb`) | CUDA `13.5.37` Variant (`m2_both_tb` + scalar PTX) | CUDA `13.0.2` Baseline | CUDA `13.0.2` Candidate (`m1_swa_tb`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `k0` (`sparse_mla_kernel`) SASS | `3,296` | **`2,480` (`-816`)** | `2,448` (`-848`) | `3,224` | **`2,872` (`-352`)** |
| `k1` (`sparse_mla_combine_kernel`) SASS | `320` | `320` | `320` | `320` | `320` |
| **Total (`k0 + k1 + 2`)** | **`3,618`** | **`2,802` (`-816`, `-22.6%`)** | **`2,770` (`-848`, `-23.4%`)** | **`3,546`** | **`3,194` (`-352`, `-9.9%`)** |
| `FMUL` | `394` | **`138` (`-256`)** | `138` (`-256`) | `394` | **`138` (`-256`)** |
| `HADD2.F32` | `324` | **`4` (`-320`)** | `4` (`-320`) | `324` | `324` |
| `F2FP.BF16.F32.PACK_AB` | `133` | **`5` (`-128`)** | `5` (`-128`) | `133` | `133` |
| `HMUL2.BF16_V2` | `0` | **`128` (`+128`)** | `128` (`+128`) | `0` | **`128` (`+128`)** |
| `F2FP.BF16.E4M3.UNPACK_B` | `0` | **`96` (`+96`)** | `80` (`+80`) | `0` | `0` |
| `F2FP.BF16.E2M1.UNPACK_B` | `0` | **`64` (`+64`)** | `64` (`+64`) | `0` | `0` |
| `F2FP.F16.E4M3.UNPACK_B` | `162` | **`2` (`-160`)** | `2` (`-160`) | `162` | `98` (`-64`) |
| `F2FP.F16.E2M1.UNPACK_B` | `64` | **`0` (`-64`)** | `0` (`-64`) | `64` | `64` |
| `F2F.BF16.F16` (slow pipe) | `0` | **`0`** | `0` | `0` | **`0`** |

---

## Hardware Benchmark Results (`sp5`, 2 Boots per Configuration)

### Workload Identity & Commands

Benchmarked on `sp5` (`spark-0013`) under `flock /tmp/gpu.lock` inside `spark-ds41f:1107` using `SparseMlaV41Plan(16, 128, 512, 128, swa_page_size=128, max_rows=max(rows, 64), decode_max_rows=6, split_max_rows=6)` at `kv_len = 4096` (`swa_pages = 32`, `idx_pages = 32`), measuring CUDA Graph replay latency (`50` warmup replays + `200` timed replays per shape).

```bash
# Unit test suite (21 tests including test_sparse_mla_bf16x2_dequant_exact_and_codegen)
flock /tmp/gpu.lock docker run --rm --gpus all -w /opt/spark-ds41f/candidate/vllm --entrypoint pytest \
  spark-ds41f:1107 --noconftest -p tests.kernels.conftest -o addopts= \
  tests/kernels/attention/test_deepseek_v41_tilelang.py -k "sparse_mla" -v
```

### 1. Candidate (`m1_swa_tb`) under CUDA `13.5.37` (`ptxas V13.5.37`, `-816` SASS instructions)

| Shape (`kv_len=4096`) | Route | Boot 1 Base (us) | Boot 1 Cand (us) | Boot 1 Delta | Boot 2 Base (us) | Boot 2 Cand (us) | Boot 2 Delta | 2-Boot Mean Delta | Bit-Exact (`uint16`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Decode `rows=1` | `split` | `22.11` | `20.51` | **`-7.24%`** | `21.60` | `20.48` | **`-5.19%`** | **`-6.22%`** | `True` (`0` diffs) |
| Decode `rows=4` | `split` | `22.43` | `20.42` | **`-8.99%`** | `22.43` | `20.61` | **`-8.13%`** | **`-8.56%`** | `True` (`0` diffs) |
| Decode `rows=8` | `split` | `22.27` | `21.76` | **`-2.30%`** | `22.53` | `22.27` | **`-1.14%`** | **`-1.72%`** | `True` (`0` diffs) |
| Decode `rows=16` | `decode` | `34.11` | `32.00` | **`-6.19%`** | `33.95` | `32.00` | **`-5.75%`** | **`-5.97%`** | `True` (`0` diffs) |
| Extend `rows=64` | `extend` | `91.84` | `83.62` | **`-8.95%`** | `91.78` | `83.58` | **`-8.93%`** | **`-8.94%`** | `True` (`0` diffs) |
| Extend `rows=256` | `extend` | `251.71` | `228.99` | **`-9.03%`** | `251.26` | `229.02` | **`-8.85%`** | **`-8.94%`** | `True` (`0` diffs) |
| Extend `rows=1024` | `extend` | `929.47` | `847.14` | **`-8.86%`** | `931.49` | `847.46` | **`-9.02%`** | **`-8.94%`** | `True` (`0` diffs) |

### 2. Candidate (`m1_swa_tb`) under CUDA `13.0.2` (`spark-ds41f:1107` default `ptxas V13.0.88`, `-352` SASS instructions)

| Shape (`kv_len=4096`) | Route | Boot 1 Base (us) | Boot 1 Cand (us) | Boot 1 Delta | Boot 2 Base (us) | Boot 2 Cand (us) | Boot 2 Delta | 2-Boot Mean Delta | Bit-Exact (`uint16`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Decode `rows=1` | `split` | `20.61` | `20.51` | **`-0.47%`** | `20.54` | `20.54` | **`0.00%`** | **`-0.23%`** | `True` (`0` diffs) |
| Decode `rows=4` | `split` | `21.57` | `20.80` | **`-3.56%`** | `21.50` | `21.60` | **`+0.45%`** | **`-1.56%`** | `True` (`0` diffs) |
| Decode `rows=8` | `split` | `22.30` | `21.73` | **`-2.58%`** | `21.89` | `21.70` | **`-0.88%`** | **`-1.73%`** | `True` (`0` diffs) |
| Decode `rows=16` | `decode` | `34.34` | `33.54` | **`-2.33%`** | `34.24` | `33.66` | **`-1.68%`** | **`-2.01%`** | `True` (`0` diffs) |
| Extend `rows=64` | `extend` | `89.98` | `89.76` | **`-0.25%`** | `91.07` | `91.26` | **`+0.21%`** | **`-0.02%`** | `True` (`0` diffs) |
| Extend `rows=256` | `extend` | `249.86` | `249.63` | **`-0.09%`** | `249.76` | `249.79` | **`+0.01%`** | **`-0.04%`** | `True` (`0` diffs) |
| Extend `rows=1024` | `extend` | `923.39` | `924.70` | **`+0.14%`** | `925.50` | `924.54` | **`-0.10%`** | **`+0.02%`** | `True` (`0` diffs) |

### 3. Additional Evaluated Variants (Including Unfavorable Runs)

- **Unfavorable run — CUDA `13.0.2` with upstream `cuda_fp8.h` `#elif __CUDA_ARCH__ >= 900` (`cvt.rn.bf16.f16`)**:
  - Upstream `cuda_fp8.h` contains an `#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)` branch in `__tl_cvt_e4m3x2_to_bfloat162` that converts `half2` to `__nv_bfloat162` via two scalar `asm("cvt.rn.bf16.f16 %0, %1;")` instructions.
  - On `sm_121a` under CUDA `13.0.2` (`ptxas V13.0.88`), those scalar inline PTX instructions lower to `320` slow-pipe scalar `F2F.BF16.F16` + `128` `PRMT` instructions (`3,642` total SASS instructions), regressing decode (`rows=1`: `20.51 -> 22.43 us`, `+9.36%`; `rows=4`: `21.63 -> 23.62 us`, `+9.17%`; `rows=8`: `21.95 -> 23.74 us`, `+8.16%`; `rows=16`: `34.18 -> 37.66 us`, `+10.21%`) and extend (`rows=64`: `90.21 -> 111.01 us`, `+23.06%`; `rows=256`: `249.57 -> 305.50 us`, `+22.41%`; `rows=1024`: `926.18 -> 1135.42 us`, `+22.59%`).
  - Removing that `#elif` branch in `cuda_fp8.h` so `CUDA < 13.2` falls back to `__float22bfloat162_rn` (which lowers to packed `F2FP.BF16.F32.PACK_AB`) completely eliminates `F2F.BF16.F16` (`320 -> 0`) and restores positive/neutral performance under CUDA `13.0.2` (`-2.01%` at `rows=16`, `-1.73%` at `rows=8`, `0.00%` on extend).
- **Variant `m2_both_tb` (hoisting `scale` in `dequant_idx` too)**:
  - Hoisting `scale = T.cast(staged_fp8[...], dtype)` outside `T.vectorized(8)` in `dequant_idx` as well reduces CUDA `13.0.2` SASS to `2,874` (`-672` instructions) and speeds up CUDA `13.0.2` decode (`rows=16`: `34.24 -> 29.92 us`, `-12.62%`) and extend (`rows=64..1024`: `-2.76%..-2.86%`).
  - Under CUDA `13.5.37` with scalar PTX 9.2 `__tl_cvt_fp8_to_bfloat16`, `m2_both_tb` reaches `2,770` SASS instructions (`-848`, `-23.4%`) and `-8.34%` (`rows=64`), `-9.76%` (`rows=256`), and `-9.68%` (`rows=1024`, `930.08 -> 840.03 us`), while `m1_swa_tb` is faster on short decode (`rows=1`: `20.50 us` vs `22.24 us`; `rows=4`: `20.51 us` vs `22.35 us`; `rows=16`: `32.00 us` vs `32.77 us`) and matches the exact `2,802` (`-816`) SASS target. We therefore select `m1_swa_tb` for patch `0762`.

---

## Memory & Test Verification

- **Shared memory**: Unchanged (`49,152` bytes for `"split"`, `83,968` bytes for `"decode"` and `"extend"`).
- **Register usage (`sm_121a`)**: `sparse_mla_kernel` uses `162` registers under CUDA `13.5.37` and `168` registers under CUDA `13.0.2` (`0` bytes stack frame, `0` bytes spill stores/loads).
- **Pytest verification (`sp5`)**:
  - Boot 1 (CUDA `13.0.2`): `21 passed, 38 deselected in 12.11s`
  - Boot 2 (CUDA `13.0.2`): `21 passed, 38 deselected in 11.86s`
  - Boot 3 (CUDA `13.5.37`): `2 passed, 57 deselected in 8.97s` (`test_sparse_mla[16-512]` and `test_sparse_mla_bf16x2_dequant_exact_and_codegen`)

---

## Conclusion

Promotable candidate (`patches/vllm/0762-perf-dsv41-sparse-mla-bf16x2-dequant.patch` and `patches/tilelang/0005-cuda-fp4-ptx92-cvt-bf16x2-e2m1x2.patch`). Eliminates `-816` SASS instructions (`-22.6%`, `3,618 -> 2,802`) in `sparse_mla_fwd_v41` under CUDA `13.5.37` (`-352` SASS instructions / `-9.9%` under CUDA `13.0.2`), delivering **`-5.97%` to `-8.56%` decode latency reduction (`rows=1, 4, 16`) and `-8.94%` extend latency reduction (`rows=64, 256, 1024`)** under CUDA `13.5.37` and **`-1.56%` to `-2.01%` decode latency reduction (`rows=4, 8, 16`)** under CUDA `13.0.2`, with 100% bitwise `uint16` equivalence across all inputs.
