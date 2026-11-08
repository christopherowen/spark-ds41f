# 2026-11-08: `mHC: fn` 24-Bit Weight Storage and CUDA Green Context L2-Prefetch Isolation Audit

## Base

- **Repository commit:** `8e5ea208f3d22bc69df91ace94d6975932c858a8` (`spark-ds41f-1107`, vLLM series `experiments/2026-10-26-spark-ds41f-candidate/1107/vllm/series` at `0761`)
- **Container image:** `spark-ds41f:1107` (`3f32a962ce04`)
- **Hardware:** `sp4` and `sp7` (`NVIDIA GB10`, `sm_121a`, 48 SMs / 24 TPCs, 24 MiB L2)

---

## Part 1: `mHC: fn` 24-Bit (`top-3-byte`) Weight Storage (`TODO.md` lines 367–371)

### Question and Hypothesis

`TODO.md` listed a candidate investigation under `TileLang kernel family`:
> **`mHC: fn` in three bytes.** `project_streams` reads `fn` (`24 x 20,480` FP32, 2 MB per sublayer) near memory bandwidth, and a TF32 MMA consumes only the top 19 bits of each value. Storing the top 24 bits would cut that traffic by 25% with bit-identical results if SM121's TF32 MMA truncates rather than rounds its inputs; verify on hardware first.

### Findings (`sp4`, 2 Container Boots)

1. **SM121 TF32 MMA Hardware Truncation Probe (`PASS`)**:
   - On `GB10` (`sm_121a`), `cute::SM80_16x8x8_F32TF32TF32F32_TN` (`tl::mma_sync<tl::DataType::kTensorFloat32, ...>`, PTX `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`) **truncates** rather than rounds its FP32 inputs:

   | Input pattern | Hex bits | SM121 TF32 MMA output `C[0,0]` | Output hex |
   | --- | --- | ---: | --- |
   | `exact_1.0` | `0x3F800000` | `1.0000000000` | `0x3F800000` |
   | `low8_only` | `0x3F8000FF` | `1.0000000000` | `0x3F800000` |
   | `just_below_half_tf32_ulp` | `0x3F800FFF` | `1.0000000000` | `0x3F800000` |
   | `exact_half_tf32_ulp` | `0x3F801000` | `1.0000000000` | `0x3F800000` |
   | `above_half_tf32_ulp` | `0x3F801001` | `1.0000000000` | `0x3F800000` |
   | `max_low13` | `0x3F801FFF` | `1.0000000000` | `0x3F800000` |

2. **Why `0092` Supersedes 24-Bit Truncation**:
   - The `TODO.md` note was written for the original Oct 4 (`0037`) `project_streams` kernel, which issued a single TF32 `T.gemm` per stream.
   - Since Oct 11 (`0092-TileLang-V4.1-mHC-the-stream-projection-at-FP32-prec.patch` and `0115`), `load_split_tf32` in `vllm/models/deepseek_v4_1/tilelang/mhc.py` splits each FP32 `fn` element into **three exact TF32 parts** (`high = f & 0xFFFFE000`, `middle = (f - high) & 0xFFFFE000`, `low = (f - high) - middle`, carrying `11 + 11 + 2 = 24` significant bits) and accumulates three TF32 MMAs (`F_shared`, `F_mid`, `F_low`), consuming **all 32 IEEE-754 bits** of `fn`.
   - Across all `86` `hc_*_fn` tensors in the `v41-flash-1017` checkpoint (`42,271,881` `torch.float32` elements), **`42,106,762 / 42,271,881` (`99.61%`) have non-zero low 8 bits (`bits & 0xFF != 0`)**, `42,266,632 / 42,271,881` (`99.99%`) have non-zero low 13 bits (`bits & 0x1FFF != 0`), every mantissa bit `0..21` has `~0.500` ones density (`0.4997..0.5006`), and `layers.1.hc_ffn_fn` spans `32` exponents (`91..129`).
   - Consequently, 24-bit `fn` (`trunc24` or RNE `round24`) alters the low 6 bits of `F_mid` and zeroes `F_low`, failing `torch.equal` on `post`, `comb`, and `pre` at every `M = 1..128`:

   | `M` | `trunc24` `torch.equal(post, comb, y, pre)` | `trunc24` max diff | `round24` `torch.equal(post, comb, y, pre)` | `round24` max diff |
   | ---: | --- | ---: | --- | ---: |
   | `1` | `[False, False, True, False]` | `8.345e-07` | `[False, False, True, False]` | `6.706e-07` |
   | `2` | `[False, False, True, False]` | `1.907e-06` | `[False, False, True, False]` | `1.073e-06` |
   | `4` | `[False, False, True, False]` | `1.818e-06` | `[False, False, True, False]` | `7.153e-07` |
   | `6` | `[False, False, True, False]` | `1.341e-06` | `[False, False, True, False]` | `1.550e-06` |
   | `8` | `[False, False, True, False]` | `2.503e-06` | `[False, False, True, False]` | `9.537e-07` |
   | `16` | `[False, False, True, False]` | `2.474e-06` | `[False, False, True, False]` | `1.132e-06` |
   | `32` | `[False, False, True, False]` | `6.378e-06` | `[False, False, True, False]` | `1.550e-06` |
   | `64` | `[False, False, True, False]` | `4.649e-06` | `[False, False, True, False]` | `1.401e-06` |
   | `128` | `[False, False, True, False]` | `4.947e-06` | `[False, False, True, False]` | `1.848e-06` |

3. **2-Boot `mhc_project_streams` (`post_pre` mode) CUDA-Graph Benchmark on `sp4`**:
   - In `vllm/models/deepseek_v4_1/nvidia/model.py` (lines 256–278), `hc_ffn_fn` and `hc_attn_fn` are the first segments prefetched into GB10's 24 MiB L2 cache by `VLLM_DS41_L2_PREFETCH=1` (`BUDGET_FFN` and `BUDGET_NEXT`), so `fn` is L2-hot during decode.
   - Replacing 128-bit `float4` loads with 24-bit `Fn_hi: uint16` + `Fn_lo: uint8` unpacking (`fp24_3g`) regresses L2-hot latency by **`+2.4%..+6.1%`** across `M = 1..64`. Dropping the third GEMM (`fp24_2g`, which is bit-identical to `fp24_3g` on 24-bit inputs since `rest = f - high` has at most 5 significant bits `12..8`) is faster (`-13.0%..-35.9%`) but is not bit-exact against 32-bit `fn`:

   | `M` | L2-Hot `base_fp32_3g` (µs, Boot 1 / Boot 2) | L2-Hot `fp24_3g` (µs, Boot 1 / Boot 2) | L2-Hot `fp24_3g` vs `base` | L2-Hot `fp24_2g` (µs, Boot 1 / Boot 2) | L2-Cold `base_fp32_3g` (µs, Boot 1 / Boot 2) | L2-Cold `fp24_3g` (µs, Boot 1 / Boot 2) | L2-Cold `fp24_2g` (µs, Boot 1 / Boot 2) |
   | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
   | `1` | `5.24 / 5.43` (`5.33`) | `5.42 / 5.54` (`5.48`) | **`+2.8%`** | `4.50 / 4.73` (`4.62`) | `11.56 / 11.83` (`11.70`) | `10.67 / 10.46` (`10.56`) | `9.84 / 9.66` (`9.75`) |
   | `2` | `5.38 / 5.30` (`5.34`) | `5.41 / 5.58` (`5.50`) | **`+3.0%`** | `4.69 / 4.53` (`4.61`) | `11.56 / 11.79` (`11.68`) | `10.87 / 10.47` (`10.67`) | `10.08 / 9.62` (`9.85`) |
   | `4` | `5.38 / 5.32` (`5.35`) | `5.58 / 5.53` (`5.55`) | **`+3.8%`** | `4.57 / 4.57` (`4.57`) | `11.60 / 11.60` (`11.60`) | `10.64 / 10.77` (`10.70`) | `9.73 / 9.50` (`9.61`) |
   | `6` | `5.36 / 5.56` (`5.46`) | `5.61 / 5.57` (`5.59`) | **`+2.4%`** | `4.56 / 4.77` (`4.67`) | `11.78 / 11.88` (`11.83`) | `10.68 / 10.82` (`10.75`) | `9.75 / 9.61` (`9.68`) |
   | `8` | `5.61 / 5.48` (`5.55`) | `5.76 / 5.94` (`5.85`) | **`+5.5%`** | `4.79 / 4.73` (`4.76`) | `11.59 / 11.74` (`11.66`) | `10.94 / 10.59` (`10.77`) | `9.77 / 9.58` (`9.67`) |
   | `16` | `6.18 / 6.20` (`6.19`) | `6.45 / 6.46` (`6.45`) | **`+4.3%`** | `5.35 / 5.43` (`5.39`) | `12.04 / 11.79` (`11.92`) | `11.17 / 10.79` (`10.98`) | `10.03 / 10.14` (`10.09`) |
   | `32` | `11.14 / 11.15` (`11.14`) | `11.81 / 11.84` (`11.83`) | **`+6.1%`** | `7.67 / 7.73` (`7.70`) | `16.22 / 16.24` (`16.23`) | `15.66 / 15.58` (`15.62`) | `10.43 / 10.34` (`10.38`) |
   | `64` | `21.34 / 21.15` (`21.25`) | `22.30 / 22.44` (`22.37`) | **`+5.3%`** | `13.78 / 13.44` (`13.61`) | `25.54 / 25.65` (`25.60`) | `25.34 / 25.31` (`25.33`) | `16.62 / 16.67` (`16.65`) |

---

## Part 2: CUDA Green Context (`CUgreenCtx`) SM Isolation for Side-Stream L2 Prefetch (`sp7`, 2 Container Boots)

### Question and Hypothesis

Evaluate whether partitioning `GB10`'s 48 SMs (24 TPCs) via CUDA Driver API Green Contexts (`cuDevSmResourceSplitByCount` + `cuGreenCtxCreate` + `cuGreenCtxStreamCreate` wrapped in `torch.cuda.ExternalStream`) to isolate `L2Prefetcher`'s side stream onto `2 SMs` (`1 TPC`, `CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING`) reduces main-stream compute interference during decode.

### Findings (`sp7`, 2 Container Boots)

1. **Functional & `%smid` Verification (`PASS`)**:
   - `cuDevSmResourceSplitByCount` with `CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING (0x1)` cleanly splits `GB10`'s 48 SMs into `2 SMs` (`[0, 1]`) and `46 SMs` (`[2..47]`), and works under `torch.cuda.CUDAGraph()` fork-join capture and replay.
   - When the `2-SM` Green Context stream is active, `l2_prefetch_kernel` executes strictly on `SMs [0, 1]`; when idle, a 48-CTA kernel on the default stream uses all `48 SMs [0..47]`.

2. **Why `CUgreenCtx` Isolation Regresses `L2Prefetcher` on `GB10`**:
   - **Cross-context `CUDAGraph` event-fork barrier (`+1.70 µs/fork` = `+5.09 µs/layer` = `+204 µs/step`)**: `L2Prefetcher.issue()` forks `self.side.wait_stream(main)` 3 times per decoder layer (`120` forks per 40-layer step). Forking within the same primary `CUcontext` (`normal_side`) costs `+0.155 µs/fork`, whereas crossing into a `CUgreenCtx` stream inside `CUDAGraph` replay costs **`+1.696 µs/fork` over `normal_side`** (`+5.09 µs/layer`).
   - **Per-SM TMA/MIO bulk-prefetch ceiling (`~145 GB/s/SM`) and tiny kernel footprint**: `l2_prefetch_kernel` uses only `4 warps` (128 threads), `22 registers`, and `0 B` shared memory, co-residing in spare warp slots at `VLLM_L2_PREFETCH_GRID=2` without evicting main-stream CTAs.

### 1. 120-Window (40 Layers × 3 Windows) `CUDAGraph` Fork-Join Overhead (`sp7`, 2 Boots)

| Mode | Boot 1 Total (µs) | Boot 2 Total (µs) | Per-Window (µs) | Delta vs `normal_side` (µs/window) | Per 3-Window Layer Overhead (µs/layer) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `no_fork` (main stream only) | `82.02` | `80.08` | `0.675` | `-0.167` | `-0.50` |
| `normal_side` (`torch.cuda.Stream()`) | **`100.53`** | **`101.56`** | **`0.842`** | **`0.000`** | **`0.00`** |
| `green_2sm` (`2 SMs`, `IGNORE_SM_COSCHEDULING`) | `305.91` | `303.19` | `2.538` | **`+1.696`** | **`+5.09`** |
| `green_4sm` (`4 SMs`, `IGNORE_SM_COSCHEDULING`) | `316.50` | `312.66` | `2.622` | `+1.780` | `+5.34` |
| `green_8sm_cosched` (`8 SMs`, coscheduled) | `312.07` | `315.65` | `2.616` | `+1.774` | `+5.32` |
| `green_split_2_46` (`46-SM` main + `2-SM` side) | `305.13` | `311.55` | `2.569` | `+1.727` | `+5.18` |

### 2. Full 3-Window Layer Overlap Benchmark at Promoted `VLLM_L2_PREFETCH_GRID=2` (`sp7`, 2 Boots, µs/layer)

| `M` | `no_prefetch` (µs) | `baseline_default` (µs) | `green_2sm` (µs) | `green_2sm` vs `baseline` | `green_4sm` (µs) | `green_8sm_cosched` (µs) | `green_split_2_46` (µs) |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `360.06 / 362.94` | **`229.05 / 231.65`** | `234.40 / 234.91` | **`+2.33% / +1.41%`** | `236.06 / 236.39` | `236.43 / 236.87` | `233.69 / 235.22` |
| `2` | `361.78 / 363.61` | **`229.68 / 230.59`** | `235.00 / 236.35` | **`+2.32% / +2.50%`** | `236.59 / 240.44` | `236.82 / 237.42` | `235.08 / 235.56` |
| `4` | `363.26 / 362.08` | **`230.37 / 229.68`** | `236.91 / 235.67` | **`+2.84% / +2.61%`** | `237.97 / 237.09` | `237.33 / 236.82` | `236.42 / 235.69` |
| `6` | `361.96 / 361.50` | **`229.14 / 229.52`** | `236.45 / 236.42` | **`+3.19% / +3.00%`** | `235.05 / 235.17` | `237.04 / 237.00` | `236.70 / 237.22` |
| `8` | `361.50 / 361.79` | **`230.08 / 230.03`** | `237.40 / 237.91` | **`+3.18% / +3.43%`** | `237.10 / 237.47` | `238.58 / 238.07` | `238.10 / 242.02` |
| `16` | `362.78 / 362.38` | **`230.30 / 230.67`** | `237.80 / 238.80` | **`+3.26% / +3.53%`** | `238.51 / 238.10` | `238.76 / 239.02` | `242.66 / 238.38` |
| `32` | `370.49 / 370.85` | **`234.19 / 234.13`** | `240.10 / 239.45` | **`+2.52% / +2.27%`** | `242.85 / 242.47` | `241.00 / 240.53` | `239.18 / 239.19` |

---

## Conclusion

Both candidates are rejected and recorded under `TODO.md` (`Tried and not adopted -> Decode kernels`), and the superseded `mHC: fn in three bytes` bullet is retired from `TileLang kernel family`.
