# Experiment: mHC Two-Launch PDL Unification & Prefill Roofline Analysis (`2026-11-06-mhc-project-finalize`)

## Objective & Hypothesis

Evaluate the `mHC` sublayer entry (`80` calls per `40`-layer DeepSeek-V4.1 forward pass) across decode (`M = 1..96`) and prefill (`M = 2048, 6144, 16384`) on NVIDIA GB10 (`SM121`, 48 SMs, ~230 GB/s effective LPDDR5X DRAM bandwidth):

1. **Single-Launch vs Two-Launch PDL at `M = 1..96`**:
   - Base Candidate `1107` (`vllm` patch `0634`) introduced `project_and_finalize_streams` with `FINALIZE_FUSE_MAX_ROWS = 1`, where the last split CTA (`1` CTA of `128` threads) executed `finalize` after an atomic arrival counter.
   - We first tested parallelizing `project_and_finalize_streams` across `block_M = 16` token rows (`8` warps/token) and compared both single-launch fused designs against the two-launch `project_streams` + `finalize` path (`Split`, chained via hardware Programmatic Dependent Launch `pdl.chain()`) across two fresh container boots on `sp8`.
   - **Hardware finding**: Even at `M = 1`, two-launch `project_streams` + `finalize` (`8.71–8.81 µs`) beats Base `0634` (`13.06–13.86 µs`) by **`-4.30..-5.15 µs/call` (`-32.9%..-37.2%`, saving `-344..-412 µs` per 40-layer pass)** and also beats parallel-CTA fused (`9.93–10.02 µs`) by **`-1.22..-1.25 µs/call`**.
   - **Root cause**: At `M = 1`, `project_streams` launches `splits = 40` CTAs across GB10's `48` SMs, leaving `8` SMs idle. Hardware PDL immediately schedules `finalize` (`1` CTA) onto one of the `8` idle SMs while `project_streams` is still running and preloads `weight` (`5,120` BF16), `scale`, and `base` into registers before `pdl.chain()` (`cudaGridDependencySynchronize()`). Inside a single fused CTA, the last arriving split CTA cannot begin `finalize` until after its final GEMM, global memory writes (`R_out, C, Q, S, P`), `__threadfence()`, and `atomicAdd`.
   - **Action (Owner directive: remove optionality and stick with the best way)**: Since two-launch `project_streams` + `finalize` is strictly faster at **every** `M >= 1`, patch `0731-mhc-project-finalize.patch` removes `project_and_finalize_streams`, `FINALIZE_FUSE_MAX_ROWS`, `_FUSE_HEADER`, `_project_finalize_call`, `_project_finalize`, `_COUNTERS`, and `_counters` completely (`-690` net lines), making the two-launch path unconditional across all `M >= 1` and eliminating `8` redundant JIT kernel compilations from `TileKernelsMHC.declare_kernels()` (`-55 s` warmup/test time).

2. **Prefill (`M = 2048, 6144, 16384`) DRAM Byte-Traffic & Bandwidth Roofline**:
   - Analyze the exact compulsory DRAM byte traffic of `project_streams` (`post_pre`) and `finalize` at `M = 2048, 6144, 16384` for both `world = 1, with_aux = False` and `world = 3, with_aux = True` (TP3 sequence-parallel unreduced ranks + fused `aux_out`), and test whether alternative CTA grid orders or tile shapes improve prefill throughput while preserving 100% bit-identity.

## Precision & Invariants

- **Zero precision reduction**: All TF32 3-way split GEMM projections (`Fn = high + middle + low`), FP32 stream folding (`T.fma` sequence), fixed `sp = 0 .. splits - 1` (`BLOCK_H = 128`, `splits = 40`) reduction of `S, P, Q`, Sinkhorn iterations (`20`), BF16 collapse RMSNorm, and MXFP8 block-32 E4M3/UE8M0 quantization (`yq, ysf`) are 100% bit-identical (`int16`, `int32`, `uint8` views) across finite and non-finite (`NaN`, `+Inf`, `-Inf`, `-0.0`) inputs.

## Base & Candidate Revisions

- **Base deployment commit**: `0b3e98934f0aa7c7c251b79dae3ac148f869304e` (Candidate `1107`)
- **Base vLLM commit**: `c47b54ae471a24b830e3515212b8c0bf5ee14604` (tree `13e4781d19bc4a18b0d3d4daad7921c465e3355e`)
- **Candidate vLLM commit**: `f733425c5faaaf74cb64ce601aae792613ca67d1` (tree `3cc497a78575411524e348295a31b9cde199d5f2`)
- **Patch**: `experiments/2026-11-06-mhc-project-finalize/vllm/0731-mhc-project-finalize.patch`

## Exact Commands

```bash
# 1. Full GPU unit test suite on sp8 (174 passed in 108.01s)
docker run --rm --gpus all --ipc=host --entrypoint pytest \
  -v /home/chris/spark-lab/mhc-project-finalize/vllm/vllm/models/deepseek_v4_1/tilelang/mhc.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/mhc.py:ro \
  -v /home/chris/spark-lab/mhc-project-finalize/vllm/tests/kernels/test_deepseek_v41_tilelang_mhc.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_tilelang_mhc.py:ro \
  -v /home/chris/spark-lab/mhc-project-finalize/vllm/tests/kernels/deepseek_v41_pdl_sources.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/deepseek_v41_pdl_sources.py:ro \
  spark-ds41f:ced-projection-51c1a32cd78c-reuse-3795255f-git --noconftest -o pythonpath=/opt/spark-ds41f/candidate/vllm \
  /opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_tilelang_mhc.py -q

# 2. CPU AST & split-order bit-identity verification
docker run --rm --entrypoint python3 \
  -v /home/chris/spark-lab/mhc-project-finalize/vllm/vllm/models/deepseek_v4_1/tilelang/mhc.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/mhc.py:ro \
  -v /home/chris/spark-lab/mhc-project-finalize/vllm/tests/kernels/deepseek_v41_pdl_sources.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/deepseek_v41_pdl_sources.py:ro \
  -v /home/chris/spark-lab/mhc-project-finalize/repo/experiments/2026-11-06-mhc-project-finalize:/exp \
  spark-ds41f:ced-projection-51c1a32cd78c-reuse-3795255f-git \
  /exp/bench_mhc_project_finalize.py --cpu-verify \
  --mhc-path /opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/mhc.py \
  --pdl-sources-path /opt/spark-ds41f/candidate/vllm/tests/kernels/deepseek_v41_pdl_sources.py \
  --out /exp/cpu-verification.json

# 3. Two-boot GPU benchmark on sp8 (boot1 & boot2)
docker run --rm --gpus all --ipc=host --entrypoint python3 \
  -v /home/chris/spark-lab/mhc-project-finalize/vllm/vllm/models/deepseek_v4_1/tilelang/mhc.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/mhc.py:ro \
  -v /home/chris/spark-lab/mhc-project-finalize/repo/experiments/2026-11-06-mhc-project-finalize:/exp \
  spark-ds41f:ced-projection-51c1a32cd78c-reuse-3795255f-git \
  /exp/bench_mhc_project_finalize.py --gpu-bench --out /exp/gpu-boot1.json
```

## 2-Boot GPU Benchmark Results on `sp8` (NVIDIA GB10)

### Table A1: Initial 2-Boot Screen — Fused (`Base0634` vs Parallel-CTA `CandFused`) vs Two-Launch PDL (`Split`) (`post_pre`, `world=1, with_aux=False`, CUDA Graph)

| `M` | `Split` (`project_streams` + `finalize`) Boot 1 / Boot 2 (µs) | `Base0634` (1-CTA finalize) Boot 1 / Boot 2 (µs) | `CandFused` (Parallel-CTA finalize) Boot 1 / Boot 2 (µs) | `Split` vs `Base0634` at `M = 1` (µs / call) | 80-Call Pass Savings (`Split` vs `Base0634`) | Bit-Exact |
|---:|---:|---:|---:|---:|---:|:---:|
| **1** | **8.77 / 8.71** | 13.06 / 13.86 | 10.02 / 9.93 | **-4.30 / -5.15 (-33.4% / -37.2%)** | **-344 / -412 µs** | PASS |
| **2** | **8.96 / 8.95** | 16.59 / 17.61 | 10.37 / 10.30 | -7.63 / -8.66 | -610 / -693 µs | PASS |
| **4** | **8.93 / 8.96** | 25.00 / 26.22 | 11.49 / 11.47 | -16.07 / -17.26 | -1,286 / -1,381 µs | PASS |
| **6** | **9.10 / 9.10** | 32.89 / 34.81 | 12.69 / 12.66 | -23.79 / -25.71 | -1,903 / -2,057 µs | PASS |
| **8** | **9.14 / 9.10** | 40.23 / 41.35 | 13.68 / 13.83 | -31.09 / -32.25 | -2,487 / -2,580 µs | PASS |
| **12** | **9.61 / 9.73** | 56.51 / 56.39 | 16.02 / 15.98 | -46.90 / -46.66 | -3,752 / -3,733 µs | PASS |
| **16** | **9.96 / 9.81** | 70.95 / 72.52 | 18.23 / 18.08 | -60.99 / -62.71 | -4,879 / -5,017 µs | PASS |
| **24** | **14.37 / 14.34** | 69.25 / 71.15 | 22.84 / 22.80 | -54.88 / -56.81 | -4,390 / -4,545 µs | PASS |
| **32** | **14.84 / 15.13** | 76.29 / 76.43 | 22.90 / 23.22 | -61.45 / -61.30 | -4,916 / -4,904 µs | PASS |
| **48** | **20.53 / 20.66** | 78.63 / 80.23 | 27.60 / 27.68 | -58.10 / -59.57 | -4,648 / -4,766 µs | PASS |
| **64** | **25.73 / 25.67** | 82.68 / 84.08 | 32.39 / 32.33 | -56.95 / -58.41 | -4,556 / -4,673 µs | PASS |
| **96** | **32.73 / 32.68** | 90.24 / 90.02 | 39.20 / 39.17 | -57.51 / -57.34 | -4,601 / -4,587 µs | PASS |

### Table A2: Follow-up 2-Boot Screen (`gpu-boot1.json` & `gpu-boot2.json`) — Unconditional Two-Launch `Split` vs `Fn` Stream-0 Pre-PDL Load Variant (`M = 1..96`, CUDA Graph)

| `M` | Canonical Two-Launch `Split` Boot 1 / Boot 2 (µs) | `Fn0` Pre-PDL Variant Boot 1 / Boot 2 (µs) | `project_only` Boot 1 / Boot 2 (µs) | `finalize_only` Boot 1 / Boot 2 (µs) | Bit-Exact |
|---:|---:|---:|---:|---:|:---:|
| **1** | **8.76 / 8.81** | 8.80 / 8.78 | 5.32 / 5.23 | 3.84 / 3.76 | PASS |
| **2** | **9.06 / 8.95** | 8.71 / 9.03 | 5.29 / 5.49 | 3.69 / 3.82 | PASS |
| **4** | **8.86 / 9.10** | 8.67 / 9.01 | 5.22 / 5.36 | 3.71 / 3.77 | PASS |
| **6** | **9.08 / 9.14** | 8.84 / 9.01 | 5.32 / 5.41 | 3.69 / 3.79 | PASS |
| **8** | **8.86 / 9.12** | 8.87 / 9.25 | 5.34 / 5.41 | 3.69 / 3.77 | PASS |
| **12** | **9.54 / 9.71** | 9.23 / 9.61 | 5.54 / 5.70 | 4.13 / 3.76 | PASS |
| **16** | **10.14 / 9.83** | 9.52 / 9.75 | 5.88 / 5.91 | 4.07 / 3.89 | PASS |
| **24** | **14.71 / 14.47** | 14.40 / 14.58 | 10.57 / 10.79 | 4.51 / 4.08 | PASS |
| **32** | **14.95 / 15.11** | 14.81 / 14.75 | 11.27 / 11.25 | 4.36 / 4.18 | PASS |
| **48** | **20.42 / 20.75** | 20.06 / 20.12 | 16.13 / 16.40 | 4.56 / 4.89 | PASS |
| **64** | **25.71 / 25.77** | 25.92 / 25.96 | 21.27 / 21.39 | 4.78 / 4.86 | PASS |
| **96** | **32.81 / 32.68** | 32.85 / 33.01 | 27.45 / 27.34 | 5.95 / 5.79 | PASS |

## Prefill (`M = 2048, 6144, 16384`) DRAM Byte-Traffic & Roofline Analysis

### Why `188.74 MiB` at `M = 6144` Undercounted `project_streams` Traffic by `4.5x`

At `M = 6144, H = 5120, MULT = 4, MIXES = 24, splits = 40`:
- `188.74 MB` (`180.00 MiB`) is only the size of `X` (`previous_output`) when `world = 3` (`3 * 6144 * 5120 * 2B = 188.74 MB`).
- In `mode = "post_pre"`, `project_streams` folds `X` into all `4` residual streams `R`, writes the `4` updated streams `R_out`, collapses them into `C` (and `Aux` when `with_aux = True`), and writes the `40`-split partials `P, S, Q`:
  1. **Read `X`**: `world * M * H * 2B` (`62.91 MB` at `world = 1`; `188.74 MB` at `world = 3`)
  2. **Read `R` (4 residual streams)**: `M * 4 * H * 2B = 251.66 MB` (`240.00 MiB`)
  3. **Read `Fn` + input mixes**: `24 * 4 * H * 4B + M * 24 * 4B = 2.56 MB`
  4. **Write `R_out` (4 updated residual streams)**: `M * 4 * H * 2B = 251.66 MB` (`240.00 MiB`)
  5. **Write `C` (collapsed stream)**: `M * H * 2B = 62.91 MB` (`60.00 MiB`)
  6. **Write `Aux` (when `with_aux = True`)**: `M * H * 2B = 62.91 MB` (`60.00 MiB`)
  7. **Write `P, S, Q` (`40` splits)**: `40 * M * (24 + 1 + 1) * 4B = 25.56 MB` (`24.38 MiB`)
  - **`project_streams` Compulsory DRAM Traffic at `M = 6144`**:
    - `world = 1, with_aux = False`: **`657.34 MB` (`626.89 MiB`)**
    - `world = 3, with_aux = True`: **`846.08 MB` (`806.89 MiB`)**
- Next, `finalize` reads `C` (`62.91 MB`) and `P, S, Q` (`25.56 MB`), and writes `y` (`62.91 MB` BF16), `yq` (`31.46 MB` E4M3), `ysf` (`0.98 MB` UE8M0), and `post, comb, pre` (`0.59 MB` FP32):
  - **`finalize` Compulsory DRAM Traffic at `M = 6144`**: **`184.42 MB` (`175.88 MiB`)**

### Table B: 2-Boot Prefill Latency, Compulsory DRAM Traffic, and Achieved GB/s on `sp8` (`gpu-boot1.json` & `gpu-boot2.json`)

| Config | `M` | Compulsory DRAM `project` / `finalize` / **Total (MB)** | `project_only` Boot 1 / Boot 2 (µs) | `finalize_only` Boot 1 / Boot 2 (µs) | Two-Launch Total Boot 1 / Boot 2 (µs) | Achieved `project` BW (GB/s) | Achieved Total BW (GB/s) | % of GB10 ~230 GB/s Roofline |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `w=1, aux=False` | **2048** | 220.42 / 61.48 / **281.90 MB** | 992.5 / 1,030.2 | 300.0 / 290.7 | **1,337.0 / 1,345.1** | 222.1 / 213.9 | **210.8 / 209.6** | **91.7% / 91.1%** |
| `w=1, aux=False` | **6144** | 657.34 / 184.42 / **841.76 MB** | 3,016.8 / 2,967.1 | 861.1 / 881.8 | **3,850.0 / 3,813.7** | 217.9 / 221.5 | **218.6 / 220.7** | **95.1% / 96.0%** |
| `w=1, aux=False` | **16384** | 1,749.64 / 491.79 / **2,241.43 MB** | 7,779.5 / 7,794.7 | 2,326.2 / 2,336.1 | **10,211.1 / 10,048.1** | 224.9 / 224.4 | **219.5 / 223.1** | **95.4% / 97.0%** |
| `w=3, aux=True` | **2048** | 283.33 / 61.48 / **344.81 MB** | 1,285.5 / 1,278.7 | 291.6 / 295.5 | **1,588.8 / 1,623.6** | 220.4 / 221.6 | **217.0 / 212.4** | **94.4% / 92.3%** |
| `w=3, aux=True` | **6144** | 846.08 / 184.42 / **1,030.50 MB** | 3,748.9 / 3,769.6 | 874.9 / 866.1 | **4,636.6 / 4,626.9** | 225.7 / 224.4 | **222.3 / 222.7** | **96.6% / 96.8%** |
| `w=3, aux=True` | **16384** | 2,252.95 / 491.79 / **2,744.74 MB** | 9,879.6 / 9,899.6 | 2,312.0 / 2,327.4 | **12,078.3 / 12,117.0** | 228.0 / 227.6 | **227.2 / 226.5** | **98.8% / 98.5%** |

### Measured Prefill Dead Ends

We also screened three bit-identical `project_streams` structural variants on `sp8` at `M = 2048, 6144, 16384`:
1. **`by`-major grid `(ceildiv(M, 16), splits)` instead of `(splits, ceildiv(M, 16))`**:
   - Measured **`+352 µs` slower at `M = 6144, w=1`** (`4,164.6 µs` vs `3,812.7 µs`) and **`+506 µs` slower at `M = 6144, w=3`** (`5,131.1 µs` vs `4,625.0 µs`).
   - Reason: With `(splits, ceildiv(M, 16))`, 40 consecutive CTAs execute `sp = 0..39` for the same 16-token row tile `by` simultaneously, coalescing the full `5,120`-column rows of `X, R, R_out, C, Aux` in DRAM/L2 and keeping the tail of `C` hot in L2 for `finalize`.
2. **`block_M = 32` (`256` threads/CTA, `16 KB` `A_shared`)**:
   - Measured **`+520 µs` slower at `M = 6144, w=1`** (`4,332.9 µs` vs `3,812.7 µs`) and **`+191 µs` slower at `M = 6144, w=3`** (`4,816.2 µs` vs `4,625.0 µs`) due to halving CTA scheduling granularity per SM.
3. **Preloading stream `0` of `Fn` before `pdl.chain()` (`Fn0` Pre-PDL Variant)**:
   - Neutral within noise at decode (`±0.02..0.62 µs`) and `M = 6144` (`-42..+24 µs`), but **`+135..+155 µs` slower at `M = 16384, w=3`** (`12,233.0 / 12,252.0 µs` vs `12,078.3 / 12,117.0 µs`) because at `M = 16384` (`40,960` CTAs = `853` waves), `99.88%` of CTAs do not wait at `pdl.chain()` and splitting stream `0` out of the `for n in T.unroll(MULT):` loop adds register pressure across all `40,960` CTAs. Kept unmodified `project_streams`.

## Conclusion

- **Promotable win (`0731-mhc-project-finalize.patch`)**: Removing `project_and_finalize_streams` and `FINALIZE_FUSE_MAX_ROWS` makes the two-launch `project_streams` + `finalize` PDL path unconditional across all `M >= 1`, saving **`4.30–5.15 µs/call` (`344–412 µs` per 40-layer forward pass, `-33.4%..-37.2%`) at `M = 1`**, removing `690` net lines of dead kernel/counter code, and eliminating `8` redundant JIT compilations per worker startup (`-55 s` in `test_deepseek_v41_tilelang_mhc.py`) while keeping prefill at **95–98.8% of GB10's ~230 GB/s effective DRAM bandwidth roofline** with 100% bit-identical outputs.
