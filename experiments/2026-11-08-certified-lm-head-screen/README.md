# 2026-11-08 — Mathematically Certified Two-Pass Target `lm_head` Vocabulary Screening on `pack12` (`GB10`, `sm_121a`)

## Summary

- **Base deployment commit**: `8e5ea208` (`Promote spark-ds41f-1107 across switchless (tp3, tp4) and switched (tp3..tp8) shapes`, vLLM patch head `2756478b60b9`, tree `ed1bdb7a7684`).
- **Candidate vLLM patch**: `0762-perf-dsv41-certified-two-pass-lm-head-vocab-screen.patch` (patch head `45094544c85c`, tree `107c0f2eb9fb`).
- **Intended causal variable**: Replace the single-pass 12-bit (`1.5 B/elem`) full-vocabulary greedy pair projection (`packed_vocab_greedy_partials` + `finish_vocab_greedy_pairs`) in `vllm/models/deepseek_v4_1/tilelang/vocab.py` (`project_head_greedy_pairs`) with a mathematically certified two-pass Cauchy-Schwarz vocabulary screening and active-column refinement pipeline (`vocab_x_norm` -> `vocab_screen_greedy_partials` -> `vocab_row_thresholds` -> `vocab_compact_tiles` -> `vocab_packed_refine_partials` -> `vocab_active_greedy_finish`), co-allocating the 8-bit (`1.0 B/elem`) sign-magnitude screening matrix `W_q` inside the single DRM dumb-buffer display carve-out allocation alongside the 12-bit packed `lm_head` weight in `PackedBf16LmHeadMethod.create_weights`.
- **Hardware & Workload**: NVIDIA GB10 (`sm_121a`) on `sp8`, real `DeepSeek-V4.1-Flash-1017` `head.weight` (`[129280, 5120]` BF16), evaluated across two container boots (`Boot 1` and `Boot 2`) for both `tp4-ring` (`N = 32,320`, `vocab = 32,320`, `K = 5,120`) and `tp3-triangle` (`N = 43,136`, `vocab = 43,094`, `K = 5,120`) across `M in [1, 2, 4, 6, 8, 16, 32, 48]`, under both `SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1` (promoted serving configuration) and `SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=0` (ordinary memory).
- **Headline Results**:
  - **Bit-exactness**: **100% bit-exact** (`0` mismatches in both `pairs[:, 0]` max logit and `pairs[:, 1]` global winning token ID across `7,500` random and adversarial near-tie rows in the 2-boot benchmark suite, plus all 107 unit tests in `test_deepseek_v41_tilelang_vocab.py`, `test_display_carveout.py`, and `test_greedy_verification.py`, with `0` PDL source violations).
  - **Active refined columns**: Prunes **98.17%–99.97%** of vocabulary columns (`12–16` active columns at `M = 1`, `97–102` at `M = 6`, `232–272` at `M = 16`, `530–601` at `M = 32`, `592–655` at `M = 48`).
  - **Speedup on `tp4-ring` (`SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1`)**: **`1.44x`** at `M = 1` (`973.71 -> 677.80 us`, **`-295.91 us`**), **`1.42x`** at `M = 6` (`967.45 -> 683.13 us`, **`-284.32 us`**), **`1.41x`** at `M = 16` (`973.50 -> 692.02 us`, **`-281.48 us`**), **`1.36x`** at `M = 32` (`978.01 -> 720.85 us`, **`-257.16 us`**), and **`1.30x`** at `M = 48` (`1043.51 -> 800.29 us`, **`-243.23 us`**).
  - **Speedup on `tp3-triangle` (`SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1`)**: **`1.45x`** at `M = 1` (`1291.48 -> 888.54 us`, **`-402.94 us`**), **`1.45x`** at `M = 6` (`1298.75 -> 896.51 us`, **`-402.24 us`**), **`1.42x`** at `M = 16` (`1291.77 -> 912.27 us`, **`-379.51 us`**), **`1.36x`** at `M = 32` (`1301.68 -> 959.53 us`, **`-342.15 us`**), and **`1.27x`** at `M = 48` (`1376.61 -> 1083.20 us`, **`-293.41 us`**).

---

## Mathematical Certification & Kernel Architecture

During greedy decode (`M = 1`) and greedy target verification (`M = 6` at `c1`, `M = 16` on copy steps, `M = 48` at `c8`), `ParallelLMHead.forward_greedy_pairs` calls `project_head_greedy_pairs(self, x, vocab=..., vocab_start=..., scratch_pairs=True)` to compute each row's exact `(max_logit, global_token_idx)` over the rank's `pack12` shard (`331 MB/rank` on TP3, `247 MB/rank` on TP4 at `1.5 B/elem`).

### 1. Offline 8-Bit Sign-Magnitude Screening Table & Cauchy-Schwarz Residual Bounds

In `build_screen_weights(packed, wq_out=...)` (executed in `1,024`-row chunks during `PackedBf16LmHeadMethod.process_weights_after_loading`):
1. Each vocabulary row $W_v \in \mathbb{R}^K$ (`K = 5,120`) is reconstructed in FP32 both with in-window 12-bit values ($W_{\text{in}, v}$) and with exact CSR exceptions applied ($W_{\text{full}, v}$).
2. Rows with extreme above-window exceptions ($\|W_{\text{full}, v}\|_\infty > 8 \|W_{\text{in}, v}\|_\infty$ or non-finite values) are flagged in `extreme_mask` (`row_bound[v] = +inf`) so they are unconditionally included in the exact refinement pass without distorting the row's 8-bit quantization scale.
3. For all normal rows, we set
   $$\text{row\_scale}[v] = \frac{\max_k |W_{\text{full}, v, k}|}{127 / 128}$$
   and encode each element as 1 sign bit (`bit 7`) plus a 7-bit rounded magnitude $u_7 \in \{0, \dots, 127\}$ (`bits 0..6`):
   $$W_{q, v, k} = (\text{sign} \ll 7) \mid u_7, \qquad \widehat{W}_{v, k} = \text{bf16}\!\left(\frac{\pm u_7}{128}\right) \cdot \text{row\_scale}[v].$$
4. On the device, `vllm_vocab_decode_q8_vec16` unpacks 16 bytes of `W_q` into 16 BF16 values in `[0, 127/128]` using a single `__byte_perm` + constant-exponent OR (`0x3F003F00u`) + BF16 subtract (`- 1.0`), with no nibble extraction or code-plane load.
5. By Cauchy-Schwarz, the inner-product approximation error between exact FP32 dot product $s_{m,v} = \langle x_m, W_{\text{full}, v}\rangle$ and the Tensor Core screening estimate $\hat{s}_{m,v} = \text{row\_scale}[v] \cdot \text{MMA}(x_m, W_{q,v} / 128)$ is bounded by
   $$|s_{m,v} - \hat{s}_{m,v}| \le \|x_m\|_2 \cdot \text{row\_bound}[v],$$
   where
   $$\text{row\_bound}[v] = 1.0005\,\|W_{\text{full}, v} - \widehat{W}_v\|_2 + 5\times 10^{-4}\,\|W_{\text{full}, v}\|_2 + 10^{-7}$$
   absorbs both the quantization residual $\|R_v\|_2$ and Tensor Core `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` accumulation rounding, and
   $$\text{row\_mma\_tol}[v] = 5\times 10^{-4}\,\|W_{\text{full}, v}\|_2 + 10^{-6}$$
   bounds the difference between scalar FP32 dot product in `vocab_row_thresholds` and Tensor Core `pack12` MMA in `vocab_packed_refine_partials` (including 1-ULP BF16 boundary rounding when `out_dtype == torch.bfloat16`).

### 2. Six-Kernel PDL-Chained Runtime Pipeline

All 6 kernels include `pdl.chain()` (`cudaGridDependencySynchronize()` before reading any pointer argument and `cudaTriggerProgrammaticLaunchCompletion()` on completion) and are registered in `tests/kernels/deepseek_v41_pdl_sources.py`:
1. **`vocab_x_norm` (`x_norm_kernel`)**: Computes $\|x_m\|_2$ (`x_norm[m]`) per row (setting `x_norm[m] = +inf` if any element of $x_m$ is NaN/Inf so all columns are refined) and zeros `active_count[0]`.
2. **`vocab_screen_greedy_partials` (`screen_vocab_greedy_partials`)**: Streams `W_q` (`1.0 B/elem`: `164 MB` on TP4, `220 MB` on TP3 — **33.3% smaller than `pack12`**), computes $\hat{s}_{m,v}$, certified lower bound $L_{m,v} = \hat{s}_{m,v} - \|x_m\|_2\,\text{row\_bound}[v]$ (for `row_bound < +inf`), and certified upper bound $U_{m,v} = \hat{s}_{m,v} + \|x_m\|_2\,\text{row\_bound}[v]$, writing per-tile argmax of $L_{m,v}$ to `(partial_vals, partial_idxs)` and per-tile max of $U_{m,v}$ to `tile_max_u[m, tile]`.
3. **`vocab_row_thresholds` (`row_thresholds_kernel`)**: Reduces `partial_vals[m, :]` to find the provisional winning column $v^*_m$, computes the exact `pack12` + CSR-exception dot product $s^{\text{exact}}_{m, v^*_m}$ for that single row, subtracts $\|x_m\|_2\,\text{row\_mma\_tol}[v^*_m]$ (plus 1 BF16 ULP when `out_dtype == torch.bfloat16`), and writes the certified threshold $T_m = \text{ row\_thresh}[m]$.
4. **`vocab_compact_tiles` (`compact_tiles_kernel`)**: Scans tiles using `tile_max_u[m, tile] >= T_m` as a fast tile skip, and within surviving tiles compacts only individual vocabulary columns $v < \text{vocab}$ satisfying $\exists m: U_{m,v} \ge T_m$ into `active_cols[0 : active_count]` via warp ballot + single-lane `atomicAdd`, then sorts `active_cols[0 : active_count]` in ascending column order so tie-breaking (`lowest column index wins`) and DRAM coalescing are preserved.
5. **`vocab_packed_refine_partials` (`refine_vocab_greedy_partials`)**: Launches `ceildiv(N, 32) × ceildiv(M, block_M)` CTAs; CTAs with `bx * 32 >= max(active_count[0], 1)` exit immediately after the PDL barrier (`> 98%` of CTAs), while active CTAs gather the exact `pack12` rows `active_cols[bx * 32 + j]`, run Tensor Core MMA + exact CSR exception correction (`vllm_vocab_refine_tile_argmax_clocal`), and write `(partial_vals[m, bx], partial_idxs[m, bx])`.
6. **`vocab_active_vocab_greedy_pairs` (`finish_active_vocab_greedy_pairs`)**: Reduces only the `ceildiv(max(active_count[0], 1), 32)` active tiles per row and writes `(max_val, vocab_start + best_col)` to `pairs[m, :2]`.

### 3. Single Display Carve-Out Co-Allocation

When `SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1`, `display_carveout.allocate` maps weights from the GB10 display carve-out via DRM dumb buffers. Because DRM dumb-buffer capacity is finite, `PackedBf16LmHeadMethod.create_weights` allocates a single `(n, k + k // 4)` BF16 buffer (`2.5 * n * k` bytes = `411 MiB` on TP4, `548 MiB` on TP3) in `display_carveout`:
- A contiguous prefix view `raw.view(-1)[: n * k].view(n, k)` is passed to `ModelWeightParameter` so `pack_bf16_weight` packs the 12-bit weight in place into the first `1.5 * n * k` bytes.
- The trailing `1.0 * n * k` bytes (`layer._carveout_screen_wq`) hold `W_q[n, k]` inside the exact same DRM carve-out allocation (`len(display_carveout._carveouts) == 1`).

---

## Exact Verification & Benchmark Commands

### 1. Full Unit Test & PDL Source Verification Suite (`sp8`)

```bash
flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host --privileged \
  -v /tmp:/tmp -v /tmp/tl_cache:/root/.tilelang \
  -v /tmp/vocab_patched.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/vocab.py:ro \
  -v /tmp/packed_bf16_patched.py:/opt/spark-ds41f/candidate/vllm/vllm/model_executor/layers/quantization/online/packed_bf16.py:ro \
  -v /tmp/test_vocab_patched.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py:ro \
  -v /tmp/vllm-certified-lm-head-0762/tests/kernels/deepseek_v41_pdl_sources.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/deepseek_v41_pdl_sources.py:ro \
  -w /opt/spark-ds41f/candidate/vllm --entrypoint pytest spark-ds41f:1107 \
  --noconftest -v -p tests.kernels.deepseek_v41_pdl_sources \
  tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py \
  tests/v1/worker/test_display_carveout.py \
  tests/v1/worker/gpu/test_greedy_verification.py
```

- **Result**: `107 passed, 0 failed`, `0` PDL violations (`receipts/pytest.txt`).

### 2. Two-Boot Hardware Benchmark & 7,500-Row Bit-Exactness Suite (`sp8`)

```bash
BLOB=/home/chris/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/blobs/4ae12b941c3facf8616ce62dd45bc6868076c2aeb82106f98123ea6c5bb64182
for CARVEOUT in 1 0; do
  for BOOT in 1 2; do
    flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host --privileged \
      -v /tmp:/tmp -v /tmp/tl_cache:/root/.tilelang \
      -v "$BLOB":/weights/head.safetensors:ro \
      -e SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=$CARVEOUT \
      --entrypoint python3 spark-ds41f:1107 /tmp/bench_two_boots.py baseline $BOOT
    flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host --privileged \
      -v /tmp:/tmp -v /tmp/tl_cache:/root/.tilelang \
      -v "$BLOB":/weights/head.safetensors:ro \
      -v /tmp/vocab_patched.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/vocab.py:ro \
      -v /tmp/packed_bf16_patched.py:/opt/spark-ds41f/candidate/vllm/vllm/model_executor/layers/quantization/online/packed_bf16.py:ro \
      -e SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=$CARVEOUT \
      --entrypoint python3 spark-ds41f:1107 /tmp/bench_two_boots.py candidate $BOOT
  done
done
```

---

## Two-Boot Hardware Results on `sp8` (`GB10`, `sm_121a`)

### Table 1 — Promoted Display Carve-Out (`SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1`), `tp4-ring` (`N = 32,320`, `vocab = 32,320`, `K = 5,120`)

| `M` | Baseline Boot 1 (`us`) | Baseline Boot 2 (`us`) | Baseline Mean ± SD (`us`) | Candidate Boot 1 (`us`) | Candidate Boot 2 (`us`) | Candidate Mean ± SD (`us`) | Active Refined Cols / `32,320` | Delta (`us`) | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `971.76` | `975.66` | `973.71 ± 2.76` | `670.35` | `685.25` | `677.80 ± 10.54` | `16` (`0.05%`) | **`-295.91`** | **`1.44x`** |
| `2` | `968.78` | `969.88` | `969.33 ± 0.78` | `678.90` | `681.73` | `680.32 ± 2.00` | `19` (`0.06%`) | **`-289.02`** | **`1.42x`** |
| `4` | `965.16` | `969.20` | `967.18 ± 2.86` | `681.68` | `685.41` | `683.55 ± 2.64` | `44` (`0.14%`) | **`-283.64`** | **`1.41x`** |
| `6` | `968.51` | `966.39` | `967.45 ± 1.50` | `683.12` | `683.14` | `683.13 ± 0.01` | `97` (`0.30%`) | **`-284.32`** | **`1.42x`** |
| `8` | `965.97` | `969.78` | `967.88 ± 2.69` | `686.66` | `687.34` | `687.00 ± 0.48` | `68` (`0.21%`) | **`-280.88`** | **`1.41x`** |
| `16` | `971.24` | `975.75` | `973.50 ± 3.19` | `690.48` | `693.56` | `692.02 ± 2.18` | `232` (`0.72%`) | **`-281.48`** | **`1.41x`** |
| `32` | `976.02` | `979.99` | `978.01 ± 2.81` | `722.69` | `719.01` | `720.85 ± 2.60` | `530` (`1.64%`) | **`-257.16`** | **`1.36x`** |
| `48` | `1043.35` | `1043.67` | `1043.51 ± 0.23` | `798.47` | `802.10` | `800.29 ± 2.57` | `592` (`1.83%`) | **`-243.23`** | **`1.30x`** |

### Table 2 — Promoted Display Carve-Out (`SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1`), `tp3-triangle` (`N = 43,136`, `vocab = 43,094`, `K = 5,120`)

| `M` | Baseline Boot 1 (`us`) | Baseline Boot 2 (`us`) | Baseline Mean ± SD (`us`) | Candidate Boot 1 (`us`) | Candidate Boot 2 (`us`) | Candidate Mean ± SD (`us`) | Active Refined Cols / `43,094` | Delta (`us`) | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `1290.69` | `1292.27` | `1291.48 ± 1.12` | `888.13` | `888.95` | `888.54 ± 0.58` | `12` (`0.03%`) | **`-402.94`** | **`1.45x`** |
| `2` | `1283.63` | `1297.28` | `1290.46 ± 9.65` | `889.43` | `890.04` | `889.74 ± 0.43` | `23` (`0.05%`) | **`-400.72`** | **`1.45x`** |
| `4` | `1283.62` | `1338.85` | `1311.24 ± 39.05` | `893.47` | `901.48` | `897.48 ± 5.66` | `57` (`0.13%`) | **`-413.76`** | **`1.46x`** |
| `6` | `1283.93` | `1313.56` | `1298.75 ± 20.95` | `894.22` | `898.80` | `896.51 ± 3.24` | `102` (`0.24%`) | **`-402.24`** | **`1.45x`** |
| `8` | `1287.17` | `1287.22` | `1287.20 ± 0.04` | `895.44` | `900.79` | `898.12 ± 3.78` | `88` (`0.20%`) | **`-389.08`** | **`1.43x`** |
| `16` | `1292.26` | `1291.28` | `1291.77 ± 0.69` | `914.37` | `910.16` | `912.27 ± 2.98` | `272` (`0.63%`) | **`-379.51`** | **`1.42x`** |
| `32` | `1302.44` | `1300.91` | `1301.68 ± 1.08` | `945.13` | `973.93` | `959.53 ± 20.36` | `601` (`1.39%`) | **`-342.15`** | **`1.36x`** |
| `48` | `1381.57` | `1371.65` | `1376.61 ± 7.01` | `1046.34` | `1120.06` | `1083.20 ± 52.13` | `655` (`1.52%`) | **`-293.41`** | **`1.27x`** |

### Table 3 — Ordinary Memory (`SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=0`), `tp4-ring` (`N = 32,320`, `vocab = 32,320`, `K = 5,120`)

| `M` | Baseline Boot 1 (`us`) | Baseline Boot 2 (`us`) | Baseline Mean ± SD (`us`) | Candidate Boot 1 (`us`) | Candidate Boot 2 (`us`) | Candidate Mean ± SD (`us`) | Active Refined Cols | Delta (`us`) | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `1038.88` | `1055.02` | `1046.95 ± 11.41` | `725.10` | `727.89` | `726.50 ± 1.97` | `16` | **`-320.46`** | **`1.44x`** |
| `2` | `1038.66` | `1043.14` | `1040.90 ± 3.17` | `735.93` | `742.21` | `739.07 ± 4.44` | `19` | **`-301.83`** | **`1.41x`** |
| `4` | `1038.99` | `1041.20` | `1040.10 ± 1.56` | `742.18` | `742.94` | `742.56 ± 0.54` | `44` | **`-297.54`** | **`1.40x`** |
| `6` | `1045.94` | `1047.52` | `1046.73 ± 1.12` | `748.09` | `748.80` | `748.45 ± 0.50` | `97` | **`-298.29`** | **`1.40x`** |
| `8` | `1042.13` | `1045.96` | `1044.05 ± 2.71` | `748.15` | `756.70` | `752.43 ± 6.05` | `68` | **`-291.62`** | **`1.39x`** |
| `16` | `1053.10` | `1044.87` | `1048.99 ± 5.82` | `773.96` | `770.71` | `772.34 ± 2.30` | `232` | **`-276.65`** | **`1.36x`** |
| `32` | `1045.98` | `1038.27` | `1042.13 ± 5.45` | `816.17` | `813.62` | `814.90 ± 1.80` | `530` | **`-227.23`** | **`1.28x`** |
| `48` | `1174.51` | `1158.05` | `1166.28 ± 11.64` | `898.61` | `916.38` | `907.50 ± 12.57` | `592` | **`-258.79`** | **`1.29x`** |

### Table 4 — Ordinary Memory (`SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=0`), `tp3-triangle` (`N = 43,136`, `vocab = 43,094`, `K = 5,120`)

| `M` | Baseline Boot 1 (`us`) | Baseline Boot 2 (`us`) | Baseline Mean ± SD (`us`) | Candidate Boot 1 (`us`) | Candidate Boot 2 (`us`) | Candidate Mean ± SD (`us`) | Active Refined Cols | Delta (`us`) | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `1383.36` | `1388.70` | `1386.03 ± 3.78` | `944.23` | `950.88` | `947.56 ± 4.70` | `12` | **`-438.48`** | **`1.46x`** |
| `2` | `1382.93` | `1381.21` | `1382.07 ± 1.22` | `955.97` | `958.60` | `957.29 ± 1.86` | `23` | **`-424.79`** | **`1.44x`** |
| `4` | `1383.71` | `1380.89` | `1382.30 ± 1.99` | `965.86` | `970.82` | `968.34 ± 3.51` | `57` | **`-413.96`** | **`1.43x`** |
| `6` | `1385.30` | `1384.03` | `1384.67 ± 0.90` | `981.89` | `968.08` | `974.99 ± 9.77` | `102` | **`-409.68`** | **`1.42x`** |
| `8` | `1400.09` | `1389.27` | `1394.68 ± 7.65` | `1026.06` | `973.77` | `999.92 ± 36.97` | `88` | **`-394.77`** | **`1.39x`** |
| `16` | `1456.57` | `1396.87` | `1426.72 ± 42.21` | `1008.04` | `1000.63` | `1004.34 ± 5.24` | `272` | **`-422.39`** | **`1.42x`** |
| `32` | `1378.43` | `1461.18` | `1419.81 ± 58.51` | `1055.23` | `1052.76` | `1054.00 ± 1.75` | `601` | **`-365.81`** | **`1.35x`** |
| `48` | `1540.21` | `1545.18` | `1542.70 ± 3.51` | `1159.09` | `1172.95` | `1166.02 ± 9.80` | `655` | **`-376.68`** | **`1.32x`** |

---

## Memory Observations & Errors Encountered During Development

- **Memory Footprint**:
  - When `SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1` (promoted serving configuration), `W_q` (`157.8 MiB` on TP4, `210.6 MiB` on TP3) is placed inside the single display carve-out allocation (`411 MiB` total on TP4, `548 MiB` total on TP3 vs `315.6 MiB` / `421.2 MiB` unpacked BF16 previously), consuming **zero** ordinary CUDA allocator memory for `W_q`.
  - Per-row metadata (`row_scale`, `row_bound`, `row_mma_tol`) uses `3 × 4 B × N` = `379 KiB` on TP4 and `505 KiB` on TP3.
  - Runtime workspace tensors (`u_buf`, `tile_max_u`, `x_norm`, `row_thresh`, `active_cols`, `active_count`) are acquired from `vllm`'s shared workspace manager (`current_workspace_manager().get_simultaneous(...)`, `~6.3 MiB` at `M = 48` on TP4, `~8.4 MiB` at `M = 48` on TP3).
- **Errors Encountered & Resolved**:
  1. **Display Carve-Out Threshold in Synthetic Unit Test**: In `test_packed_bf16_lm_head_carveout_shares_single_buffer_with_screen_wq`, using `n, k = 8,192, 4,096` resulted in a packed 12-bit weight of `48 MiB < display_carveout._MIN_WEIGHT_BYTES` (`64 MiB`), causing `confirm_streamed_weights` to reject the synthetic test weight. Increasing the synthetic test dimensions to `n, k = 16,384, 4,096` (`96 MiB` packed `>= 64 MiB`) resolved the test.
  2. **Extreme Out-of-Window `pack12` Exceptions**: When a synthetic row contains an outlier many orders of magnitude larger than its in-window elements (`weight[301, 11] = 300.0`), scaling the entire row to `300.0` would underflow the normal elements in `W_q` and inflate `row_bound`. Detecting `full_amax > 8.0 * in_amax` (`extreme_mask`) and setting `row_bound[v] = +inf` guarantees that extreme-exception rows are always refined by `vocab_packed_refine_partials` while keeping `W_q` well-scaled.

---

## Conclusion

Certified two-pass Cauchy-Schwarz `lm_head` vocabulary screening (`0762`) is **100% bit-exact** (`0` mismatches across `7,500` benchmark rows and `107` unit tests, `0` PDL violations) and delivers a **`1.27x–1.46x` speedup** across all greedy decode and speculative target verification batch sizes (`M = 1..48`), saving **`243–296 us/step` on `tp4-ring`** and **`293–414 us/step` on `tp3-triangle`**. Ready for promotion review.
