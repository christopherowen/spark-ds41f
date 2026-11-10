# 2026-11-08 — 12-Bit Packed BF16 (`pack12`) Kernel & Memory Screening + Unconditional Bounds-Masked Loads & Byte-Level Vocab SWAR (`vLLM 0762`)

## Base & Candidate Identity

- **Base deployment commit**: `cc20ed18b` (`spark-ds41f-1107`)
- **Base vLLM source**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` + 641 patches (`0001..0761`, patch head `2756478b60b90700b9918b457aeb38657da637b5`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate vLLM source**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` + 642 patches (`0001..0762`, patch `vllm/0762-perf-dsv41-unconditional-pack12-plane-loads-and-swar.patch`, patch head `530ecffb0ab2b659a243828835d6dabf5e9333e0`, tree `2812133965ee0c217fe8869bfd0099869be52ded`, `patchset_sha256` `e09d3e860b176b9b4b9f9469b465f21b40b59b1324e6006ee46f88c2e168b047`)
- **Hardware**: Isolated DGX Spark node `sp7` (`NVIDIA GB10`, SM121, `spark-ds41f:1107`)

---

## Overview of Screened Levers

We screened five 12-bit packed BF16 (`packed_bf16` / `pack12`) kernel and memory optimizations alongside unconditional bounds-masked plane/offset loading on `sp7` across two independent container boots:

1. **Unconditional Bounds-Masked Row-Offset Loads + PDL Chain in `gemm.py` (`vLLM 0762`, Promoted in PR)**:
   - In `_packed_bf16_gemm` and `packed_bf16_gemm_two` (`vllm/models/deepseek_v4_1/tilelang/gemm.py`), every CTA thread previously called `vllm_packed_bf16_filter_exc(..., 0, K, ...)` before the K-loop, which branched on `if (num_exceptions <= 0) return;` and then ran a `while (lo < hi)` upper-bound binary search over `exception_entries` for `col < K` even though `col_end == K` covers the entire row (`0..K`).
   - Furthermore, `_packed_bf16_gemm` (`packed_bf16_gemm_decode`) and `packed_bf16_gemm_two` were missing `pdl.chain()` despite running on the decode stream (`TL_DISABLE_WARP_SPECIALIZED: True`).
   - Adding `vllm_packed_bf16_load_row_bounds` (unconditional `has_exc ? __ldg(&row_offsets[row]) : 0` without any branch or binary search loop), making `vllm_packed_bf16_filter_exc` load `row_offsets` unconditionally, and chaining `_packed_bf16_gemm` and `packed_bf16_gemm_two` with `pdl.chain()` delivers **1.01x–1.41x speedup** across non-split-K packed BF16 GEMMs (`compressor_up_5120x512`, `indexer_wq_b_4096x1280`, `indexer_wk_128x5120`, `compressor_down_512x10240`) with **100% bit-exact (`0 ULP`) output parity** and **0 PDL violations across 449 launched kernels**.

2. **Byte-Level `__byte_perm` SWAR Unpack + Direct Per-Fragment-Column Exception Accumulation in `vocab.py` (`vLLM 0762`, Promoted in PR)**:
   - Upgraded `vllm_vocab_decode_vec16` in `vllm/models/deepseek_v4_1/tilelang/vocab.py` from 16-bit halfword SWAR (`8` `__byte_perm`s + `24` bit-ops = `32` ops per 4-byte code word, `128` ops per 32 BF16 values) to byte-level `__byte_perm` SWAR (`4` `__byte_perm`s + `14` bit-ops = `18` ops per 4-byte code word, `72` ops per 32 BF16 values, matching `gemm.py`), cutting inner-loop unpack ALU by **44%**.
   - Replaced the post-K-loop tile-wide CSR exception broadcast loop in `packed_vocab_projection`, `packed_vocab_greedy_partials`, and `packed_vocab_markov_greedy_partials` (which iterated all 128/256 CTA threads across every exception in the `block_N`-row tile `row_offsets[first] .. row_offsets[first + block_N]` and guarded updates with `if j == row:`) with `vllm_vocab_apply_row_exceptions` / `vllm_vocab_apply_row_exceptions_clocal`, where each thread loads `row_offsets[col0 .. col0 + 2]` directly for its 2 MMA fragment columns (`c_frag[frag_idx]`, `c_frag[frag_idx + 1]`) and accumulates only its own column exceptions in list order.
   - Delivers **1.03x–1.59x speedup** at real checkpoint exception density (`0.00566%`, saving up to **-447 us** at `M=128` and **-9.65 ms** at `M=2048` on TP4, and **-687 us** at `M=128` and **-14.03 ms** at `M=2048` on TP3) and **2.27x–3.50x speedup** at stressed exception density (`0.33%`), with **100% bit-for-bit (`0 ULP`) output parity**.

3. **Lossless 12-Bit Token Embedding (`embed_tokens.weight` in `embedding.py`, Already Active via `0671`)**:
   - Verified across all 129,280 rows of real `DeepSeek-V4.1-Flash-1017` `model.embed_tokens.weight` (`[129280, 5120]` BF16): `0` bit mismatches (`0 ULP`) across all `661,913,600` elements.
   - Saves **78.91 MiB/rank** on TP4 (`315.62 MiB -> 236.72 MiB` + `0.33 MiB` CSR metadata = `78.58 MiB` net) and **105.21 MiB/rank** on TP3 (`420.86 MiB -> 315.64 MiB` + `0.45 MiB` CSR metadata = `104.77 MiB` net).
   - Confirmed `vllm_packed_vocab_embedding_row` in `embedding.py` already issues `row_offsets[row..row+1]` in parallel with the 16-byte coherent plane loads before the `if (exc_start < exc_end)` branch, running in **4.15–4.75 us** across decode batches (`M=6..96`) in both regular unified memory and GB10 display carve-out (`SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1`).

4. **Even-Base Exponent Window (`[2B, 2B + 15]`) + Verbatim Low-Byte Unpack ALU (Measured Dead End — Rejected)**:
   - Keeping the BF16 low byte (`E0 | M6..M0`) verbatim and packing `[a.S | b.S | (b.o >> 1) | (a.o >> 1)]` into the shared meta byte reduces unpack ALU from 18 ops/word (`cur15`) to 10 ops/word (`even16`).
   - However, because code `0` can no longer represent exact zero (`0x0000` has `E0 = 0`, which maps to `2B + 0` instead of `0.0`), all exact zeros become exceptions and the even-base alignment constraint (`2B`) shifts compressor/indexer tensors off their optimal odd exponent base (`base = 113`), inflating exception counts on real `DeepSeek-V4.1-Flash-1017` weights by **2.68x–29.59x** (`lm_head` `37,461 -> 100,456`; `embed_tokens` `113,128 -> 3,347,764` including `3,221,291` exact zeros; `compressor` `down_proj` `17,995 -> 72,226`; `indexer` `wq_b` `18,774 -> 62,331`).
   - Even at matched exception counts, the 8 fewer ALU ops per word are completely hidden behind shared-memory `ldmatrix` / `mma.sync` latency (`1.00x` across GEMMs).

5. **Direct Register-to-MMA `B`-Fragment Layout & Group-Sentinel Escape (`meta == 0u`) & Multi-Chain Accumulators (`kChains`) (Measured Trade-offs — Not Promoted)**:
   - **Direct `B`-fragment layout**: Eliminating `B_shared` in `packed_bf16_gemm_splitk` speeds up decode by `1.04x–1.27x` *only* when weights are pre-tiled offline into `[N // 8, K // 128, ...]` MMA super-atoms; keeping row-major `[N, K + K // 2]` weights and transposing in registers via `__shfl_xor_sync` incurs 8-way shared-memory bank conflicts (`0.54x–0.99x`), and pre-tiling breaks `packed_bf16_unpack` compatibility used at prefill (`M > 256`). In `vocab.py`, `ROW_TILES = (16, 32, 64, 128)` already fits `block_M = 128` in `90,112 B <= 101,376 B` shared memory via `act_stages = 2`.
   - **Group-sentinel escape (`meta == 0u`)**: On real `DeepSeek-V4.1-Flash-1017` weights (`0.22%–0.36%` element exception rate on compressor/indexer), `24.8%–36.7%` of 128-column groups and **92.8%–98.7% of 32-lane warps** contain at least one exception, so `__any_sync(0xffffffffu, meta == 0u)` triggers on >92% of warps and provides no fast-path skip.
   - **Multi-chain MMA accumulators (`kChains in {1, 2, 4}`)**: On SM121 (`GB10`), a single warp processes `K=128` (`8` `k16` steps) across `2` independent `n8` tiles (`16` `mma.sync` instructions per K-group) interleaved with shared-memory unpack ALU, which already hides the `mma.sync` pipeline latency. Adding `kChains = 2` or `4` doubles or quadruples FP32 accumulator registers (`16 -> 32 -> 64` regs/thread) and adds end-of-K reduction overhead, resulting in `0.91x–1.00x` across `M in {1, 6, 12, 16}`.

---

## Measured Results

### 1. Real `DeepSeek-V4.1-Flash-1017` Checkpoint Exception & Group-Escape Census (`sp7`)

| Weight Tensor | Shape | Total Elements | `cur15` Base `[B, B+14]` | `cur15` Exceptions (%) | `cur15` CSR Bytes | `even16` Base `[2B, 2B+15]` | `even16` Exceptions (%) | `even16` / `cur15` Exc Ratio | `cur15` Warp-32 Escape Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lm_head.weight` (full) | `(129280, 5120)` | `661,913,600` | `110` | `37,461` (`0.00566%`) | `0.66 MiB` | `110` | `100,456` (`0.01518%`) | `2.68x` | `1.39%` |
| `lm_head.weight` (TP4 rank 0) | `(32320, 5120)` | `165,478,400` | `110` | `8,546` (`0.00516%`) | `0.16 MiB` | `110` | `23,608` (`0.01427%`) | `2.76x` | `1.29%` |
| `lm_head.weight` (TP3 rank 0) | `(43096, 5120)` | `220,651,520` | `110` | `11,297` (`0.00512%`) | `0.21 MiB` | `110` | `31,260` (`0.01417%`) | `2.77x` | `1.28%` |
| `embed_tokens.weight` (full) | `(129280, 5120)` | `661,913,600` | `110` | `113,128` (`0.01709%`) | `1.35 MiB` | `110` | `3,347,764` (`0.50577%`) | `29.59x` | `3.28%` |
| `compressor.down_proj` (L0) | `(512, 10240)` | `5,242,880` | `113` | `17,995` (`0.34323%`) | `0.07 MiB` | `114` | `72,226` (`1.37760%`) | `4.01x` | `98.24%` |
| `compressor.up_proj` (L0) | `(5120, 512)` | `2,621,440` | `114` | `9,917` (`0.37830%`) | `0.06 MiB` | `114` | `18,434` (`0.70320%`) | `1.86x` | `98.73%` |
| `indexer.wq_b` (L0) | `(4096, 1280)` | `5,242,880` | `113` | `18,774` (`0.35809%`) | `0.09 MiB` | `114` | `62,331` (`1.18887%`) | `3.32x` | `98.58%` |
| `indexer.wk` (L0) | `(128, 5120)` | `655,360` | `113` | `2,322` (`0.35431%`) | `0.01 MiB` | `114` | `7,722` (`1.17828%`) | `3.33x` | `98.31%` |

---

### 2. Promoted Change 1: Unconditional Row-Bounds Load + PDL Chain in `_packed_bf16_gemm` (`gemm.py`, 2 Container Boots on `sp7`)

| Shape `(N, K)` | `M` | Baseline Boot 1 (us) | Baseline Boot 2 (us) | Candidate `0762` Boot 1 (us) | Candidate `0762` Boot 2 (us) | Mean Speedup | Bit-Exact (`0 ULP`) |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `compressor_up` `(5120, 512)` | `1` | `11.48` | `11.33` | `9.85` | `9.69` | **1.17x** (`-1.64 us`) | `True` |
| `compressor_up` `(5120, 512)` | `6` | `14.64` | `11.33` | `10.40` | `9.64` | **1.30x** (`-2.97 us`) | `True` |
| `compressor_up` `(5120, 512)` | `16` | `11.33` | `11.39` | `9.68` | `9.62` | **1.18x** (`-1.71 us`) | `True` |
| `compressor_up` `(5120, 512)` | `32` | `14.50` | `14.48` | `12.79` | `12.75` | **1.13x** (`-1.72 us`) | `True` |
| `compressor_up` `(5120, 512)` | `64` | `18.49` | `18.48` | `15.25` | `15.27` | **1.21x** (`-3.23 us`) | `True` |
| `compressor_up` `(5120, 512)` | `128` | `32.76` | `32.73` | `31.11` | `31.17` | **1.05x** (`-1.61 us`) | `True` |
| `compressor_up` `(5120, 512)` | `512` | `105.00` | `104.74` | `101.75` | `101.81` | **1.03x** (`-3.09 us`) | `True` |
| `indexer_wq_b` `(4096, 1280)` | `1` | `16.39` | `16.40` | `13.28` | `13.34` | **1.23x** (`-3.09 us`) | `True` |
| `indexer_wq_b` `(4096, 1280)` | `6` | `16.53` | `16.50` | `13.48` | `13.46` | **1.23x** (`-3.05 us`) | `True` |
| `indexer_wq_b` `(4096, 1280)` | `16` | `16.54` | `16.57` | `13.85` | `13.87` | **1.19x** (`-2.70 us`) | `True` |
| `indexer_wq_b` `(4096, 1280)` | `32` | `27.91` | `27.91` | `24.58` | `24.64` | **1.13x** (`-3.30 us`) | `True` |
| `indexer_wq_b` `(4096, 1280)` | `64` | `35.66` | `35.69` | `31.40` | `31.43` | **1.14x** (`-4.26 us`) | `True` |
| `indexer_wq_b` `(4096, 1280)` | `128` | `52.56` | `52.54` | `48.93` | `48.95` | **1.07x** (`-3.61 us`) | `True` |
| `indexer_wq_b` `(4096, 1280)` | `512` | `181.79` | `181.74` | `175.36` | `175.40` | **1.04x** (`-6.39 us`) | `True` |
| `indexer_wk` `(128, 5120)` | `1` | `43.58` | `43.58` | `42.97` | `42.96` | **1.01x** (`-0.62 us`) | `True` |
| `indexer_wk` `(128, 5120)` | `6` | `43.58` | `43.57` | `42.99` | `42.97` | **1.01x** (`-0.60 us`) | `True` |
| `indexer_wk` `(128, 5120)` | `16` | `43.99` | `43.98` | `42.94` | `42.95` | **1.02x** (`-1.04 us`) | `True` |
| `indexer_wk` `(128, 5120)` | `32` | `49.73` | `49.72` | `48.06` | `48.07` | **1.03x** (`-1.66 us`) | `True` |
| `indexer_wk` `(128, 5120)` | `64` | `62.25` | `62.24` | `60.68` | `60.68` | **1.03x** (`-1.57 us`) | `True` |

---

### 3. Promoted Change 2: Byte-Level `__byte_perm` SWAR + Per-Fragment-Column Exception Accumulation in `vocab.py` (2 Container Boots on `sp7`)

#### A. At Real Checkpoint Exception Density (`0.00566%`, `~9,365` exceptions on TP4 / `~12,487` on TP3)

| Shape `(N, K)` | `M` | Baseline Full Logits (us) [Boot 1 / Boot 2] | Candidate `0762` Full Logits (us) [Boot 1 / Boot 2] | Full Logits Speedup | Baseline Greedy Argmax (us) [Boot 1 / Boot 2] | Candidate `0762` Greedy Argmax (us) [Boot 1 / Boot 2] | Greedy Speedup | Bit-Exact (`0 ULP`) |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `lm_head` TP4 `(32320, 5120)` | `1` | `1092.4 / 1091.7` | `1051.6 / 1051.8` | **1.04x** (`-40.4 us`) | `1038.3 / 1068.9` | `1031.2 / 1031.7` | **1.02x** (`-22.2 us`) | `True` |
| `lm_head` TP4 `(32320, 5120)` | `6` | `1060.6 / 1065.7` | `1046.5 / 1046.4` | **1.02x** (`-16.7 us`) | `1046.7 / 1039.2` | `1031.6 / 1035.1` | **1.01x** (`-9.6 us`) | `True` |
| `lm_head` TP4 `(32320, 5120)` | `12` | `1085.3 / 1096.6` | `1052.1 / 1048.1` | **1.04x** (`-40.9 us`) | `1079.9 / 1046.3` | `1043.6 / 1037.1` | **1.02x** (`-22.8 us`) | `True` |
| `lm_head` TP4 `(32320, 5120)` | `16` | `1097.4 / 1072.8` | `1046.2 / 1051.6` | **1.03x** (`-36.2 us`) | `1055.9 / 1082.8` | `1041.0 / 1045.8` | **1.03x** (`-26.0 us`) | `True` |
| `lm_head` TP4 `(32320, 5120)` | `24` | `1134.8 / 1134.0` | `1078.0 / 1078.2` | **1.05x** (`-56.3 us`) | `1124.2 / 1126.9` | `1068.2 / 1068.2` | **1.05x** (`-57.4 us`) | `True` |
| `lm_head` TP4 `(32320, 5120)` | `32` | `1138.0 / 1143.6` | `1079.8 / 1078.8` | **1.06x** (`-61.5 us`) | `1130.5 / 1123.7` | `1067.6 / 1066.5` | **1.06x** (`-60.1 us`) | `True` |
| `lm_head` TP4 `(32320, 5120)` | `48` | `1218.2 / 1213.7` | `1147.0 / 1147.8` | **1.06x** (`-68.6 us`) | `1159.4 / 1166.9` | `1096.4 / 1097.0` | **1.06x** (`-66.5 us`) | `True` |
| `lm_head` TP4 `(32320, 5120)` | `64` | `1221.3 / 1219.9` | `1146.5 / 1147.9` | **1.06x** (`-73.4 us`) | `1170.5 / 1164.0` | `1099.4 / 1097.2` | **1.06x** (`-69.0 us`) | `True` |
| `lm_head` TP4 `(32320, 5120)` | `96` | `1517.8 / 1516.7` | `1240.9 / 1241.3` | **1.22x** (`-276.2 us`) | `1451.8 / 1455.2` | `1205.5 / 1207.8` | **1.20x** (`-246.9 us`) | `True` |
| `lm_head` TP4 `(32320, 5120)` | `107` | `1582.7 / 1584.9` | `1269.0 / 1272.2` | **1.25x** (`-313.2 us`) | `1516.4 / 1515.1` | `1221.4 / 1217.5` | **1.24x** (`-296.3 us`) | `True` |
| `lm_head` TP4 `(32320, 5120)` | `128` | `1773.6 / 1785.0` | `1327.0 / 1327.6` | **1.34x** (`-452.0 us`) | `1693.9 / 1697.8` | `1259.6 / 1261.2` | **1.35x** (`-435.5 us`) | `True` |
| `lm_head` TP4 `(32320, 5120)` | `512` | `7382.6 / 7406.0` | `4873.0 / 4870.9` | **1.52x** (`-2522.4 us`) | — | — | — | `True` |
| `lm_head` TP4 `(32320, 5120)` | `2048` | `30631.4 / 30678.5` | `20985.5 / 20979.5` | **1.46x** (`-9672.5 us`) | — | — | — | `True` |
| `lm_head` TP3 `(43096, 5120)` | `1` | `1422.4 / 1419.6` | `1389.0 / 1383.8` | **1.02x** (`-34.6 us`) | `1385.1 / 1421.7` | `1355.0 / 1372.4` | **1.03x** (`-39.7 us`) | `True` |
| `lm_head` TP3 `(43096, 5120)` | `6` | `1402.0 / 1403.5` | `1375.8 / 1377.7` | **1.02x** (`-26.0 us`) | `1394.5 / 1389.5` | `1364.8 / 1368.1` | **1.02x** (`-25.6 us`) | `True` |
| `lm_head` TP3 `(43096, 5120)` | `24` | `1496.5 / 1495.5` | `1446.5 / 1445.0` | **1.03x** (`-50.3 us`) | `1485.0 / 1492.4` | `1433.9 / 1435.6` | **1.04x** (`-54.0 us`) | `True` |
| `lm_head` TP3 `(43096, 5120)` | `48` | `1609.0 / 1609.9` | `1550.9 / 1551.4` | **1.04x** (`-58.3 us`) | `1536.0 / 1538.4` | `1479.8 / 1480.8` | **1.04x** (`-56.9 us`) | `True` |
| `lm_head` TP3 `(43096, 5120)` | `96` | `1982.3 / 1986.2` | `1654.4 / 1656.8` | **1.20x** (`-328.7 us`) | `1904.6 / 1906.8` | `1582.5 / 1581.5` | **1.20x** (`-323.7 us`) | `True` |
| `lm_head` TP3 `(43096, 5120)` | `107` | `2074.4 / 2069.5` | `1642.8 / 1647.5` | **1.26x** (`-426.8 us`) | `1991.5 / 1995.1` | `1577.9 / 1574.1` | **1.26x** (`-417.3 us`) | `True` |
| `lm_head` TP3 `(43096, 5120)` | `128` | `2368.1 / 2368.2` | `1680.8 / 1680.2` | **1.41x** (`-687.7 us`) | `2267.9 / 2266.2` | `1593.5 / 1595.4` | **1.42x** (`-672.6 us`) | `True` |
| `lm_head` TP3 `(43096, 5120)` | `512` | `9899.6 / 9896.8` | `6208.5 / 6209.2` | **1.59x** (`-3689.4 us`) | — | — | — | `True` |
| `lm_head` TP3 `(43096, 5120)` | `2048` | `40233.7 / 40176.2` | `26201.3 / 26196.7` | **1.53x** (`-14006.0 us`) | — | — | — | `True` |

#### B. At Stressed Exception Density (`0.33%`, matching compressor/indexer exception density, recorded in `receipts/boot-1.json` & `receipts/boot-2.json`)

- **TP4 `(32320, 5120)`**: `2.32x–3.46x` speedup (`M=1: 2635.7 -> 1127.6 us`, `M=6: 3014.8 -> 1112.9 us`, `M=48: 4939.3 -> 1553.9 us`, `M=96: 8491.9 -> 2455.5 us`, `M=2048: 180.59 ms -> 63.68 ms`), `exact_0_ulp = True`.
- **TP3 `(43096, 5120)`**: `2.27x–3.50x` speedup (`M=1: 3468.0 -> 1501.4 us`, `M=6: 3964.6 -> 1478.6 us`, `M=48: 6522.0 -> 2038.0 us`, `M=96: 11381.1 -> 3251.0 us`, `M=2048: 209.74 ms -> 76.59 ms`), `exact_0_ulp = True`.

---

## Verification & Receipts

- **Unit & PDL Source Tests (2 Container Boots on `sp7`)**:
  - `receipts/pytest-boot1.txt`: `274 passed, 0 failed` (`PDL sources: 449 kernels launched, 242 with PDL, 0 violations`) across `test_deepseek_v41_tilelang_vocab.py`, `test_deepseek_v41_tilelang_linear.py`, `test_deepseek_v41_tilelang_embedding.py`, and `test_deepseek_v41_tilelang_pdl.py`.
  - `receipts/pytest-boot2.txt`: second container boot verification pass.
- **Two-Boot Benchmark Receipts**:
  - `receipts/boot-1.json` and `receipts/boot-2.json` generated by `screen_pack12_kernels.py`.
- **Configuration & Patch Checks**:
  - `bin/spark doctor`, `bin/spark --cluster-config experiments/2026-11-08-pack12-unconditional-loads-swar/tp3.json doctor`, `bin/spark --cluster-config experiments/2026-11-08-pack12-unconditional-loads-swar/tp4.json doctor`, and `git diff --check` all pass cleanly.
