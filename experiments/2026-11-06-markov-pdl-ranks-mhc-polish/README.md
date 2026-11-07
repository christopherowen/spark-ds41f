# 2026-11-06: Markov Greedy Warp-Shuffle Epilogue, PDL Gather Chain, Prompt Ranks Barrier Fusion & SP/TP mHC/Engram Rank-Major Boundary Fusion (`0744..0747`)

## Base & Environment

- **Deployment Repository Base**: `239fad997eac7f42bd9a3f001a6958b29080a2cf` (`1107` with `0001..0743`)
- **vLLM Base (`0743` head)**: `a75357bba81e31e4f3085700f94b6a0b29aed44a` (tree `45757842cce62c7da54f3873918839866e8c040d`)
- **vLLM Candidate (`0747` head)**: `136d075ddadc12fa2c9b6cf69a058dccb7212f29` (tree `b1ed082358c89d07facc26990ccd9b24dc1f2ee6`, branch `feat/markov-pdl-ranks-mhc-polish`)
- **Hardware & Container**: `sp8` (`NVIDIA GB10`, SM121a), container image `spark-ds41f:1107`, 2 independent container boots (`boot1.json`, `boot2.json`, `summary_2boot.json`)

## Patch Series (`0744..0747`)

1. `0744-perf-dsv41-warp-shuffle-vocab-greedy-epilogue-and-L2.patch` (`a5264e68f`)
   - **Option B (Warp-Shuffle MMA Epilogue)**: Replaces full `[block_M, 64]` shared-memory staging (`C_shared`) and block-wide `T.sync_threads()` in `bf16_vocab_markov_greedy_partials`, `packed_vocab_markov_greedy_partials`, `packed_vocab_greedy_partials`, and `nvfp4_vocab_greedy_partials` with an in-register warp-shuffle (`__shfl_xor_sync` across MMA fragment lanes `offset = 2, 1`) reduction (`vllm_vocab_tile_argmax_clocal`) that writes only per-warp winners (`[block_M, WARPS_N]`) to shared memory and reduces them on `tx < block_M`. Reduces `bf16_vocab_markov_greedy_partials` register count from **126 to 43 registers**.
   - **Option A (Step-Aware L2-Hot Dispatch + Scratch Reuse)**: Adds `_use_fused_markov_greedy_step(tp_size, step, num_reqs)` in `DSparkSpeculator._sample_sequential` and persistent output buffer reuse in `markov_bias_local` / `vocab.project_head(..., out=...)` so L2-cold `pos == 0` (`M >= 3`) uses the fused kernel while L2-hot `pos >= 1` (and `M <= 2` on `TP > 1`) dispatches to unfused `markov_bias_local + greedy_markov_shard_pair`.
2. `0745-perf-dsv41-add-PDL-chain-to-gather_columns_kernel-be.patch` (`c59c74a88`)
   - Adds `pdl.chain()` (`T.pdl_sync()` / `T.pdl_trigger()`) to `gather_columns_kernel` (`vllm/models/deepseek_v4_1/tilelang/gather.py`) and registers `"gather_columns_into"` in `CHAIN` in `tests/kernels/deepseek_v41_pdl_sources.py` so `compress_kernel` overlaps its pre-wait static weight and slot-metadata preload with `gather_columns_into`.
3. `0746-perf-sample-broadcast-target-logit-and-fuse-barriers.patch` (`acba70056`)
   - In `topk_log_softmax_kernel(..., include_ranks=True)` (`vllm/v1/worker/gpu/sample/tilelang_logprob.py`): loads `selected_rank_shared[0]` once on thread 0 overlapped with the initial global vector read and broadcasts it at the first barrier before `fold_max`; accumulates `rank_count` inside `fold_max` and warp-reduces `rank_count` into `rank_partial` before `fold_max`'s block barrier; and fuses `se + ties` block reduction into `_row_reduce_pair` so `include_ranks=True` uses **3 `T.sync_threads()` barriers total** (zero extra block barriers compared to `include_ranks=False`). Also hoists `if write_scores:` before the vocabulary scan loop in standalone `ranks_kernel`.
4. `0747-perf-dsv41-fuse-rank-major-shards-in-mhc.post-and-en.patch` (`136d075dd`)
   - Extends `mhc.post` (`post_streams`) and `engram_post_gate` / `engram_post_gate_blocked` (`_reconstruct_stream`) with a compile-time `world` dimension and `T.StridedTensor` input declarations so SP prefill `RankOrderParts` (`[world, rows, hidden]` BF16) are summed in FP32 directly inside the consumer kernel without launching `previous_output.sum()`.
   - In `Engram.forward` and `Engram.declare_kernels` on non-SP TP steps (`sp_rows is None and tp_size > 1`), all-gathers `self.wkv.local_product(rows)` along `dim=0` (`[tp_size, num_tokens, block_cols]`) and feeds it directly to `engram_post_gate_blocked` / `engram_gate_blocked`, eliminating `concat_rank_shards` and `.contiguous()` slice copies.

## 2-Boot Correctness & PDL Verification

- **Boot 1**: 9 pytest suites (`624` GPU tests passed, `0` failed, `0` PDL violations).
- **Boot 2**: 5 targeted pytest suites (`356` GPU tests passed, `0` failed, `0` PDL violations).
- **Bit-Identity (`torch.equal`)**: Verified across all 4 interventions on both container boots.

## 2-Boot Latency Summary (`[boot1, boot2]`)

### Item 1: DSpark Markov Greedy Projection (12-Bit Packed Production Head)

| Shape | `M` | `1107` `hot/step` (µs) | `optB` `hot/step` (µs) | `optAB` `hot/step` (µs) | `1107` 5-Step (`cold0+4*hot`, µs) | `optAB` 5-Step (`cold0+4*hot`, µs) | `optAB` vs `1107` 5-Step (µs / %) |
|---|---:|---|---|---|---|---|---|
| TP1 (`129280x256`) | 4 | `[244.28, 241.80]` | `[241.09, 237.98]` | `[241.09, 237.98]` | `[1255.14, 1244.97]` | `[1240.32, 1229.06]` | `[-14.82, -15.91]` (`[-1.2%, -1.3%]`) |
| TP1 (`129280x256`) | 16 | `[247.34, 245.98]` | `[246.83, 245.41]` | `[246.83, 245.41]` | `[1270.06, 1261.26]` | `[1265.17, 1258.59]` | `[-4.89, -2.67]` (`[-0.4%, -0.2%]`) |
| TP3 (`43136x256`) | 1 | `[22.17, 21.85]` | `[21.22, 20.66]` | `[19.68, 19.07]` | `[216.19, 215.15]` | `[209.13, 206.61]` | `[-7.06, -8.54]` (`[-3.3%, -4.0%]`) |
| TP3 (`43136x256`) | 4 | `[27.58, 26.86]` | `[25.00, 24.47]` | `[23.43, 23.09]` | `[235.15, 232.33]` | `[225.73, 222.84]` | `[-9.42, -9.49]` (`[-4.0%, -4.1%]`) |
| TP3 (`43136x256`) | 8 | `[27.79, 27.60]` | `[25.63, 25.08]` | `[24.71, 23.90]` | `[236.46, 235.69]` | `[229.97, 226.44]` | `[-6.49, -9.25]` (`[-2.7%, -3.9%]`) |
| TP3 (`43136x256`) | 12 | `[28.76, 28.43]` | `[25.99, 25.69]` | `[25.81, 24.79]` | `[248.93, 247.47]` | `[240.52, 234.59]` | `[-8.41, -12.88]` (`[-3.4%, -5.2%]`) |
| TP3 (`43136x256`) | 16 | `[29.84, 29.60]` | `[26.60, 26.36]` | `[27.24, 26.18]` | `[257.34, 259.84]` | `[250.07, 246.27]` | `[-7.27, -13.57]` (`[-2.8%, -5.2%]`) |
| TP4 (`32320x256`) | 1 | `[16.71, 16.92]` | `[15.86, 15.96]` | `[14.59, 14.37]` | `[165.41, 166.26]` | `[159.70, 158.62]` | `[-5.71, -7.64]` (`[-3.5%, -4.6%]`) |
| TP4 (`32320x256`) | 4 | `[20.69, 20.63]` | `[18.69, 18.70]` | `[17.56, 17.57]` | `[179.64, 179.13]` | `[169.37, 169.51]` | `[-10.27, -9.62]` (`[-5.7%, -5.4%]`) |
| TP4 (`32320x256`) | 8 | `[21.13, 21.17]` | `[19.16, 19.14]` | `[18.54, 18.25]` | `[181.66, 181.91]` | `[173.49, 172.37]` | `[-8.17, -9.54]` (`[-4.5%, -5.2%]`) |
| TP4 (`32320x256`) | 12 | `[22.14, 22.84]` | `[19.53, 19.63]` | `[19.37, 19.34]` | `[191.93, 198.90]` | `[180.93, 182.03]` | `[-11.00, -16.87]` (`[-5.7%, -8.5%]`) |
| TP4 (`32320x256`) | 16 | `[22.49, 22.50]` | `[19.68, 18.70]` | `[20.24, 19.72]` | `[199.15, 202.20]` | `[186.75, 187.98]` | `[-12.40, -14.22]` (`[-6.2%, -7.0%]`) |

### Item 2: PDL-Chained `gather_columns_into -> compress_kernel`

| Config | `M=1` (`1107` -> `PDL`, %) | `M=4` (`1107` -> `PDL`, %) | `M=16` (`1107` -> `PDL`, %) | `M=64` (`1107` -> `PDL`, %) | `M=128` (`1107` -> `PDL`, %) |
|---|---|---|---|---|---|
| `tp3_ratio1` | `[-22.55%, -23.26%]` | `[-24.08%, -24.16%]` | `[-25.52%, -24.79%]` | `[-19.85%, -25.51%]` | `[-22.90%, -21.95%]` |
| `tp3_ratio2` | `[-8.85%, -8.62%]` | `[-7.53%, -8.16%]` | `[-7.98%, -8.35%]` | `[-6.75%, -6.99%]` | `[-5.07%, -4.85%]` |
| `tp4_ratio1` | `[-23.62%, -23.09%]` | `[-23.75%, -23.81%]` | `[-24.44%, -24.32%]` | `[-21.22%, -21.26%]` | `[-20.47%, -21.04%]` |
| `tp4_ratio2` | `[-8.22%, -8.10%]` | `[-7.97%, -8.00%]` | `[-7.39%, -7.75%]` | `[-6.26%, -6.16%]` | `[-5.43%, -5.45%]` |

### Item 3: Fused Prompt Scores + Ranks (`topk_log_softmax_kernel(..., include_ranks=True)`)

| PR `#912` End-to-End Prompt Case | Reference (µs) | Candidate 3-Barrier (µs) | 2-Boot Speedup (%) |
|---|---|---|---|
| `prompt-1025-8193-0-raw_logits` | `[481.20, 444.27]` | `[448.26, 431.11]` | `[-6.85%, -2.96%]` |
| `prompt-1025-8193-0-raw_logprobs` | `[730.94, 725.92]` | `[425.57, 424.53]` | `[-41.78%, -41.52%]` |
| `prompt-1025-8193-5-raw_logits` | `[438.94, 439.97]` | `[433.42, 429.73]` | `[-1.26%, -2.33%]` |
| `prompt-1025-8193-5-raw_logprobs` | `[688.85, 681.55]` | `[533.86, 533.04]` | `[-22.50%, -21.79%]` |
| `prompt-2049-8193-0-raw_logits` | `[770.64, 773.76]` | `[738.13, 738.62]` | `[-4.22%, -4.54%]` |
| `prompt-2049-8193-0-raw_logprobs` | `[1157.87, 1207.89]` | `[741.36, 740.70]` | `[-35.97%, -38.68%]` |
| `prompt-2049-8193-5-raw_logits` | `[752.78, 750.34]` | `[744.22, 743.47]` | `[-1.14%, -0.91%]` |
| `prompt-2049-8193-5-raw_logprobs` | `[1192.42, 1197.49]` | `[937.28, 936.43]` | `[-21.40%, -21.80%]` |
| `prompt-1025-129280-5-raw_logits` | `[4544.78, 4551.17]` | `[4537.36, 4543.28]` | `[-0.16%, -0.17%]` |
| `prompt-1025-129280-5-raw_logprobs` | `[7284.80, 7305.52]` | `[4935.47, 4937.30]` | `[-32.25%, -32.42%]` |

### Item 4: SP Prefill & Non-SP TP Rank-Major Boundary Fusion (`#927`)

- **SP Prefill (`RankOrderParts`, `rows = 228..1366`)**:
  - `mhc.post`: TP3 **`[-9.13%, -10.48%]` to `[-26.50%, -29.64%]`** (`[98.64, 100.74] -> [72.50, 70.88] µs` at `rows=228`); TP4 **`[-8.48%, -9.49%]` to `[-30.07%, -23.15%]`** (`[120.30, 118.52] -> [84.12, 91.09] µs` at `rows=228`).
  - `engram_post_gate`: TP3 **`[-7.30%, -10.50%]` to `[-10.66%, -10.17%]`**; TP4 **`[-6.35%, -7.24%]` to `[-9.91%, -8.72%]`**.
  - `engram_post_gate_blocked`: TP3 **`[-8.36%, -8.28%]` to `[-10.98%, -11.39%]`**; TP4 **`[-7.14%, -7.06%]` to `[-9.46%, -9.59%]`**.
- **Non-SP TP Engram `dim=0` All-Gather + `engram_post_gate_blocked` (`rows = 1..512`)**:
  - TP3: `rows=1` **`[-10.89%, -9.45%]`**, `rows=6` **`[-38.39%, -39.20%]`**, `rows=16` **`[-44.06%, -43.01%]`**, `rows=48` **`[-36.45%, -40.66%]`**, `rows=128` **`[-79.41%, -77.04%]`** (`[147.32, 136.00] -> [30.33, 31.22] µs`), `rows=512` **`[-54.48%, -56.80%]`** (`[723.66, 759.50] -> [329.42, 328.07] µs`).
  - TP4: `rows=1` **`[-12.81%, -12.61%]`**, `rows=6` **`[-15.78%, -15.16%]`**, `rows=16` **`[-21.45%, -20.10%]`**, `rows=48` **`[-11.69%, -11.65%]`**, `rows=128` **`[-60.42%, -58.32%]`** (`[75.18, 72.45] -> [29.75, 30.20] µs`), `rows=512` **`[-34.78%, -35.17%]`** (`[499.25, 502.77] -> [325.60, 325.92] µs`).
