# 2026-11-16-c60-prefill (candidate-60)

- **Base commit**: `de66d013e` (`candidate-59` + `#1218`), also rebased past `#1219` (`1ed6de867`) at Step 2
- **Branch**: `c60-prefill` (final overlay commit `5fc2363bc`)
- **Cluster profiles**: `tp4-switched.json` / `tp4-switched-dev.json` (`sp1..sp4`), `tp3-switched.json` / `tp3-switched-dev.json` (`sp5..sp7`) (and matching `<shape>.json` / `<shape>-dev.json` configs extending `experiments/2026-10-26-spark-ds41f-candidate/candidate-55/<shape>.json` on `spark-ds41f:candidate-55`, `VLLM_DS41_LSA=0`)
- **Goal**: Stepwise prefill, prompt-cache, Engram-shield, startup, and tokenization improvements over `candidate-59` with zero decode, determinism, or quality regressions.

## Step 1 (`3b75644ba`) — PR `#992` `bf16x2` Producer Casts, BF16 Indexer SMEM / Warp-Scan Top-$k$, and SP Compressor Overlap

### Changes
1. **PR `#992` (`producer-mxfp8-bf16x2-cast`)**:
   - `overlay/vllm/vllm/models/deepseek_v4_1/tilelang/{activation.py,columns.py,front.py,gemm.py,mhc.py,norm.py,quant_common.py}`.
2. **`indexer.py` (`overlay/vllm/vllm/models/deepseek_v4_1/tilelang/indexer.py`)**:
   - Stored `contrib` pairwise row partials as packed `bfloat16_t` (`__float22bfloat162_rn` into `contrib_bf16`) in `__tl_score_rows_dense_fast` (`rows >= 64`) and `__tl_score_row_candidates_fast`, and removed redundant `__syncthreads()` barriers.
   - Replaced `_select_macros.block_scan` serial shared-memory Hillis-Steele scan with intra-warp `T.shfl_up` + warp-spine prefix scan (`warp_offset=256` when `threads >= 512`, and `warp_sums=hist` shared across `compact` in `indexer_topk_segment_v41` and `indexer_topk_merge_v41`).
3. **`attention.py` (`overlay/vllm/vllm/models/deepseek_v4_1/attention.py`)**:
   - Launched SP prefill compressor projection (`project_sp_local` + `gather_many_async`) before `sp_rows.gather_mxfp8_async` to overlap with `indexer.weights_proj` (`wait_comp()` before compressor consumption).
   - Deferred `write_swa_records` into a `finish_front` closure waited after indexer weight preparation, and gated persistent KV caching with `cacheable = index_rows <= chunk_rows`.

### Verification (`sp8`, GB10 `sm_121a`, `spark-ds41f:candidate-55` + final `5fc2363bc` overlay)
- `bench_indexer_step1.py` (`27` tensors checked vs `c59`: **`0` diffs**, bit-exact):
  - `dense_r1_32k_total`: `32.83 -> 33.39 us` (`0.983x`); `dense_r16_32k_total`: `110.28 -> 75.29 us` (**`1.465x`**); `dense_r256_32k_total`: `735.80 -> 569.63 us` (**`1.292x`**); `dense_r256_131k_total`: `2867.08 -> 2155.68 us` (**`1.330x`**).
  - `cand_r16_score`: `32.78 -> 27.18 us` (**`1.206x`**); `cand_r256_score`: `485.82 -> 392.44 us` (**`1.238x`**); `cand_r256_total`: `545.36 -> 430.48 us` (**`1.267x`**).
  - `topk_32768_g1`: `91.46 -> 81.78 us` (**`1.118x`**); `topk_32768_g8`: `34.97 -> 30.63 us` (**`1.142x`**); `topk_131072_g1`: `435.30 -> 403.39 us` (**`1.079x`**); `topk_262144_g1`: `828.64 -> 773.00 us` (**`1.072x`**); `topk_262144_g8`: `670.02 -> 664.66 us` (`1.008x`).
- `bench_pr992_step1.py` (CUDA graph): `act_swiglu_m256` `2.73 -> 2.53 us` (`1.083x`), `mhc_finalize_m256` `10.82 -> 10.47 us` (`1.033x`), `cast_mxfp8_m256` `2.28 -> 2.25 us` (`1.012x`), `front_m256` `3.13 -> 3.10 us` (`1.008x`), `act_swiglu_m1024` `6.03 -> 5.80 us` (`1.040x`), `norm_cast_m1024` `216.78 -> 215.27 us` (`1.007x`).

### Step 1 Cluster Gate (`sp1`, `tp4-switched`, commit `3b75644ba`)
- Receipts archived in `receipts/step1/tp4-switched/`.
- **`screen` (`20261116T090746-screen-tp4-switched`, 2 boots `c60`, `c60-again` vs `c59` recorded)**:
  - `model` (`c60 -> c60-again`): `128/128` greedy identical; `100.00%` identical distributions over `32,768` positions (`128` sequences), `KL mean 0.000e+00`, perplexity `1.31888 -> 1.31888 (+0.000%)`.
  - `grid`: `decode geometric mean (20)` `+0.05% ±0.13%` (`margin 0.27%`, repeat `+0.00%`, **`same`**).
  - `c16`: `output tok/s` `268.393 / 268.503` (`-0.14% ±0.84%`, **`same`**); `decode tok/s, paired` `-0.21% ±0.42%` (**`same`**).
  - `prefill` (`default-text`): `32,768` `9,426.21 / 9,446.29` (`+1.23% ±4.67%`, `same`); `262,144` `9,784.08 / 9,804.99` (**`+1.70% ±0.30%`, `better`**, margin `0.47%`).
- **`bench` (`20261116T090905-bench-prefill-tp4-switched`, `--prefill-text source`, 2 boots `b1`, `b2`)**:
  - `32,768`: `b1 = 9,551.38` (`9,532.81`..`9,569.95`, TTFT `3.042s`), `b2 = 9,592.23` (`9,588.05`..`9,596.41`, TTFT `3.029s`) -> mean **`9,571.81 tok/s`** (`c59` `9,548.45`).
  - `131,072`: `b1 = 10,020.42` (`9,970.22`..`10,070.62`, TTFT `11.510s`), `b2 = 9,979.79` (`9,928.28`..`10,031.29`, TTFT `11.557s`) -> mean **`10,000.10 tok/s`** (`+113.9 tok/s` vs `c59` `9,886.17`).
  - `262,144`: `b1 = 9,864.89` (`9,864.04`..`9,865.73`, TTFT `24.317s`), `b2 = 9,864.64` (`9,861.84`..`9,867.43`, TTFT `24.317s`) -> mean **`9,864.76 tok/s`** (`+107.5 tok/s` vs `c59` `9,757.28`).
  - Memory/thermals (`b1`/`b2`): `sp1` `min_available_gib` `17.25` / `17.15 GiB`, `sp2..sp4` `19.50–19.83 GiB`, `swap_growth_gib = 0.0`, `thermal_slowdown_ms = 0.0`, `power_cap_ms = 0.0`.

## Step 2 (`d63784647` + `783ab927b`) — Engram Prefill Shield (`#983`), Hybrid KV Retention (`#1010`), MM Fastpath (`#978`), and `PREFILL_MIX_CHUNK` (`#1150`)

### Changes
1. **PR `#983`**: Clock-Pro 2-bit Engram prefill shield and pinned two-plane scales (`SPARK_DS41F_ENGRAM_PIN_SCALES`) ported onto `c59`'s SoA reader (`overlay/vllm/vllm/models/deepseek_v4_1/engram_rows/{_reader.c,_row_plan.h,rows.py}`, `tests/kernels/test_deepseek_v41_engram_rows.py`).
2. **PR `#1010`**: hybrid KV retention and eviction (`overlay/vllm/vllm/v1/core/{kv_cache_coordinator.py,single_type_kv_cache_manager.py,sched/scheduler.py,sched/async_scheduler.py}`, `tests/v1/core/test_prefix_cache_retention_probe.py`).
3. **PR `#978`**: multimodal prefix-cache fastpath (`overlay/vllm/vllm/multimodal/{hasher.py,inputs.py,media/image.py,processing/processor.py}`, `renderers/base.py`, `v1/engine/{async_llm.py,core.py}`, `tests/multimodal/test_prefix_cache_fastpath.py`).
4. **PR `#1150` (evaluated, dropped in cleanup)**: `SPARK_DS41F_PREFILL_MIX_CHUNK` was tested during Step 2 and later removed (`scheduler.py`, `tests/v1/core/test_prefill_mix_chunk.py`) during optionality cleanup rather than promoted as a default-off knob.

### Failed Boot Record & Verification (`sp8`, `sp1`)
- **Initial boot failure (`20261116T094817-screen-tp4-switched`, `receipts/step2/tp4-switched/failed-start-20261116T095150Z.txt`)**: Commit `d63784647` omitted `Callable` from `from collections.abc import ...` in `async_llm.py`, raising `NameError: name 'Callable' is not defined` at engine startup; resolved in `783ab927b`.
- **`shield_bench.json` (`1,050,000` rows, `350,000` owned, `decode_m16` under prefill flood on `sp8`)**:
  - `hot_working_set`: `fifo_0750_exact_sim` hit rate fell to `0.1914` interleaved / `0.2344` (`m8`) and `0.4106` (`m16`) post-prefill; `two_plane_pin_scales_cache64k` preserved **`1.0000`** interleaved and **`1.0000`** post-prefill hit rates with `0.0` disk read calls/step (`31.37 us` interleaved vs `68.79 us` unpacked `packed_cache64k`; `20.02 us` post-prefill).
  - `zipf85_working_set`: interleaved hit `0.1465 -> 0.8604`, post-prefill hit `0.3071 -> 0.8562` (`86.18 us`).
  - `200K`-row `zipf m32` lookup benchmark: `two_plane_pin_scales_cache64k` vs unpinned `two_plane_cache64k` halved read calls/step (`32.0 -> 16.72`) and transfer bytes (`135.38 -> 67.25 KiB/step`), cutting mean latency `97.48 -> 84.44 us`.
- **`bench_step2_mm.py` (4×768×768 PNG Turn-2 repeat vs `c59`)**:
  - `turn2_load_bytes_4x768_ms`: `31.270 -> 1.656 ms` (**`18.88x`**).
  - `turn2_proc_apply_4x768_ms`: `54.025 -> 5.269 ms` (**`10.25x`**).
- **Step 2 Cluster Gate (`20261116T100535-screen-tp4-switched`, `receipts/step2/tp4-switched/`)**:
  - `model` (`c60 -> c60-again`): `128/128` greedy identical, `100.00%` identical distributions over `32,768` positions, `KL mean 0.000e+00`, perplexity `1.31888 -> 1.31888 (+0.000%)`.
  - `grid` (20 shapes): `+0.06% ±0.14%` (`margin 0.27%`, repeat `+0.01%`, **`same`**).
  - `c16`: `output tok/s` `268.457 / 268.607` (`-0.11% ±0.84%`, `same`); `decode tok/s, paired` `-0.13% ±0.32%` (`same`).
  - `prefill` (`default-text`): `32,768` `9,418.14 / 9,398.58` (`+0.93% ±4.79%`, `same`); `262,144` `9,769.20 / 9,747.21` (**`+1.32% ±0.53%`, `better`**, margin `0.70%`).

## Step 3 (`5fc2363bc`) — Post-Load GPU Vectorization & Warmup (`#1006`) + Piece-Level Parallel Tokenizer Cache

### Changes
1. **PR `#1006` GPU post-load vectorizations, pressure guard, lazy imports, and first-prefill warmup**:
   - `vocab.py`: direct in-place BF16 packing above `_PACK_SAFE_DIRECT_ROW = 3,072` (`pack_bf16_weight(in_place=True)`).
   - `scale_codes.py`, `moe.py`, `gemm.py`, `attention.py`: vectorized `encode_expert_scale_codes`, `int64` `tile_expert_weights`, and vectorized `pack_weight_tiles`/`unpack_weight_tiles`.
   - `vllm/model_executor/model_loader/utils.py`: `release_device_memory_under_pressure(target_device)` after `process_weights_after_loading` when reserved CUDA memory grows.
   - `vllm/models/deepseek_v41/__init__.py`, `vllm/models/deepseek_v4_1/nvidia/vl_model.py`, `vllm/models/deepseek_v41/nvidia/vl_model.py`, `vllm/tokenizers/registry.py`: lazy imports.
   - `vllm/models/deepseek_v4_1/nvidia/model.py`: `_warm_model_runtime` hook in `prepare.declare(..., warm=_warm_model_runtime)` pre-building `_build_l2_prefetch()` and `sp_prefill._side_stream()` before request #1.
   - **Negative result (`postload_cache` dropped)**: on-disk caching of packed BF16 weights and expert scale codes was tested during Step 3 staging and measured slower on GB10 unified memory than recomputing with PR `#1006`'s vectorized GPU kernels (`encode_expert_scale_codes`, `pack_bf16_weight`), so `postload_cache` was removed before qualification.
2. **Piece-level prompt tokenizer cache & parallel encoding (`overlay/vllm/vllm/tokenizers/deepseek_v41.py`, `tests/tokenizers_/test_deepseek_v41.py`)**:
   - Added `_BoundedPieceCache`, structural segment splitting (`_split_prompt_segments`), and BPE-invariant cut-after-`\n` parallel chunking (`_split_parallel_newline_chunks` with `encode_batch`) integrated into `_DeepseekV41Tokenizer.encode`, `__call__`, and `apply_chat_template`.

## Final Qualification (`candidate-60`, commit `5fc2363bc`, `receipts/final/`)

Note: `tp4-ring` (`dgx1..dgx4`) remained held by `tp2-direct-slots` and was not touched; qualification ran on `tp4-switched` (`sp1..sp4`) and `tp3-switched` (`sp5..sp7`).

### `tp4-switched` (`sp1..sp4`, `receipts/final/tp4-switched/`)

- **`screen-tp4-switched`** (`20261116T104515-screen-tp4-switched`, 2 boots `c60`, `c60-again` vs recorded `c59`):
  - `model` (`c60 -> c60-again`): `128/128` greedy identical; `100.00%` identical distributions over `32,768` positions (`128` sequences), `KL mean 0.000e+00`, perplexity `1.31888 -> 1.31888 (+0.000%)`.
  - `grid` (20 shapes, trimmed geomean): `c59 -> c60` `-0.12% ±0.13%` (`same`), `c59 -> c60-again` `-0.20% ±0.13%` (`better`); 2-boot aggregate **`-0.07% ±0.13%`** (`repeat -0.08%`, `margin 0.27%`, **`same`**).
  - `c16`: `output tok/s` `269.673 / 268.893` (mean `269.283`, **`+0.17% ±0.80%`**, `same` vs `c59` `268.777 / 268.877`); `decode tok/s, paired` **`+0.15% ±0.35%`** (`same`, window tok/s `+0.33% ±0.80%`).
  - `prefill` (`default-text` prose):
    - `32,768`: `9,412.37 / 9,458.52` (mean **`9,435.44 tok/s`**, `+1.22% ±4.69%`, `same` vs `c59` `9,371.86 / 9,272.09`; `+2.82%` / `+258.6 tok/s` vs `c58-final` `9,176.85`).
    - `262,144`: `9,817.79 / 9,848.57` (mean **`9,833.18 tok/s`**, **`+2.10% ±0.64%`, `better`**, margin `0.84%` vs `c59` `9,618.70 / 9,643.25`; **`+4.50%` / `+423.0 tok/s`** vs `c58-final` `9,410.19`).
  - Startup `min_available_gib`: `sp1 13.68 / 13.59`, `sp2..sp4 19.709–21.581 GiB`.
- **`gate-tp4-switched`** (`20261116T104518-gate-tp4-switched`, 2 boots):
  - All 8 determinism and greedy checks (`determinism`, `greedy`, `determinism2`, `greedy2` across Boot 1 & Boot 2): `identical: true`, `token_diffs: 0`, `logprob_diffs: 0`, matching `c58-final`/`c59` SHA-256 hashes (`18d61961...` / `d6568649...` / `0a2dacdd...` / `12fd8257...`).
  - Distinct-prompt throughput (`foreign_requests = 0`): `c16-distinct` `269.51 / 270.41` (**`269.96 tok/s`**), `c8-distinct` `201.44 / 200.06` (**`200.75 tok/s`**), `c4-distinct` `157.19 / 157.22` (**`157.21 tok/s`**).
- **`c1-decisions-tp4-switched`** (`20261116T104521-c1-decisions-tp4-switched`, 4 boots):
  - `83.03 / 82.81 / 82.87 / 82.93` -> mean **`82.91 tok/s`** (`35.74 steps/s`, `2.328 tok/step`, `ttft_p50 0.112–0.114 s`, `0` failed, `0` foreign requests; vs `c59` `82.94 tok/s`).
- **`bench` 2-boot `source` prefill** (`20261116T104524-bench-prefill-tp4-switched`, `c60-prefill-b1` / `b2`):
  - `32,768`: `9,484.52 / 9,556.93` (**`9,520.73 tok/s`**, TTFT `3.063 / 3.040s`).
  - `131,072`: `9,921.43 / 9,927.42` (**`9,924.43 tok/s`**, TTFT `11.625 / 11.617s`, `+38.3 tok/s` vs `c59` `9,886.17`).
  - `262,144`: `9,809.46 / 9,850.93` (**`9,830.20 tok/s`**, TTFT `24.454 / 24.351s`, **`+72.9 tok/s`** vs `c59` `9,757.28`).
  - Prefill memory/thermals: `sp1` `min_available_gib` `17.35` / `17.43 GiB`, `sp2..sp4` `19.64–19.93 GiB`, `swap_growth_gib = 0.0`; `max_gpu_c` up to `76.0 C` (`b1`) / `74.0 C` (`b2`), `thermal_slowdown_ms = 0.0`, `power_cap_ms = 0.0`.
- **`bench` 2-boot `quality,decode`** (`20261116T104527-bench-decode-tp4-switched`, `c60-b1` / `b2`, `5/5` quality pass both boots, `concurrency = [1, 2, 4, 8, 16]`):

| Case | Boot 1 (`tok/s`) | Boot 2 (`tok/s`) | 2-Boot Mean (`tok/s`) | 2-Boot Window (`tok/s`) | 2-Boot `step_ms` | vs `c59` |
|---|---:|---:|---:|---:|---:|---:|
| `prose-c1` | `79.17` | `79.30` | `79.23` | `83.48` | `26.73 ms` | `-0.04%` |
| `prose-c2` | `126.04` | `120.17` | `123.11` | `131.75` | `33.32 ms` | `+2.63%` |
| `prose-c4` | `241.53` | `255.71` | `248.62` | `268.77` | `31.53 ms` | `-4.17%` |
| `prose-c8` | `415.76` | `423.76` | `419.76` | `452.93` | `37.56 ms` | `-2.10%` |
| `prose-c16` | `646.65` | `599.57` | `623.11` | `668.04` | `49.50 ms` | `+3.54%` |
| `code-c1` | `98.45` | `98.01` | `98.23` | `105.26` | `29.64 ms` | `-0.10%` |
| `code-c2` | `160.42` | `167.17` | `163.80` | `179.47` | `33.50 ms` | `-1.33%` |
| `code-c4` | `269.88` | `268.94` | `269.41` | `294.16` | `41.50 ms` | `-6.54%` |
| `code-c8` | `556.07` | `491.15` | `523.61` | `579.85` | `41.27 ms` | `+8.78%` |
| `code-c16` | `784.13` | `763.46` | `773.80` | `849.95` | `52.62 ms` | `-1.99%` |

- Decode memory/thermals (`c60-b1` / `b2`): `sp1` `min_available_gib` `21.71` / `21.72 GiB`, `sp2..sp4` `24.00–24.08 GiB`, `swap_growth_gib = 0.0`, `max_gpu_c` up to `65.0 C`, `thermal_slowdown_ms = 0.0`, `power_cap_ms = 0.0`.

### `tp3-switched` (`sp5..sp7`, `receipts/final/tp3-switched/`)

- **`screen-tp3-switched`** (`20261116T104533-screen-tp3-switched`, 2 boots vs recorded `c59`):
  - `model` (`c60 -> c60-again`): `128/128` greedy identical; `100.00%` identical distributions over `32,768` positions (`128` sequences), `KL mean 0.000e+00`, perplexity `1.31608 -> 1.31608 (+0.000%)`.
  - `grid` (15 shapes, trimmed geomean): `c59 -> c60` `-0.25% ±0.11%` (`better`), `c59 -> c60-again` `-0.19% ±0.12%` (`better`); 2-boot aggregate **`-0.03% ±0.12%`** (`repeat +0.07%`, `floor 0.30%`, `margin 0.33%`, **`same`**).
  - `c8`: `output tok/s` `158.727 / 158.520` (`-0.25% ±1.37%`, **`same`**); `decode tok/s, paired` **`-0.23% ±0.43%`** (`same`).
  - Startup `min_available_gib`: `sp5 5.965 / 5.898 GiB`, `sp6 7.755 / 7.755 GiB`, `sp7 7.458 / 7.516 GiB` (well above the `5 GiB` startup guard).
- **`gate-tp3-switched`** (`20261116T104536-gate-tp3-switched`, 2 boots):
  - All 8 determinism and greedy checks (`identical: true`, `token_diffs: 0`, `logprob_diffs: 0`), matching `c58-final`/`c59` SHA-256 hashes (`5c22ce41...` / `bce6092e...` / `26a739a9...` / `12cfee9b...`).
  - Distinct-prompt throughput (`foreign_requests = 0`): `c8-distinct` `153.59 / 154.77` (**`154.18 tok/s`**), `c4-distinct` `117.56 / 118.88` (**`118.22 tok/s`**).

## Conclusion

- `candidate-60` passed every Step 1, Step 2, and final qualification gate on both `tp4-switched` (`sp1..sp4`) and `tp3-switched` (`sp5..sp7`), catching and fixing the Step 2 `async_llm.py` `Callable` import omission and dropping `postload_cache` where direct GPU vectorization ran faster on GB10.
- Cumulative `c60` prefill throughput reached **`9,833.18 tok/s`** (`+2.10% ±0.64%` **`better`** over `c59`; `+4.50%` / `+423.0 tok/s` over `c58-final`) at `262K` `default-text` prose and **`9,924.43 / 9,830.20 tok/s`** at `131K / 262K` `source` prefill (after hitting **`10,000.10 tok/s`** mean at `131K` `source` in Step 1) with **zero decode, determinism, or quality regressions**.

## Post-Qualification Phase 1 Additions, Lever Measurements & Hardening

### Phase 1 PRs, Tool Discovery, and Config Cleanliness
- Incorporated **PR `#1212`** (rebased `#1047`, literal image-token escaping in `vllm/tokenizers/deepseek_v41.py`) and verified **PR `#1013` Part 1** (`StructuredOutputManager._get_constraint_start` and `DeepseekV41ModelState.prepare_attn` `ubatch_idx=0`) is already present in `spark-ds41f:candidate-55`.
- Implemented inline deferred tool-schema discovery (`defer_loading: true` + `tool_reference` inside `tool_result` in `vllm/entrypoints/anthropic/serving.py`, `vllm/tokenizers/deepseek_v41.py`, `vllm/tokenizers/deepseek_v41_encoding.py`, tested in `test_anthropic_messages_conversion.py` and `test_deepseek_v41.py`) so mid-session tool discovery preserves top-of-prompt KV prefixes.
- Removed default-off `SPARK_DS41F_PREFILL_MIX_CHUNK` (`scheduler.py`), simplified `can_gather_mxfp8_async` (`attention.py`, `kernels.py`), preserved `TP2` (`scripts/topology.py`, `tests/test_topology.py`, `tests/test_switched_topology.py`) and all 9 switched/triangle profiles (`tp3`, `tp3-switched`, `tp3-triangle-4k`, `tp4`, `tp4-switched..tp8-switched`), and kept one shape-independent default in `l2_prefetch.py` with shape overrides in config.

### Prefill Lever Screenings (`sp8`, GB10)
- **Indexer two-pass / head-subset CS (`IndexerV41Plan`, `rows=4096`, `page_size=256`, `topk=512`)**:
  - `131K`: `score = 30.18 ms` (145.7 TFLOPs), `select_blocks = 4.62 ms`, `gather_select = 1.68 ms`, `plain_select = 6.06 ms`.
  - `262K`: `score = 58.20 ms`, `select_blocks = 9.87 ms`, `gather_select = 1.68 ms`, `plain_select = 12.51 ms`.
  - `524K`: `score = 113.70 ms` (154.7 TFLOPs), `select_blocks = 19.83 ms`, `gather_select = 1.68 ms`, `plain_select = 25.04 ms`.
  - `1M`: `score = 277.43 ms`, `select_blocks = 39.32 ms`, `gather_select = 1.71 ms`, `plain_select = 49.86 ms`.
  - Only encoder layers `[2, 8, 14]` execute full-chunk dense scoring; checkpoint inspection (`weights_proj + wq_b`) showed uniform head importance (`0.341..0.363` top-8/32, `0.613..0.621` top-16/32), so head-subset bounds cannot skip significant compute -> skipped.
- **`wq_b` split proxy (`M=16384` BF16)**: `1.132 ms` single vs `1.134 ms` `4096 + 12288` -> skipped; SP `defer_q` (`_sp_front`/`finish_front`) and `_warm_model_runtime` already cover front overlap and warmup without global `CUDA_MODULE_LOADING=EAGER`.
- **BPE boundary alignment**: skipped runtime prefix-token rewriting because mutating prefix spelling shifts deterministic logprobs/sampling; retained `_BoundedPieceCache`.

### Code Hardening & Quality / Sanitizer Verification (`sp8` & host)
- Hardened `sp_prefill.py` (`_record_stream` guard + `try..finally: main.wait_stream(side)` across `project_exchange_rows`, `project_reduce_scatter`, `_project_exchange`), `engram_rows/rows.py` (`__init__` cleanup on failure, `0 <= count <= max_lookups` & `ids_flat.numel() >= count` bounds in `stage()`, idempotent `close()` with `torch.cuda.is_initialized()`), `deepseek_v41.py` / `deepseek_v41_encoding.py` (`Mapping` check + `None` text guards), and fixed `test_deepseek_v41_engram_rows.py:1040` (`pin_scales=False`) and `test_prefix_cache_retention_probe.py` (`long_prefill_token_threshold=4096`, `VLLM_SINK_REJECTION=0`).
- **vLLM pytest suites on `sp8`**:
  - 6-file overlay suite: **579 passed, 8 deselected** (`@pytest.mark.asyncio` per `config/release-tests.json`).
  - 12-file IKET / PDL / confidence / startup / vocab / MoE / attention / stop-strings / fastpath / IPC / retention / DSpark suite: **1,851 passed, 7 skipped, 1 xfailed** (on all non-retention files) + **1,464 passed, 0 skipped, 0 failed** in `test_prefix_cache_retention_probe.py` (`1,415` non-scheduler + `49` scheduler cases under `-p deepseek_models`); **1,942+ passed** across tokenizer + SP prefill + Step 4 regression tests (`2/2` new tests passed).
- **NVIDIA `compute-sanitizer` on `sp8`** (mounting `/usr/local/cuda-13.0/compute-sanitizer`):
  - `test_indexer_reads_only_visible_rows` (`initcheck` + `PYTORCH_NO_CUDA_MEMORY_CACHING=1`): **1 passed, 0 errors**.
  - `memcheck`: **11/11 kernel tests passed, 0 errors**.
  - `synccheck`: **11/11 kernel tests passed, 0 errors**.
  - `initcheck` (all 11 kernel tests): **11 passed**; 18 `initcheck` errors reported on `test_moe_experts[idx_dtype0-1-384-576]` (`moe_grouped_fp8_fp4_gemm_kernel+0x28f0`), verified as reading unused bytes `18..19` of the `uint32`-aligned `h_sf` workspace row (`sf_width = 20` vs `18` written scale bytes at `intermediate=576`, ignored by 64-col tail MMA `ki = 0, 1`).
  - `racecheck`: **0 errors** across indexer, top-$k$, sparse MLA, attention front, `moe_swiglu_cast`, and Engram; flagged `cp.async.mbarrier.arrive.noinc` in `moe_grouped_fp8_fp4_gemm_kernel` (verified `block_M=16` `T.sync_warp()` accesses only warp-local `SFB_shared[w*32..w*32+31]`).
- **Host C Sanitizers (`ASan` + `UBSan`) on `sp8`**: compiled `engram_rows/_reader.c` via `CC="gcc -fsanitize=address,undefined -fno-omit-frame-pointer"` into `VLLM_NATIVE_CACHE_DIR` and ran targeted `test_deepseek_v41_engram_rows.py` suite (Clock-Pro shield, `pin_scales=True/False`, packed columns, bounds/close) under `libasan.so` + `libubsan.so`: **14 passed, 0 errors**.
