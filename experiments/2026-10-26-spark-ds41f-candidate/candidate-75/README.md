# `candidate-75` — DFly2 block speculative decoding (`AngelSlim/DS-V4.1-Flash-DFly2-B6`)

## Base and single variable

- **Base deployment commit**: `890f8654bb` (`spark-ds41f-1126`, promoted baseline `2026-11-26-spark-ds41f-1126`).
- **Base vLLM series**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` + `candidate-74` (`0001..0869`, 742 patches, patch head `d8784b841c360f523ec20e44bf8fb0898f36d3e4`, tree `f3783b0ec61851ba481f6a3a316d0a76d0e171d8`).
- **Candidate vLLM series**: `experiments/2026-10-26-spark-ds41f-candidate/candidate-75/vllm/series` (`0001..0870`, 743 patches):
  - **Build 1 (`spark-ds41f:candidate-75`, `a422fd193b`)**: patch head `58422e4d0f76627ceca1fd4072154c97d39e1baa`, tree `1847ed52251a050c289cc780f8fa7509ccaa7f77`, patchset SHA-256 `3df5afb5e785964aaf8caf9c40f112292cbdf526af098b5ed2d91bab50fc4163`, OCI digest `sha256:734c15d77c58d02e3accfb5af15e44893fa13ea1fcd3796e48f943886db73011` (`dgx4`, `475 s`, `lowest MemAvailable 81.7 GiB`, distributed to `dgx1..dgx3`).
  - **Build 2 (`spark-ds41f:candidate-75-r2`, `8d0a48ea53`)**: patch head `8cc34082513b7e0225c0a42d4a963b721403bacf`, tree `f740b9860ca616f193c1fbed0ba6fe3f53bf100f`, patchset SHA-256 `c57c48f8811c77acba9c0ba6f9cf0a4304a151d2e4cd4172cd836859c31ecd65`, OCI digest `sha256:2eca750da593795cb940dd25207b70fb50ffa71cb77657ee3d185d2feb4c3d4c` (`dgx4`, `459 s`, `lowest MemAvailable 82.7 GiB`).
- **Intended variable**: replace the built-in `DSpark` drafter with Tencent Hunyuan AI Infra's standalone **DFly2** checkpoint (`AngelSlim/DS-V4.1-Flash-DFly2-B6`, Apache-2.0, `arXiv:2611.34776`), keeping target weights (`DeepSeek-V4.1-Flash-1017`), GLP-27 steering, kernel backend (`tilelang`), sparknet `v0.6.0` transports, KV pool bytes, stream count, and adaptive-verification curves unchanged per shape.

## Checkpoint provenance and fleet distribution (`dgx1..dgx4`)

- **Repository**: `https://huggingface.co/AngelSlim/DS-V4.1-Flash-DFly2-B6`
- **Pinned revision**: `d08e55a6897b1974307ea9771038838de5986e66`
- **Host cache path on `dgx1`, `dgx2`, `dgx3`, `dgx4`**:
  `/home/swank/.cache/huggingface/hub/models--AngelSlim--DS-V4.1-Flash-DFly2-B6/snapshots/d08e55a6897b1974307ea9771038838de5986e66`

| File | Size (bytes) | SHA-256 (`dgx1..dgx4` verified) |
| --- | ---: | --- |
| `config.json` | `4,144` | `0d53324203ab18c69927da96aa27566195602b9ab0aac4867b2819d6a46a051e` |
| `model.safetensors.index.json` | `197,299` | `7192e8294c56f3a9064d7b3dfa4494dc7486009f3229e98e9c6d0d2fbdaf6786` |
| `model-00001-of-00004.safetensors` | `2,735,851,872` | `3a8d253390daebb7f4279ad4edd0eb0a89561cf0bae610931fbc4c47a497345a` |
| `model-00002-of-00004.safetensors` | `2,735,851,872` | `95da61fce834e4f2d1f3cc5a80656bd91ce624164cbc7f8d16726cab3e9f8701` |
| `model-00003-of-00004.safetensors` | `2,735,851,872` | `a2a1480dedc6bbcc9f8f54c928ca6ae11eaf27a979b8de90beb8bdb8bd540e2e` |
| `model-00004-of-00004.safetensors` | `639,969,728` | `0e42e694d739a7a4c07eb6c93be86a55b180f077273f11b129206d126a489a1d` |
| `README.md` | `16,548` | `d5b893bb16a1c1f1290cdf1bf08c1c92774a3fbf08015bd3ff906d215c1c19f4` |
| `LICENSE` | `11,629` | `14cac091b400701631c226dc33d34384cd19aedbf465c8765ad86ac60e3e0a6e` |
| `.gitattributes` | `1,519` | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` |

All fragmented DFly2 and target shards on `dgx1..dgx3` were defragmented inside the maintenance window before running `load_format=direct` benchmarks.

## Architecture and SM121 integration (`0870`)

1. **Zero drafter KV cache, zero prefill drafter overhead**:
   - `DFly2` shares the target model's SWA KV cache (`128` tokens, FP8 E4M3 + UE8M0) and compressed-MLA main KV cache (`512` top-k tokens, E2M1 + E4M3) plus the target's top-512 DSA index buffers at target layers `28, 32, 36`.
   - `uses_draft_kv_cache() == False`, `num_lookahead_tokens == 0`, `num_prefill_lookahead_tokens == 0`, and `separate_draft_input_tokens == 6` (`block_size = 6`, `num_speculative_tokens = 5`): 100% of `--kv-cache-memory-bytes` stays allocated to the 40 target layers, target `mtp.*` weights are skipped (`+250 MiB`/rank net resident delta over DSpark on `TP3`), and draft `lm_head` aliases `target_model.lm_head`.
2. **SM121 TileLang kernel path (`vllm/models/deepseek_v4_1/tilelang/dfly2.py`)**:
   - `sparse_mla_fwd_v41(..., segments_only=True, row_stride_div=6|1)` reuses the warp-specialized SM121 decode kernel across the fixed `640`-latent context window regardless of sequence length.
   - `@tilelang.jit` `dfly2_merge_backbone_kernel` fuses the 6-token bidirectional block-key RoPE dot products, learned `attn_sink`, and log2 online softmax merge.
   - `@tilelang.jit` `dfly2_merge_correction_kernel` fuses the 5-step sequential self-forced correction block-key dot products (`<= j`), `attn_sink`, `block_sink`, and gated relative bias (`tanh(gate_logits + block_gate_bias)` + `block_rel_bias`).
   - `@tilelang.jit` `dfly2_swiglu_kernel` fuses the correction MLP's unclamped FP32-product SwiGLU with `pdl.chain()`.
   - `@tilelang.jit` `dfly2_prepare_step_kernel` extracts `anchor_positions`, `prev_hidden`, `mtp_valid`, block slots, SWA slot tables, trimmed active main page-table columns, and `< anchor_pos` rejection/CED-aligned top-512 indices in one launch.
3. **Topology & memory coverage**:
   - Supports all 16 shapes (`tp4-ring`, `tp3-triangle`, `tp3-triangle-4k`, `tp2-direct`, `tp1`, `pp2-tp2-ring`, and all switched profiles), uneven `TP3` `64 -> 72` head / `8 -> 9` `o_groups` padding, dedicated workspace lane `1`, per-batch `[1..max_num_seqs] * 6` `FULL_DECODE_ONLY` CUDA graphs (`16` graphs on `tp3-triangle` captured in `0.12 s`), vocabulary-parallel top-2 greedy draft sampling, and cached `mask_token_id` embeddings.

## Measured `tp3-triangle` performance (`spark-ds41f:candidate-75` Boot 1 vs recorded `spark-ds41f-1126`)

Measured on `dgx1..dgx3` (`tp3-triangle`, `tp3-dev.json`, receipts in `receipts/tp3-triangle/`):
- **Boot 1 (cold JIT)**: ready `+207.8 s` (`+253/+255/+250` TileLang files compiled), `lowest MemAvailable`: `dgx1 6.503 GiB`, `dgx2 8.247 GiB`, `dgx3 8.066 GiB` (`0.0 GiB` swap). `DirectModelLoader` loaded `3.00 GB`/rank DFly2 weights in `0.63 s`; `16/16` DFly2 `FULL_DECODE_ONLY` CUDA graphs captured in `0.12 s`.
- **Boot 2 (warm JIT, stopped after ready per owner direction)**: ready **`+79.5 s`** (`+0/+0/+0` JIT files, vs `1126`'s `+78.5 s`), `lowest MemAvailable`: `dgx1 6.499 GiB`, `dgx2 8.294 GiB`, `dgx3 8.053 GiB` (`0.0 GiB` swap).

### Headline suites (`candidate-75` Boot 1 vs `manifests/suites/2026-11-26-spark-ds41f-1126-tp3*`)

| Suite / Step | `candidate-75` (DFly2) | `spark-ds41f-1126` (DSpark) | Delta | Notes |
| --- | ---: | ---: | ---: | --- |
| `screen-tp3` `model` (`128` prompts) | `rank1 = 1.0`, `ppl = 1.3161` | `rank1 = 1.0`, `ppl = 1.3161` | Identical | `0` failed, `outputs_sha256 = b795e6d4c813...`, `acceptance_length = 2.088` |
| `single-stream` `c1` (`tok/s`) | **`54.76`** (`±5.51%`, `acc = 1.421`) | **`71.28`** (`±3.64%`, `acc = 1.470`) | **`-23.2%`** | `22.81` vs `29.18 steps/s`; `TPOT p50 = 18.02 ms` |
| `screen-tp3` `c8` (`tok/s`) | **`151.40`** (`±1.78%`, `acc = 1.372`) | **`172.33`** (`acc = 1.424`) | **`-12.1%`** | `8.98` vs `9.91 steps/s`; `TPOT p50 = 52.47` vs `45.98 ms` |
| `sampled` `c1` (`tok/s`) | **`53.07`** (`±4.20%`, `acc = 1.338`) | **`69.12`** (`acc = 1.374`) | **`-23.2%`** | `22.95 steps/s` |
| `sampled` `c8` (`tok/s`) | **`148.73`** (`±1.36%`, `acc = 1.319`) | **`167.07`** (`acc = 1.333`) | **`-11.0%`** | `9.04` vs `10.05 steps/s` |
| `sampled` `c16` (`tok/s`) | **`198.02`** (`±0.78%`, `acc = 1.417`) | **`219.95`** (`acc = 1.452`) | **`-10.0%`** | `6.75` vs `7.27 steps/s` |
| `prefill` `32K` (`tok/s`) | **`7,037.8`** (`±2.16%`) | **`7,326.6`** (`±1.30%`) | **`-3.9%`** | `0` failed |
| `prefill` `262K` (`tok/s`) | **`7,254.3`** (`±0.36%`) | **`7,543.7`** (`±0.36%`) | **`-3.8%`** | `0` failed |

### Engine grid (`grid.json`, 100 replays per shape)

| Shape | C75 `step_ms` | C75 `fwd` | C75 `draft` | `1126` `step_ms` | `1126` `fwd` | `1126` `draft` | `step_ms` delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `decode 1x2 rows @4K` | `26.31` | `12.60` | **`12.42`** | `16.23` | `12.52` | **`2.39`** | `+62.1%` |
| `decode 4x8 rows @4K` | `29.70` | `13.54` | **`14.85`** | `17.94` | `13.60` | **`3.00`** | `+65.6%` |
| `decode 8x16 rows @4K` | `32.53` | `14.99` | **`16.21`** | `20.03` | `15.18` | **`3.49`** | `+62.4%` |
| `decode 1x3 rows @4K` | `26.59` | `12.76` | **`12.54`** | `16.49` | `12.78` | **`2.40`** | `+61.2%` |
| `decode 4x12 rows @4K` | `30.23` | `14.23` | **`14.68`** | `18.57` | `14.22` | **`3.01`** | `+62.8%` |
| `decode 8x24 rows @4K` | `34.98` | `17.59` | **`16.06`** | `22.58` | `17.71` | **`3.50`** | `+54.9%` |
| `decode 1x6 rows @4K` | `27.20` | `13.28` | **`12.63`** | `16.94` | `13.22` | **`2.40`** | `+60.6%` |
| `decode 2x12 rows @4K` | `29.21` | `14.24` | **`13.67`** | `18.30` | `14.19` | **`2.77`** | `+59.6%` |
| `decode 4x24 rows @4K` | `33.63` | `17.61` | **`14.69`** | `22.06` | `17.68` | **`3.04`** | `+52.4%` |
| `decode 8x48 rows @4K` | `40.25` | `22.81` | **`16.11`** | `27.82` | `22.89` | **`3.57`** | `+44.7%` |
| `decode 1x3 rows @64K` | `26.57` | `12.74` | **`12.52`** | `16.51` | `12.77` | **`2.41`** | `+60.9%` |
| `decode 8x24 rows @64K` | `34.80` | `17.49` | **`15.97`** | `22.49` | `17.65` | **`3.49`** | `+54.7%` |
| `decode 1x3 rows @256K` | `26.54` | `12.73` | **`12.50`** | `16.47` | `12.72` | **`2.39`** | `+61.2%` |
| `decode 8x24 rows @256K` | `34.87` | `17.53` | **`16.00`** | `22.62` | `17.74` | **`3.52`** | `+54.1%` |
| `decode 1x3 rows @1016K` | `26.61` | `12.78` | **`12.52`** | `16.51` | `12.75` | **`2.42`** | `+61.2%` |

## Defects found and fixed in `candidate-75-r2` (`0a45c7d5bd`, `8d0a48ea53`)

1. **Initial `release-tests` gate run on Build 1 (`receipts/release-tests/verdict-initial-pre-fix.json`, `dgx4`, `3,096 s`)**: `12,081 passed, 356 skipped, 7 failed` — all 7 were CPU/mock-stub test regressions (`mxfp4.py` string `config.device`, `gpu_worker.py` `_SpecConfig` stub lacking `use_dfly2`, `speculative.py` error-message regex in `test_adaptive_verification.py`, and module-scoped `_init_single_rank_distributed` losing `_TP` after `conftest.py` cleanup in `chunk005`). All 7 were fixed and verified passing.
2. **Docstring placement**: moved the early return below `_target_feeds_hc_residual`'s docstring (`speculator.py`) and the device check below `apply_rope_cpu`'s docstring (`tilelang/dfly2.py`).
3. **Efficiency fixes verified on SM121 GPU (`dgx4`)**:
   - Removed the `active_steps == num_speculative_steps` eager-fallback guard in `DFly2Speculator.propose()` so dynamic `K < 5` stays on the captured `FULL_DECODE_ONLY` CUDA graph and zeroes tail confidence columns.
   - Cached the constant `mask_token_id` embedding vector (`cache_mask_embed()`), shrinking backbone embedding lookup and TP all-reduce by `6x` (verified bit-exact).
   - Trimmed `main_page_table` copy in `dfly2_prepare_step_kernel` from `4,096` columns to `cdiv(anchor_pos, main_page_size) + 1` active pages.
   - Added `@tilelang.jit` `dfly2_swiglu_kernel` with `pdl.chain()` for `DFly2HiddenCorrection._mlp_local` (`5x` calls at `B=8` improved from `0.105 ms` -> `0.018 ms`, bit-exact `max_diff = 0.0`).
   - Kept the compact FP8/FP4 `sparse_mla_fwd_v41` path in `DFly2AttentiveCorrection` after SM121 profiling showed all `5x` calls combined take `0.045 ms` (`B=1`) / `0.087 ms` (`B=8`) from L2 cache (`215 KB` compact records vs `640 KB` BF16).
4. **Linux repo unit suite (`dgx4`)**: **`955 passed, 0 failed`** (`39.4 s`).
5. **In-image `175`-file `release-tests` gate bundle on `spark-ds41f:candidate-75-r2` (`dgx4`)**: running (`receipts/release-tests/verdict.json` pending completion).

## Conclusion

- `DFly2-B6` integrates cleanly on SM121 (`0` KV cache overhead, `0` prefill drafter work, `+79.5 s` warm boot, `6.50–8.29 GiB` memory headroom, exact target quality `rank1 = 1.0`, `perplexity = 1.3161`), and `candidate-75-r2` resolves all gate-test and integration defects.
- However, on GB10 (`273 GB/s` LPDDR5X UMA), `DFly2`'s architectural 3-layer backbone + 5 sequential `HiddenCorrection`/`AttentiveCorrection` steps + 5 sequential `ParallelLMHead` sweeps stream `~3.2 GB`/rank per step (`drafter_ms = 12.42–16.21 ms` vs `DSpark`'s `2.39–3.57 ms`), while native co-trained `DSpark` (`1017`) also edges out distilled `DFly2-B6` on acceptance (`1.470 / 1.424 / 1.452` vs `1.421 / 1.372 / 1.417`). `config/cluster.json` remains on promoted `spark-ds41f:1126` (`DSpark`).
