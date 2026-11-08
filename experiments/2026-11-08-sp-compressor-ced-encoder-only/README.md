# SP Ratio-2 Compressor Projection and CED Encoder-Only Prefill Chunks

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18b6ca6521b602f09cb3ba326774166d1c`
- **Base vLLM patch head**: `320021e3c4dbb202d911ab70e2cd924c75420488` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`, 641 patches)
- **Candidate vLLM patch head**: `f467c438ac1e1c8fe11ff9669715f6b5841aaaeb` (tree `1c9482de34b4c136ac1e063cd29110f946af136e`, `patchset_sha256` `f6d96ec684419c7f798b8e02882036ca12f19aedcdccbbf9c802382c77b7b3ba`, 642 patches)
- **Patch**: `vllm/0762-perf-dsv41-sp-ratio2-compressor-and-ced-encoder-only-chunks.patch`

## Problem & Sublayer Timing Breakdown

CUDA-event instrumentation across a `16,399`-token prefill chunk on `sp1..sp4` (`tp4-switched`, 4x DGX Spark over 200G RoCE) exposed two lossless prefill bottlenecks in `spark-ds41f:1107`:

1. **Full-row all-gather and replicated front/compressor projections on encoder layers 2, 8, and 14 (`compress_ratio == 2`)**:
   - In `vllm/models/deepseek_v4_1/compressor.py`, `DeepseekCompressor._sp_enabled` was restricted to `self.compress_ratio == 1`.
   - Because encoder layers 2, 8, and 14 have `compress_ratio == 2` (`512` value channels + `512` gate channels, `shards=5, blocked=True`), `_sp_enabled` was `False` on those layers, which also disabled `DeepseekV4Attention._gather_front` in `vllm/models/deepseek_v4_1/attention.py`.
   - Consequently, on layers 2, 8, and 14 during Sequence Parallel (`SP`) prefill:
     - Each layer first all-gathered the full `[16399, 5120]` BF16 hidden state (`168 MiB`) across TP ranks before attention;
     - Every rank ran `fused_wqa_wkv` on all `16,399` rows (`4,099` local rows x 4) instead of only `sp_rows.local_rows`;
     - Every rank ran `fused_wkv_wgate` on all `16,399` rows and then executed an extra column all-gather for the compressor outputs (`~16.0 ms/layer` on layers 2, 8, and 14 vs `5.4 ms` on layer 20).

2. **Redundant decoder execution (`Layers 20..39`) on non-final multi-chunk prefill steps**:
   - Under Contextual Encoder-Decoder (`CED`), only the final `CED_WINDOW = 128` prompt positions (`[prefill_len - 128, prefill_len)`) enter the decoder layers (`Layers 20..39`) once `prepare_global_kv` at the boundary layer (`Layer 20`) has written the compressed global KV cache.
   - For multi-chunk prompts (`> 16,384` tokens), non-final prefill chunks ending at or before `floor = prefill_len - 128` select `0` real decoder replay rows. To keep compact decoder tensors non-empty (`count >= 1`), `select_rows_into` pads `count = 1` with row index `0`.
   - Previously, non-final chunks still executed Layer 20's decoder Sparse MLA attention and MoE plus all 19 subsequent decoder layers (`Layers 21..39`, including 20 MoE all-to-all dispatches/combines and 20 Sparse MLA indexer + attention passes) on that single dummy row, followed by two `[16384, 5120]` `scatter_rows` zero-fills (`~28–32 ms` per non-final chunk).

## Changes

1. **SP local-row projection for `compress_ratio == 2` KV compressors and `_gather_front` on Layers 2, 8, and 14** (`vllm/models/deepseek_v4_1/compressor.py`, `vllm/models/deepseek_v4_1/attention.py`):
   - Enable `DeepseekCompressor._sp_enabled` for `self.compress_ratio in (1, 2)` when `self._tilelang_weight_views` is available.
   - For `compress_ratio == 2`, project `sp_rows.local_rows` directly via `tilelang_linear.bf16_gemm_two(hidden_states[:sp_rows.local_rows], wkv_view, wgate_view, self.kv_out_dim, self.hidden_size, out_dtype="float32", shards=split["shards"], blocked=split["blocked"])` and gather `(local_values, local_gates)` across ranks with `sp_rows.gather_many(local_values, local_gates)` before `compressor_Stage1`.
   - Enable `DeepseekV4Attention._gather_front` on compressor layers (`self.use_compressor and self.compressor._sp_enabled`) so Layers 2, 8, and 14 project `wq_a/wkv` on `sp_rows.local_rows` and gather the narrower projected outputs alongside the local compressor projections, cutting compressor + attention front time on Layers 2, 8, and 14 by `~8 ms/layer` (`~24 ms` per `16,384`-token chunk).

2. **CED `encoder_only` early exit at the boundary layer on non-final prefill chunks** (`vllm/models/deepseek_v4_1/ced.py`, `vllm/models/deepseek_v4_1/nvidia/model_state.py`, `vllm/models/deepseek_v4_1/nvidia/model.py`):
   - In `DeepseekV41ModelState._stage_ced`, compute `encoder_only` when not capturing/forcing full output and every request `i` in the step is a non-final prefill chunk ending at or before its CED decoder replay floor (`num_computed_prefill_tokens_np[i] + num_scheduled_tokens[i] <= floors[i]`).
   - At the CED boundary layer (`idx == self.ced_decoder_start`, Layer 20) in `DeepseekV4Model.forward` and `DeepseekV4DecoderLayer.forward`, when `boundary_encoder_only` is active on the last PP rank outside CUDA graph capture, run `mhc.pre` and `attn.prepare_global_kv` to populate the global KV cache and immediately return preallocated zero hidden/auxiliary states without executing Layer 20's decoder attention/FFN or Layers 21..39.

## Bit-Exactness & Unit Test Verification

- **GPU bit-exactness check (`TP=3` and `TP=4`)**: Verified on GB10 (`sp1`) that `bf16_gemm_two(512, 5120, out_dtype="float32", shards=5, blocked=True)` on local SP rows matches the column-parallel `project_parts` + `gather_columns_into` path with `max_abs_diff = 0.0` (`torch.equal == True` for both `values` and `gates` at `TP=3` and `TP=4`).
- **Two-boot GPU `pytest` suite on `sp1` (`spark-ds41f:1107`)**:
  - `receipts/gpu-pytest-boot1.txt`: `347 passed in 166.46s`
  - `receipts/gpu-pytest-boot2.txt`: `347 passed in 166.78s`

## Prefill Benchmark Results (`sp1..sp4`, `tp4-switched`)

Measured with `bin/spark --cluster-config config/cluster.json --shape tp4-switched bench` (`servebench prefill`, 3 repeats per context length, deterministic source-text slice) against the recorded `spark-ds41f:1107` `tp4-switched` reference (`manifests/benchmarks/2026-11-07-spark-ds41f-1107-switched.json`):

| Target Prompt Tokens | Actual Prompt Tokens | Recorded `1107` `tp4-switched` (tok/s) | Candidate `0762` `tp4-switched` (tok/s) | Candidate Repeats (tok/s) | Candidate TTFT Mean (s) | Speedup vs `1107` |
|---:|---:|---:|---:|---|---:|---:|
| `32,768` | `32,761` | `8,471.55 ± 1.70%` | **`8,667.45 ± 0.47%`** | `[8651.13, 8667.36, 8683.86]` | `3.780 s` | **`+2.31%`** |
| `65,536` | `65,529` | `8,885.39 ± 0.14%` | **`8,924.73 ± 2.32%`** | `[8963.67, 8981.69, 8828.84]` | `7.343 s` | **`+0.44%`** (peak **`8,981.69`**) |
| `131,072` | `130,839` | `8,612.63 ± 0.90%` | **`8,756.80 ± 1.20%`** | `[8795.00, 8711.49, 8763.91]` | `14.942 s` | **`+1.67%`** |
| `262,144` | `262,080` | `8,209.23 ± 1.11%` | **`8,349.31 ± 0.84%`** | `[8341.61, 8325.56, 8380.76]` | `31.389 s` | **`+1.71%`** |

Full benchmark output is recorded in `receipts/servebench-prefill-tp4-switched.json`.
