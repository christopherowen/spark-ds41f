# Candidate `candidate-69`

Date: 2026-11-22.
Base: `candidate-68` (`vLLM 0001..0852`, tree `740f884e927d790cdc48e5b6fe60056a6a0925c1`).

## Delta Over `candidate-68`

1. **vLLM patch `0853`** (`0853-feat-dsv41-enable-pipeline-parallelism-with-disk-Eng.patch`):
   - Removes the `pipeline_parallel_size != 1` refusal on disk-backed Engram (`engram.py`).
   - Scopes `disk_engram` and Engram hash registration to PP stages that own local Engram layers while keeping `file_backed_engram` true on all stages so `DirectModelLoader` skips streaming Engram tables on Stage 1; enables `supports_aux_hidden_states_over_pp = True`; packs/collects aux hidden states through `IntermediateTensors`; offsets `_graph_out` by `_aux_slot_base_cached` (`model.py`).
   - Instantiates `VocabParallelEmbedding` on non-first PP stages in `DSparkDeepseekV4Model`, sets `has_own_embed_tokens = True`, maps `embed.weight -> model.embed_tokens.weight`, and verifies it loaded (`dspark.py`).
   - Skips projected PP KV-cache groups with empty `layer_names` in `get_kv_cache_config_from_groups` (`kv_cache_utils.py`).
   - Gates `LayerProgress` (`VLLM_LATE_ADMISSION_LAYER`) with `and not self.use_pp`, preserves `model_inputs["input_ids"]` on non-first PP stages when `requires_raw_input_tokens` is set (`model_runner.py`, `cudagraph_utils.py`), and declares TileLang `post_update_num_computed_tokens_calls` when `not self.is_last_pp_rank`.
2. **vLLM patch `0854`** (`0854-perf-v1-balance-V2-PP-async-decode-microbatches-acro.patch`):
   - Caps per-step scheduled running decodes at `ceil(decode_running / pp_size)` under `V2+PP+async` (`scheduler.py`), preventing concurrent requests from lockstepping into a single microbatch with alternating empty pipeline bubble steps.
3. **Config-driven `PP=2, TP=[2-4]` deployment plumbing & candidate cluster shapes**:
   - Adds `pp2-tp2-ring` (`oneshot-pp2-ring4`, 4 switchless ring Sparks), `pp2-tp2-switched` (4 switched Sparks), `pp2-tp3-switched` (6 switched Sparks), and `pp2-tp4-switched` (8 switched Sparks) shape profiles (`pp2-tp*.json` and `*-dev.json`) to `candidate-69/` (leaving promoted `config/cluster.json` unchanged until release promotion).
   - `scripts/recipe.py` and `scripts/topology.py` resolve `pipeline_parallel_size`, derive `TP = nodes // PP`, align `draft_tensor_parallel_size = tp`, strip `VLLM_LATE_ADMISSION_LAYER` for `pp > 1` and `NCCL_SWITCHLESS_BIDIRECTIONAL` for `oneshot-pp2-ring4`, and map direct dual-stripe `sparknet` TP=2 pairs + NCCL RoCE PP=2 pipes on 4-node switchless rings.
4. **Live `PP=2, TP=2` overlay evidence on `spark-ds41f:1120`** ([2026-11-20-pp2-tp2](../../../2026-11-20-pp2-tp2/README.md)):
   - Verified live over 200 GbE RoCE (`oneshot-pp2-ring4`) on `dgx1..dgx4`: balanced dual-stripe RDMA on both `sparknet` TP=2 (`14.74 GB` tx/rx per stripe) and NCCL RoCE PP=2 (`628 / 608 MB` tx per stripe), `LRU 5/5` quality across both boots, `44K` 3-chunk and `131K` multi-chunk CED prefill (`5,698 tok/s`), `20.66M` KV tokens (`19.70x` 1M ctx), `21.9–25.0 GiB` min `MemAvailable`, `0` errors across all 4 ranks.
   - Patch `0854` eliminated lockstepped pipeline bubbles at low concurrency (`code-c2` `+24.7%` to `92.8 ± 1.5% tok/s`, `prose-c2` `+15.6%` to `76.2 ± 1.8% tok/s` with CI collapsing from `±20.0%` to `±1.8%`, `code-c4` `+6.7%` to `147.8 ± 4.4% tok/s`).

## Build & Verification Status

Composed `candidate-69` (`vLLM 0001..0854`, `727` patches, `patch_head = c8dcce14a9f1bc1a8c1c0f991c9c2206f6f5fa54`, `expected_tree = 514f3b1d311b91e73e522c19e9fc7b1568b85745`, `patchset_sha256 = 1f5ab013ba20fa4f3166986e9e354fc1fd32f90968e425ad350c1d56bdcfbe44`). Image build (`spark-ds41f:candidate-69`) and 4-node distribution pending build window.
