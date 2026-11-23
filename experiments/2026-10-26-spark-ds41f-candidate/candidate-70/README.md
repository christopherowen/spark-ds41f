# Candidate `candidate-70`

Date: 2026-11-23.
Base: `spark-ds41f-1122` / `candidate-69-r2` (`vLLM 0001..0857`, tree `8e0683a9276389957e49b7a26bd2bd062d788558`).

## Scope & Zero-Degradation Guarantee

`candidate-70` promotes only the verified "pure wins" from the `tp1`, `tp2`, and `pp2` performance screen while keeping every `tp3-*`, `tp4-*`, and `tp5..tp8-*` cluster profile byte-for-byte identical to `spark-ds41f-1122` (only image tag, `expected_labels`, and `upstreams_config` change):

- Patch `0858` executes strictly inside `ExpertSlotCache` (`VLLM_DS41_EXPERT_SLOTS > 0`: `tp1`, `tp2-direct`, `tp2-switched`).
- Patch `0859` executes strictly under `pipeline_parallel_size > 1` (`pp2-*`).
- Patch `0860` executes strictly under `pipeline_parallel_size > 1` when `enable_adaptive_verification=true` or `VLLM_SINK_REJECTION=1`; `pp2-*` profiles keep `enable_adaptive_verification=false` and `VLLM_SINK_REJECTION=0` in `candidate-70` so the image carries the PP adaptive verification machinery for follow-up `c1_scale` tuning without introducing the uncalibrated `code-c2` dip.
- Empirical benchmark-captured expert hotlists are intentionally excluded until calibrated on a held-out multi-domain corpus.

## Delta Over `spark-ds41f-1122` (`candidate-69-r2`)

1. **vLLM patch `0858`** (`0858-perf-dsv41-expert-centric-prefill-streaming-for-slot.patch`):
   - Adds expert-centric prefill streaming (`_run_expert_centric_prefill`, default ON via `VLLM_DS41_EXPERT_PREFILL_MODE=expert`) in `vllm/models/deepseek_v4_1/expert_slots/cache.py`.
   - When a prefill batch routes to more uncached cold experts than fit in the dynamic scratch slots (`E_cold > num_dynamic_slots`), partitions the cold experts into waves of at most `num_dynamic_slots`, streams each cold expert from NVMe exactly once per layer, accumulates every pass in exact FP32 (`acc_fp32`), adds `shared` once in FP32, and casts once to `plan.out_dtype`.
   - Eliminates O(rows / sub-batch) repeated NVMe thrashing during long-context prefill while preserving exact numerical parity (`LRU 5/5` on both `tp1` and `tp2-direct`).
2. **vLLM patch `0859`** (`0859-perf-dsv41-enable-intra-stage-prefill-sequence-paral.patch`):
   - Enables intra-stage prefill sequence parallelism (`sp_prefill`) under `pipeline_parallel_size > 1` when `tp > 1`, keeping the full-row `IntermediateTensors` PP wire ABI intact (`sp.local_slice` on non-first PP entry, `sp.gather_many` on non-last PP exit).
   - Fixes `prepare_disk_engram` when Engram layers `[1, 14]` are split across separate PP stages.
   - Zero behavior change when `pipeline_parallel_size == 1`.
3. **vLLM patch `0860`** (`0860-feat-v1-support-lockstep-PP-adaptive-verification-an.patch`):
   - Adds lockstep `PPHandler` confidence broadcasting (`pp_size`-deep queue) so pinned-`row_cost` adaptive verification can run identically across all PP stages, and enables `VLLM_SINK_REJECTION=1` on `PP > 1` (by sending `sampled_token_ids` and `num_nans` from the last PP rank over `PPHandler`).
   - Dormant on `pp_size == 1` and when `enable_adaptive_verification=false` / `VLLM_SINK_REJECTION=0`.
4. **Profile pure wins (`tp1`, `tp2-direct`, `tp2-switched`, `pp2-*`)**:
   - `tp1`: `VLLM_DS41_EXPERT_SLOTS=108` (up from `96`, `+12` resident experts/layer), `kv_cache_memory_bytes=6442450944` (`6.00 GiB` = `6.44M` tokens = `24.58x` 256K ctx / `6.14x` 1M ctx), `max_num_batched_tokens=8192` (`long_prefill_token_threshold=8168`), keeping PIECEWISE adaptive verification (`row_cost=0.5`, `DEAD_ROWS_TAU=0.6`).
   - `tp2-direct` & `tp2-switched`: `VLLM_DS41_EXPERT_SLOTS=252` (up from `240`, `+12` resident experts/layer), `kv_cache_memory_bytes=6442450944` (`6.00 GiB` = `6.44M` tokens = `24.58x` 256K ctx / `6.14x` 1M ctx), `max_num_batched_tokens=8192` (`long_prefill_token_threshold=8168`), and PIECEWISE adaptive verification (`row_cost=0.25`, `DEAD_ROWS_TAU=0.45`).
   - `pp2-tp2-ring`, `pp2-tp2-switched`, `pp2-tp3-switched`, `pp2-tp4-switched`: unchanged runtime configuration over `spark-ds41f-1122`, picking up `0859` intra-stage `sp_prefill`.

## Pre-Build Screen Evidence (Overlay on `spark-ds41f:1122`, branch `tp1-tp2-pp2-perf`)

| Shape | Baseline (`1120`/`1122`) | Pure-Win Screen Arm | Prefill | Decode (`prose-c1 / c4 / code-c1 / c4`) | Min `MemAvailable` | Quality |
| --- | --- | --- | --- | --- | --- | --- |
| `tp1` (`dgx1`) | `1120-tp1` (`96/48`, 8 GiB KV, 4K) | `108/48`, 6 GiB KV (`6.44M` tok), 8K, `0858` | `1K`: `332.6` (**`13.6x`**), `8K`: `1,229.3`, `32K`: `2,113.9` (**`~88x`**) | `6.7` (**`+53.1%`**) / `15.3` (**`+40.2%`**) / `6.6` (**`+37.7%`**) / `19.4` (**`+71.2%`**) | `9.06 GiB` | `LRU 5/5` |
| `tp2-direct` (`dgx1..2`) | `1120-direct2` (`240/64`, 8 GiB KV, 4K) | `252/64`, 6 GiB KV (`6.44M` tok), 8K, `0858`, `rc=0.25/tau=0.45` | `32K`: `4,596.6` (**`17.8x`**), `131K`: `4,897.4` (**`20.8x`**) | `21.6` (**`+22.0%`**) / `73.0` (**`+37.8%`**) / `24.8` (**`+11.7%`**) / `72.5` (**`+12.9%`**) | `5.88 / 7.82 GiB` | `LRU 5/5` |
| `pp2-tp2-ring` (`dgx1..4`) | `1122-pp2tp2ring` | `0859` intra-stage `sp_prefill` | `32K`: `6,470.9` (**`+16.1%`**), `131K`: `6,655.3` (**`+16.2%`**) | unchanged | `16.25–20.76 GiB` | `LRU 5/5` |

## Build & Verification Status

Built `spark-ds41f:candidate-70-r2` (`sha256:e392b4962a65cb5ac08b41d9c1aae8c823b25696462ba9e571b0a69c729d4e37`, vLLM `0001..0860`, `733` patches, `patch_head = cf8c3fe64faa3c90cfb1c55ec7c5574041533d07`, `expected_tree = 13dc1169a68dc0273e631b02e58fbd85f7fad9c7`, `patchset_sha256 = 8e335a808a2a93f4d2092a3584bb419fccb7b75194d1110557c4cb93c1bd01c9`) in `479 s` on `dgx4` (lowest `MemAvailable = 82.2 GiB`; GPU import smoke passed for DS4.1 SM12x, TileLang, SparkNet, and B12X-free import).

- **Repository checks**: All 17 `candidate-70` profiles pass `bin/spark doctor`; all `tp3-*`, `tp4-*`, `tp5..tp8-*`, and `pp2-*` profiles resolve byte-for-byte identically to `releases/spark-ds41f-1122/cluster.json` (except image tag, `expected_labels`, and `upstreams_config`); full Linux repository unit test suite `929 passed, 0 failed` (`31.1 s`).
- **Zero-overlay built-image GPU unit tests on `dgx4` (`--gpus all`)**: **`857 passed, 2 skipped, 0 failed, 0 errors`** (`742 passed, 2 skipped` in `104.3 s` + `115 passed` in `0.34 s`) across all 16 suites: `test_deepseek_v41_expert_slots.py`, `test_deepseek_v4_1_sp_prefill.py`, `test_adaptive_verification.py`, `test_deepseek_v4_1_glp.py`, `test_deepseek_v41_serving_contract.py`, `test_deepseek_v41_postload_startup.py`, `test_jit_warmup_migration.py`, `test_deepseek_v4.py`, `test_deepseek_v41.py`, `test_deepseek_v41_dsml_provenance.py`, `test_deepseek_v41_structural_closers.py`, tokenizers (`test_deepseek_v4.py`, `test_deepseek_v41.py`, `test_deepseek_v41_effort.py`, `test_deepseek_v41_renderer_pool.py`), and `test_responses_custom_tools.py`.
- **Distribution**: Identical OCI digest `sha256:e392b4962a65cb5ac08b41d9c1aae8c823b25696462ba9e571b0a69c729d4e37` verified on all four nodes (`dgx1`, `dgx2`, `dgx3`, `dgx4`).
