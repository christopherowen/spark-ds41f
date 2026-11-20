# Config-driven `PP=2, TP=[2-4]` (live proof on `PP=2, TP=2`)

- **Base commit:** `2a65d3a8f` (`origin/main`, `spark-ds41f:1120`)
- **Causal variable:** Pipeline parallelism (`pipeline_parallel_size: 2`) with `TP = nodes // PP` (`TP=2` on 4 nodes, `TP=3` on 6 nodes, `TP=4` on 8 nodes), config-driven through `scripts/recipe.py`, `scripts/topology.py`, `spark_ds41f/cluster.py`, `spark_ds41f/config.py`, and `spark_ds41f/doctor.py`, plus the pure-Python vLLM PP fixes in `patches/vllm/0001-dsv41-pipeline-parallel-engram-dspark-aux.patch`.

## Architecture and fabric fit

DeepSeek-V4.1-Flash (40 target layers) partitions at layer 20 (`0..19 | 20..39`) with zero cross-stage KV, indexer, Engram, or DSpark dependencies:

- **Stage 0 (`0..19`):** input embedding (`embed_tokens`), vision tower, both Engram layers (`1, 14`, disk-backed), all CED encoder layers, KV/indexer owners `2, 8, 14`.
- **Stage 1 (`20..39` + `mtp.0..2`):** `ced_decoder_start = 20` (fresh KV/indexer/candidate owner), all DSpark aux tap layers (`37, 38, 39`), DSpark drafter (with own `VocabParallelEmbedding` loaded from `embed.weight`), and `lm_head`.

Fabric transports supporting `PP=2`:

- **`oneshot-switched` (`TP=2, 3, 4` on 4/6/8 switched Sparks):** `pp2-tp2-switched.json`, `pp2-tp3-switched.json`, `pp2-tp4-switched.json`. Uses the switched clique directly (`sparknet` `oneshot-direct` for each TP stage, NCCL over RoCE for PP P2P).
- **`oneshot-pp2-ring4` (`TP=2` on 4 switchless ring Sparks, zero recabling):** `pp2-tp2-ring.json`. `topology.pp2_ring4_node_map` swaps ranks 2 and 3 from a `tp4-ring` map so Stage 0 (`dgx1=0 <-> dgx2=1`) and Stage 1 (`dgx4=2 <-> dgx3=3`) run direct dual-stripe CX7 `sparknet` `oneshot-direct` (keyed by TP-local peer rank `"1"`/`"0"`) while PP pipes (`dgx1 <-> dgx4`, `dgx2 <-> dgx3`) run direct dual-stripe CX7 NCCL with `NCCL_IB_SUBNET_AWARE_ROUTING=1` and `NCCL_RUNTIME_CONNECT=1`.
- **`nccl-socket` (`TP=2..4` over Ethernet LAN):** `pp2-tp2-socket.json`. Used to prove the 4-node `PP=2, TP=2` stack live on `dgx1..dgx4` while `dgx1..dgx3` remained physically cabled in `tp3-triangle` and `dgx4`'s CX7 ports were down.

## vLLM patches

1. `patches/vllm/0001-dsv41-pipeline-parallel-engram-dspark-aux.patch` (core PP=2 support, applies cleanly with zero fuzz on `spark-ds41f:1120`):
   - `vllm/models/deepseek_v4_1/common/engram.py`: remove the `pipeline_parallel_size != 1` refusal on disk Engram.
   - `vllm/models/deepseek_v4_1/nvidia/model.py`: scope `disk_engram` and Engram hash registration to PP stages that hold local Engram layers while keeping `file_backed_engram` true on all stages so `DirectModelLoader` skips streaming Engram tables on Stage 1; set `supports_aux_hidden_states_over_pp = True`; pack/collect aux hidden states through `IntermediateTensors`.
   - `vllm/models/deepseek_v4_1/nvidia/dspark.py`: on non-first PP stages, instantiate `VocabParallelEmbedding` on `DSparkDeepseekV4Model`, set `has_own_embed_tokens = True`, add `"embed."` to `checkpoint_weight_name_prefixes`, map `"embed.weight" -> "model.embed_tokens.weight"`, and assert it loaded.
   - `vllm/v1/core/kv_cache_utils.py`: skip projected PP KV-cache groups with empty `layer_names` in `get_kv_cache_config_from_groups` so `UniformTypeKVCacheSpecs` (e.g. DSpark draft KV layers on Stage 0) does not emit `KVCacheTensor`s for another stage's layers (`StopIteration` in `allocate_kv_cache`).
   - `vllm/v1/worker/gpu/model_runner.py`: gate `LayerProgress` (`VLLM_LATE_ADMISSION_LAYER`) with `and not self.use_pp`; preserve `model_inputs["input_ids"]` on non-first PP stages when `model.requires_raw_input_tokens` is set; declare TileLang's `post_update_num_computed_tokens_calls` when `not self.is_last_pp_rank`.
   - `vllm/v1/worker/gpu/cudagraph_utils.py`: preserve `model_inputs["input_ids"]` during CUDA graph capture on non-first PP stages when `requires_raw_input_tokens(model)` is set.
2. `patches/vllm/0002-cuda-communicator-allow-socket-without-roce.patch` (Ethernet functional test helper):
   - `vllm/distributed/device_communicators/cuda_communicator.py`: skip `sparknet` RoCE on TP groups when `NCCL_IB_DISABLE=1` (keeping RoCE fail-closed otherwise), and view `torch.uint32` as `torch.int32` around `pynccl_comm.all_gather` (needed by `tilelang/wo.py`'s scale-factor gather).

## Live verification (`PP=2, TP=2` on `dgx1..dgx4`, 2026-11-21)

Cluster cabling during run: `dgx1..dgx3` physically cabled as `tp3-triangle`; `dgx4` CX7 ports down, `enP7s7` 10 GbE LAN up (`10.0.1.71..74`). Tested with `pp2-tp2-socket.json` on `spark-ds41f:1120` with `overlay/vllm/` bind-mounted read-only.

### Boot history (all runs recorded)

1. **`ee470425a` (`00:05Z`, failed at +25 s):** Gloo 4-node rendezvous succeeded; `CudaCommunicator.__init__` raised `ValueError: sparknet's one-shot collectives need custom all-reduce`. Fixed with `NCCL_IB_DISABLE=1` guard in `0002`.
2. **`8589bda17` (`01:50Z`, failed during `_prepare_kernels`):** All 4 ranks loaded model cleanly (`67.99 GiB` on Stage 0, `72.01 GiB` on Stage 1 with `DSpark draft model loaded: 98 params`, `Eagle3` aux layers `(37, 38, 39)` verified over PP). Failed in `tilelang/wo.py:328` (`pynccl_comm.all_gather` rejected `torch.uint32`). Fixed by viewing `uint32` as `int32` in `0002`.
3. **`35516a0a5` (`01:57Z`, failed during `allocate_kv_cache`):** Compiled/cached all 284 TP=2 TileLang kernels and computed `GPU KV cache size: 20,661,635 tokens (19.70x 1M)`. Failed with `StopIteration` in `allocate_kv_cache` due to empty projected DSpark KV group on Stage 0. Fixed in `kv_cache_utils.py` (`0001`).
4. **`b922ea64f` (`02:02Z`, failed during `warmup.py`):** Allocated KV cache on all 4 ranks, ran first prefill + PP0→PP1 P2P transfer + sampling + DSpark proposal; Stage 0 hit `KernelNotPreparedError: post_update_num_computed_tokens_kernel('int64')`. Fixed by adding `or not self.is_last_pp_rank` in `model_runner.py:2706` (`0001`).
5. **`cd3d78283` (`02:06Z`, failed during `capture_model`):** Passed entire `warmup_kernels` cycle; Stage 1 hit `ValueError: DeepSeek V4 vision MoE routing requires input_ids` in `cudagraph_utils.py:1165`. Fixed with `if not requires_raw_input_tokens(model):` (`0001`).
6. **`1f10097ef` (`02:11Z`, failed post-init):** Completed 100% of engine init in `60.33 s` (26/26 FULL CUDA graphs on Stage 0 `0.25 GiB`, target + DSpark graphs on Stage 1 `0.34 GiB`); `EngineCore` rejected inherited `VLLM_LATE_ADMISSION_LAYER=30` under PP. Fixed in `scripts/recipe.py` (strip `VLLM_LATE_ADMISSION_LAYER` automatically when `pp > 1`).
7. **`a1ac04721` (`02:14Z`, PASSED — API ready at `+124.9s`):**

### Live proof results (`a1ac04721`, `pp2-tp2-socket.json`, 10 GbE TCP)

- **`bin/spark --cluster-config experiments/2026-11-20-pp2-tp2/pp2-tp2-socket.json doctor --live`:** `configuration OK; live cluster matches with 1 warning` (only shard fragmentation warning on `dgx2, dgx3`).
- **Memory & KV capacity:**
  - Stage 0 (`dgx1` PP0/TP0, `dgx2` PP0/TP1): `67.99 GiB` weights, `16.5 GiB` KV, `0.25 GiB` CUDA graphs, `MemAvailable`: `22.55 GiB` (`dgx1`), `24.50 GiB` (`dgx2`).
  - Stage 1 (`dgx4` PP1/TP0, `dgx3` PP1/TP1): `72.01 GiB` weights (incl. DSpark + own `embed_tokens`), `16.5 GiB` KV, `0.34 GiB` CUDA graphs, `MemAvailable`: `21.57 GiB` (`dgx4`), `22.18 GiB` (`dgx3`).
  - KV pool: `20,661,635` tokens (`19.70x` 1M-context concurrency).
- **Greedy (`temperature=0`) correctness & determinism:**
  - `c1` (`17 * 23 + 19`): `129` tok in `7.66 s` (`16.8 tok/s`), repeat `129` tok in `7.81 s` (`16.5 tok/s`), answer `'410'`, **byte-identical content + reasoning: `True`**.
- **Concurrent PP microbatch pipelining:**
  - `c4`: `396` completion tok in `8.98 s` (`44.1 agg tok/s`), all 4 coherent & accurate.
  - `c8`: `792` completion tok in `10.22 s` (`77.5 agg tok/s`), **byte-identical vs `c4`: `True`**.
- **Long & multi-chunk prefill (CED layer-20 boundary across PP0→PP1):**
  - `12,669`-token single-chunk prefill + `41` completion tok in `9.40 s`: needle `'ORBIT-7429'` recalled (`True`).
  - `21,074`-token **two-chunk prefill** (`> max_num_batched_tokens=16384`, exercising `encoder_only` CED on Chunk 1 across PP0→PP1) + `64` completion tok in `15.42 s`: both keys `'DELTA-9012/SIGMA-3456'` recalled (`True`).
- **DSpark 5-token speculative decoding on Stage 1 + `broadcast_drafts` to Stage 0 (`/metrics`):**
  - `485` draft rounds (`2,425` draft tokens proposed) → `1,272` accepted (**`3.62` tokens/step** mean acceptance length; per-position: pos0 `78.6%`, pos1 `62.7%`, pos2 `48.0%`, pos3 `41.4%`, pos4 `31.5%`).
- **Post-ready logs (`dgx1..dgx4`):** `0` errors, `0` tracebacks, `0` unprepared kernels across all 4 ranks.

## Conclusion

Config-driven `PP=2, TP=[2-4]` works end-to-end on DeepSeek-V4.1-Flash with disk Engram, CED chunked prefill, FULL CUDA graphs, and DSpark 5-token speculative decoding. Production `tp3-triangle` was cleanly restored after the live `PP=2, TP=2` verification. Throughput numbers above were measured over 10 GbE Ethernet TCP (`nccl-socket`) solely as a functional proof; `pp2-tp2-ring.json` and `pp2-tp{2,3,4}-switched.json` are ready for full-speed 200 GbE RoCE / switched benchmarking once 4–8 RDMA-cabled Sparks are available.
