# Config-driven `PP=2, TP=[2-4]` (live proof on `PP=2, TP=2`)

- **Base commit:** `06535980fd` (`origin/main`, `spark-ds41f:1120`; earlier `nccl-socket` proof ran on `2a65d3a8f`)
- **Causal variable:** Pipeline parallelism (`pipeline_parallel_size: 2`) with `TP = nodes // PP` (`TP=2` on 4 nodes, `TP=3` on 6 nodes, `TP=4` on 8 nodes), config-driven through `scripts/recipe.py`, `scripts/topology.py`, `spark_ds41f/cluster.py`, `spark_ds41f/config.py`, and `spark_ds41f/doctor.py`, plus the pure-Python vLLM PP fixes in `patches/vllm/0001-dsv41-pipeline-parallel-engram-dspark-aux.patch` and `patches/vllm/0003-pp-async-balance-decode-microbatches.patch`.

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
2. `patches/vllm/0002-cuda-communicator-allow-socket-without-roce.patch` (Ethernet functional test helper, only mounted by `pp2-tp2-socket.json`):
   - `vllm/distributed/device_communicators/cuda_communicator.py`: skip `sparknet` RoCE on TP groups when `NCCL_IB_DISABLE=1` (keeping RoCE fail-closed otherwise), and view `torch.uint32` as `torch.int32` around `pynccl_comm.all_gather` (needed by `tilelang/wo.py`'s scale-factor gather over NCCL; not needed under `sparknet` RoCE).
3. `patches/vllm/0003-pp-async-balance-decode-microbatches.patch` (PP microbatch balancing, zero-fuzz apply on `spark-ds41f:1120`):
   - `vllm/v1/core/sched/scheduler.py`: under `V2+PP+async`, `AsyncScheduler` enforces `next_decode_eligible_step = current_step + pp_size`. Without per-step balancing, concurrent requests that arrive together lockstep into a single microbatch and leave alternating pipeline steps empty (50% bubble + bimodal variance). Capping per-step scheduled running decodes at `ceil(decode_running / pp_size)` splits concurrent requests evenly across the `pp_size` pipeline slots and self-rebalances as requests finish.

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

## Live 200 GbE RoCE verification (`oneshot-pp2-ring4` on `dgx1..dgx4`, 2026-11-22)

Cluster rewired to the standard 4-node ConnectX-7 ring (`tp4-ring` / `40-cx7.yaml.ring`); all 8 ring paths (`10.12.1/2`, `10.14.1/2`, `10.23.1/2`, `10.34.1/2`) verified reachable and `bin/spark --cluster-config experiments/2026-11-20-pp2-tp2/pp2-tp2-ring.json doctor --live` passed cleanly (`spark-ds41f:1120`, core PP overlay only — no `0002` socket helper).

### Commands

```bash
./bin/spark --cluster-config experiments/2026-11-20-pp2-tp2/pp2-tp2-ring.json cluster start --change-shape --replace --apply
./bin/spark --cluster-config experiments/2026-11-20-pp2-tp2/pp2-tp2-ring.json doctor --live
./bin/spark --cluster-config experiments/2026-11-20-pp2-tp2/pp2-tp2-ring.json bench \
  --suites quality,decode,prefill \
  --prefill-sizes 32768,131072 --prefill-repeats 2 \
  --compare manifests/benchmarks/2026-11-20-spark-ds41f-1120-tp4.json
```

### Boot 1 (`502eb15c1`, unmodified `scheduler.py` — API ready at `+113.0 s`)

- **Fabric & RDMA proof:** `sparknet` `oneshot-direct` initialized cleanly on Stage 0 (`dgx1 <-> dgx2`, `10.12.*`) and Stage 1 (`dgx4 <-> dgx3`, `10.34.*`); NCCL RoCE initialized cleanly on PP P2P (`10.14.*`, `10.23.*`) and `pp_broadcast`. Hardware RDMA counters on `dgx1` across a functional sweep (`c1`–`c16` + 44K 3-chunk CED prefill):
  - `rocep1s0f0` / `roceP2p1s0f0` (`TP0 <-> TP1`, `10.12.*`): `14,737.6 MB` tx/rx and `14,737.7 MB` tx / `14,737.8 MB` rx (100% balanced dual-stripe `sparknet` RoCE).
  - `rocep1s0f1` / `roceP2p1s0f1` (`PP0 <-> PP1`, `10.14.*`): `628.3 MB` tx / `4.5 MB` rx and `608.1 MB` tx / `2.6 MB` rx (balanced dual-stripe NCCL RoCE P2P + `broadcast_drafts`).
- **Functional & quality checks:**
  - Greedy `c1` deterministic (`'410'`, `42.3 tok/s`, repeat identical `True`).
  - `44,041`-token three-chunk CED prefill completed in `8.39 s` and recalled `'DELTA-9012/SIGMA-3456'`.
  - Sonar `quality`: **LRU 5/5 pass**.
- **Memory & thermals:** `min MemAvailable` `18.04 GiB` (`dgx1`), `20.55 GiB` (`dgx2`), `19.80 GiB` (`dgx4`), `19.83 GiB` (`dgx3`); `0.0 GiB` swap growth; `0.0 ms` thermal slowdown; peak GPU temp `66.0 C` (`75.0 W` on Stage 0, `43.4 W` on Stage 1).
- **Prefill (vs `2026-11-20-spark-ds41f-1120-tp4.json`):**
  - `32,768` tok: **`5,539.5 ± 4.8% tok/s`** (`n=2`, ref `9,762.9`, `-43.3%`).
  - `131,072` tok: **`5,698.1 ± 0.6% tok/s`** (`n=2`, ref `10,097.1`, `-43.6%`).
- **Scheduler lockstep finding:** Per-sample decode speeds were sharply bimodal when concurrent requests arrived in the same step vs across two steps (`prose-c2`: `78.4` pipelined vs `61.7/61.9/61.8` lockstepped; `prose-c4`: `144.0/143.8` vs `113.7/113.6`; `code-c8`: `311.7` vs `253.1`).

### Boot 2 (`0b37b5a50`, with `0003-pp-async-balance-decode-microbatches.patch` — API ready at `+105.5 s`)

- **Quality, memory, logs:** Sonar `quality`: **LRU 5/5 pass**; `min MemAvailable` `21.92–25.01 GiB` across `dgx1..dgx4`, `0.0 GiB` swap growth, `0.0 ms` thermal slowdown, `35–41 W` peak decode power, **`0` errors / tracebacks / unprepared kernels** across all 4 ranks.
- **Decode comparison (aggregate tok/s and step time, 95% CI):**

| Point | Boot 1 (unmodified) | Boot 2 (`0003` balanced) | Boot 2 vs Boot 1 | `tp4-ring` 1120 ref |
|---|---:|---:|---:|---:|
| `prose-c1` | `40.4 ± 0.6%` (`56.4 ms`) | `40.2 ± 1.6%` (`56.7 ms`) | `-0.4%` (same) | `84.9` (`25.1 ms`) |
| `prose-c2` | `66.0 ± 20.0%` (`70.3 ms`) | **`76.2 ± 1.8%`** (`59.9 ms`) | **`+15.6%`** (CI `20% -> 1.8%`) | `150.4` (`26.7 ms`) |
| `prose-c4` | `128.8 ± 21.5%` (`72.3 ms`) | `123.9 ± 1.7%` (`74.2 ms`) | `-3.8%` (CI `21.5% -> 1.7%`) | `266.9` (`28.4 ms`) |
| `prose-c8` | `226.1 ± 8.8%` (`81.7 ms`) | `217.6 ± 9.2%` (`85.0 ms`) | `-3.7%` (same) | `453.8` (`33.4 ms`) |
| `prose-c16` | `367.0 ± 2.6%` (`100.9 ms`) | `354.2 ± 0.7%` (`104.6 ms`) | `-3.5%` (slower) | `657.3` (`47.5 ms`) |
| `code-c1` | `50.7 ± 0.8%` (`59.3 ms`) | `50.3 ± 1.1%` (`59.8 ms`) | `-0.8%` (same) | `105.1` (`27.3 ms`) |
| `code-c2` | `74.5 ± 0.5%` (`82.0 ms`) | **`92.8 ± 1.5%`** (`65.1 ms`) | **`+24.7%`** (`-20.6%` step) | `186.0` (`28.2 ms`) |
| `code-c4` | `138.5 ± 0.0%` (`88.3 ms`) | **`147.8 ± 4.4%`** (`82.6 ms`) | **`+6.7%`** (`-6.4%` step) | `329.0` (`30.3 ms`) |
| `code-c8` | `279.9 ± 13.7%` (`87.9 ms`) | `262.7 ± 10.9%` (`93.7 ms`) | `-6.1%` (same) | `560.9` (`35.9 ms`) |
| `code-c16` | `446.7 ± 12.4%` (`110.8 ms`) | `448.7 ± 10.8%` (`110.2 ms`) | `+0.5%` (same) | `873.9` (`45.2 ms`) |

- **DSpark acceptance:** `prose` `2.361 tok/step` (`0.272` acc/ver, full 5-draft verification) and `code` `3.161 tok/step` (`0.432` acc/ver), matching full-depth DSpark on `tp4-ring`.

## Conclusion

Config-driven `PP=2, TP=[2-4]` is proven end-to-end on DeepSeek-V4.1-Flash over full 200 GbE RoCE (`oneshot-pp2-ring4`) with disk Engram, multi-chunk CED prefill (`32K`/`131K`), FULL CUDA graphs, DSpark 5-token speculative decoding, and `LRU 5/5` quality.

- On **4 Sparks**, `PP=2, TP=2` is expectedly ~44–52% slower than `PP=1, TP=4`: each stage has half the TP memory bandwidth (`2x` weight bytes per Spark per layer, so `c1` is `56.7 ms` vs `25.1 ms`), and at `c8`/`c16` MoE expert saturation means splitting into two `4+4` / `8+8` microbatches reads nearly all active experts twice per cycle relative to `TP=4` reading them once across 4 ranks (`354–367` vs `657 tok/s` prose; `449` vs `874 tok/s` code).
- Balancing V2+PP+async decode microbatches (`0003`) eliminates lockstepped pipeline bubbles at low concurrency (`code-c2` `+24.7%`, `prose-c2` `+15.6%` with `c2`/`c4` variance collapsing from `±20%` to `±1.7%`, achieving `1.90x` `c1 -> c2` scaling).
- All `PP=2` runtime, fabric, and vLLM overlay pieces are now validated live on 200 GbE RoCE and ready to run **`pp2-tp3-switched.json`** (6 Sparks) and **`pp2-tp4-switched.json`** (8 Sparks, where each stage retains full `TP=4` `~12.5 ms` bandwidth without 8-way TP collective overhead) as soon as switched hardware returns. Production `tp4-ring` (`config/cluster.json`, `spark-ds41f:1120`) was restored cleanly after testing.
