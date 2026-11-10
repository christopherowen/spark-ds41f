# 2026-11-08 — TP=2 Native-Weight Zero-Copy `io_uring` Routed MoE Expert Slot Cache

## Summary

- **Base deployment commit**: `cc20ed18b` (`1107` vLLM patch series head `0761` / `320021e3c4dbb202d911ab70e2cd924c75420488`)
- **Candidate patch**: `experiments/2026-11-08-tp2-nvme-expert-slot-cache/vllm/0762-feat-dsv41-add-zero-copy-io_uring-routed-MoE-expert-.patch` (patch head `792d5df648324a076808ebfee6a3c9fd8b0e4392`, tree `9031d0e65d639e94243830d3106116b854865700`)
- **Causal variable**: Add a zero-copy `cudaHostAlloc` (`pin_memory`) + `O_DIRECT` `io_uring` (`IORING_OP_READ_FIXED` + `IOSQE_FIXED_FILE`) routed MoE expert slot manager (`vllm/models/deepseek_v4_1/expert_slots/`) with:
  1. In-kernel `ExpertToSlot[e]` dispatch remapping (`slot_map=True` in `moe_dispatch_v41`),
  2. Two-phase role-major `w13`/`w2` NVMe streaming (`wait_w2` callback between `moe_swiglu_cast` and `down_proj`),
  3. LBA-sorted multi-miss scheduling & contiguous slot pairing (`allocate_dynamic_slots_lba` + C `qsort` by ascending expert file offset),
  4. `CountDecayLFU` replacement with cold-prefill protection and unread lookahead probation (`probationary=True`, `score=0.0` for speculative prefetches until demanded by the router),
  5. Router-bias (`e_score_correction_bias`) / hotlist static seeding (`rank_experts_by_router_bias`), layer-adaptive Shannon-entropy slot budgeting (`plan_layer_slot_budgets`), and UMA `/proc/meminfo` `MemAvailable` auto-budgeting (`compute_uma_slot_budget`),
  6. Zero-staging chunked Boot 1 preparation (`populate_slots_from_chunk_bytes`) + background `O_DIRECT` layer-file writer (`write_layer_expert_file` / `ExpertDiskCache`) with an `8 GiB` free-disk-space safety floor and atomic `<layer>.tmp.<pid>` publication (`loading_done()`), plus Boot 2+ cached startup (`fill_from_disk_cache`) with 4-expert byte-for-byte spot verification (`verify_spot_experts` on `[0, 1, sampled, E - 1]`) and automatic rebuild on mismatch.
- **Outcome**:
  - **Memory footprint at `TP=2` (`H=5120, I=1152, E=384, num_slots=288, static_slots=240, dynamic_slots=48`)**: Reduces resident MoE weight memory per layer from `3,612,575,232 B` (`3.364 GiB/layer`) to `2,629,804,032 B` (`2.449 GiB/layer`), saving **`937.24 MiB/layer`, `36.61 GiB/rank` across the model's 40 MoE layers**. That is not enough on its own: at `288` slots a rank still needs about `108 GiB` of weights against about `99 GiB` for weights and KV together, so no KV fits. About `256` slots per layer or fewer are needed (section 1). The receipts' `saved_gib_58_layers_per_rank` (`53.086`) multiplies by 58 layers, which this model does not have; section 1 supersedes it.
  - **Not yet served**: no two-node `TP=2` run has happened. Nothing in serving sets a layer's `slot_cache_config`, so the cache runs only in its tests and this benchmark.
  - **Bitwise parity**: `torch.equal(out_ref, out_slot) == True` (`max_abs_diff = 0.0`) across all-hit decode (`rows=1`, `rows=5`), role-major `w13`/`w2` overlapped cold misses, LBA-sorted multi-miss reads, cross-layer router lookahead prefetches, multi-window prefill batches (`>512` routes), and Boot 2+ warm cached startup.
  - **Startup & disk-cache speedup (`boot-1` & `boot-2` on `sp6`)**: Zero-staging chunked Boot 1 preparation (`chunk_experts=64`) + background `O_DIRECT` write completes main-thread layer build in `24.90 s` / `21.40 s` (`25.39 s` / `21.91 s` after `loading_done()`, vs `32.78 s` synchronous write previously), while Boot 2+ cached start with 4-expert byte-for-byte spot verification (`[0, 1, sampled, 383]`, skipping the remaining `380` experts) + LBA-sorted `O_DIRECT` `io_uring` static-slot fill completes in **`1.635 s` / `1.733 s` (`15.53x` / `12.64x` speedup, mean `14.09x`)**.
  - **All-hit decode latency (`boot-1` & `boot-2` on `sp6`)**: Zero latency penalty (`242.06 µs` / `240.88 µs` for 288-slot zero-copy `pin_memory` vs `251.62 µs` / `250.89 µs` for 384-expert `cudaMalloc` at `rows=1`; `1161.05 µs` / `1139.14 µs` vs `1171.36 µs` / `1171.65 µs` at `rows=5`).
  - **Router-bias static seeding (`rank_experts_by_router_bias` on real `DeepSeek-V4.1-Flash` `layers.0.ffn.gate.bias`)**: `86` of the top `240` highest-bias experts in Layer 0 have expert index $\ge 240$ (including 6 of the top 10: `[331, 50, 364, 279, 169, 291, 65, 41, 381, 367]`). Seeding `static_slots` from `e_score_correction_bias` raises 200-step router-driven hit rate from `94.83%` (`62` misses with naive `0..239` seeding) to **`100.00%` (`0` misses, `100.0%` miss reduction)**.
  - **Unread lookahead probation (`probationary=True, score=0.0`)**: Under noisy cross-layer lookahead predictions (2 true + 4 false tail predictions per step alongside a 44-expert recurring decode working set in `48` dynamic slots), probationary lookahead raises demand hit rate from `33.50%` (`798` misses with eager lookahead scoring) to **`68.00%` (`384` misses, `-51.9%` demand misses)**.
  - **LBA-sorted multi-miss scheduling (`allocate_dynamic_slots_lba` + `_reader.c` `qsort`)**: Issues `100%` (`146 / 146`) of multi-miss `io_uring` batches in strictly ascending LBA file-offset order (`92` contiguous destination slot pairs), achieving **`8.74–9.14 GB/s`** effective NVMe bandwidth on 4-expert (`34.844 MiB`) cold misses.

---

## 1. TP=2 Native-Weight Sharding & Memory Budget on Two 128 GiB GB10 Nodes

Every dimension of `DeepSeek-V4.1-Flash` divides cleanly by `TP=2` with zero head padding:

| Architectural Dimension | Full Model | `TP=3` (`tp3-triangle`) | `TP=2` (2× DGX Spark) |
|---|---:|---:|---:|
| Decoder layers (`num_hidden_layers`), every one MoE | `40` (plus 3 DSpark drafter layers) | `40` | **`40`** |
| Attention query heads (`num_attention_heads`) | `64` | `24 /rank` (padded to `72`) | **`32 /rank` (exact)** |
| Output groups (`o_groups`) | `8` | `3 /rank` (padded to `9`) | **`4 /rank` (exact)** |
| KV latent (`num_key_value_heads`) | `1` (512 wide) | replicated | **replicated** |
| Indexer heads (`index_n_heads`) | `32` | replicated | **replicated** |
| MoE `moe_intermediate_size` | `2,304` | `768 /rank` (`6 × 128`) | **`1,152 /rank` (`9 × 128`, zero tail waste)** |
| Shared expert `intermediate_size` | `2,304` | `768 /rank` | **`1,152 /rank` (exact)** |
| Vocabulary (`vocab_size`) | `129,280` | `43,094 /rank` | **`64,640 /rank` (exact)** |

### Per-Rank Memory Breakdown at `TP=2`

At `TP=2` (`hidden_size = 5120`, per-rank `moe_intermediate_size = 1152`), each routed expert shard has:
- `w13_tiles` (`gate_up_proj`, `[2304, 2560]` packed NVFP4 `e2m1` tiled in `128 × 64 B` TMA tiles): `5,898,240 B` (`5.625 MiB`, `1,440 × 4 KiB` pages)
- `w2_tiles` (`down_proj`, `[5120, 576]` packed NVFP4 `e2m1` tiled in `128 × 64 B` TMA tiles): `2,949,120 B` (`2.8125 MiB`, `720 × 4 KiB` pages)
- `w13_codes` (`uint8` dictionary scale codes, `narrow_codes=True`): `184,320 B` (`180 KiB`)
- `w2_codes` (`uint8` dictionary scale codes, `narrow_codes=True`): `92,160 B` (`90 KiB`)
- `w13_base` (`2,304 B`) + `w2_base` (`5,120 B`): `7,424 B`
- Page-aligned role-major pack record (`expert_stride`):
  - `w13` role slab: `5,898,240 B` (`w13_tiles`) + `188,416 B` (`w13_meta`, 4 KiB aligned) = `6,086,656 B`
  - `w2` role slab: `2,949,120 B` (`w2_tiles`) + `98,304 B` (`w2_meta`, 4 KiB aligned) = `3,047,424 B`
  - Total per-expert pack stride: **`9,134,080 B` (`8.711 MiB` = `2,230 × 4 KiB` pages)**

A resident slot holds `9,131,264 B` (the tiles, codes and bases above, without the pack file's 4 KiB alignment: `2,629,804,032 B / 288`). One slot in each of the 40 MoE layers is `365,250,560 B` (`0.340 GiB`) per rank.

The budget is built from measured anchors and two estimates:

- **Other weights, about `10.4 GiB` per rank (estimate).** A ring rank loads `72.79 GiB` of weights (`Model loading took`, [2026-10-26-tp3-budget-memory](../2026-10-26-tp3-budget-memory/README.md)). At TP4 the routed experts in this benchmark's full layout are half its layer, `1,806,287,616 B × 40 = 67.29 GiB`, which leaves `5.50 GiB` of other weights. About `0.6 GiB` of that is replicated (Q-A/KV, router, indexer, compressor); the rest is sharded and doubles at TP2.
- **Weights plus KV, about `99.0 GiB` per rank (estimate).** The triangle at an 8,192-token prefill budget loads `94.12 GiB` of weights beside a `4.91 GiB` KV pool, and its lowest free memory is `6.67 GiB` (same experiment). Keeping that low point, a TP2 rank at the same prefill budget has about `99.0 GiB` for weights and KV together; a 16,384 budget costs about `3.5 GiB` more. TP2's wider per-rank activations are not measured, so this is an upper estimate.
- **KV: `1,100,393` tokens per GiB per rank**, from the ring's startup line (`11,554,122` tokens in `11,274,289,152 B`). Every rank holds the whole KV latent, so this does not change with TP: a full 1M context is `0.95 GiB` per rank.

| Resident slots per layer | Routed experts / rank | All weights / rank | Left for KV / rank | KV tokens | Full 1M contexts | 100K sessions |
|---|---:|---:|---:|---:|---:|---:|
| `384`, all resident (`134.58 GiB` in the production layout with `uint16` codes) | `130.62 GiB` | `141.0 GiB` | `-42.0 GiB`: does not fit | — | — | — |
| `288`, this benchmark (`240` static + `48` dynamic) | `97.97 GiB` | `108.4 GiB` | `-9.3 GiB`: does not fit | — | — | — |
| `256` | `87.08 GiB` | `97.5 GiB` | `1.6 GiB` | `1.7M` | `1` | `17` |
| `248` | `84.36 GiB` | `94.8 GiB` | `4.3 GiB` | `4.7M` | `4` | `47` |
| `240` | `81.64 GiB` | `92.0 GiB` | `7.0 GiB` | `7.7M` | `7` | `77` |
| `232` | `78.92 GiB` | `89.3 GiB` | `9.7 GiB` | `10.7M` | `10` | `107` |
| `224` | `76.20 GiB` | `86.6 GiB` | `12.4 GiB` | `13.7M` | `13` | `137` |

Each slot per layer trades `0.340 GiB`, about `374K` KV tokens. At most `257` slots per layer leave room for one full 1M context. For comparison, the four-node ring holds `11.55M` KV tokens (`11.02` full 1M contexts) with every expert resident.

An earlier version of this table counted 58 MoE layers, 24.08 GiB of non-routed weights and a 115 GiB budget. DS4.1 Flash has 40 decoder layers, all MoE, and no dense MLP layers.

---

## 2. Architecture & Implementation

1. **In-Kernel `ExpertToSlot[e]` Remapping (`vllm/models/deepseek_v4_1/tilelang/moe.py`)**:
   - `moe_dispatch_v41(..., slot_map=True)` accepts `ExpertToSlot: T.Tensor((num_experts,), "int32")`.
   - Routes (`Order`, `RoutePos`, `LayoutWeights`) are still grouped by logical expert ID `e in 0..num_experts-1` so reduction order in `moe_combine` is untouched (`torch.equal`), while `TileExpert[offset // block_M + t] = ExpertToSlot[tx]` translates each emitted tile's expert index to its physical slot `s in 0..num_slots-1` inside the dispatch kernel itself (`0` extra kernel launches).
   - `MoeExpertsV41Plan` adds `num_slots`, `narrow_codes`, `slot_map`, and an optional `wait_w2: Callable[[], None]` hook invoked between `swiglu_kernel` and `down_kernel`.
2. **Zero-Copy `cudaHostAlloc` (`pin_memory`) + Role-Major LBA-Sorted `O_DIRECT` `io_uring` (`READ_FIXED`) Reader (`vllm/models/deepseek_v4_1/expert_slots/`)**:
   - `alloc_pinned_aligned` allocates 4 KiB-aligned `pin_memory` storage and wraps it in a zero-copy CUDA tensor view (`as_cuda_zero_copy` via `__cuda_array_interface__`) sharing the identical unified LPDDR5x physical address (`host.data_ptr() == cuda.data_ptr()`).
   - `_reader.c` opens the layer pack file with `O_DIRECT`, registers the file descriptor via `io_uring_register_files` (`IOSQE_FIXED_FILE`), and registers `2 * dynamic_slots + 1` fixed buffers via `io_uring_register_buffers` (`IORING_OP_READ_FIXED`) covering each dynamic slot's `w13_tiles` (`5.625 MiB`), each dynamic slot's `w2_tiles` (`2.8125 MiB`), and the 4 KiB-aligned metadata staging buffer (`_meta_staging_host`).
   - When multiple experts miss in a single step ($M > 1$), `CountDecayLFU.allocate_dynamic_slots_lba` sorts both the missed logical expert IDs (`e_0 < e_1 < ... < e_{M-1}`, ascending LBA file offset) and the selected victim dynamic slots (`s_0 < s_1 < ... < s_{M-1}`, ascending physical buffer address) before binding `e_k -> s_k`, and `py_submit_reads` in `_reader.c` sorts request pairs by `(expert, dyn_slot)` so all `ROLE_W13` SQEs and `ROLE_W2` SQEs are issued in strictly monotonic ascending file-offset order (`lba_ordered_submissions` and `contiguous_slot_pairs` tracked in `reader_stats()`).
   - `ensure_active()` waits only for `ROLE_W13` before `gate_up_proj` and returns `wait_w2` so `w2_tiles` and `w2_meta` transfer over NVMe concurrently with `gate_up_proj` and `moe_swiglu_cast`.
3. **`CountDecayLFU` with Cold-Prefill Protection & Unread Lookahead Probation (`cache.py`)**:
   - Static slots `0..static_slots-1` stay permanently pinned; dynamic slots `static_slots..num_slots-1` are managed by `CountDecayLFU`. During prefill (`rows >= prefill_protect_rows`), dynamic slots whose decode score exceeds `hot_threshold` are protected from one-shot cold prefill eviction unless all unprotected dynamic slots are already active.
   - `prefetch_async(predicted_experts)` binds newly prefetched dynamic slots with `probationary=True`, `score=0.0`, and `last_used=0`, protecting them only until the next `ensure_active()` call on that layer. Inside `ensure_active()`, any probationary slot actually demanded by the router is promoted (`promote_probationary`, `lookahead_promoted += 1`) and receives a normal demand LFU score bump, while unconsumed lookahead slots have `probationary=False` cleared (`clear_probation`, `lookahead_evictable += count`) while retaining `score=0.0, last_used=0` so mispredicted lookahead reads are first in line for eviction and never displace warm decode experts.
4. **Router-Bias Static Seeding, Layer-Adaptive Entropy Slot Planner, & UMA `MemAvailable` Auto-Budgeting (`cache.py`)**:
   - `rank_experts_by_router_bias(e_score_correction_bias, gate_weight=None, hotlist_counts=None)` ranks experts `0..E-1` from highest to lowest routing prior so each layer's `static_slots` pin the checkpoint's true high-bias experts (wired automatically via `router_bias` / `layer.e_score_correction_bias` in `TileLangExperts`).
   - `plan_layer_slot_budgets(layer_biases, total_slot_budget, min_slots_per_layer=64, max_slots_per_layer=384, dynamic_slots_per_layer=48)` distributes a fixed global slot budget across MoE layers proportionally to each layer's routing Shannon entropy so concentrated low-entropy layers use fewer static slots and diffuse high-entropy layers receive more static slots at the identical total memory footprint.
   - `compute_uma_slot_budget(num_layers=58, expert_bytes=9_134_080, non_routed_bytes=0, kv_reserve_bytes=0, system_reserve_bytes=8*(1<<30))` reads Linux `/proc/meminfo` `MemAvailable` on integrated-memory GB10 hosts, subtracts system, KV, and non-routed reserves, and returns a clamped `[64, 384]` per-layer slot budget. Its `num_layers` default is `58`; DS4.1 Flash has 40 MoE layers, so a caller must pass `num_layers=40` (nothing in serving calls it yet).
5. **Zero-Staging Chunked Build, Background `O_DIRECT` Layer Writer, & 4-Expert Spot-Checked Cached Start (`cache.py` + `_reader.c` + `tilelang.py`)**:
   - On Boot 1 (when the per-layer NVMe file does not yet exist), `ExpertDiskCache._build_layer_chunked` prepares experts in bounded slices (`chunk_experts=64`) so no full-384-expert staging tensor sits in memory at once, populates resident `ExpertSlotCache` slots directly in RAM (`populate_slots_from_chunk_bytes`), and enqueues the `4 KiB` metadata header (`DS41EXPK` magic, format version, `ExpertPackMeta` geometry, and `checkpoint_key`) plus `4 KiB`-aligned expert chunks onto a single background writer thread (`write_layer_expert_file`).
   - `write_layer_expert_file` verifies `shutil.disk_usage(parent).free` remains above the `8 GiB` safety floor (`MIN_FREE_DISK_BYTES`), streams `4 KiB`-aligned pages through an `mmap` bounce buffer to `<layer>.tmp.<pid>` with `O_DIRECT` (falling back cleanly to buffered write + `fsync` when `O_DIRECT` is unsupported), and atomically publishes via `os.replace`. `ExpertDiskCache.loading_done()` waits for the background queue to drain and cleans up any stale `.tmp.<dead_pid>` files (`cleanup_stale_tmp_files`).
   - On Boot 2+ (when the per-layer NVMe file exists), `verify_spot_experts` validates the `4 KiB` header and file size (`4,096 + 384 × 9,134,080 = 3,507,490,816 B`) and performs a **4-expert byte-for-byte spot check** on `[0, 1, sampled, E - 1]` (`select_spot_check_experts`), running `MoeExpertsV41Plan.prepare_weight` on only those 4 experts (`1.04%` of the layer) and comparing their `9,134,080 B` records byte-for-byte against the NVMe file while skipping the other `380` experts. On pass, `ExpertSlotCache.fill_from_disk_cache(..., static_only=True)` reads the `static_slots` router-bias-ranked experts in LBA-sorted order via `O_DIRECT` `io_uring` (`READ_FIXED`). On any header or byte mismatch, `ExpertDiskCache` logs a warning and transparently rebuilds and atomically replaces the layer file from all `384` experts.

---

## 3. Verification & 2-Boot Benchmark Results (`sp6`, `spark-ds41f:1107`)

### Exact Commands

Unit and regression tests (`sp6`):
```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined \
  --entrypoint pytest \
  -v /tmp/vllm-tp2-slot-cache/vllm/models/deepseek_v4_1/tilelang/moe.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py:ro \
  -v /tmp/vllm-tp2-slot-cache/vllm/models/deepseek_v4_1/expert_slots:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/expert_slots:ro \
  -v /tmp/vllm-tp2-slot-cache/vllm/model_executor/layers/fused_moe/tilelang.py:/opt/spark-ds41f/candidate/vllm/vllm/model_executor/layers/fused_moe/tilelang.py:ro \
  -v /tmp/vllm-tp2-slot-cache/tests/kernels/moe/test_deepseek_v41_expert_slots.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/moe/test_deepseek_v41_expert_slots.py:ro \
  spark-ds41f:1107 --noconftest -o addopts="" \
  /opt/spark-ds41f/candidate/vllm/tests/kernels/moe/test_deepseek_v41_expert_slots.py \
  /opt/spark-ds41f/candidate/vllm/tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py -v
```
- `tests/kernels/moe/test_deepseek_v41_expert_slots.py` (`12 passed`) + `tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py` (`29 passed`): **`41 passed`**
- `tests/kernels/moe/test_deepseek_v41_tilelang_moe.py`: **`139 passed`**

Two-boot `TP=2` geometry benchmark (`E=384, top_k=6, H=5120, I=1152, num_slots=288, static_slots=240, dynamic_slots=48`) on idle `sp6` against real `DeepSeek-V4.1-Flash` `layers.{0..39}.ffn.gate.bias` checkpoint tensors:
```bash
REAL_SF=$(readlink -f /home/chris/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170/model-00001-of-00048.safetensors)
for b in boot-1 boot-2; do
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined --entrypoint python3 \
    -v /tmp/vllm-tp2-slot-cache/vllm/models/deepseek_v4_1/tilelang/moe.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py:ro \
    -v /tmp/vllm-tp2-slot-cache/vllm/models/deepseek_v4_1/expert_slots:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/expert_slots:ro \
    -v /tmp/vllm-tp2-slot-cache/vllm/model_executor/layers/fused_moe/tilelang.py:/opt/spark-ds41f/candidate/vllm/vllm/model_executor/layers/fused_moe/tilelang.py:ro \
    -v /tmp/vllm-tp2-slot-cache/bench_tp2_expert_slots_v2.py:/tmp/bench_tp2_expert_slots_v2.py:ro \
    -v "$REAL_SF":/mnt/ckpt_shard1.safetensors:ro \
    spark-ds41f:1107 /tmp/bench_tp2_expert_slots_v2.py "$b"
done
```

### Measured Results (`boot-1` and `boot-2` on `sp6`, `receipts/tp2_boot1.json` & `receipts/tp2_boot2.json`)

| Metric (`E=384, top_k=6, H=5120, I=1152, slots=288 [240 static + 48 dynamic]`) | `boot-1` | `boot-2` | Mean |
|---|---:|---:|---:|
| Full 384-expert layer bytes | `3,612,575,232 B` (`3.364 GiB`) | `3,612,575,232 B` (`3.364 GiB`) | `3.364 GiB` |
| 288-slot zero-copy layer bytes | `2,629,804,032 B` (`2.449 GiB`) | `2,629,804,032 B` (`2.449 GiB`) | `2.449 GiB` |
| Saved per rank across the 40 MoE layers (the receipts' `saved_gib_58_layers_per_rank` multiplies by 58) | `36.61 GiB` | `36.61 GiB` | **`36.61 GiB`** |
| Bitwise parity (`all_hit` & `cold_miss_overlapped`, `rows=1` & `rows=5`) | `max_abs_diff = 0.0` (`True`) | `max_abs_diff = 0.0` (`True`) | **`0.0` (exact)** |
| All-hit decode `rows=1`: 384-expert `cudaMalloc` | `251.62 µs` | `250.89 µs` | `251.26 µs` |
| All-hit decode `rows=1`: 288-slot zero-copy `pin_memory` | `242.06 µs` | `240.88 µs` | **`241.47 µs` (`-9.79 µs`)** |
| All-hit decode `rows=5` (MTP-5): 384-expert `cudaMalloc` | `1171.36 µs` | `1171.65 µs` | `1171.51 µs` |
| All-hit decode `rows=5` (MTP-5): 288-slot zero-copy `pin_memory` | `1161.05 µs` | `1139.14 µs` | **`1150.10 µs` (`-21.41 µs`)** |
| 1-expert cold miss (`8.711 MiB`) — synchronous full wait (`p50`) | `1.604 ms` | `1.601 ms` | `1.603 ms` |
| 1-expert cold miss (`8.711 MiB`) — **role-major `w13`/`w2` split wait** (`p50`) | **`1.375 ms` (`-14.3%`)** | **`1.462 ms` (`-8.7%`)** | **`1.419 ms` (`-11.5%`)** |
| 1-expert cold miss (`8.711 MiB`) — **cross-layer router lookahead prefetch** (`p50`) | **`0.379 ms` (`-76.4%`)** | **`0.366 ms` (`-77.1%`)** | **`0.373 ms` (`-76.7%`)** |
| 4-expert LBA-sorted cold miss (`34.844 MiB`) — sync / role-split / lookahead (`p50`) | `4.179` / `4.029` / `2.658 ms` (`8.74 GB/s`) | `3.999` / `3.867` / `2.448 ms` (`9.14 GB/s`) | **`4.089` / `3.948` / `2.553 ms` (`8.94 GB/s`)** |
| LBA-ordered multi-miss submissions (`lba_ordered_submissions / total_submissions`) | `146 / 146` (`92` contiguous pairs) | `146 / 146` (`92` contiguous pairs) | **`100.0%` (`92` contiguous pairs)** |
| 200-step Zipfian ($\alpha=1.15$) trace + step-100 cold-prefill burst: route hit rate | `90.19%` (`1250 / 1386`) | `90.19%` (`1250 / 1386`) | **`90.19%`** |
| 200-step Zipfian ($\alpha=1.15$) trace: step `p50` / `mean` latency | `0.346 ms` / `0.838 ms` | `0.339 ms` / `0.679 ms` | **`0.343 ms` / `0.759 ms`** |
| **Router-bias static seeding** (`layers.0.ffn.gate.bias`, `86/240` top experts $\ge 240$): naive `0..239` vs `rank_experts_by_router_bias` hit rate | `94.83%` (`62` misses) $\to$ **`100.00%` (`0` misses)** | `94.83%` (`62` misses) $\to$ **`100.00%` (`0` misses)** | **`+5.17 pp` (`100.0%` miss reduction)** |
| **Unread lookahead probation** (noisy lookahead: 2 true + 4 false tail predictions + 44 warm decode experts): eager vs probationary demand hit rate | `33.50%` (`798` misses) $\to$ **`68.00%` (`384` misses)** | `33.50%` (`798` misses) $\to$ **`68.00%` (`384` misses)** | **`+34.50 pp` (`-51.9%` demand misses)** |
| **Layer-adaptive entropy budget planner** (`40` checkpoint layers, `11,520` total slots) | Min `258` slots (L34) .. Max `303` slots (L0), sum `11,520` | Min `258` slots (L34) .. Max `303` slots (L0), sum `11,520` | **Exact `11,520`-slot conservation** |
| **Cold Boot 1 chunked build (`chunk_experts=64`)**: main-thread ready / background `O_DIRECT` write drained (`3,507,490,816 B`) | `24.903 s` / `25.393 s` (`384` experts prepared) | `21.400 s` / `21.906 s` (`384` experts prepared) | **`23.152 s` / `23.650 s`** |
| **Warm Boot 2+ cached start (`verify_spot_experts` on 4 experts + `O_DIRECT` `io_uring` 240-slot static fill)** | **`1.635 s`** (`[0, 1, 169, 383]` checked, `380` skipped) | **`1.733 s`** (`[0, 1, 96, 383]` checked, `380` skipped) | **`1.684 s` (`14.09x` speedup)** |

### Errors Encountered & Resolved During Validation

1. **Initial population metadata staging size (`Bad address` at slot 58)**:
   - On the first 288-slot `TP=2` run, `_meta_staging_host` was sized to `dynamic_slots * meta_stride` (`48 × 8,192 B = 393,216 B`), while initial population opened a temporary reader with `dynamic_slots = num_slots = 288`. Fixed in `cache.py` and `_reader.c` by raising `RING_DEPTH` to `1024u`, allocating a temporary `num_slots * meta_stride` (`2.25 MiB`) staging buffer for the initial population pass (freed immediately after closing the initial reader), and submitting initial population reads in batches of `128` slots (`512` SQEs $\le 1024$).

---

## 4. Conclusion

The zero-copy `pin_memory` + role-major `O_DIRECT` `io_uring` (`IORING_OP_READ_FIXED`) expert slot cache reduces per-rank resident MoE weight memory at `TP=2` by `36.61 GiB` across the 40 MoE layers at `288` resident slots per layer (not yet enough to leave room for KV: about `256` slots or fewer are needed, section 1), preserves bit-identical output (`torch.equal`), matches or slightly improves all-hit decode latency on GB10 unified LPDDR5x, cuts 1-expert cold-miss latency by `11.5%` via role-major `w13`/`w2` split waiting and by `76.7%` via cross-layer router lookahead prefetching, reaches `8.74–9.14 GB/s` on 4-expert LBA-sorted multi-miss reads, eliminates `100%` of cold-start static misses on Layer 0 router-bias traces via `rank_experts_by_router_bias`, and doubles demand hit rate (`33.50% -> 68.00%`) under noisy lookahead via unread lookahead probation.
