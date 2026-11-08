# 2026-11-08 — TP=2 Native-Weight Zero-Copy `io_uring` Routed MoE Expert Slot Cache

## Summary

- **Base deployment commit**: `cc20ed18b` (`1107` vLLM patch series head `0761` / `320021e3c4dbb202d911ab70e2cd924c75420488`)
- **Candidate patch**: `experiments/2026-11-08-tp2-nvme-expert-slot-cache/vllm/0762-feat-dsv41-add-zero-copy-io_uring-routed-MoE-expert-.patch`
- **Causal variable**: Add a zero-copy `cudaHostAlloc` (`pin_memory`) + `O_DIRECT` `io_uring` (`IORING_OP_READ_FIXED` + `IOSQE_FIXED_FILE`) routed MoE expert slot manager (`vllm/models/deepseek_v4_1/expert_slots/`) with in-kernel `ExpertToSlot[e]` dispatch remapping (`slot_map=True` in `moe_dispatch_v41`), two-phase role-major `w13`/`w2` NVMe streaming (`wait_w2` callback between `moe_swiglu_cast` and `down_proj`), `CountDecayLFU` replacement with cold-prefill protection, and cross-layer router lookahead prefetching (`RouterLookaheadPredictor`).
- **Outcome**:
  - **Memory footprint at `TP=2` (`H=5120, I=1152, E=384, num_slots=288, static_slots=240, dynamic_slots=48`)**: Reduces resident MoE weight memory per layer from `3,612,575,232 B` (`3.364 GiB/layer`) to `2,629,804,032 B` (`2.449 GiB/layer`), saving **`937.24 MiB/layer` (`53.086 GiB/rank` across all 58 MoE layers)** and bringing total per-rank native-weight footprint on a two-node `TP=2` GB10 cluster from `123.75 GiB/rank` down to **`94.83 GiB/rank`** (`~20.2 GiB/rank` headroom on a `115 GiB` usable GB10 node).
  - **Bitwise parity**: `torch.equal(out_ref, out_slot) == True` (`max_abs_diff = 0.0`) across all-hit decode (`rows=1`, `rows=5`), role-major `w13`/`w2` overlapped cold misses, cross-layer router lookahead prefetches, and multi-window prefill batches (`>512` routes).
  - **All-hit decode latency (`boot-1` & `boot-2` on `sp5`)**: Zero latency penalty (`247.12 µs` / `252.46 µs` for 288-slot zero-copy `pin_memory` vs `250.86 µs` / `252.30 µs` for 384-expert `cudaMalloc` at `rows=1`; `1146.14 µs` / `1171.82 µs` vs `1173.45 µs` / `1176.56 µs` at `rows=5`).
  - **Cold-miss latency & overlap (`boot-1` & `boot-2` on `sp5`)**: Role-major `w13`/`w2` split wait reduces 1-expert (`8.711 MiB`) cold-miss latency by **`15.0–16.2%`** (`1.079 ms -> 0.917 ms` on `boot-1`, `1.466 ms -> 1.229 ms` on `boot-2`), and cross-layer router lookahead prefetching reduces it by **`68.0–75.0%`** (`0.345 ms` on `boot-1`, `0.366 ms` on `boot-2`).

---

## 1. TP=2 Native-Weight Sharding & Memory Budget on Two 128 GiB GB10 Nodes

Every dimension of `DeepSeek-V4.1-Flash` divides cleanly by `TP=2` with zero head padding:

| Architectural Dimension | Full Model | `TP=3` (`tp3-triangle`) | `TP=2` (2× DGX Spark) |
|---|---:|---:|---:|
| Attention query heads (`num_attention_heads`) | `40` | `14 /rank` (padded to `42`) | **`20 /rank` (exact)** |
| Attention KV heads (`num_key_value_heads`) | `8` | `3 /rank` (padded to `9`) | **`4 /rank` (exact)** |
|DSA Indexer heads (`index_n_heads`) | `64` | `22 /rank` (padded to `66`) | **`32 /rank` (exact)** |
| Dense MLP `intermediate_size` | `12,288` | `4,096 /rank` | **`6,144 /rank` (exact)** |
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

| Component (58 MoE layers, `TP=2`) | 100% Resident (`384` experts) | `288` Slots (`75%`: `240` static + `48` dynamic) | `256` Slots (`66.7%`: `208` static + `48` dynamic) |
|---|---:|---:|---:|
| Replicated + sharded non-routed weights (`embed`, MLA/DSA, dense MLPs, shared experts, routers, mHC, `pack12` `lm_head`, DSpark drafter) | `24.08 GiB /rank` | `24.08 GiB /rank` | `24.08 GiB /rank` |
| Routed MoE experts across 58 layers (`narrow_codes=True`) | `94.71 GiB /rank` | **`71.03 GiB /rank`** | **`63.14 GiB /rank`** |
| **Total Model Weights per Rank** | **`118.79 GiB /rank`** (`123.75 GiB` w/ `uint16` codes) | **`95.11 GiB /rank`** (**`-28.64 GiB`** vs `uint16` 384-exp) | **`87.22 GiB /rank`** (**`-36.53 GiB`** vs `uint16` 384-exp) |
| **Remaining Headroom on `115 GiB` Usable GB10 Node** | `-3.79 GiB` (OOM) | **`+19.89 GiB /rank`** | **`+27.78 GiB /rank`** |

---

## 2. Architecture & Implementation

1. **In-Kernel `ExpertToSlot[e]` Remapping (`vllm/models/deepseek_v4_1/tilelang/moe.py`)**:
   - `moe_dispatch_v41(..., slot_map=True)` accepts `ExpertToSlot: T.Tensor((num_experts,), "int32")`.
   - Routes (`Order`, `RoutePos`, `LayoutWeights`) are still grouped by logical expert ID `e in 0..num_experts-1` so reduction order in `moe_combine` is untouched (`torch.equal`), while `TileExpert[offset // block_M + t] = ExpertToSlot[tx]` translates each emitted tile's expert index to its physical slot `s in 0..num_slots-1` inside the dispatch kernel itself (`0` extra kernel launches).
   - `MoeExpertsV41Plan` adds `num_slots`, `narrow_codes`, `slot_map`, and an optional `wait_w2: Callable[[], None]` hook invoked between `swiglu_kernel` and `down_kernel`.
2. **Zero-Copy `cudaHostAlloc` (`pin_memory`) + Role-Major `O_DIRECT` `io_uring` (`READ_FIXED`) Reader (`vllm/models/deepseek_v4_1/expert_slots/`)**:
   - `alloc_pinned_aligned` allocates 4 KiB-aligned `pin_memory` storage and wraps it in a zero-copy CUDA tensor view (`as_cuda_zero_copy` via `__cuda_array_interface__`) sharing the identical unified LPDDR5x physical address (`host.data_ptr() == cuda.data_ptr()`).
   - `_reader.c` opens the layer pack file with `O_DIRECT`, registers the file descriptor via `io_uring_register_files` (`IOSQE_FIXED_FILE`), and registers `2 * dynamic_slots + 1` fixed buffers via `io_uring_register_buffers` (`IORING_OP_READ_FIXED`) covering each dynamic slot's `w13_tiles` (`5.625 MiB`), each dynamic slot's `w2_tiles` (`2.8125 MiB`), and the 4 KiB-aligned metadata staging buffer (`_meta_staging_host`).
   - `submit_reads()` queues all `ROLE_W13` SQEs (`w13_tiles` zero-copy into the GPU tensor + `w13_meta` into staging) ahead of all `ROLE_W2` SQEs in a single `io_uring_submit()` syscall.
   - `ensure_active()` waits only for `ROLE_W13` before `gate_up_proj` and returns `wait_w2` so `w2_tiles` and `w2_meta` transfer over NVMe concurrently with `gate_up_proj` and `moe_swiglu_cast`.
3. **`CountDecayLFU` with Cold-Prefill Protection & Cross-Layer Router Lookahead (`cache.py`)**:
   - Static slots `0..static_slots-1` stay permanently pinned; dynamic slots `static_slots..num_slots-1` are managed by `CountDecayLFU` (exponential decay every `decay_interval` steps). During prefill (`rows >= prefill_protect_rows`), dynamic slots whose decode score exceeds `hot_threshold` are protected from one-shot cold prefill eviction unless all unprotected dynamic slots are already active.
   - `RouterLookaheadPredictor` computes `topk(hidden_states @ next_gate_weight.T)` on layer $L$'s output activation and calls `next_cache.prefetch_async(predicted_experts)` before layer $L+1$'s attention block.

---

## 3. Verification & 2-Boot Benchmark Results (`sp5`, `spark-ds41f:1107`)

### Exact Commands

Unit and regression tests (`sp6` & `sp5`):
```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined \
  --entrypoint pytest -v /tmp/vllm-tp2-slot-cache:/workspace -w /workspace spark-ds41f:1107 \
  -o addopts="" --noconftest \
  tests/kernels/moe/test_deepseek_v41_expert_slots.py \
  tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py \
  tests/kernels/moe/test_deepseek_v41_tilelang_moe.py
```
- `tests/kernels/moe/test_deepseek_v41_expert_slots.py` + `tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py`: **`34 passed`**
- `tests/kernels/moe/test_deepseek_v41_tilelang_moe.py`: **`139 passed`**

Two-boot `TP=2` geometry benchmark (`E=384, top_k=6, H=5120, I=1152, num_slots=288, static_slots=240, dynamic_slots=48`) on idle `sp5`:
```bash
for b in boot-1 boot-2; do
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined \
    -e TILELANG_CACHE_DIR=/tmp/tilelang-cache --entrypoint python3 \
    -v /tmp:/tmp \
    -v /tmp/tp2-overlay/vllm/models/deepseek_v4_1/tilelang/moe.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py:ro \
    -v /tmp/tp2-overlay/vllm/models/deepseek_v4_1/expert_slots:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/expert_slots:ro \
    -v /tmp/tp2-overlay/vllm/model_executor/layers/fused_moe/tilelang.py:/opt/spark-ds41f/candidate/vllm/vllm/model_executor/layers/fused_moe/tilelang.py:ro \
    spark-ds41f:1107 /tmp/tp2-overlay/bench_tp2_expert_slots.py "$b"
done
```

### Measured Results (`boot-1` and `boot-2` on `sp5`)

| Metric (`E=384, top_k=6, H=5120, I=1152, slots=288 [240 static + 48 dynamic]`) | `boot-1` | `boot-2` | Mean |
|---|---:|---:|---:|
| Full 384-expert layer bytes | `3,612,575,232 B` (`3.364 GiB`) | `3,612,575,232 B` (`3.364 GiB`) | `3.364 GiB` |
| 288-slot zero-copy layer bytes | `2,629,804,032 B` (`2.449 GiB`) | `2,629,804,032 B` (`2.449 GiB`) | `2.449 GiB` |
| Saved across 58 MoE layers per rank | `53.086 GiB` | `53.086 GiB` | **`53.086 GiB`** |
| Bitwise parity (`all_hit` & `cold_miss_overlapped`, `rows=1` & `rows=5`) | `max_abs_diff = 0.0` (`True`) | `max_abs_diff = 0.0` (`True`) | **`0.0` (exact)** |
| All-hit decode `rows=1`: 384-expert `cudaMalloc` | `250.86 µs` | `252.30 µs` | `251.58 µs` |
| All-hit decode `rows=1`: 288-slot zero-copy `pin_memory` | `247.12 µs` | `252.46 µs` | **`249.79 µs` (`-1.79 µs`)** |
| All-hit decode `rows=5` (MTP-5): 384-expert `cudaMalloc` | `1173.45 µs` | `1176.56 µs` | `1175.01 µs` |
| All-hit decode `rows=5` (MTP-5): 288-slot zero-copy `pin_memory` | `1146.14 µs` | `1171.82 µs` | **`1158.98 µs` (`-16.03 µs`)** |
| 1-expert cold miss (`8.711 MiB`) — synchronous full wait (`p50`) | `1.079 ms` | `1.466 ms` | `1.273 ms` |
| 1-expert cold miss (`8.711 MiB`) — **role-major `w13`/`w2` split wait** (`p50`) | **`0.917 ms` (`-15.0%`)** | **`1.229 ms` (`-16.2%`)** | **`1.073 ms` (`-15.7%`)** |
| 1-expert cold miss (`8.711 MiB`) — **cross-layer router lookahead prefetch** (`p50`) | **`0.345 ms` (`-68.0%`)** | **`0.366 ms` (`-75.0%`)** | **`0.356 ms` (`-72.0%`)** |
| 200-step Zipfian ($\alpha=1.15$) trace + step-100 cold-prefill burst: route hit rate | `90.12%` (`1249 / 1386`) | `90.12%` (`1249 / 1386`) | **`90.12%`** |
| 200-step Zipfian ($\alpha=1.15$) trace: step `p50` / `mean` latency | `0.390 ms` / `0.589 ms` | `0.342 ms` / `0.580 ms` | **`0.366 ms` / `0.585 ms`** |

### Errors Encountered & Resolved During Validation

1. **Initial population metadata staging size (`Bad address` at slot 58)**:
   - On the first 288-slot `TP=2` run, `_meta_staging_host` was sized to `dynamic_slots * meta_stride` (`48 × 8,192 B = 393,216 B`), while initial population opened a temporary reader with `dynamic_slots = num_slots = 288`. Fixed in `cache.py` and `_reader.c` by raising `RING_DEPTH` to `1024u`, allocating a temporary `num_slots * meta_stride` (`2.25 MiB`) staging buffer for the initial population pass (freed immediately after closing the initial reader), and submitting initial population reads in batches of `128` slots (`512` SQEs $\le 1024$).

---

## 4. Conclusion

The zero-copy `pin_memory` + role-major `O_DIRECT` `io_uring` (`IORING_OP_READ_FIXED`) expert slot cache reduces per-rank resident MoE weight memory at `TP=2` by `53.086 GiB` across 58 layers (`288` resident slots per layer), preserves bit-identical output (`torch.equal`), matches or slightly improves all-hit decode latency on GB10 unified LPDDR5x, and cuts cold-miss latency by `15.7%` via role-major `w13`/`w2` split waiting and by `72.0%` via cross-layer router lookahead prefetching.
