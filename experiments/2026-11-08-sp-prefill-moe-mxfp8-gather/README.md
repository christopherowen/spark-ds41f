# Sequence-Parallel MoE Pre-Gather MXFP8 Cast, Local Routing & Tail-Mask Caching (`2026-11-08-sp-prefill-moe-mxfp8-gather`)

## Base & Intended Causal Variable

- **Base commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`origin/main`, Candidate `1107`, vLLM patch series at `0761`)
- **Runtime image**: `spark-ds41f:1107` on `sp5` (`NVIDIA GB10`)
- **Candidate vLLM patch**: `experiments/2026-11-08-sp-prefill-moe-mxfp8-gather/vllm/0762-perf-dsv41-cache-SP-local-routing-masks-and-skip-SP-.patch`
- **Intended causal variable**:
  1. Audit and benchmark the active `0573`/`0720` sequence-parallel prefill MoE input contract (`mhc.post_pre` local `MXFP8` cast + `router.route_local` + `sp_rows.gather_mxfp8` + `provide_mxfp8` / `provide_routes` reuse across `shared_experts` and `TileLangExperts`) against post-gather full-`T` `MXFP8` quantization and full-`T` routing (`TODO.md` lines 260–265).
  2. Eliminate the remaining per-layer redundant work in `vllm/models/deepseek_v4_1/tilelang/router.py` where `_local_mask(mask, rows, sp_rows)` re-executed `sp_rows.local_slice(mask[:sp_rows.num_tokens])` uncached on every encoder layer (`0..19`), and fix `_partials_mask` (`local_rows <= 256`) so SP tail pad rows `[real_rows:local_rows]` are padded with `True` (`skip`) instead of `False` (`route`).

## Trace of the SP Prefill MoE Pipeline in `1107`

1. **Pre-gather MXFP8 quantization (`mhc.post_pre`) & local routing (`sp_moe_inputs`)**:
   - On the 20 encoder layers (`0..19`), `self.mhc.post_pre(...)` outputs this rank's `L = ceil(T / TP)` rows in both `BF16` (`x`) and `MXFP8` (`x_q = MXFP8Rows(x, values, scales)`).
   - `kernels.sp_moe_inputs(sp_rows, x, x_q, self.ffn, input_ids)` calls `router.route_local(x, sp_rows, input_ids=input_ids)` on the `L` local `BF16` rows, then `sp_rows.gather_mxfp8(x_q, weights, ids)` to all-gather `(x_q.values[:L], x_q.scales[:L], weights, ids)` in one grouped PyNCCL collective (`5,352` B/row vs `10,240` B/row `BF16`, a **`-47.73%`** collective payload reduction).
   - Both `shared_experts.gate_up_proj` (`tilelang_mxfp8_input = True`) and `TileLangExperts` (`MoeExpertsV41Plan.run`) consume the gathered `MXFP8Rows` via `provide_mxfp8(gathered_q)` without recasting, and `TileLangGateRouter._compute_routing` consumes `provide_routes(gathered_q.rows, weights, ids)` without re-running the gate GEMM or top-k selection on all `T` rows.
2. **Remaining redundant per-layer tail-rank mask allocation & copy (`_local_mask`)**:
   - While `_routing_masks` cached the full `T`-row `image` and `routed` masks on `ForwardContext`, `_local_mask(mask, rows, sp_rows)` called `sp_rows.local_slice(mask[:sp_rows.num_tokens])` uncached on every encoder layer (`0..19`).
   - Whenever `T % TP != 0` (every power-of-2 chunk size `T = 256, 512, 1024, 2048, 4096, 8192, 16384` on `TP=3`, and any non-multiple of `4` on `TP=4`), the tail rank (`rank = TP - 1`, where `real_rows < local_rows`) allocated `new_zeros` and copied `[:real_rows]` twice per layer (`4` GPU kernel launches per layer, `80` across the 20 encoder layers), adding **`8.7–20.5 us/layer` (`+3%` to `+44%` local router latency)** on the tail rank immediately before the synchronous `gather_mxfp8` collective.
   - When `ForwardContext` was not active, `_routing_masks` also omitted caching `_ds41_image_token_mask` on `sp_rows`.
3. **SP tail-pad polarity defect in `_partials_mask` (`local_rows <= 256`)**:
   - When `local_rows <= 256` (`205 <= T <= 768` on `TP=3`, `205 <= T <= 1024` on `TP=4`) and `T % TP != 0`, `_partials_mask` passed `is_padding` (`True` = skip, `False` = route) through `_local_mask(is_padding, rows, sp_rows)`, which zero-padded `[real_rows:local_rows]` with `False` (**active / route**) instead of `True` (**padding / skip**).
   - Passing `pad=True` in `_partials_mask` pads `[real_rows:local_rows]` with `True` so `route_partials` skips unused SP tail pad rows with `topk_idx = -1, topk_weights = 0.0` (matching `TileKernelsRouter` at `local_rows > 256`) and avoids triggering `T.device_assert(finite)` on uninitialized/NaN tail pad rows.

## Exact Commands

```bash
# 2-boot hardware benchmark on sp5 under /tmp/gpu.lock
flock /tmp/gpu.lock bash -c '
  docker run --rm --gpus all --ipc=host --entrypoint "" -v /tmp:/tmp spark-ds41f:1107 python3 /tmp/bench_sp_moe.py 1 /tmp/boot1_sp_moe.json &&
  docker run --rm --gpus all --ipc=host --entrypoint "" -v /tmp:/tmp spark-ds41f:1107 python3 /tmp/bench_sp_moe.py 2 /tmp/boot2_sp_moe.json
'

# 2-boot pytest verification on sp5 under /tmp/gpu.lock
flock /tmp/gpu.lock bash -c '
  for boot in 1 2; do
    docker run --rm --gpus all --ipc=host --entrypoint "" \
      -v /tmp/vllm-patch-work/vllm/models/deepseek_v4_1/tilelang/router.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/router.py:ro \
      -v /tmp/vllm-patch-work/tests/kernels/moe/test_deepseek_v41_tilelang_router.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/moe/test_deepseek_v41_tilelang_router.py:ro \
      -w /opt/spark-ds41f/candidate/vllm spark-ds41f:1107 \
      pytest --noconftest tests/models/test_deepseek_v4_1_sp_prefill.py tests/kernels/moe/test_deepseek_v41_tilelang_router.py
  done
'
```

## 2-Boot Hardware Results (`sp5`, NVIDIA GB10)

### Collective Payload per Encoder Layer (`20` Encoder Layers)

| Shape (`T`) | `TP=3` `L` | `TP=3` BF16 Gather / Rank | `TP=3` MXFP8+Routes Gather / Rank | `TP=4` `L` | `TP=4` BF16 Gather / Rank | `TP=4` MXFP8+Routes Gather / Rank | Payload Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024` | `342` | `3,502,080 B` | `1,830,384 B` | `256` | `2,621,440 B` | `1,370,112 B` | **`-47.73%`** |
| `2048` | `683` | `6,993,920 B` | `3,655,416 B` | `512` | `5,242,880 B` | `2,740,224 B` | **`-47.73%`** |
| `4096` | `1366` | `13,987,840 B` | `7,310,832 B` | `1024` | `10,485,760 B` | `5,480,448 B` | **`-47.73%`** |
| `6144` | `2048` | `20,971,520 B` | `10,960,896 B` | `1536` | `15,728,640 B` | `8,220,672 B` | **`-47.73%`** |
| `8192` | `2731` | `27,965,440 B` | `14,616,312 B` | `2048` | `20,971,520 B` | `10,960,896 B` | **`-47.73%`** |
| `16384` | `5462` | `55,930,880 B` | `29,232,624 B` | `4096` | `41,943,040 B` | `21,921,792 B` | **`-47.73%`** |

### `TP=3` Per-Layer Preparation & Tail-Rank Router Latency (`tp3.json`)

| `T` | `L` | Post-Gather Cast+Route (`us`, B1 / B2, Mean ± CI) | Tail `route_local` Uncached `1107` (`us`, B1 / B2, Mean ± CI) | Tail `route_local` Cached `0762` (`us`, B1 / B2, Mean ± CI) | Tail Mask Cache Delta (`0762` vs `1107`) | Pre-Gather Prep Delta vs Post-Gather | Saved Prep / `20` Encoder Layers |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024` | `342` | `97.49` / `90.91` (`94.20 ± 3.29`) | `52.88` / `54.57` (`53.73 ± 0.84`) | `34.91` / `39.77` (`37.34 ± 2.43`) | **`-30.54%`** | **`-60.22%`** | **`1.14 ms`** |
| `2048` | `683` | `279.79` / `327.13` (`303.46 ± 23.67`) | `78.16` / `81.34` (`79.75 ± 1.59`) | `69.30` / `72.76` (`71.03 ± 1.73`) | **`-10.94%`** | **`-76.50%`** | **`4.65 ms`** |
| `4096` | `1366` | `625.36` / `629.78` (`627.57 ± 2.21`) | `116.23` / `117.83` (`117.03 ± 0.80`) | `109.60` / `106.40` (`108.00 ± 1.60`) | **`-7.71%`** | **`-82.79%`** | **`10.39 ms`** |
| `6144` | `2048` | `945.77` / `939.72` (`942.75 ± 3.02`) | `167.26` / `166.23` (`166.75 ± 0.52`) | `148.29` / `147.35` (`147.82 ± 0.47`) | **`-11.35%`** | **`-84.32%`** | **`15.90 ms`** |
| `8192` | `2731` | `1252.50` / `1263.23` (`1257.87 ± 5.37`) | `232.30` / `240.39` (`236.34 ± 4.04`) | `222.15` / `220.17` (`221.16 ± 0.99`) | **`-6.39%`** | **`-82.41%`** | **`20.73 ms`** |
| `16384` | `5462` | `2467.58` / `2453.27` (`2460.43 ± 7.15`) | `437.45` / `433.23` (`435.34 ± 2.11`) | `421.35` / `423.64` (`422.50 ± 1.14`) | **`-2.95%`** | **`-82.83%`** | **`40.76 ms`** |

### `TP=4` Per-Layer Preparation & Tail-Rank Router Latency (`tp4.json`)

| `T` | `L` | Post-Gather Cast+Route (`us`, B1 / B2, Mean ± CI) | Tail `route_local` Uncached `1107` (`us`, B1 / B2, Mean ± CI) | Tail `route_local` Cached `0762` (`us`, B1 / B2, Mean ± CI) | Tail Mask Cache Delta (`0762` vs `1107`) | Pre-Gather Prep Delta vs Post-Gather | Saved Prep / `20` Encoder Layers |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024` | `256` | `100.78` / `91.64` (`96.21 ± 4.57`) | `43.24` / `50.58` (`46.91 ± 3.67`) | `26.41` / `26.33` (`26.37 ± 0.04`) | **`-43.42%`** | **`-72.53%`** | **`1.40 ms`** |
| `2048` | `512` | `276.30` / `271.83` (`274.06 ± 2.24`) | `54.45` / `53.06` (`53.76 ± 0.70`) | `41.70` / `41.52` (`41.61 ± 0.09`) | **`-22.58%`** | **`-84.82%`** | **`4.65 ms`** |
| `4096` | `1024` | `628.62` / `630.25` (`629.43 ± 0.81`) | `86.15` / `84.45` (`85.30 ± 0.85`) | `76.47` / `74.72` (`75.59 ± 0.88`) | **`-11.38%`** | **`-88.00%`** | **`11.08 ms`** |
| `6144` | `1536` | `950.19` / `925.20` (`937.70 ± 12.50`) | `129.00` / `130.84` (`129.92 ± 0.92`) | `109.05` / `110.82` (`109.94 ± 0.88`) | **`-15.39%`** | **`-88.27%`** | **`16.56 ms`** |
| `8192` | `2048` | `1368.59` / `1260.96` (`1314.78 ± 53.81`) | `167.70` / `159.61` (`163.66 ± 4.04`) | `150.11` / `148.25` (`149.18 ± 0.93`) | **`-8.80%`** | **`-88.63%`** | **`23.31 ms`** |
| `16384` | `4096` | `2454.93` / `2640.50` (`2547.72 ± 92.79`) | `327.42` / `338.22` (`332.82 ± 5.40`) | `330.50` / `327.85` (`329.18 ± 1.32`) | **`-1.06%`** | **`-87.06%`** | **`44.37 ms`** |

## Bit-Exactness, Tests & Memory Observations

- **Bit-exactness (`torch.equal`)**: Verified `100%` bit-exact (`torch.equal` on `MXFP8` `values`, `scales`, `topk_weights.view(torch.int32)`, `topk_ids`, `shared_experts` BF16 output, and `MoeExpertsV41Plan` combined output) across all ranks `0..TP-1`, all `T in (1024, 2048, 4096, 6144, 8192, 16384)`, and `TP in (3, 4)` across both boots.
- **2-boot pytest suite**: `pytest --noconftest tests/models/test_deepseek_v4_1_sp_prefill.py tests/kernels/moe/test_deepseek_v41_tilelang_router.py` passed **`262/262` in `48.55s` (Boot 1)** and **`262/262` in `48.04s` (Boot 2)**.
- **Memory observations**: `0` additional steady-state memory; caching `_local_mask` on `ForwardContext` reuses the 2 boolean tensors (`local_rows` bytes each, `<= 5.5 KiB`) computed on Encoder Layer 0 across Layers `1..19`, eliminating `38` transient tensor allocations and `76` GPU kernel launches (`new_zeros`/`new_ones` + `copy_`) on the tail rank per prefill chunk.

## Conclusion

1. The core sequence-parallel MoE pre-gather MXFP8 cast and local routing contract (`0573`, `0720`, `0651`) is already active in `1107`, cutting per-rank collective volume by **`47.73%`** and reducing per-layer MoE input preparation time by **`60.3%–84.3%` on `TP=3`** (`1.14–40.76 ms` saved per 20-layer encoder pass across `T = 1024..16384`) and **`72.6%–88.7%` on `TP=4`** (`1.40–44.37 ms` saved per 20-layer encoder pass).
2. Patch `0762` eliminates the remaining per-layer redundant tail-rank mask slicing/padding in `_local_mask` (`-3.0%` to `-30.5%` tail-rank `route_local` latency on `TP=3`, `-1.1%` to `-43.7%` on `TP=4`, removing `76` GPU kernel launches per 20-layer encoder chunk) and fixes `_partials_mask` (`local_rows <= 256`) to pad SP tail rows with `True` (`skip`).
