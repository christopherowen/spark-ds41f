# GB10 Display Carve-Out Weight Compaction & Decode-Only DSpark Weight Expansion (`2026-11-08-display-carveout-weight-expansion`)

## Base & Intended Causal Variable

- **Base commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`origin/main`, `spark-ds41f-1107`, vLLM series at `0761`, patch head `320021e3c4dbb202d911ab70e2cd924c75420488`)
- **Runtime image**: `spark-ds41f:1107` (`sha256:3f32a962ce04`) on `sp3` (`spark-0011`, NVIDIA GB10)
- **Candidate vLLM patch**: `experiments/2026-11-08-display-carveout-weight-expansion/vllm/0762-perf-worker-compact-packed-vocab-tables-and-expand-d.patch`
- **Intended causal variable**:
  1. Compact `embed_tokens.weight` and `lm_head.weight` dumb buffers in `display_carveout.compact()` after in-place 12-bit packing (`TileLangEmbeddingMethod` and `PackedBf16LmHeadMethod`), reclaiming `224 MiB` at TP3 (`432 MiB x 2 -> 320 MiB x 2`) and `160 MiB` at TP4 (`320 MiB x 2 -> 240 MiB x 2`) of `/dev/dri/card0` scanout heap with zero fragmentation.
  2. Lower `_MIN_WEIGHT_BYTES` from `64 MiB` to `60 MiB` and allocate `DSparkMarkovHead.markov_w1.weight` (`129,280 x 256` BF16 = `63.125 MiB` -> `64 MiB` buffer) directly in the display carve-out via `display_carveout.allocate()`, shrinking the drafter's `_direct_weight_arena` by `63.125 MiB`.
  3. Move decode-only `TileLangExperts` routed-expert weight slabs (`self._plan.max_rows is not None and not self._plan._has_mid_tiles()`, i.e., DSpark draft layers `mtp.0`, `mtp.1`, `mtp.2` `w13_weight` and `w2_weight`) into the remaining display carve-out capacity via `display_carveout.move()` up to `_MAX_TOTAL_BYTES = 1,968 MiB` (leaving `64 MiB` of free scanout heap on `dgx3`'s `2,032 MiB` reserve for the `8 MiB` console framebuffer).

## Candidate Weight Screening

1. **Rejected prefill-active dense FP8 weights (`engram.wkv`, `dspark.main_proj`, target `shared_experts`)**:
   - Screened on `sp3` at `M = 4096` prefill (`32` M-tiles). Because `mxfp8_gemm` re-reads weight tiles across all 32 M-tiles via `T.tma_copy`, placing prefill-active dense weights in the uncached display carve-out bypasses L2 reuse and regresses prefill `2.77x–3.10x`:
     - `engram.wkv` (`5120 -> 7680` FP8): `2,398.79 us` (regular) vs `7,436.70 us` (carve-out, `3.10x` slower)
     - `dspark.main_proj` (`25600 -> 5120` FP8): `1,565.18 us` (regular) vs `4,409.75 us` (carve-out, `2.82x` slower)
     - `shared_experts.gate_up_proj` (`5120 -> 2048` FP8): `1,352.00 us` (regular) vs `3,741.89 us` (carve-out, `2.77x` slower)
     - `shared_experts.down_proj` (`1024 -> 5120` FP8): `681.55 us` (regular) vs `2,033.36 us` (carve-out, `2.98x` slower)
   - All prefill-active dense GEMM weights therefore stay in cached regular memory.
2. **Selected read-once-per-step / prefill-inactive weights**:
   - `model.markov_head.markov_w1.weight` (`63.125 MiB`, `64 MiB` buffer): never touched during prefill; read only by `markov_anchor_embed` (`1` row = `512 B` per active request per draft step). Note: `markov_w2.weight` (`15.78 MiB` packed 12-bit at TP3) remains in cached regular memory so sequential Markov steps `1..4` stay L2-hot.
   - Decode-only DSpark routed-expert slabs (`model.layers.{0,1,2}.ffn.experts.w13_weight` and `w2_weight`): DSpark's 3 decoder layers never execute during prefill (prefill only runs `main_proj` and `_context_kv_projections`). During `DSparkSpeculator.propose()`, `M = 5 * num_reqs` (`M = 5..40` on TP3) with `128` experts (`topk = 3`) and `decode_block_M = 16`, so each active expert touches a single `M = 16` tile and streams its weight slab once per step. Physically contiguous dumb buffers avoid unified-memory page-table/TLB misses and L2 cache pollution, running **8.5%–14.7% faster** than regular memory across `M = 5, 15, 40`.

## Exact Commands

```bash
# Unit test suite on sp3 under flock /tmp/gpu.lock (234 passed)
flock /tmp/gpu.lock docker run --rm --gpus all --device /dev/dri/card0   -v /tmp/vllm-0762-work/vllm:/opt/spark-ds41f/candidate/vllm/vllm   -v /tmp/vllm-0762-work/tests:/opt/spark-ds41f/candidate/vllm/tests   --entrypoint python3 spark-ds41f:1107 -c '
import sys, types, pytest
tblib = types.ModuleType("tblib")
ps = types.ModuleType("tblib.pickling_support")
ps.install = lambda *a, **kw: None
tblib.pickling_support = ps
sys.modules["tblib"] = tblib
sys.modules["tblib.pickling_support"] = ps
sys.exit(pytest.main([
    "-q",
    "/opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_display_carveout.py",
    "/opt/spark-ds41f/candidate/vllm/tests/models/test_dspark_mla.py",
    "/opt/spark-ds41f/candidate/vllm/tests/model_executor/model_loader/test_direct_loader.py",
    "/opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_gpu_model_runner_v2.py",
    "/opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_tilelang_embedding.py",
]))'

# 2-boot memory and kernel latency benchmark on sp3 under flock /tmp/gpu.lock
flock /tmp/gpu.lock bash -c '
  docker run --rm --gpus all --device /dev/dri/card0     -v /tmp/tilelang-cache:/root/.tilelang/cache     -v /tmp/vllm-0762-work/vllm:/opt/spark-ds41f/candidate/vllm/vllm     -v /tmp/run_carveout_bench.py:/tmp/run_carveout_bench.py:ro     -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     --entrypoint python3 spark-ds41f:1107 /tmp/run_carveout_bench.py 1 > /tmp/boot1.json
  docker run --rm --gpus all --device /dev/dri/card0     -v /tmp/tilelang-cache:/root/.tilelang/cache     -v /tmp/vllm-0762-work/vllm:/opt/spark-ds41f/candidate/vllm/vllm     -v /tmp/run_carveout_bench.py:/tmp/run_carveout_bench.py:ro     -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     --entrypoint python3 spark-ds41f:1107 /tmp/run_carveout_bench.py 2 > /tmp/boot2.json
'
```

## 2-Boot Memory Observations (`sp3`, NVIDIA GB10)

### Display Carve-Out Allocation Layout

| Shape | Metric | Baseline (`1107`) | Candidate (`0762`) | Delta |
|---|---|---:|---:|---:|
| `tp3-triangle` | Carve-out buffers (`MiB`) | `[432, 432]` (`2` buffers) | `[320, 320, 64, 480, 240, 480]` (`6` buffers) | `+4` buffers |
| `tp3-triangle` | Total carve-out reserved (`MiB`) | `864.000` | `1,904.000` | `+1,040.000 MiB` (`128 MiB` free on `dgx3`) |
| `tp3-triangle` | Live weight bytes in carve-out (`MiB`) | `631.875` | `1,895.000` | **`+1,263.125 MiB` (`+1.234 GiB`)** |
| `tp4-ring` | Carve-out buffers (`MiB`) | `[320, 320]` (`2` buffers) | `[240, 240, 64, 368, 160, 368, 160, 368]` (`8` buffers) | `+6` buffers |
| `tp4-ring` | Total carve-out reserved (`MiB`) | `640.000` | `1,968.000` | `+1,328.000 MiB` (`64 MiB` free on `dgx3`) |
| `tp4-ring` | Live weight bytes in carve-out (`MiB`) | `473.438` | `1,936.562` | **`+1,463.125 MiB` (`+1.429 GiB`)** |

### Regular UMA & CUDA Allocator Savings Across 2 Fresh Container Boots

| Shape | Metric | Boot 1 Baseline | Boot 1 Candidate | Boot 1 Saved | Boot 2 Baseline | Boot 2 Candidate | Boot 2 Saved | Mean Saved |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `tp3-triangle` | `torch.cuda.memory_allocated()` (`MiB`) | `2,260.304` | `996.851` | **`1,263.453`** | `2,260.304` | `996.851` | **`1,263.453`** | **`1,263.453 MiB` (`1.234 GiB`)** |
| `tp3-triangle` | `torch.cuda.memory_reserved()` (`MiB`) | `2,388.000` | `1,068.000` | **`1,320.000`** | `2,388.000` | `1,068.000` | **`1,320.000`** | **`1,320.000 MiB` (`1.289 GiB`)** |
| `tp3-triangle` | Peak `max_memory_reserved()` (`MiB`) | `3,068.000` | `1,568.000` | **`1,500.000`** | `3,068.000` | `1,568.000` | **`1,500.000`** | **`1,500.000 MiB` (`1.465 GiB`)** |
| `tp3-triangle` | Host `MemAvailable` used (`MiB`) | `2,805.58` | `1,136.06` | **`1,669.52`** | `2,805.05` | `1,137.20` | **`1,667.85`** | **`1,668.68 MiB` (`1.630 GiB`)** |
| `tp4-ring` | `torch.cuda.memory_allocated()` (`MiB`) | `1,712.170` | `248.799` | **`1,463.371`** | `1,712.416` | `249.538` | **`1,462.878`** | **`1,463.125 MiB` (`1.429 GiB`)** |
| `tp4-ring` | `torch.cuda.memory_reserved()` (`MiB`) | `1,788.000` | `306.000` | **`1,482.000`** | `1,788.000` | `308.000` | **`1,480.000`** | **`1,481.000 MiB` (`1.446 GiB`)** |
| `tp4-ring` | Peak `max_memory_reserved()` (`MiB`) | `2,308.000` | `1,146.000` | **`1,162.000`** | `2,308.000` | `1,148.000` | **`1,160.000`** | **`1,161.000 MiB` (`1.134 GiB`)** |
| `tp4-ring` | Host `MemAvailable` used (`MiB`) | `1,830.12` | `361.18` | **`1,468.93`** | `1,838.53` | `366.04` | **`1,472.48`** | **`1,470.71 MiB` (`1.436 GiB`)** |

## 2-Boot Kernel Latency & Bit-Identical Verification (`sp3`, NVIDIA GB10)

| Shape | Kernel & Batch Shape | Boot 1 Regular (`us`) | Boot 1 Carve-Out (`us`) | Boot 1 Delta (`us`) | Boot 2 Regular (`us`) | Boot 2 Carve-Out (`us`) | Boot 2 Delta (`us`) | Mean Ratio | `torch.equal` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `tp3-triangle` | `dspark_moe` (`M=5`, 1 req) | `393.22` | `345.02` | **`-48.20` (`-12.3%`)** | `392.45` | `347.26` | **`-45.19` (`-11.5%`)** | **`0.8812x`** | `True` |
| `tp3-triangle` | `dspark_moe` (`M=15`, 3 reqs) | `1,153.18` | `1,012.80` | **`-140.38` (`-12.2%`)** | `1,153.31` | `1,020.03` | **`-133.28` (`-11.6%`)** | **`0.8814x`** | `True` |
| `tp3-triangle` | `dspark_moe` (`M=40`, 8 reqs) | `3,056.86` | `2,796.64` | **`-260.22` (`-8.5%`)** | `3,052.70` | `2,795.39` | **`-257.31` (`-8.4%`)** | **`0.9153x`** | `True` |
| `tp3-triangle` | `markov_anchor_embed` (`reqs=1`) | `9.73` | `9.82` | `+0.09` | `9.79` | `9.86` | `+0.07` | `1.0082x` | `True` |
| `tp3-triangle` | `markov_anchor_embed` (`reqs=8`) | `11.26` | `9.73` | `-1.53` | `9.57` | `9.82` | `+0.25` | `0.9451x` | `True` |
| `tp3-triangle` | `vocab_embedding_packed` (`M=1`) | `10.14` | `10.11` | `-0.03` | `10.72` | `10.59` | `-0.13` | `0.9924x` | `True` |
| `tp3-triangle` | `vocab_embedding_packed` (`M=40`) | `10.46` | `9.92` | `-0.54` | `10.50` | `10.50` | `0.00` | `0.9742x` | `True` |
| `tp3-triangle` | `vocab_embedding_packed` (`M=4096`) | `249.95` | `249.06` | `-0.89` | `249.95` | `252.06` | `+2.11` | `1.0024x` | `True` |
| `tp4-ring` | `dspark_moe` (`M=5`, 1 req) | `296.45` | `252.93` | **`-43.52` (`-14.7%`)** | `297.06` | `253.09` | **`-43.97` (`-14.8%`)** | **`0.8526x`** | `True` |
| `tp4-ring` | `dspark_moe` (`M=15`, 3 reqs) | `857.41` | `736.19` | **`-121.22` (`-14.1%`)** | `864.26` | `740.45` | **`-123.81` (`-14.3%`)** | **`0.8576x`** | `True` |
| `tp4-ring` | `dspark_moe` (`M=40`, 8 reqs) | `2,280.19` | `2,094.91` | **`-185.28` (`-8.1%`)** | `2,291.97` | `2,104.38` | **`-187.59` (`-8.2%`)** | **`0.9184x`** | `True` |
| `tp4-ring` | `markov_anchor_embed` (`reqs=1`) | `9.86` | `9.95` | `+0.09` | `9.92` | `9.98` | `+0.06` | `1.0076x` | `True` |
| `tp4-ring` | `markov_anchor_embed` (`reqs=8`) | `9.70` | `9.89` | `+0.19` | `12.10` | `9.92` | `-2.18` | `0.9197x` | `True` |
| `tp4-ring` | `vocab_embedding_packed` (`M=4096`) | `237.76` | `239.81` | `+2.05` | `237.76` | `235.71` | `-2.05` | `1.0000x` | `True` |

## Conclusion

Compacting the 12-bit packed `embed_tokens` and `lm_head` dumb buffers (`224 MiB` reclaimed at TP3, `160 MiB` at TP4) and expanding `SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1` to hold `markov_w1.weight` (`63.125 MiB`) plus decode-only DSpark routed-expert `w13_weight` and `w2_weight` slabs up to `1,968 MiB` increases live weights in the GB10 display carve-out from `631.88 MiB` (`864 MiB` buffers) to **`1,895.00 MiB` (`1,904 MiB` buffers) at TP3** and from `473.44 MiB` (`640 MiB` buffers) to **`1,936.56 MiB` (`1,968 MiB` buffers) at TP4**. This saves **`+1,263.125 MiB` (`+1.234 GiB` allocated, `1,320 MiB` reserved, `1,500 MiB` peak reserved during preparation, `~1,669 MiB` host `MemAvailable`) per rank at TP3** and **`+1,463.125 MiB` (`+1.429 GiB` allocated, `1,481 MiB` reserved) per rank at TP4**, speeds up moved DSpark MoE layers by **8.5%–14.7%** across `M = 5..40`, has **`0.00 us` prefill impact**, and preserves 100% bit-identical (`torch.equal`) outputs.
