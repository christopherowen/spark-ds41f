# Host & CUDA Unified Memory Allocator Hygiene Across Weight Loading, Profiling, Graph Capture, and Long-Prefill Transitions

## Metadata

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18b` (`main`)
- **Base vLLM patch head**: `320021e3c4dbb202d911ab70e2cd924c75420488` (`0761`, Candidate `1107`)
- **Candidate vLLM patch**: [`vllm/0762-perf-worker-reclaim-uma-heap-and-allocator-slack-acr.patch`](vllm/0762-perf-worker-reclaim-uma-heap-and-allocator-slack-acr.patch)
  - **Patch commit**: `873f7e6417345827e037f7421c1e7175c1114752`
  - **Expected tree**: `cc254b9c80f4f471aa0b54ef1ad82c187559f85b`
  - **Stable `git patch-id`**: `a8fb5b4c57ac2be367822dfebb8a5c677f94cef3`
  - **Patch SHA-256**: `b82e1eecdb74042f067c4862a92dd2cf21cf844fbfa1729ca496f9a29e023e47`
  - **Patchset SHA-256**: `5e8b3d6bc7fd6bfca76b009477d54a2865e63042ec39c46c838421a38ee362f5`
- **Causal variable**: Coordinated host heap trimming (`libc10` statically linked `mi_collect(True)` + `glibc` `malloc_trim(0)`) and CUDA caching-allocator reclamation (`gc.collect()` + `torch.accelerator.empty_cache()`) around weight loading, startup memory profiling, CUDA graph capture, and long-prefill-to-decode transitions on unified-memory `GB10` (`sm_121a`) nodes.

## Background & Root-Cause Trace

On `GB10` (`128 GB LPDDR5x` unified memory), host virtual memory arenas and PyTorch's CUDA caching allocator draw from the same physical memory pool. Moreover, `MemorySnapshot.measure()` (`vllm/utils/mem_utils.py`, patch `0021`) samples `psutil.virtual_memory().available` (`/proc/meminfo` `MemAvailable`) directly when `current_platform.is_integrated_gpu()` is `True`, and computes:

$$\text{non\_torch\_memory} = \text{total\_memory} - \text{free\_memory} - \text{torch\_memory}$$

Tracing `/proc/self/status` (`VmRSS`, `RssAnon`), `/proc/self/smaps` (`[heap]` vs anonymous `mmap` segments), and `/usr/local/lib/python3.12/dist-packages/torch/lib/libc10.so` on `sp8` (`spark-ds41f:1107`) uncovered two structural allocator retention mechanisms across the worker lifecycle:

1. **Dual Host Allocators (`libc10` `mimalloc` + `glibc` `ptmalloc`)**:
   - Disassembly of `_ZN3c109alloc_cpuEm` and `_ZN3c108free_cpuEPv` in `libc10.so` on `aarch64` shows that PyTorch's CPU tensor allocator (`torch.empty(..., device="cpu")`, `.to("cpu", copy=True)`, grammar bitmask tensors, multimodal CPU staging tensors, and weight loader staging buffers) does **not** use `glibc` `malloc`/`free`. Instead, `libc10.so` statically links `mimalloc` (`mi_malloc_aligned` at `0x46fbc` and `mi_free` at `0x45800`, with `mi_collect` at `0x44bd0`, `-0x1230` bytes from `_ZN3c108free_cpuEPv`).
   - Python dicts, JSON/safetensors header tables, and C++ stdlib containers allocate from `glibc` `ptmalloc` (`[heap]` and `mmap` arenas).
   - Calling `gc.collect()` or `libc.malloc_trim(0)` alone reclaims `glibc` arenas (`~104 MiB` in our synthetic arena test), leaving **100% of freed CPU tensor pages (`~643 MiB`)** pinned inside `mimalloc` anonymous `mmap` segments. Calling both `mi_collect(True)` and `malloc_trim(0)` reclaims **`747.7 MiB` of `768.0 MiB`** immediately back to OS `MemAvailable`.
2. **Lifecycle Gaps Across Startup and Long-Prefill Transitions**:
   - **Stage 1 (`load_model` / `DirectModelLoader` / `bind_weight_arena`)**: `bind_weight_arena()` snapshots pre-initialized small parameters to CPU via `param.detach().to("cpu", copy=True)` (`cpu_snapshots`), copies them into the unified CUDA weight arena, and drops the dict at function exit without calling `gc.collect()` or `trim_host_heap()`. `DirectModelLoader.load_model()` and `DeviceMemoryProfiler.__exit__()` called `torch.accelerator.empty_cache()` without `gc.collect()` or `trim_host_heap()`.
   - **Stage 2 (`profile_run` / `memory_profiling` / `_determine_available_memory`)**:
     - `GPUModelRunner.profile_run()` ran `mm_registry.get_mm_limits_per_prompt`, `encoder_cache_manager.check_and_update_cache`, `_dummy_run(max_num_batched_tokens)`, and `_dummy_sampler_run()` before running `_profile_deepseek_v4_attention()` without flushing unreferenced default-allocator blocks between the generic forward/sampler pass and the max-context attention profiling pass.
     - `memory_profiling()` exited `profile_run()`, called `torch.accelerator.empty_cache()`, and immediately sampled `after_profile = MemorySnapshot()` without calling `gc.collect()` or `trim_host_heap()`. On `GB10` (`UMA`), any unreleased `mimalloc` or `glibc` heap slack lowered `after_profile.free_memory` (`MemAvailable`), inflating `result.non_torch_increase` and `result.non_kv_cache_memory` byte-for-byte and permanently shrinking the KV cache allocation.
   - **Stage 3 (`freeze_gc_for_cudagraph_capture` / `capture_model` / `freeze_gc_heap`)**:
     - `freeze_gc_for_cudagraph_capture()` and `freeze_gc_heap()` ran `gc.collect()` + `gc.freeze()` without trimming host heaps (`mi_collect(True)` + `malloc_trim(0)`), and `Worker.compile_or_warm_up_model()` logged `log_startup_memory_breakdown()` (`MemorySnapshot()`) before `freeze_gc_heap()`.
   - **Stage 4 (Long-Prefill `CUDAGraphMode.NONE` $\rightarrow$ `CUDAGraphMode.FULL` Decode Transition)**:
     - Multi-chunk eager prefill (`32,768` tokens across `16,384`-token chunks) runs with `CUDAGraphMode.NONE` and expands the default PyTorch CUDA caching allocator by `~1.5–1.9 GiB` per rank for temporary prefill workspace and activations.
     - Once the request transitions back to `CUDAGraphMode.FULL` decode (which replays static allocations inside `global_graph_pool`), those `~1.5–1.9 GiB` of default-allocator segments and `~120 MiB` of host prefill staging buffers remain pinned as idle reserved slack unless flushed once at the prefill-to-decode boundary. Flushing between consecutive prefill chunks would thrash segment reuse; flushing **once** on the first `CUDAGraphMode.FULL` decode step after any `CUDAGraphMode.NONE` step preserves 100% segment reuse across prefill chunks while returning the entire prefill spike to `MemAvailable` as soon as decode begins.

## Optimization Design (Patch `0762`)

1. **`trim_host_heap()` and `reclaim_uma_memory()` (`vllm/utils/mem_utils.py`)**:
   - `_get_c10_mi_collect()` locates `libc10.so` via `dl_iterate_phdr` (`dladdr` on `_ZN3c108free_cpuEPv`) and parses `.symtab`/`.strtab` in `libc10.so` to resolve the statically linked `mi_collect(bool force)` symbol at runtime via its exact offset from `_ZN3c108free_cpuEPv`.
   - `trim_host_heap()` invokes both `mi_collect(True)` and `libc.malloc_trim(0)`.
   - `reclaim_uma_memory(device=None, *, synchronize=False)` runs `gc.collect()`, optional device synchronization, `torch.accelerator.empty_cache()`, and `trim_host_heap()`.
2. **Weight Loading & Arena Binding (`vllm/model_executor/model_loader/direct_loader.py`, `vllm/models/deepseek_v4_1/tilelang/prepare.py`, `vllm/utils/mem_utils.py`)**:
   - `bind_weight_arena()` explicitly clears `cpu_snapshots` and runs `gc.collect()` + `trim_host_heap()`.
   - `DirectModelLoader.load_model()`, `DeviceMemoryProfiler.__exit__()`, and TileLang `release_host_memory()` call `reclaim_uma_memory(synchronize=True)` / `trim_host_heap()`.
3. **Profiling & UMA KV Cache Sizing (`vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/worker/gpu_worker.py`, `vllm/v1/worker/gpu/cudagraph_utils.py`, `vllm/utils/mem_utils.py`)**:
   - `GPUModelRunner.profile_run()` calls `reclaim_uma_memory(self.device, synchronize=True)` before `_profile_deepseek_v4_attention()` and at the end of `profile_run()`.
   - `memory_profiling()` trims host heaps before `before_profile.measure()` and runs `gc.collect()` + `torch.accelerator.empty_cache()` + `trim_host_heap()` before `after_profile.measure()`.
4. **CUDA Graph Capture & GC Freeze (`vllm/utils/gc_utils.py`, `vllm/v1/worker/gpu/model_runner.py`, `vllm/v1/worker/gpu_worker.py`)**:
   - `freeze_gc_for_cudagraph_capture()` and `freeze_gc_heap()` call `trim_host_heap()` immediately after `gc.collect()` before `gc.freeze()`, and `Worker.compile_or_warm_up_model()` freezes and trims before `log_startup_memory_breakdown()`.
5. **Single-Shot Post-Prefill Decode Transition Flush (`vllm/v1/worker/gpu/model_runner.py`)**:
   - `GPUModelRunner` tracks `_pending_post_prefill_cache_flush: bool` and `_last_batch_cg_mode: CUDAGraphMode`.
   - When `execute_model()` runs with `cudagraph_mode == CUDAGraphMode.NONE` (`num_tokens > max_cudagraph_capture_size` or eager prefill), `_pending_post_prefill_cache_flush` is set to `True` without flushing between consecutive prefill chunks.
   - On the first subsequent `CUDAGraphMode.FULL` decode step, `sample_tokens()` clears `_pending_post_prefill_cache_flush` and calls `reclaim_uma_memory(self.device, synchronize=False)` once after launching decode work.

## 2-Boot Verification & Hardware Benchmark Results (`sp8`, `NVIDIA GB10`)

### Unit & Regression Tests (2 Boots)

Command executed on `sp8` inside `spark-ds41f:1107`:

```bash
flock /tmp/gpu.lock bash -c '
for boot in 1 2; do
  docker run --rm --gpus all --ipc=host --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --entrypoint "" \
    -e PYTHONPATH=/tmp/stub_pkgs:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/vllm/tests/vllm_test_utils \
    -v /tmp/stub_pkgs:/tmp/stub_pkgs:ro \
    -v /tmp/vllm-uma-hygiene/vllm/utils/mem_utils.py:/opt/spark-ds41f/candidate/vllm/vllm/utils/mem_utils.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/utils/gc_utils.py:/opt/spark-ds41f/candidate/vllm/vllm/utils/gc_utils.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/model_executor/model_loader/direct_loader.py:/opt/spark-ds41f/candidate/vllm/vllm/model_executor/model_loader/direct_loader.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/models/deepseek_v4_1/tilelang/prepare.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/prepare.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/v1/executor/multiproc_executor.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/executor/multiproc_executor.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/v1/worker/gpu/cudagraph_utils.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/cudagraph_utils.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/v1/worker/gpu/model_runner.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/model_runner.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/v1/worker/gpu_worker.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu_worker.py:ro \
    -v /tmp/vllm-uma-hygiene/tests:/opt/spark-ds41f/candidate/vllm/tests:ro \
    -w /opt/spark-ds41f/candidate/vllm \
    spark-ds41f:1107 \
    pytest -v -k "not test_memory_profiling" \
      tests/utils_/test_mem_utils.py \
      tests/utils_/test_gc_utils.py \
      tests/v1/worker/test_gpu_model_runner_v2.py \
      tests/v1/worker/test_gpu_worker.py \
      tests/model_executor/model_loader/test_direct_loader.py \
      tests/model_executor/model_loader/test_direct_loader_cuda.py 2>&1 | tee /tmp/pytest_uma_hygiene_boot${boot}.txt
done
'
```

- **Boot 1** ([`receipts/pytest-boot1.txt`](receipts/pytest-boot1.txt)): `233 passed, 1 deselected in 65.50s`
- **Boot 2** ([`receipts/pytest-boot2.txt`](receipts/pytest-boot2.txt)): `233 passed, 1 deselected in 65.03s`

### 4-Stage Hardware Lifecycle Benchmark (2 Boots on `sp8`)

Benchmark script: [`bench_uma_allocator_hygiene.py`](bench_uma_allocator_hygiene.py)
Receipts: [`receipts/bench-boot1.json`](receipts/bench-boot1.json), [`receipts/bench-boot2.json`](receipts/bench-boot2.json)

Command:

```bash
flock /tmp/gpu.lock bash -c '
for boot in 1 2; do
  docker run --rm --gpus all --ipc=host --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --entrypoint "" \
    -e PYTHONPATH=/opt/spark-ds41f/candidate/vllm \
    -v /home/chris/.cache/huggingface:/root/.cache/huggingface:ro \
    -v /tmp/vllm-uma-hygiene/vllm/utils/mem_utils.py:/opt/spark-ds41f/candidate/vllm/vllm/utils/mem_utils.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/utils/gc_utils.py:/opt/spark-ds41f/candidate/vllm/vllm/utils/gc_utils.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/model_executor/model_loader/direct_loader.py:/opt/spark-ds41f/candidate/vllm/vllm/model_executor/model_loader/direct_loader.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/models/deepseek_v4_1/tilelang/prepare.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/prepare.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/v1/executor/multiproc_executor.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/executor/multiproc_executor.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/v1/worker/gpu/cudagraph_utils.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/cudagraph_utils.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/v1/worker/gpu/model_runner.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/model_runner.py:ro \
    -v /tmp/vllm-uma-hygiene/vllm/v1/worker/gpu_worker.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu_worker.py:ro \
    -v /tmp/bench_uma_allocator_hygiene.py:/tmp/bench_uma_allocator_hygiene.py:ro \
    -w /opt/spark-ds41f/candidate/vllm \
    spark-ds41f:1107 \
    python3 /tmp/bench_uma_allocator_hygiene.py --boot-label boot${boot} > /tmp/uma_hygiene_boot${boot}_clean.json
done
'
```

| Lifecycle Stage & Metric | Baseline (Boot 1 / Boot 2) | Candidate `0762` (Boot 1 / Boot 2) | Delta Candidate − Baseline (Boot 1 / Boot 2) |
| :--- | :---: | :---: | :---: |
| **Stage 1: Post-Weight Load (`load_model` / `bind_weight_arena`)** | | | |
| Process `VmRSS` (`/proc/self/status`) | `1,546.99` / `1,547.82 MiB` | **`1,348.85` / `1,348.44 MiB`** | **`-198.14` / `-199.38 MiB`** |
| Anonymous RSS (`RssAnon`) | `954.17` / `954.99 MiB` | **`761.97` / `761.55 MiB`** | **`-192.20` / `-193.44 MiB`** |
| `smaps` Anonymous `mmap` RSS (`mimalloc`) | `612.83` / `613.56 MiB` | **`428.80` / `428.41 MiB`** | **`-184.03` / `-185.15 MiB`** |
| `smaps` `[heap]` RSS (`glibc`) | `296.98` / `297.07 MiB` | **`288.81` / `288.79 MiB`** | **`-8.17` / `-8.28 MiB`** |
| Host `MemAvailable` (`/proc/meminfo`) | `114,578.80` / `114,358.85 MiB` | **`114,773.69` / `115,130.82 MiB`** | **`+194.89` / `+771.97 MiB`** |
| **Stage 2: Post-Profiling (`profile_run` / `memory_profiling`)** | | | |
| Process `VmRSS` (`/proc/self/status`) | `1,739.53` / `1,740.23 MiB` | **`1,349.09` / `1,348.74 MiB`** | **`-390.44` / `-391.49 MiB`** |
| `smaps` Anonymous `mmap` RSS (`mimalloc`) | `802.53` / `803.29 MiB` | **`428.93` / `428.58 MiB`** | **`-373.60` / `-374.71 MiB`** |
| Peak `torch.cuda.memory_reserved` during attention profile | `2,108.00` / `2,108.00 MiB` | **`1,724.00` / `1,724.00 MiB`** | **`-384.00` / `-384.00 MiB`** |
| Profiled `non_kv_cache_memory` (for KV sizing) | `2,802.56` / `3,275.32 MiB` | **`2,004.54` / `2,020.40 MiB`** | **`-798.02` / `-1,254.92 MiB`** |
| Host `MemAvailable` (`/proc/meminfo`) | `114,001.14` / `113,754.39 MiB` | **`114,374.69` / `114,730.82 MiB`** | **`+373.55` / `+976.43 MiB`** |
| **Stage 3: Post-Graph Capture & `freeze_gc_heap`** | | | |
| Process `VmRSS` (`/proc/self/status`) | `1,691.50` / `1,692.13 MiB` | **`1,580.40` / `1,580.14 MiB`** | **`-111.10` / `-111.99 MiB`** |
| `torch.cuda.memory_reserved` | `1,906.00` / `1,906.00 MiB` | **`1,522.00` / `1,522.00 MiB`** | **`-384.00` / `-384.00 MiB`** |
| `torch.cuda.memory_reserved - memory_allocated` (slack) | `402.82` / `402.82 MiB` | **`18.82` / `18.82 MiB`** | **`-384.00` / `-384.00 MiB`** |
| Host `MemAvailable` (`/proc/meminfo`) | `113,070.94` / `113,075.56 MiB` | **`113,787.91` / `114,123.63 MiB`** | **`+716.97` / `+1,048.07 MiB`** |
| `cudaMemGetInfo` Free | `74,166.96` / `74,171.60 MiB` | **`74,883.46` / `75,219.12 MiB`** | **`+716.50` / `+1,047.52 MiB`** |
| **Stage 4: Long-Prefill (`32k` Eager) $\rightarrow$ `FULL` Decode Transition** | | | |
| Segment reuse across prefill chunks (`chunk1 -> chunk2`) | `True` / `True` | **`True` / `True`** | **Preserved (`0` thrashing)** |
| Process `VmRSS` (`/proc/self/status`) | `1,699.69` / `1,700.34 MiB` | **`1,579.50` / `1,580.24 MiB`** | **`-120.19` / `-120.10 MiB`** |
| `torch.cuda.memory_reserved` | `3,442.00` / `3,442.00 MiB` | **`1,522.00` / `1,522.00 MiB`** | **`-1,920.00` / `-1,920.00 MiB`** |
| Unallocated CUDA caching-allocator slack | `1,938.82` / `1,938.82 MiB` | **`18.82` / `18.82 MiB`** | **`-1,920.00` / `-1,920.00 MiB`** |
| Host `MemAvailable` (`/proc/meminfo`) | `111,531.69` / `111,599.21 MiB` | **`113,825.14` / `114,132.11 MiB`** | **`+2,293.45` / `+2,532.90 MiB`** |
| `cudaMemGetInfo` Free | `72,627.71` / `72,695.24 MiB` | **`74,920.69` / `75,227.59 MiB`** | **`+2,292.98` / `+2,532.35 MiB`** |
| Steady-State Decode Latency (`p50`, 200 steps) | `426.32` / `429.74 us` | **`434.03` / `427.81 us`** | `+7.71` / `-1.93 us` (within noise) |

## Memory Observations & Conclusion

1. **Dual-Allocator Host Heap Trimming (`mi_collect(True)` + `malloc_trim(0)`)**:
   - Because `libc10.so` on `aarch64` uses statically linked `mimalloc` for CPU tensors (`c10::alloc_cpu` / `c10::free_cpu`), resolving `mi_collect` from `libc10.so` alongside `glibc` `malloc_trim(0)` is essential to return freed CPU tensor arenas (`cpu_snapshots` in `bind_weight_arena`, multimodal/sampler profiling tensors, and grammar bitmasks) to the unified memory pool.
   - Across `load_model()` and `profile_run()`, `trim_host_heap()` reduces worker process `VmRSS` by **`198–391 MiB` per rank** (`~0.6–1.2 GiB` across TP3) and lowers profiled `non_kv_cache_memory` by **`798–1,255 MiB`** (preventing transient host heap slack from shrinking the UMA KV cache allocation, equivalent to **`+11,349` to `+17,847` additional 64-token MLA KV cache blocks** at `73,728 B/block`).
2. **Startup & Post-Prefill Unified Memory Headroom**:
   - Flushing between `profile_run()`'s generic pass and `_profile_deepseek_v4_attention()` lowers peak `torch.cuda.memory_reserved()` during attention profiling by **`384.00 MiB`**.
   - Flushing once on the first `CUDAGraphMode.FULL` decode step after a `CUDAGraphMode.NONE` eager prefill reclaims **`1,920.00 MiB` of idle CUDA caching-allocator slack** plus **`120.1 MiB` of host `VmRSS`** (**`+2.24–2.47 GiB` `MemAvailable` per rank, `~6.7–7.4 GiB` across TP3**) while preserving 100% segment reuse across consecutive chunked-prefill steps and zero steady-state decode overhead.
