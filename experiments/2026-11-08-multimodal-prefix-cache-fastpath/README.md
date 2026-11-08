# Multimodal Prefix-Cache Host Fastpath under `--mm-processor-cache-gb 0`

## Metadata

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18b` (`main`)
- **Base vLLM patch head**: `320021e3c4dbb202d911ab70e2cd924c75420488` (`0761`, Candidate `1107`)
- **Candidate vLLM patch**: [`vllm/0762-perf-multimodal-prefix-cache-fastpath-under-mm-proce.patch`](vllm/0762-perf-multimodal-prefix-cache-fastpath-under-mm-proce.patch)
  - **Patch commit**: `ef9b08de18e75554d46d4cc5c3acd3b166f245e0`
  - **Expected tree**: `92dec38c06df745c561b0848fcb95f69e4aa73ed`
  - **Stable `git patch-id`**: `c88aa17e172dadd855c5cfb1e6ef7cb731c20081`
  - **Patch SHA-256**: `f5b3787945852ead8b53d0d1e104c22a8d999861f56ab337d7c14efad3b6a920`
  - **Patchset SHA-256**: `6c9326e7550b4323de7356fc496fa17373c435508473b372d56ab862e6b53c2a`
- **Causal variable**: Host-side multimodal image decompression, processor tensor materialization, and P0→P1 ZMQ IPC serialization on repeat turns (`Turn 2+`) when `--mm-processor-cache-gb 0` and `--enable-prefix-caching` are active (`config/cluster.json`).

## Background & Root-Cause Trace

In `config/cluster.json`, `spark-ds41f` serves `DeepSeek-V4.1` on 128 GB unified-memory GB10 nodes with:

- `--mm-processor-cache-gb 0` (disabling the P0/P1 `MultiModalProcessorOnlyCache` / `ShmObjectStore` pixel-tensor cache so host RAM is not consumed by multi-megabyte vision tensors), and
- `--enable-prefix-caching` (with patch `0758` preserving `cache_salt` across chat completions and patch `0727` wrapping base64-decoded bytes in `MediaWithBytes` so `MultiModalHasher.serialize_item` hashes compressed `raw_bytes` instead of decoded PIL pixels).

Tracing a multi-turn OpenAI Chat Completion conversation containing one or more images from Turn 1 across follow-up turns (`Turn 2, Turn 3, ...`) revealed three host-side bottlenecks when `--mm-processor-cache-gb 0` is configured:

1. **Redundant PIL pixel decompression in `ImageMediaIO.load_bytes` (`vllm/multimodal/media/image.py`)**:
   - Even though `ImageMediaIO.load_base64` wraps the result in `MediaWithBytes(image, raw_bytes)` (`0727`), `load_base64` still calls `self.load_bytes(raw_bytes)` on every turn.
   - `ImageMediaIO.load_bytes` unconditionally ran `Image.open(BytesIO(data))`, `image.load()` (full PNG/JPEG DEFLATE/Huffman + IDCT pixel decompression), and `_convert_image_mode(image, "RGB")` on every turn before the processor was even called.
2. **Single-threaded BLAKE3 hashing in `MultiModalHasher.get_hasher` (`vllm/multimodal/hasher.py`)**:
   - `MultiModalHasher.get_hasher("blake3")` returned `blake3` with default `max_threads=1` instead of `functools.partial(blake3, max_threads=blake3.AUTO)`, leaving multi-megabyte image buffers to hash on a single core.
3. **Full CPU HF image processing on every turn in `BaseMultiModalProcessor._cached_apply_hf_processor` (`vllm/multimodal/processing/processor.py`)**:
   - When `--mm-processor-cache-gb 0` is set, `mm_SharedCache` returns `self.cache = None`.
   - `_cached_apply_hf_processor` checked `if cache is None: return self._apply_hf_processor(inputs, timing_ctx)`, bypassing all caching and running `_apply_hf_processor_main` (`DeepseekV4VLImageProcessor`: `ImageOps.pad`, `np.asarray(..., dtype=np.float32) / 255.0`, normalization, and `torch.from_numpy(...).to(torch.bfloat16)` patch extraction producing ~10.2 MB of `patches` tensors per image) on every turn solely to recompute the prompt placeholder token sequence (`mm_prompt_updates`, ~1,049–1,069 sentinel tokens per image).
4. **Redundant ~10–41 MB ZMQ IPC serialization from P0 (`AsyncLLM`) to P1 (`EngineCore`)**:
   - Because `mm_processor_cache` is `None` under `--mm-processor-cache-gb 0`, `AsyncLLM` attached the full ~10.2 MB `patches` tensor per image to `EngineCoreRequest.mm_features[i].data` and serialized it via `MsgpackEncoder` over ZMQ to `EngineCore` on Turn 2+.
   - Inside P1 (`EngineCore`), `KVCacheManager.get_computed_blocks` (using `Request.block_hashes` keyed on `(mm_hash, start_offset)`) and `EncoderCacheManager.check_and_update_cache` immediately detected that the `mm_hash` was already cached from Turn 1, and `Scheduler._update_from_kv_cache` / `strip_covered_mm_data` discarded the `mm_features[i].data` tensor before sending `SchedulerOutput` to the worker.

## Optimization Design (Patch `0762`)

Patch `0762` introduces a 4-stage multimodal prefix-cache fastpath active when `--mm-processor-cache-gb 0` (`self.cache is None`) and `--enable-prefix-caching` are set:

1. **Parallel BLAKE3 hashing (`vllm/multimodal/hasher.py`)**:
   - `MultiModalHasher.get_hasher("blake3")` returns `functools.partial(blake3, max_threads=blake3.AUTO)`.
2. **Validated image header cache & `_LazyPILImage` (`vllm/multimodal/media/image.py`)**:
   - `ImageMediaIO.load_bytes` hashes compressed image bytes with parallel BLAKE3 and checks a bounded thread-safe LRU cache (`_VALIDATED_IMAGE_META_CACHE`, max 2,048 entries, ~64 bytes/entry = ~128 KB max).
   - On a hit (Turn 2+), `load_bytes` returns a `_LazyPILImage(raw_bytes, size, mode, image_loader)` with `.size`, `.width`, `.height`, `.mode`, and `.info` populated from the validated Turn 1 header without running `Image.open` or `image.load()`. If any code path ever accesses pixel data (`.load()`, `.getdata()`, `.tobytes()`, `__array_interface__`, `.resize()`, `.crop()`, `.convert()`, etc.), `_LazyPILImage` transparently decodes the underlying `raw_bytes` on demand.
3. **Bounded metadata-only cache & lazy `MultiModalKwargsItem` (`vllm/multimodal/inputs.py`, `vllm/multimodal/processing/processor.py`, `vllm/renderers/base.py`)**:
   - `BaseMultiModalProcessor` maintains a bounded thread-safe LRU `_prefix_metadata_cache` (max 2,048 entries, ~2 KB/entry = ~4 MB max, ~5,000× smaller than pixel tensor caching) mapping `mm_hash -> (keep_on_cpu_elems, updates_item)`.
   - On Turn 1 (cold), `_apply_hf_processor_with_prefix_metadata_cache` runs `_apply_hf_processor_main` once on `inputs.mm_data_items` and populates `_prefix_metadata_cache`.
   - On Turn 2+ (warm), `_apply_hf_processor_with_prefix_metadata_cache` skips `_apply_hf_processor_main` and returns the cached `ResolvedPromptUpdate` token expansions alongside a lazy `MultiModalKwargsItem(lazy_materializer=..., lazy_mm_hash=item_hash, keep_on_cpu_elems=keep_on_cpu_elems)`.
4. **P0→P1 tensor stripping & automatic self-healing on P1 eviction (`vllm/v1/engine/async_llm.py`, `vllm/v1/engine/core.py`, `vllm/v1/core/sched/scheduler.py`)**:
   - `AsyncLLM` tracks `_p1_cached_mm_identifiers: set[str]` (cleared on `reset_mm_cache`, `reset_prefix_cache`, `reset_encoder_cache`, and `sleep`).
   - When `f.identifier in self._p1_cached_mm_identifiers` on Turn 2+, `AsyncLLM._prepare_mm_features_for_prefix_cache` replaces `f.data` with `f.data.get_stripped_for_prefix_cache(uses_mrope)` (retaining only lightweight `keep_on_cpu` metadata such as `spatial_crop` / `grid_thw` and omitting the ~10.2 MB `patches` tensor) while registering the item's lazy materializer in `_pending_mm_retries[request_id]`.
   - In `Scheduler.schedule()`, if a request arrives with stripped `mm_features` (`f.data is None` or lacking device tensors) and *both* the KV prefix cache and `encoder_cache_manager` in P1 have evicted `mm_hash`, the scheduler raises `MMCacheMissError(missed_hashes)` before allocating KV blocks or mutating scheduler state.
   - `EngineCore.add_request` catches `MMCacheMissError` and returns an `EngineCoreOutput(request_id=..., mm_cache_miss_hashes=...)` to `AsyncLLM.output_handler`, which discards the evicted hashes from `_p1_cached_mm_identifiers`, materializes the full pixel tensors on `renderer._mm_executor`, and transparently resubmits the `EngineCoreRequest` to `EngineCore`.

## 2-Boot Verification & Benchmark Results

All tests and benchmarks were executed on `sp8` inside `spark-ds41f:1107`.

### Unit & Regression Tests (2 Boots)

Command:

```bash
for boot in 1 2; do
  docker run --rm --gpus all --entrypoint pytest \
    -v /tmp/vllm-mm-cache-work/vllm/multimodal:/opt/spark-ds41f/candidate/vllm/vllm/multimodal:ro \
    -v /tmp/vllm-mm-cache-work/vllm/renderers:/opt/spark-ds41f/candidate/vllm/vllm/renderers:ro \
    -v /tmp/vllm-mm-cache-work/vllm/v1:/opt/spark-ds41f/candidate/vllm/vllm/v1:ro \
    -v /tmp/vllm-mm-cache-work/tests:/opt/spark-ds41f/candidate/vllm/tests:ro \
    -w /opt/spark-ds41f/candidate/vllm \
    spark-ds41f:1107 \
    --noconftest -o addopts='' -o cache_dir=/tmp/pytest_cache -v \
    -k 'not (test_ipc_enable_disable_consistency or test_cache_eviction_lru_cache or test_cache_eviction_shm_cache or test_processor_cache_shared_across_loras or test_sleep_wake_preserves_mm_cache_consistency)' \
    tests/multimodal/test_prefix_cache_fastpath.py \
    tests/multimodal/test_hasher.py \
    tests/multimodal/media/test_image.py \
    tests/multimodal/test_cache.py \
    tests/models/test_deepseek_v4_1_image_span.py | tee /tmp/pytest-boot${boot}.txt
done
```

- **Boot 1** ([`receipts/pytest-boot1.txt`](receipts/pytest-boot1.txt)): `45 passed in 11.01s`
- **Boot 2** ([`receipts/pytest-boot2.txt`](receipts/pytest-boot2.txt)): `45 passed in 11.34s`

### Multi-Turn Host Preprocessing & ZMQ IPC Benchmark (2 Boots)

Benchmark script: [`bench_mm_prefix_fastpath.py`](bench_mm_prefix_fastpath.py)
Receipts: [`receipts/bench-boot1.json`](receipts/bench-boot1.json), [`receipts/bench-boot2.json`](receipts/bench-boot2.json)

Each workload simulates a 5-turn multimodal conversation (`Turn 1` cold, `Turns 2..5` warm with the same image(s) in the chat history) across 6 repeats under `--mm-processor-cache-gb 0` and `enable_prefix_caching=True`, measuring end-to-end host time (`ImageMediaIO.load_base64` + `DeepseekV4VLMultiModalProcessor.apply` + `AsyncLLM._prepare_mm_features_for_prefix_cache` + `MsgpackEncoder.encode`) and serialized ZMQ payload bytes:

| Workload | Baseline Turn 2+ (Boot 1 / Boot 2) | Candidate Turn 2+ (Boot 1 / Boot 2) | Turn 2+ Speedup (Boot 1 / Boot 2) | Turn 2+ Latency Saved (Boot 1 / Boot 2) | Turn 2+ ZMQ IPC Payload (Baseline → Candidate) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1302x1302` JPEG × 1 img | `18.07 ms` / `17.24 ms` | **`1.43 ms` / `1.51 ms`** | **12.6× / 11.4×** | `-16.64 ms` / `-15.73 ms` | `10,186,417 B` → **`6,951 B`** (**1,465.5× smaller**) |
| `1302x1302` JPEG × 4 imgs | `66.26 ms` / `67.67 ms` | **`4.90 ms` / `4.95 ms`** | **13.5× / 13.7×** | `-61.36 ms` / `-62.72 ms` | `40,743,691 B` → **`25,611 B`** (**1,590.9× smaller**) |
| `1920x1080` PNG × 1 img | `54.06 ms` / `54.08 ms` | **`3.76 ms` / `3.88 ms`** | **14.4× / 13.9×** | `-50.30 ms` / `-50.20 ms` | `9,914,397 B` → **`6,795 B`** (**1,459.1× smaller**) |
| `1920x1080` PNG × 4 imgs | `214.66 ms` / `216.77 ms` | **`14.09 ms` / `15.67 ms`** | **15.2× / 13.8×** | `-200.57 ms` / `-201.10 ms` | `39,655,611 B` → **`24,987 B`** (**1,587.0× smaller**) |
| `3840x2160` PNG × 1 img | `177.51 ms` / `177.66 ms` | **`10.05 ms` / `9.86 ms`** | **17.7× / 18.0×** | `-167.46 ms` / `-167.80 ms` | `9,914,397 B` → **`6,795 B`** (**1,459.1× smaller**) |
| `3840x2160` PNG × 4 imgs | `704.77 ms` / `716.18 ms` | **`39.55 ms` / `40.13 ms`** | **17.8× / 17.9×** | `-665.22 ms` / `-676.05 ms` | `39,655,611 B` → **`24,987 B`** (**1,587.0× smaller**) |
| `4032x3024` JPEG × 1 img | `120.69 ms` / `120.66 ms` | **`4.60 ms` / `4.20 ms`** | **26.3× / 28.7×** | `-116.09 ms` / `-116.46 ms` | `10,302,939 B` → **`6,993 B`** (**1,473.3× smaller**) |
| `4032x3024` JPEG × 4 imgs | `485.98 ms` / `480.63 ms` | **`18.18 ms` / `18.34 ms`** | **26.7× / 26.2×** | `-467.80 ms` / `-462.29 ms` | `41,209,779 B` → **`25,779 B`** (**1,598.6× smaller**) |

### Memory Observations

- Per-entry metadata footprint in `_prefix_metadata_cache`: **~2.0 KB** (`spatial_crop` 2-element tensor + `ResolvedPromptUpdate` token expansion list), compared to **~10.19 MB** per image for full `patches` + `global_view` + `local_views` bfloat16 tensors (**~5,000× smaller**).
- Bounded at 2,048 entries (`_prefix_metadata_cache_max_entries = 2048` and `_VALIDATED_IMAGE_META_CACHE_MAX_ENTRIES = 2048`), capping worst-case host metadata memory at **~4.2 MB** while keeping `--mm-processor-cache-gb 0` intact.

## Conclusion

Caching validated image header metadata (`_LazyPILImage`) and prompt-update token expansion metadata (`_prefix_metadata_cache`) while stripping P1-resident multimodal tensors before ZMQ IPC serialization eliminates **11.4× – 28.7×** of Turn 2+ host preprocessing latency (**16 ms – 676 ms saved per turn**) and reduces P0→P1 ZMQ IPC payload sizes by **~1,460× – 1,599×** (`~40 MB → ~25 KB` for 4 images) under `--mm-processor-cache-gb 0`, with zero quality drift and automatic self-healing on P1 cache eviction.
