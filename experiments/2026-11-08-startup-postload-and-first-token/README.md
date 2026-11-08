# Post-Load Weight Preparation, On-Disk Artifact Cache, and First-Request Prefill Warmup

## Metadata

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`)
- **Base vLLM patch head**: `320021e3c4dbb202d911ab70e2cd924c75420488` (`0761`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate vLLM patch**: `0762-perf-dsv41-vectorize-postload-add-artifact-cache-and.patch`
  - **Patch head**: `06d47ca498cc5b914d9ea29dce3bde8d8415411b`
  - **Expected tree**: `5eef95329d15ebbfbdad0dcf90382c8fdd682fa5`
  - **Patchset SHA-256**: `01dd1c044d15a7cb2161c0a2dacdbb3d83b8a9c9c39cb9e6a2a8f2c2e1ce189e`
- **Hardware**: `sp1` (`spark-0009`, NVIDIA GB10 `SM121`)

## Motivation & Causal Variables

1. **MoE Post-Load Scale Encoding and Weight Tiling (`scale_codes.py`, `moe.py`, `model_loader/utils.py`)**:
   - `_row_deltas` previously ran separate `e > 0`, `torch.where`, `amin`, `any`, and `max` reductions even when all scale exponents were positive (`row_min > 0`).
   - `pack_expert_scale_codes` and `MoeExpertsV41Plan.prepare_weight` previously computed 4-bit nibbles (`_pack_nibbles`) and `overlay_byte_codes` copies even when all experts were narrow (`delta <= 3`).
   - `tile_expert_weights` permuted 64-byte FP4 TMA slabs byte-by-byte (`uint8` / `int8`) rather than as 8-byte `int64` words.
   - `process_weights_after_loading` called `release_device_memory_under_pressure(target_device)` after every submodule, invoking `torch.cuda.empty_cache()` whenever reserved memory exceeded 80% of total memory even when the submodule allocated no new segments (`torch.cuda.memory_reserved(target_device) <= reserved_before`).

2. **In-Place 12-Bit BF16 Packing (`vocab.py`) and TP3 Rank 2 `wq_b` Trimming (`gemm.py`, `attention.py`)**:
   - `pack_bf16_weight(in_place=True)` previously staged and copied every `1,024`-row chunk across all `out_features` rows (`43` kernel launches + `43` staging `.copy_()` calls on TP3 `lm_head` `[43264, 7168]`). Because row `r`'s packed output occupies `[1.5 * K * r, 1.5 * K * (r + 1))` and row `r`'s BF16 input occupies `[2 * K * r, 2 * K * (r + 1))`, any launch covering rows `[start, end)` with `start >= 3,072` and `3 * end <= 4 * start` writes strictly below `2 * K * start` without overlapping unread input rows.
   - On TP3 rank 2 (`real_groups = 4 < padded_groups = 6`), `_trim_padded_groups` previously unpacked `12,288` `wq_b` rows (`unpack_weight_tiles`), sliced `[:8192]`, and repacked `8,192` rows (`pack_weight_tiles`) across all 59 layers even when `tilelang_weight_packed` was already `True` and `(q_rows, k_q) == (8192, 1280)` is in `TILE_CONTIGUOUS_FP8`.

3. **Per-Rank On-Disk Post-Load Artifact Cache (`postload_cache.py`)**:
   - Added an optional per-rank on-disk post-load artifact cache (`VLLM_POSTLOAD_CACHE_DIR` / `SPARK_DS41F_POSTLOAD_CACHE_DIR`) for prepared `pack12` (`PackedBf16Weight`) and `moe_scale` (`encode_expert_scale_codes`) buffers:
     - **First Start (Background Atomic Write)**: After `process_weights_after_loading` computes the prepared tensors, a single background thread copies the tensors to host memory on a dedicated CUDA stream and writes them to a PID-scoped staging file (`*.tmp.<pid>`) before atomically publishing via `os.replace` when free disk space is `>= 8 GiB` (`MIN_FREE_DISK_BYTES = 8 * (1 << 30)`). `loading_done()` joins queued writes at the end of model post-load processing, and `cleanup_dead_pid_staging_files` removes stale `*.tmp.<pid>` files left by dead processes.
     - **Cached Start (Direct Read + 4-Slice Byte-for-Byte Spot Check)**: Subsequent starts read the prepared buffer directly from disk (`O_DIRECT` into page-aligned `mmap` when `VLLM_POSTLOAD_CACHE_ODIRECT=1`, or unbuffered `readinto` into a reusable scratch tensor released at `loading_done()`) and verify integrity without hashing the entire buffer by running the reference transform on 4 slices (`first = 0`, `second = min(1, n - 1)`, `last = n - 1`, and 1 per-boot random index `spot_check_indices(n)`) from the loaded checkpoint tensor **before** mutating `weight` in place, comparing them byte-for-byte against the cached file. Any mismatch or I/O error falls back to full computation and replaces the cache file.

4. **API Server / Tokenizer Startup & First-Request Prefill Warmup (`deepseek_v41/__init__.py`, `vl_model.py`, `tokenizers/registry.py`, `nvidia/model.py`)**:
   - Converted `vllm/models/deepseek_v41/__init__.py`, `vllm/models/deepseek_v4_1/nvidia/vl_model.py`, and `vllm/models/deepseek_v41/nvidia/vl_model.py` to lazy `__getattr__` imports so resolving `DeepseekV41ForCausalLM` in the API server does not eagerly import `tilelang` and `flashinfer`.
   - Passed `config=model_config.hf_config` into `get_tokenizer` for `TokenizersBackend` subclasses (`DeepseekV41Tokenizer`) to avoid reloading `AutoConfig.from_pretrained`.
   - Warmed `self._build_l2_prefetch()` and `sp_prefill._side_stream(first_param.device)` inside `DeepseekV4Model.declare_kernels(warm=...)` so the first eager prefill request after startup does not lazily construct 59 layers of L2 prefetch device pointer tables or lazily allocate the SP side stream on the request critical path.

## Exact Verification & Benchmark Commands

```bash
# 2-boot hardware benchmark on sp1 (GB10)
flock /tmp/gpu.lock bash -c '
  docker run --rm --gpus all --ipc=host --entrypoint python3 $MOUNTS spark-ds41f:1107 \
    /exp/bench_startup_postload.py /exp/receipts/bench-boot1.json
  docker run --rm --gpus all --ipc=host --entrypoint python3 $MOUNTS spark-ds41f:1107 \
    /exp/bench_startup_postload.py /exp/receipts/bench-boot2.json
'

# 2-boot unit and contract test suite on sp1 (GB10)
flock /tmp/gpu.lock bash -c '
  docker run --rm --gpus all --ipc=host --entrypoint python3 $MOUNTS spark-ds41f:1107 \
    -m pytest --noconftest -p no:cacheprovider -v \
    tests/kernels/test_deepseek_v41_postload_startup.py \
    tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py \
    tests/tokenizers_/test_deepseek_v41.py \
    tests/v1/attention/test_sparse_indexer_decode_seq_lens.py \
    > receipts/pytest-boot1.txt 2>&1
  docker run --rm --gpus all --ipc=host --entrypoint python3 $MOUNTS spark-ds41f:1107 \
    -m pytest --noconftest -p no:cacheprovider -v \
    tests/kernels/test_deepseek_v41_postload_startup.py \
    tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py \
    tests/tokenizers_/test_deepseek_v41.py \
    tests/v1/attention/test_sparse_indexer_decode_seq_lens.py \
    > receipts/pytest-boot2.txt 2>&1
'
```

## 2-Boot Hardware Results (`sp1`, NVIDIA GB10 `SM121`)

Raw JSON receipts: `receipts/bench-boot1.json` and `receipts/bench-boot2.json`.

| Metric | Boot 1 Baseline (`1107`) | Boot 1 Candidate (`0762`) | Boot 2 Baseline (`1107`) | Boot 2 Candidate (`0762`) | Speedup / Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| **58-Layer MoE Post-Load (Cold Compute)** | `1.7026 s` | `0.9153 s` | `1.7045 s` | `0.9162 s` | **1.86x** (`-0.788 s`) |
| **58-Layer MoE Post-Load (Total w/ `empty_cache` / BG Flush)** | `1.7165 s` | `0.9890 s` | `1.7186 s` | `0.9850 s` | **1.74x** (`-0.731 s`) |
| **58-Layer MoE Post-Load (Cached + 4-Expert Spot Check)** | `1.7026 s` | `0.6962 s` | `1.7045 s` | `0.6931 s` | **2.45x–2.46x** (`-1.009 s`) |
| **`lm_head` `[43264, 7168]` In-Place `pack12` (Cold)** | `14.04 ms` | `11.92 ms` | `13.80 ms` | `11.16 ms` | **1.18x–1.24x** (`-2.38 ms`) |
| **TP3 Rank 2 `wq_b` Trim (`59` layers `[12288, 1280]` -> `[8192, 1280]`)** | `16.153 ms` | `0.0549 ms` | `15.542 ms` | `0.0556 ms` | **280x–294x** (`-15.8 ms`) |
| **Total Post-Load Weight Prep (Cold)** | `1.7980 s` | `1.0640 s` | `1.7979 s` | `1.0597 s` | **1.69x–1.70x** (`-0.736 s`) |
| **Total Post-Load Weight Prep (Cached)** | `1.7980 s` | `0.9839 s` | `1.7979 s` | `0.9896 s` | **1.82x–1.83x** (`-0.811 s`) |
| **Model Class Resolution (`_try_load_model_cls`)** | `7,107.04 ms` | `4,395.74 ms` | `7,284.00 ms` | `4,490.58 ms` | **1.62x** (`-2.75 s`) |
| **First-Prefill L2 Prefetch + SP Stream Setup (`59` layers)** | `1.142 ms` | `0.0033 ms` | `1.132 ms` | `0.0032 ms` | **346x–354x** (`-1.13 ms`) |
| **Bit-Identical (`torch.equal`) Cold & Cached** | — | `true` | — | `true` | **100% exact** |
| **2-Boot Pytest Suite (`receipts/pytest-boot{1,2}.txt`)** | — | `170 passed (17.20s)` | — | `170 passed (17.11s)` | **0 failures** |

## Memory Observations

- Staging only the first `3,072` overlapping rows in `pack_bf16_weight(in_place=True)` keeps peak temporary allocation at a single `1,024`-row staging slab (`11.02 MiB` for `K = 7168`) while eliminating `40` staging `.copy_()` passes (`432.5 MiB` of device-to-device copies) on `lm_head`.
- `_READ_SCRATCH` in `postload_cache.py` is released inside `loading_done()` at the end of `process_weights_after_loading`, leaving `0 B` of steady-state host or device memory overhead.
- Pre-warming `_build_l2_prefetch()` and `sp_prefill._side_stream()` inside `DeepseekV4Model.declare_kernels(warm=...)` allocates the exact same per-layer L2 prefetch device pointer tables (`59` layers) and SP side stream that were previously allocated lazily on the first eager prefill request, so steady-state serving memory is unchanged (`0 B` delta after the first request).

## Conclusion

Vectorizing MoE scale-code encoding and 8-byte TMA slab permutation, replacing full-row staging in `pack_bf16_weight(in_place=True)` with non-overlapping geometric direct writes (`3 * end <= 4 * start`), slicing packed `wq_b` tiles directly on TP3 rank 2, adding the per-rank on-disk post-load artifact cache with 4-slice byte-for-byte spot verification, making `deepseek_v41` package imports lazy, and pre-warming L2 prefetch and SP prefill state during `declare_kernels` cuts post-load preparation time by **1.70x cold / 1.82x cached** and model class resolution by **2.75 s (1.62x)** with 100% bit-identical weights and outputs across 2 fresh container boots.
