# SP Prefill Local `attention_front` (`MXFP8` `q_cast` + `528 B` SWA Record Gather) & Local-Only `index_weights`

- **Date**: 2026-11-08
- **Base commit**: `8e5ea208f3d22bc69df91ace94d6975932c858a8` (`spark-ds41f-1107`, vLLM `0761` `2756478b60b90700b9918b457aeb38657da637b5`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate patch**: `experiments/2026-11-08-sp-prefill-local-attention-front/vllm/0762-perf-dsv41-sp-prefill-local-attention-front.patch` (vLLM `0762` `3515dba65257e6d4260b2362ce2842c4c832d9cb`, tree `31d076ef636c4581ba5f8830c7909817d0370377`)
- **Hardware**: `sp4` (`spark-0012`, NVIDIA GB10, `spark-ds41f:1107`)

## Causal Variable & Rationale

In `vllm/models/deepseek_v4_1/attention.py`, during sequence-parallel prefill (`sp_rows is not None`) in all **54 of 59 layers** without a KV compressor (`self.compressor is None` and `replicated_sp_prefill == True`):

1. **Redundant BF16 `qr` gather and `T`-row `cast_mxfp8`**:
   - Baseline `_gather_front` runs `self.q_norm(qr, out=front[:, :1280])` (`2,560 B/row` BF16) on local `L` rows, all-gathers BF16 `qr` across all TP ranks, and then runs `kernels.cast_mxfp8(qr)` on all `T` gathered rows on every rank.
   - Neither `self.wq_b` (`Q-B`) nor `self.indexer.wq_b` (`indexer Q-B`) reads the BF16 `qr` values when `mxfp8_inputs` (`q_cast`) is supplied; normalizing and casting `qr` to `MXFP8` (`1,280 B` `E4M3` values + `40 B` `UE8M0` scales = `1,320 B/row`) on local `real_rows` before the all-gather cuts `q` collective payload by **`48.44%` (`2,560 -> 1,320 B/row`)** and cuts `cast_mxfp8` rows by `TP` (`T -> real_rows`).
2. **Redundant BF16 `kv` gather and `T`-row RoPE + SWA quantization**:
   - Baseline `_gather_front` runs `self.kv_norm(kv, out=front[:, 1280:1792])` (`1,024 B/row` BF16) on local `L` rows, all-gathers BF16 `kv`, and then runs `self._cache_context_kv(kv, positions)` (`rope_` + `write_cache(..., cache_kind="swa")`) on all `T` gathered rows on every rank, while `forward_mqa` never references its `kv` tensor argument once `swa_cache_layer.kv_cache` is populated.
   - Normalizing `kv` (`512`), applying NeoX RoPE on the `64` dim-pairs using `positions[start:start + real]`, and quantizing to the `528 B/row` SWA KV cache record (`512 B` `E4M3` + `16 B` `UE8M0` scales) on local `real_rows` before the all-gather cuts `kv` collective payload by **`48.44%` (`1,024 -> 528 B/row`)**, cuts `kv_norm + rope_ + SWA quantization` rows by `TP`, and reduces the post-gather step to a pure 528-byte paged scatter (`write_swa_records`).
3. **Unnecessary `index_weights` gather (`copy_columns` + `split_columns`)**:
   - Baseline `_gather_front` packs `self.indexer.weights_proj(hidden_states)` (`[L, 32]` BF16 = `64 B/row`) via `copy_columns`, all-gathers it across all TP ranks, and unpacks it via `split_columns`, only for `_index_query` to slice `index_weights[first:last]` where `[first, last) = [split.start, split.start + split.real_rows)` (this rank's own local rows) whenever `_INDEX_SPLIT_CHECK` is `False`.
   - Keeping `index_weights = self.indexer.weights_proj(hidden_states)[:real]` rank-local when `not _INDEX_SPLIT_CHECK` eliminates `copy_columns`, `split_columns`, and `64 B/row` of collective traffic.

### Implementation Summary (Option B Promoted)

- `vllm/models/deepseek_v4_1/tilelang/front.py`:
  - Hoists `_make_norm(split, M)` to module scope so `attention_front_kernel` and `sp_attention_front_kernel` share the exact two-warp `1280`/`512` RMS-norm reduction.
  - Adds `sp_attention_front_kernel(eps)` / `sp_attention_front(...)` / `sp_front_call(eps)`: one CTA per local row (`real_rows <= L`) with 128 threads reads `qr_kv[r, :1792]` (stride `1792`), writes `q_values[r, :1280]` (`E4M3`) and `q_scales[r, :40]` (`UE8M0`), and writes the rotated SWA record `records[r, :528]` (`512 B` `E4M3` + `16 B` `UE8M0` scales) using `positions[start:start + real]`.
- `vllm/models/deepseek_v4_1/tilelang/cache_writer.py`:
  - Adds `swa_records_kernel(page_size, slot_dtype)` / `write_swa_records(...)` / `swa_records_call(page_size, slot_dtype)`: scatters gathered `[T, 528]` SWA records into `swa_cache_layer.kv_cache` at `slot_mapping[:T]` via 128-bit `dsv41_copy_b128` vector copies (`33 x 16 B` per row) with PDL synchronization.
- `vllm/models/deepseek_v4_1/attention.py`:
  - Replaces `_gather_front` with `_sp_front`, registering `sp_front_call` and `swa_records_call` in `_tilelang_calls` when `self.compressor is None` (removing the old `copy_columns_kernel` and `split_columns_kernel` declarations).
  - All-gathers `(local_q_cast, local_records)` in a single `sp_rows.gather_mxfp8` call (`1,848 B/row` vs `3,648 B/row`, **`-49.34%`**) and scatters SWA records via `write_swa_records`.
  - Keeps `index_weights` rank-local (`[:sp_rows.real_rows]`) unless `_INDEX_SPLIT_CHECK` is enabled; updates `_index_query` to accept either rank-local `[real_rows, 32]` or full `[T, 32]` `index_weights`.

## Collective Payload Comparison (54 Non-Compressor Layers)

| Variant | Per-Row Gather Bytes | Savings vs Baseline | Total Gather Payload across 54 Layers (`T=16,384`, `TP=3`) | Total Gather Payload across 54 Layers (`T=16,384`, `TP=4`) |
| :--- | ---: | ---: | ---: | ---: |
| **Baseline `1107` (`_gather_front`)** | `3,648 B/row` (`2,560 B` `qr` + `1,024 B` `kv` + `64 B` `index_weights`) | — | `3,078.47 MiB` | `3,078.00 MiB` |
| **Option A (Unfused Local `q_cast` + Local Rotated BF16 `kv` + Local `index_weights`)** | `2,344 B/row` (`1,320 B` `q_cast` + `1,024 B` `kv`) | `-35.75%` (`-1,304 B/row`) | `1,978.05 MiB` (`-1,100.42 MiB`) | `1,977.75 MiB` (`-1,100.25 MiB`) |
| **Option B (Promoted: `sp_attention_front` + `528 B` SWA Record Gather + Local `index_weights`)** | **`1,848 B/row`** (`1,320 B` `q_cast` + `528 B` SWA record) | **`-49.34%` (`-1,800 B/row`)** | **`1,559.49 MiB` (`-1,518.98 MiB`)** | **`1,559.25 MiB` (`-1,518.75 MiB`)** |

## 2-Boot GPU Kernel Latency Benchmarks (`sp4`, NVIDIA GB10)

Measured across 2 fresh container boots of `spark-ds41f:1107` (`40` warmup iterations, `200` timed iterations with CUDA events; receipts in `receipts/boot1.json` and `receipts/boot2.json`):

| Shape (`TP`, `T`, `L`) | Baseline `1107` Kernel Time (Boot 1 / Boot 2) | Option A Kernel Time (Boot 1 / Boot 2) | **Option B (`sp_attention_front` + `write_swa_records`) (Boot 1 / Boot 2)** | **Option B Speedup (Boot 1 / Boot 2)** | **54-Layer Kernel Time Saved (Boot 1 / Boot 2)** | `sp_attention_front(L)` Alone (Boot 1 / Boot 2) | `write_swa_records(T)` Alone (Boot 1 / Boot 2) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TP=3, T=4096 (L=1366)` | `243.68 us` (IQR `5.89`) / `242.91 us` (IQR `2.94`) | `78.08 us` / `78.05 us` (`3.12x` / `3.11x`) | **`25.12 us`** (IQR `0.29`) / **`24.99 us`** (IQR `0.22`) | **`9.70x` / `9.72x`** | **`11.80 ms` / `11.77 ms`** (`-218.6 / -217.9 us/layer`) | `20.42 us` / `20.80 us` | `14.69 us` / `13.44 us` |
| `TP=3, T=8192 (L=2731)` | `568.29 us` (IQR `2.88`) / `566.40 us` (IQR `4.13`) | `173.18 us` / `167.14 us` (`3.28x` / `3.39x`) | **`62.75 us`** (IQR `1.18`) / **`52.35 us`** (IQR `1.06`) | **`9.06x` / `10.82x`** | **`27.30 ms` / `27.76 ms`** (`-505.5 / -514.0 us/layer`) | `17.73 us` / `19.71 us` | `16.54 us` / `15.30 us` |
| `TP=3, T=16384 (L=5462)` | `1108.96 us` (IQR `2.43`) / `1119.39 us` (IQR `2.94`) | `341.95 us` / `344.16 us` (`3.24x` / `3.25x`) | **`223.20 us`** (IQR `1.25`) / **`222.46 us`** (IQR `1.18`) | **`4.97x` / `5.03x`** | **`47.83 ms` / `48.43 ms`** (`-885.8 / -896.9 us/layer`) | `127.17 us` / `122.02 us` | `23.81 us` / `21.73 us` |
| `TP=4, T=4096 (L=1024)` | `233.50 us` (IQR `2.02`) / `235.42 us` (IQR `2.18`) | `76.61 us` / `75.26 us` (`3.05x` / `3.13x`) | **`25.12 us`** (IQR `0.70`) / **`24.99 us`** (IQR `0.26`) | **`9.30x` / `9.42x`** | **`11.25 ms` / `11.36 ms`** (`-208.4 / -210.4 us/layer`) | `19.39 us` / `19.58 us` | `15.23 us` / `15.14 us` |
| `TP=4, T=8192 (L=2048)` | `548.26 us` (IQR `1.31`) / `546.91 us` (IQR `3.26`) | `144.58 us` / `133.31 us` (`3.79x` / `4.10x`) | **`32.99 us`** (IQR `1.70`) / **`38.94 us`** (IQR `1.02`) | **`16.62x` / `14.04x`** | **`27.82 ms` / `27.43 ms`** (`-515.3 / -508.0 us/layer`) | `19.74 us` / `20.67 us` | `16.64 us` / `16.26 us` |
| `TP=4, T=16384 (L=4096)` | `1062.18 us` (IQR `2.37`) / `1073.02 us` (IQR `3.10`) | `286.82 us` / `289.06 us` (`3.70x` / `3.71x`) | **`189.50 us`** (IQR `1.79`) / **`193.09 us`** (IQR `17.50`) | **`5.61x` / `5.56x`** | **`47.12 ms` / `47.52 ms`** (`-872.7 / -879.9 us/layer`) | `41.15 us` / `42.05 us` | `23.81 us` / `25.44 us` |

## Bit-Exactness & Test Verification

1. **Bit-Exactness (`bench_sp_attention_front.py`, 2 Boots)**:
   - Verified across **14 shapes** (`TP in {3, 4}` x `T in {60, 65, 205, 206, 207, 512, 4096}`, covering both exact multiples of `TP` and non-divisible token counts where `real_rows < local_rows` on tail ranks):
     - `q_values` (`[T, 1280]` `float8_e4m3fn` viewed as `uint8`): **`0` byte diffs** (`torch.equal == True`).
     - `q_scales` (`[T, 40]` `uint8` `UE8M0`): **`0` byte diffs** (`torch.equal == True`).
     - `swa_cache_layer.kv_cache` (`[num_blocks, page_size, 528]` `uint8`): **`0` byte diffs** (`torch.equal == True`).
     - `index_weights` (`[real_rows, 32]` `bfloat16` viewed as `uint16`): **`0` uint16 diffs** (`torch.equal == True`).
     - `index_slice_mxfp8` (`q_cast.slice_rows(start, start + real_rows)`): **`0` byte diffs** (`torch.equal == True`).
2. **Unit & Integration Test Suite (2 Fresh Container Boots)**:
   - Command:
     ```bash
     flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host --entrypoint pytest \
       -v /tmp/vllm-git-1107/vllm/models/deepseek_v4_1:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1:ro \
       -v /tmp/vllm-git-1107/tests:/opt/spark-ds41f/candidate/vllm/tests:ro \
       -w /opt/spark-ds41f/candidate/vllm spark-ds41f:1107 --noconftest -v \
       tests/kernels/test_deepseek_v41_tilelang_front.py \
       tests/models/test_deepseek_v4_1_attention.py \
       tests/models/test_sp_prefill.py \
       tests/models/test_sp_prefill_e2e.py
     ```
   - Boot 1 (`receipts/pytest_boot1.txt`): **`368 passed in 24.91s`** (`PDL sources: 33 kernels launched, 28 with PDL, 0 violations`).
   - Boot 2 (`receipts/pytest_boot2.txt`): **`368 passed in 22.72s`** (`PDL sources: 33 kernels launched, 28 with PDL, 0 violations`).
3. **Memory Observations**:
   - Eliminates the `[L, 1824]` (`3,648 B/row`) and `[T, 1824]` (`3,648 B/row`) BF16 `_gather_front` staging buffers and the `[T, 1280]` BF16 `qr` and `[T, 512]` BF16 `kv` all-gather outputs on all 54 non-compressor layers, replacing them with `[L, 1280]` + `[L, 40]` `local_q_cast` (`1,320 B/row`) and `[L, 528]` `local_records` (`528 B/row`), saving **`5,448 B/row` of peak transient prefill activation buffers** (`85.125 MiB` at `T=16,384`).

## Conclusion

Promoted Option B (`0762-perf-dsv41-sp-prefill-local-attention-front.patch`): reduces `_sp_front` collective payload by **`49.34%` (`3,648 -> 1,848 B/row`, `-1,518.75 MiB` per 16K prefill across 54 layers)** and speeds up the per-layer SP attention front kernel sequence by **`4.97x–16.62x`** (`-47.12` to `-48.43 ms` of GPU kernel time per 16K prefill across 54 layers on GB10) with **100% bit-exact outputs (`0` byte/uint16 diffs)**.
