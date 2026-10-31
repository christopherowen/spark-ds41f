# Column-Masked Layer-Major Host Engram Hash and Known-Only D2H Stage Elision

## Hypothesis

In tensor-parallel serving with disk-backed Engram tables (`TP=3` or `TP=4`), the host prefill n-gram hash (`HostEngramHash.step_ids`) and `EngramDiskRows.stage` perform redundant host CPU and GPU-to-host work on every prefill chunk:

1. **Redundant `np.remainder` across unowned TP columns**: Each rank's `EngramDiskRows` reader filters every lookup against its `columns` mask (`8` of `24` columns per rank at `TP=3`, `6` of `24` columns per rank at `TP=4`). However, `HostEngramHash` expands `mixed2`, `mixed3`, and `mixed4` via `np.repeat(..., 8)` + `np.concatenate` and computes 64-bit integer division (`np.remainder`) for all `24` columns on every TP rank.
2. **Non-contiguous per-layer slices**: `HostEngramHash.step_ids` allocates `[T, layers, 24]` in token-major order, so `early[:, i]` passed to each layer's `rows.stage(...)` is non-contiguous and triggers a per-layer `np.ascontiguousarray` allocation and copy on the runner thread.
3. **Redundant D2H copy of GPU-hashed IDs when all lookups are known on the host**: During pure prefill steps without draft verification (`has_unknown == False`) when `check_early == False`, the reader plans and reads 100% of the step's disk rows from the host-hashed `early` array and never inspects the mapped `self.ids` buffer (unless `check_early` is enabled). Yet `EngramDiskRows.stage` still enqueues an asynchronous D2H copy `self.ids[:count].copy_(ids.view(-1)[:count], non_blocking=True)` (`3.15 MB` per layer at `T = 16384`, or `6.29 MB` across both Engram layers) before signaling `self._ids_seq_device`.

We hypothesize that:
- Backing `HostEngramHash.step_ids` and `hash_positions` with a layer-major `[layers, T, 24]` buffer returned as its `[T, layers, 24]` transpose view makes every per-layer slice `early[:, layer]` C-contiguous (`0` copy in `stage`).
- Broadcasting `[layers, T, 3, 1]` directly against `[layers, 1, 3, 8]` (for `ALL_COLUMNS`) and hashing only the rank's owned columns `columns` (for `TP > 1` when `check_early` is disabled) cuts Grace CPU `step_ids` + `stage` slice time by >50% at `TP=3` and ~60% at `TP=4`.
- Exposing `native.has_unknown(early)` and passing `has_unknown` into `native.submit` allows `EngramDiskRows.stage` to scan `early` once in C and elide `self.ids[:count].copy_` whenever `!has_unknown && !self.check_early`.

## Changes

Patch: `0001-engram-column-masked-host-hash-and-stage.patch` (against `vllm` commit `1f18be14470339c1227ed0432b896a8ca09811b5`).

- `vllm/models/deepseek_v4_1/engram_rows/host_hash.py`:
  - Precompute reshaped geometry views `_multipliers_u64`, `_primes_3x8`, `_offsets_3x8`, and cached per-mask column plans `_column_plan(columns)`.
  - Store step/position planes in `[layers, T, 24]` layout and return `planes.transpose(1, 0, 2)` (`[T, layers, 24]`) so each layer's `[:, layer]` slice is C-contiguous.
  - Add keyword argument `columns: int = ALL_COLUMNS` to `hash_positions` and `step_ids`. When `columns != ALL_COLUMNS`, compute `np.remainder` + `offsets` only for the set bits of `columns` and fill unowned columns with `-1` (`DEAD_ID`).
- `vllm/models/deepseek_v4_1/nvidia/model_state.py`:
  - In `_early_engram_ids`, pass the rank's `column_mask` to `self._host_engram_hash.step_ids(..., columns=columns)` when `not ENGRAM_CHECK_EARLY_IDS` and `not check_early`.
- `vllm/models/deepseek_v4_1/engram_rows/_reader.c` & `rows.py`:
  - Add `has_unknown(early)` to `_reader` and accept an optional precomputed `has_unknown` flag in `submit` so `early` is scanned only once.
  - In `EngramDiskRows.stage`, skip `np.ascontiguousarray` when `early` is already C-contiguous `int64`, and elide `self.ids[:count].copy_(...)` when `not has_unknown and not self.check_early` while still writing `_sequence` to `_ids_seq_device`.
- `tests/kernels/test_deepseek_v41_engram_host_placement.py` & `tests/kernels/test_deepseek_v41_engram_rows.py`:
  - Add `test_step_ids_layer_slices_are_contiguous_and_column_mask_matches_full_hash` and `test_column_masked_early_ids_skip_d2h_ids_copy_and_match_reference`.

## Verification & Benchmark Results (`dgx4`, `spark-ds41f:1030`)

### Correctness

- `pytest --noconftest -q tests/kernels/test_deepseek_v41_engram_host_placement.py`: `6 passed`
- `pytest --noconftest -q tests/kernels/test_deepseek_v41_engram_rows.py`: `22 passed`
- `pytest -q tests/models/test_deepseek_v4_1_engram.py -k "test_disk_engram_rows_the_host_hashed_equal_the_gpu_rows or test_disk_rows_read_whole_table_sources"`: `3 passed`

### Grace CPU `HostEngramHash.step_ids` + Layer-Slice Benchmark (`layers = 2`, `vocab = 129,280`)

| Prefill Tokens (`T`) | Mode | Legacy `step_ids` + 2x `ascontiguousarray` (µs) | New `step_ids` (µs, 0-copy layer slice) | Latency Delta | D2H Copy Elided (2 layers) |
|---:|---|---:|---:|---:|---:|
| `2048` | `all_24_cols` (`TP=1`) | `625.2` | `553.1` | `-11.5%` | `0.79 MB` |
| `2048` | `tp3_8_cols` (`TP=3`) | `625.2` | `292.5` | `-53.2%` | `0.79 MB` |
| `2048` | `tp4_6_cols` (`TP=4`) | `625.2` | `250.0` | `-60.0%` | `0.79 MB` |
| `4096` | `all_24_cols` (`TP=1`) | `1,240.3` | `1,092.6` | `-11.9%` | `1.57 MB` |
| `4096` | `tp3_8_cols` (`TP=3`) | `1,240.3` | `569.2` | `-54.1%` | `1.57 MB` |
| `4096` | `tp4_6_cols` (`TP=4`) | `1,240.3` | `485.8` | `-60.8%` | `1.57 MB` |
| `16384` | `all_24_cols` (`TP=1`) | `6,230.2` | `4,755.5` | `-23.7%` | `6.29 MB` |
| `16384` | `tp3_8_cols` (`TP=3`) | `6,230.2` | `2,517.0` | `-59.6%` | `6.29 MB` |
| `16384` | `tp4_6_cols` (`TP=4`) | `6,230.2` | `2,150.2` | `-65.5%` | `6.29 MB` |
