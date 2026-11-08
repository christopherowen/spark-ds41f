# Disk Engram Direct Cache Insert, Aligned Tag-Split Cache & In-Step Miss Deduplication

- **Date**: 2026-11-08
- **Base commit**: `cc20ed18b` (`spark-ds41f-1107`, vLLM patch `0761` `320021e3c4dbb202d911ab70e2cd924c75420488`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate patch**: `vllm/0762-perf-dsv41-direct-insert-packed-engram-cache-and-ali.patch` (`be312021678a254dc6c374dd628a961d91f41bda`, tree `00c6b813fe0ab2c201afb2ccca4f5a60c39afdcf`)
- **Intended causal variable**: User-space disk Engram row cache memory layout, packed-row cache insertion path, and in-step miss planning in `vllm/models/deepseek_v4_1/engram_rows/_row_plan.h` and `_reader.c`.

## Rationale

In `0750-Fuse-multi-CTA-disk-Engram-step-and-wire-packed-cach.patch`, the 4-way set-associative user-space row cache (`65,536` rows = `16,384` sets) in `vllm/models/deepseek_v4_1/engram_rows/_row_plan.h` used an array of 1,096-byte interleaved `ple_cache_set_t` structs and a two-hop miss insertion path:

1. **Interleaved tag/data cache sets and unaligned weight rows**:
   - `sizeof(ple_cache_set_t)` was `1,096` bytes (`4 * 8B` tags + `1B` clock + `1,024B` weights + `32B` scales + `7B` tail padding).
   - Because `1,096 % 64 == 8`, the 32-byte `ids[4]` tag array straddled two 64-byte CPU cache lines on 37.5% of sets, `weights` started at byte offset `33` (unaligned for 64-byte cache lines and 16-byte NEON loads/stores), and tag probes spanned an 18 MiB address range larger than Grace's 1 MiB per-core L2 cache.
   - Splitting the cache into 64-byte-aligned `cache_tags` (`512 KiB` total for 64K rows, `32 B` per set, two sets per 64B cache line, zero cache-line straddling), `cache_clocks` (`16 KiB`), 64-byte-aligned prefaulted `cache_weights` (`16 MiB`), and 8-byte-aligned prefaulted `cache_scales` (`512 KiB`) keeps all tag probes resident in L2 and aligns every 256-byte weight row to a 64-byte cache line.

2. **Double-copy bounce through `miss_weights` / `miss_scales` on packed single-row misses**:
   - Previously, `ple_stash_job()` copied every `io_uring` completion for a packed 264-byte row into `reader->miss_weights[lookup * 256]` and `reader->miss_scales[lookup * 8]`, and `ple_cache_commit()` then copied the same 264 bytes a second time from `miss_weights`/`miss_scales` into the cache set.
   - For non-page-straddling packed rows (`reader->packed && fragment->length == PLE_PACKED_ROW_BYTES`), `ple_stash_job()` now calls `ple_cache_insert()` directly from the `io_uring` slot buffer (`job->saved_buffer`), eliminating the intermediate bounce copy and `miss_staged` bookkeeping. Two-plane and page-straddling fragments stage into dense `miss_idx` (`0 .. miss_count - 1`) offsets instead of sparse `lookup` stride.

3. **In-step duplicate miss deduplication and planning fast paths**:
   - When multiple lookups in the same step miss on the same `(layer, id)` row (e.g., repeated n-gram hashes across concurrent sequences or speculative draft trees), `ple_plan()` previously emitted duplicate fragments, duplicate `miss_ids` entries, and duplicate `ple_cache_insert()` writes into the same cache set.
   - `ple_plan()` now filters duplicate misses within a step via a 64-entry stack filter (`step_seen_id`), bypasses 64-bit `/` + `%` + `bsearch()` when `shard == 0` (the common single-shard layout), compacts `ple_fragment_t` from 40 B to 32 B, and uses insertion sort in `ple_sort()` when `fragment_count <= 32`.

## Commands & Workload Identity

Executed on `sp7` (`GB10`, kernel `7.0.0-1019-nvidia-64k`, NVMe `/mnt/nvme` `O_DIRECT`) inside `spark-ds41f:1107` under `flock /tmp/gpu.lock` across 2 fresh container boots:

```bash
# Unit test verification (Boot 1 and Boot 2)
flock /tmp/gpu.lock docker run --rm --gpus=all --ipc=host \
  --cap-add=IPC_LOCK --ulimit=memlock=-1:-1 \
  --security-opt=seccomp=unconfined --security-opt=label=disable \
  --entrypoint python3 \
  -v ... \
  spark-ds41f:1107 -m pytest --noconftest -v tests/kernels/test_deepseek_v41_engram_rows.py

# 2-boot cache and packed-layout benchmark (Boot 1 and Boot 2)
flock /tmp/gpu.lock docker run --rm --gpus=all --ipc=host \
  --cap-add=IPC_LOCK --ulimit=memlock=-1:-1 \
  --security-opt=seccomp=unconfined --security-opt=label=disable \
  --entrypoint python3 \
  -v /mnt/nvme:/mnt/nvme \
  -v ... \
  spark-ds41f:1107 -c "import json, bench_and_verify as b; print(json.dumps(b.run_cache_and_packed_benchmark(bench_dir='/mnt/nvme/engram_bench_test'), indent=2))"
```

## Results

### 1. Correctness & Unit Tests (`receipts/pytest-boot1.txt`, `receipts/pytest-boot2.txt`)

- **Boot 1**: `52 passed in 55.52s`
- **Boot 2**: `52 passed in 55.90s`
- Includes `test_packed_row_cache_direct_insert_and_duplicate_miss_dedup[0]` (aligned 264B direct cache insert + duplicate in-step miss IDs) and `test_packed_row_cache_direct_insert_and_duplicate_miss_dedup[3828]` (page-boundary-straddling fragment assembly + cache commit).

### 2. Two-Boot Benchmark Comparison vs Recorded `0750` Baseline (`receipts/bench-cache-boot1.json`, `receipts/bench-cache-boot2.json`)

Comparison against the recorded 2-boot baseline from `experiments/2026-11-06-engram-step-and-cache/receipts/bench-cache-boot{1,2}.json` (mean of Boot 1 and Boot 2, in microseconds per step):

| Regime | Batch (`M`) | `0750` `two_plane_cache64k` (us) | `0762` `two_plane_cache64k` (us) | Delta | `0750` `packed_cache64k` (us) | `0762` `packed_cache64k` (us) | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Cold Miss** (0% hit) | 1 | 21.76 (23.9, 19.6) | **20.85** (19.6, 22.1) | **-4.2%** | 18.88 (21.5, 16.3) | **18.25** (16.5, 20.0) | **-3.3%** |
| **Cold Miss** (0% hit) | 4 | 32.08 (31.8, 32.4) | **30.80** (29.2, 32.4) | **-4.0%** | 22.67 (22.8, 22.5) | **21.60** (20.2, 23.0) | **-4.7%** |
| **Cold Miss** (0% hit) | 16 | 61.05 (61.8, 60.3) | **59.55** (56.6, 62.5) | **-2.5%** | 40.49 (37.4, 43.6) | **37.35** (35.0, 39.7) | **-7.8%** |
| **Cold Miss** (0% hit) | 32 | 92.19 (93.5, 90.9) | **85.30** (84.6, 86.0) | **-7.5%** | 78.76 (78.8, 78.8) | **69.15** (65.8, 72.5) | **-12.2%** |
| **Zipfian Working Set** (~95% hit) | 1 | 15.83 (16.2, 15.4) | **15.80** (14.4, 17.2) | **-0.2%** | 15.07 (15.4, 14.8) | **15.05** (14.7, 15.4) | **-0.1%** |
| **Zipfian Working Set** (~94% hit) | 4 | 15.95 (16.2, 15.7) | **16.65** (15.4, 17.9) | +4.4% | 15.56 (15.8, 15.3) | **14.90** (14.6, 15.2) | **-4.2%** |
| **Zipfian Working Set** (~95% hit) | 16 | 24.48 (24.5, 24.4) | **24.65** (24.1, 25.2) | +0.7% | 21.14 (21.3, 21.0) | **20.60** (19.6, 21.6) | **-2.6%** |
| **Zipfian Working Set** (~95% hit) | 48 | 33.76 (33.3, 34.2) | **33.40** (33.3, 33.5) | **-1.1%** | 25.83 (25.5, 26.2) | **24.70** (24.5, 24.9) | **-4.4%** |
| **Zipfian Working Set** (~95% hit) | 128 | 51.31 (52.4, 50.2) | **49.95** (48.8, 51.1) | **-2.7%** | 33.34 (33.5, 33.2) | **31.90** (30.7, 33.1) | **-4.3%** |
| **Hot Cache** (100% hit) | 1 | 31.12 (46.8, 15.4) | **21.40** (27.6, 15.2) | **-31.2%** | 46.98 (78.6, 15.3) | **15.00** (15.2, 14.8) | **-68.1%** |
| **Hot Cache** (100% hit) | 4 | 15.76 (16.2, 15.3) | **15.20** (15.6, 14.8) | **-3.6%** | 20.39 (25.2, 15.6) | **14.85** (14.6, 15.1) | **-27.2%** |
| **Hot Cache** (100% hit) | 16 | 15.52 (15.9, 15.2) | **14.70** (14.7, 14.7) | **-5.3%** | 15.20 (15.4, 15.0) | **14.85** (14.9, 14.8) | **-2.3%** |
| **Hot Cache** (100% hit) | 32 | 15.90 (16.4, 15.4) | **15.20** (15.4, 15.0) | **-4.4%** | 15.88 (16.4, 15.3) | **15.20** (15.5, 14.9) | **-4.3%** |
| **Hot Cache** (100% hit) | 64 | 16.40 (16.8, 16.0) | **15.80** (15.8, 15.8) | **-3.7%** | 16.32 (16.8, 15.8) | **16.00** (16.2, 15.8) | **-2.0%** |
| **Hot Cache** (100% hit) | 128 | 17.52 (17.8, 17.2) | **17.40** (17.5, 17.3) | **-0.7%** | 19.01 (20.4, 17.6) | **17.45** (17.6, 17.3) | **-8.2%** |

## Memory Observations

- Separating the 4-way set-associative cache arrays removes the 7 bytes of struct tail padding per set (`16,384 * 7 B = 112 KiB` saved) and compacts `ple_fragment_t` from 40 B to 32 B (`8 B` saved per fragment slot).
- Total user-space cache footprint for `65,536` rows (`16,384` sets) is `512 KiB` (`cache_tags`) + `16 KiB` (`cache_clocks`) + `16 MiB` (`cache_weights`) + `512 KiB` (`cache_scales`) = `17.02 MiB` (down from `17.13 MiB`), prefaulted at `ple_plan_init()` time so no minor page faults occur on the 64 KiB page kernel during cold-miss steps.

## Conclusion

Splitting and 64B-aligning the user-space Engram row cache arrays, inserting packed single-fragment misses directly from the `io_uring` slot buffer into the cache without bouncing through `miss_weights`/`miss_scales`, and deduplicating repeated in-step misses reduces `packed_cache64k` step latency by **3.3%–12.2% on cold misses**, **2.6%–4.4% on Zipfian working sets (`M >= 4`)**, and **2.0%–8.2% on hot cache steps** while remaining bit-exact.
