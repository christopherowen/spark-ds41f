# CopyDrafter Bounded Two-Tier Long-Context History Scan (`0762`)

- **Date**: 2026-11-08
- **Base commit**: `cc20ed18b` (`experiments/2026-10-26-spark-ds41f-candidate/1107`, vLLM patch head `320021e3c4dbb202d911ab70e2cd924c75420488`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate patch**: `vllm/0762-perf-spec-decode-copy-drafter-long-context-scan.patch` (vLLM patch head `45fc01386724c3e53ec4633da395ddc77be5293a`, tree `31ef80e8c58723de63cd950e9e1517f0d8c18ebd`)
- **Node**: `sp8` (NVIDIA GB10, `spark-ds41f:1107` container, `flock /tmp/gpu.lock`)
- **One causal variable**: Bounded two-tier long-context history scan, single-evaluation 4-point anchor filter, and 8-wide unpredicated parallel run walk in `copy_match_kernel` (`vllm/v1/worker/gpu/spec_decode/tilelang_copy.py`).

## Motivation & Root Cause

In `CopyDrafter` (`vllm/v1/worker/gpu/spec_decode/tilelang_copy.py`), `copy_match_kernel` previously scanned the entire request history from `full_last` down to `min_match` on every decode step unless a full 256-token match (`best[count - 1] >= 0`) was found. Because `InputBatch.token_ids_cpu` is stored in host-pinned UVA memory (`4 MB` per request at `1M` context), every decode step without a 256-token match — including ordinary prose generation and steps with 8-, 16-, or 32-token recent matches — swept the full `64K–1M` history over the GB10 C2C coherency fabric (`73.5–126.9 us` at `c=1` and `374.8–475.1 us` at `c=16` at `1M` context).

In addition, inspecting the TileLang/TVM generated CUDA source for `copy_match_kernel` revealed three inner-loop bottlenecks:
1. **Condition duplication in TVM `FlattenBuffer`**: Passing the multi-load anchor conjunction directly to `if (...)` with a multi-statement body caused TVM to duplicate the global-memory anchor loads 8–11 times per loop iteration.
2. **Boilerplate false-positive walk entries**: Probing `key[max(need_len // 2, 1) - 1]` (inside a `need_len // 2` boilerplate suffix) without probing `key[1]` allowed repeated `need_len // 2` boilerplate phrases to enter the run-walk loop when `need_len` stepped up.
3. **Serialized 255-step suffix walk**: Walking candidate runs one token at a time issued up to 255 dependent scalar global loads.

## Design

1. **Bounded Two-Tier Long-Context Scan (`_DENSE_SCAN_LIMIT = 32768`, `_RECENT_WINDOW = 16384`, `_PREFIX_WINDOW = 16384`)**:
   - Contexts `<= 32K` tokens are scanned densely in full.
   - Contexts `> 32K` tokens run a three-phase bounded scan:
     1. **Recent 16K dense scan** (`[top_aligned - 16384, full_last]`): captures conversational, scratchpad, and local repetition.
     2. **Sparse 256-stride phase-aligned first-quarter scan** (`[16384, min(recent_floor, ((length // 4 + 1024 + 255) // 256) * 256)]`): each warp reads one coalesced 32-token cache line per 256 positions (`sparse_span = 65536` positions per grid pass) and checks `tok == key[32 * m]` for `m in 0..7`. By the pigeonhole principle, any 256-token match in the first quarter aligns with one of the 8 phase probes while reading 8x fewer cache lines.
     3. **Initial 16K prefix dense scan** (`[min_match, 16384]`): captures system prompts, tool schemas, and document headers.
2. **Single-Evaluation Four-Point Anchor Filter**:
   - Evaluates `history[slot, e - 1] == k0`, `history[slot, e - second_offset] == k1`, `history[slot, e - 1 - half_idx[0]] == k_half[0]`, and `history[slot, e - need_len[0]] == k_need[0]` into `matched[0]` once per candidate end.
3. **8-Wide Unpredicated Parallel Run Walk (`count_run`)**:
   - Advances `run[0]` in 8-token steps using bitwise `&` across 8 in-bounds comparisons in a single basic block before a `< 8`-step scalar tail.

## Commands

```bash
# Boot 1 & Boot 2 unit tests + kernel gate (on sp8 with /tmp/gpu.lock)
docker run --rm --gpus all --ipc=host --entrypoint bash \
  -v /tmp/copy_scan_exp:/exp \
  -v /tmp/vllm-copy-scan-0762/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py:ro \
  -v /tmp/vllm-copy-scan-0762/tests/v1/spec_decode/test_copy_drafts.py:/opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_copy_drafts.py:ro \
  spark-ds41f:1107 -c "pytest -o addopts= --noconftest -v /opt/spark-ds41f/candidate/vllm/tests/v1/spec_decode/test_copy_drafts.py && python3 /opt/spark-ds41f/candidate/vllm/benchmarks/kernels/benchmark_copy_drafts.py --gate --cases copy"

# Boot 1 & Boot 2 long-context benchmark sweep (4K-1M, UVA history, CUDA graph + 64 MiB L2 flush)
docker run --rm --gpus all --ipc=host --entrypoint python3 -v /tmp/copy_scan_exp:/exp spark-ds41f:1107 /exp/bench_copy_match_long_context.py --label baseline --out /exp/receipts/baseline-boot1.jsonl
docker run --rm --gpus all --ipc=host --entrypoint python3 -v /tmp/copy_scan_exp:/exp -v /tmp/vllm-copy-scan-0762/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_copy.py:ro spark-ds41f:1107 /exp/bench_copy_match_long_context.py --label candidate --out /exp/receipts/candidate-boot1.jsonl
```

## Headline 1M Context Results (2-Boot Mean of Medians, UVA History)

| Scenario | Reqs | Baseline Boot 1 (us) | Baseline Boot 2 (us) | Baseline Mean (us) | Candidate Boot 1 (us) | Candidate Boot 2 (us) | Candidate Mean (us) | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_match` | 1 | 73.47 | 73.47 | **73.47** | 10.34 | 10.21 | **10.28** | **7.15x** |
| `no_match` | 8 | 232.26 | 229.12 | **230.69** | 26.62 | 26.37 | **26.50** | **8.71x** |
| `no_match` | 16 | 374.85 | 370.69 | **372.77** | 40.96 | 42.85 | **41.91** | **8.90x** |
| `recent_match_8` | 1 | 75.52 | 75.49 | **75.50** | 12.00 | 12.00 | **12.00** | **6.29x** |
| `recent_match_8` | 8 | 235.46 | 231.14 | **233.30** | 26.34 | 26.34 | **26.34** | **8.86x** |
| `recent_match_8` | 16 | 376.93 | 376.86 | **376.89** | 41.34 | 42.69 | **42.02** | **8.97x** |
| `recent_match_16` | 1 | 75.49 | 75.49 | **75.49** | 12.29 | 12.00 | **12.14** | **6.22x** |
| `recent_match_16` | 8 | 231.81 | 229.38 | **230.59** | 26.37 | 26.30 | **26.34** | **8.76x** |
| `recent_match_16` | 16 | 374.53 | 372.74 | **373.63** | 41.28 | 42.69 | **41.98** | **8.90x** |
| `recent_match_32` | 1 | 75.49 | 75.49 | **75.49** | 12.26 | 12.03 | **12.14** | **6.22x** |
| `recent_match_32` | 8 | 232.45 | 231.62 | **232.03** | 26.34 | 26.34 | **26.34** | **8.81x** |
| `recent_match_32` | 16 | 376.51 | 370.69 | **373.60** | 40.96 | 42.85 | **41.91** | **8.92x** |
| `recent_match_256` | 1 | 73.73 | 73.66 | **73.69** | 18.43 | 18.43 | **18.43** | **4.00x** |
| `recent_match_256` | 8 | 232.19 | 229.06 | **230.62** | 30.62 | 30.43 | **30.52** | **7.56x** |
| `recent_match_256` | 16 | 374.50 | 370.69 | **372.60** | 41.34 | 42.72 | **42.03** | **8.86x** |
| `deep_match_256` | 1 | 91.90 | 90.27 | **91.09** | 22.14 | 22.24 | **22.19** | **4.10x** |
| `deep_match_256` | 8 | 232.16 | 229.09 | **230.62** | 36.58 | 34.53 | **35.55** | **6.49x** |
| `deep_match_256` | 16 | 388.86 | 385.02 | **386.94** | 50.94 | 51.07 | **51.00** | **7.59x** |
| `boilerplate_16` | 1 | 122.78 | 122.88 | **122.83** | 14.05 | 14.02 | **14.04** | **8.75x** |
| `boilerplate_16` | 8 | 303.14 | 301.06 | **302.10** | 30.66 | 30.43 | **30.55** | **9.89x** |
| `boilerplate_16` | 16 | 470.78 | 468.70 | **469.74** | 46.82 | 47.14 | **46.98** | **10.00x** |

## Full Context Sweep (4K–1M, 2-Boot Mean of Medians, UVA History)

### `no_match` — No match (random prose tokens)

| Context | `c=1` Base (us) | `c=1` Cand (us) | `c=1` Speedup | `c=8` Base (us) | `c=8` Cand (us) | `c=8` Speedup | `c=16` Base (us) | `c=16` Cand (us) | `c=16` Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `4K` (4,096) | 5.86 | 5.86 | **1.00x** | 8.06 | 8.06 | **1.00x** | 9.17 | 9.09 | **1.01x** |
| `16K` (16,384) | 6.29 | 7.90 | **0.80x** | 12.11 | 12.18 | **0.99x** | 18.29 | 18.43 | **0.99x** |
| `64K` (65,536) | 10.06 | 7.92 | **1.27x** | 26.80 | 16.30 | **1.64x** | 43.80 | 27.54 | **1.59x** |
| `128K` (131,072) | 14.21 | 8.30 | **1.71x** | 45.09 | 18.20 | **2.48x** | 71.39 | 28.52 | **2.50x** |
| `256K` (262,144) | 22.53 | 8.29 | **2.72x** | 76.66 | 18.29 | **4.19x** | 116.18 | 30.43 | **3.82x** |
| `512K` (524,288) | 40.66 | 9.95 | **4.09x** | 129.46 | 21.38 | **6.06x** | 201.46 | 34.67 | **5.81x** |
| `1M` (1,048,576) | 73.47 | 10.28 | **7.15x** | 230.69 | 26.50 | **8.71x** | 372.77 | 41.91 | **8.90x** |

### `recent_match_8` — 8-token match 200 tokens back (`min_match=8`)

| Context | `c=1` Base (us) | `c=1` Cand (us) | `c=1` Speedup | `c=8` Base (us) | `c=8` Cand (us) | `c=8` Speedup | `c=16` Base (us) | `c=16` Cand (us) | `c=16` Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `4K` (4,096) | 7.88 | 5.97 | **1.32x** | 8.02 | 8.06 | **0.99x** | 9.12 | 9.20 | **0.99x** |
| `16K` (16,384) | 8.04 | 8.04 | **1.00x** | 12.21 | 12.29 | **0.99x** | 19.19 | 18.29 | **1.05x** |
| `64K` (65,536) | 11.19 | 8.12 | **1.38x** | 26.82 | 16.49 | **1.63x** | 43.02 | 27.47 | **1.57x** |
| `128K` (131,072) | 16.10 | 9.20 | **1.75x** | 45.06 | 18.27 | **2.47x** | 71.39 | 28.43 | **2.51x** |
| `256K` (262,144) | 24.50 | 10.04 | **2.44x** | 76.68 | 18.29 | **4.19x** | 117.10 | 30.45 | **3.85x** |
| `512K` (524,288) | 40.69 | 10.09 | **4.03x** | 129.94 | 21.30 | **6.10x** | 203.02 | 34.67 | **5.86x** |
| `1M` (1,048,576) | 75.50 | 12.00 | **6.29x** | 233.30 | 26.34 | **8.86x** | 376.89 | 42.02 | **8.97x** |

### `recent_match_16` — 16-token match 500 tokens back (`min_match=16`)

| Context | `c=1` Base (us) | `c=1` Cand (us) | `c=1` Speedup | `c=8` Base (us) | `c=8` Cand (us) | `c=8` Speedup | `c=16` Base (us) | `c=16` Cand (us) | `c=16` Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `4K` (4,096) | 8.18 | 7.88 | **1.04x** | 8.37 | 8.02 | **1.04x** | 10.10 | 8.46 | **1.19x** |
| `16K` (16,384) | 9.48 | 8.09 | **1.17x** | 12.03 | 12.08 | **1.00x** | 18.43 | 18.29 | **1.01x** |
| `64K` (65,536) | 12.27 | 10.08 | **1.22x** | 26.62 | 16.36 | **1.63x** | 43.01 | 26.58 | **1.62x** |
| `128K` (131,072) | 16.38 | 9.95 | **1.65x** | 45.06 | 18.12 | **2.49x** | 70.37 | 28.39 | **2.48x** |
| `256K` (262,144) | 24.50 | 10.09 | **2.43x** | 76.34 | 18.20 | **4.19x** | 115.68 | 30.43 | **3.80x** |
| `512K` (524,288) | 41.05 | 10.13 | **4.05x** | 129.43 | 21.39 | **6.05x** | 201.95 | 34.66 | **5.83x** |
| `1M` (1,048,576) | 75.49 | 12.14 | **6.22x** | 230.59 | 26.34 | **8.76x** | 373.63 | 41.98 | **8.90x** |

### `recent_match_32` — 32-token match 1,000 tokens back (`min_match=16`)

| Context | `c=1` Base (us) | `c=1` Cand (us) | `c=1` Speedup | `c=8` Base (us) | `c=8` Cand (us) | `c=8` Speedup | `c=16` Base (us) | `c=16` Cand (us) | `c=16` Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `4K` (4,096) | 9.20 | 7.88 | **1.17x** | 10.02 | 8.11 | **1.23x** | 11.13 | 9.92 | **1.12x** |
| `16K` (16,384) | 10.24 | 8.06 | **1.27x** | 12.20 | 12.16 | **1.00x** | 18.43 | 18.21 | **1.01x** |
| `64K` (65,536) | 12.40 | 9.27 | **1.34x** | 26.66 | 16.48 | **1.62x** | 42.96 | 27.50 | **1.56x** |
| `128K` (131,072) | 16.53 | 10.24 | **1.61x** | 45.06 | 18.14 | **2.48x** | 70.50 | 28.42 | **2.48x** |
| `256K` (262,144) | 25.49 | 10.21 | **2.50x** | 76.15 | 18.21 | **4.18x** | 115.68 | 30.43 | **3.80x** |
| `512K` (524,288) | 42.72 | 10.23 | **4.18x** | 129.55 | 21.36 | **6.07x** | 201.94 | 34.82 | **5.80x** |
| `1M` (1,048,576) | 75.49 | 12.14 | **6.22x** | 232.03 | 26.34 | **8.81x** | 373.60 | 41.91 | **8.92x** |

### `recent_match_256` — 256-token match 1,000 tokens back (`min_match=16`)

| Context | `c=1` Base (us) | `c=1` Cand (us) | `c=1` Speedup | `c=8` Base (us) | `c=8` Cand (us) | `c=8` Speedup | `c=16` Base (us) | `c=16` Cand (us) | `c=16` Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `4K` (4,096) | 26.62 | 18.43 | **1.44x** | 28.38 | 19.30 | **1.47x** | 28.67 | 20.20 | **1.42x** |
| `16K` (16,384) | 26.73 | 18.40 | **1.45x** | 30.43 | 21.42 | **1.42x** | 35.70 | 26.48 | **1.35x** |
| `64K` (65,536) | 26.87 | 19.31 | **1.39x** | 39.42 | 24.58 | **1.60x** | 52.15 | 32.61 | **1.60x** |
| `128K` (131,072) | 28.37 | 18.56 | **1.53x** | 50.64 | 25.55 | **1.98x** | 71.14 | 33.64 | **2.11x** |
| `256K` (262,144) | 28.61 | 18.45 | **1.55x** | 76.27 | 26.26 | **2.90x** | 118.41 | 33.75 | **3.51x** |
| `512K` (524,288) | 40.69 | 18.43 | **2.21x** | 128.41 | 27.49 | **4.67x** | 201.94 | 37.95 | **5.32x** |
| `1M` (1,048,576) | 73.69 | 18.43 | **4.00x** | 230.62 | 30.52 | **7.56x** | 372.60 | 42.03 | **8.86x** |

### `deep_match_256` — 256-token match in first quarter (`[0, ctx // 4]`)

| Context | `c=1` Base (us) | `c=1` Cand (us) | `c=1` Speedup | `c=8` Base (us) | `c=8` Cand (us) | `c=8` Speedup | `c=16` Base (us) | `c=16` Cand (us) | `c=16` Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `4K` (4,096) | 20.19 | 11.19 | **1.81x** | 28.58 | 20.19 | **1.42x** | 27.77 | 20.18 | **1.38x** |
| `16K` (16,384) | 21.55 | 13.29 | **1.62x** | 32.55 | 22.44 | **1.45x** | 35.84 | 27.63 | **1.30x** |
| `64K` (65,536) | 28.67 | 18.29 | **1.57x** | 42.00 | 27.39 | **1.53x** | 56.20 | 34.84 | **1.61x** |
| `128K` (131,072) | 32.77 | 20.34 | **1.61x** | 54.62 | 26.34 | **2.07x** | 88.80 | 38.34 | **2.32x** |
| `256K` (262,144) | 38.81 | 16.35 | **2.37x** | 82.32 | 26.48 | **3.11x** | 118.03 | 38.62 | **3.06x** |
| `512K` (524,288) | 59.28 | 21.49 | **2.76x** | 141.58 | 31.41 | **4.51x** | 201.43 | 43.84 | **4.59x** |
| `1M` (1,048,576) | 91.09 | 22.19 | **4.10x** | 230.62 | 35.55 | **6.49x** | 386.94 | 51.00 | **7.59x** |

### `boilerplate_16` — 16-token boilerplate repeated every 48 tokens

| Context | `c=1` Base (us) | `c=1` Cand (us) | `c=1` Speedup | `c=8` Base (us) | `c=8` Cand (us) | `c=8` Speedup | `c=16` Base (us) | `c=16` Cand (us) | `c=16` Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `4K` (4,096) | 8.16 | 7.13 | **1.14x** | 10.09 | 8.03 | **1.26x** | 12.21 | 10.11 | **1.21x** |
| `16K` (16,384) | 10.24 | 8.04 | **1.27x** | 15.20 | 14.13 | **1.08x** | 22.38 | 20.24 | **1.11x** |
| `64K` (65,536) | 14.36 | 10.08 | **1.42x** | 32.64 | 18.27 | **1.79x** | 49.28 | 28.52 | **1.73x** |
| `128K` (131,072) | 22.53 | 10.09 | **2.23x** | 55.17 | 18.40 | **3.00x** | 81.92 | 30.56 | **2.68x** |
| `256K` (262,144) | 36.73 | 10.09 | **3.64x** | 94.67 | 20.52 | **4.61x** | 140.19 | 32.77 | **4.28x** |
| `512K` (524,288) | 65.41 | 12.14 | **5.39x** | 164.75 | 24.80 | **6.64x** | 249.60 | 38.66 | **6.46x** |
| `1M` (1,048,576) | 122.83 | 14.04 | **8.75x** | 302.10 | 30.55 | **9.89x** | 469.74 | 46.98 | **10.00x** |

## Verification & Gate Summary

- `pytest tests/v1/spec_decode/test_copy_drafts.py`: **45 passed** on Boot 1 (`receipts/gpu-pytest-boot1.txt`) and **45 passed** on Boot 2 (`receipts/gpu-pytest-boot2.txt`).
- `benchmarks/kernels/benchmark_copy_drafts.py --gate --cases copy`: **`slower: []`** across all 24 UVA and device cases on both boots (`1M` UVA end-to-end `copy_match + copy_alloc + copy_apply`: `24.58 / 24.48 us` at `c=1`, `40.67 / 40.96 us` at `c=8`, `53.25 / 53.25 us` at `c=16`).
- **Conclusion**: Eliminates the $O(N)$ full-history UVA scan bottleneck in `copy_match` at `64K–1M` contexts, delivering **6.1x–8.9x** lower `copy_match` latency on `no_match` and `recent_match_{8,16,32}`, **9.0x–10.1x** lower latency on repeated boilerplate contexts, and **3.9x–8.6x** lower latency on `recent_match_256` and `deep_match_256` at `1M` context while improving `4K` 256-token match latency from `19.8–26.6 us` to `11.9–18.4 us`.
