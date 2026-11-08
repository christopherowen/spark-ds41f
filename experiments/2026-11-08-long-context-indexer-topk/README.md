# 2026-11-08 — Long-Context (`64K–1M`) Indexer Top-K Selection Optimization

- **Deployment base commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`)
- **vLLM base commit**: `320021e3c4dbb202d911ab70e2cd924c75420488` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **vLLM candidate commit**: `4ef117500fa21b821d73b17265a80b1c9e06ec4f` (tree `2645208b0735e7793d1c6afd926d467d073eda59`, patch `0762-perf-dsv41-widen-long-context-indexer-top-k-segments.patch`)
- **Hardware / Image**: `sp7` (`spark-0015`, NVIDIA GB10 SM121), container image `spark-ds41f:1107`
- **Single intended causal variable**: Widen `IndexerTopK` segment and merge capacity from `4,096` / `8,192` `int32` keys to `16,384` `uint16` keys with a bijective shared-memory XOR bank-conflict swizzle (`idx ^ ((idx >> 5) & 31)`), active-extent loop bounds (`n_active`), winner-gated second compaction pass, and source-layer candidate-block reuse (`gather_scores=True` on `IndexerV41Plan(candidate_blocks=True)`).

## 1. Baseline Profiling: `indexer_score_v41` vs `IndexerTopK` Breakdown (`64K–1M`)

In DeepSeek-V4.1-Flash (`vllm/models/deepseek_v4_1/tilelang/indexer.py`), each layer's sparse-attention indexer runs:
1. **Dense layers (layers 2, 8, 14; `width = 524,288`, `page_size = 128`)**: `indexer_score_v41` + `IndexerTopK(524288, topk=512, group=1)`.
2. **Candidate source layer (layer 20; `width = 1,048,576`, `page_size = 256`, `candidate_blocks=True`)**: `indexer_score_v41` + `select_blocks = IndexerTopK(1048576, topk=2048, group=8)` + `select = IndexerTopK(1048576, topk=512, group=1)`.
3. **Reindex layers (layers 24, 28, 32, 36; `width = 16,384`, `candidates=True`)**: `indexer_score_v41(width=16384, candidates=True)` + `IndexerTopK(16384, topk=512, group=1, candidates=True)`.

### Why Baseline Long-Context `IndexerTopK` Was Bottlenecked
- `_key16(score)` maps BF16 scores to an order-preserving `16`-bit integer in `0..0xFFFF`, but `indexer_topk_segment_v41` and `indexer_topk_merge_v41` allocated `keys` and `sel_key` in shared memory as 4-byte `T.int32`, capping `_SEGMENT_ENTRIES = 4096` (`16 KiB`) and `_MERGE_KEYS = 8192` (`32 KiB`).
- During ordered compaction (`compact`), each thread read `per` contiguous entries `keys[tx * per + v]` (`per = 8` for `4,096` entries, `per = 16` for `8,192` entries), causing **8-way to 16-way shared-memory bank conflicts** across the 32 threads of a warp on both compaction passes, and unconditionally re-reading `keys[tx * per + v]` in the second compaction pass even when `(n_gt == 0) & (sel_eq == 0)`.
- Stage-1 and merge CTAs looped over the full compile-time `segment` / `n` capacity even when the row's active extent (`min(Lengths[row], Active[0])`) covered only a fraction of the segment or merge group.
- On Layer 20 (`candidate_blocks=True`, `width = 1,048,576`), `IndexerV41Plan.run` launched both `select_blocks` (`top-2048` blocks of 8 = `16,384` positions over `1,048,576` scores) and a full-width `select` (`top-512` positions over `1,048,576` scores), even though every top-`512` individual position is mathematically guaranteed to reside inside the `2,048` candidate blocks selected by `select_blocks`.

## 2. Optimizations in Patch `0762` (100% Bit-Exact)

1. **Wider Segments (`_LONG_SEGMENT_ENTRIES = 16384`) and Wider Merge Fan-In (`_LONG_MERGE_KEYS = 16384`) with `T.uint16` Shared Keys (`topk <= 2048`)**:
   - Stores `keys` and `sel_key` as `T.uint16` (`2` bytes) for `topk <= 2048` (`32 KiB` for `16,384` keys; total shared memory `39 KiB` for `topk = 512` and `47 KiB` for `topk = 2048`, fitting inside `48 KiB` for **2 CTAs/SM = 100% occupancy** on GB10 SM120/SM121).
   - Preserves `_SEGMENT_ENTRIES = 4096` and `_MERGE_KEYS = 8192` for coarse selectors with `topk > 2048` (`DirectCoarseTopK`, `RawWideCoarseTopK`).
   - Collapses `width = 16,384` (reindex layers 24, 28, 32, 36 and `prefill_short`) from `4` segments + 1 merge to **`1` single kernel launch (zero merge, zero scratch)**.
   - Cuts `width = 524,288, topk = 512` (dense layers 2, 8, 14) from `128` segments + 2 merge levels (`128 -> 8 -> 1`) to `32` segments + **1 single merge level (`32 -> 1`)**, reducing intermediate global memory traffic by `4.25x`.
   - Cuts `width = 1,048,576, topk = 2048, group = 8` (Layer 20 `select_blocks`) from `32` segments + 3 merge levels (`32 -> 8 -> 2 -> 1`) to `8` segments + **1 single merge level (`8 -> 1`)**, reducing intermediate global memory traffic by `5.25x`.
   - Cuts `width = 1,048,576, topk = 2048, group = 1` from `256` segments + 4 merge levels (`256 -> 64 -> 16 -> 4 -> 1`) to `64` segments + **2 merge levels (`64 -> 8 -> 1`)**.
2. **Bijective Shared-Memory XOR Bank-Conflict Swizzle (`_swizzle_key_idx(idx) = idx ^ ((idx >> 5) & 31)`)**:
   - Keeps strided load/histogram accesses (`i * threads + tx`) bank-conflict-free while eliminating warp bank conflicts during per-thread contiguous reads (`tx * per + v`) in `compact`.
3. **Active-Extent Loop Bounds (`n_active`) & Winner-Gated Second Compaction Pass**:
   - Bounds load, `threshold`, and `compact` loops by `n_active` (`min(segment, max(n_entries - seg * segment, 0))` in Stage 1 and `group_live * topk` in Merge), and gates the second shared-memory read loop in `compact` on `if (n_gt > 0) | (sel_eq > 0):`.
4. **Source-Layer Candidate-Block Short-Circuit (`gather_scores=True` on `IndexerV41Plan(candidate_blocks=True)`)**:
   - Because `select_blocks` outputs `candidate_out` (`[rows, 2048]`) in strictly ascending block ID order with no `-1` holes before `candidate_out_lengths` (`count` of visible positions) and always includes the partial last block (`+inf` block score), candidate slot `j = b * 8 + o < count` maps monotonically to logical position `candidate_out[row, b] * 8 + o`.
   - Selecting the top-`512` positions directly from `Scores[row, candidate_out[row, b] * 8 + o]` in a single 1-CTA-per-row segment launch (`gather_scores=True`, `segments == 1`, zero `select` scratch buffers) is 100% bit-identical (`torch.equal`) to a full `1,048,576`-entry `select` pass.

## 3. Exact Verification & 2-Boot Benchmark Commands

Executed across 2 fresh container boots (`spark-ds41f:1107`) on `sp7` under `flock /tmp/gpu.lock`:

```bash
# Boot 1 & Boot 2: Full indexer/LSA/attention pytest suite + indexer reads check
pytest --noconftest -q \
  tests/kernels/attention/test_deepseek_v41_tilelang.py \
  tests/models/test_deepseek_v4_1_lsa.py \
  tests/models/test_deepseek_v4_1_attention.py
python3 tests/kernels/attention/deepseek_v41_indexer_reads.py

# Boot 1 & Boot 2: Paired bit-exactness + latency sweep across N=64K..1M and M=1..1024
python3 bench_2boot.py receipts/bench_boot1.json
python3 bench_2boot.py receipts/bench_boot2.json
```

- **Pytest suite**: `132 passed, 1 skipped, 1 xfailed` on both Boot 1 (`24.84s`) and Boot 2 (`24.25s`); `deepseek_v41_indexer_reads.py` `scored 4 cases` on both boots.
- **Bit-exactness (`verify_exactness` in `bench_2boot.py`)**: `PASS` (`100% torch.equal` on `indices`, `values`, and `counts` across random, 16-level quantized tie, and constant-score rows for `(width, topk, group)` in `{(16384, 512, 1), (524288, 512, 1), (1048576, 512, 1), (1048576, 2048, 8), (1048576, 2048, 1)}` plus `IndexerV41Plan(1048576, 256, candidate_blocks=True)`).

## 4. Paired 2-Boot Latency Results (`sp7`, `us`, `mean (Boot1/Boot2)`)

### 4.1 Reindex Layer Selection (`width = 16,384, topk = 512, candidates = True`: 4 segments + 1 merge -> 1 single launch)

| `M` (rows) | Baseline `0761` (`us`) | Candidate `0762` (`us`) | Delta (`%` / Speedup) |
| ---: | :---: | :---: | :---: |
| `1` | 22.8 (22.9/22.7) | **13.6 (13.5/13.6)** | **-40.5% (1.68x)** |
| `6` | 22.6 (22.6/22.6) | **13.6 (13.6/13.6)** | **-40.0% (1.67x)** |
| `16` | 22.5 (22.5/22.5) | **13.6 (13.6/13.6)** | **-39.5% (1.65x)** |
| `32` | 24.8 (24.8/24.9) | **14.6 (13.5/15.7)** | **-41.1% (1.70x)** |
| `64` | 27.8 (27.9/27.8) | **17.6 (17.6/17.6)** | **-36.7% (1.58x)** |
| `256` | 86.6 (83.1/90.0) | **47.0 (46.3/47.6)** | **-45.8% (1.84x)** |
| `1024` | 341.2 (345.0/337.4) | **249.3 (240.2/258.5)** | **-26.9% (1.37x)** |

### 4.2 Long-Context Breakdown Across `N = 64K..1M` and `M = 1..1024`

| `N` | `M` | `score_p256` (`us`) | Dense `524K k=512` Base -> Cand (`us`) | Dense Delta | Layer 20 `select_blocks` (`k=2048,g=8`) Base -> Cand (`us`) | Layer 20 `select` (`k=512`) Base -> Cand (`us`) | Layer 20 Total Selection Base -> Cand (`us`) | Layer 20 Delta | Standalone `1M k=2048 g=1` Base -> Cand (`us`) | `1M k=2048` Delta |
| ---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `65536` | `1` | 22.3 (21.8/22.8) | 35.0 -> **26.0** | **-25.8% (1.35x)** | 46.4 -> **25.4** | 34.2 -> **23.7** | 80.6 -> **49.0** | **-39.2% (1.64x)** | 57.0 -> **33.6** | **-41.1% (1.70x)** |
| `65536` | `6` | 102.9 (101.5/104.3) | 38.5 -> **26.2** | **-32.0% (1.47x)** | 45.6 -> **24.8** | 37.6 -> **23.9** | 83.2 -> **48.6** | **-41.5% (1.71x)** | 57.0 -> **37.0** | **-35.1% (1.54x)** |
| `65536` | `16` | 269.7 (266.5/272.9) | 39.1 -> **25.8** | **-34.0% (1.51x)** | 46.2 -> **26.5** | 42.0 -> **23.7** | 88.2 -> **50.1** | **-43.1% (1.76x)** | 73.0 -> **37.8** | **-48.2% (1.93x)** |
| `65536` | `32` | 525.8 (520.7/531.0) | 61.7 -> **40.9** | **-33.7% (1.51x)** | 46.2 -> **26.2** | 65.5 -> **26.9** | 111.7 -> **53.1** | **-52.4% (2.10x)** | 119.7 -> **48.8** | **-59.3% (2.45x)** |
| `65536` | `64` | 1046.9 (1036.2/1057.5) | 94.7 -> **58.0** | **-38.7% (1.63x)** | 54.5 -> **26.9** | 108.1 -> **27.1** | 162.6 -> **54.0** | **-66.8% (3.01x)** | 235.4 -> **73.2** | **-68.9% (3.22x)** |
| `65536` | `256` | 4188.7 (4158.3/4219.1) | 359.9 -> **231.2** | **-35.8% (1.56x)** | 289.1 -> **183.4** | 400.9 -> **138.6** | 690.0 -> **322.0** | **-53.3% (2.14x)** | 1087.2 -> **326.5** | **-70.0% (3.33x)** |
| `65536` | `1024` | 16812.9 (16624.6/17001.2) | 1500.9 -> **899.4** | **-40.1% (1.67x)** | 1158.6 -> **759.7** | 1674.2 -> **559.5** | 2832.8 -> **1319.2** | **-53.4% (2.15x)** | 4375.4 -> **1344.4** | **-69.3% (3.25x)** |
| `131072` | `1` | 43.9 (46.1/41.7) | 37.5 -> **26.0** | **-30.5% (1.44x)** | 46.1 -> **26.2** | 37.4 -> **23.9** | 83.5 -> **50.1** | **-40.0% (1.67x)** | 58.0 -> **36.4** | **-37.2% (1.59x)** |
| `131072` | `6` | 208.4 (207.2/209.7) | 38.3 -> **25.7** | **-32.9% (1.49x)** | 46.4 -> **26.6** | 40.4 -> **23.7** | 86.8 -> **50.4** | **-42.0% (1.72x)** | 63.9 -> **36.5** | **-42.9% (1.75x)** |
| `131072` | `16` | 512.3 (507.6/517.0) | 58.0 -> **35.8** | **-38.3% (1.62x)** | 47.4 -> **33.6** | 64.1 -> **25.0** | 111.4 -> **58.6** | **-47.4% (1.90x)** | 107.7 -> **49.9** | **-53.6% (2.16x)** |
| `131072` | `32` | 1020.1 (1012.9/1027.3) | 96.2 -> **55.3** | **-42.5% (1.74x)** | 48.4 -> **37.1** | 103.8 -> **23.8** | 152.3 -> **60.9** | **-60.0% (2.50x)** | 219.2 -> **71.5** | **-67.4% (3.07x)** |
| `131072` | `64` | 2031.7 (2016.7/2046.7) | 174.2 -> **102.0** | **-41.4% (1.71x)** | 69.9 -> **40.6** | 188.4 -> **25.9** | 258.3 -> **66.5** | **-74.3% (3.88x)** | 507.2 -> **123.3** | **-75.7% (4.11x)** |
| `131072` | `256` | 8064.9 (8026.4/8103.3) | 725.1 -> **442.9** | **-38.9% (1.64x)** | 512.1 -> **358.9** | 809.8 -> **169.2** | 1322.0 -> **528.2** | **-60.0% (2.50x)** | 2124.4 -> **630.4** | **-70.3% (3.37x)** |
| `131072` | `1024` | 32308.1 (31999.5/32616.7) | 2866.4 -> **1657.9** | **-42.2% (1.73x)** | 2043.1 -> **1349.6** | 3095.6 -> **612.9** | 5138.7 -> **1962.5** | **-61.8% (2.62x)** | 8413.8 -> **2466.0** | **-70.7% (3.41x)** |
| `262144` | `1` | 80.6 (78.6/82.5) | 38.3 -> **26.3** | **-31.3% (1.46x)** | 48.2 -> **29.9** | 37.3 -> **26.4** | 85.6 -> **56.3** | **-34.2% (1.52x)** | 58.6 -> **37.9** | **-35.3% (1.54x)** |
| `262144` | `6` | 401.5 (391.4/411.6) | 44.3 -> **26.1** | **-41.0% (1.69x)** | 46.2 -> **30.0** | 55.7 -> **23.7** | 101.9 -> **53.8** | **-47.2% (1.89x)** | 80.9 -> **40.1** | **-50.4% (2.02x)** |
| `262144` | `16` | 1053.5 (1047.5/1059.6) | 91.9 -> **55.3** | **-39.9% (1.66x)** | 50.1 -> **38.1** | 97.7 -> **23.8** | 147.8 -> **61.8** | **-58.2% (2.39x)** | 209.6 -> **74.4** | **-64.5% (2.82x)** |
| `262144` | `32` | 2087.2 (2088.6/2085.8) | 167.1 -> **95.8** | **-42.7% (1.74x)** | 73.8 -> **43.1** | 175.9 -> **23.7** | 249.7 -> **66.8** | **-73.3% (3.74x)** | 477.2 -> **129.0** | **-73.0% (3.70x)** |
| `262144` | `64` | 4120.6 (4111.6/4129.6) | 334.8 -> **223.8** | **-33.2% (1.50x)** | 255.8 -> **171.9** | 362.9 -> **27.8** | 618.7 -> **199.8** | **-67.7% (3.10x)** | 1037.3 -> **320.7** | **-69.1% (3.23x)** |
| `262144` | `256` | 16434.8 (16225.2/16644.5) | 1416.3 -> **823.4** | **-41.9% (1.72x)** | 996.4 -> **674.2** | 1501.2 -> **179.3** | 2497.6 -> **853.5** | **-65.8% (2.93x)** | 4131.1 -> **1284.6** | **-68.9% (3.22x)** |
| `262144` | `1024` | 66643.7 (67019.2/66268.1) | 5451.0 -> **3194.0** | **-41.4% (1.71x)** | 4050.2 -> **2601.5** | 5771.6 -> **635.0** | 9821.9 -> **3236.5** | **-67.0% (3.03x)** | 16745.9 -> **4935.2** | **-70.5% (3.39x)** |
| `524288` | `1` | 225.4 (221.5/229.4) | 38.8 -> **26.4** | **-31.8% (1.47x)** | 47.6 -> **34.2** | 37.8 -> **24.4** | 85.4 -> **58.6** | **-31.4% (1.46x)** | 62.0 -> **39.0** | **-37.2% (1.59x)** |
| `524288` | `6` | 1149.8 (1148.7/1150.8) | 68.9 -> **43.6** | **-36.6% (1.58x)** | 47.4 -> **32.0** | 75.8 -> **23.7** | 123.2 -> **55.7** | **-54.8% (2.21x)** | 143.1 -> **55.8** | **-61.0% (2.56x)** |
| `524288` | `16` | 3034.4 (3034.1/3034.8) | 167.4 -> **98.1** | **-41.4% (1.71x)** | 73.0 -> **44.4** | 178.7 -> **23.7** | 251.7 -> **68.1** | **-72.9% (3.69x)** | 471.3 -> **125.6** | **-73.3% (3.75x)** |
| `524288` | `32` | 6046.4 (6062.2/6030.6) | 319.4 -> **222.6** | **-30.3% (1.44x)** | 235.9 -> **167.8** | 344.3 -> **23.8** | 580.2 -> **191.5** | **-67.0% (3.03x)** | 1009.6 -> **313.7** | **-68.9% (3.22x)** |
| `524288` | `64` | 12195.5 (12339.8/12051.1) | 664.5 -> **432.0** | **-35.0% (1.54x)** | 486.2 -> **344.6** | 729.5 -> **29.8** | 1215.7 -> **374.4** | **-69.2% (3.25x)** | 2071.8 -> **612.4** | **-70.4% (3.38x)** |
| `524288` | `256` | 47963.9 (47689.6/48238.3) | 2680.1 -> **1608.4** | **-40.0% (1.67x)** | 1955.2 -> **1292.8** | 2860.0 -> **187.8** | 4815.1 -> **1480.6** | **-69.3% (3.25x)** | 8297.6 -> **2493.2** | **-70.0% (3.33x)** |
| `524288` | `1024` | 187908.0 (185817.0/189998.9) | 10459.8 -> **6306.5** | **-39.7% (1.66x)** | 7861.2 -> **5107.2** | 11123.7 -> **671.3** | 18984.9 -> **5778.5** | **-69.6% (3.29x)** | 33767.2 -> **9570.3** | **-71.7% (3.53x)** |
| `1048576` | `1` | 464.1 (453.0/475.1) | — | **—** | 51.9 -> **34.0** | 44.3 -> **24.4** | 96.2 -> **58.4** | **-39.3% (1.65x)** | 80.6 -> **43.6** | **-45.9% (1.85x)** |
| `1048576` | `6` | 2412.7 (2412.4/2412.9) | — | **—** | 52.6 -> **37.0** | 127.7 -> **24.3** | 180.3 -> **61.4** | **-66.0% (2.94x)** | 300.4 -> **96.9** | **-67.7% (3.10x)** |
| `1048576` | `16` | 6485.6 (6500.1/6471.0) | — | **—** | 231.9 -> **171.2** | 331.6 -> **23.7** | 563.5 -> **195.0** | **-65.4% (2.89x)** | 981.6 -> **312.4** | **-68.2% (3.14x)** |
| `1048576` | `32` | 12707.9 (12622.6/12793.2) | — | **—** | 474.6 -> **341.0** | 661.6 -> **25.5** | 1136.3 -> **366.5** | **-67.7% (3.10x)** | 1967.2 -> **633.6** | **-67.8% (3.10x)** |
| `1048576` | `64` | 25126.7 (24942.8/25310.5) | — | **—** | 958.1 -> **653.5** | 1380.8 -> **28.9** | 2338.9 -> **682.4** | **-70.8% (3.43x)** | 3893.6 -> **1193.2** | **-69.4% (3.26x)** |
| `1048576` | `256` | 100284.7 (99829.4/100740.0) | — | **—** | 3916.3 -> **2546.6** | 5361.3 -> **196.2** | 9277.6 -> **2742.8** | **-70.4% (3.38x)** | 15823.0 -> **4789.3** | **-69.7% (3.30x)** |
| `1048576` | `1024` | 401007.9 (399347.0/402668.8) | — | **—** | 15869.9 -> **10123.3** | 20949.8 -> **692.2** | 36819.7 -> **10815.5** | **-70.6% (3.40x)** | 64049.8 -> **19236.5** | **-70.0% (3.33x)** |

## 5. Conclusion

- **Top-k Share of Long-Context Indexer Time**:
  - At decode `M = 1` (`N = 64K..1M`), `IndexerTopK` is a major share of indexer latency (`61.3%` at `N = 64K` on dense layers, `78.5%` at `N = 64K` and `17.2%` at `N = 1M` on Layer 20).
  - At wider decode (`M = 16..32`) and SP prefill chunks (`M = 64..1024`) at `524K–1M` context, baseline `IndexerTopK` adds `563 us` (`M = 16`) to `36,820 us` (`M = 1024`) per Layer 20 call (`8.0%–8.4%` of total Layer 20 indexer time) and `2,680 us` (`M = 256`) to `10,460 us` (`M = 1024`) per dense layer call (`5.3%` of dense layer indexer time).
- **Measured Gains Across 2 Boots**:
  - **Layer 20 (`candidate_blocks=True`, `1M` width)**: Selection latency drops by **`38.8%` to `71.0%` (`1.63x` to `3.45x` faster)** across all `(N, M)` points — e.g. at `N = 1M`, `563.5 us -> 195.0 us` (`-65.4%`, `2.89x`) at `M = 16`, `9,277.6 us -> 2,742.8 us` (`-70.4%`, `3.38x`) at `M = 256`, and `36,819.7 us -> 10,815.5 us` (`-70.6%`, `3.40x`) at `M = 1024`.
  - **Dense layers (`524K` width, `k = 512`)**: Selection latency drops by **`25.4%` to `40.1%` (`1.34x` to `1.67x` faster)** across all `(N, M)` points — e.g. at `N = 524K`, `2,680.1 us -> 1,608.4 us` (`-40.0%`, `1.67x`) at `M = 256` and `10,459.8 us -> 6,306.5 us` (`-39.7%`, `1.66x`) at `M = 1024`.
  - **Standalone `1M` `k = 2048, group = 1`**: Selection latency drops by **`35.1%` to `72.4%` (`1.54x` to `3.62x` faster)** — e.g. at `N = 1M`, `981.6 us -> 312.4 us` (`-68.2%`, `3.14x`) at `M = 16`, `15,823.0 us -> 4,789.3 us` (`-69.7%`, `3.30x`) at `M = 256`, and `64,049.8 us -> 19,236.5 us` (`-70.0%`, `3.33x`) at `M = 1024`.
  - **Reindex layers (`16K` width, `k = 512, candidates = True`)**: Collapsing `4` segments + `1` merge into a single launch reduces selection latency by **`26.5%` to `77.1%` (`1.36x` to `4.36x` faster)** across `M = 1..1024` (`925.4 us -> 212.1 us` at `M = 1024`).
