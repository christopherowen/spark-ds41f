# 64 KiB Chunked, `%globaltimer`-Paced Side-Stream L2 Prefetch (`2026-11-08-l2-prefetch-pacing`)

## Summary & Goal

In `spark-ds41f-1107` (`vLLM 0044`, `0048`, `0487`, `0572`, `0752`), each DeepSeek V4.1 decoder layer issues three side-stream `cp.async.bulk.prefetch.L2.global.L2::cache_hint` (`evict_last`) plans during communication and attention windows when `M <= 32` (`VLLM_L2_PREFETCH_MAX_TOKENS = 32`):

1. **`WO` (`16 MiB` budget)**: prefetches `o_proj.b` (`43.75 MiB` total on TP3; first `16 MiB` of MXFP4 weight + scale words) during attention (`15-95 us` window).
2. **`FFN` (`16 MiB` budget)**: prefetches `mhc.hc_ffn_*`, `ffn_norm.weight`, `gate_up_proj` (`3.5 MiB`), and `down_proj` (`1.75 MiB`) (~`5.28 MiB` total) during the post-attention `all_reduce` (`60-120 us` window).
3. **`NEXT` (`18 MiB` budget)**: prefetches the next layer's `mhc.hc_attn_*`, `attn_norm.weight`, `wqa_wkv` (`9.28 MiB`), and `wq_b` (`8.72 MiB` slice) during the post-MoE `all_reduce` (`70-130 us` window).
4. **DSpark speculative drafter (`4 MiB` budget)**: prefetches the Markov `W2` projection (`15 MiB` 12-bit packed BF16) during the drafter's `all_reduce`.

Both the active TileLang backend (`vllm/models/glm5next/nvidia/tilelang/l2_prefetch.py`, selected when `VLLM_TILELANG_ENABLED=1`) and the CuTe DSL backend (`vllm/models/glm5next/nvidia/l2_prefetch.py`) launched `grid * 128` threads (`256` threads at `VLLM_L2_PREFETCH_GRID=2` on TP4; `512` threads at `VLLM_L2_PREFETCH_GRID=4` on TP3) where every thread issued unpaced `4 KiB` (`4096`-byte) `cp.async.bulk.prefetch.L2` requests in a tight loop (`4,096` bulk-prefetch requests for a `16 MiB` plan).

This experiment evaluates three hardware questions on GB10 (`sm_121a`, 273 GB/s LPDDR5X, 48 MiB L2 with 32 MiB persisting set-aside, `sp8`):

1. **`cp.async.bulk.prefetch.L2` chunk size & TMA FIFO saturation**: What chunk sizes does GB10 honor for `cp.async.bulk.prefetch.L2.global.L2::cache_hint`, and what happens when `256`–`512` threads issue `4,096` back-to-back `4 KiB` bulk-prefetch requests?
2. **`%globaltimer`-paced single-thread issue vs. unpaced multi-CTA flood**: Does pacing `64 KiB` bulk-prefetch chunks from a single issuer thread (`CTA 0, lane 0`) after an initial `128 KiB` (`2`-chunk) burst improve effective L2 hit rates and reduce memory-controller contention during concurrent decode window kernels?
3. **Prefetch placement & SWA vs. Indexer `WO` budget gating**: Are all prefetch call sites in `model.py` and `dspark/speculator.py` properly gated on skipped paths, and does reducing `BUDGET_WO` on faster SWA layers (`10 MiB` vs `16 MiB`) help or hurt?

---

## Base Commit & Hardware Environment

- **Base commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`, image `spark-ds41f:1107`, vLLM `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` + `0001..0761`, `patch_head` `320021e3c4dbb202d911ab70e2cd924c75420488`, `expected_tree` `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`).
- **Candidate patch**: `experiments/2026-11-08-l2-prefetch-pacing/vllm/0762-perf-dsv41-64k-chunked-globaltimer-paced-l2-prefetch.patch` (`patch_head` `e8b8e44f37f305d71ec493ecbc0430c87a8a79a6`, `expected_tree` `15b3c956d5f575722650c7233a4562b9eca45b16`, `642` patches).
- **Hardware node**: `sp8` (`NVIDIA GB10`, `sm_121a`, CUDA `13.2`, TileLang `0.1.15+snapshot.35f15411cf3af0fc06a1cb0097215b1372a40464`).

---

## Hardware Diagnosis & Audit Findings

### 1. Call-Site & Gating Audit (`model.py` & `dspark/speculator.py`)

We audited every L2 prefetch call site across `vllm/models/deepseek_v4_1/`:

- **Decoder layer (`vllm/models/deepseek_v4_1/nvidia/model.py`)**:
  - `_issue_wo_prefetch(hidden_states)` is gated by `self._l2_prefetch_wo is not None` and `hidden_states.shape[0] <= self._l2_prefetch_max_tokens` (`32`), and is skipped when `_attention_front_ran` is active (`prefill` only, `rows > 32`).
  - `FFN` and `NEXT` prefetches inside `forward()` are gated by `num_tokens <= self._l2_prefetch_max_tokens` (`32`) and `self._tp_size > 1`.
  - `_l2_prefetch_end_step` joins the side stream once at the end of `DeepseekV41Model.forward()` (`num_tokens <= 32` and `tp_size > 1`).
- **DSpark speculative drafter (`vllm/models/deepseek_v4_1/nvidia/dspark/speculator.py`)**:
  - In `propose()` (lines 1185–1215), step `0` prefetches `W2` during `tp_all_reduce(h0)`.
  - For draft steps `k >= 1`, `if k >= active_draft_tokens: break` (line 1252) executes **before** `_l2_prefetch_w2.prefetch(stream)` (line 1254) and `tp_all_reduce(h)` (line 1255). Skipped draft steps (`k >= active_draft_tokens`, when adaptive confidence halts drafting early) exit the loop before issuing any prefetch.

### 2. GB10 `cp.async.bulk.prefetch.L2` Chunk Size & TMA Prefetch FIFO Behavior

Microbenchmarking `cp.async.bulk.prefetch.L2.global.L2::cache_hint` on GB10 (`sm_121a`) against a cold `16 MiB` buffer (flushed via a `64 MiB` L2 thrash buffer before each measurement, then read by a `ld.global.L1::no_allocate.v4.u32` consumer kernel; cold read `107.0 us`, hot L2 read `39.9 us`) revealed two hardware properties:

1. **64 KiB maximum hardware bulk-prefetch chunk size**:
   - `cp.async.bulk.prefetch.L2` honors `chunk = 65536` (`64 KiB`) requests (`256` requests for `16 MiB`), whereas `128 KiB` and `1 MiB` requests are silently ignored by hardware.
2. **Multi-CTA 4 KiB flood overflows the TMA prefetch FIFO**:
   - Under the existing kernel (`grid=2, block=128, chunk=4096` = `256` threads issuing `4,096` `4 KiB` requests), the subsequent `16 MiB` consumer read takes `71.5–74.8 us` (`40.5–41.2%` effective L2 hit rate).
   - Under `grid=4, block=128, chunk=4096` (`512` threads issuing `4,096` `4 KiB` requests, the `tp3-triangle` configuration in `config/cluster.json`), the consumer read degrades to `81.7 us` (`28.8–31.7%` effective L2 hit rate), and in the 4-layer decode chain `grid=4` is **`40–42 us/layer` slower than `grid=2` even when the concurrent window kernel performs zero memory accesses (`pure_wait`)** because `512` concurrent threads overflowing the TMA prefetch FIFO stall at kernel completion.
   - Issuing `64 KiB` (`65536`-byte) chunks from a single active thread (`CTA 0, lane 0`) with an initial `128 KiB` (`2`-chunk) burst followed by `200 ns` `%globaltimer` pacing reduces the `16 MiB` consumer read to **`67.3 us`** (`46.3–49.8%` effective L2 hit rate, **`-10.0%` faster read than `grid=2`** and **`-17.6%` faster read than `grid=4`**) while keeping kernel launch duration at `~48–50 us` for `16 MiB` (fitting cleanly inside the `40–120 us` decode attention and `all_reduce` overlap windows).

### 3. Screening Sweep & Rejected Hypothesis (SWA `WO` Budget Reduction)

Before locking the candidate, we screened pacing intervals (`0 ns`, `100 ns`, `200 ns`, `300 ns`, `400 ns`) and also tested whether reducing `BUDGET_WO` on the 3 SWA layers per cycle from `16 MiB` to `10 MiB` (`cand_tl_64k_p200_swa10m`) improved end-to-end layer latency (`3 SWA + 1 Indexer` heterogeneous chain):

| Variant | `M=1` `pure_wait` (`us/layer`) | `M=1` `mem_active` (`us/layer`) | `M=6` `pure_wait` (`us/layer`) | `M=6` `mem_active` (`us/layer`) | `M=8` `pure_wait` (`us/layer`) | `M=8` `mem_active` (`us/layer`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `off` (no prefetch) | `508.93` | `596.48` | `1028.10` | `1115.65` | `1245.18` | `1332.99` |
| `orig_tp4_g2_4k` (`grid=2, 4 KiB`) | `484.35` | `576.26` | `1022.21` | `1114.62` | `1240.83` | `1335.55` |
| `orig_tp3_g4_4k` (`grid=4, 4 KiB`) | `525.06` | `619.26` | `1065.73` | `1159.42` | `1282.56` | `1378.82` |
| `cand_tl_64k_p200` (`64 KiB, 200 ns`, `16 MiB` `WO`) | **`467.20`** | **`558.59`** | **`1003.78`** | **`1099.01`** | **`1224.19`** | **`1318.91`** |
| `cand_tl_64k_p300` (`64 KiB, 300 ns`, `16 MiB` `WO`) | `468.22` | `559.10` | `1004.80` | `1099.52` | `1224.96` | `1318.66` |
| `cand_tl_64k_p200_swa10m` (`10 MiB` SWA `WO`) | `473.60` | `563.71` | `1011.20` | `1103.62` | `1230.85` | `1325.31` |

- **Conclusion on SWA `WO` budget**: Reducing SWA `BUDGET_WO` from `16 MiB` to `10 MiB` was **`+5.1 to +7.4 us/layer` slower** than keeping the full `16 MiB` `WO` budget with `64 KiB` + `200 ns` pacing, because `64 KiB` + `200 ns` pacing completes `16 MiB` in `~50 us` without memory-controller saturation, warming an extra `6 MiB` of `o_proj.b` into L2 before `wo.a` + `wo.b` execute. We therefore kept all segment budgets (`16 MiB / 16 MiB / 18 MiB / 4 MiB`) and `_SLICE_BYTES = 4096` unchanged.

---

## 2-Boot A/B Benchmark Results on `sp8` (`spark-ds41f:1107`)

### Exact Reproduction Command

```bash
flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host \
  -v "$PWD/experiments/2026-11-08-l2-prefetch-pacing:/work" \
  --entrypoint bash spark-ds41f:1107 -c '
    cd /opt/spark-ds41f/candidate/vllm &&
    patch -p1 < /work/vllm/0762-perf-dsv41-64k-chunked-globaltimer-paced-l2-prefetch.patch &&
    PYTHONPATH=/opt/spark-ds41f/candidate/vllm python3 /work/bench_l2_prefetch_ab.py \
      --boot 1 --replays 35 --out /work/receipts/boot1.json
  '
```

Receipts from two independent container boots on `sp8`:
- `experiments/2026-11-08-l2-prefetch-pacing/receipts/boot1.json`
- `experiments/2026-11-08-l2-prefetch-pacing/receipts/boot2.json`
- `experiments/2026-11-08-l2-prefetch-pacing/receipts/gpu-pytest-boot1.txt` (`34 passed, 1 skipped`)
- `experiments/2026-11-08-l2-prefetch-pacing/receipts/gpu-pytest-boot2.txt` (`34 passed, 1 skipped`)

### 1. Standalone `16 MiB` Weight Prefetch + Consumer Read (`35` reps/boot)

| Arm | Boot 1 Prefetch (`us`) | Boot 1 Consumer Read (`us`, Median ± CI95) | Boot 1 Effective L2 Hit | Boot 2 Prefetch (`us`) | Boot 2 Consumer Read (`us`, Median ± CI95) | Boot 2 Effective L2 Hit |
| :--- | ---: | :---: | ---: | ---: | :---: | ---: |
| `off` (cold `16 MiB` read) | `2.7` | `107.01 ± 0.90` | `0.0%` | `2.7` | `105.60 ± 0.91` | `0.0%` |
| `orig_tp4_g2_4k` (`grid=2, 4 KiB`) | `27.3` | `74.82 ± 0.87` | `40.5%` | `27.3` | `71.49 ± 0.90` | `41.2%` |
| `orig_tp3_g4_4k` (`grid=4, 4 KiB`) | `104.3` | `81.73 ± 0.42` | `31.7%` | `104.0` | `81.73 ± 0.25` | `28.8%` |
| **`cand_tl_64k_p200ns` (TileLang)** | **`48.1`** | **`67.33 ± 0.08`** | **`49.8%`** | **`47.9`** | **`67.33 ± 0.07`** | **`46.3%`** |
| **`cand_cute_64k_p200ns` (CuTe DSL)** | **`47.9`** | **`67.33 ± 0.08`** | **`49.8%`** | **`47.9`** | **`67.33 ± 0.08`** | **`46.3%`** |

### 2. 4-Layer Heterogeneous Decode Chain (`3 SWA + 1 Indexer`, `pure_wait` Windows, `us/layer`)

All 20 TileLang kernels per layer (`attention_entry`, `wqa_wkv`, `q_norm`, `wq_b`, `wo.a`, `wo.b`, `ffn_entry`, `gate_up`, `swiglu`, `down`, routed MoE `plan.run`) plus real `WO`, `FFN`, and `NEXT` L2 prefetch plans captured into a CUDA graph per arm and replayed in interleaved round-robin order (`35` replays/boot after cold `64 MiB` L2 flush). Output digests verified 100% bit-exact across all arms and rows:

| Rows (`M`) | `off` (`boot1 / boot2`) | `orig_tp4_g2_4k` (`boot1 / boot2`, Mean ± CI95) | `orig_tp3_g4_4k` (`boot1 / boot2`, Mean ± CI95) | `cand_tl_64k_p200ns` (`boot1 / boot2`, Mean ± CI95) | Delta vs `orig_tp4_g2_4k` (`boot1 / boot2`) | Delta vs `orig_tp3_g4_4k` (`boot1 / boot2`) | Bit-Exact |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1`** | `508.93 / 509.12` | `482.62 / 485.13` (`483.86 ± 0.66`) | `524.55 / 524.98` (`524.83 ± 0.56`) | **`467.06 / 470.12`** (`468.58 ± 0.58`) | **`-15.56 / -15.01 us` (`-3.22% / -3.09%`)** | **`-57.49 / -54.86 us` (`-10.96% / -10.45%`)** | `True` |
| **`2`** | `619.26 / 618.49` | `596.39 / 595.42` (`595.77 ± 0.58`) | `637.28 / 635.87` (`636.69 ± 0.58`) | **`578.41 / 577.32`** (`578.02 ± 0.55`) | **`-17.98 / -18.10 us` (`-3.01% / -3.04%`)** | **`-58.87 / -58.55 us` (`-9.24% / -9.21%`)** | `True` |
| **`4`** | `821.63 / 822.10` | `809.77 / 807.46` (`808.85 ± 0.90`) | `851.43 / 851.97` (`851.65 ± 0.87`) | **`791.98 / 790.59`** (`791.27 ± 0.86`) | **`-17.79 / -16.87 us` (`-2.20% / -2.09%`)** | **`-59.45 / -61.38 us` (`-6.98% / -7.20%`)** | `True` |
| **`6`** | `1027.39 / 1028.02` | `1022.45 / 1023.40` (`1023.16 ± 1.07`) | `1066.10 / 1064.70` (`1065.36 ± 1.15`) | **`1004.39 / 1004.77`** (`1004.75 ± 1.09`) | **`-18.06 / -18.63 us` (`-1.77% / -1.82%`)** | **`-61.71 / -59.93 us` (`-5.79% / -5.63%`)** | `True` |
| **`8`** | `1245.42 / 1245.32` | `1241.06 / 1242.01` (`1241.32 ± 1.15`) | `1280.62 / 1283.66` (`1282.34 ± 1.18`) | **`1223.78 / 1224.69`** (`1224.22 ± 1.18`) | **`-17.28 / -17.32 us` (`-1.39% / -1.39%`)** | **`-56.84 / -58.97 us` (`-4.44% / -4.59%`)** | `True` |
| **`16`** | `2014.43 / 2014.46` | `2007.62 / 2007.34` (`2007.01 ± 1.38`) | `2047.38 / 2048.13` (`2047.52 ± 1.34`) | **`1986.74 / 1986.70`** (`1987.21 ± 1.34`) | **`-20.88 / -20.64 us` (`-1.04% / -1.03%`)** | **`-60.64 / -61.43 us` (`-2.96% / -3.00%`)** | `True` |
| **`24`** | `2692.15 / 2697.81` | `2694.40 / 2695.85` (`2695.36 ± 1.88`) | `2730.02 / 2731.94` (`2731.02 ± 2.13`) | **`2675.25 / 2679.82`** (`2678.00 ± 1.98`) | **`-19.15 / -16.03 us` (`-0.71% / -0.59%`)** | **`-54.77 / -52.12 us` (`-2.01% / -1.91%`)** | `True` |
| **`32`** | `3328.91 / 3331.49` | `3328.46 / 3333.14` (`3331.34 ± 2.26`) | `3362.94 / 3369.05` (`3366.08 ± 2.34`) | **`3313.09 / 3315.90`** (`3314.95 ± 2.31`) | **`-15.37 / -17.24 us` (`-0.46% / -0.52%`)** | **`-49.85 / -53.15 us` (`-1.48% / -1.58%`)** | `True` |

Across all 61 decoder layers (`M = 1..8`), `-15.0 to -18.6 us/layer` corresponds to **`-0.92 to -1.13 ms/step`** vs. `grid=2` (`tp4-ring`) and **`-3.35 to -3.76 ms/step`** vs. `grid=4` (`tp3-triangle`).

### 3. 4-Layer Heterogeneous Decode Chain (`3 SWA + 1 Indexer`, `mem_active` Concurrent Windows, `us/layer`)

When the concurrent window kernels actively issue dependent memory reads while the side-stream L2 prefetch runs:

| Rows (`M`) | `off` (`boot1 / boot2`) | `orig_tp4_g2_4k` (`boot1 / boot2`, Mean ± CI95) | `orig_tp3_g4_4k` (`boot1 / boot2`, Mean ± CI95) | `cand_tl_64k_p200ns` (`boot1 / boot2`, Mean ± CI95) | Delta vs `orig_tp4_g2_4k` (`boot1 / boot2`) | Delta vs `orig_tp3_g4_4k` (`boot1 / boot2`) | Bit-Exact |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1`** | `596.32 / 599.26` | `574.34 / 580.42` (`577.79 ± 0.87`) | `616.76 / 620.86` (`618.73 ± 0.90`) | **`552.63 / 562.50`** (`557.88 ± 0.85`) | **`-21.71 / -17.92 us` (`-3.78% / -3.09%`)** | **`-64.13 / -58.36 us` (`-10.40% / -9.40%`)** | `True` |
| **`2`** | `706.59 / 706.08` | `688.35 / 688.45` (`688.38 ± 0.84`) | `729.69 / 730.94` (`730.36 ± 0.91`) | **`667.05 / 664.63`** (`666.16 ± 0.81`) | **`-21.30 / -23.82 us` (`-3.09% / -3.46%`)** | **`-62.64 / -66.31 us` (`-8.58% / -9.07%`)** | `True` |
| **`4`** | `912.64 / 911.09` | `901.79 / 901.51` (`901.66 ± 1.07`) | `946.46 / 945.44` (`946.08 ± 1.04`) | **`881.87 / 882.11`** (`882.06 ± 0.98`) | **`-19.92 / -19.40 us` (`-2.21% / -2.15%`)** | **`-64.59 / -63.33 us` (`-6.82% / -6.70%`)** | `True` |
| **`6`** | `1115.65 / 1117.70` | `1116.57 / 1118.92` (`1117.75 ± 1.25`) | `1159.42 / 1161.47` (`1160.38 ± 1.24`) | **`1097.17 / 1103.54`** (`1100.32 ± 1.24`) | **`-19.40 / -15.38 us` (`-1.74% / -1.37%`)** | **`-62.25 / -57.93 us` (`-5.37% / -4.99%`)** | `True` |
| **`8`** | `1332.51 / 1336.58` | `1336.06 / 1340.67` (`1338.28 ± 1.29`) | `1378.41 / 1381.89` (`1380.12 ± 1.32`) | **`1320.11 / 1321.47`** (`1320.78 ± 1.26`) | **`-15.95 / -19.20 us` (`-1.19% / -1.43%`)** | **`-58.30 / -60.42 us` (`-4.23% / -4.37%`)** | `True` |
| **`16`** | `2101.50 / 2104.32` | `2113.02 / 2116.10` (`2114.54 ± 1.54`) | `2155.01 / 2158.08` (`2156.45 ± 1.58`) | **`2095.71 / 2100.76`** (`2098.24 ± 1.48`) | **`-17.31 / -15.34 us` (`-0.82% / -0.72%`)** | **`-59.30 / -57.32 us` (`-2.75% / -2.66%`)** | `True` |
| **`24`** | `2780.16 / 2784.26` | `2792.45 / 2796.54` (`2794.50 ± 2.02`) | `2832.38 / 2836.48` (`2834.43 ± 2.11`) | **`2782.63 / 2787.08`** (`2784.85 ± 1.98`) | **`-9.82 / -9.46 us` (`-0.35% / -0.34%`)** | **`-49.75 / -49.40 us` (`-1.76% / -1.74%`)** | `True` |
| **`32`** | `3418.11 / 3422.21` | `3431.42 / 3430.91` (`3431.17 ± 2.34`) | `3471.36 / 3473.41` (`3472.38 ± 2.40`) | **`3418.84 / 3423.09`** (`3420.97 ± 2.30`) | **`-12.58 / -7.82 us` (`-0.37% / -0.23%`)** | **`-52.52 / -50.32 us` (`-1.51% / -1.45%`)** | `True` |

Notice that at `M >= 6` under `mem_active`, the unpaced `orig_tp4_g2_4k` and `orig_tp3_g4_4k` kernels actually **regress against `off`** because their unpaced `4 KiB` prefetch flood stalls the concurrent memory-active window kernel by more than it saves in the consumer GEMM, whereas `cand_tl_64k_p200ns` (`64 KiB` chunked + `200 ns` `%globaltimer` pacing) remains a net win across the entire decode range.
