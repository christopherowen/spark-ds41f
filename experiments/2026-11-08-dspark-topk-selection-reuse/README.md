# DSpark Target-to-Drafter and Step-0 Anchor Indexer Top-K Selection Reuse

## Metadata

- **Date:** 2026-11-08
- **Base commit:** `8e5ea208f3d22bc69df91ace94d6975932c858a8` (`spark-ds41f-1107`, vLLM patch head `2756478b60b9`, tree `ed1bdb7a7684`)
- **Candidate patch:** `experiments/2026-11-08-dspark-topk-selection-reuse/vllm/0762-perf-dsv41-dspark-target-and-anchor-topk-selection-reuse.patch` (vLLM patch head `c7ae910ad89f`, tree `4418bf7a9180`)
- **Node:** `sp5` (NVIDIA GB10, `spark-ds41f:1107`)
- **Variable tested:** Reusing the target model's accepted anchor row top-$k$ KV selection (`layers.36.attn.topk_indices_buffer`) and intra-draft step-0 anchor top-$k$ selection across the 5 parallel query positions of each request in `DSpark` (`layers.40..42`).

## Architectural Analysis of `1107`

1. **`1107` Baseline (`compress_ratio = 0`, Window-Only Draft Attention):**
   - In `DeepSeek-V4.1-Flash` (`config.json`), `compress_ratios` has 43 entries: `[0, 0, 2..2 (x18), 1..1 (x20), 0, 0, 0]`, where the 3 DSpark draft layers (`layers.40, 41, 42`, loaded from `mtp.0, mtp.1, mtp.2`) have `compress_ratio = 0`.
   - The checkpoint carries no `mtp.{0,1,2}.attn.indexer.*` or `mtp.{0,1,2}.attn.compressor.*` weights. Consequently, in `1107`, DSpark draft layers execute window-only sparse MLA (`swa_width = 136`, `idx_width = 0`), spend `0.00 ms` in indexer scoring, and maintain constant $O(1)$ attention latency (`0.035–0.036 ms/step` at `c1`, `0.064–0.066 ms/step` at `c8` on `TP=4`) from `4K` to `256K`.
2. **Indexed Draft Attention (`compress_ratio = 1`, `idx_width = 512`) and Target Top-$k$ Reuse:**
   - Because `SparseMlaV41Plan` reads both the draft layer's uncompressed sliding-window cache (`swa_cache`, `136` slots of `528 B`) and the compressed MLA cache (`idx_cache`, `512` selected positions of `288 B`) through the same query heads (`wq_a -> q_norm -> wq_b`, `[H, 512]`) and output projections (`wo_a -> wo_b`), enabling `compress_ratio = 1` on DSpark draft layers requires **zero new checkpoint weights** when reusing the target decoder's compressed MLA cache (`layers.20.attn.kv_cache`) and `topk_indices_buffer` (`layers.36.attn.topk_indices_buffer`).
   - During each speculative decode cycle, the target model verifies `1 + K = 6` candidate tokens per request at `layers.0..39`. At layer 36 (the final target `index_source_layer_id`), the target indexer scores the full `KPool` and populates `layers.36.attn.topk_indices_buffer` (`[num_verify_tokens, 512]`).
   - After rejection sampling, each request $r$ has $n_{\text{rej}}[r] \in [0, K]$ rejected tokens, so its accepted bonus token (the DSpark anchor position) was evaluated at target row:
     $$\text{anchor\_row}[r] = \max(q_{\text{end}}[r] - n_{\text{rej}}[r] - 1,\, q_{\text{start}}[r]) = \text{clamp}(q_{\text{start}}[r] + (\text{draft\_req\_pos}[r] - \text{target\_req\_pos}[r]),\, q_{\text{start}}[r],\, \max(q_{\text{end}}[r] - 1, q_{\text{start}}[r]))$$
   - Because `target_meta.query_start_loc`, `target_meta.request_positions`, and `draft_swa.request_positions` are persistent static-address GPU buffers updated in place by `DeepseekV41B12xMetadataBuilder` before draft execution, resolving `anchor_row` from `draft_request_positions - target_request_positions` (or explicit `num_rejected`) captures and replays directly inside `CUDAGraphMode.FULL` without CPU-GPU synchronization.
   - Furthermore, in `tilelang/metadata.py` (`_window_macro` with `draft=True`), DSpark's `swa_cache` (`WINDOW_SIZE = 128`, `swa_width = 136`) already covers all 5 newly drafted tail positions (`anchor_pos .. anchor_pos + 4`) in uncompressed form, while the target's `idx_cache` holds historical tokens up to `anchor_pos`. Broadcasting each request's anchor top-$k$ row across its 5 draft query rows (`gather_dspark_anchor_topk_indices` in `"target"` mode, or `broadcast_dspark_step0_topk_indices` in `"anchor"` step-0 mode) eliminates `100%` (`5/5` rows) of draft indexer scoring in `"target"` mode and `80%` (`4/5` rows) in `"anchor"` mode.
   - Because the target model verifies every drafted token on the next verification step, reusing top-$k$ selection in the drafter is **100% lossless (`KL = 0`)** with respect to target model output quality.

## Exact Commands

```bash
# 2-boot GPU latency + checkpoint agreement benchmark on sp5:
for boot in 1 2; do
  docker run --rm --gpus all --ipc=host --entrypoint python3 \
    -v /home/chris/projects/spark-ds41f/.work/upstreams/vllm/vllm/models/deepseek_v4_1:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1:ro \
    -v /home/chris/.cache/huggingface:/root/.cache/huggingface:ro \
    -v /tmp/tilelang_cache:/root/.tilelang/cache \
    -v /home/chris/projects/spark-ds41f/experiments/2026-11-08-dspark-topk-selection-reuse/bench_dspark_topk_reuse.py:/tmp/bench_dspark_topk_reuse.py:ro \
    -v /home/chris/projects/spark-ds41f/experiments/2026-11-08-dspark-topk-selection-reuse/receipts:/tmp/bench_dspark_results \
    spark-ds41f:1107 \
    /tmp/bench_dspark_topk_reuse.py /tmp/bench_dspark_results/boot${boot}.json
done

# 2-boot GPU unit test suite on sp5:
for boot in 1 2; do
  docker run --rm --gpus all --ipc=host --entrypoint pytest \
    -v /home/chris/projects/spark-ds41f/.work/upstreams/vllm/vllm/models/deepseek_v4_1:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1:ro \
    -v /home/chris/projects/spark-ds41f/.work/upstreams/vllm/tests/models:/tmp/test_models:ro \
    -v /tmp/tilelang_cache:/root/.tilelang/cache \
    spark-ds41f:1107 \
    -v /tmp/test_models/test_deepseek_v4_1_dspark_topk_reuse.py \
       /tmp/test_models/test_deepseek_v4_1_attention.py \
    | tee experiments/2026-11-08-dspark-topk-selection-reuse/receipts/pytest-boot${boot}.txt
done
```

## 2-Boot Benchmark Results on `sp5` (`CUDAGraphMode.FULL`)

### Table 1: 3-Layer DSpark Attention + Indexer Latency (`ms/step`, Boot 1 / Boot 2 / Mean)

Modes compared across `4K`, `16K`, `64K`, `128K`, and `256K` context lengths:
- **`1107` Window-Only (`compress_ratio=0`)**: 3 draft layers with `swa_width=136`, `idx_width=0` (`0` indexer rows).
- **Full Re-Index (1-src, `5N` rows)**: 1 shared draft indexer scoring all `5` query rows/req + 3 draft layers with `swa_width=136, idx_width=512`.
- **Full Re-Index (3-src, `15N` rows)**: 3 per-layer draft indexers scoring all `5` query rows/req + 3 draft layers with `swa_width=136, idx_width=512`.
- **Step-0 Anchor Reuse (`1N` row)**: 1 draft indexer scoring only step `0` (`1` row/req) + `broadcast_dspark_step0_topk_indices` + 3 draft layers with `idx_width=512`.
- **Target Anchor Reuse (`0` rows)**: `gather_dspark_anchor_topk_indices` from target layer 36 (`0` draft indexer rows) + 3 draft layers with `idx_width=512`.

| Shape | Load | Context | `1107` Window-Only (`ms`) | Full Re-Index 1-src (`ms`) | Full Re-Index 3-src (`ms`) | Step-0 Anchor Reuse (`ms`) | Target Anchor Reuse (`ms`) | Speedup (`Target` vs `1-src` / `3-src`) |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TP=4** | **c1** (`5` rows) | `4K` | `0.0350 / 0.0359` (**0.0355**) | `0.0553 / 0.0558` (**0.0556**) | `0.0848 / 0.0891` (**0.0870**) | `0.0778 / 0.0780` (**0.0779**) | `0.0797 / 0.0799` (**0.0798**) | `0.70x / 1.09x` |
| **TP=4** | **c1** (`5` rows) | `16K` | `0.0361 / 0.0348` (**0.0355**) | `0.0785 / 0.0803` (**0.0794**) | `0.1586 / 0.1618` (**0.1602**) | `0.0876 / 0.0881` (**0.0879**) | `0.0797 / 0.0799` (**0.0798**) | `0.99x / 2.01x` |
| **TP=4** | **c1** (`5` rows) | `64K` | `0.0355 / 0.0354` (**0.0355**) | `0.1483 / 0.1515` (**0.1499**) | `0.3641 / 0.3669` (**0.3655**) | `0.1079 / 0.1081` (**0.1080**) | `0.0838 / 0.0848` (**0.0843**) | **1.78x / 4.34x** |
| **TP=4** | **c1** (`5` rows) | `128K` | `0.0367 / 0.0348` (**0.0358**) | `0.2331 / 0.2591` (**0.2461**) | `0.6268 / 0.6348` (**0.6308**) | `0.1305 / 0.1309` (**0.1307**) | `0.0844 / 0.0854` (**0.0849**) | **2.90x / 7.43x** |
| **TP=4** | **c1** (`5` rows) | `256K` | `0.0353 / 0.0350` (**0.0352**) | `0.4233 / 0.4343` (**0.4288**) | `1.1718 / 1.1686` (**1.1702**) | `0.1901 / 0.1865` (**0.1883**) | `0.0817 / 0.0819` (**0.0818**) | **5.24x / 14.31x** |
| **TP=4** | **c8** (`40` rows) | `4K` | `0.0665 / 0.0638` (**0.0652**) | `0.2057 / 0.2030` (**0.2044**) | `0.3263 / 0.3270` (**0.3267**) | `0.1976 / 0.1963` (**0.1970**) | `0.1925 / 0.1948` (**0.1937**) | **1.06x / 1.69x** |
| **TP=4** | **c8** (`40` rows) | `16K` | `0.0647 / 0.0635` (**0.0641**) | `0.3336 / 0.3350` (**0.3343**) | `0.7056 / 0.7112` (**0.7084**) | `0.2299 / 0.2317` (**0.2308**) | `0.1929 / 0.1929` (**0.1929**) | **1.73x / 3.67x** |
| **TP=4** | **c8** (`40` rows) | `64K` | `0.0635 / 0.0635` (**0.0635**) | `0.8217 / 0.8258` (**0.8238**) | `2.1651 / 2.1615` (**2.1633**) | `0.3327 / 0.3321` (**0.3324**) | `0.1938 / 0.1943` (**0.1941**) | **4.24x / 11.15x** |
| **TP=4** | **c8** (`40` rows) | `128K` | `0.0662 / 0.0635` (**0.0649**) | `1.4823 / 1.4787` (**1.4805**) | `4.1089 / 4.1075` (**4.1082**) | `0.4716 / 0.4718` (**0.4717**) | `0.1936 / 0.1960` (**0.1948**) | **7.60x / 21.09x** |
| **TP=4** | **c8** (`40` rows) | `256K` | `0.0637 / 0.0635` (**0.0636**) | `2.8918 / 2.8860` (**2.8889**) | `8.3037 / 8.2902` (**8.2970**) | `0.7739 / 0.7686` (**0.7713**) | `0.1974 / 0.1990` (**0.1982**) | **14.58x / 41.86x** |
| **TP=3** | **c1** (`5` rows) | `4K` | `0.0362 / 0.0397` (**0.0380**) | `0.0694 / 0.0664` (**0.0679**) | `0.0961 / 0.1018` (**0.0990**) | `0.0886 / 0.0887` (**0.0887**) | `0.0921 / 0.0935` (**0.0928**) | `0.73x / 1.07x` |
| **TP=3** | **c1** (`5` rows) | `16K` | `0.0356 / 0.0369` (**0.0363**) | `0.0898 / 0.0939` (**0.0919**) | `0.1683 / 0.1722` (**0.1703**) | `0.0981 / 0.0994` (**0.0988**) | `0.0938 / 0.0908` (**0.0923**) | **1.00x / 1.85x** |
| **TP=3** | **c1** (`5` rows) | `64K` | `0.0363 / 0.0353` (**0.0358**) | `0.1582 / 0.1597` (**0.1590**) | `0.3791 / 0.3751` (**0.3771**) | `0.1184 / 0.1184` (**0.1184**) | `0.0933 / 0.0936` (**0.0935**) | **1.70x / 4.03x** |
| **TP=3** | **c1** (`5` rows) | `128K` | `0.0363 / 0.0369` (**0.0366**) | `0.2433 / 0.2487` (**0.2460**) | `0.6370 / 0.6349` (**0.6360**) | `0.1450 / 0.1400` (**0.1425**) | `0.0938 / 0.0914` (**0.0926**) | **2.66x / 6.87x** |
| **TP=3** | **c1** (`5` rows) | `256K` | `0.0365 / 0.0359` (**0.0362**) | `0.4360 / 0.4311` (**0.4336**) | `1.1852 / 1.1880` (**1.1866**) | `0.2018 / 0.1966` (**0.1992**) | `0.0964 / 0.0955` (**0.0960**) | **4.52x / 12.36x** |
| **TP=3** | **c8** (`40` rows) | `4K` | `0.0758 / 0.0808` (**0.0783**) | `0.2284 / 0.2265` (**0.2275**) | `0.3469 / 0.3475` (**0.3472**) | `0.2173 / 0.2156` (**0.2165**) | `0.2129 / 0.2140` (**0.2135**) | **1.07x / 1.63x** |
| **TP=3** | **c8** (`40` rows) | `16K` | `0.0763 / 0.0777` (**0.0770**) | `0.3550 / 0.3540` (**0.3545**) | `0.7273 / 0.7273` (**0.7273**) | `0.2519 / 0.2518` (**0.2519**) | `0.2144 / 0.2151` (**0.2148**) | **1.65x / 3.39x** |
| **TP=3** | **c8** (`40` rows) | `64K` | `0.0789 / 0.0777` (**0.0783**) | `0.8779 / 0.8436` (**0.8608**) | `2.1936 / 2.1824` (**2.1880**) | `0.3527 / 0.3524` (**0.3526**) | `0.2146 / 0.2147` (**0.2147**) | **4.01x / 10.19x** |
| **TP=3** | **c8** (`40` rows) | `128K` | `0.0758 / 0.0778` (**0.0768**) | `1.5128 / 1.5054` (**1.5091**) | `4.1414 / 4.1868` (**4.1641**) | `0.4933 / 0.4923` (**0.4928**) | `0.2158 / 0.2180` (**0.2169**) | **6.96x / 19.20x** |
| **TP=3** | **c8** (`40` rows) | `256K` | `0.0758 / 0.0779` (**0.0769**) | `2.9200 / 2.9119` (**2.9160**) | `8.3355 / 8.3572` (**8.3464**) | `0.8109 / 0.8132` (**0.8121**) | `0.2199 / 0.2216` (**0.2208**) | **13.21x / 37.80x** |

### Table 2: Total 5-Step DSpark Draft Latency, End-to-End Step Time (`ms/step`), and Net Throughput (`tok/s`)

| Shape | Load | Context | `1107` Window Draft (`ms`) | Full 1-src Draft (`ms`) | Full 3-src Draft (`ms`) | Step-0 Reuse Draft (`ms`) | Target Reuse Draft (`ms`) | E2E Savings (`Target` vs `1-src` / `3-src`, `ms/step`) | Net `tok/s` (`1107` / `Full 1-src` / `Full 3-src` / `Target Reuse`) |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TP=4** | **c1** | `4K` | `4.4555` | `4.4756` | `4.5070` | `4.4979` | `4.4998` | `-0.024 / +0.007` | `90.40 / 90.36 / 90.29 / 90.31` |
| **TP=4** | **c1** | `16K` | `4.4555` | `4.4994` | `4.5802` | `4.5079` | `4.4998` | `0.000 / +0.080` | `84.13 / 84.05 / 83.90 / 84.05` |
| **TP=4** | **c1** | `64K` | `4.4555` | `4.5699` | `4.7855` | `4.5280` | `4.5043` | `+0.066 / +0.281` | `67.19 / 67.05 / 66.80 / 67.13` |
| **TP=4** | **c1** | `128K` | `4.4558` | `4.6661` | `5.0508` | `4.5507` | `4.5049` | `+0.161 / +0.546` | `52.87 / 52.71 / 52.44 / 52.83` |
| **TP=4** | **c1** | `256K` | `4.4552` | `4.8488` | `5.5902` | `4.6083` | `4.5018` | **+0.347 / +1.088** | `37.10 / 36.96 / 36.70 / 37.09` |
| **TP=4** | **c8** | `4K` | `5.9252` | `6.0644` | `6.1867` | `6.0570` | `6.0537` | `+0.011 / +0.133` | `459.10 / 458.11 / 457.24 / 458.18` |
| **TP=4** | **c8** | `16K` | `5.9241` | `6.1943` | `6.5684` | `6.0908` | `6.0529` | `+0.141 / +0.516` | `407.06 / 405.54 / 403.46 / 406.33` |
| **TP=4** | **c8** | `64K` | `5.9235` | `6.6838` | `8.0233` | `6.1924` | `6.0541` | `+0.630 / +1.969` | `286.60 / 284.49 / 280.85 / 286.23` |
| **TP=4** | **c8** | `128K` | `5.9249` | `7.3405` | `9.9682` | `6.3317` | `6.0548` | `+1.286 / +3.913` | `205.27 / 203.26 / 199.64 / 205.08` |
| **TP=4** | **c8** | `256K` | `5.9236` | `8.7489` | `14.1570` | `6.6313` | `6.0582` | **+2.691 / +8.099** | `130.95 / 129.32 / 126.32 / 130.87` |
| **TP=3** | **c1** | `4K` | `4.8180` | `4.8479` | `4.8790` | `4.8687` | `4.8728` | `-0.025 / +0.006` | `81.77 / 81.72 / 81.66 / 81.67` |
| **TP=3** | **c1** | `16K` | `4.8163` | `4.8719` | `4.9503` | `4.8788` | `4.8723` | `0.000 / +0.078` | `75.77 / 75.69 / 75.57 / 75.68` |
| **TP=3** | **c1** | `64K` | `4.8158` | `4.9390` | `5.1571` | `4.8984` | `4.8735` | `+0.066 / +0.284` | `60.14 / 60.03 / 59.82 / 60.09` |
| **TP=3** | **c1** | `128K` | `4.8166` | `5.0260` | `5.4160` | `4.9225` | `4.8726` | `+0.153 / +0.543` | `47.15 / 47.03 / 46.81 / 47.12` |
| **TP=3** | **c1** | `256K` | `4.8162` | `5.2136` | `5.9666` | `4.9792` | `4.8760` | **+0.338 / +1.091** | `32.98 / 32.87 / 32.66 / 32.97` |
| **TP=3** | **c8** | `4K` | `6.4183` | `6.5675` | `6.6872` | `6.5565` | `6.5535` | `+0.014 / +0.134` | `413.38 / 412.51 / 411.83 / 412.60` |
| **TP=3** | **c8** | `16K` | `6.4170` | `6.6945` | `7.0673` | `6.5919` | `6.5548` | `+0.140 / +0.513` | `365.64 / 364.38 / 362.71 / 365.01` |
| **TP=3** | **c8** | `64K` | `6.4183` | `7.2008` | `8.5280` | `6.6926` | `6.5547` | `+0.646 / +1.973` | `256.18 / 254.45 / 251.56 / 255.88` |
| **TP=3** | **c8** | `128K` | `6.4168` | `7.8491` | `10.5041` | `6.8328` | `6.5569` | `+1.292 / +3.947` | `183.29 / 181.67 / 178.75 / 183.13` |
| **TP=3** | **c8** | `256K` | `6.4169` | `9.2560` | `14.6864` | `7.1521` | `6.5608` | **+2.695 / +8.126** | `116.82 / 115.52 / 113.11 / 116.75` |

### Table 3: Top-$k$ Index Overlap, 3-Layer DSpark Cosine Similarity, and 5-Step Draft Token Agreement (`4K..256K`)

Evaluated using real `DeepSeek-V4.1-Flash` checkpoint weights (`layers.36.attn.{wq_a,q_norm,indexer}`, `mtp.{0,1,2}.{attn_norm,hc_attn_scale,attn}`, `mtp.2.{norm,markov_head}`, `head.weight`, and `compress_rope_theta = 160000`) across 24 trials $\times$ 5 draft positions = 120 draft positions per context length on both Boot 1 and Boot 2 (100% deterministic across boots):

| Context | Top-$k$ Overlap by Step (`s=0..4`, `%`) | Mean Overlap (`%`) | 3-Layer Output Cosine Sim (`s=0..4`) | Mean Cosine Sim | Top-1 Draft Token Agreement (`s=0..4`, `%`) | Mean Top-1 Agreement (`%`) |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`4K`** | `100.0 / 97.29 / 96.63 / 95.94 / 95.56` | **97.08%** | `1.000000 / 0.999975 / 0.999971 / 0.999967 / 0.999964` | **0.999975** | `100.0 / 100.0 / 95.83 / 91.67 / 91.67` | **95.83%** |
| **`16K`** | `100.0 / 98.92 / 98.60 / 98.39 / 98.30` | **98.84%** | `1.000000 / 0.999985 / 0.999983 / 0.999982 / 0.999981` | **0.999986** | `100.0 / 100.0 / 100.0 / 95.83 / 91.67` | **97.50%** |
| **`64K`** | `100.0 / 98.71 / 98.43 / 98.27 / 98.12` | **98.70%** | `1.000000 / 0.999984 / 0.999982 / 0.999981 / 0.999980` | **0.999985** | `100.0 / 100.0 / 100.0 / 100.0 / 100.0` | **100.00%** |
| **`128K`** | `100.0 / 98.71 / 98.43 / 98.35 / 98.15` | **98.73%** | `1.000000 / 0.999984 / 0.999982 / 0.999982 / 0.999981` | **0.999986** | `100.0 / 100.0 / 100.0 / 100.0 / 100.0` | **100.00%** |
| **`256K`** | `100.0 / 98.78 / 98.54 / 98.37 / 98.21` | **98.78%** | `1.000000 / 0.999985 / 0.999983 / 0.999981 / 0.999981` | **0.999986** | `100.0 / 100.0 / 100.0 / 100.0 / 95.83` | **99.17%** |

## Unit Tests

- **vLLM GPU unit tests (`tests/models/test_deepseek_v4_1_dspark_topk_reuse.py` + `tests/models/test_deepseek_v4_1_attention.py`):**
  - `test_broadcast_dspark_step0_topk_when_seq_len_le_topk` (`seq_len <= 512`, including straddling `510 -> 514` across steps `0..4`): **PASSED**
  - `test_broadcast_dspark_step0_topk_when_seq_len_gt_topk` (`seq_len > 512`, `4K` and `128K`): **PASSED**
  - `test_gather_dspark_anchor_topk_rejected_slots_overwritten_next_step` (multi-request rejection offsets `q_end - num_rejected - 1`, position-offset equivalence `q_start + (draft_req_pos - target_req_pos)`, all-rejected clamping, and next-step overwrite of rejected draft slots): **PASSED**
  - `test_dspark_topk_reuse_cudagraph_capture_and_replay` (`torch.cuda.CUDAGraph` capture and multi-replay bit-exactness on GPU): **PASSED**
  - `test_deepseek_v4_attention_dspark_target_and_step0_topk_reuse` (end-to-end `DeepseekV4Attention.forward_mqa` across `"target"`, `"anchor"`, and `"full"` modes): **PASSED**
  - Boot 1 (`receipts/pytest-boot1.txt`): **`70 passed in 13.88s`**
  - Boot 2 (`receipts/pytest-boot2.txt`): **`70 passed in 14.17s`**
- **Deployment repository unit tests (`tests/test_dspark_topk_reuse.py`):**
  - **`5 passed`** (`Ran 5 tests in 0.08s — OK`).

## Conclusion

1. In `1107`, `DeepSeek-V4.1-Flash` configures `compress_ratios = [0, 0, 0]` on all 3 DSpark draft layers (`layers.40..42`, `mtp.0..2`), so production DSpark draft attention is already window-only (`swa_width = 136`, `idx_width = 0`, `0.00 ms` indexer work).
2. When indexed draft attention (`compress_ratio = 1`, `idx_width = 512`) is enabled on DSpark draft layers, reusing the target model's accepted anchor row top-$k$ selection (`gather_dspark_anchor_topk_indices`, `"target"` mode) requires zero new checkpoint weights, eliminates `100%` of draft indexer work (`5/5` rows), keeps 3-layer DSpark indexed attention latency constant $O(1)$ (`~0.080–0.085 ms/step` at `c1` and `~0.193–0.198 ms/step` at `c8` on `TP=4`) across `4K..256K`, saves up to **`2.69 ms/step`** (`14.58x` faster attention) vs 1 shared draft indexer and **`8.10 ms/step`** (`41.86x` faster attention) vs 3 per-layer draft indexers at `256K c8`, and preserves **`97.1%–98.8%`** top-$k$ index overlap, **`0.999975–0.999986`** 3-layer output cosine similarity, and **`95.8%–100.0%` (`98.5%` mean)** 5-step Markov top-1 draft token agreement.
