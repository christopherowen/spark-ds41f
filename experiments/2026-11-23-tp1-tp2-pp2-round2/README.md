# Round 2 `tp1`, `tp2`, and `pp2` Performance Exploration (`2026-11-23`)

- **Date:** 2026-11-23
- **Base image:** `spark-ds41f:candidate-70-r2` (`sha256:e392b4962a65cb5ac08b41d9c1aae8c823b25696462ba9e571b0a69c729d4e37`, vLLM `0001..0860`, tree `13dc1169a68d`)
- **Base deployment commit:** `290d2994df` (`candidate-70`)
- **Branch:** `tp1-tp2-pp2-round2`
- **Model:** DeepSeek-V4.1-Flash-1017 (`hidden=5120`, `moe_intermediate=2304`, `384` routed experts, `top_k=6`, `40` layers / `39` MoE)

---

## Executive Summary of Findings

| Track | Idea Tested | Verdict | Measured Impact |
|---|---|---|---|
| **`pp2-tp2-ring`** (translates to `pp2-tp{3,4}-switched`) | **Coalesced metadata-free PP P2P (`VLLM_PP_FAST_P2P=1`)** + async zero-sync PP adaptive verification + `PP_USE_ENGINE_REQS=1` + `VLLM_SINK_REJECTION=1` | **HUGE PURE WIN** | **All 12/12 points positive (`LRU 5/5`, zero regressions):** `prose-c1 +18.2%` (`47.4`), `prose-c2 +6.8%` (`81.4`), `prose-c4 +18.4%` (`146.0`), `prose-c8 +14.8%` (`241.4`), `code-c1 +7.3%` (`54.0`), **`code-c2 +2.1%` (`95.1`, fixed `-5.2%` dip!)**, `code-c8 +28.0%` (`301.4`), `code-c16 +11.8%` (`498.3`), prefill `+14.3% / +15.7%`. Isolated `VLLM_PP_FAST_P2P=1` cut **`4.7–7.7 ms/step`** (`+7.3%..+11.6%` decode) and `-31%` `c16` TTFT (`0.700 -> 0.481 s`). *Next round: move PP P2P into `dgx-spark-networking` (`sparknet`) so TP and PP share one RDMA transport.* |
| **`tp1` & `tp2-direct`** | **Sonar-disjoint 5-domain held-out hotlist** (`4,013,544` routes across `prose`/`code`/`math`/`agent`/`ops`, zero Sonar overlap) + **`FAST_LOOKAHEAD=1`** | **HUGE PURE WIN** | **`tp2-direct` (vs `1120-direct2` / vs `c70`):** `prose-c1 29.4` (**`+66.3%`** / `+36%`), `prose-c4 95.5` (**`+80.3%`** / `+31%`), `code-c1 32.9` (**`+48.4%`** / `+33%`), `code-c4 99.7` (**`+55.3%`** / `+38%`). **`tp1` (vs `1120-tp1` / vs `c70`):** `prose-c1 9.2` (**`+108.8%`** / `+37%`), `prose-c4 30.0` (**`+174.2%`** / `+96%`), `code-c1 8.4` (**`+75.4%`** / `+27%`), `code-c4 28.5` (**`+150.8%`** / `+47%`). `LRU 5/5`, `8.51 GiB` free RAM on `tp1`, `5.43 / 7.23 GiB` on `tp2`. |
| **`tp1` & `tp2-direct`** | **BF16 fused single-sync `L+1` lookahead (`VLLM_DS41_EXPERT_FAST_LOOKAHEAD=1`)** | **PURE WIN** | Removes 40× `6.3 MB` FP32 gate-weight casts/step and cuts 80 -> 40 D2H syncs/step. **`tp2`:** `+2–3%` decode (`22.2 / 74.2 / 25.5`). **`tp1`:** `+36.6%` `prose-c4` (`15.3 -> 20.9`) and `+6.7%` `code-c4` (`19.4 -> 20.7`) even without a hotlist. |
| **`tp2` & `tp3` & `pp2-tp{2,3}`** | **TileLang MoE `down_num_stages=3` when `intermediate % 128 == 0`** | **CLEAN KERNEL WIN** | Standalone GB10 microbenchmark: strictly faster at every tested row count on both `TP=2` (`inter=1152`: `-1.2%..-3.3%`) and `TP=3` (`inter=768`: `-0.7%..-2.5%`). Leaves `TP=4` (`inter=576`, 64-wide tail) unchanged. `stages=4` and `prefill_M=128` exceed GB10's 99 KiB SMEM limit. |
| **`tp1` & `tp2-direct`** | **Greedy marginal-frequency knapsack layer slot allocation (`EMPIRICAL_BUDGET=1`)** | **MARGINAL / SKIP** | Ties uniform slots on `tp2` (`29.4 / 96.9 / 32.0 / 98.0`) and trims step time only `~1.5%` on `tp1` (`9.2 / 30.0 / 8.4 / 28.6`), while adding `+70..128 s` first-boot JIT compile overhead from heterogeneous per-layer `num_slots`. Uniform slots capture ~99% of the gain cleanly. |
| **`tp1` & `tp2-direct`** | **Dual-bank double-buffered cold-expert prefill (`DOUBLE_BUFFER_PREFILL=1`)** | **REJECT / MARGINAL** | After fixing stale `slot_to_expert` bank cleanup: `tp2` `32K` `+3.8%` (`4,771` vs `4,597`), `131K` `-1.0%` (`4,847` vs `4,897`); per-layer cached `tmp_out`/`acc_fp32` buffers (`~12.5 GiB` across 39 layers) tripped `tp1`'s 4 GiB bench memory floor (`2.08 GiB`). `0858` single-bank streaming already captures ~97% of prefill bandwidth cleanly. |
| **`tp2-direct`** | **`L+2` lookahead (`LOOKAHEAD_DEPTH=2`)** | **DEAD END** | **`-10%..-15%` decode regression** (`19.4 / 66.3 / 22.0 / 63.5`): predicting 2 layers ahead mispredicts too often, evicting warm dynamic slots and wasting NVMe bandwidth. |

---

## Detailed Measurements

### 1. `pp2-tp2-ring` (`dgx1..dgx4`, vs `manifests/benchmarks/2026-11-22-spark-ds41f-1122-pp2tp2ring.json`)

Full Sonar `screen` suite (`model` + `grid` + `c16` + `prefill 32K/262K`):
- `pp2-r2-a-sink-only` (`VLLM_SINK_REJECTION=1`): `outputs_sha256 = 78175f5d5dee...`, `rank1=1.0`, `ppl=1.3102`, `c16 = 177.15 ± 1.40% tok/s` (`ttft_p50 = 0.700 s`), `prefill 32K/262K = 5,905 / 6,327 tok/s` (1 prefill repeat failed).
- `pp2-r2-c-fast-p2p` (`VLLM_PP_FAST_P2P=1`): **bit-identical `outputs_sha256 = 78175f5d5dee...`**, `rank1=1.0`, `ppl=1.3102`, `0` failures, `c16 = 176.20 ± 1.82% tok/s` (**`ttft_p50 = 0.481 s`, `-31.3%`**), `model score_s = 26.12 s` (vs `29.42 s`, `-11.2%`), `prefill 32K/262K = 5,799 / 6,242 tok/s`.

Fast matrix (`--suites quality,decode,prefill --concurrency 1,2,4,8,16 --prefill-sizes 32768,131072`):

| Metric | `1122` Ref | `pp2-r2-b-async-av` (no fast P2P) | `pp2-r2-d-combined` (+ `VLLM_PP_FAST_P2P=1`) | Fast P2P Delta (`d` vs `b`) |
|---|---|---|---|---|
| Quality | `LRU 5/5` | `LRU 5/5` | **`LRU 5/5`** | — |
| `prose-c1` | `40.1` | `47.1` (`+17.3%`) | **`47.4` (`+18.2%`)** | `+0.6%` |
| `prose-c2` | `76.2` | `75.2` (`-1.3%`) | **`81.4` (`+6.8%`)** | **`+8.2%`** (`-4.65 ms/step`) |
| `prose-c4` | `123.3` | `130.8` (`+6.0%`) | **`146.0` (`+18.4%`)** | **`+11.6%`** (`-7.66 ms/step`) |
| `prose-c8` | `210.2` | `223.2` (`+6.2%`) | **`241.4` (`+14.8%`)** | **`+8.2%`** (`-6.05 ms/step`) |
| `prose-c16` | `385.9` | `409.1` (`+6.0%`) | **`398.9` (`+3.4%`)** | `-2.5%` (noise `±22%`) |
| `code-c1` | `50.3` | `54.1` (`+7.4%`) | **`54.0` (`+7.3%`)** | `-0.2%` |
| `code-c2` | `93.2` | `88.3` (`-5.2%` FAIL) | **`95.1` (`+2.1%` PASS)** | **`+7.7%`** (`-4.85 ms/step`) |
| `code-c4` | `148.2` | `149.1` (`+0.6%`) | **`153.6` (`+3.7%`)** | **`+3.0%`** |
| `code-c8` | `235.5` | `281.0` (`+19.3%`) | **`301.4` (`+28.0%`)** | **`+7.3%`** |
| `code-c16` | `445.5` | `487.0` (`+9.3%`) | **`498.3` (`+11.8%`)** | **`+2.3%`** |
| Prefill `32K` | `5,571.7` | `6,277.9` (`+12.7%`) | **`6,368.2` (`+14.3%`)** | `+1.4%` |
| Prefill `131K` | `5,727.5` | `6,664.1` (`+16.4%`) | **`6,624.4` (`+15.7%`)** | `-0.6%` |
| Min `MemAvailable` | `21.9–25.0 GiB` | `16.18–18.86 GiB` | **`16.13–19.24 GiB`** | `0.0 GiB` swap |

### 2. Multi-Domain Held-Out Calibration Corpus (`results/hotlist_multidomain_heldout.json`)

Captured via `scripts/calib_corpus.py` (`40` Sonar-disjoint prompts across 5 balanced domains: `3,072` tokens each of `prose`, `code`, `math`, `agent`, `ops` = `15,360` decode tokens = **`4,013,544` layer-expert routes across 40 layers**):
- Mean top-60 expert coverage (`tp1` static slots, 15.6% of experts): **`60.3%`**
- Mean top-188 expert coverage (`tp2` static slots, 49.0% of experts): **`91.7%`**

### 3. `tp2-direct` (`dgx1..dgx2`, `252/64`, `6.0 GiB` KV, `8K` batch, vs `1120-direct2`)

| Arm | `prose-c1` | `prose-c4` | `code-c1` | `code-c4` | Prefill `32K / 131K` | Min `MemAvailable` | Quality |
|---|---|---|---|---|---|---|---|
| `1120-direct2` baseline | `17.7` | `53.0` | `22.2` | `64.2` | `259 / 236` | — | `LRU 5/5` |
| `candidate-70` (`0858` + `rc=0.25/tau=0.45`) | `21.6` (`+22.0%`) | `73.0` (`+37.8%`) | `24.8` (`+11.7%`) | `72.5` (`+12.9%`) | `4,597 / 4,897` | `5.88 / 7.82 GiB` | `LRU 5/5` |
| `tp2-r2-a-fast-la` | `22.2` (`+25.5%`) | `74.2` (`+40.1%`) | `25.5` (`+14.9%`) | `69.8` (`+8.7%`) | `4,530 / 4,900` | `5.38 / 6.97 GiB` | `LRU 5/5` |
| `tp2-r2-b-la-depth2` (dead end) | `19.4` (`+9.6%`) | `66.3` (`+25.2%`) | `22.0` (`-0.7%`) | `63.5` (`-1.1%`) | `4,516 / 4,906` | `5.40 / 7.25 GiB` | `LRU 5/5` |
| `tp2-r2-c-dblbuf-fixed` (prefill only) | — | — | — | — | `4,771 / 4,847` | `5.04 / 7.14 GiB` | `LRU 5/5` |
| **`tp2-r2-d-heldout-hotlist`** | **`29.4` (`+66.3%`)** | **`95.5` (`+80.3%`)** | **`32.9` (`+48.4%`)** | **`99.7` (`+55.3%`)** | `4,381 / 4,894` | `5.43 / 7.23 GiB` | `LRU 5/5` |
| `tp2-r2-e-empirical-budget` (`+128 s` JIT) | `29.4` (`+66.3%`) | `96.9` (`+83.0%`) | `32.0` (`+44.3%`) | `98.0` (`+52.6%`) | `4,666 / 4,774` | `5.26 / 6.83 GiB` | `LRU 5/5` |

### 4. `tp1` (`dgx1`, `108/48`, `6.0 GiB` KV, `8K` batch, vs `1120-tp1`)

| Arm | `prose-c1` | `prose-c4` | `code-c1` | `code-c4` | Prefill `1K / 8K / 32K` | Min `MemAvailable` | Quality |
|---|---|---|---|---|---|---|---|
| `1120-tp1` baseline | `4.4` | `10.9` | `4.8` | `11.4` | `24.4 / — / —` | — | `LRU 5/5` |
| `candidate-70` (`0858` + `108/48`) | `6.7` (`+53.1%`) | `15.3` (`+40.2%`) | `6.6` (`+37.7%`) | `19.4` (`+71.2%`) | `333 / 1,229 / 2,114` | `9.06 GiB` | `LRU 5/5` |
| `tp1-r2-a-fast-la-dblbuf` (decode before dblbuf abort) | `6.4` (`+45.2%`) | `20.9` (`+90.8%`) | `6.3` (`+32.2%`) | `20.7` (`+82.5%`) | *(aborted on 4 GiB mem floor: `2.08 GiB`)* | `5.33 / 2.08 GiB` | `LRU 5/5` |
| **`tp1-r2-b-heldout-hotlist`** | **`9.2` (`+108.8%`)** | **`30.0` (`+174.2%`)** | **`8.4` (`+75.4%`)** | **`28.5` (`+150.8%`)** | `243 / 1,532 / 2,116` | `8.51 GiB` | `LRU 5/5` |
| `tp1-r2-c-empirical-budget` (`+70 s` JIT) | `9.2` (`+109.1%`) | `30.0` (`+174.4%`) | `8.4` (`+76.1%`) | `28.6` (`+151.8%`) | `243 / 1,539 / 2,128` | `8.34 GiB` | `LRU 5/5` |

### 5. TileLang MoE Tile Schedule Microbenchmark (`dgx4` GB10)

| Shape / Config | `r=6` | `r=12` | `r=24` | `r=192` | `r=4096` | Notes |
|---|---|---|---|---|---|---|
| `TP=2` (`inter=1152, narrow=True`) `s3/2` (current) | `1.228 ms` | `2.266 ms` | `4.650 ms` | `13.905 ms` | `18.730 ms` | baseline |
| **`TP=2` (`inter=1152, narrow=True`) `down_stages=3`** | **`1.187 ms` (`-3.3%`)** | **`2.231 ms` (`-1.5%`)** | **`4.581 ms` (`-1.5%`)** | **`13.743 ms` (`-1.2%`)** | **`18.416 ms` (`-1.7%`)** | strictly faster across all rows |
| `TP=2` `stages=4/3`, `4/4`, `prefill_M=128` | ERROR | — | — | — | — | exceeds GB10 99 KiB SMEM |
| `TP=3` (`inter=768, narrow=True`) `s3/2` (current) | `0.866 ms` | `1.737 ms` | `3.163 ms` | `9.231 ms` | `13.735 ms` | baseline |
| **`TP=3` (`inter=768, narrow=True`) `down_stages=3`** | **`0.844 ms` (`-2.5%`)** | **`1.714 ms` (`-1.3%`)** | **`3.149 ms` (`-0.4%`)** | **`9.057 ms` (`-1.9%`)** | **`13.641 ms` (`-0.7%`)** | strictly faster across all rows |
| `TP=1` (`inter=2304, slots=108`) `s3/2` vs `s3/3` | `2.430` vs `2.427` | — | `6.269` vs `6.245` | — | `19.285` vs `19.284` | flat/slightly faster |
