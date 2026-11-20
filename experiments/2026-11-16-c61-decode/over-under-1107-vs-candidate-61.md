# `1107` vs `candidate-61` Over/Under Scorecard (By Cluster Type)

> **Build tested**: `spark-ds41f:candidate-61` (`sha256:44e8aa6bc4d6c2445eb38aafa3a05ea8d710b9bcbb69a2f4acc6c7f73dcc39ad`, deployment commit `37a270e71`, `release-tests`: `11,918 passed, 0 failed`).
>
> **Run standard**: Every run below executes immediately after a fresh machine reboot on every participating node, passes `5/5` quality checks on every boot (`0` failed requests, `0.0 ms` thermal slowdown), stays above the `5.5 GiB` startup `MemAvailable` floor, and cools nodes between suites/steps and before every prefill sample.
>
> **Snapshot timestamp**: `2026-11-18T23:02Z` (`2026-11-19 00:02 CET`).
>
> **Baseline**: [2026-11-07-spark-ds41f-1107.json](../../manifests/baselines/2026-11-07-spark-ds41f-1107.json) and its immutable suite/benchmark receipts in `manifests/suites/2026-11-07-spark-ds41f-1107-*` and `manifests/benchmarks/2026-11-07-spark-ds41f-1107*.json`.

---

## Executive Summary Across Cluster Types

| Cluster Shape | Status | KV Pool (`1107 -> c61`) | Headline Decode (`1107 -> c61`) | Headline Cold Prefill (`32K..1M`) | Overall Verdict |
|---|---|---|---|---|---|
| **`tp4-ring`** (`dgx1..4`) | **2-Boot COMPLETE** | `10.50 -> 16.50 GiB` (`+57.1%`) | `grid -7.96%`; `c1 +6.78%` paired (`82.62 -> 88.94`); `c8 +2.25%` (`207.45 -> 213.59`); `c16 +2.48%` (`272.86 -> 280.62`); `sampled +2.50%..+8.70%` | `32K +13.0%`, `131K +16.8%`, `262K +21.7%`, `524K +31.8%`, `1M +56.3%` (`9,394` vs `6,012`) | **ALL OVER** |
| **`tp4-switched`** (`sp1..4`) | **Boot 1 bench + release (`stall-15`) DONE; Boot 2 IN PROGRESS** | `10.50 -> 16.50 GiB` (`+57.1%`) | Boot 1 `grid -9.89%`; `c8 +6.98%`, `c16 +6.58%` paired; `bench` 10/10 points **`+10.3%..+30.5%`** | Boot 1 `bench`: `32K +14.2%`, `131K +16.3%`, `262K +21.3%`, `524K +31.6%`, `1M +56.4%` (`9,330` vs `5,966`) | **ALL OVER** *(Boot 1)* |
| **`tp3-switched`** (`sp5..7`) | **2-Boot `bench` + `1m` & Boot 1 `servebench` DONE; Boot 2 `servebench` IN PROGRESS (`cool_below=49`)** | `4.906 -> 4.906 GiB` (tie) | Live `servebench` Boot 1: `c1 +12.17%`, `c4 +6.16%`, `c8 +7.35%`, `sampled +6.23%..+11.28%`; `bench` 2-boot: `5` **OVER** (`+4.6%..+18.7%`), `3` **UNDER** (`-1.2%..-9.7%`); `grid` `1x*` **+76%..+79% slower** | `32K 7,250–7,281` (`+12.4%..+12.8%`), `1M 6,885–6,974` (`+55.3%..+57.3%` vs `1107 tp3-triangle`) | **MIXED** *(live serving OVER; 2 synthetic points UNDER)* |
| **`tp5-switched`** (`sp1..4, sp8`) | **2-Boot COMPLETE** | `10.50 -> 24.00 GiB` (`+128.6%`) | 2-boot `bench` 10/10 points **OVER**: `prose +3.8%..+23.3%`, `code +2.9%..+13.5%` | `10,583 / 10,930 / 10,906 / 10,697 / 10,285` tok/s (`32K..1M`, new record) | **ALL OVER** |
| **`tp2-direct`** (`dgx1..2`) | **2-Boot COMPLETE** | *OOM ->* `8.00 GiB` (`256K` ctx) | `prose-c1/c2/c4 = 17.72 / 28.98 / 53.00`, `code-c1/c2/c4 = 22.18 / 34.42 / 64.26` | `32K 258.9`, `131K 235.7`, `262K 244.5` tok/s | **NEW (OVER)** |
| **`tp2-switched`** (`sp5..6`) | **QUEUED** (after `tp3-switched` Boot 2) | *OOM ->* `8.00 GiB` (`256K` ctx) | Pending | Pending | **PENDING** |
| **`tp6-switched`** (`sp1..6`) | **QUEUED** (after `sp1..8` join) | `10.50 -> 14.00 GiB` (`+33.3%`) | `1107` bar: `98.47..637.93` prose, `107.74..782.80` code | Pending (`32K..1M`) | **PENDING** |
| **`tp7-switched`** (`sp1..7`) | **QUEUED** | `10.50 -> 18.00 GiB` (`+71.4%`) | `1107` bar: `98.95..601.03` prose, `106.21..740.40` code | Pending (`32K..1M`) | **PENDING** |
| **`tp8-switched`** (`sp1..8`) | **QUEUED** | `10.50 -> 28.00 GiB` (`+166.7%`) | `1107` bar: `bench` + `release` / `single-stream` / `sampled` | Pending (`32K..1M`) | **PENDING** |
| **`tp3-triangle`** & **`-4k`** (`dgx1..3`) | **AWAITING RECABLE** | `4.906` / `2.200 GiB` | Profiles validated (`doctor`/`lint` clean); physically ring-cabled currently | Pending physical recable | **WAITING** |

---

## Regressions & Watchlist Items Observed So Far

Only **`tp3-switched`** exhibits any negative deltas against `1107`, isolated to two synthetic measurements while real end-to-end serving suites on `tp3-switched` are strongly positive:

1. **`tp3-switched` `servebench engine` (`grid`) single-row steps (`1x2 / 1x3 / 1x6`)**:
   - In the isolated `grid` microbench (`tp3-switched` Boot 1), all `1x*` single-request shapes regressed from `18.74–19.60 ms` (`1107` trimmed mean; `18.94–19.75 ms` raw mean) to `33.40–34.42 ms` (`33.60–34.53 ms` raw mean in Boot 1/Boot 2, `+75.6%..+79.0%` slower step latency), lifting the 15-shape geometric mean to **`+22.20%`** worse despite multi-request shapes (`2x12..8x48`) being **`-0.58%..-9.86%` faster**.
   - **Root cause (`drafter_ms` at `num_reqs == 1` on `TP3` dummy padding steps)**: Inspecting `grid.json` (`1107` and `c60` vs `c61` Boot 1/Boot 2 on `sp5`) shows target `forward_ms` actually *improved* by `7.3%–7.8%` on every `1x*` shape (`1x2 @4K`: `14.632 -> 13.502 / 13.544 ms`; `1x3 @4K`: `14.763 -> 13.681 / 13.797 ms`; `1x6 @4K`: `15.435 -> 14.245 / 14.283 ms`; `1x3 @1016K`: `14.782 -> 13.626 ms`), and multi-request `drafter_ms` (`2x12..8x48`) also improved (`2.942–3.781 ms -> 2.800–3.573 ms`). The entire `+15.9 ms` jump is inside `drafter_ms` **strictly when `num_reqs == 1` on `TP3`** (`2.84–2.87 ms` in `1107` and `2.81 ms` in `c60` vs `18.73–19.02 ms` in `c61` Boot 1/Boot 2, whereas `tp4-ring` and `tp4-switched` `1x2 @4K` `drafter_ms` is `1.97 ms`). In `c61` ([0820](vllm/0820-perf-spec-decode-optimize-DSpark-margin-confidence-h.patch#L3673-L3686), [0819](vllm/0819-perf-dsv41-tune-vocab-head-grid-width-and-packed-BF1.patch#L385-L400)), `use_screened_markov` activates exclusively for `num_reqs == 1` greedy drafting, routing into `screen_head_base` and persistent `block_N=8, grid_x=40` Markov refinement (`project_head_screened_markov_greedy_pairs`); on default `engine_grid` dummy padding rows (`real_rows=False`, null KV slot per `servebench/engine.py`, `TP3` `64 -> 72` padded attention heads), screening falls back to full-shard persistent refinement across the 5 draft steps rather than pruning.
   - **Live `c1` serving is unaffected**: With real finite KV slots on `tp3-switched`, `prose-c1` total `step_ms` (target + MoE + drafter) drops from `36.79 ms` to `31.36 ms` (`-14.8%`), `code-c1` drops from `39.31 ms` to `34.85 ms` (`-11.3%`), `servebench single-stream c1` improves **`+12.17%`** paired (`62.27 -> 68.50` tok/s; `26.33 -> 28.73` steps/s, `+9.11%`), and `sampled c1` improves **`+11.28%`** paired (`61.64 -> 68.40` tok/s). On `tp4-ring` and `tp4-switched`, `grid` `1x*` shapes improved by `-8.6%` to `-16.3%`.

2. **`tp3-switched` synthetic identical-prompt `bin/spark bench` decode (`3` of `8` points)**:
   - Under `bin/spark bench` (2-boot average), **`code-c8`** dropped `-9.69%` (`378.23 -> 341.57` tok/s), **`prose-c4`** dropped `-4.25%` (`205.77 -> 197.02` tok/s), and **`prose-c1`** was slightly under (`-1.23%`, `62.38 -> 61.61` tok/s), while `prose-c2` (`+15.49%`), `prose-c8` (`+4.63%`), `code-c1` (`+6.14%`), `code-c2` (`+15.30%`), and `code-c4` (`+18.66%`) were **OVER**.
   - **Root cause (stricter `TP3` adaptive verification row cost vs `distinct_outputs: 1` identical prompts)**: [tp3-switched.json](tp3-switched.json#L32) raises `adaptive_verification_row_cost` (`0.035 -> 0.049`) and `single_req_row_cost_scale` (`1.79`, effective `c1` row cost `0.0877`) to trim speculative rows for real distinct-prompt MoE expert loads on 3 ranks. Because `bin/spark bench` runs **`distinct_outputs: 1`** (identical prompt copies across every stream), trimming rows yields little or no MoE expert-weight savings in those 3 cases while lowering `tokens_per_step`:
     - **`code-c8`**: All 8 streams repeat the exact same code prompt/tokens and hit the same subset of MoE experts, so cutting `verified_per_draft` (`2.988 -> 2.469` / `2.457`) lowers `tokens_per_step` (`2.885 -> 2.666` / `2.669`, `-7.5%`) while `step_ms` stays flat/slightly higher (`57.18 -> 57.24 / 58.23 ms`). On 8 *distinct* prompts in `servebench c8`, trimming rows (`26.00 -> 23.01` rows/step, `wasted 7.87 -> 5.62`) speeds up engine step rate **`+11.82%`** (`8.687 -> 9.713 steps/s`), winning **`+7.35%` paired (`+7.29%` output, `157.29 -> 168.75` tok/s, OVER)**.
     - **`prose-c4` & `prose-c1`**: At `prose-c4` (4 identical prompts), cutting `verified_per_draft` (`2.317 -> 1.623 / 1.618`) drops `tokens_per_step` `-18.8%` (`2.302 -> 1.869`) against a `-15.0%` `step_ms` drop (`41.81 -> 35.54 / 35.51 ms`), whereas distinct prompts in `servebench c4` (`13.12 -> 11.62` rows/step) gain `+11.73%` step rate (`13.387 -> 14.957 steps/s`) for **`+6.16%` paired (`+6.98%` output, `121.43 -> 129.91` tok/s, OVER)**. At `prose-c1`, `single_req_row_cost_scale=1.79` trims that single fixed prompt (`verified_per_draft` `2.826 -> 2.098`, `tokens_per_step` `-15.9%` vs `step_ms` `-14.8%` `36.79 -> 31.36 ms`), whereas across `servebench single-stream c1`'s diverse prompts (`3.74 -> 3.57` rows/step), `tokens_per_step` rises `+1.99%` (`2.371 -> 2.418`) and step rate rises `+9.11%` (`26.333 -> 28.733 steps/s`) for **`+12.17%` paired (`+11.27%` output, `62.35 -> 69.38` tok/s, OVER)** (`1107` single-request `c1` vs `c61`).

3. **`tp3-switched` Boot 2 thermal cooling margin (`sp6`)**:
   - On `sp5..sp7` (stock fan curve; no `dgx-fan-control.service`), the initial Boot 2 `release-tp3` run halted at `1000000:r3` when `sp6` hit the `83 °C` guard (`thermal_slowdown_ms = 0.0`). Lowering inter-sample `cool_below` from `52 °C` to `49 °C` kept `sp6 <= 79.9 °C` across continuous `1M` prefills; the clean Boot 2 `release-tp3` run is underway now.

---

## 1. `tp4-ring` (`dgx1..dgx4`) — 2-Boot COMPLETE

### 1a. `servebench` Release Qualification (`2-Boot`: `c61` & `c61-again` vs `1107-tp4`)

| Suite / Metric | `1107-tp4` | `c61` Boot 1 | `c61` Boot 2 | Paired / Mean Delta | Repeat Drift | Verdict |
|---|---:|---:|---:|---|---|---|
| KV Cache / Rank | `10.50 GiB` | `16.50 GiB` | `16.50 GiB` | `+6.00 GiB (+57.1%)` | — | **OVER** |
| Min Startup `MemAvail` (head `dgx1`) | — | `17.54 GiB` | `17.62 GiB` | `>= 5.5 GiB` (`19.35..19.59` workers) | — | **PASS** |
| `release` `grid` geomean (20 shapes) | — | `-8.02%` | `-7.90%` | **`-7.96% ±0.11%`** (all 20 shapes faster) | `+0.12%` | **OVER** |
| `release c16` output tok/s | `272.86` | `280.43` | `280.80` | `+2.77% ±1.05%` (`+2.48% ±1.15%` paired) | `+0.13%` (`+0.07%` paired) | **OVER** |
| `release c8` output tok/s | `207.45` | `213.57` | `213.60` | `+2.95% ±1.39%` (`+2.25% ±1.15%` paired) | `+0.02%` (`+0.10%` paired) | **OVER** |
| `release poisson` output / paired dec | `180.17 / 31.72` | `180.29 / 33.62` | `180.25 / —` | `+0.07%` output / **`+6.38% ±1.30%`** paired | `-0.03%` | **SAME / OVER** |
| `single-stream c1` output tok/s | `82.62` | `89.06` | `88.81` | **`+7.79% ±4.83%`** (`+6.78%` paired) | `-0.28%` (`-0.16%` paired) | **OVER** |
| `sampled c16` output tok/s | `268.31` | `276.58` | `275.98` | `+3.08% ±1.05%` (`+2.50%` paired) | `-0.22%` | **OVER** |
| `sampled c8` output tok/s | `203.78` | `210.02` | `209.00` | `+3.06% ±1.00%` (`+2.79%` paired) | `-0.48%` | **OVER** |
| `sampled c1` output tok/s | `79.99` | `87.07` | `87.01` | **`+8.85% ±3.69%`** (`+8.70%` paired) | `-0.08%` | **OVER** |
| `stall-3` prefill (`131K`) under dec | `6,804.1` | `7,733.6` | `7,779.2` | **`+13.66% ±2.02%`** (TTFT p50 `1.017->0.726/0.626s`) | `+0.59%` | **OVER** |
| `stall-11` prefill (`131K`) under dec | `6,386.5` | `7,157.5` | `7,218.3` | **`+12.07% ±1.00%`** (TTFT p50 `1.439->1.140/1.241s`) | `+0.85%` | **OVER** |
| `stall-15` prefill (`131K`) under dec | `6,893.8` | `7,851.5` | `7,866.1` | **`+13.89% ±2.54%`** (worst wait `-13.99%`) | `+0.19%` | **OVER** |
| `release prefill 32,768` tok/s | `8,508.0` | `9,614.6` | `9,628.0` | **`+13.01% ±1.86%`** | `+0.14%` | **OVER** |
| `release prefill 131,072` tok/s | `8,657.2` | `10,108.5` | `10,103.4` | **`+16.76% ±0.55%`** | `-0.05%` | **OVER** |
| `release prefill 262,144` tok/s | `8,260.6` | `10,056.2` | `10,059.1` | **`+21.74% ±0.24%`** | `+0.03%` | **OVER** |
| `release prefill 524,288` tok/s | `7,459.6` | `9,833.1` | `9,835.6` | **`+31.82% ±0.19%`** | `+0.03%` | **OVER** |
| `release prefill 1,000,000` tok/s | `6,012.2` | `9,397.1` | `9,391.1` | **`+56.30% ±0.64%`** | `-0.06%` | **OVER** |

### 1b. `tp4-ring` 2-Boot `bin/spark bench` Matrix (`c61-bench-tp4-c61-b{1,2}` & `-1m`)

| Metric | `c61` Boot 1 | `c61` Boot 2 | 2-Boot Mean |
|---|---:|---:|---:|
| `prose-c1 / c2 / c4 / c8 / c16` | `84.94 / 150.39 / 266.94 / 453.78 / 657.35` | `84.88 / 150.35 / 266.97 / 455.31 / 671.59` | `84.91 / 150.37 / 266.96 / 454.55 / 664.47` |
| `code-c1 / c2 / c4 / c8 / c16` | `105.12 / 186.00 / 329.02 / 560.88 / 873.94` | `104.96 / 185.99 / 328.27 / 561.01 / 873.93` | `105.04 / 186.00 / 328.65 / 560.95 / 873.94` |
| `bench prefill 32K / 131K / 262K` | `9,762.9 / 10,097.1 / 10,068.1` | `9,695.1 / 10,046.1 / 10,032.7` | `9,729.0 / 10,071.6 / 10,050.4` |
| `bench prefill 524K / 1M` (`-1m`) | `9,874.1 / 9,457.6` | `9,870.4 / 9,470.3` | `9,872.3 / 9,464.0` |

---

## 2. `tp4-switched` (`sp1..sp4`) — Boot 1 (`bench` + `release` through `stall-15`) COMPLETE

### 2a. `tp4-switched` Boot 1 `servebench release` vs `1107-tp4-switched`

| Metric | `1107-tp4-switched` | `c61` Boot 1 | Delta vs `1107` | Verdict |
|---|---:|---:|---|---|
| KV Cache / Rank | `10.50 GiB` | `16.50 GiB` | `+6.00 GiB (+57.1%)` | **OVER** |
| Min Startup `MemAvail` (`sp1 / sp2..4`) | — | `15.46 / 17.06–17.09 GiB` | `>= 5.5 GiB` | **PASS** |
| `grid` geomean (20 shapes) | — | all 20 shapes `-1.52%..-16.27%` | **`-9.89% ±0.15%`** | **OVER** |
| `release c16` output / paired dec | `262.47 / 16.37` | `280.51 / 17.43` | **`+6.87%` output / `+6.58%` paired** | **OVER** |
| `release c8` output / paired dec | `199.52 / 25.24` | `213.06 / 26.67` | **`+6.79%` output / `+6.98%` paired** | **OVER** |
| `release poisson` output / paired dec | `180.16 / 29.04` | `180.26 / 33.61` | `+0.06%` output / **`+15.67%` paired** | **SAME / OVER** |
| `stall-3` prefill (`131K`) under dec | `6,659.4` | `7,653.9` | **`+14.93% ±2.05%`** (worst wait `-20.13%`) | **OVER** |
| `stall-11` prefill (`131K`) under dec | `6,367.0` | `7,123.2` | **`+11.88% ±0.95%`** (TTFT p50 `-20.72%`) | **OVER** |
| `stall-15` prefill (`131K`) under dec | `6,808.8` | `7,765.7` | **`+14.05% ±3.93%`** (worst wait `-17.19%`) | **OVER** |
| `prefill`, `single-stream`, `sampled`, Boot 2 | *(recorded)* | Running now | Pending | **IN PROGRESS** |

### 2b. `tp4-switched` Boot 1 `bin/spark bench` vs `1107-switched`

| Workload | `1107` | `c61` Boot 1 | Delta vs `1107` | Verdict |
|---|---:|---:|---|---|
| `prose-c1 / c2 / c4` | `73.34 / 128.39 / 212.76` | `84.48 / 150.38 / 267.89` | **`+15.19% / +17.13% / +25.91%`** | **OVER** |
| `prose-c8 / c16` | `396.10 / 587.04` | `458.22 / 663.07` | **`+15.68% / +12.95%`** | **OVER** |
| `code-c1 / c2 / c4` | `94.96 / 149.85 / 266.86` | `104.76 / 186.41 / 329.16` | **`+10.32% / +24.40% / +23.35%`** | **OVER** |
| `code-c8 / c16` | `490.12 / 677.32` | `568.28 / 884.08` | **`+15.95% / +30.53%`** | **OVER** |
| `prefill 32K / 131K / 262K` | `8,471.5 / 8,612.6 / 8,209.2` | `9,672.7 / 10,020.4 / 9,958.3` | **`+14.18% / +16.35% / +21.31%`** | **OVER** |
| `prefill 524K / 1,000K` | `7,417.0 / 5,965.5` | `9,762.9 / 9,330.0` | **`+31.63% / +56.40%`** | **OVER** |

---

## 3. `tp3-switched` (`sp5..sp7`) — 2-Boot `bench` & Boot 1 `servebench` COMPLETE

> `1107` did not record a completed `prefill` step on `tp3-switched` (`sp6` halted thermally in `1107`), so `tp3-switched` cold prefill is also shown against `1107-tp3` (`tp3-triangle`, identical `4.906 GiB` KV and `6,144` batched-token budget). Boot 2 `release-tp3` is currently re-running cleanly with `cool_below=49 °C`.

### 3a. `tp3-switched` Boot 1 `servebench` vs `1107-tp3-switched`

| Suite / Metric | `1107-tp3-sw` | `c61` Boot 1 | Delta vs `1107` | Verdict |
|---|---:|---:|---|---|
| KV Cache / Rank | `4.906 GiB` | `4.906 GiB` | Same (`5.23x` 1M) | **TIE** |
| Min Startup `MemAvail` (`sp5 / sp6,sp7`) | — | `6.81 / 8.36, 8.04 GiB` (`b2`: `6.84 / 8.34, 8.05`) | `>= 5.5 GiB` floor | **PASS** |
| `release-tp3` `grid` (`1x*` 5 shapes) | `18.74–19.60 ms` | `33.40–34.42 ms` | **`+75.6%..+79.0%` slower** | **UNDER** |
| `release-tp3` `grid` (10 multi-row shapes) | `20.49–28.81 ms` | `18.61–27.99 ms` | `-0.58%..-9.86%` (`8` faster, `2` same) | **OVER** |
| `release-tp3` `grid` geomean (15 shapes) | — | — | **`+22.20% ±0.21%`** *(pulled by `1x*`)* | **UNDER** |
| `release-tp3 c8` output / paired dec | `157.29 / 19.86` | `168.75 / 21.00` | **`+7.29%` output / `+7.35%` paired** | **OVER** |
| `release-tp3 c4` output / paired dec | `121.43 / 30.04` | `129.91 / 32.22` | **`+6.98%` output / `+6.16%` paired** | **OVER** |
| `release-tp3 poisson` output / paired | `118.96 / 27.73` | `119.59 / 31.92` | `+0.53%` output / **`+12.53%` paired** | **SAME / OVER** |
| `single-stream c1` output / paired | `62.35 / 62.27` | `69.38 / 68.50` | **`+11.27%` output / `+12.17%` paired** | **OVER** |
| `sampled c16 / c8 / c1` paired decode | `19.41 / 19.40 / 61.64` | `20.44 / 20.55 / 68.40` | **`+6.23% / +7.11% / +11.28%`** | **OVER** |
| `stall-3 / 6 / 7` prefill (`131K`) under dec | `4,663 / 4,329 / 4,633` | `5,271 / 4,945 / 5,308` | **`+13.04% / +14.24% / +14.58%`** | **OVER** |
| `stall-3 / 6 / 7` worst decode wait (s) | `1.100 / 1.176 / 1.174` | `0.964 / 1.044 / 1.037` | **`-12.36% / -11.25% / -11.74%`** | **OVER** |
| `prefill 32K..1M` (`suite` Boot 1 vs `1107-tp3`) | `6,452 / 6,480 / 6,135 / 5,514 / 4,434` | `7,250 / 7,482 / 7,412 / 7,230 / 6,885` | **`+12.4% / +15.5% / +20.8% / +31.1% / +55.3%`** | **OVER** *(vs tri)* |

### 3b. `tp3-switched` 2-Boot `bin/spark bench` vs `1107-switched3`

| Workload | `1107-switched3` | `c61` Boot 1 | `c61` Boot 2 | 2-Boot Mean | Delta vs `1107` | Verdict |
|---|---:|---:|---:|---:|---|---|
| `prose-c1` | `62.38` | `61.62` | `61.61` | `61.61` | `-1.23%` | **UNDER** *(synthetic; `servebench c1 +12.17%`)* |
| `prose-c2` | `95.30` | `110.04` | `110.08` | `110.06` | **`+15.49%`** | **OVER** |
| `prose-c4` | `205.77` | `197.07` | `196.97` | `197.02` | `-4.25%` | **UNDER** *(synthetic; `servebench c4 +6.16%`)* |
| `prose-c8` | `354.56` | `371.31` | `370.67` | `370.99` | **`+4.63%`** | **OVER** |
| `code-c1` | `73.15` | `77.54` | `77.74` | `77.64` | **`+6.14%`** | **OVER** |
| `code-c2` | `121.64` | `140.40` | `140.09` | `140.25` | **`+15.30%`** | **OVER** |
| `code-c4` | `218.77` | `258.82` | `260.38` | `259.60` | **`+18.66%`** | **OVER** |
| `code-c8` | `378.23` | `344.45` | `338.69` | `341.57` | `-9.69%` | **UNDER** *(synthetic; `servebench c8 +7.35%`)* |
| `bench prefill 32K / 131K / 262K` | *(none)* | `7,253.1 / 7,482.5 / 7,441.8` | `7,307.8 / 7,486.6 / 7,465.4` | `7,280.5 / 7,484.6 / 7,453.6` | *(+12.8%..+21.5% vs tri)* | **OVER** *(vs tri)* |
| `bench prefill 524K / 1M` | *(none)* | `7,286.2 / 6,984.7` | `7,259.8 / 6,963.7` | `7,273.0 / 6,974.2` | *(+31.9%..+57.3% vs tri)* | **OVER** *(vs tri)* |

---

## 4. `tp5-switched` (`sp1..sp4, sp8`) — 2-Boot COMPLETE

Measured on `sp1, sp2, sp3, sp4, sp8` (`config/nodes-switched5.local.json` rank-4 remapped cleanly while `sp5..sp7` ran `tp3-switched`), `10/10` points **OVER** `1107-switched5`.

| Workload | `1107-switched5` | `c61` Boot 1 | `c61` Boot 2 | 2-Boot Mean | Delta vs `1107` | Verdict |
|---|---:|---:|---:|---:|---|---|
| KV Cache / Rank | `10.50 GiB` | `24.00 GiB` | `24.00 GiB` | `24.00 GiB` | `+13.50 GiB (+128.6%)` | **OVER** |
| Min Startup `MemAvail` (`sp1 / sp8`) | — | `11.02 / 14.35 GiB` | `11.07 / 14.15 GiB` | `>= 5.5 GiB` | — | **PASS** |
| `prose-c1` | `83.43` | `86.17` | `87.08` | `86.62` | **`+3.82%`** | **OVER** |
| `prose-c2` | `144.96` | `151.45` | `149.97` | `150.71` | **`+3.97%`** | **OVER** |
| `prose-c4` | `247.60` | `282.95` | `282.51` | `282.73` | **`+14.19%`** | **OVER** |
| `prose-c8` | `392.29` | `482.39` | `485.09` | `483.74` | **`+23.31%`** | **OVER** |
| `prose-c16` | `604.55` | `730.77` | `727.58` | `729.17` | **`+20.61%`** | **OVER** |
| `code-c1` | `91.10` | `103.82` | `102.93` | `103.38` | **`+13.48%`** | **OVER** |
| `code-c2` | `156.16` | `171.22` | `169.64` | `170.43` | **`+9.14%`** | **OVER** |
| `code-c4` | `279.91` | `317.11` | `317.49` | `317.30` | **`+13.36%`** | **OVER** |
| `code-c8` | `500.52` | `518.98` | `511.39` | `515.18` | **`+2.93%`** | **OVER** |
| `code-c16` | `745.84` | `801.18` | `800.73` | `800.95` | **`+7.39%`** | **OVER** |
| `prefill 32K / 131K / 262K` | *(none)* | `10,647.4 / 10,929.1 / 10,936.7` | `10,518.2 / 10,931.3 / 10,874.9` | `10,582.8 / 10,930.2 / 10,905.8` | — | **NEW RECORDED** |
| `prefill 524K / 1,000K` | *(none)* | `10,706.5 / 10,317.5` | `10,686.9 / 10,252.0` | `10,696.7 / 10,284.8` | — | **NEW RECORDED** |

---

## 5. `tp2-direct` (`dgx1..dgx2`) — 2-Boot COMPLETE & `tp2-switched` (`sp5..sp6`) — QUEUED

New `candidate-61` capability (`1107` could not boot on 2 nodes due to `232 GiB` model footprint); `8.00 GiB` KV pool, `262,144` max context, adaptive verification disabled.

| Metric | `1107` | `c61 tp2-direct` (`b1 / b2` -> Mean) | `c61 tp2-switched` | Verdict |
|---|---:|---:|---:|---|
| Min Startup `MemAvail` | *OOM* | `9.96 / 9.89 GiB` (`dgx1`), `11.96 / 11.96 GiB` (`dgx2`) | *Queued* | **NEW (OVER)** |
| `prose-c1 / c2 / c4` | — | `17.70/17.73 -> 17.72`, `28.95/29.00 -> 28.98`, `52.96/53.04 -> 53.00` | *Queued* | **NEW** |
| `code-c1 / c2 / c4` | — | `22.17/22.19 -> 22.18`, `34.39/34.45 -> 34.42`, `64.20/64.32 -> 64.26` | *Queued* | **NEW** |
| `prefill 32K / 131K / 262K` | — | `258.8/259.0 -> 258.9`, `235.6/235.7 -> 235.7`, `244.5/244.5 -> 244.5` | *Queued* | **NEW** |

---

## 6. Pending `1107` Comparison Bars (`tp6..tp8-switched`, `tp3-triangle`, `tp3-triangle-4k`)

| Shape | KV Target (`1107 -> c61`) | `1107` `prose-c1 / c16` (`bin/spark`) | `1107` `code-c1 / c16` (`bin/spark`) | `1107` `servebench` Bars | Execution Status |
|---|---|---|---|---|---|
| `tp6-switched` (`sp1..6`) | `10.50 -> 14.00 GiB` | `98.47 / 637.93` | `107.74 / 782.80` | — | Queued after `sp1..8` join (`bench 32K..1M`) |
| `tp7-switched` (`sp1..7`) | `10.50 -> 18.00 GiB` | `98.95 / 601.03` | `106.21 / 740.40` | — | Queued after `tp6-switched` (`bench 32K..1M`) |
| `tp8-switched` (`sp1..8`) | `10.50 -> 28.00 GiB` | `112.71 / 658.91` | `132.68 / 750.89` | `release c16=370.58`, `c8=289.84`; `single c1=115.47`; `sampled c1/c8/c16=112.73/286.58/363.38`; `prefill 32K..1M` `11,176..8,751` | Queued after `tp7-switched` (`bench + full suites`) |
| `tp3-triangle` (`dgx1..3`) | `4.906 -> 4.906 GiB` | `64.28` (`c1` suite) | *(after recable)* | `c4=123.73`, `c8=160.80`; `prefill 32K..1M` `6,452..4,434` | Awaiting physical `dgx1..3` recable |
| `tp3-triangle-4k` (`dgx1..3`) | `2.200 -> 2.200 GiB` | *(256K ctx)* | *(after recable)* | — | Awaiting physical `dgx1..3` recable |
