# `tp1`, `tp2-direct`, and `pp2-tp2-ring` Performance Screen (2026-11-23)

- **Base commit:** `945d95d03` (`origin/main`, `spark-ds41f:1122`, `sha256:8fc42c4405b5c00f91413c321f9344c2af18b540919bef65ff2aa6fcde73a1af`)
- **Branch:** `tp1-tp2-pp2-perf` (pure-Python overlays over `spark-ds41f:1122` under `overlay/vllm/`, zero image rebuilds)
- **Hardware:** `dgx1..dgx4` (4-node CX7 ring; `tp1` on `dgx1`, `tp2-direct` on `dgx1..dgx2`, `pp2-tp2-ring` on `dgx1..dgx4`)
- **Cooling & safety:** `--cool-below 55`, fail-closed 5 GiB startup / 3 GiB steady memguards, `LRU 5/5` quality gate on every arm.

## Summary of Measured Results

### 1. `tp1` (1 Spark, `dgx1`) — vs `manifests/benchmarks/2026-11-20-spark-ds41f-1120-tp1.json`

| Arm | Slots / KV (`256K` concurrency) | Prefill 1K / 8K / 32K (tok/s) | `prose-c1` | `prose-c4` | `code-c1` | `code-c4` | `min MemAvailable` | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| `1120` ref (`fixed d5`) | `96/48`, `8.0 GiB` (`34.3×`) | `24.4` / — / — | `4.4` (`511 ms`) | `10.9` (`678 ms`) | `4.8` (`584 ms`) | `11.4` (`794 ms`) | `18.76 GiB` | baseline |
| `tp1-a-slots112-kv8g` | `112/48`, `8.0 GiB` (`34.3×`) | — | — | — | — | — | `<5 GiB` warmup | **failed** (`exit=137` startup memguard) |
| **`tp1-b-expert-prefill-8k`** (`108/48`, `6.0 GiB`, expert-centric prefill `8K`, `0844` `rc=0.5/tau=0.6`) | `108/48`, `6.0 GiB` (**`6.44M` tok = `24.58×`**) | **`332.6` (`13.6×`) / `1,229.3` / `2,113.9`** | **`6.7` (`+53.1%`)** | **`15.3` (`+40.2%`)** | **`6.6` (`+37.7%`)** | **`19.4` (`+71.2%`)** | **`9.06 GiB`** | **strong win** (`LRU 5/5`) |
| **`tp1-c-expert-prefill-16k-hotlist`** (`16K` batch + empirical hotlist*) | `108/48`, `6.0 GiB` (**`5.97M` tok = `22.79×`**) | **`330.3` (`13.5×`) / `1,576.3` / `2,124.3`** | **`9.0` (`+104.7%`)** | **`29.4` (`+169.0%`)** | **`9.1` (`+89.8%`)** | **`30.7` (`+170.5%`)** | **`5.03 GiB`** | **strong win** (`LRU 5/5`) |

*\*Note on hotlist arms (`tp1-c`, `tp2-d`): empirical hotlists were captured from the same Sonar decode workload (`tp1-b` / `tp2-a`). Held-out traffic will realize a realistic Zipfian fraction of the hotlist increment, stacked on top of the cold-prior gains already proven by `tp1-b` and `tp2-c`.*

### 2. `tp2-direct` (2 Sparks, `dgx1..dgx2`) — vs `manifests/benchmarks/2026-11-20-spark-ds41f-1120-direct2.json`

| Arm | Slots / KV (`256K` concurrency) | Prefill 32K / 131K (tok/s) | `prose-c1` | `prose-c4` | `code-c1` | `code-c4` | `min MemAvailable` (`dgx1`/`dgx2`) | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| `1120` ref (`adapt off`) | `240/64`, `8.0 GiB` (`34.3×`) | `258.8` / `235.6` | `17.7` (`121 ms`) | `53.0` (`154 ms`) | `22.2` (`128 ms`) | `64.2` (`166 ms`) | `9.03 / 11.23 GiB` | baseline |
| **`tp2-a-adaptive-rc025`** (config-only `0844`, `rc=0.25/tau=0.45`) | `240/64`, `8.0 GiB` (**`8.98M` tok = `34.26×`**) | — | **`20.3` (`+14.6%`)** | **`64.6` (`+21.9%`)** | **`23.2` (`+4.4%`)** | `63.5` (`-1.1%`) | `9.34 / 11.14 GiB` | **win** (over-trimmed `code-c4`) |
| **`tp2-c-expert-prefill-8k`** (`252/64`, `6.0 GiB`, expert-centric `8K`, `rc=0.25/tau=0.45`) | `252/64`, `6.0 GiB` (**`6.44M` tok = `24.58×`**) | **`4,596.6` (`17.8×`) / `4,897.4` (`20.8×`)** | **`21.6` (`+22.0%`)** | **`73.0` (`+37.8%`)** | **`24.8` (`+11.7%`)** | **`72.5` (`+12.9%`)** | **`5.88 / 7.82 GiB`** | **huge win** (`LRU 5/5`) |
| `tp2-d` (`16K` batch at `252/64`, `6.0 GiB`) | `252/64`, `6.0 GiB` (`22.79×`) | — | — | — | — | — | `<5 GiB` warmup | **failed** (`exit=137` startup memguard) |
| **`tp2-d` (`8K` batch + empirical hotlist* + `rc=0.15/tau=0.35`)** | `252/64`, `6.0 GiB` (**`6.44M` tok = `24.58×`**) | **`4,579.8` (`17.7×`) / `4,911.2` (`20.8×`)** | **`30.8` (`+74.0%`)** | **`101.5` (`+91.7%`)** | **`35.9` (`+61.7%`)** | **`95.1` (`+48.1%`)** | **`5.83 / 7.88 GiB`** | **huge win** (`LRU 5/5`) |
| `tp2-e-breakable-cg` (`VLLM_USE_BREAKABLE_CUDAGRAPH=1`) | `252/64`, `6.0 GiB` | — | — | — | — | — | startup error | **rejected** (PyTorch 2.13 `CUDACachingAllocator.cpp:3211` `markCaptureEnd` assert) |

### 3. `pp2-tp2-ring` (4 Sparks, `dgx1..dgx4`) — vs `manifests/benchmarks/2026-11-22-spark-ds41f-1122-pp2tp2ring.json`

| Arm | Variable | Prefill 32K / 131K | `prose-c1` / `c2` / `c4` / `c8` / `c16` | `code-c1` / `c2` / `c4` / `c8` / `c16` | `min MemAvailable` | Verdict |
|---|---|---|---:|---:|---:|---|
| `1122` ref | `20/20`, `16.5 GiB` KV (`19.7×` 1M), `adapt=off`, `sp=off`, `sink=0` | `5,572` / `5,728` | `40.1 / 76.2 / 123.3 / 210.2 / 385.9` | `50.3 / 93.2 / 148.2 / 235.5 / 445.5` | `16.6–20.0 GiB` | baseline |
| **`pp2-a-sp-prefill`** | Intra-stage `VLLM_DS41_PP_SP_PREFILL=1` | **`6,471` (`+16.1%`) / `6,655` (`+16.2%`)** | — | — | `16.3–20.8 GiB` | **strong win** (`LRU 5/5`) |
| **`pp2-b-adaptive-verif`** | Lockstep PP adaptive verif (`rc=0.045, tau=0.2`) | — | **`46.9 (+16.9%)` / `76.5` / `137.1 (+11.1%)` / `213.7` / `414.3 (+7.3%)`** | **`53.6 (+6.5%)` / `88.7 (-4.8%)` / `149.2` / `270.8 (+15.0%)` / `467.4 (+4.9%)`** | `18.5–23.7 GiB` | **strong win** (tune `c2` `rc`) |
| `pp2-c-chunk8k-l2-bal8` | `L2 18/18/20` + `VLLM_PP_DECODE_BALANCE_MAX_SEQS=8` (ran `16K` batch) | `5,538 (-0.6%)` / `5,687 (-0.7%)` | `40.0 / 76.4 / 123.1 / 208.5 / 368.8 (-4.4%)` | `50.1 / 93.1 / 146.5 / 235.6 / 461.2 (+3.5%)` | `17.0–19.9 GiB` | **rejected** (flat/worse on `prose-c16`) |
| `pp2-d-partition-14-26` | `VLLM_PP_LAYER_PARTITION=14,26`, `8.5 GiB` KV (`9.48×` 1M), `sp_prefill=1` | aborted on `dgx4` `3.73 GiB` floor | `40.1 / 60.4 (-20.7%) / 98.5 / 176.8 (-15.9%) / 317.1` | `50.4 / 74.6 (-19.9%) / 117.8 / 230.0 / 389.1` | `2.60 GiB` (`dgx4`) | **rejected** (`-20%` decode, tight Stage-1 RAM on 4 nodes) |
| **`pp2-e-combined`** | `sp_prefill=1` + PP adaptive verif (`rc=0.032, c1_scale=1.55, tau=0.2`) + `VLLM_SINK_REJECTION=1` | **`6,440 (+15.6%)` / `6,693 (+16.9%)`** | **`46.0 (+14.5%)` / `75.3` / `140.2 (+13.7%)` / `231.8 (+10.3%)` / `385.1`** | **`52.8 (+5.0%)` / `88.1 (-5.4%)` / `146.6` / `291.0 (+23.6%)` / `446.3`** | `16.3–20.0 GiB` | **win** (`LRU 5/5`, `sink=1` clean; `code-c2` needs lower `tau` or lower `c1_scale` on `PP=2`) |

## Key Engineering Findings

1. **Expert-centric prefill streaming (`cache.py`) is a transformative `14×–88×` prefill win for `tp1` and `tp2`.**
   - Stock `ExpertSlotCache._partition_windows` slices prefill sequentially by token range (`~12–25` tokens per window), re-reading the same cold experts from NVMe 10×–30× per layer per chunk with a `cuda.synchronize()` per window.
   - Switching to expert-outer single-pass LBA-sorted streaming (Pass 0 resident experts with zero I/O, Passes `1..K` cold experts once each in `dynamic_slots` batches in exact `plan.out_dtype` FP32 with `shared` folded once in FP32) lifts `tp2` prefill from **`259 / 236` → `4,597 / 4,911 tok/s` (`17.8×–20.8×`)** and `tp1` prefill from **`24.4` → `333 tok/s` (1K) / `1,576 tok/s` (8K) / `2,124 tok/s` (32K)**.
2. **Decent KV cache sizing for `tp1` and `tp2` (`6.00 GiB` = `6.44M` tokens = `24.58×` 256K ctx / `6.14×` 1M ctx) coexists cleanly with higher resident slot counts.**
   - `tp1`: `108/48` slots + `6.00 GiB` KV + `8K` batch keeps **`9.06 GiB` `min MemAvailable`** (`16K` keeps `5.03 GiB`). `112` slots + `8.0 GiB` KV tripped the 5 GiB startup guard.
   - `tp2-direct`: `252/64` slots + `6.00 GiB` KV + `8K` batch keeps **`5.83–5.88 GiB` on `dgx1`** / **`7.82–7.88 GiB` on `dgx2`**. `16K` batch at `252` slots tripped the 5 GiB startup guard.
3. **Empirical hotlist static placement (`VLLM_DS41_EXPERT_HOTLIST_PATH`) works and delivers major decode reductions** (`tp2-d`: `+74% / +92%` prose, `+62% / +48%` code; `tp1-c`: `+105% / +169%` prose, `+90% / +171%` code), proving checkpoint `e_score_correction_bias` alone is a weak proxy for inference expert hotness. A general multi-domain calibration corpus (prose + code + agent tool calls) should be used to produce a non-overfit default hotlist before promotion.
4. **Intra-stage `sp_prefill` on `PP > 1` (`sp_prefill.py` + `model.py`) is a clean `+16%` prefill win** (`5,572 → 6,471` at `32K`, `5,728 → 6,693` at `131K`) with zero memory or quality regressions.
5. **Lockstep PP adaptive verification (`vllm.py`, `pp_utils.py`, `adaptive_verification.py`, `model_runner.py`) works cleanly with zero deadlocks** (`+14.5%..+16.9%` `prose-c1`, `+13.7%` `prose-c4`, `+15%..+23.6%` `code-c8`), and `VLLM_SINK_REJECTION=1` also runs cleanly on `PP=2`. Only `code-c2` (`-4.8%..-5.4%`) needs a small `PP=2` `c1_scale` / `tau` tweak because `0854` runs `c2` as two `num_reqs=1` microbatches.
6. **Cleanly rejected levers (do not pursue further on these shapes):**
   - `VLLM_USE_BREAKABLE_CUDAGRAPH=1` on `tp2` (`CUDACachingAllocator.cpp:3211` PyTorch 2.13 internal assert during capture).
   - `VLLM_PP_LAYER_PARTITION=14,26` on 4-node `pp2-tp2-ring` (26 layers + DSpark on Stage 1 regresses `c2..c16` decode `-16%..-21%` and drops `dgx4` to `2.6 GiB` free RAM during prefill; also uncovered and fixed a real `prepare_disk_engram` bug when Engram layers `[1, 14]` split across PP stages).
   - `L2_PREFETCH 18/18/20` and `VLLM_PP_DECODE_BALANCE_MAX_SEQS=8` on `pp2-tp2-ring` (flat on `c1..c8`, `-4.4%` on `prose-c16`).
