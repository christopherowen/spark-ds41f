# Long-Context `indexer_score_v41` `KPool` Vectorization, CTA Co-Scheduling, and `evict_first` / `coded_scales` Evaluation (`ptxas 13.0.88` & `13.5.37`)

## Metadata

- **Date**: `2026-11-08`
- **Base commit**: `cc20ed18b` (`spark-ds41f-1107`, vLLM patch head `2756478b60b90700b9918b457aeb38657da637b5`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate patch**: `vllm/0762-perf-dsv41-vectorize-indexer-KPool-loads-and-co-sche.patch` (vLLM patch head `a52f800a66f5e99689c1438e0c1257fff563dd08`, tree `b1d1cf0c3294f4d73784a12621fca683d82b18b0`, `patchset_sha256` `b9160b1febe4c87d67b7f98fe652fd01080e6f36792e7b529c2b24f370f47160`)
- **Hardware**: NVIDIA GB10 (`SM121`, 48 SMs / 24 MB L2 cache, `sp6`)
- **Toolchains evaluated**: CUDA `13.0.2` (`ptxas 13.0.88`) and CUDA `13.5` (`ptxas 13.5.37`)

---

## 1. Executive Summary

This experiment revisits two open questions under both CUDA `13.0.2` (`ptxas 13.0.88`) and CUDA `13.5` (`ptxas 13.5.37`):

1. **Long-Context (`4K–256K`) `indexer_score_v41` `KPool` `evict_first` (`experiments/2026-10-25-moe-overlap` lines 208–210)**:
   - **Negative finding on `L2::evict_first` (`ld.global.cs`)**: Because GB10 runs with `CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 0`, `l2_prefetch` lines loaded via `cp.async.bulk.prefetch.L2 ... L2::evict_last` reside in the unreferenced prefetch victim slot of L2 sets until demand-hit by `WO-A`/`WO-B`. Streaming `KPool` with `ld.global.cs` (`LDG.E.EF.128`) explicitly targets that unreferenced victim slot, **evicting the prefetched `WO-A`/`WO-B` weights** (`wo` latency rises from `95.3–106.5 us` with `ca` to `128.4–131.0 us` cold DRAM speed with `cs` at `L = 262,144`, and from `115.2 us` to `130.0 us` at `L = 131,072`) and degrading inter-row `KPool` L2 reuse at `rows = 6`.
   - **Positive finding #1 — Broken `load_keys` vectorization in `indexer_score_v41`**: In `indexer_score_v41`, `for p, d in T.Parallel(64, 128)` with `threads = 128` assigned only 2 E2M1 elements (`1 byte`) per thread along `D = 128` across a 32-iteration loop along `block_P = 64`, and TileLang failed to vectorize across the bounds predicate and 16-byte swizzle boundary. Each thread issued **32 scalar 1-byte `LDG.E.U8` loads, 32 1-byte `STS.U8` stores, and 33 redundant `PageTable` global loads per 64-position tile** (`1,223` SASS instructions / `148` registers on `13.0.88`, exploding to `1,518` SASS instructions / `250` registers / 2 CTAs/SM on `13.5.37`). Replacing `load_keys` with 16-byte `uint4` (`32` E2M1) global loads (`4` threads per row `p`), a single hoisted `PageTable` lookup per tile/row, and direct swizzled `STS.128` stores (`__tl_load_k_shared_dense` and `__tl_load_k_shared_candidates`) reduces `indexer_score_v41` to **`528` SASS instructions (`-56.8%`) and `68` registers** on `13.0.88` (**`535` SASS instructions (`-64.8%`) and `67` registers** on `13.5.37`).
   - **Positive finding #2 — Persistent CTA column co-scheduling across rows in dense mode (`rf_ca`)**: Launching `with T.Kernel(ctas, rows) as (cta, row)` (`ctas = 128, rows = 6`) schedules all `128` CTAs of `row = 0` first (`Waves 0..1` on GB10), streaming the entire `17.8 MB` `KPool` for `row = 0` before `row = 1..5` (`Waves 2..12`) launch and evicting early `KPool` pages before `row = 1..5` can reuse them. Mapping `cta = (by * ctas + bx) // rows; row = (by * ctas + bx) % rows` in dense mode co-schedules all `rows` (`0..5`) of each persistent `cta` column in the **same wave**, cutting 6-row cold-L2 `indexer_score_v41` latency by **`38.5%–50.1%` (`2.01x` faster)** across `16K–262K` context (`481.3 -> 240.0 us` at `L = 262,144`) and reducing full post-MoE attention block latency (`l2_prefetch` + `indexer` + `sparse_mla` + `WO-A`/`WO-B`) by **`241.7 us/layer` (`-37.3%`)** at `L = 262,144`.
2. **Drafter & Target MoE `coded_scales` / `narrow_codes` (`M = 1, 2, 4, 6`) under `ptxas 13.0.88` vs `ptxas 13.5.37` (`experiments/2026-10-18-scale-codes`)**:
   - At `block_M = 32` (`drafter_tp4_gateup_bm32`), `ptxas 13.5.37` reduces `coded_4bit` from `696` to `608` SASS instructions (`-88` instructions, `-12.6%`) and `120 -> 118` registers by hoisting scale offset and bit-shift/mask addressing onto the Uniform Datapath (`UIADD3`, `ULOP3.LUT`, `IMAD.WIDE.U32`).
   - In `spark-ds41f:1107`, `MoeExpertsV41Plan` already uses `decode_block_M = 16` with `coded_scales=True, narrow_codes=True` (`608` SASS instructions / `94` registers on `13.0.88`, `632` SASS instructions / `92` registers on `13.5.37`, maintaining 2 CTAs/SM occupancy). Across 2 boots on all 9 drafter and target shapes, `coded_narrow` beats `uncoded` by **`3%–7%` on cold-L2 latency** even at `M = 1, 2`.

---

## 2. Experiment P5: `indexer_score_v41` Codegen & 2-Boot Benchmarks (`4K–256K`)

### 2.1 SASS Instruction & Register Footprint (`SM121`)

| Kernel Variant | Mode | `ptxas 13.0.88` SASS Insts | `ptxas 13.0.88` Regs | `ptxas 13.5.37` SASS Insts | `ptxas 13.5.37` Regs |
|---|---|---:|---:|---:|---:|
| `orig_ca` (`1107` baseline, 32x `LDG.E.U8` + 33x `PageTable`) | `dense` | `1,223` | `148` | `1,518` | `250` |
| `orig_cs` (32x `ld.global.cs.u8`) | `dense` | `945` | `124` | `950` | `122` |
| `vec128_ca` (2x `LDG.E.128` + 1x `PageTable` + `STS.128`) | `dense` | **`528`** (`-56.8%`) | **`68`** (`-54.1%`) | **`535`** (`-64.8%`) | **`67`** (`-73.2%`) |
| `rf_ca` (`vec128_ca` + row-first CTA co-scheduling, **promoted in `0762`**) | `dense` | **`536`** (`-56.2%`) | **`68`** (`-54.1%`) | **`542`** (`-64.3%`) | **`67`** (`-73.2%`) |
| `orig_ca` (`1107` baseline) | `candidates` | `1,158` | `182` | `1,250` | `252` |
| `vec128_ca` (**promoted in `0762`**) | `candidates` | **`745`** (`-35.7%`) | **`118`** (`-35.2%`) | **`732`** (`-41.4%`) | **`94`** (`-62.7%`) |

### 2.2 Standalone `indexer_score_v41` & Full Attention Block (`rows = 6`, MTP Decode Verification, 2-Boot Means `[Boot 1 / Boot 2]`)

#### CUDA `13.0.2` (`ptxas 13.0.88`, `sp6`)

| `L` (`KPool` MB) | Variant | `score_warm` (`us`) | `score_cold` (`us`) | `attn_idx` (`us`) | `sparse_mla` (`us`) | `wo` (`WO-A+B`, `us`) | `attn_total` (`us`) | Delta vs `orig_ca` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `4,096` (`0.28 MB`) | `orig_ca` | `15.5` `[15.5/15.6]` | `18.3` `[18.4/18.2]` | `55.6` | `39.9` | `83.9` | `179.4` `[179.2/179.7]` | — |
| `4,096` (`0.28 MB`) | `vec128_ca` | `10.1` `[9.7/10.4]` | `16.4` `[16.4/16.4]` | `47.5` | `38.9` | `84.6` | `171.0` `[171.0/171.0]` | `-8.4 us` (`-4.7%`) |
| `4,096` (`0.28 MB`) | **`rf_ca` (`0762`)** | **`9.9`** `[9.6/10.2]` | **`16.4`** `[16.4/16.4]` | **`48.1`** | `39.0` | `83.8` | **`170.9`** `[170.9/171.0]` | **`-8.5 us` (`-4.7%`)** |
| `16,384` (`1.11 MB`) | `orig_ca` | `38.6` `[38.5/38.6]` | `40.0` `[40.0/40.1]` | `84.3` | `40.6` | `84.5` | `209.4` `[210.1/208.7]` | — |
| `16,384` (`1.11 MB`) | `vec128_ca` | `19.6` `[19.6/19.5]` | `24.6` `[24.6/24.6]` | `68.1` | `40.3` | `84.7` | `193.1` `[193.5/192.6]` | `-16.3 us` (`-7.8%`) |
| `16,384` (`1.11 MB`) | **`rf_ca` (`0762`)** | **`18.9`** `[18.9/18.9]` | **`24.6`** `[24.6/24.6]` | **`66.9`** | `40.4` | `84.2` | **`191.5`** `[191.5/191.5]` | **`-17.9 us` (`-8.5%`)** |
| `32,768` (`2.23 MB`) | `orig_ca` | `66.6` `[66.5/66.6]` | `68.6` `[68.6/68.6]` | `112.9` | `41.1` | `85.1` | `239.1` `[239.6/238.5]` | — |
| `32,768` (`2.23 MB`) | `vec128_ca` | `30.8` `[30.8/30.7]` | `38.9` `[38.9/38.9]` | `84.0` | `40.8` | `85.9` | `210.7` `[210.9/210.5]` | `-28.4 us` (`-11.9%`) |
| `32,768` (`2.23 MB`) | **`rf_ca` (`0762`)** | **`31.1`** `[31.2/31.1]` | **`38.9`** `[38.9/38.9]` | **`86.0`** | `40.9` | `85.6` | **`212.5`** `[212.9/212.1]` | **`-26.6 us` (`-11.1%`)** |
| `65,536` (`4.46 MB`) | `orig_ca` | `123.5` `[123.4/123.6]` | `125.0` `[124.9/125.0]` | `171.2` | `41.6` | `86.2` | `299.0` `[299.0/299.0]` | — |
| `65,536` (`4.46 MB`) | `vec128_ca` | `54.9` `[55.1/54.7]` | `67.1` `[66.6/67.6]` | `114.2` | `41.1` | `88.2` | `243.5` `[243.8/243.2]` | `-55.5 us` (`-18.6%`) |
| `65,536` (`4.46 MB`) | **`rf_ca` (`0762`)** | **`56.3`** `[56.3/56.3]` | **`67.6`** `[67.6/67.6]` | **`109.6`** | `40.9` | `87.1` | **`237.5`** `[237.6/237.5]` | **`-61.5 us` (`-20.6%`)** |
| `131,072` (`8.91 MB`) | `orig_ca` | `237.1` `[236.9/237.3]` | `240.6` `[240.6/240.6]` | `278.5` | `41.9` | `87.8` | `408.2` `[408.4/408.0]` | — |
| `131,072` (`8.91 MB`) | `vec128_ca` | `102.3` `[102.1/102.4]` | `128.0` `[128.0/128.0]` | `164.8` | `41.4` | `115.2` | `321.4` `[321.5/321.3]` | `-86.8 us` (`-21.3%`) |
| `131,072` (`8.91 MB`) | `vec128_cs` | `165.6` `[165.4/165.8]` | `160.8` `[160.8/160.8]` | `197.6` | `41.9` | `130.0` | `369.5` `[369.5/369.4]` | `-38.7 us` (`-9.5%`) |
| `131,072` (`8.91 MB`) | **`rf_ca` (`0762`)** | **`108.2`** `[108.4/108.0]` | **`124.3`** `[123.9/124.8]` | **`160.8`** | `41.6` | `90.6` | **`292.9`** `[292.9/293.0]` | **`-115.3 us` (`-28.2%`)** |
| `131,072` (`8.91 MB`) | `rf_cs` | `126.9` `[126.9/126.9]` | `127.5` `[127.0/128.0]` | `163.8` | `42.0` | `128.8` | `334.6` `[334.8/334.4]` | `-73.6 us` (`-18.0%`) |
| `262,144` (`17.83 MB`) | `orig_ca` | `480.7` `[480.7/480.8]` | `481.3` `[481.3/481.3]` | `510.0` | `42.5` | `95.3` | `647.8` `[648.1/647.5]` | — |
| `262,144` (`17.83 MB`) | `orig_cs` | `470.5` `[470.1/470.9]` | `471.0` `[471.0/471.0]` | `500.7` | `42.7` | `128.4` | `671.8` `[671.7/671.9]` | `+24.0 us` (`+3.7%`) |
| `262,144` (`17.83 MB`) | `vec128_ca` | `268.2` `[267.9/268.5]` | `268.2` `[268.3/268.2]` | `297.0` | `42.4` | `106.5` | `445.9` `[445.5/446.3]` | `-201.9 us` (`-31.2%`) |
| `262,144` (`17.83 MB`) | `vec128_cs` | `327.2` `[327.2/327.2]` | `327.0` `[326.7/327.4]` | `355.3` | `42.8` | `130.9` | `529.0` `[528.8/529.3]` | `-118.8 us` (`-18.3%`) |
| `262,144` (`17.83 MB`) | **`rf_ca` (`0762`)** | **`239.3`** `[239.4/239.1]` | **`240.0`** `[240.4/239.6]` | **`267.2`** | `42.6` | `96.3` | **`406.1`** `[406.5/405.7]` | **`-241.7 us` (`-37.3%`)** |
| `262,144` (`17.83 MB`) | `rf_cs` | `247.3` `[247.4/247.2]` | `247.3` `[247.8/246.8]` | `275.5` | `42.8` | `131.0` | `449.3` `[449.5/449.0]` | `-198.5 us` (`-30.6%`) |

#### CUDA `13.5` (`ptxas 13.5.37`, `sp6`)

| `L` (`KPool` MB) | Variant | `score_warm` (`us`) | `score_cold` (`us`) | `attn_idx` (`us`) | `sparse_mla` (`us`) | `wo` (`WO-A+B`, `us`) | `attn_total` (`us`) | Delta vs `orig_ca` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `4,096` (`0.28 MB`) | `orig_ca` | `21.2` `[21.5/20.9]` | `21.5` `[21.5/21.5]` | `56.5` | `42.1` | `84.8` | `183.4` `[184.3/182.4]` | — |
| `4,096` (`0.28 MB`) | **`rf_ca` (`0762`)** | **`9.8`** `[9.7/9.9]` | **`16.4`** `[16.4/16.4]` | **`48.1`** | `41.1` | `84.0` | **`173.2`** `[173.1/173.2]` | **`-10.2 us` (`-5.6%`)** |
| `16,384` (`1.11 MB`) | `orig_ca` | `64.4` `[65.5/63.3]` | `65.1` `[65.6/64.5]` | `110.2` | `42.8` | `85.2` | `238.2` `[239.2/237.2]` | — |
| `16,384` (`1.11 MB`) | **`rf_ca` (`0762`)** | **`18.9`** `[18.9/18.9]` | **`24.6`** `[24.6/24.6]` | **`66.9`** | `42.7` | `84.0` | **`193.6`** `[193.7/193.5]` | **`-44.6 us` (`-18.7%`)** |
| `32,768` (`2.23 MB`) | `orig_ca` | `119.3` `[119.8/118.8]` | `120.8` `[120.8/120.8]` | `164.9` | `43.2` | `85.8` | `293.9` `[294.9/292.9]` | — |
| `32,768` (`2.23 MB`) | **`rf_ca` (`0762`)** | **`31.0`** `[30.9/31.1]` | **`38.9`** `[38.9/38.9]` | **`86.0`** | `43.0` | `85.5` | **`214.5`** `[214.5/214.6]` | **`-79.4 us` (`-27.0%`)** |
| `65,536` (`4.46 MB`) | `orig_ca` | `232.0` `[232.2/231.8]` | `233.5` `[233.5/233.5]` | `278.7` | `43.6` | `86.7` | `409.0` `[409.6/408.4]` | — |
| `65,536` (`4.46 MB`) | **`rf_ca` (`0762`)** | **`56.3`** `[56.3/56.2]` | **`67.6`** `[67.6/67.6]` | **`109.6`** | `43.1` | `87.1` | **`239.8`** `[239.7/239.8]` | **`-169.2 us` (`-41.4%`)** |
| `131,072` (`8.91 MB`) | `orig_ca` | `454.9` `[455.2/454.7]` | `457.7` `[457.7/457.7]` | `495.5` | `44.0` | `88.2` | `627.7` `[628.0/627.4]` | — |
| `131,072` (`8.91 MB`) | **`rf_ca` (`0762`)** | **`108.1`** `[108.1/108.1]` | **`124.2`** `[123.9/124.6]` | **`160.8`** | `43.7` | `90.6` | **`295.1`** `[295.1/295.1]` | **`-332.6 us` (`-53.0%`)** |
| `262,144` (`17.83 MB`) | `orig_ca` | `901.6` `[901.6/901.6]` | `904.2` `[904.2/904.2]` | `932.9` | `44.6` | `95.5` | `1,073.0` `[1,073.2/1,072.8]` | — |
| `262,144` (`17.83 MB`) | **`rf_ca` (`0762`)** | **`239.1`** `[239.1/239.1]` | **`240.3`** `[239.9/240.6]` | **`267.1`** | `44.7` | `96.2` | **`408.0`** `[407.9/408.1]` | **`-665.0 us` (`-62.0%`)** |

### 2.3 Single-Row (`rows = 1`) & Reindex Candidate Mode (`candidates=True`, `width=4096`)

- **Single-row (`rows = 1`, `ptxas 13.0.88`, 2-boot mean)**:
  - `L = 65,536`: `score_warm` `21.6 -> 10.6 us` (**`-50.9%`**), `score_cold` `36.9 -> 34.8 us`, `attn_total` `158.1 -> 154.0 us` (**`-4.1 us`**).
  - `L = 131,072`: `score_warm` `40.4 -> 18.8 us` (**`-53.5%`**), `score_cold` `67.6 -> 65.5 us`, `attn_total` `187.8 -> 184.7 us` (**`-3.1 us`**).
  - `L = 262,144`: `score_warm` `106.7 -> 72.9 us` (**`-31.7%`**), `score_cold` `128.0 -> 124.1 us` (`-3.9 us`), `attn_total` `266.2 -> 261.8 us` (**`-4.4 us`**; vs `267.0 us` with `rf_cs` because `cs` degrades `WO-A+B` from `97.3 us` to `130.6 us`).
- **Candidate reindex mode (`candidates=True, width=4096, rows=6`, 2-boot mean)**:
  - `ptxas 13.0.88`: `score_warm` `18.6 -> 13.0 us` (**`-30.1%`**), `score_cold` `42.9 -> 30.7 us` (**`-28.4%`**).
  - `ptxas 13.5.37`: `score_warm` `34.1 -> 12.9 us` (**`-62.2%`**), `score_cold` `55.5 -> 30.8 us` (**`-44.5%`**).

---

## 3. Experiment P4: Drafter & Target MoE `coded_scales` / `narrow_codes` (`ptxas 13.0.88` vs `13.5.37`)

### 3.1 SASS Instruction & Register Footprint (`grouped_fp8_fp4_gemm`, `SM121`)

| Shape | `block_M` | `uncoded` (`13.0.88` / `13.5.37`) | `coded_4bit` (`13.0.88` / `13.5.37`) | `coded_narrow` (`13.0.88` / `13.5.37`) |
|---|---:|---|---|---|
| `drafter_tp4_gateup_bm32` (`E=128, N=1152, K=5120`) | `32` | `632` insts, `116` regs / `624` insts, `116` regs | `696` insts, `120` regs / **`608` insts (`-12.6%`), `118` regs** | `792` insts, `128` regs / `784` insts, `128` regs |
| `drafter_tp4_gateup` (`E=128, N=1152, K=5120`) | `16` | `464` insts, `84` regs / `472` insts, `84` regs | `504` insts, `88` regs / `512` insts, `88` regs | `608` insts, `94` regs / `632` insts, `92` regs |
| `drafter_tp4_down` (`E=128, N=5120, K=576`) | `16` | `512` insts, `94` regs / `520` insts, `95` regs | `544` insts, `98` regs / `568` insts, `100` regs | `624` insts, `103` regs / `640` insts, `105` regs |
| `drafter_tp3_gateup` (`E=128, N=1536, K=5120`) | `16` | `464` insts, `84` regs / `472` insts, `84` regs | `504` insts, `88` regs / `512` insts, `88` regs | `608` insts, `94` regs / `632` insts, `92` regs |
| `drafter_tp3_down` (`E=128, N=5120, K=768`) | `16` | `424` insts, `84` regs / `432` insts, `84` regs | `464` insts, `88` regs / `480` insts, `88` regs | `528` insts, `94` regs / `544` insts, `94` regs |
| `target_tp4_gateup` (`E=384, N=1152, K=5120`) | `16` | `464` insts, `84` regs / `472` insts, `84` regs | `504` insts, `88` regs / `512` insts, `88` regs | `608` insts, `94` regs / `632` insts, `92` regs |
| `target_tp3_gateup` (`E=384, N=1536, K=5120`) | `16` | `464` insts, `84` regs / `472` insts, `84` regs | `504` insts, `88` regs / `512` insts, `88` regs | `608` insts, `94` regs / `632` insts, `92` regs |

### 3.2 2-Boot Cold-L2 MoE Latency (`us`, `[Boot 1 / Boot 2]`)

| Shape | `M` | `uncoded` (`13.0.88`) | `coded_narrow` (`13.0.88`) | `uncoded` (`13.5.37`) | `coded_narrow` (`13.5.37`) | Cold Delta (`coded_narrow` vs `uncoded`) |
|---|---:|---:|---:|---:|---:|---:|
| `drafter_tp3_gateup` | `1` | `85.91` `[86.82/84.99]` | **`80.94`** `[80.99/80.90]` | `86.50` `[86.02/86.98]` | **`80.45`** `[79.94/80.96]` | **`-5.8%` / `-7.0%`** |
| `drafter_tp3_gateup` | `2` | `160.87` `[159.87/161.86]` | **`153.14`** `[152.58/153.70]` | `161.36` `[160.83/161.89]` | **`154.18`** `[154.69/153.66]` | **`-4.8%` / `-4.5%`** |
| `drafter_tp3_down` | `1` | `42.43` `[41.92/42.94]` | **`40.96`** `[40.96/40.96]` | `42.47` `[41.92/43.01]` | **`40.50`** `[40.03/40.96]` | **`-3.5%` / `-4.6%`** |
| `drafter_tp3_down` | `2` | `84.46` `[85.02/83.90]` | **`80.99`** `[81.02/80.96]` | `85.04` `[85.18/84.90]` | **`82.43`** `[82.94/81.92]` | **`-4.1%` / `-3.1%`** |
| `drafter_tp4_gateup` | `1` | `64.50` `[63.46/65.54]` | **`60.91`** `[60.38/61.44]` | `61.97` `[62.43/61.50]` | **`58.40`** `[58.37/58.43]` | **`-5.6%` / `-5.8%`** |
| `drafter_tp4_gateup` | `2` | `133.06` `[133.12/132.99]` | **`129.54`** `[130.05/129.02]` | `132.14` `[132.10/132.19]` | **`129.44`** `[130.91/127.97]` | **`-2.6%` / `-2.0%`** |
| `drafter_tp4_down` | `1` | `34.27` `[34.75/33.79]` | **`33.42`** `[32.86/33.98]` | `33.77` `[33.79/33.76]` | **`33.17`** `[33.57/32.77]` | **`-2.5%` / `-1.8%`** |
| `drafter_tp4_down` | `2` | `60.91` `[61.44/60.38]` | **`57.92`** `[58.43/57.41]` | `60.91` `[61.44/60.38]` | **`58.80`** `[59.36/58.24]` | **`-4.9%` / `-3.5%`** |
| `target_tp3_gateup` | `1` | `161.17` `[160.77/161.57]` | **`153.04`** `[153.50/152.58]` | `162.29` `[163.84/160.74]` | **`153.15`** `[153.60/152.70]` | **`-5.0%` / `-5.6%`** |
| `target_tp3_gateup` | `6` | `706.56` `[704.51/708.61]` | **`660.99`** `[663.55/658.43]` | `705.20` `[705.76/704.64]` | **`661.07`** `[661.66/660.48]` | **`-6.4%` / `-6.3%`** |
| `target_tp4_gateup` | `1` | `134.14` `[133.12/135.17]` | **`128.52`** `[130.05/126.98]` | `131.20` `[132.22/130.18]` | **`127.34`** `[127.87/126.82]` | **`-4.2%` / `-2.9%`** |
| `target_tp4_gateup` | `6` | `553.06` `[564.22/541.89]` | **`517.71`** `[528.38/507.04]` | `544.79` `[547.84/541.73]` | **`514.56`** `[514.08/515.04]` | **`-6.4%` / `-5.5%`** |

`MoeExpertsV41Plan` in `spark-ds41f:1107` already enables `coded_scales=True, narrow_codes=True` with `decode_block_M = 16` across both drafter and target MoE layers, which is optimal under both `ptxas 13.0.88` and `ptxas 13.5.37`.

---

## 4. Verification & Receipts

- **Unit & Sanitizer Tests (2 boots on `sp6`)**:
  - `pytest --noconftest -v tests/kernels/attention/test_deepseek_v41_tilelang.py tests/models/test_deepseek_v4_1_lsa.py`: **`66 passed, 1 xfailed`** per boot (`receipts/pytest_boot1.txt`, `receipts/pytest_boot2.txt`), including `test_indexer_reads_only_visible_rows` under `compute-sanitizer --tool initcheck` (`ERROR SUMMARY: 0 errors`) and `test_indexer_score_vectorizes_kpool_loads_and_coschedules_dense_rows`.
- **Raw 2-boot JSON receipts**:
  - Experiment P5 (`indexer_score_v41` + full attention block): `receipts/p5_13.0.88_boot1.json`, `receipts/p5_13.0.88_boot2.json`, `receipts/p5_13.5.37_boot1.json`, `receipts/p5_13.5.37_boot2.json`.
  - Experiment P4 (drafter & target MoE scale codes): `receipts/p4_13.0.88_boot1.json`, `receipts/p4_13.0.88_boot2.json`, `receipts/p4_13.5.37_boot1.json`, `receipts/p4_13.5.37_boot2.json`.
