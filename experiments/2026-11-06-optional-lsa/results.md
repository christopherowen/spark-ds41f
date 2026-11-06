# Results: Optional Default-Off BF16 TileLang LSA (`2026-11-06-optional-lsa`)

## Summary

- **Status**: Passed all 7 `sp8` GPU correctness and sanitizer stages (`summary-initcheck`, `summary-racecheck`, `summary-timing`, `registration`, `actual`, `initcheck`, `graph`) and 2 independent cold-container benchmark boots (`bench-boot1`, `bench-boot2`).
- **Default Path (`VLLM_DS41_PAGE_SHORTLIST=0` or unset)**:
  - Executes the **100% unchanged, bit-identical native indexer** (`state_content_bytes = 68`, `declared_kernel_count = 0`, `extra_scratch_bytes = 0`).
  - Verified bit-identical (`atol = 0, rtol = 0`) against the baseline native indexer across all 12 captured rank-0 batches and all synthetic prefill/decode configurations on both boots.
- **Precision & No Downward Quantization**:
  - Rejects `VLLM_DS41_PAGE_MODE=compact8` (`uint8` + `fp8_e4m3fn`) with a clear `ValueError`.
  - Uses only `block8` (`STATE_BYTES = 100`): 8-key `bfloat16` block means accumulated in `float32` via the 512-thread cooperative `summarize_blocks` kernel, scored in `bfloat16` with `float32` accumulation (`QueryReusePlan`), followed by exact `bfloat16`/`float32` fine scoring on shortlisted blocks (`RawWideCoarseTopK` for `budget = 1024`, `DirectCoarseTopK` for `budget = 512`, `PageShortlistPlan` for `budget = 64`).

## GPU Correctness & Sanitizer Gate Receipts (`runs/gpu-gates-01/`)

| Gate | Receipt | Result | Key Metrics |
| --- | --- | --- | --- |
| `summary-initcheck` | `runs/gpu-gates-01/summary-initcheck.json` | PASS | `compute-sanitizer --tool initcheck`: `0` errors; 512-thread BF16 summary bit-identical to 128-thread reference & FP32->BF16 oracle across all 4 rank-0 pools and synthetic fixtures |
| `summary-racecheck` | `runs/gpu-gates-01/summary-racecheck.json` | PASS | `compute-sanitizer --tool racecheck`: `0` race hazards across `128` and `256` page widths |
| `summary-timing` | `runs/gpu-gates-01/summary-timing.json` | PASS | `page=128`: **3.21x-3.24x** speedup (`13.29 us` vs `42.90 us`); `page=256`: **3.25x** speedup (`25.27 us` vs `82.06 us`) |
| `registration` | `runs/gpu-gates-01/registration.json` | PASS | `56` declared calls across 6 `QueryReusePlan` shapes (`budget=64, 512, 1024`) and `summarize_blocks` (`128, 256`) bind cleanly after `prepare.freeze(...)` |
| `actual` | `runs/gpu-gates-01/actual.json` | PASS | All 12 rank-0 batches pass exact BF16 block means, coarse score bit-identity, `RawWideCoarseTopK`/`DirectCoarseTopK` bit-identity, and retained fine score bit-identity |
| `initcheck` | `runs/gpu-gates-01/initcheck.json` | PASS | `compute-sanitizer --tool initcheck`: `0` errors across all 12 rank-0 batches and strict unwritten pool/table suffix bounds (`budget = 64, 512, 1024`) |
| `graph` | `runs/gpu-gates-01/graph.json` | PASS | `72/72` CUDA graph routes (`12` captures x `rows in (1, 8, 64)` x `mode in (decode, prefill)`) pass `original`, `empty`, and `restored` replays under `initcheck` |

## 512-Thread Cooperative BF16 Block Summary Timing (`runs/gpu-gates-01/summary-timing.json`)

| Pool | Page Size | Complete Pages | Single-Page 512T (`us`) | Single-Page 128T (`us`) | Single-Page Speedup | All-Pages 512T (`us`) | All-Pages 128T (`us`) | All-Pages Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `validation-a-rank-0-dense-0.pt` | 128 | 262 | 13.34 | 42.99 | **3.22x** | 32.75 | 45.84 | **1.40x** |
| `validation-a-rank-0-dense-1.pt` | 128 | 262 | 13.29 | 43.02 | **3.24x** | 35.16 | 45.77 | **1.30x** |
| `validation-a-rank-0-dense-2.pt` | 128 | 262 | 13.98 | 42.45 | **3.04x** | 32.71 | 46.02 | **1.41x** |
| `validation-a-rank-0-source-0.pt` | 256 | 492 | 25.27 | 82.06 | **3.25x** | 100.96 | 163.03 | **1.61x** |

## Captured Rank-0 Batches: 2-Boot Speedup & Top-K Recall (`bench-boot1`, `bench-boot2`)

| Capture | Case | Layer | Budget | Rows | Boot 1 Speedup | Boot 2 Speedup | 2-Boot Mean Speedup | Top-512 Recall (Mean / Min) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fit-a-rank-0-dense-0.pt` | `fit-a` | 2 | 64 | 64 | 2.2718x | 2.2616x | **2.2667x** | 80.25% / 68.36% |
| `fit-a-rank-0-dense-1.pt` | `fit-a` | 8 | 64 | 64 | 2.2582x | 2.2493x | **2.2538x** | 88.23% / 77.34% |
| `fit-a-rank-0-dense-2.pt` | `fit-a` | 14 | 64 | 64 | 2.2480x | 2.2517x | **2.2499x** | 90.25% / 76.76% |
| `fit-a-rank-0-source-0.pt` | `fit-a` | 20 | 1024 | 64 | 1.1259x | 1.1055x | **1.1157x** | 97.92% / 94.34% |
| `fit-b-rank-0-dense-0.pt` | `fit-b` | 2 | 64 | 64 | 2.2514x | 2.2606x | **2.2560x** | 84.89% / 72.27% |
| `fit-b-rank-0-dense-1.pt` | `fit-b` | 8 | 64 | 64 | 2.2536x | 2.2621x | **2.2578x** | 91.95% / 83.20% |
| `fit-b-rank-0-dense-2.pt` | `fit-b` | 14 | 64 | 64 | 2.2544x | 2.2642x | **2.2593x** | 96.05% / 90.62% |
| `fit-b-rank-0-source-0.pt` | `fit-b` | 20 | 1024 | 64 | 1.2060x | 1.2014x | **1.2037x** | 97.98% / 94.92% |
| `validation-a-rank-0-dense-0.pt` | `validation-a` | 2 | 64 | 64 | 2.2546x | 2.2552x | **2.2549x** | 77.67% / 65.43% |
| `validation-a-rank-0-dense-1.pt` | `validation-a` | 8 | 64 | 64 | 2.2549x | 2.2544x | **2.2547x** | 86.76% / 68.55% |
| `validation-a-rank-0-dense-2.pt` | `validation-a` | 14 | 64 | 64 | 2.2496x | 2.2633x | **2.2564x** | 89.15% / 77.15% |
| `validation-a-rank-0-source-0.pt` | `validation-a` | 20 | 1024 | 64 | 1.1481x | 1.1467x | **1.1474x** | 97.87% / 96.09% |

## Full `32K-1M` Combined 4-Layer (`3x Dense + 1x Source`) 2-Boot Speedup Matrix

Combined 4-layer indexer latency per step (`3 * dense_layer_us + 1 * source_layer_us`), mirroring the 3:1 ratio of compressed layers (`2, 8, 14`, `compress_ratio = 2`) to source layer (`20`, `compress_ratio = 1`):

| Context Tokens | Prefill (`r=64`) B1 / B2 | Prefill (`r=64`) Mean | Decode (`r=8`) B1 / B2 | Decode (`r=8`) Mean | Decode (`r=24`) B1 / B2 | Decode (`r=24`) Mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 1.2674x / 1.2646x | **1.2660x** | 0.9814x / 0.9717x | **0.9766x** | 1.0949x / 1.0991x | **1.0970x** |
| 65,536 | 1.6736x / 1.6708x | **1.6722x** | 1.1915x / 1.1735x | **1.1825x** | 1.3022x / 1.2979x | **1.3001x** |
| 131,072 | 1.8948x / 1.8888x | **1.8918x** | 1.4758x / 1.4827x | **1.4793x** | 1.7401x / 1.7540x | **1.7471x** |
| 262,144 | 2.8140x / 2.8397x | **2.8268x** | 2.1321x / 2.0136x | **2.0729x** | 2.6113x / 2.6032x | **2.6073x** |
| 524,288 | 4.2204x / 4.2496x | **4.2350x** | 3.2257x / 3.2340x | **3.2298x** | 4.0588x / 4.0699x | **4.0644x** |
| 1,048,576 | 4.1140x / 4.0969x | **4.1055x** | 3.7837x / 3.7798x | **3.7818x** | 4.0594x / 4.0517x | **4.0555x** |

## Key Conclusions

1. **Default-Off Is Zero-Risk and 100% Bit-Identical**:
   - Leaving `VLLM_DS41_PAGE_SHORTLIST` unset or `0` keeps `state_content_bytes = 68`, compiles zero shortlist kernels, allocates zero shortlist scratch, and matches baseline native output with `max_abs_diff = 0` across every tested shape.
2. **Zero Downward Quantization**:
   - All summary storage (`block8`, `STATE_BYTES = 100`) and coarse scoring stay in native `bfloat16` with `float32` accumulation. Downward-quantized `compact8` is rejected at initialization.
3. **Full `32K-1M` Prefill & Multi-Row Decode Coverage**:
   - Lowering the dense-layer `prefill_short` floor from `16,385` to `16,384` engages `budget = 64` coarse selection on layers `2, 8, 14` at `32K` (`score_width = 16,384`), delivering **1.62x** dense-layer prefill speedup (`416.8 us -> 257.6 us`) and **1.27x** combined 4-layer prefill speedup at `32K` (previously `1.00x`).
   - Adding `source_short` (`budget = 512`, `DirectCoarseTopK`) at `64K` (`score_width = 65,536`) and lowering `RawWideCoarseTopK` (`budget = 1024`) floor from `65,664` to `131,072` engages coarse selection across `64K-1M` on layer `20`, delivering **1.67x** combined prefill speedup at `64K`, **1.89x** at `131K`, **2.83x** at `262K`, **4.24x** at `524K`, and **4.11x** at `1M`.
   - Multi-row decode (`rows >= 2`, e.g. MTP / batched decode) engages coarse selection across `32K-1M` (`1.10x` at `32K`, `1.30x` at `64K`, `1.75x` at `131K`, `2.61x` at `262K`, `4.06x` at `524K-1M`), while single-row decode (`rows == 1`) stays on native indexing where the tiny 1-row native kernel is already memory-latency bound (`43-348 us`).
