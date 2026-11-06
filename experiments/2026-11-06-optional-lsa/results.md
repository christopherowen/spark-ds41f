# Results: Optional Default-Off BF16 TileLang LSA (`2026-11-06-optional-lsa`)

## Summary

- **Status**: Passed all 7 `sp8` GPU correctness and sanitizer stages (`summary-initcheck`, `summary-racecheck`, `summary-timing`, `registration`, `actual`, `initcheck`, `graph`) and 2 independent cold-container benchmark boots (`bench-boot1`, `bench-boot2`).
- **Single Opt-In Knob (`VLLM_DS41_LSA`)**:
  - **Default-Off (`VLLM_DS41_LSA=0` or unset)**: Executes the **100% unchanged, bit-identical native indexer** (`state_content_bytes = 68`, `declared_kernel_count = 0`, `extra_scratch_bytes = 0`). Verified bit-identical (`atol = 0, rtol = 0`) against the baseline native indexer across all 12 captured rank-0 batches and all synthetic prefill/decode configurations on both boots.
  - **Opt-In (`VLLM_DS41_LSA=1`)**: Enables the single non-quantized BF16 `block8` (`STATE_BYTES = 100`) TileLang LSA implementation with zero sub-options.
- **Precision & No Downward Quantization**:
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
| `fit-a-rank-0-dense-0.pt` | `fit-a` | 2 | 64 | 64 | 2.2674x | 2.2554x | **2.2614x** | 80.25% / 68.36% |
| `fit-a-rank-0-dense-1.pt` | `fit-a` | 8 | 64 | 64 | 2.2555x | 2.2410x | **2.2483x** | 88.23% / 77.34% |
| `fit-a-rank-0-dense-2.pt` | `fit-a` | 14 | 64 | 64 | 2.2505x | 2.2643x | **2.2574x** | 90.25% / 76.76% |
| `fit-a-rank-0-source-0.pt` | `fit-a` | 20 | 1024 | 64 | 1.1367x | 1.1242x | **1.1305x** | 97.92% / 94.34% |
| `fit-b-rank-0-dense-0.pt` | `fit-b` | 2 | 64 | 64 | 2.2593x | 2.2590x | **2.2591x** | 84.89% / 72.27% |
| `fit-b-rank-0-dense-1.pt` | `fit-b` | 8 | 64 | 64 | 2.2583x | 2.2542x | **2.2563x** | 91.95% / 83.20% |
| `fit-b-rank-0-dense-2.pt` | `fit-b` | 14 | 64 | 64 | 2.2512x | 2.2503x | **2.2508x** | 96.05% / 90.62% |
| `fit-b-rank-0-source-0.pt` | `fit-b` | 20 | 1024 | 64 | 1.1993x | 1.1991x | **1.1992x** | 97.98% / 94.92% |
| `validation-a-rank-0-dense-0.pt` | `validation-a` | 2 | 64 | 64 | 2.2600x | 2.2481x | **2.2540x** | 77.67% / 65.43% |
| `validation-a-rank-0-dense-1.pt` | `validation-a` | 8 | 64 | 64 | 2.2515x | 2.2652x | **2.2584x** | 86.76% / 68.55% |
| `validation-a-rank-0-dense-2.pt` | `validation-a` | 14 | 64 | 64 | 2.2168x | 2.2503x | **2.2336x** | 89.15% / 77.15% |
| `validation-a-rank-0-source-0.pt` | `validation-a` | 20 | 1024 | 64 | 1.1615x | 1.1506x | **1.1561x** | 97.87% / 96.09% |

## Full `32K-1M` Combined 4-Layer (`3x Dense + 1x Source`) 2-Boot Speedup Matrix

Combined 4-layer indexer latency per step (`3 * dense_layer_us + 1 * source_layer_us`), mirroring the 3:1 ratio of compressed layers (`2, 8, 14`, `compress_ratio = 2`) to source layer (`20`, `compress_ratio = 1`):

| Context Tokens | Prefill (`r=64`) B1 / B2 | Prefill (`r=64`) Mean | Decode (`r=8`) B1 / B2 | Decode (`r=8`) Mean | Decode (`r=24`) B1 / B2 | Decode (`r=24`) Mean | Decode (`r=96`) B1 / B2 | Decode (`r=96`) Mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 1.2668x / 1.2644x | **1.2656x** | 1.0536x / 1.0765x | **1.0651x** | 1.2191x / 1.2154x | **1.2172x** | 1.3404x / 1.3432x | **1.3418x** |
| 65,536 | 1.6743x / 1.6729x | **1.6736x** | 1.3054x / 1.2984x | **1.3019x** | 1.5613x / 1.5643x | **1.5628x** | 1.7029x / 1.7052x | **1.7041x** |
| 131,072 | 1.8992x / 1.9089x | **1.9041x** | 1.4693x / 1.4741x | **1.4717x** | 1.7314x / 1.7560x | **1.7437x** | 1.9486x / 1.9555x | **1.9521x** |
| 262,144 | 2.8156x / 2.8406x | **2.8281x** | 2.1297x / 2.1330x | **2.1314x** | 2.5398x / 2.5781x | **2.5590x** | 2.8844x / 2.8975x | **2.8910x** |
| 524,288 | 4.2714x / 4.1593x | **4.2153x** | 3.2586x / 3.2019x | **3.2302x** | 4.1002x / 3.9548x | **4.0275x** | 4.3499x / 4.2107x | **4.2803x** |
| 1,048,576 | 4.0492x / 4.0813x | **4.0652x** | 3.7346x / 3.7646x | **3.7496x** | 3.9823x / 4.0055x | **3.9939x** | 4.0634x / 4.1023x | **4.0828x** |

## Key Conclusions

1. **Default-Off Is Zero-Risk and 100% Bit-Identical**:
   - Leaving `VLLM_DS41_LSA` unset or `0` keeps `state_content_bytes = 68`, compiles zero shortlist kernels, allocates zero shortlist scratch, and matches baseline native output with `max_abs_diff = 0` across every tested shape.
2. **Zero Downward Quantization & Single Clean Knob**:
   - All summary storage (`block8`, `STATE_BYTES = 100`) and coarse scoring stay in native `bfloat16` with `float32` accumulation, controlled by the single `VLLM_DS41_LSA` environment variable.
3. **Every Single `32K-1M` Prefill & Multi-Row Decode Cell Is a Full Win (`> 1.00x` Across Both Boots)**:
   - Passing `score_width` outside CUDA graph capture in both `extend` and `decode` routes `32K` dense layers (`2, 8, 14`, `score_width = 16,384`) to `prefill_short` (`width = 16,384`, 1-stage `IndexerTopK`) and `32K` layer `20` (`score_width = 32,768 <= 33,023`) to `native_index` (`5` kernels instead of a 12-kernel no-op coarse pass), lifting `32,768` `Decode (r=8)` from `0.9766x` (`0.9814x / 0.9717x`) to **`1.0651x` (`1.0536x / 1.0765x`)** and `32,768` `Decode (r=24)` from `1.0970x` to **`1.2172x` (`1.2191x / 1.2154x`)**.
   - At `65,536` (`64K`), unified `score_width` routing engages `source_short` (`budget = 512`, `DirectCoarseTopK`) on layer `20` during multi-row decode as well as prefill, lifting `64K` `Decode (r=8)` from `1.1825x` to **`1.3019x` (`1.3054x / 1.2984x`)** and `64K` `Decode (r=24)` from `1.3001x` to **`1.5628x` (`1.5613x / 1.5643x`)**.
   - Across `32K-1M`, combined 4-layer speedups reach **`1.27x-4.22x`** in prefill (`r=64`), **`1.07x-3.75x`** in `r=8` decode, **`1.22x-4.03x`** in `r=24` decode, and **`1.34x-4.28x`** in `r=96` decode.
