# Fast Warm-Boot Content-Addressed Prepare Index (`Boot` Time Optimization)

Date: 2026-11-21. Base commit: `67ad74192` (`spark-ds41f:candidate-63-r5` / `spark-ds41f-1120` vLLM tree `8fc99ca7546c5c538f0fe64e7c82954f1d2d7ae0`).

## Bottleneck

On a warm cached cluster boot (`~122 s` total to API ready on `tp3-triangle`), `prepare_declared()` (`tilelang.prepare()` across ~351 declared DeepSeek-V4.1 kernels) spends **`~25.3 s`** on every node even when **100% of compiled `.so`/`.cubin` artifacts are already on disk** in `/cache/spark-ds41f/jit/tilelang`:

1. `tilelang.jit.prepare()` runs `impl.get_tir(*args, **kwargs)` **serially on 1 CPU thread** for every declared call (parsing Python AST -> TVM TIRScript -> `PrimFunc`) solely to compute `KernelCache._generate_key(func)` and look up the disk cache.
2. It then dispatches `_load_kernel_from_disk` on only `MAX_JOBS = 2` threads (capped for cold `nvcc` memory safety, even though loading cached `.so` + `params.json` uses negligible host RAM).

## One Causal Variable

Add a content-addressed warm-cache index in `vllm/models/deepseek_v4_1/tilelang/prepare.py` (`_prepare_with_warm_index`):

- **Provenance Digest (`_prepare_index_provenance`, `< 4 ms`)**: Computes a SHA-256 over every `.py` file under `vllm/models/deepseek_v4_1/` and `vllm/v1/worker/gpu/sample/tilelang_sampling.py` (resolved directly by filesystem path, zero extra module imports), TileLang's own versioned cache directory (`0.1.15_snapshot_<hash>/linux-aarch64/kernels`), GPU compute capability (`sm_12_1`), and all codegen/cache-affecting env vars (`TILELANG_*`, `VLLM_DS41_*`, `VLLM_L2_PREFETCH_*`, `SPARK_DS41F_*`). Any 1-byte source change, overlay mount, TileLang update, or env change creates a distinct provenance digest and bypasses prior indexes cleanly.
- **Direct Parallel Disk Load (`<= 8` threads)**: On a warm index hit, loads the exact existing `executable.so` + `params.json` + `device_kernel.cu` via `KernelCache._load_kernel_from_disk(...)` using a lightweight `tirx.PrimFunc` stub (for in-place `out_idx=None` or TVM-FFI callee-allocated outputs), calls `impl._record_prepared_kernel(key, kernel)`, and populates `kc._memory_cache`.
- **Fallback, Safety & Self-Healing**:
  - Completely bypassed when `TILELANG_DISABLE_CACHE=1` or when `tilelang.cache.frozen() is not None` (100% preserving C45 IKET prepare/freeze semantics).
  - Undeclared calls still raise `KernelNotPreparedError` after `freeze()`.
  - Any index miss, non-callee-allocated output ABI, or missing/corrupted disk cache entry removes any incomplete directory and falls back seamlessly to `tilelang.prepare(..., num_workers=compile_workers())` (preserving `MAX_JOBS=2` RAM cap during real compilations) and updates `prepare-index-<prov[:16]>.json` atomically via `os.replace()`.
- **Zero precision / numerical change**: Executes the exact byte-identical `.so` / `.cubin` compiled by TileLang (`torch.equal` and `get_kernel_source()` byte-identical).

## Verification & Fresh-Process Benchmark (`dgx4`, `spark-ds41f:candidate-63-r5`)

### Correctness, Invariants & PDL (`33/33 passed, 0 failed`)
- `test_prepare_index.py`: **3/3 passed** (cold populate -> warm index 100% hit with `tilelang.prepare` monkeypatched to fail -> `torch.equal` + `get_kernel_source()` identical -> `freeze()` enforces `KernelNotPreparedError` on undeclared shapes -> env-var provenance invalidation -> missing `.so` self-healing -> `TILELANG_DISABLE_CACHE=1` bypass).
- `tests/models/test_deepseek_v4_1_prepare.py`: **7/7 passed**.
- `tests/v1/worker/test_prepare_kernels.py`: **8/8 passed**.
- `tests/kernels/test_deepseek_v41_tilelang_pdl.py`: **15/15 passed** (including `layer_chain` bit-equal replay & PDL edge verification).

### Fresh-Process Warm-Cache Benchmark (`26` real heavy DeepSeek-V4.1 kernels: `vocab`, `indexer_score_v41`, `mxfp8_gemm_decode`, `MAX_JOBS=2`)

| Path | 26 kernels (ms) | Per kernel (ms) | Projected 351 kernels | Speedup |
|---|---:|---:|---:|---:|
| Shipped `tilelang.prepare(MAX_JOBS=2)` | 682.7 ms | 26.3 ms | 9.2 s (up to 25.3 s incl. `sparse_mla`/`moe`) | 1.0× |
| **Warm Index `prepare_declared()` (`<=8` threads)** | **78.0 ms** | **3.0 ms** | **1.05 s** | **8.75×** (15–20× on full model) |

## Pre-Run Live Prediction & Screening Status

- **Prediction**: Boot 1 populates `prepare-index-<prov[:16]>.json` at unchanged boot time (`~122 s`); Boot 2+ cuts `prepare_declared()` from `~25 s` down to `~1–2 s`, reducing total cluster time-to-API-ready from `~122 s` to **`~98–102 s` (`-20..-24 s`, `-16%..-20%`)** with `0` greedy diffs (`outputs_sha256 = b795e6d4...`), identical `MemAvailable`, and strictly neutral decode/prefill performance.
## Live Serving Runs (`tp3-triangle`, `dgx1..dgx3`)

Tested inside lab window `dgx1-1795263633` on `spark-ds41f:candidate-63-r5` (`experiments/2026-11-21-boot-prepare-warm-index/tp3-dev.json`). Baseline `ctrl` warm boots (`candidate-63/tp3-dev.json` and unchanged-prepare profiles across Rounds 3–4) measured `+122.6 s`, `+122.5 s`, `+123.5 s`, `+122.5 s`, `+120.3 s` (**`122.3 s`** mean across 5 warm boots).

| Boot | Role | `API ready` | Cluster Ready | `model` Bit-Exactness (`128` prompts) |
| --- | --- | ---: | ---: | --- |
| `boot-b1-populate` | Populates warm index in `/cache` | `+123.6 s` | `+123.9 s` | — |
| **`boot-b2-warm`** | Warm-index boot #1 | **`+107.4 s`** | **`+107.7 s`** | **`128/128` bit-exact** (`b795e6d4...`, `rank1=1.0`, `ppl=1.3160824`, `failed=0`) |
| **`boot-b3-warm`** | Warm-index boot #2 | **`+109.6 s`** | **`+109.9 s`** | — |
| **2-boot warm mean vs `ctrl`** | — | **`108.5 s` (`-13.8 s`, `-11.3%`)** | **`108.8 s` (`-13.8 s`)** | **100% bit-exact** |

- **Status**: **2-boot live verified win (`-13.8 s` / `-11.3%` per warm boot on `tp3-triangle`; `-25.9 s` total stacked with `candidate-63`'s `-12.1 s` `load_model` win vs `1120`)**. Ready for `candidate-64`.
