# 2026-11-15-c59-prefill (candidate-59)

- **Base commit**: `bca856d25` (`candidate-58`)
- **Branch**: `c59-prefill`
- **Cluster profiles**: `tp4-switched-c59-dev.json`, `tp3-switched-c59-dev.json`, `tp4-c59-dev.json`
- **Goal**: Stepwise prefill lift with zero prefill, decode, determinism, or quality regressions vs `candidate-58`.

## Step 1 — Targeted Host/Decompress PR Bundle + Dense Indexer Paired-Row Reduction

### Changes
1. **Bounded UTF-8 suffix detokenizer priming**:
   - `overlay/vllm/vllm/v1/engine/detokenizer.py`, `overlay/vllm/tests/detokenizer/test_check_stop_strings.py`.
2. **Zero-copy `int32` `PickleBuffer` IPC for `NewRequestData` admission**:
   - `overlay/vllm/vllm/v1/core/sched/output.py`, `overlay/vllm/vllm/v1/request.py`, `overlay/vllm/vllm/v1/worker/gpu/buffer_utils.py`, `overlay/vllm/tests/v1/core/test_new_request_data_ipc.py`.
3. **BF16x2 dequant + CUDA 13.5/PTX 9.2 `cvt.rn.bf16x2.e2m1x2` in `sparse_mla.py`**:
   - `overlay/vllm/vllm/models/deepseek_v4_1/tilelang/sparse_mla.py`, `overlay/vllm/tests/kernels/attention/test_deepseek_v41_tilelang.py`.
4. **Dense `indexer_score_v41` paired-row SMEM reduction & `2 <= rows < 64` `R=2` key-tile reuse** (without modifying `attention.py`):
   - `overlay/vllm/vllm/models/deepseek_v4_1/tilelang/indexer.py`:
     - Routed `2 <= rows < 64` to `__tl_score_rows_dense_fast<2>` (keeping `<1>` for `rows < 2` and `<4>` for `rows >= 64`).
     - Expanded `contrib` shared memory to `[2 * block_P, heads]` (`[128, 32]` float32) and paired consecutive rows inside `same_k` tiles (`rp += 2`, writing `sub=0,1` before a single `__syncthreads()`, then reducing all 128 threads concurrently in strictly increasing head order `h = 0..31` with half as many barriers and zero ULP diffs).
     - Also screened adding `__tl_score_rows_dense_fast<8>` for `rows >= 128`; discarded `<8>` because instantiating `B_reg[8]` in the single kernel increased register usage across all dispatch branches and slowed `rows=1` (`13.6 -> 19.7 us`), `rows=64` (`302.2 -> 353.3 us`), and `rows=4096` (`19987.6 -> 20145.0 us`).

### Verification & Microbenchmark (`sp8`, GB10 `sm_121a`, `spark-ds41f:candidate-55` + Step 1 overlay)

- Detokenizer + IPC unit tests (`pytest -q -p tblib_stub tests/detokenizer/test_check_stop_strings.py tests/v1/core/test_new_request_data_ipc.py`): `17 passed in 3.69s`.
- Stop string termination test (`pytest -q -p tblib_stub -p offline_skip tests/detokenizer/test_stop_string_while_stop_model_terminates.py`): `2 passed in 0.36s`.
- Sparse MLA + BF16x2 + Indexer + PDL source audit (`pytest -q -p tblib_stub -p offline_skip -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/attention/test_deepseek_v41_tilelang.py -k "sparse_mla or bf16x2 or indexer or pages"`): `36 passed, 1 skipped, 34 deselected, 1 xfailed in 33.21s` (`PDL sources: 39 kernels launched, 33 with PDL, 0 violations`).
- `IndexerV41Plan` (`kv=65536`, `page_size=64`) bit-exactness check vs `candidate-58` saved reference (`idx_ref_65k.pt`): **0 diffs across all tested shapes**.

| Shape (`kv=65536`, `causal`, `span`) | `c58` score (`us`) | `c59` `<1,2,4,8>` paired (`us`, dropped) | `c59` `<1,2,4>` paired (`us`) | Speedup (`score`) |
|---|---:|---:|---:|---:|
| `rows=1, causal=0, span=1` | `13.6` | `19.7` | `13.5` | `+0.7%` |
| `rows=2, causal=0, span=2` | `21.7` | `23.8` | `15.6` | **`1.39x`** |
| `rows=6, causal=0, span=6` | `54.5` | `48.1` | `38.1` | **`1.43x`** |
| `rows=12, causal=0, span=6` | `103.2` | `91.2` | `74.3` | **`1.39x`** |
| `rows=24, causal=0, span=6` | `193.8` | `179.4` | `134.4` | **`1.44x`** |
| `rows=48, causal=0, span=6` | `380.0` | `347.3` | `261.3` | **`1.45x`** |
| `rows=64, causal=0, span=64` | `302.2` | `353.3` | `285.9` | **`1.06x`** |
| `rows=65, causal=0, span=65` | `310.4` | `363.6` | `294.1` | **`1.06x`** |
| `rows=256, causal=0, span=256` | `1221.7` | `1300.5` | `1153.3` | **`1.06x`** |
| `rows=1024, causal=0, span=1024` | `4745.3` | `4781.0` | `4517.0` | **`1.05x`** |
| `rows=1024, causal=1, span=1024` | `4711.5` | `4754.3` | `4490.5` | **`1.05x`** |
| `rows=4096, causal=1, span=4096` | `19987.6` | `20145.0` | `18815.1` | **`1.06x`** (`-1.17 ms`) |

### Step 1 Cluster Gate (`sp1`, `tp4-switched`, commit `967b99435`)

- Receipts archived in `receipts/step1/tp4-switched/`.
- **`screen` (`20261115T221417-screen-tp4-switched`, 2 boots `c59`, `c59-again` vs `c58-final` recorded)**:
  - `model` (`c59` -> `c59-again` + recorded determinism): `128/128` greedy identical; `100.00%` identical distributions over `32,768` positions (`128` sequences), `KL mean 0.000e+00`, perplexity `1.31888 -> 1.31888`.
  - `prefill` (`default-text`): `32,768`: `9320.125` / `9284.973` (`+1.37% ±1.86%`, `same`, margin `2.05%`); `262,144`: `9592.128` / `9605.529` (`+2.01% ±0.76%`, **`better`**, margin `0.98%`).
  - `c16`: `output tok/s` `268.580` / `268.400` (`-0.02% ±0.91%`, `same` vs `c58-final` `268.720` / `268.343`); `decode tok/s, paired` `-0.06% ±0.31%` (`same`).
  - `grid`: `decode geometric mean (20)` **`+0.34% ±0.14%` worse** (`+0.47%` / `+0.36%`), concentrated in `@4K` medium-batch shapes (`8x16 @4K` `+0.58%/+2.26%`, `8x24 @4K` `+1.53%/+1.14%`, `8x48 @4K` `+1.42%/+1.74%`).
- **`bench` (`20261115T221457-bench-prefill-tp4-switched`, `prefill` `--prefill-text source`, 2 boots `b1`, `b2`)**:
  - `32,768`: `b1 = 9472.7` (`9450.1`..`9495.3`), `b2 = 9406.4` (`9403.6`..`9409.2`) (vs `c58-final` `9385.9` / `9423.6`).
  - `131,072`: `b1 = 9848.1` (`9793.3`..`9903.0`), `b2 = 9830.8` (`9777.2`..`9884.5`) (vs `c58-final` `9742.7` / `9779.6`).
  - `262,144`: `b1 = 9706.9` (`9673.8`..`9739.9`), `b2 = 9680.1` (`9653.4`..`9706.8`) (vs `c58-final` `9634.7` / `9627.5`).
  - Memory/thermals (`b1`/`b2`): `sp1` `min_available_gib` `16.97` / `16.93 GiB` (`swap_growth_gib = 0.0`); `max_gpu_c` up to `73.0 C` (`b1`) / `72.0 C` (`b2`), `thermal_slowdown_ms = 0.0`, `power_cap_ms = 0.0` across `sp1..sp4`.

### Step 1 Decode Regression Fix (`indexer.py`)

- **Root cause**: Routing `2 <= rows < 64` to `__tl_score_rows_dense_fast<2>` improved synthetic dense-only shapes (`span <= 6`) on `sp8`, but in real serving `@4K` decode it serialized cross-request queries, forced halted-thread CTAs on ragged KV spans, and doubled `contrib` SMEM from `8 KiB` to `16 KiB`, hurting `@4K` batch decode occupancy.
- **Fix**:
  - Restored `row_blocks = T.if_then_else(rows >= 64, T.ceildiv(rows, 4), rows)` and removed the `<2>` branch (`2 <= rows < 64` uses `<1>` again; `rows >= 64` keeps `<4>`).
  - Kept `contrib` at `[block_P, heads]` (`8 KiB` float32 SMEM) and stored the `2`-row partial sums into `bfloat16_t* contrib_bf16` inside `__tl_score_rows_dense_fast` (`rows >= 64` only).
- **Verification (`sp8`)**: `pytest -q -p tblib_stub -p offline_skip -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/attention/test_deepseek_v41_tilelang.py -k "indexer"` -> `12 passed, 1 skipped, 57 deselected, 1 xfailed in 55.20s` (`PDL sources: 15 kernels launched, 15 with PDL, 0 violations`).

## Step 2 — Fuse `prefill_gate_up` -> `moe_swiglu_cast` in `moe.py`

### Changes
- Added 64-col `(gate, up)` slab interleaving (`_interleave_gate_up_slabs`) for `prefill_gate_up` weights (`w13`) so adjacent 128-col output blocks hold `[gate_64, up_64]` pairs.
- Fused SwiGLU (`w_row` scaling, BF16 cast rounding, warp-level `32`-col MXFP8 block-scale reduction) directly into `grouped_fp8_fp4_gemm(..., fuse_swiglu=True)` via `C_shared` (`T.bfloat16`) and C++ `__tl_fused_swiglu_epilogue<BLOCK_M, ROW_STEP, HAS_LIMIT>` with `if (rows_shared[ep_i] < 0) break;` early exit on padding rows, eliminating the prefill `h` buffer and separate `moe_swiglu_cast` kernel launch.
- Updated `overlay/vllm/tests/kernels/moe/test_deepseek_v41_tilelang_moe.py` for slab-interleaved fused epilogue parity and prefill tests.

### Verification & Microbenchmark (`sp8`, GB10 `sm_121a`, `spark-ds41f:candidate-55` + Step 2 overlay)

- `pytest -q -p tblib_stub -p offline_skip -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/moe/test_deepseek_v41_tilelang_moe.py`: **`140 passed, 14 warnings in 202.19s`** (`PDL sources: 143 kernels launched, 135 with PDL, 0 violations`).
- `nvcc -cubin -arch=sm_121 -Xptxas=-v` on fused `moe_grouped_fp8_fp4_gemm` (`inter=576` and `768`): `Used 81 registers, used 1 barriers, 1024 bytes smem`, `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`.
- 25-iteration interleaved `MoeExpertsV41Plan.run` A/B median vs `c58` (`E=384, topk=6, H=5120`, bit-exact `uint16_diffs=0`, `max_abs=0.000e+00` on all 6 shapes):

| Shape | `c58` (`us`) | `c59` Step 2 (`us`) | Delta / Speedup | Bit diffs |
|---|---:|---:|---:|---:|
| `inter=576, rows=2048` | `9291.2` | `8908.5` | **`-382.7 us` (`1.043x`)** | `0` |
| `inter=576, rows=4096` | `11130.7` | `10687.6` | **`-443.0 us` (`1.041x`)** | `0` |
| `inter=576, rows=6144` | `13327.1` | `12637.0` | **`-690.1 us` (`1.055x`)** | `0` |
| `inter=768, rows=2048` | `11467.0` | `11470.9` | `+3.9 us` (`1.000x`) | `0` |
| `inter=768, rows=4096` | `13489.0` | `13066.5` | **`-422.5 us` (`1.032x`)** | `0` |
| `inter=768, rows=6144` | `16321.5` | `15951.3` | **`-370.1 us` (`1.023x`)** | `0` |

### Step 2 Cluster Gate (`sp1`, `tp4-switched`, commit `336ba005a`)

- Receipts archived in `receipts/step2/tp4-switched/`. Includes both the `indexer.py` `rows >= 64` decode fix and the fused `prefill_gate_up -> moe_swiglu_cast` kernel.
- **`screen` (`20261115T230729-screen-tp4-switched`, 2 boots `c59`, `c59-again` vs `c58-final` recorded)**:
  - `model` (`c59` -> `c59-again`): `128/128` greedy identical; `100.00%` identical distributions over `32,768` positions (`128` sequences), `KL mean 0.000e+00`, perplexity `1.31888 -> 1.31888`.
  - `prefill` (`default-text`): `32,768`: `9224.952` / `9385.513` (`+1.40% ±5.80%`, `same`, repeat `+1.74%`, boot 2 `9385.513` `+2.28% ±1.01%` better); `262,144`: `9572.537` / `9627.823` (`+2.02% ±0.76%`, **`better`**, margin `1.11%`).
  - `c16`: `output tok/s` `268.967` / `268.820` (`+0.13% ±0.87%`, `same` vs `c58-final` `268.720` / `268.343`); `decode tok/s, paired` `+0.09% ±0.46%` (`same`).
  - `grid`: `decode geometric mean (20)` `+0.24% ±0.13%` (`margin 0.27%`, repeat `+0.07%`, **`same`** vs `c58-final`) — Step 1 `@4K` regression eliminated.
- **`bench` (`20261115T230732-bench-prefill-tp4-switched`, `prefill` `--prefill-text source`, 2 boots `b1`, `b2`)**:
  - `32,768`: `b1 = 9483.11` (`9449.52`..`9516.71`), `b2 = 9487.26` (`9444.32`..`9530.20`) (vs `c58-final` `9385.90` / `9423.59`, Step 1 `9472.72` / `9406.39`).
  - `131,072`: `b1 = 9880.00` (`9847.28`..`9912.72`), `b2 = 9852.12` (`9830.05`..`9874.19`) (vs `c58-final` `9742.71` / `9779.57`, Step 1 `9848.14` / `9830.82`).
  - `262,144`: `b1 = 9723.23` (`9681.61`..`9764.84`), `b2 = 9693.67` (`9685.57`..`9701.77`) (vs `c58-final` `9634.65` / `9627.49`, Step 1 `9706.88` / `9680.10`).
  - Memory/thermals (`b1`/`b2`): `sp1` `min_available_gib` `17.25` / `17.21 GiB` (`swap_growth_gib = 0.0`); `max_gpu_c` up to `73.0 C` (`b1`) / `75.0 C` (`b2`), `thermal_slowdown_ms = 0.0`, `power_cap_ms = 0.0` across `sp1..sp4`.

## Step 3 — PR `#1005` MoE Dispatch Scan, Sync, and SwiGLU Preload Optimization

### Changes
- Cherry-picked PR `#1005` onto `overlay/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py` and `overlay/vllm/tests/kernels/moe/test_deepseek_v41_tilelang_moe.py`, while preserving `c58` BF16 SwiGLU intermediate rounding (`x_bf16[j] = T.cast(val, in_dtype); x[j] = T.cast(x_bf16[j], T.float32)`), Step 2's `slab_interleaved` layout and `fuse_swiglu` epilogue, explicit `T.sync_threads()` both before and after `C_shared` in `fuse_swiglu`, and `if (gather or trigger_after_loop) and expert < 0:` early return:
  - Replaced `_dispatch_macros.layout()` 9-step shared-memory Hillis-Steele scan (25 `__syncthreads()` in fused `moe_dispatch_v41`) with two-level warp-shuffle prefix scan (7 `__syncthreads()`).
  - Enabled `TL_DISABLE_SAFE_MEMORY_ACCESS` across `moe_dispatch_v41`, `moe_combine_v41`, `grouped_fp8_fp4_gemm`, and `moe_swiglu_cast`, plus `TL_DISABLE_THREAD_STORAGE_SYNC` on `grouped_fp8_fp4_gemm`.
  - Removed redundant `T.ptx_commit_group()` / `T.ptx_wait_group(stages - 1)` in `grouped_fp8_fp4_gemm` (already synchronized by `loaded[s]`), switched `block_M == 16 and sfb_copies == 1` coded-scale expansion sync from `T.sync_threads()` to `T.sync_warp()`, and unified non-gather metadata preload before `pdl.wait()` / `pdl.chain()`.
  - Added cascaded `orders[0]` / `orders[1]` / `Weights` preloading and early empty-slab exit in `moe_swiglu_cast` via `@T.macro def compute_row(...)`.

### Verification & Microbenchmark (`sp8`, GB10 `sm_121a`, `spark-ds41f:candidate-55` + Step 3 overlay)

- `pytest -q -p tblib_stub -p offline_skip -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/moe/test_deepseek_v41_tilelang_moe.py tests/models/test_deepseek_v4_1_l2_schedule.py`: **`157 passed, 14 warnings in 197.27s`** (`PDL sources: 143 kernels launched, 135 with PDL, 0 violations`).
- 25-iter interleaved eager `MoeExpertsV41Plan.run` A/B median (`Step 2` vs `Step 3`, `E=384, topk=6, H=5120`, `uint16_diffs=0`, `max_abs=0.000e+00` on all 16 shapes):
  - `inter=576`: `rows=1` `120.9 -> 130.5 us` (`+9.5`), `rows=6` `650.9 -> 633.1 us` (`-17.8`), `rows=24` `2370.8 -> 2346.5 us` (`-24.3`), `rows=64` `4562.6 -> 4499.2 us` (`-63.4`), `rows=1024` `7961.6 -> 7853.2 us` (`-108.5`), `rows=2048` `8772.0 -> 8650.4 us` (`-121.6`), `rows=4096` `10365.0 -> 10320.0 us` (`-45.0`), `rows=6144` `12453.7 -> 12509.7 us` (`+56.0`).
  - `inter=768`: `rows=1` `406.5 -> 410.8 us` (`+4.3`), `rows=6` `805.3 -> 808.7 us` (`+3.4`), `rows=24` `2889.7 -> 2900.9 us` (`+11.3`), `rows=64` `5715.3 -> 5740.5 us` (`+25.2`), `rows=1024` `10020.5 -> 10035.1 us` (`+14.6`), `rows=2048` `11130.0 -> 11215.9 us` (`+85.8`), `rows=4096` `13293.4 -> 13090.3 us` (`-203.1`), `rows=6144` `15938.5 -> 15749.7 us` (`-188.9`).
- 30-iter interleaved CUDA-graph decode `MoeExpertsV41Plan.run` A/B median (`Step 2` vs `Step 3`, `uint16_diffs=0` on all 12 shapes):
  - `inter=576`: `rows=1` `121.7 -> 100.5 us` (**`-21.2 us` / `1.211x`**), `rows=6` `612.1 -> 603.5 us` (`-8.6`), `rows=12` `1266.5 -> 1253.1 us` (`-13.5`), `rows=24` `2348.1 -> 2324.0 us` (`-24.1`), `rows=48` `3582.7 -> 3531.7 us` (`-51.0`), `rows=64` `4549.4 -> 4477.0 us` (`-72.4`).
  - `inter=768`: `rows=1` `152.2 -> 140.4 us` (**`-11.8 us` / `1.084x`**), `rows=6` `773.0 -> 773.8 us` (`+0.8`), `rows=12` `1599.8 -> 1585.0 us` (`-14.8`), `rows=24` `2900.8 -> 2885.6 us` (`-15.2`), `rows=48` `4819.9 -> 4781.8 us` (`-38.1`), `rows=64` `5724.5 -> 5691.8 us` (`-32.7`).
