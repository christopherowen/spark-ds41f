# Cold-Boot / Warm-Boot Optimization on `tp3-triangle`

Date: 2026-11-22.
Base image: `spark-ds41f:candidate-65` (`sha256:4d6be6fa9e4086f60e2b5bc1beb4be9b1efc94a6ed0ff35c6ff45f5bc35d5c0a`, vLLM `0001..0844`, tree `34b01980b09e`).
Cluster: `dgx1..dgx3` (`tp3-triangle`, 16 seats; lab window `dgx1-1795347285`, production `spark-ds41f:1120` restored cleanly on close).

## Exact Baseline Stage Timeline (`candidate-65/tp3.json`, `API ready +114.4 s`, `97.10 s` inside container)

1. **Pre-launch (`cluster start` host stop + MemAvailable settle + `drop_caches/compact` + docker run)**: `11.1 s`
2. **Container launch → APIServer banner**: `~6.2 s`
3. **`APIServer → EngineCore → Worker_TP0` (`0.00 → 14.41 s`)**: `14.41 s` (`1.79 s` APIServer init + `3.94 s` `EngineCore` spawn + `8.68 s` `Worker_TP0` spawn — `dgx2` reached `dist_init_start` at `5.49 s` and waited `~9 s` for `dgx1`).
4. **NCCL / distributed init (`14.41 → 20.10 s`)**: `5.69 s` (`3.49 s` `pynccl` + `2.20 s` MP groups).
5. **Model skeleton + `bind_weight_arena` (`20.56 → 30.97 s`)**: `10.41 s` (`8.25 s` `create_model` including `90.6 GiB` throwaway MoE `torch.zeros` + `2.16 s` `bind_weight_arena`).
6. **Direct loader NVMe weight read (`30.97 → 49.41 s`)**: `18.44 s` (`100.59 GB` payload / `166.84 GB` disk read = `9.05 GB/s` disk; hardware floor for unsharded safetensors).
7. **Post-load weight processing (`49.47 → 67.39 s`)**: **`17.92 s`** (`18.31 s` on `dgx2`) — GPU work for 40 MoE layers was only `1.95 s` (`1.28 s` fused), while `~15 s` was 40 consecutive `release_device_memory_under_pressure` (`gc.collect + synchronize + empty_cache + malloc_trim`) cycles because `MemAvailable (~16 GiB) < (1 - 0.8) * 128 GiB (25.6 GiB)`.
8. **DSpark draft model load (`68.17 → 70.98 s`)**: `2.81 s`.
9. **TileLang kernel preparation (`71.36 → 83.75 s`)**: **`12.39 s`** — `prepare_declared` (`353` model calls) hit `0838`'s warm index in `1.27 s`, **but `TileLangKernels.compile_many()` still called `tilelang.prepare()` directly**, burning **`10.06 s`** re-elaborating all `302` step kernels from `declare_step_kernels()` on every boot.
10. **Encoder profile + KV cache alloc (`83.75 → 87.60 s`)**: `3.85 s`.
11. **Screened vocab warmup + CUDA graph capture (`87.60 → 96.05 s`)**: `8.45 s` (16-seat capture up to 96 rows).
12. **EngineCore tokenizer + Uvicorn ready (`96.05 → 97.10 s`)**: `1.05 s`.

## Isolated Arms & Live Measurements (`tp3-triangle`, `spark-ds41f:candidate-65`)

All surviving arms verified `0 / 4` greedy diffs (`128` tokens, `temperature=0.0`, full reasoning + content JSON equality) against the `candidate-65` baseline.

| Arm / Run | Commit | `API ready` (wall) | Inside container (`first_log → ready`) | `Model loading` | `_prepare_kernels` | `dgx1` `free_memory` | Greedy diffs | Verdict |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Baseline** (`candidate-65/tp3.json`) | `bcf59eb397` | `+114.4 s` | `97.10 s` | `50.59 s` | `12.4 s` | `~7.7 GiB` | ref | Baseline |
| **Arm 1** (`tp3-arm1-moe-postload.json`, cold index) | `98b8f24ce4` | `+114.0 s` | `97.24 s` | **`36.93 s`** (`-13.66 s`) | `24.8 s` *(rebuilt index after `moe.py` edit)* | `7.73 GiB` | `0 / 4` | **SURVIVED** (`-13.66 s` `load_model`) |
| **Arm 1+2 Boot 1** (`tp3-arm12-combined.json`, populates v2 index) | `98b8f24ce4` | `+111.9 s` | — | `37.07 s` | `25.0 s` *(0 fast, 353+301 indexed)* | `7.73 GiB` | — | Index populate |
| **Arm 1+2 Boot 2** (`tp3-arm12-combined.json`, warm v2 index) | `98b8f24ce4` | **`+89.3 s`** (`-25.1 s`) | **`71.54 s`** (`-25.56 s`) | **`36.97 s`** (`-13.62 s`) | **`2.7 s`** (`353` in `0.43 s` + `301` in `0.35 s`, **`-9.7 s`**) | `7.70 GiB` | `0 / 4` | **SURVIVED** |
| **Arm 1+2+3** (`tp3-arm123-defer-moe-alloc.json`) | `8f0bbfe854` | **`+88.1 s`** (`-26.3 s`) | **`70.67 s`** (`-26.43 s`) | **`34.23 s`** (`-16.36 s`, `create_model` `8.21 → 5.46 s`) | `2.7 s` (`0.43 + 0.35 s`) | `7.84 GiB` | `0 / 4` | **SURVIVED** |
| **Arm 1+2+3+4** (`tp3-arm1234-forkserver.json`) | `8f0bbfe854` | `+90.4 s` (`+2.3 s` vs Arm 3) | `72.72 s` | `34.2 s` | `2.7 s` | `7.39 GiB` (`-0.45 GiB`) | — | **DROPPED** (EngineCore spawns a 2nd forkserver + costs 450 MB RAM) |
| **Arm 1+2+3+5 Boot 1** (`tp3-arm1235-worker-fork.json`) | `b5c52d7c00` | **`+84.8 s`** (`-29.6 s`) | **`67.97 s`** (`-29.13 s`) | `35.17 s` | `2.8 s` (`0.44 + 0.35 s`) | **`8.07 GiB`** (`+0.37 GiB`) | `0 / 4` | **SURVIVED** (`Worker_TP0` `8.64 → 4.31 s`) |
| **Arm 1+2+3+5 Boot 2** (`tp3-arm1235-worker-fork.json`) | `b5c52d7c00` | **`+83.9 s`** (`-30.5 s`) | — | `35.06 s` | `2.8 s` (`0.43 + 0.34 s`) | **`7.97 GiB`** | `0 / 4` | **SURVIVED** (**2-boot mean `+84.35 s`, `-30.05 s` / `-26.3%`**) |

*(Note: an initial Arm 3 draft using `device="meta"` hit PyTorch's `variable.set_data` meta-to-CUDA type check and was replaced with a 1-byte CUDA `.expand(shape)` view in `8f0bbfe854`; an initial Arm 5 draft referenced unimported `sys.argv` and was switched to `parallel_config.numa_bind` in `b5c52d7c00`.)*

## Remaining Non-NVMe Floor Notes
- NVMe weight read (`18.2 s` for `100.59 GB` payload / `166.84 GB` disk read @ `9.15 GB/s`) is now the single largest stage.
- Remaining `process_weights_after_loading` (`5.07 s`) is dominated by `build_screen_weights` (`1.56 s` in PyTorch over 14 alphas; `40×` MoE fused retile + scale codes is `1.32 s`, `pack_bf16_weight` is `0.90 s`).
- All surviving optimizations (`Arm 1 + Arm 2 + Arm 3 + Arm 5`) are folded cleanly into `candidate-66` as vLLM patches `0846` and `0847`.
