# Full 43-Layer Forward-Pass IKET Recording and IKET Runtime Optimizations

## Overview

This experiment completes the full 43-layer (`40` target layers + `3` DSpark speculative drafter layers) DeepSeek-V4.1-Flash forward-pass IKET recording on `dgx4` (TP4 rank shard shape: `16` attention heads, `2` `o_groups`, `moe_intermediate_size=576`, `c1` decode with `5` speculative drafts = `6` verified rows/step) and measures four concrete runtime and memory optimizations to the IKET profiling pipeline itself.

## 1. End-to-End 43-Layer Recording & IKET Runtime Improvements

All runs execute the same 43-layer configuration (`FULL_TEXT_OVERRIDES`: `2` SWA + `18` CSA + `20` HCA target layers + `3` DSpark drafter SWA layers, `VLLM_LATE_ADMISSION_LAYER=30`) on a single `dgx4` GPU with deterministic dummy weights (`uint8_fill=0`), `1` 32-token prompt, `8` generated tokens (`1` prefill step + `5` speculative decode steps between `iket_step_marker` launches), followed by `5` `engine_grid` steps (`1` warmup + `4` timed replays at `context_len=36, num_tokens=6`).

| Metric | Run 1: Plain (`mode=off`) | Run 2: Base (`--warp-trace`) | Run 3a: Lean IKET (2-Pass) | Run 3b: Lean IKET (1-Pass + Lean Startup) | Improvement (`3a` -> `3b`) |
|---|---:|---:|---:|---:|---:|
| Total profiling wall time (`wall_s`) | `36.1 s` (warm) / `158.4 s` (cold) | `243.48 s` | `384.52 s` | **`71.96 s`** | **`5.34x` faster** |
| Engine ready time (`boot_s`) | `28.0 s` (warm) / `158.4 s` (cold) | `28.7 s` | `223.9 s` (dry) / `41.8 s` (measured) | **`27.4 s`** | **`1.53x` faster warm (`8.17x` vs cold dry)** |
| Instrumented TileLang compile (248 calls) | N/A | N/A | `156.3 s` (cold) -> `12.0 s` (warm cache) | **`12.0 s`** (warm cache) | **`13.0x` faster** |
| `run-iket` passes required | `1` (unprofiled) | `1` (`--warp-trace`) | `2` (`tracker` + `iket`) | **`1` (`--single-pass-max-ts 42`)** | **Eliminates `tracker` + `gen-config` pass** |
| Max timestamp slots / warp (`maxTsCntPerWarp`) | `0` | `0` (`24 B`/warp) | `72` (`616 B`/warp) | **`42` (`376 B`/warp)** | **`1.64x` fewer bytes/warp** |
| Device context buffer pool (`pool_gb`) | `0.00 GB` | `4.50 GB` | `25.09 GB` | **`8.50 GB`** | **`2.95x` smaller device pool** |
| Total launches recorded | — | `39,537` | `36,008` | **`33,016`** | `-2,992` non-measured startup launches |
| Total warps recorded | — | `94,994,313` | `39,577,461` | **`20,462,180`** | **`1.93x` fewer total warps (`-19.12M`)** |
| Measured window launches (`iket_step_marker`) | — | `8,355` (all kernels) | `8,289` (TileLang) | **`8,289` (TileLang)** | **100.0% identical measured launches** |
| Measured window warps (`iket_step_marker`) | — | `5,918,949` | `5,915,989` | **`5,915,989`** | **100.0% identical measured warps** |
| `iket.decoded_results.json` size | — | `8,880.42 MiB` | `3,084.77 MiB` | **`1,517.21 MiB`** | **`2.03x` smaller decoded JSON** |
| `iket.data.json` size (`--skip-raw-data-dump`) | — | `0.0 MiB` (`34 B`) | `0.0 MiB` (`34 B`) | **`0.0 MiB` (`34 B`)** | **Saves `1.5–4.5 GiB` disk I/O** |
| `engine_grid` decode `step_ms` (mean of 4) | `19.91 ms` | `49.12 ms` | `71.12 ms` | **`54.56 ms`** | **`-16.56 ms/step` (`1.11x` of `--warp-trace`)** |
| `engine_grid` target `forward_ms` (mean of 4) | `11.77 ms` | `20.45 ms` | `44.34 ms` | **`29.41 ms`** | **`-14.93 ms/step`** |
| `engine_grid` `drafter_ms` (mean of 4) | `6.89 ms` | `19.23 ms` | `22.39 ms` | **`21.82 ms`** | **`-0.57 ms/step`** |
| Generated `token_ids` (8 tokens) | `[88916, 68371, ...]` | `[88916, 68371, ...]` | `[88916, 68371, ...]` | **`[88916, 68371, ...]`** | **100% bit-for-bit identical** |

## 2. Root-Cause Breakdown of IKET Overhead & Optimizations

1. **Where the 25.09 GB Context Buffer Pool Came From (43-Layer `tracker.json` Analysis):**
   - Inspecting `run3a_2pass/dry/tracker/pid_0xd6/tracker.json` (`39,577,461` total warps) showed that **`85.1%` (`33,661,472` warps)** occurred outside the `iket_step_marker` measured window (`5,915,989` warps).
   - Specifically, three `512`-token (`max_num_batched_tokens=512`) dummy prefill passes during worker startup accounted for **`23,680,239` warps (`59.8%` of the entire pool, `15.02 GB` at `ts=72`)**:
     - `GPUModelRunner.profile_run` (`skip_attn=True`, 512 tokens): `7,048,318` warps (`3.89 GiB` at `ts=72`), dominated by 40 layers of `moe_grouped_fp8_fp4_gemm_kernel` (`grid=(688, 36, 1)` = `99,072` warps/launch) and `moe_swiglu_cast_kernel_kernel`.
     - `GPUModelRunner._profile_deepseek_v4_attention` (512 tokens): `7,064,570` warps (`3.90 GiB` at `ts=72`), which exists to measure transient peak activation memory for automatic KV cache admission (`memory_profiling`), even though `--kv-bytes` (`kv_cache_memory_bytes`) explicitly skips memory profiling.
     - `kernel_warmup.flashinfer_autotune` (512 tokens): `9,567,351` warps (`5.28 GiB` at `ts=72`), which runs a full 512-token dummy prefill across all 43 layers and logs `Saved 0 configs` because the TileLang attention/linear/MoE backends are active (production `config/cluster.json` passes `--no-enable-flashinfer-autotune`).
   - By contrast, a 43-layer `c1` decode step (`6` tokens) launches only **`~480,000` warps (`0.17 GiB` at `ts=42`)** — **20x fewer warps** than a 512-token dummy prefill pass.
2. **Startup Dummy-Run & Warmup Optimization under `--lean-capture` (`harness/iket_step.py`):**
   - Pre-reserves the full `max_num_tokens` (`512`) mHC and extend-attention workspace specs in `current_workspace_manager()` across both target and drafter workspace lanes before `profile_run`, so workspace buffer byte sizes are unchanged.
   - Skips `_profile_deepseek_v4_attention` when `kv_cache_memory_bytes` (`--kv-bytes`) is explicitly configured and caps `profile_run`'s `_dummy_run` token count to `min(max_num_tokens, max(64, requests * 8))`.
   - Disables `flashinfer_autotune` and skips `run_mixed_prefill_decode_warmup` when `requests < 2`.
   - **Impact:** Eliminates **`19,115,281` non-measured startup warps (`-48.3%` of all warps)**, cuts `boot_s` from `41.8 s` to `27.4 s`, and halves `iket.decoded_results.json` (`3,084.77 MiB` -> `1,517.21 MiB`) and `IKET::Finalize` exit decode time.
3. **DMA Weight Initialization (`same_dummy_weights` in `harness/iket_step.py`):**
   - Replaces `ATen FillFunctor` (`param.zero_()`, `param.fill_()`, `torch.zeros()`) with `cudaMemsetAsync` and caches prepared MoE/MXFP8 scale tensors via `cudaMemcpyAsync(DeviceToDevice)`.
   - Prevents `run-iket --warp-trace` from recording **`151.3M` weight-init warps (`3.64 GB` of device pool)** across the 43 layers.
4. **1-Pass Tracker-Free Mode (`--single-pass-max-ts`) & Raw-Data Bypass (`--skip-raw-data-dump` in `harness/iket_profile.py`):**
   - `libsmodel_injection.so` auto-discovers `.iket` ELF sections in instrumented TileLang cubins and ignores per-site `instrumentations` in `instrument.config.json` for `kNativeDump`. Writing a minimal stub config with `maxTsCntPerWarp: 42` (above the 43-layer static bound of `38` events/warp verified in `trace_counts_43l.txt`) eliminates the entire `tracker` dry run and `gen-config` pass.
   - Binary-patching the two `IKET::Finalize` raw-dump branch sites in a local copy of `libsmodel_injection.so` (`--skip-raw-data-dump`) skips serializing and writing the multi-GB `iket.data.json` while keeping `iket.decoded_results.json` intact.

## 3. Full 43-Layer Decode Step Breakdown (5 Decode Steps, `1,404` TileLang Launches / Step)

Across the 5 decode steps between `iket_step_marker` launches (`7,017` TileLang launches of `115` kernel shapes, `1,404` TileLang launches per step out of `1,415` total GPU kernels per step), the per-step kernel span sums to **`47,764.0 us/step`** under uninstrumented `--warp-trace` ([`table_43l_base.md`](table_43l_base.md)) and **`49,462.2 us/step`** under 1-warp phase-granularity IKET ([`table_43l.md`](table_43l.md)) — only **`+3.55%` total kernel span difference**:

| Kernel | Grid | Launches / Step | `--warp-trace` Span / Launch (`us`) | Lean IKET Span / Launch (`us`) | Lean IKET Span / Step (`us`) | Share | Last Warp: `wait` | `load` | `compute` | `store` | `unattributed` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `l2_prefetch_kernel` | `(2, 1, 1)` | `119` | `69.18` | `71.10` | `8,314.7` | `16.8%` | `0%` | `99%` | `0%` | `0%` | `1%` |
| `vocab_packed_projection_kernel` | `(1, 4040, 1)` | `2` | `4,077.82` | `4,017.57` | `8,037.3` | `16.2%` | `73%` | `9%` | `9%` | `0%` | `9%` |
| `moe_grouped_fp8_fp4_gemm_kernel` (`gate_up`) | `(9, 36, 1)` | `40` | `117.58` | `118.29` | `4,666.6` | `9.4%` | `66%` | `23%` | `17%` | `1%` | `0%` |
| `moe_grouped_fp8_fp4_gemm_kernel` (`down`) | `(40, 36, 1)` | `40` | `69.23` | `75.15` | `2,955.6` | `6.0%` | `17%` | `45%` | `14%` | `3%` | `21%` |
| `mhc_project_streams_kernel` | `(40, 1, 1)` | `86` | `20.18` | `23.49` | `2,244.5` | `4.5%` | `37%` | `23%` | `11%` | `2%` | `27%` |
| `gemm_mxfp8_kernel` | `(28, 1, 1)` | `43` | `53.12` | `52.26` | `2,204.3` | `4.5%` | `83%` | `5%` | `8%` | `0%` | `4%` |
| `attention_front_kernel` | `(4, 1, 1)` | `43` | `49.73` | `49.44` | `2,018.8` | `4.1%` | `86%` | `1%` | `1%` | `1%` | `12%` |
| `mhc_finalize_kernel` | `(6, 1, 1)` | `80` | `22.06` | `22.72` | `1,906.1` | `3.9%` | `66%` | `5%` | `10%` | `10%` | `9%` |
| `gemm_fp8w_bf16_partials_kernel` | `(32, 4, 1)` | `43` | `38.66` | `37.34` | `1,617.8` | `3.3%` | `60%` | `17%` | `9%` | `0%` | `14%` |
| `gemm_bf16_kernel` | `(2020, 1, 1)` | `5` | `266.27` | `324.22` | `1,613.6` | `3.3%` | `6%` | `7%` | `0%` | `0%` | `87%` |
| `gemm_mxfp8_kernel` | `(64, 1, 1)` | `51` | `24.10` | `26.56` | `1,541.0` | `3.1%` | `56%` | `10%` | `9%` | `0%` | `25%` |
| `gemm_bf16_partials_kernel` | `(6, 8, 1)` | `40` | `33.78` | `36.40` | `1,463.8` | `3.0%` | `15%` | `5%` | `0%` | `0%` | `79%` |
| `gemm_mxfp8_kernel` | `(36, 1, 1)` | `43` | `30.40` | `33.66` | `1,423.1` | `2.9%` | `74%` | `11%` | `10%` | `0%` | `5%` |
| `moe_swiglu_cast_kernel_kernel` | `(8, 36, 1)` | `40` | `35.82` | `34.99` | `1,373.2` | `2.8%` | `94%` | `1%` | `1%` | `1%` | `4%` |
| `gemm_splitk_reduce_cast_kernel` | `(16, 2, 1)` | `43` | `30.43` | `29.38` | `1,266.0` | `2.6%` | `94%` | `1%` | `0%` | `0%` | `4%` |
| `sparse_mla_kernel_1` | `(16, 6, 1)` | `40` | `28.00` | `23.33` | `967.3` | `2.0%` | `6%` | `1%` | `0%` | `0%` | `94%` |
| `sparse_mla_kernel` | `(5, 1, 6)` | `38` | `25.09` | `24.02` | `949.4` | `1.9%` | `19%` | `43%` | `20%` | `3%` | `14%` |

### Stitched Perfetto Trace (`full-43l-tp4-c1.pftrace`)

Running `scripts/iket_perfetto.py` on `run3b_1pass` rebased against `run2_base` (`--base-iket`) produced `full-43l-tp4-c1.pftrace` ([`perfetto_43l.log`](perfetto_43l.log)):
- **Scope:** Decode step `3 of 6` (`54,566.0 us` span)
- **Size & Packets:** `22.41 MiB` (`23,496,767` bytes), `543,269` Perfetto packets
- **Coverage:** `1,404` paired TileLang launches (`0` unpaired IKET launches, `0` unpaired base TileLang launches, `0` grid mismatches), `141,355` CTAs across `48` SMs, `1,404` phase-traced warps, and `1,415` total kernels (including the `11` non-TileLang kernels per step).

## Files in This Experiment

- [`experiment_summary.json`](experiment_summary.json): Complete timings, pool sizes, trace statistics, token IDs, and logprobs across `run1_plain`, `run2_base`, `run3a_2pass`, and `run3b_1pass`.
- [`table_43l.md`](table_43l.md) & [`table_43l.json`](table_43l.json): Full 43-layer kernel and phase attribution report from `scripts/iket_table.py` on `run3b_1pass`.
- [`table_43l_base.md`](table_43l_base.md) & [`table_43l_base.json`](table_43l_base.json): Uninstrumented `--warp-trace` 43-layer baseline timeline report from `run2_base`.
- [`trace_counts_43l.txt`](trace_counts_43l.txt): Verification from `harness/trace_counts.py` that all `100` instrumented kernel types record `1` warp and stay within their static per-warp event bound (`max=38 <= 42`).
- [`perfetto_43l.log`](perfetto_43l.log): Summary of the stitched 43-layer Perfetto trace (`full-43l-tp4-c1.pftrace`).
- [`harness/iket_step.py`](harness/iket_step.py), [`harness/iket_profile.py`](harness/iket_profile.py), [`harness/trace_counts.py`](harness/trace_counts.py): 43-layer step harness (`--full-layers`, `--lean-capture`, `--no-warmup-request`, DMA weight init) and 1-pass / raw-data-bypass `run-iket` wrapper.
- [`scripts/iket_perfetto.py`](scripts/iket_perfetto.py), [`scripts/iket_table.py`](scripts/iket_table.py), [`scripts/step_bundle.py`](scripts/step_bundle.py), [`scripts/step_diff.py`](scripts/step_diff.py): Trace analysis, Perfetto stitching, and bundle generation scripts.
- [`overlay/`](overlay/): Unified `vllm` overlay (`iket.py`, `envs.py`, `gpu_worker.py`, and `test_deepseek_v41_tilelang_iket.py`).
- [`test_iket_43l_and_runtime.py`](test_iket_43l_and_runtime.py) & [`test_iket_profile.py`](test_iket_profile.py): Unit tests verifying the 43-layer config, 1-pass stub generator, ELF patcher, KV cache deduction, and lean startup hooks.
