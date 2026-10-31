# Streaming L2 Cache Hints (`ld.global.cs`) in Mapped-Host Engram Gather

## Problem

On an Engram layer (layers 1 and 15 in DeepSeek-V4.1), the preceding layer's MoE all-reduce window issues an `ENGRAM` L2 weight prefetch for `engram.wkv` (`11.47 MB` at TP3, `8.60 MB` at TP4) so the projection GEMM immediately following `engram_rows_gather` can read its weights from GB10's `24 MiB` L2 cache.

Between that prefetch and the `engram.wkv` GEMM, `engram_rows_gather_kernel` in `vllm/models/deepseek_v4_1/tilelang/engram_rows.py` gathers and dequantizes the step's staged Engram rows from zero-copy mapped host memory (`cudaHostAllocMapped` buffers `rows` and `scales`) plus the step's row ID tensor `ids`. Each step's staged rows are consumed once and overwritten by the reader on the next step; they are never reused by subsequent kernels in the forward pass.

Previously, `engram_rows_gather_kernel` loaded `ids[t, head]`, `scales[lookup, ...]`, and `rows[lookup, ...]` using default `.ca` (cache at all levels, normal eviction priority) global loads. On prefill or multi-step burst decode over GB10's unified LPDDR5X memory, those one-shot mapped-host reads allocate lines in the GPU L2 cache and evict portions of the prefetched `engram.wkv` weight matrix before the projection GEMM runs.

## Solution

Update `engram_rows_gather_kernel` in `vllm/models/deepseek_v4_1/tilelang/engram_rows.py` to use streaming `L2::evict_first` PTX global loads (`ld.global.cs.*`):

1. **`engram_ld_cs_s64`**: Streams `ids[t, head]` (`int64`) via `ld.global.cs.s64`.
2. **`engram_ld_cs_u8`**: Streams the UE8M0 block scale byte `scales[lookup, tx * vec // 32]` via `ld.global.cs.u8`.
3. **`engram_ld_cs_u32`**: Streams 4 packed `float8_e4m3fn` bytes `rows[lookup, tx * 4 .. tx * 4 + 3]` in a single 32-bit `ld.global.cs.u32` transaction per thread and unpacks the 4 bytes in registers before scaling and casting to `bfloat16`.

Marking these one-shot mapped-host loads as `cs` (`L2::evict_first`) instructs GB10's L2 cache controller to allocate streaming lines with first-evict priority instead of displacing reusable weight and activation cache lines.

## Patch

- [`0001-engram-gather-streaming-cache-hints.patch`](0001-engram-gather-streaming-cache-hints.patch) — formatted against `vllm` commit `1f18be14470339c1227ed0432b896a8ca09811b5`.

## Verification & Benchmark (`dgx4`, GB10 `SM121`)

### Correctness

Executed inside `spark-ds41f:1030` on `dgx4`:

```bash
pytest -o addopts="" --noconftest -q tests/kernels/test_deepseek_v41_engram_rows.py
```

- **`22 passed`** (including `test_gather_matches_b12x_lookup_bits` bit-exact verification across full and column-sharded tables, and `test_gather_kernel_uses_streaming_cache_hints_and_matches_reference` verifying `ld.global.cs.s64`, `ld.global.cs.u8`, and `ld.global.cs.u32` emission in the compiled CUDA kernel source).

### L2 Retention & Downstream `engram.wkv` GEMM Latency on GB10

Measured on `dgx4` (`NVIDIA GB10`, `SM121`) across 50 iterations of:
1. Issuing the `L2Prefetcher` plan for `engram.wkv` (`M = 16, N = 1600, K = 7168`, `11.47 MB` MXFP8 weight + block scales),
2. Running `engram_rows_gather_kernel` (`256` tokens, `TP3_RANK0` column mask) across mapped-host row/scale buffers, and
3. Timing both `engram_rows_gather_kernel` and the immediately following `mxfp8_gemm_decode` (`engram.wkv`):

| Gather Kernel Policy | Gather Median (`µs`) | Gather `p95` (`µs`) | `engram.wkv` GEMM Median (`µs`) | `engram.wkv` GEMM `p95` (`µs`) | `engram.wkv` `p95 - p05` Spread (`µs`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (`ld.global` default `.ca`) | `5.12` | `5.47` | `24.78` | `25.57` | `2.34` |
| **Streaming (`ld.global.cs`, `L2::evict_first`)** | **`5.02` (`-2.0%`)** | `5.51` | **`18.94` (`-23.6%`)** | **`20.26` (`-20.8%`)** | **`1.92` (`-17.9%`)** |
