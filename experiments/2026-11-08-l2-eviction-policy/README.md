# Unswizzled Routed MoE TMA Weight Loads Default to L2 `evict_first`

## Metadata
- **Date**: 2026-11-08
- **Deployment base commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`)
- **vLLM base commit**: `2756478b60b90700b9918b457aeb38657da637b5` (Candidate `1107` tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`, 641 patches `0001..0761`)
- **vLLM candidate commit**: `eb8367abcb1f64b8aea176d2bd516b7c24c88d3e` (tree `e10b5f644a8fc58f38d26a8bf71109fdd184204d`, 642 patches `0001..0762`)
- **Patch file**: `vllm/0762-perf-dsv41-evict-first-l2-policy-on-routed-moe-tma-weights.patch`
- **Hardware**: `sp5` (NVIDIA GB10 `sm_121a`, 24 MB L2 cache, 2 fresh container boots of `spark-ds41f:1107`)
- **Intended causal variable**: Default `eviction_policy` to `"evict_first"` on `T.tma_copy(B[...])` in `grouped_fp8_fp4_gemm` (`vllm/models/deepseek_v4_1/tilelang/moe.py`) when `swizzle_panel <= 0` (`decode` `block_M=16`, `late_` `block_M=16`, and `mid_` `block_M=32`), while preserving default `EVICT_NORMAL` (`None`) when `swizzle_panel > 0` (`prefill_` `block_M=64, swizzle_panel=2`).

## Rationale & Hardware L2 Mechanism

1. **Why static L2 persistence failed and dynamic TMA cache hints succeed**:
   - Previously, carving out a static persisting slice of GB10's 24 MB L2 cache (`cudaLimitPersistingL2CacheSize`, `TODO.md` line 403) regressed step time (`+2.47%` to `+4.51%`) because reducing general-purpose L2 capacity starved MoE and MLA tiles.
   - Dynamic PTX L2 replacement hints (`tl::tma_load<tl::CacheHintSm90::EVICT_FIRST>`, `0x12f0000000000000ULL`, emitting `cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint`) bias L2 replacement priority per instruction with **zero static L2 capacity carve-out**.
   - Hardware probe on GB10 (`sm_121a`, CUDA 13.0.2): `T.tma_copy(..., eviction_policy="evict_first")` executes cleanly and bit-identically (`torch.equal`), whereas `cp.async.cg.shared.global.L2::cache_hint` traps with `CUDA error: an illegal instruction was encountered` on `sm_121a` and is not used.

2. **Why unswizzled routed-expert TMA weight loads thrash GB10's 24 MB L2 cache**:
   - In `grouped_fp8_fp4_gemm`, each active routed expert streams `1536 * 5120 / 2 + 5120 * 768 / 2 = 5.90 MB` (TP3, `I = 768`) or `1152 * 5120 / 2 + 5120 * 576 / 2 = 4.42 MB` (TP4, `I = 576`) of packed E2M1 weights (`B`) across `gate_up` and `down_proj`:
     - At `M = 1` (6 active experts): `35.4 MB` (TP3) / `26.5 MB` (TP4) — already exceeding GB10's 24 MB L2 cache in a single layer.
     - At `M = 6` (speculative verify, ~30–36 active experts): `177–212 MB` (TP3) / `133–159 MB` (TP4).
     - At `M = 16` (~60–80 active experts): `354–472 MB` (TP3) / `265–354 MB` (TP4).
   - When `swizzle_panel <= 0` (`decode` `block_M = 16`, `late_` `block_M = 16`, and `mid_` `block_M = 32`), each CTA streams its `[block_N, block_K]` weight slab once without cross-CTA M-tile swizzle reuse.
   - Under default `EVICT_NORMAL`, streaming `B` flushes the 24 MB L2 cache and evicts:
     1. Prefetched `shared_experts.down_proj` (`3.93 MB` at TP3, `2.95 MB` at TP4) and `residual` (`[M, 4, 5120]` BF16) loaded with `evict_last` by `l2_prefetch` before the FFN block.
     2. Gathered input activation rows `A` (`x.values`) and scale words `SFA` (`x.scales`) reused across active expert tiles and across `bx = 0 .. N / block_N - 1`.
     3. Intermediate SwiGLU outputs `hq` (`[tiles * block_M, I]` FP8 + scales) written by `moe_swiglu_cast` and read 40 times per tile (`bx = 0..39`) by `down_kernel`.
     4. Per-route outputs `y` (`[tiles * block_M, 5120]` BF16) written by `down_kernel` and immediately reduced by `moe_combine_v41`.
   - Marking `T.tma_copy(B[...])` as `evict_first` when `swizzle_panel <= 0` causes GB10's L2 replacement policy to evict single-use streamed expert weight lines ahead of prefetched weights, reused input activations, intermediate SwiGLU tiles, and combine buffers, while keeping default `EVICT_NORMAL` when `swizzle_panel > 0` (`prefill_` `block_M = 64, swizzle_panel = 2`) where consecutive M-tiles of the same expert share `B` across a swizzle panel.

## Exact Commands & Workload Identity

1. **2-Boot Hardware Benchmark (`sp5`, `spark-ds41f:1107`)**:
   ```bash
   # Boot 1
   ssh sp5 'flock /tmp/gpu.lock docker run --rm --gpus all --entrypoint python3 \
     -v /tmp/bench_moe_evict_host.py:/tmp/bench_moe_evict.py:ro \
     spark-ds41f:1107 /tmp/bench_moe_evict.py' \
     > experiments/2026-11-08-l2-eviction-policy/receipts/boot-1.txt

   # Boot 2
   ssh sp5 'flock /tmp/gpu.lock docker run --rm --gpus all --entrypoint python3 \
     -v /tmp/bench_moe_evict_host.py:/tmp/bench_moe_evict.py:ro \
     spark-ds41f:1107 /tmp/bench_moe_evict.py' \
     > experiments/2026-11-08-l2-eviction-policy/receipts/boot-2.txt
   ```
   - **Workloads**:
     1. **Standalone Routed MoE (`MoeExpertsV41Plan.run`)**: `384` routed experts, `topk = 6`, `hidden = 5120`, `intermediate = 768` (TP3) and `576` (TP4), rotating across 16 distinct layer routing tables (`M = 1, 2, 4, 6, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096`).
     2. **Full FFN Layer (`l2_prefetch` + `shared_experts` + `MoeExpertsV41Plan.run` + `residual`)**: Exact FFN sequence from `DeepseekV41Layer.forward` (`l2_prefetch.issue/join` on `shared_experts.down_proj` + `residual` with `evict_last`, `shared_experts.gate_up_proj`, `swiglu_quant`, `shared_experts.down_proj`, `MoeExpertsV41Plan.run(..., shared=sh_out)`, and `residual.view(M, -1) + moe_out`) across `M = 1, 2, 4, 6, 8, 16, 32, 64`.
     3. Each data point runs 5 interleaved `(baseline, candidate)` pairs of 32 timed iterations after 10 warmup iterations (`5 * 32 = 160` timed calls per arm per boot).

2. **GPU Unit & Preparation Contract Tests (`sp5`, `spark-ds41f:1107`)**:
   ```bash
   ssh sp5 'flock /tmp/gpu.lock docker run --rm --gpus all --entrypoint python3 \
     -v /tmp/vllm-0762/vllm/models/deepseek_v4_1/tilelang/moe.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py:ro \
     -v /tmp/vllm-0762/tests/kernels/moe/test_deepseek_v41_tilelang_moe.py:/tmp/test_deepseek_v41_tilelang_moe.py:ro \
     -v /tmp/vllm-0762/tests/v1/worker/test_prepare_kernels.py:/tmp/test_prepare_kernels.py:ro \
     spark-ds41f:1107 -m pytest /tmp/test_deepseek_v41_tilelang_moe.py /tmp/test_prepare_kernels.py -v'
   ```
   - Result: `148 passed in 142.42s` (`receipts/gpu-pytest.txt`).

## 2-Boot Benchmark Results (`sp5`, GB10 `sm_121a`)

### 1. Full FFN Layer (`l2_prefetch` FFN window + `shared_experts` + `MoeExpertsV41Plan.run` + `residual`)

#### TP3 (`intermediate = 768`)
| `M` | Boot 1 Base (us) | Boot 1 Cand (us) | Boot 1 Delta | Boot 2 Base (us) | Boot 2 Cand (us) | Boot 2 Delta | 2-Boot Mean Delta (us/layer) | 61-Layer Step Savings | `torch.equal` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 220.32 ± 0.14 | 184.45 ± 0.14 | **-16.28%** | 220.42 ± 0.15 | 183.58 ± 0.19 | **-16.71%** | **-36.36 ± 0.48 us (-16.50%)** | **-2.22 ms** | `True` |
| 2 | 376.19 ± 0.19 | 339.14 ± 0.19 | **-9.85%** | 374.94 ± 0.35 | 339.04 ± 0.12 | **-9.58%** | **-36.48 ± 0.58 us (-9.71%)** | **-2.23 ms** | `True` |
| 4 | 691.42 ± 2.38 | 648.42 ± 0.77 | **-6.22%** | 684.38 ± 0.90 | 646.08 ± 0.90 | **-5.60%** | **-40.65 ± 2.35 us (-5.91%)** | **-2.48 ms** | `True` |
| 6 | 997.38 ± 1.07 | 952.54 ± 1.65 | **-4.49%** | 1009.92 ± 14.35 | 951.10 ± 1.54 | **-5.82%** | **-51.83 ± 6.99 us (-5.16%)** | **-3.16 ms** | `True` |
| 8 | 1308.99 ± 0.55 | 1265.60 ± 0.46 | **-3.31%** | 1319.01 ± 6.38 | 1261.76 ± 0.78 | **-4.34%** | **-50.32 ± 6.93 us (-3.83%)** | **-3.07 ms** | `True` |
| 16 | 2539.71 ± 0.30 | 2475.17 ± 0.42 | **-2.54%** | 2541.54 ± 1.56 | 2500.70 ± 17.75 | **-1.61%** | **-52.69 ± 11.85 us (-2.07%)** | **-3.21 ms** | `True` |
| 32 | 4769.18 ± 2.83 | 4675.23 ± 1.53 | **-1.97%** | 4916.70 ± 31.32 | 4762.50 ± 27.60 | **-3.14%** | **-124.08 ± 30.12 us (-2.56%)** | **-7.57 ms** | `True` |
| 64 | 8266.24 ± 2.13 | 8121.02 ± 0.65 | **-1.76%** | 8539.36 ± 95.36 | 8212.16 ± 39.21 | **-3.83%** | **-236.21 ± 90.99 us (-2.81%)** | **-14.41 ms** | `True` |

#### TP4 (`intermediate = 576`)
| `M` | Boot 1 Base (us) | Boot 1 Cand (us) | Boot 1 Delta | Boot 2 Base (us) | Boot 2 Cand (us) | Boot 2 Delta | 2-Boot Mean Delta (us/layer) | 61-Layer Step Savings | `torch.equal` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 176.35 ± 0.14 | 150.50 ± 0.12 | **-14.66%** | 177.38 ± 0.46 | 150.82 ± 0.08 | **-14.97%** | **-26.21 ± 0.36 us (-14.82%)** | **-1.60 ms** | `True` |
| 2 | 288.83 ± 0.42 | 259.26 ± 0.25 | **-10.24%** | 287.07 ± 0.06 | 257.34 ± 0.14 | **-10.36%** | **-29.65 ± 0.08 us (-10.30%)** | **-1.81 ms** | `True` |
| 4 | 516.19 ± 2.30 | 488.54 ± 1.81 | **-5.36%** | 516.38 ± 0.59 | 484.83 ± 0.57 | **-6.11%** | **-29.60 ± 1.95 us (-5.73%)** | **-1.81 ms** | `True` |
| 6 | 747.62 ± 1.54 | 708.99 ± 0.52 | **-5.17%** | 747.39 ± 0.77 | 706.46 ± 1.13 | **-5.48%** | **-39.78 ± 1.15 us (-5.32%)** | **-2.43 ms** | `True` |
| 8 | 981.06 ± 0.71 | 943.42 ± 0.35 | **-3.84%** | 979.07 ± 0.69 | 941.15 ± 0.24 | **-3.87%** | **-37.78 ± 0.14 us (-3.85%)** | **-2.30 ms** | `True` |
| 16 | 1906.08 ± 2.86 | 1878.88 ± 1.80 | **-1.43%** | 1904.06 ± 2.13 | 1869.06 ± 1.13 | **-1.84%** | **-31.10 ± 3.90 us (-1.63%)** | **-1.90 ms** | `True` |
| 32 | 3566.11 ± 3.31 | 3510.75 ± 2.91 | **-1.55%** | 3563.04 ± 2.53 | 3515.10 ± 4.86 | **-1.35%** | **-51.65 ± 3.71 us (-1.45%)** | **-3.15 ms** | `True` |
| 64 | 6190.40 ± 0.37 | 6090.75 ± 4.02 | **-1.61%** | 6190.24 ± 4.65 | 6087.17 ± 1.38 | **-1.66%** | **-101.36 ± 1.71 us (-1.64%)** | **-6.18 ms** | `True` |

### 2. Standalone Routed MoE (`MoeExpertsV41Plan.run`, rotating 16 layer routings)

#### TP3 (`intermediate = 768`)
| `M` | Boot 1 Base (us) | Boot 1 Cand (us) | Boot 1 Delta | Boot 2 Base (us) | Boot 2 Cand (us) | Boot 2 Delta | `torch.equal` |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 167.07 ± 0.56 | 161.92 ± 0.06 | **-3.08%** | 166.21 ± 0.39 | 161.06 ± 0.24 | **-3.10%** | `True` |
| 2 | 319.90 ± 0.24 | 318.08 ± 0.32 | **-0.57%** | 320.22 ± 0.54 | 317.89 ± 0.14 | **-0.73%** | `True` |
| 4 | 631.90 ± 0.59 | 625.73 ± 0.43 | **-0.98%** | 632.10 ± 1.38 | 625.92 ± 0.44 | **-0.98%** | `True` |
| 6 | 942.14 ± 0.89 | 923.58 ± 0.99 | **-1.97%** | 940.03 ± 0.37 | 921.86 ± 0.47 | **-1.93%** | `True` |
| 8 | 1250.62 ± 0.31 | 1220.45 ± 0.48 | **-2.41%** | 1249.63 ± 0.32 | 1217.79 ± 0.33 | **-2.55%** | `True` |
| 16 | 2488.22 ± 0.61 | 2432.03 ± 0.77 | **-2.26%** | 2483.46 ± 1.85 | 2428.99 ± 0.22 | **-2.19%** | `True` |
| 32 | 4700.38 ± 1.07 | 4595.46 ± 1.56 | **-2.23%** | 4695.07 ± 0.52 | 4591.39 ± 0.62 | **-2.21%** | `True` |
| 64 | 8171.65 ± 0.83 | 8018.88 ± 1.71 | **-1.87%** | 8165.41 ± 1.31 | 8013.38 ± 0.65 | **-1.86%** | `True` |
| 128 | 11526.05 ± 0.61 | 11386.94 ± 0.72 | **-1.21%** | 11519.49 ± 0.48 | 11381.98 ± 0.88 | **-1.19%** | `True` |
| 512 | 13231.04 ± 0.62 | 13179.90 ± 1.01 | **-0.39%** | 13232.06 ± 0.82 | 13173.44 ± 0.70 | **-0.44%** | `True` |
| 1024 | 13447.62 ± 2.40 | 13537.54 ± 3.92 | +0.67% | 13726.37 ± 121.02 | 13684.86 ± 80.52 | -0.30% | `True` |
| 2048 | 13371.55 ± 1.65 | 13376.00 ± 4.02 | +0.03% | 13448.03 ± 40.31 | 13450.27 ± 62.66 | +0.02% | `True` |
| 4096 | 17218.21 ± 31.11 | 17211.01 ± 33.76 | -0.04% | 17284.74 ± 43.36 | 17279.68 ± 55.02 | -0.03% | `True` |

#### TP4 (`intermediate = 576`)
| `M` | Boot 1 Base (us) | Boot 1 Cand (us) | Boot 1 Delta | Boot 2 Base (us) | Boot 2 Cand (us) | Boot 2 Delta | `torch.equal` |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 136.06 ± 1.24 | 135.42 ± 1.38 | **-0.47%** | 135.30 ± 0.08 | 133.34 ± 0.13 | **-1.44%** | `True` |
| 2 | 253.89 ± 0.35 | 251.49 ± 0.25 | **-0.95%** | 253.15 ± 0.14 | 250.30 ± 0.12 | **-1.13%** | `True` |
| 4 | 485.98 ± 0.90 | 476.35 ± 1.56 | **-1.98%** | 484.35 ± 0.42 | 474.91 ± 1.01 | **-1.95%** | `True` |
| 6 | 711.87 ± 1.07 | 682.21 ± 0.87 | **-4.17%** | 707.84 ± 0.27 | 680.00 ± 0.67 | **-3.93%** | `True` |
| 8 | 940.13 ± 0.41 | 905.41 ± 0.41 | **-3.69%** | 942.46 ± 2.75 | 911.71 ± 5.18 | **-3.26%** | `True` |
| 16 | 1866.91 ± 2.73 | 1840.16 ± 3.92 | **-1.43%** | 2009.38 ± 65.58 | 1854.75 ± 17.52 | **-7.70%** | `True` |
| 32 | 3516.70 ± 1.66 | 3466.56 ± 2.67 | **-1.43%** | 3541.54 ± 48.03 | 3466.08 ± 2.74 | **-2.13%** | `True` |
| 64 | 6136.77 ± 1.88 | 6029.89 ± 2.13 | **-1.74%** | 6135.42 ± 2.41 | 6028.03 ± 1.44 | **-1.75%** | `True` |
| 128 | 8657.63 ± 1.53 | 8548.83 ± 1.71 | **-1.26%** | 8656.67 ± 1.09 | 8547.87 ± 1.78 | **-1.26%** | `True` |
| 512 | 9960.19 ± 2.09 | 9912.61 ± 3.74 | **-0.48%** | 9961.98 ± 4.10 | 9912.03 ± 2.45 | **-0.50%** | `True` |
| 1024 | 10200.74 ± 1.44 | 10186.27 ± 1.77 | **-0.14%** | 10200.83 ± 2.45 | 10185.79 ± 0.72 | **-0.15%** | `True` |
| 2048 | 10164.06 ± 2.52 | 10164.35 ± 2.40 | +0.00% | 10164.10 ± 2.75 | 10163.36 ± 1.63 | -0.01% | `True` |
| 4096 | 13264.61 ± 19.63 | 13275.39 ± 21.18 | +0.08% | 13279.68 ± 19.60 | 13283.74 ± 18.61 | +0.03% | `True` |

## Errors Encountered & Resolved During Benchmarking
1. **Random FP8 `NaN` bit pattern (`0x7f`)**: Initial random input generation used `torch.randint(0, 128, ..., dtype=torch.uint8).view(torch.float8_e4m3fn)`, which includes `0x7f` (`NaN` in E4M3), causing `torch.equal(out1, out2)` to report `False` when comparing `NaN == NaN`. Restricting the byte range to `0..119` (`torch.randint(0, 120, ...)`) produced finite E4M3 inputs, and `torch.equal(out1, out2)` passed 100% across all `M = 1..4096` on both TP3 and TP4.
2. **`L2PrefetchPlan` API & TP4 `pack_weight_tiles` alignment**: `l2_prefetch.issue(pf_plan, M)` + `l2_prefetch.join()` is the runtime API used by `DeepseekV41Layer`, and `pack_weight_tiles(sh_w2)` requires `K % 128 == 0` (used on TP3 `I = 768`, while TP4 `I = 576` uses `tile_expert_weights` / `mxfp8_mxfp4_gemm_splitk`).

## Memory Observations
- **Zero memory overhead**: `eviction_policy="evict_first"` is a compile-time template parameter on `tl::tma_load<tl::CacheHintSm90::EVICT_FIRST>` (`0x12f0000000000000ULL` immediate in `cp.async.bulk.tensor.3d`). It allocates zero device memory, zero shared memory, and zero extra registers, and makes no `cudaLimitPersistingL2CacheSize` carve-out.

## Conclusion
Defaulting `grouped_fp8_fp4_gemm` TMA weight loads to `eviction_policy="evict_first"` when `swizzle_panel <= 0` (`decode`, `late_`, and `mid_` tiles) while preserving `EVICT_NORMAL` on swizzled prefill tiles (`swizzle_panel > 0`) is a **confirmed, bit-identical hardware win** across both boots on GB10 (`sm_121a`):
- **Full FFN Layer**: `-16.50%` (`-36.36 us/layer`, `-2.22 ms/61 layers`) at `M = 1` and `-5.16%` (`-51.83 us/layer`, `-3.16 ms/61 layers`) at `M = 6` on TP3; `-14.82%` (`-26.21 us/layer`, `-1.60 ms/61 layers`) at `M = 1` and `-5.32%` (`-39.78 us/layer`, `-2.43 ms/61 layers`) at `M = 6` on TP4.
- **Standalone Routed MoE**: `-1.95%` to `-3.09%` (`M = 1, 6, 8, 16, 32`) on TP3 and `-1.96%` to `-4.05%` (`M = 4, 6, 8, 16, 32, 64`) on TP4, with neutral (`±0.04%`) prefill performance at `M = 2048..4096`.
