# Routed MoE `moe_swiglu_cast` Empty-Slab Early Exit & Post-Gate Dual-Stream `shared_experts` Overlap

## Overview

- **Base commit**: `8e5ea208f3d22bc69df91ace94d6975932c858a8` (Candidate `1107`, image `spark-ds41f:1107`)
- **Upstream vLLM base**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`
- **vLLM patch series**: `experiments/2026-11-08-moe-swiglu-and-dual-stream-graph/vllm/series` (`642` active patches)
  - Patch added: `0762-perf-dsv41-skip-empty-moe-swiglu-slabs-and-restore-p.patch`
  - Patch head: `2756478b60b90700b9918b457aeb38657da637b5` -> `e2c90df7ca7611dda40bebd16ca951bd85b1f322`
  - Expected tree: `ed1bdb7a7684da33d403a055521ac0c4dd7ca670` -> `1648d2a2a84790f9c1a89e08d48d80ea5a92bd67`
- **Hardware**: `sp6` (`spark-0014`, NVIDIA GB10 `sm_121a`)
- **Causal variables (`TODO.md` lines 181-186 & 206-209)**:
  1. **Routed MoE `moe_swiglu_cast` empty-slab and 1-row-tile early exit + `rows_per_cta = 4` at `intermediate = 768` (`vllm/models/deepseek_v4_1/tilelang/moe.py`)**:
     - In decode (`tile_m = 16`, `rows_per_cta = 4`, `slabs = 4`), `moe_route_pack` packs valid routed rows contiguously from row `0` of each active expert tile (`r = 0, 1, ...`) and fills unused entries with `-1`. At `M = 1..6` (`6..36` total routed token-expert assignments across up to `36` active experts), each active tile has `1.0-1.5` valid rows on average, so **75% of active-tile CTAs (`slab in {1, 2, 3}`) have `orders[0] == -1`** (zero valid rows), and **68%-100% of `slab == 0` CTAs have `orders[1] == -1`** (only row `0` valid).
     - Previously, `moe_swiglu_cast` launched `rows_per_cta = 4 if warps >= 4 else 2` (using only `2` rows/CTA on TP3 where `intermediate = 768` and `warps = 3`, doubling active-tile slabs from `4` to `8`), unconditionally preloaded all `rows_per_cta` `Order` and `Weights` entries before `pdl.chain()`, and after `pdl.chain()` executed an unrolled `for r in T.unroll(rows_per_cta):` loop with per-element `orders[r] >= 0` predicates across every row even when `orders[0] < 0` or `orders[1] < 0`.
     - We set `rows_per_cta = 2 if intermediate > 1152 else 4` (so TP3 `intermediate = 768` also uses `4` rows/CTA), cascade `Order`/`Weights` preloads before `pdl.chain()` (`orders[0]` first; only if `orders[0] >= 0` load `weights[0]` and `orders[1]`; only if `orders[1] >= 0` load `weights[1]` and `r in 2..rows_per_cta-1`), exit empty slabs immediately after `pdl.chain()` via `if expert >= 0 and orders[0] >= 0:`, and guard rows `1..rows_per_cta-1` behind `if rows_per_cta > 1 and orders[1] >= 0:`.
     - We also investigated fusing SwiGLU + MXFP8 directly into the `FC1` (`moe_mxfp4_mxfp8_ArgK_gemm`) epilogue: because `FC1` tiles the `2 * intermediate` column dimension by `block_N = 64` (`18` independent N-tiles per expert on TP4, `24` on TP3), gate (`0..intermediate-1`) and up (`intermediate..2*intermediate-1`) columns are computed by different CTAs, and `MXFP8` scale reduction requires the full row maximum across `intermediate` outputs. Keeping `moe_swiglu_cast` as a PDL-chained kernel while eliminating empty-slab and padded-row work preserves `FC1`'s `block_N = 64` occupancy and bit-exactness.
  2. **Decode dual-stream `shared_experts` overlap with pre-recorded `_input_ready_event` (`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`, `shared_experts.py`)**:
     - In `MoERunner._forward_impl`, `if workspace_size == 0 and self.shared_experts is not None:` previously ran `shared_experts.maybe_forward_async(hidden_states)` *before* `_apply_quant_method` even when `fold_shared == True` (where `workspace_size == 0` is always true because shared experts accumulate into the routed combine buffer). That bypassed `router._post_gate_hook` in `_apply_quant_method` and launched `shared_experts` on the side stream before the router gate GEMM, causing SM contention with the router gate GEMM and cold-L2 FFN weight prefetch (`_l2pf_ffn`).
     - Furthermore, when `router._post_gate_hook` fired after `partials_kernel`, `maybe_forward_async` recorded `_input_ready_event` on `current_stream()` *after* `partials_kernel`, forcing `_shared_experts_stream` to wait for `partials_kernel` before starting `shared_experts` even though `shared_experts_input` (`hidden_states`) was already ready before `router.select_experts`.
     - We guard the pre-`_apply_quant_method` launch with `and not getattr(self._quant_method.modular_experts, 'fold_shared', False)`, add `SharedExpertsRunner.record_input_ready(shared_experts_input)` to record `_input_ready_event` before `router.select_experts`, and pass `input_ready_recorded=True` inside `_post_gate_hook` so `_shared_experts_stream` waits only on `hidden_states` readiness while launching on the host after the router gate GEMM.

## Verification & Unit Tests

- **Bit-exactness**: `100%` bit-exact (`torch.equal`, `0` uint16 diffs, `max_abs_diff == 0.0`) across all standalone `moe_swiglu_cast` shapes (`intermediate in {576, 768, 1152, 1536}`, `M in {1, 2, 3, 6, 12, 24, 48}`) and all full FFN sublayer modes (`serial_1stream`, `base_1107`, `candidate_hook_only`, `candidate_swiglu_only`, `candidate_full`) across both `TP=4` and `TP=3` (both ungraphed eager and captured `torch.cuda.CUDAGraph` replay) on both boots.
- **Pytest suite (2 fresh container boots on `sp6`)**:
  - Boot 1 (`receipts/pytest_boot1.txt`): **160 passed in 29.01s**
  - Boot 2 (`receipts/pytest_boot2.txt`): **160 passed in 28.45s**
  - Files tested: `tests/models/test_deepseek_v4_1_l2_schedule.py`, `tests/kernels/moe/test_deepseek_v41_tilelang_moe.py`, `tests/kernels/test_deepseek_v41_tilelang_pdl.py`.

## Exact Commands

```bash
# Boot 1 & Boot 2 pytest
docker run --rm --gpus all --ipc=host \
  -v /home/chris/projects/spark-ds41f/.work/upstreams/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py:ro \
  -v /home/chris/projects/spark-ds41f/.work/upstreams/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:/opt/spark-ds41f/candidate/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:ro \
  -v /home/chris/projects/spark-ds41f/.work/upstreams/vllm/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:/opt/spark-ds41f/candidate/vllm/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:ro \
  -v /home/chris/projects/spark-ds41f/.work/upstreams/vllm/tests:/opt/vllm-tests:ro \
  spark-ds41f:1107 pytest --noconftest -q \
    /opt/vllm-tests/models/test_deepseek_v4_1_l2_schedule.py \
    /opt/vllm-tests/kernels/moe/test_deepseek_v41_tilelang_moe.py \
    /opt/vllm-tests/kernels/test_deepseek_v41_tilelang_pdl.py

# Boot 1 & Boot 2 benchmarks
docker run --rm --gpus all --ipc=host \
  -v /home/chris/projects/spark-ds41f/experiments/2026-11-08-moe-swiglu-and-dual-stream-graph:/workspace \
  -v /home/chris/projects/spark-ds41f/.work/upstreams/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py:ro \
  -v /home/chris/projects/spark-ds41f/.work/upstreams/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:/opt/spark-ds41f/candidate/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:ro \
  -v /home/chris/projects/spark-ds41f/.work/upstreams/vllm/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:/opt/spark-ds41f/candidate/vllm/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:ro \
  spark-ds41f:1107 python3 /workspace/bench_moe_swiglu_and_dual_stream.py --boot <1|2> --out /workspace/receipts/boot<1|2>.json
```

## 1. Standalone `moe_swiglu_cast` 2-Boot Benchmark (`torch.cuda.CUDAGraph` Replay & Eager)

Workload: `E = 72`, `top_k = 6`, `tile_m = 16`, captured inside `torch.cuda.CUDAGraph` (`35` interleaved windows $\times$ `20` graph replays per boot).

### `intermediate = 576` (TP4 Routed Expert Shard)

| `M` | `limit` | Active Tiles | Eager Base B1/B2 (us) | Eager Cand B1/B2 (us) | Graph Base B1/B2 (us) | Graph Base Mean (us) | Graph Cand B1/B2 (us) | Graph Cand Mean (us) | $\Delta$ Graph (us/layer) | $\Delta$ (58L ms/step) | Speedup (%) | Bit-Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `1` | `96` | `6` | `9.92/11.65` | `9.98/11.87` | `1.054/1.074` | `1.064` | `0.979/1.040` | **`1.010`** | **`-0.054`** | **`-0.003`** | **`+5.12%`** | `PASS` |
| `2` | `192` | `12` | `10.88/11.87` | `10.24/11.58` | `1.072/1.070` | `1.071` | `1.022/1.030` | **`1.026`** | **`-0.045`** | **`-0.003`** | **`+4.20%`** | `PASS` |
| `3` | `288` | `18` | `9.73/10.21` | `9.66/9.86` | `1.182/1.189` | `1.185` | `1.061/1.035` | **`1.048`** | **`-0.137`** | **`-0.008`** | **`+11.60%`** | `PASS` |
| `6` | `576` | `36` | `9.57/11.46` | `9.57/11.46` | `1.274/1.330` | `1.302` | `1.152/1.146` | **`1.149`** | **`-0.153`** | **`-0.009`** | **`+11.75%`** | `PASS` |
| `12` | `1152` | `65` | `9.98/11.26` | `9.70/11.30` | `2.110/2.144` | `2.127` | `1.883/1.877` | **`1.880`** | **`-0.247`** | **`-0.014`** | **`+11.61%`** | `PASS` |
| `24` | `2304` | `123` | `11.14/11.10` | `10.91/9.92` | `3.200/3.242` | `3.221` | `2.469/2.515` | **`2.492`** | **`-0.729`** | **`-0.042`** | **`+22.63%`** | `PASS` |
| `48` | `4608` | `201` | `12.67/12.90` | `11.33/11.30` | `5.163/5.192` | `5.178` | `3.712/3.715` | **`3.713`** | **`-1.464`** | **`-0.085`** | **`+28.28%`** | `PASS` |

### `intermediate = 768` (TP3 Routed Expert Shard, `rows_per_cta: 2 -> 4`)

| `M` | `limit` | Active Tiles | Eager Base B1/B2 (us) | Eager Cand B1/B2 (us) | Graph Base B1/B2 (us) | Graph Base Mean (us) | Graph Cand B1/B2 (us) | Graph Cand Mean (us) | $\Delta$ Graph (us/layer) | $\Delta$ (58L ms/step) | Speedup (%) | Bit-Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `1` | `96` | `6` | `11.78/12.00` | `11.90/12.10` | `1.301/1.330` | `1.316` | `1.293/1.306` | **`1.300`** | **`-0.016`** | **`-0.001`** | **`+1.22%`** | `PASS` |
| `2` | `192` | `12` | `9.98/12.19` | `9.89/12.26` | `1.310/1.362` | `1.336` | `1.336/1.358` | **`1.347`** | **`+0.011`** | **`+0.001`** | **`-0.82%`** | `PASS` |
| `3` | `288` | `18` | `9.92/12.06` | `9.79/12.13` | `1.342/1.376` | `1.359` | `1.342/1.368` | **`1.355`** | **`-0.004`** | **`-0.000`** | **`+0.29%`** | `PASS` |
| `6` | `576` | `36` | `10.18/11.01` | `9.92/10.56` | `1.491/1.510` | `1.501` | `1.389/1.438` | **`1.413`** | **`-0.087`** | **`-0.005`** | **`+5.80%`** | `PASS` |
| `12` | `1152` | `65` | `11.30/11.33` | `11.17/10.88` | `2.253/2.272` | `2.263` | `1.848/1.859` | **`1.853`** | **`-0.409`** | **`-0.024`** | **`+18.08%`** | `PASS` |
| `24` | `2304` | `123` | `11.36/10.62` | `11.42/10.56` | `2.678/2.701` | `2.689` | `2.307/2.333` | **`2.320`** | **`-0.369`** | **`-0.021`** | **`+13.74%`** | `PASS` |
| `48` | `4608` | `201` | `11.94/11.87` | `11.68/11.71` | `3.811/3.842` | `3.827` | `3.312/3.344` | **`3.328`** | **`-0.499`** | **`-0.029`** | **`+13.03%`** | `PASS` |

### `intermediate = 1152` (TP4 Unsharded / TP2 Shard)

| `M` | `limit` | Active Tiles | Eager Base B1/B2 (us) | Eager Cand B1/B2 (us) | Graph Base B1/B2 (us) | Graph Base Mean (us) | Graph Cand B1/B2 (us) | Graph Cand Mean (us) | $\Delta$ Graph (us/layer) | $\Delta$ (58L ms/step) | Speedup (%) | Bit-Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `1` | `96` | `6` | `48.16/48.80` | `47.14/48.51` | `1.648/1.690` | `1.669` | `1.451/1.437` | **`1.444`** | **`-0.225`** | **`-0.013`** | **`+13.48%`** | `PASS` |
| `2` | `192` | `12` | `47.55/48.10` | `46.27/47.26` | `1.734/1.717` | `1.726` | `1.461/1.440` | **`1.450`** | **`-0.275`** | **`-0.016`** | **`+15.94%`** | `PASS` |
| `3` | `288` | `18` | `47.39/47.71` | `45.98/46.69` | `1.792/1.778` | `1.785` | `1.488/1.485` | **`1.486`** | **`-0.299`** | **`-0.017`** | **`+16.72%`** | `PASS` |
| `6` | `576` | `36` | `48.26/47.07` | `47.10/46.30` | `2.000/2.016` | `2.008` | `1.624/1.643` | **`1.634`** | **`-0.374`** | **`-0.022`** | **`+18.65%`** | `PASS` |
| `12` | `1152` | `65` | `10.72/11.10` | `10.43/11.10` | `2.410/2.442` | `2.426` | `2.094/2.096` | **`2.095`** | **`-0.331`** | **`-0.019`** | **`+13.64%`** | `PASS` |
| `24` | `2304` | `123` | `11.65/10.11` | `11.42/10.14` | `3.728/3.749` | `3.739` | `3.114/3.123` | **`3.119`** | **`-0.620`** | **`-0.036`** | **`+16.58%`** | `PASS` |
| `48` | `4608` | `201` | `13.86/14.02` | `12.70/13.89` | `5.949/5.954` | `5.951` | `4.853/4.936` | **`4.894`** | **`-1.057`** | **`-0.061`** | **`+17.76%`** | `PASS` |

### `intermediate = 1536` (TP3 Unsharded Shard)

| `M` | `limit` | Active Tiles | Eager Base B1/B2 (us) | Eager Cand B1/B2 (us) | Graph Base B1/B2 (us) | Graph Base Mean (us) | Graph Cand B1/B2 (us) | Graph Cand Mean (us) | $\Delta$ Graph (us/layer) | $\Delta$ (58L ms/step) | Speedup (%) | Bit-Exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `1` | `96` | `6` | `51.07/11.90` | `49.06/11.90` | `2.170/1.805` | `1.987` | `1.830/1.646` | **`1.738`** | **`-0.249`** | **`-0.014`** | **`+12.55%`** | `PASS` |
| `2` | `192` | `12` | `50.62/11.74` | `49.73/11.87` | `2.261/1.907` | `2.084` | `1.944/1.766` | **`1.855`** | **`-0.229`** | **`-0.013`** | **`+10.99%`** | `PASS` |
| `3` | `288` | `18` | `51.42/11.78` | `49.95/11.74` | `2.286/1.962` | `2.124` | `2.021/1.821` | **`1.921`** | **`-0.203`** | **`-0.012`** | **`+9.56%`** | `PASS` |
| `6` | `576` | `36` | `11.78/11.36` | `11.62/11.17` | `2.213/2.208` | `2.211` | `2.010/2.026` | **`2.018`** | **`-0.193`** | **`-0.011`** | **`+8.71%`** | `PASS` |
| `12` | `1152` | `65` | `9.79/11.39` | `9.76/11.36` | `3.010/2.947` | `2.978` | `2.712/2.693` | **`2.703`** | **`-0.276`** | **`-0.016`** | **`+9.27%`** | `PASS` |
| `24` | `2304` | `123` | `11.97/12.22` | `11.26/11.62` | `4.261/4.283` | `4.272` | `3.640/3.682` | **`3.661`** | **`-0.611`** | **`-0.035`** | **`+14.30%`** | `PASS` |
| `48` | `4608` | `201` | `14.34/13.86` | `13.63/13.70` | `6.085/6.078` | `6.082` | `5.131/5.080` | **`5.106`** | **`-0.976`** | **`-0.057`** | **`+16.05%`** | `PASS` |

## 2. Full FFN Sublayer 2-Boot Benchmark (`mhc_pre + router + shared_experts + routed_moe`)

Workload: `H = 5120`, `E = 72`, `top_k = 6`, `fold_shared = True`, comparing:
- `serial_1stream`: 1-stream serial execution (`shared_experts` + `router` + `routed_moe` on the primary stream)
- `base_1107`: Candidate `1107` baseline (where `workspace_size == 0` launches `shared_experts` on the side stream before `_apply_quant_method`)
- `candidate_swiglu_only`: Pre-gate `shared_experts` launch + optimized `moe_swiglu_cast`
- `candidate_full`: Restored `router._post_gate_hook` with pre-recorded `_input_ready_event` + optimized `moe_swiglu_cast`

### `TP = 4` (`I_routed = 576`, `I_shared = 576`, `2*I_shared = 1152`) — Warm-L2 Captured CUDA Graph Replay (us/layer & 58-Layer ms/step)

| `M` | `serial_1stream` B1/B2 (us) | `base_1107` B1/B2 (us) | `cand_swiglu_only` B1/B2 (us) | `cand_full` B1/B2 (us) | $\Delta$ `cand_full` vs `serial_1stream` (us / 58L ms) | $\Delta$ `cand_swiglu_only` vs `1107` (us / 58L ms) | $\Delta$ `cand_full` vs `1107` (us / 58L ms) | Bit-Exact (`0` u16 diff) |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `1` | `249.49 / 251.37` (`250.43`) | `238.98 / 239.56` (`239.27`) | `238.86 / 239.56` (`239.21`) | `239.09 / 239.86` (`239.48`) | **`-10.95 us` (`-0.635 ms`)** | **`-0.06 us` (`-0.003 ms`)** | **`+0.21 us` (`+0.012 ms`)** | `PASS` |
| `2` | `358.14 / 359.99` (`359.06`) | `347.25 / 349.33` (`348.29`) | `347.47 / 349.13` (`348.30`) | `347.49 / 349.03` (`348.26`) | **`-10.81 us` (`-0.627 ms`)** | **`+0.01 us` (`+0.001 ms`)** | **`-0.03 us` (`-0.002 ms`)** | `PASS` |
| `3` | `435.43 / 433.48` (`434.46`) | `429.91 / 427.35` (`428.63`) | `430.23 / 427.98` (`429.11`) | `430.12 / 428.20` (`429.16`) | **`-5.30 us` (`-0.307 ms`)** | **`+0.48 us` (`+0.028 ms`)** | **`+0.53 us` (`+0.031 ms`)** | `PASS` |
| `6` | `738.56 / 737.61` (`738.09`) | `731.30 / 732.89` (`732.10`) | `731.58 / 733.36` (`732.47`) | `731.87 / 733.12` (`732.50`) | **`-5.59 us` (`-0.324 ms`)** | **`+0.38 us` (`+0.022 ms`)** | **`+0.40 us` (`+0.023 ms`)** | `PASS` |
| `12` | `1313.10 / 1312.78` (`1312.94`) | `1304.31 / 1304.04` (`1304.17`) | `1304.20 / 1303.10` (`1303.65`) | `1304.31 / 1303.19` (`1303.75`) | **`-9.19 us` (`-0.533 ms`)** | **`-0.52 us` (`-0.030 ms`)** | **`-0.42 us` (`-0.025 ms`)** | `PASS` |
| `24` | `2429.05 / 2433.77` (`2431.41`) | `2417.20 / 2423.28` (`2420.24`) | `2417.90 / 2422.82` (`2420.36`) | `2417.56 / 2422.50` (`2420.03`) | **`-11.38 us` (`-0.660 ms`)** | **`+0.12 us` (`+0.007 ms`)** | **`-0.21 us` (`-0.012 ms`)** | `PASS` |
| `48` | `3576.03 / 3585.84` (`3580.94`) | `3559.16 / 3562.59` (`3560.88`) | `3557.49 / 3561.62` (`3559.55`) | `3557.58 / 3561.52` (`3559.55`) | **`-21.39 us` (`-1.240 ms`)** | **`-1.32 us` (`-0.077 ms`)** | **`-1.32 us` (`-0.077 ms`)** | `PASS` |

### `TP = 4` (`I_routed = 576`, `I_shared = 576`, `2*I_shared = 1152`) — Cold-L2 + `_l2pf_ffn` Prefetch Captured CUDA Graph Replay (us/layer & 58-Layer ms/step)

| `M` | `serial_1stream` B1/B2 (us) | `base_1107` B1/B2 (us) | `cand_swiglu_only` B1/B2 (us) | `cand_full` B1/B2 (us) | $\Delta$ `cand_full` vs `serial_1stream` (us / 58L ms) | $\Delta$ `cand_swiglu_only` vs `1107` (us / 58L ms) | $\Delta$ `cand_full` vs `1107` (us / 58L ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `254.98 / 258.05` (`256.51`) | `241.47 / 242.69` (`242.08`) | `243.71 / 243.71` (`243.71`) | `242.69 / 243.52` (`243.11`) | **`-13.41 us` (`-0.778 ms`)** | **`+1.63 us` (`+0.095 ms`)** | **`+1.03 us` (`+0.059 ms`)** |
| `2` | `363.52 / 364.54` (`364.03`) | `352.26 / 352.26` (`352.26`) | `352.38 / 353.09` (`352.74`) | `352.06 / 354.30` (`353.18`) | **`-10.85 us` (`-0.629 ms`)** | **`+0.48 us` (`+0.028 ms`)** | **`+0.92 us` (`+0.053 ms`)** |
| `3` | `439.33 / 437.63` (`438.48`) | `435.20 / 432.13` (`433.66`) | `434.24 / 433.15` (`433.69`) | `432.13 / 430.91` (`431.52`) | **`-6.96 us` (`-0.404 ms`)** | **`+0.03 us` (`+0.002 ms`)** | **`-2.14 us` (`-0.124 ms`)** |
| `6` | `742.40 / 744.77` (`743.59`) | `733.02 / 736.06` (`734.54`) | `733.18 / 736.29` (`734.73`) | `736.26 / 737.28` (`736.77`) | **`-6.82 us` (`-0.395 ms`)** | **`+0.19 us` (`+0.011 ms`)** | **`+2.23 us` (`+0.129 ms`)** |
| `12` | `1315.84 / 1317.54` (`1316.69`) | `1305.60 / 1309.70` (`1307.65`) | `1306.62 / 1307.55` (`1307.09`) | `1303.55 / 1304.58` (`1304.07`) | **`-12.62 us` (`-0.732 ms`)** | **`-0.57 us` (`-0.033 ms`)** | **`-3.59 us` (`-0.208 ms`)** |
| `24` | `2432.00 / 2434.24` (`2433.12`) | `2421.18 / 2428.00` (`2424.59`) | `2420.96 / 2427.20` (`2424.08`) | `2417.66 / 2423.81` (`2420.73`) | **`-12.39 us` (`-0.718 ms`)** | **`-0.51 us` (`-0.030 ms`)** | **`-3.86 us` (`-0.224 ms`)** |
| `48` | `3575.84 / 3583.01` (`3579.43`) | `3554.34 / 3560.45` (`3557.39`) | `3554.14 / 3559.42` (`3556.78`) | `3555.17 / 3559.42` (`3557.30`) | **`-22.13 us` (`-1.284 ms`)** | **`-0.62 us` (`-0.036 ms`)** | **`-0.10 us` (`-0.006 ms`)** |

### `TP = 3` (`I_routed = 768`, `I_shared = 768`, `2*I_shared = 1536`) — Warm-L2 Captured CUDA Graph Replay (us/layer & 58-Layer ms/step)

| `M` | `serial_1stream` B1/B2 (us) | `base_1107` B1/B2 (us) | `cand_swiglu_only` B1/B2 (us) | `cand_full` B1/B2 (us) | $\Delta$ `cand_full` vs `serial_1stream` (us / 58L ms) | $\Delta$ `cand_swiglu_only` vs `1107` (us / 58L ms) | $\Delta$ `cand_full` vs `1107` (us / 58L ms) | Bit-Exact (`0` u16 diff) |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `1` | `308.10 / 307.97` (`308.04`) | `298.40 / 297.80` (`298.10`) | `298.75 / 297.78` (`298.26`) | `298.11 / 297.92` (`298.01`) | **`-10.02 us` (`-0.581 ms`)** | **`+0.16 us` (`+0.010 ms`)** | **`-0.09 us` (`-0.005 ms`)** | `PASS` |
| `2` | `434.88 / 439.32` (`437.10`) | `424.63 / 431.23` (`427.93`) | `425.12 / 431.85` (`428.49`) | `424.68 / 428.50` (`426.59`) | **`-10.51 us` (`-0.610 ms`)** | **`+0.56 us` (`+0.032 ms`)** | **`-1.34 us` (`-0.078 ms`)** | `PASS` |
| `3` | `612.79 / 612.68` (`612.73`) | `602.10 / 601.75` (`601.92`) | `602.16 / 602.06` (`602.11`) | `602.21 / 602.15` (`602.18`) | **`-10.55 us` (`-0.612 ms`)** | **`+0.18 us` (`+0.011 ms`)** | **`+0.26 us` (`+0.015 ms`)** | `PASS` |
| `6` | `995.22 / 1000.65` (`997.93`) | `986.63 / 986.74` (`986.68`) | `987.35 / 986.45` (`986.90`) | `987.00 / 986.84` (`986.92`) | **`-11.01 us` (`-0.639 ms`)** | **`+0.22 us` (`+0.012 ms`)** | **`+0.24 us` (`+0.014 ms`)** | `PASS` |
| `12` | `1836.58 / 1840.88` (`1838.73`) | `1827.38 / 1830.12` (`1828.75`) | `1826.84 / 1830.66` (`1828.75`) | `1827.93 / 1831.05` (`1829.49`) | **`-9.24 us` (`-0.536 ms`)** | **`+0.00 us` (`+0.000 ms`)** | **`+0.74 us` (`+0.043 ms`)** | `PASS` |
| `24` | `3172.68 / 3174.10` (`3173.39`) | `3157.82 / 3161.51` (`3159.66`) | `3158.39 / 3160.98` (`3159.68`) | `3157.42 / 3161.32` (`3159.37`) | **`-14.02 us` (`-0.813 ms`)** | **`+0.02 us` (`+0.001 ms`)** | **`-0.30 us` (`-0.017 ms`)** | `PASS` |
| `48` | `4687.07 / 4695.74` (`4691.40`) | `4666.47 / 4673.73` (`4670.10`) | `4665.62 / 4674.27` (`4669.94`) | `4665.99 / 4674.06` (`4670.02`) | **`-21.38 us` (`-1.240 ms`)** | **`-0.16 us` (`-0.009 ms`)** | **`-0.08 us` (`-0.004 ms`)** | `PASS` |

### `TP = 3` (`I_routed = 768`, `I_shared = 768`, `2*I_shared = 1536`) — Cold-L2 + `_l2pf_ffn` Prefetch Captured CUDA Graph Replay (us/layer & 58-Layer ms/step)

| `M` | `serial_1stream` B1/B2 (us) | `base_1107` B1/B2 (us) | `cand_swiglu_only` B1/B2 (us) | `cand_full` B1/B2 (us) | $\Delta$ `cand_full` vs `serial_1stream` (us / 58L ms) | $\Delta$ `cand_swiglu_only` vs `1107` (us / 58L ms) | $\Delta$ `cand_full` vs `1107` (us / 58L ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `313.18 / 313.15` (`313.16`) | `301.89 / 302.05` (`301.97`) | `303.10 / 301.89` (`302.50`) | `302.21 / 301.06` (`301.63`) | **`-11.53 us` (`-0.669 ms`)** | **`+0.52 us` (`+0.030 ms`)** | **`-0.34 us` (`-0.019 ms`)** |
| `2` | `441.50 / 443.39` (`442.44`) | `426.82 / 432.13` (`429.48`) | `429.06 / 433.15` (`431.11`) | `430.08 / 432.10` (`431.09`) | **`-11.35 us` (`-0.659 ms`)** | **`+1.63 us` (`+0.095 ms`)** | **`+1.62 us` (`+0.094 ms`)** |
| `3` | `617.47 / 616.45` (`616.96`) | `605.18 / 604.99` (`605.09`) | `605.18 / 604.99` (`605.09`) | `606.66 / 604.16` (`605.41`) | **`-11.55 us` (`-0.670 ms`)** | **`+0.00 us` (`+0.000 ms`)** | **`+0.32 us` (`+0.019 ms`)** |
| `6` | `1000.26 / 1004.54` (`1002.40`) | `991.23 / 993.28` (`992.25`) | `991.42 / 992.26` (`991.84`) | `989.06 / 990.05` (`989.55`) | **`-12.85 us` (`-0.745 ms`)** | **`-0.42 us` (`-0.024 ms`)** | **`-2.70 us` (`-0.157 ms`)** |
| `12` | `1841.15 / 1846.24` (`1843.70`) | `1830.75 / 1832.99` (`1831.87`) | `1831.97 / 1833.79` (`1832.88`) | `1830.91 / 1831.94` (`1831.43`) | **`-12.27 us` (`-0.712 ms`)** | **`+1.01 us` (`+0.059 ms`)** | **`-0.44 us` (`-0.026 ms`)** |
| `24` | `3179.55 / 3174.43` (`3176.99`) | `3154.98 / 3161.92` (`3158.45`) | `3160.90 / 3162.11` (`3161.51`) | `3158.02 / 3166.21` (`3162.11`) | **`-14.88 us` (`-0.863 ms`)** | **`+3.06 us` (`+0.177 ms`)** | **`+3.66 us` (`+0.213 ms`)** |
| `48` | `4685.63 / 4695.04` (`4690.34`) | `4672.00 / 4677.63` (`4674.82`) | `4671.52 / 4678.66` (`4675.09`) | `4675.62 / 4679.52` (`4677.57`) | **`-12.77 us` (`-0.740 ms`)** | **`+0.27 us` (`+0.016 ms`)** | **`+2.75 us` (`+0.160 ms`)** |

## Memory & Register Observations

- **Zero additional memory footprint**: No new workspace buffers, persistent tensors, or CUDA streams are allocated. `_shared_experts_stream` and `_input_ready_event` are already allocated once per `SharedExpertsRunner`.
- **Kernel compilation & register profile**: `moe_swiglu_cast` uses `0 B` dynamic shared memory (`1` CTA per slab, `96-128` threads) and compiles cleanly via TileLang TVM FFI for all `intermediate in {576, 768, 1152, 1536}`.

## Errors Encountered During Development

1. **`from __future__ import annotations` with nested `@T.prim_func` in benchmark script**: Defining `@T.prim_func` inside a helper function in a module with `from __future__ import annotations` caused `typing.get_type_hints` inside TileLang's decorator to fail resolving local closure variables (`rows`, `intermediate`). Removing `from __future__ import annotations` in `bench_moe_swiglu_and_dual_stream.py` resolved the issue.
2. **Container editable vLLM bind mount**: Mounting the source `.work/upstreams/vllm/vllm` directory over `/usr/local/lib/python3.12/dist-packages/vllm` hid compiled C++/CUDA extensions (`.so`) that live in `/opt/spark-ds41f/candidate/vllm/vllm` inside `spark-ds41f:1107`. Mounting only the three modified `.py` files into `/opt/spark-ds41f/candidate/vllm/...` preserved all compiled extensions.

## Conclusion

1. **Standalone `moe_swiglu_cast`**: Skipping empty slabs (`orders[0] < 0`) and 1-row-tile trailing rows (`orders[1] < 0`), cascading `Order`/`Weights` preloads before `pdl.chain()`, and using `rows_per_cta = 4` at `intermediate = 768` reduces `moe_swiglu_cast` latency inside captured CUDA graphs across every shape and batch size:
   - `intermediate = 576` (TP4): **`+4.20%` to `+28.28%` faster** (`-0.045` to `-1.464 us/layer`, `-0.003` to `-0.085 ms` per 58-layer step).
   - `intermediate = 768` (TP3): **`+3.47%` to `+18.08%` faster** (`-0.040` to `-0.499 us/layer`, `-0.002` to `-0.029 ms` per 58-layer step).
   - `intermediate = 1152`: **`+13.48%` to `+18.65%` faster** (`-0.225` to `-1.057 us/layer`).
   - `intermediate = 1536`: **`+8.71%` to `+16.05%` faster** (`-0.193` to `-0.976 us/layer`).
2. **Dual-Stream `shared_experts` Overlap**: Overlapping `shared_experts` on a secondary CUDA stream inside captured decode CUDA graphs saves **`5.3` to `22.1 us/layer` (`-0.31` to `-1.28 ms` per 58-layer step) on TP4** and **`9.2` to `21.4 us/layer` (`-0.54` to `-1.24 ms` per 58-layer step) on TP3** vs 1-stream serial execution, with `100%` bit-exact outputs (`0` uint16 diffs). Fixing `MoERunner._forward_impl` so `fold_shared == True` no longer bypasses `router._post_gate_hook` (while pre-recording `_input_ready_event` before `router.select_experts`) additionally improves cold-L2 + `_l2pf_ffn` verify-batch (`M = 6..24`) latency by up to **`-3.86 us/layer` (`-0.224 ms/step`)** over `1107`.
