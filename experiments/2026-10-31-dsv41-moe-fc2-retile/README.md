# 2-Stage Pipeline for Decode MoE `down` (`FC2`) Grouped GEMM on SM120 / SM121

## Problem

In `MoeExpertsV41Plan` (`vllm/models/deepseek_v4_1/tilelang/moe.py`), both the `gate_up` (`FC1`, `K = 5120`) and `down` (`FC2`, `K = 576` at TP4 or `K = 768` at TP3) grouped FP8×FP4 GEMMs (`grouped_fp8_fp4_gemm`) were compiled with `num_stages = 3` at `block_M = 32, block_N = 128, block_K = 128`.

Each pipeline stage of `grouped_fp8_fp4_gemm` at `block_M = 32, block_N = 128` allocates:
- `A_shared`: `32 × 128` bytes = `4 KiB`
- `B_shared`: `128 × 128` unpacked FP4 bytes = `16 KiB`
- `SFA_shared`, `SFB_shared`, `SFC8`: `~1.5 KiB`
- **Total per stage**: `~21.5 KiB`

With `num_stages = 3`, per-CTA shared memory is `~64.5 KiB`, which exceeds half of GB10's (`SM121`) `99 KiB` shared memory per SM and caps residency at **1 CTA per SM** (`48` active CTAs across GB10's 48 SMs).

While `gate_up` (`K = 5120`, `40` K-blocks) benefits from 3-stage TMA pipelining across its deep K loop, decode `down` (`rows < 16`, `not late`) runs only **5 K-blocks** (`K = 576` at TP4, 4 full tiles + 64-wide tail) or **6 K-blocks** (`K = 768` at TP3) across a `40 × tiles` grid (`240` CTAs at `rows = 1`, `1440` CTAs at `rows = 6`) where many tiles exit early (`TileExpert[t] < 0`). At 1 CTA/SM, CTA launch/prologue/epilogue and wave tail latency dominate.

## Solution

Configure `MoeExpertsV41Plan` with `down_num_stages = 2` (default `2`) for the decode `down` kernel (`rows < LATE_TRIGGER_ROWS`), while keeping `num_stages = 3` for `gate_up`, `late_gate_up`, `late_down`, `prefill_gate_up`, and `prefill_down`:

- At `down_num_stages = 2`, per-CTA shared memory drops to `~43 KiB <= 49.5 KiB`, doubling residency to **2 CTAs per SM** (`96` co-resident CTAs on GB10).
- Requires **zero** changes to `prepare_weights` (`block_N = 128` is unchanged).
- Produces **bit-identical** outputs (`out2 == out3` to the bit) because the K-reduction order within each row is identical.

## Benchmark Results on `dgx4` (GB10, `spark-ds41f:1030`)

### TP4 `down_proj` (`N = 5120, K = 576`, 384 experts, top-6)

| Decode `rows` | `num_stages=3` (1 CTA/SM) | `num_stages=2` (2 CTAs/SM) | Speedup | Per-step savings (58 MoE layers) |
|---:|---:|---:|---:|---:|
| `1` | `99.55 us` | `58.59 us` | **1.70x** | `-2.38 ms` |
| `2` | `115.90 us` | `66.56 us` | **1.74x** | `-2.86 ms` |
| `4` | `226.53 us` | `191.52 us` | **1.18x** | `-2.03 ms` |
| `6` | `294.08 us` | `259.33 us` | **1.13x** | `-2.02 ms` |
| `8` | `362.69 us` | `332.06 us` | **1.09x** | `-1.78 ms` |
| `12` | `506.91 us` | `468.13 us` | **1.08x** | `-2.25 ms` |

### TP3 `down_proj` (`N = 5120, K = 768`, 384 experts, top-6)

| Decode `rows` | `num_stages=3` (1 CTA/SM) | `num_stages=2` (2 CTAs/SM) | Speedup | Per-step savings (58 MoE layers) |
|---:|---:|---:|---:|---:|
| `1` | `99.52 us` | `56.51 us` | **1.76x** | `-2.49 ms` |
| `2` | `122.02 us` | `70.78 us` | **1.72x** | `-2.97 ms` |
| `4` | `272.67 us` | `235.78 us` | **1.16x** | `-2.14 ms` |
| `6` | `355.55 us` | `328.83 us` | **1.08x** | `-1.55 ms` |
| `8` | `448.32 us` | `419.94 us` | **1.07x** | `-1.65 ms` |
| `12` | `627.71 us` | `609.34 us` | **1.03x** | `-1.07 ms` |

## Verification

Verified on `dgx4` inside `spark-ds41f:1030`:
- `pytest -q --noconftest -o addopts="" -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/moe/test_deepseek_v41_tilelang_moe.py -k "test_decode_down_two_stage_pipeline_gives_three_stage_bits or test_moe_experts or test_tile_height"` -> `46 passed`, `0 PDL violations`.
- Bit-for-bit equality confirmed across all tested row counts (`rows = 1..4096`) and both TP4 (`K = 576`) and TP3 (`K = 768`).
