# TP3 NCCL SP Channels and Sparse-MLA Head-Block / Decode Occupancy

## Base commit and candidate

- **Deployment base commit**: `0b3e98934f0aa7c7c251b79dae3ac148f869304e`
- **Base runtime profile**: `experiments/2026-10-26-spark-ds41f-candidate/1107/tp3.json` (`spark-ds41f:1107`, vLLM patch head `c47b54ae471a24b830e3515212b8c0bf5ee14604`, tree `13e4781d19bc4a18b0d3d4daad7921c465e3355e`)
- **Open levers**: `TODO.md` lines 532-546
  1. NCCL channels and balanced switchless Ring policy for TP3 sequence-parallel (SP) prefill collectives (`TODO.md` line 532).
  2. Sparse-MLA prefill and multi-segment decode occupancy with few heads per rank (`TODO.md` line 543).

## Audit Findings

### 1. TP3 NCCL Channels & SP Prefill Collectives (`TODO.md` line 532)

1. **Current TP3 vs TP4 Configuration (`1107/tp3.json` vs `1107/tp4.json`)**:
   | Variable | TP3 (`tp3-triangle`) | TP4 (`tp4-ring`) | `scripts/topology.py` (`oneshot-direct`, `N=3`) |
   | --- | --- | --- | --- |
   | `NCCL_ALGO` | unset (Tree + Ring) | `"Ring"` | `"Ring"` |
   | `NCCL_MAX_NCHANNELS` | `"8"` | `"4"` | `"4"` |
   | `NCCL_MIN_NCHANNELS` | unset | `"4"` | `"4"` |
   | `NCCL_BUFFSIZE` | `"1048576"` (1 MiB) | `"4194304"` (4 MiB) | `"4194304"` (4 MiB) |
   | `NCCL_SWITCHLESS_BIDIRECTIONAL` | unset (`0`, unidirectional) | `"2"` (balanced CW/CCW + NIC root) | `"2"` |
   | `NCCL_MIN_TRAFFIC_PER_CHANNEL` | unset (32 KiB default) | `"512"` | `"512"` |
   | `NCCL_THREAD_THRESHOLDS` | unset | `"-2 -2 -2 1 1 1"` | `"-2 -2 -2 1 1 1"` |

2. **Actual TP3 Rank-0 NCCL Init Log (`experiments/2026-10-03-tp3-revalidation/hardware/runs.tar.gz:dgx1-probe.log`)**:
   - Topology search across the two merged dual-port ConnectX-7 RoCE NICs (`NET/0-4`, `NET/0-5`) finds 32 virtual channels and caps at `NCCL_MAX_NCHANNELS=8`:
     - `8 coll channels, 8 collnet channels, 0 nvls channels, 8 p2p channels, 2 p2p channels per peer`
     - `Ring 00 : 2 -> 0 -> 1` through `Ring 07 : 2 -> 0 -> 1` (all 8 rings unidirectional clockwise)
     - `Trees [0]..[7]` enabled alongside Ring
     - `P2P Chunksize set to 131072` (clamped from GB10s 262,144 B default by `NCCL_BUFFSIZE=1048576 / NCCL_STEPS(8) = 131072` in `src/init.cc:836`)

3. **Why TP3 SP Prefill Collectives Hit PyNCCL**:
   - `SparknetOneShotAllReduce` caps `should_reduce_scatter`, `should_exchange`, and `should_all_to_all_rows` at `_slot_bytes = 4 MiB` (`4,194,304` B).
   - At TP3s 6,144-token prefill chunk (`L = 2,048` rows/rank), attention front `sp_rows.gather` (`21 MiB` or `60 MiB`), WO 2-slice `exchange_into` (`30 MiB/slice`), MoE `sp.gather_mxfp8` (`31.35 MiB`), and MoE `reduce_scatter_parts` (`60 MiB`) all exceed `4 MiB` and execute on PyNCCL (`comm.all_gather` or `comm.send`/`comm.recv`).
   - With unidirectional rings (`Ring 00..07: 2 -> 0 -> 1`), `ncclAllGather` sends 100% of ring traffic over only one of the two direct peer cables per rank while leaving the counter-clockwise direct peer cable idle, launches 8 collective CTAs (16.7% of GB10s 48 SMs) that contend with overlapped GEMMs in `project_reduce_scatter`, and clamps P2P `send`/`recv` chunk size to 128 KiB.
   - `patches/nccl/0002..0004` (`balanceSwitchlessRings`) already supports `nranks == 3` (`tp3-triangle`) under `NCCL_SWITCHLESS_BIDIRECTIONAL=2`, alternating ring direction and NIC root across 4 channels (`ch0`: CW/root0, `ch1`: CCW/root1, `ch2`: CCW/root0, `ch3`: CW/root1) while raising `NCCL_BUFFSIZE` to `4 MiB` (`P2P Chunksize = 262,144` B).

4. **100% Bit-Identity Proof (NCCL Arm)**:
   - In `vllm/models/deepseek_v4_1/tilelang/collectives.py` and `sp_prefill.py`, every PyNCCL call is a pure unreduced byte transfer (`comm.all_gather` or `comm.send`/`comm.recv`); all reductions run locally on the GPU via `rank_order_sum` (FP32 accumulation in fixed rank order `0, 1, ..., world - 1`). Changing NCCL channels, ring directions, or buffer sizes alters zero floating-point operations and preserves 100% bit-identical outputs.

### 2. Sparse-MLA Prefill & Multi-Segment Decode Occupancy (`TODO.md` line 543)

1. **Compiled SM121 Kernel Resource Usage (`sparse_mla_fwd_v41`)**:
   - `block_H = 32, threads = 256` (`__launch_bounds__(256, 1)`): `89,344` B (`87.25` KiB) dynamic shared memory in `extend`, `87,040` B (`85.0` KiB) in `decode`.
   - `block_H = 16, threads = 128` (`__launch_bounds__(128, 1)`): `70,912` B (`69.25` KiB) in `extend`, `68,608` B (`67.0` KiB) in `decode`.
   - Because GB10 (SM121) allows `99` KiB (`101,376` B) shared memory per SM and has `48` SMs, every `sparse_mla_fwd_v41` CTA is strictly `1` CTA per SM (`48` CTAs per wave).
   - Full-chunk prefill (`rows = 6,144` on layers `0..4`, `rows = 2,048` on SP layers `5..44`) grids over `(head_blocks=1, rows)` and already splits each sequence token-wise across SMs (`128.0` and `42.67` waves of 48 SMs = `100.0%` and `99.2%` SM wave occupancy), attending to at most `128` SWA + `512` top-k = `640` entries (`5` segments of `128`).

2. **Two Concrete Occupancy & Efficiency Gaps Fixed in `vllm/0731`**:
   - **50% Wasted Warps & Tensor-Core MMAs on 16-Head Ranks (TP3 Rank 2 and TP4 All Ranks)**:
     - `DeepseekV4Attention._trim_padded_groups()` (`vllm/models/deepseek_v4_1/attention.py:1755-1815`) trims the zero-padded 3rd group on TP3 Rank 2 (`tp_rank = 2`), so TP3 has `n_local_heads = 24` on Ranks 0 & 1 and `n_local_heads = 16` on Rank 2 (and TP4 has `n_local_heads = 16` on all 4 ranks).
     - `SparseMlaV41Plan.routes` previously hardcoded `block_H = 32, threads = 256` (8 warps) for `extend` and `decode` even when `heads = 16`, where `head_blocks = cdiv(16, 32) = 1` (same as `cdiv(16, 16) = 1`) and 4 of the 8 warps in every CTA executed `Q*K^T` and `P*V` MMAs, softmax, and `wo_input` on zero-padded dummy heads `16..31` and discarded them.
     - Capping `block_H = min(block_H, tilelang.cdiv(heads, 16) * 16)` and scaling `threads` accordingly switches 16-head `extend` and `decode` to `block_H = 16, threads = 128` (4 warps), cutting tensor-core MMA and warp work by 50% and dynamic shared memory by `18.4` KiB/CTA on TP3 Rank 2 and all TP4 ranks. Adding `skip_inactive: bool = True` to `sparse_mla_fwd_v41` preserves inactive-segment and dead-row skipping on `decode` at `block_H = 16` while keeping single-wave `split` (`skip_inactive=False`) branch-free.
   - **Under-Filled `< 1` Wave Transition (`rows = 29..47`) on Indexed Layers**:
     - `DECODE_MAX_WAVES = 3` previously set `decode_max_rows = 3 * 48 // (5 * 1) = 28` on 5-segment indexed layers. For `rows = 29..47` (short prefill tails, prefix-cached chunks, or mixed batches), `SparseMlaV41Plan` switched from segment-split `decode` (`5 * rows = 145..235` CTAs) to unsplit `extend` (`1 * rows = 29..47` CTAs < 48 SMs), leaving up to 19 SMs idle while each active SM ran all 5 KV segments serially.
     - Setting `decode_max_rows = max(DECODE_MAX_WAVES * sms // (segments * head_blocks), sms // head_blocks if segments > 1 else 0)` raises `decode_max_rows` on 5-segment indexed layers from `28` to `48` (keeping segment-split `decode` until unsplit `extend` fills a full wave of 48 SMs) while leaving 1-segment SWA layers (including the DSpark drafter in `_draft_workspace_lane`) unchanged at `144`. Peak workspace memory across both lanes changes by `0` bytes.

3. **100% Bit-Identity Proof (`vllm/0731`)**:
   - `segment_chunks` (`4`), `block_I` (`32`), FP32 partial accumulators, the explicit `FOLD` expression (`L = T.fma(l, b, T.fmul(L, a))`), and `wo_input` rotation/quantization are unchanged; `extend`, `decode`, and `split` at `block_H = 16` and `block_H = 32` are bit-identical (`differ == 0` in `int16` view).

## Hypothesis

1. **Arm 1 (`tp3-c4-ring.json` and `candidate.json`)**: Replacing TP3s default 8-channel unidirectional NCCL policy (1 MiB buffer) with 4 Ring channels (`tp3-c4-ring.json`) and the full balanced bidirectional 4-channel Ring + 4 MiB switchless policy (`candidate.json`) will reduce PyNCCL SP collective latency and SM contention during cooled 32K and 64K prefill without regressing decode throughput or altering output bits.
2. **Arm 2 (`tp3-mla-occupancy.json`)**: Right-sizing 16-head `extend`/`decode` CTAs to `block_H = 16, threads = 128` and keeping segment-split `decode` on indexed layers up to 1 full wave (`rows <= 48`) will eliminate 50% wasted warps/MMAs on TP3 Rank 2 (and all TP4 ranks) and eliminate `< 1` wave SM under-utilization on `rows = 29..48` while keeping 100% bit-identical outputs and zero workspace memory increase.

## Intended delta

Each arm changes one causal variable at a time from `base.json` (`1107/tp3.json`):
- `base.json`: Unchanged Candidate `1107` TP3 control (`spark-ds41f:1107`).
- `tp3-c4-ring.json`: Isolates `NCCL_ALGO="Ring"`, `NCCL_MIN_NCHANNELS="4"`, `NCCL_MAX_NCHANNELS="4"` on `spark-ds41f:1107` (zero image rebuild).
- `candidate.json`: Full switchless `tp3-triangle` balanced 4-channel Ring + 4 MiB NCCL policy (`NCCL_ALGO="Ring"`, `NCCL_MIN_NCHANNELS="4"`, `NCCL_MAX_NCHANNELS="4"`, `NCCL_BUFFSIZE="4194304"`, `NCCL_SWITCHLESS_BIDIRECTIONAL="2"`, `NCCL_MIN_TRAFFIC_PER_CHANNEL="512"`, `NCCL_THREAD_THRESHOLDS="-2 -2 -2 1 1 1"`) on `spark-ds41f:1107` (zero image rebuild).
- `tp3-mla-occupancy.json`: Isolates `vllm/0731-perf-dsv41-right-size-sparse-mla-head-block-and-deco.patch` with unchanged TP3 NCCL environment.
- `tp3-combined.json`: Combines `candidate.json` NCCL policy + `vllm/0731`.

## Quality and safety gates

- **100% bit-identity and exact precision**: Zero precision reduction or rounding changes anywhere. Greedy token outputs and logprobs must match `base.json` bit-for-bit (`0` token diffs).
- **Memory floor**: Host `MemAvailable` >= `4.0 GiB` on all ranks; zero workspace or KV-cache capacity regression.
- **Request-failure tolerance**: `0` failed requests, `0` NCCL/SparkNet errors or timeouts.
- **Verifyrank-0 NCCL init log**: Confirm `4 coll channels`, bidirectional ring pairs (`2 -> 0 -> 1` and `1 -> 0 -> 2`), and `P2P Chunksize set to 262144` under `candidate.json`.

## Workloads and Screening Plan (`TODO.md` lines 532-546)

1. **Lean screening on cooled 32K and 64K prefill**:
   - Compare `candidate.json` (and `tp3-c4-ring.json`) against recorded `1107` TP3 reference runs (and `base.json` if needed) with at least two boots per candidate arm (boot 1 + repeat boot 2) on cooled 32K and 64K prefill prompts (`temperature=0`, `seed=42`).
   - Record rank-0 `docker logs` NCCL init lines, TTFT / prefill tok/s at 32K and 64K, decode TPS, and exact output token hashes under `runs/`.
2. **Full `bin/spark bench` suite**:
   - Run only for an arm that clears the 32K/64K prefill screen with repeatable gain and zero regressions.

## Acceptance criteria

- Repeatable prefill throughput improvement (lower TTFT at 32K and 64K across both boots) with non-overlapping confidence intervals vs control, no decode TPS or memory regression, and 100% bit-identical outputs.
