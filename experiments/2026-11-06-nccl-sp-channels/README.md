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
   - **Wave-Boundary Ablation (`rows = 29..48`, `DECODE_MAX_WAVES = 3` vs `5`, `receipts/screen-mla-waves5-boot{1,2}.jsonl`)**:
     - We tested raising `decode_max_rows` on 5-segment indexed layers from `28` (`DECODE_MAX_WAVES = 3`) to `48` (`sms // head_blocks`, 5 waves). Across two fresh container boots on GB10 (`sp8`), `decode_max_rows = 48` sped up `rows = 29..48` when only 2 of 5 segments were active (`partial_256`: `1.32x..1.95x`), **but regressed `rows = 29..48` when all 5 segments were active (`full_640`: `+28%..+105%` slower)** because single-wave `extend` (`29..48` CTAs <= 48 SMs) folds all 5 segments in registers (`0.049 ms` warm / `0.061..0.069 ms` cold) without writing `2.36 MiB` `Partial` to global memory or launching the 24-wave merge kernel (`0.056..0.105 ms` warm / `0.095..0.128 ms` cold).
     - Therefore, refined `vllm/0731` (`609f622114a9ddac55ef5ab329294a71c5a061d2`, tree `74723c61264412617284b752d87ee1db4348c3de`) retains `DECODE_MAX_WAVES = 3` (`decode_max_rows = 28`) and isolates the strictly beneficial `block_H = 16, threads = 128` (`heads <= 16`) + `skip_inactive: bool = True` change.

4. **Phase 2A Standalone 2-Boot GPU Benchmark Results on GB10 (`sp8`, `receipts/screen-mla-boot{1,2}.jsonl`)**:
   - **GPU `pytest` contracts**: `20 passed` on `boot1` (`11.58s`, `receipts/gpu-pytest-boot1.txt`) and `boot2` (`11.52s`, `receipts/gpu-pytest-boot2.txt`), `0` PDL violations.
   - **100% `int16` bit-identity**: `86/86` checks across `boot1` and `boot2` passed with `int16_differ == 0` across all shapes and explicit `extend`/`decode`/`split` routes.
   - **2-Boot Median Latency Summary (`screen_mla.py`, 15 timed iterations/point, alternating order on `boot2`)**:
     | Heads | `idx_width` | `rows` | Context Mode | Cache Mode | Route | Control `boot1` / `boot2` (ms) | Candidate `0731` `boot1` / `boot2` (ms) | Delta % | Speedup |
     | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
     | `16` | `512` | `24` | `full_640` | `warm` | `decode` | `0.0444` / `0.0407` | `0.0387` / `0.0370` | **`-11.09%`** | **`1.125x`** |
     | `16` | `512` | `28` | `full_640` | `warm` | `decode` | `0.0410` / `0.0410` | `0.0389` / `0.0388` | **`-5.23%`** | **`1.055x`** |
     | `16` | `512` | `29` | `full_640` | `warm` | `extend` | `0.0493` / `0.0491` | `0.0434` / `0.0430` | **`-12.22%`** | **`1.139x`** |
     | `16` | `512` | `48` | `full_640` | `warm` | `extend` | `0.0510` / `0.0512` | `0.0451` / `0.0449` | **`-12.05%`** | **`1.137x`** |
     | `16` | `512` | `1024` | `full_640` | `64MiB_wash` | `extend` | `1.0749` / `1.0769` | `0.9611` / `0.9592` | **`-10.76%`** | **`1.121x`** |
     | `16` | `512` | `1024` | `full_640` | `warm` | `extend` | `1.0628` / `1.0630` | `0.9459` / `0.9459` | **`-11.01%`** | **`1.124x`** |
     | `16` | `512` | `2048` | `full_640` | `64MiB_wash` | `extend` | `2.1081` / `2.1071` | `1.8728` / `1.8624` | **`-11.39%`** | **`1.129x`** |
     | `16` | `512` | `2048` | `full_640` | `warm` | `extend` | `2.0893` / `2.0929` | `1.8613` / `1.8415` | **`-11.46%`** | **`1.129x`** |
     | `16` | `512` | `4096` | `full_640` | `64MiB_wash` | `extend` | `4.1766` / `4.1756` | `3.7013` / `3.6996` | **`-11.39%`** | **`1.129x`** |
     | `16` | `512` | `4096` | `full_640` | `warm` | `extend` | `4.1719` / `4.1634` | `3.6883` / `3.6846` | **`-11.55%`** | **`1.131x`** |
     | `16` | `512` | `6144` | `full_640` | `64MiB_wash` | `extend` | `6.2228` / `6.2227` | `5.5211` / `5.5262` | **`-11.23%`** | **`1.127x`** |
     | `16` | `512` | `6144` | `full_640` | `warm` | `extend` | `6.2114` / `6.2124` | `5.5080` / `5.5077` | **`-11.33%`** | **`1.128x`** |
     | `16` | `0` | `6144` | `full_640` | `64MiB_wash` | `extend` | `1.7958` / `1.7804` | `1.5941` / `1.5937` | **`-10.86%`** | **`1.122x`** |
     | `24` | `512` | `6144` | `full_640` | `64MiB_wash` | `extend` | `6.3280` / `6.3246` | `6.3281` / `6.3260` | `+0.01%` | `1.000x` |
   - Across 40 indexed layers (`-0.699 ms/layer`) + 38 SWA-only layers (`-0.194 ms/layer`), `vllm/0731` saves **`35.34 ms` per 6,144-token prefill chunk** (`~212 ms` at 32K/36K prefill, `~389 ms` at 64K prefill) on TP3 Rank 2 and all TP4 ranks.

3. **100% Bit-Identity Proof (`vllm/0731`)**:
   - `segment_chunks` (`4`), `block_I` (`32`), FP32 partial accumulators, the explicit `FOLD` expression (`L = T.fma(l, b, T.fmul(L, a))`), and `wo_input` rotation/quantization are unchanged; `extend`, `decode`, and `split` at `block_H = 16` and `block_H = 32` are bit-identical (`differ == 0` in `int16` view).


### 3. Phase 2B 2-Boot 3-Rank TP3 NCCL Collective Benchmark (`dgx1-3`, `receipts/screen-nccl-tp3-boot{1,2}.jsonl`)

- **Bit-identity**: `56/56` checks across `boot1` and `boot2` passed (`differ_ag == 0`, `differ_a2a == 0`, `differ_rank_order_ar == 0`).
- **Rank-0 NCCL Init Topology (`receipts/nccl-init-boot1-*-rank0.txt`)**:
  - `control_c8_1m`: `coll_channels=8, p2p_channels=8, p2p_chunk=131072`, 8 unidirectional rings (`Ring 00..07 : 2 -> 0 -> 1`).
  - `cand_c4_1m`: `coll_channels=4, p2p_channels=4, p2p_chunk=131072`, 4 unidirectional rings (`Ring 00..03 : 2 -> 0 -> 1`).
  - `cand_c4_4m_bidir`: `coll_channels=4, p2p_channels=4, p2p_chunk=262144`, 2 CW + 2 CCW rings (`Ring 00,03: 2->0->1`, `Ring 01,02: 1->0->2`).
  - `cand_c4_8m_bidir`: `coll_channels=4, p2p_channels=4, p2p_chunk=524288`, 2 CW + 2 CCW rings (`8 MiB` buffer).
  - `cand_c6_8m_bidir` (**winner**, updated in `candidate.json` and `tp3-combined.json`): `coll_channels=6, p2p_channels=8, p2p_chunk=524288`, **3 CW rings (`Ring 00..02 : 2 -> 0 -> 1`) + 3 CCW rings (`Ring 03..05 : 1 -> 0 -> 2`)** (`8 MiB` buffer).
- **Aggregate Latency Across All 8 Collectives (2-boot mean of 3-rank synchronized medians)**:
  | Token Count `T` | `control_c8_1m` (ms) | `cand_c4_1m` | `cand_c4_4m_bidir` | `cand_c4_8m` | `cand_c4_8m_bidir` | `cand_c6_8m` | `cand_c6_8m_bidir` (**Winner**) |
  | ---: | ---: | --- | --- | --- | --- | --- | --- |
  | `16` | `0.5093` | `0.6248` (`+22.7%`) | `0.4575` (`-10.2%`) | `0.4436` (`-12.9%`) | `0.4431` (`-13.0%`) | `0.5172` (`+1.6%`) | **`0.4437` (`-12.9%`, `1.148x`)** |
  | `2048` | `6.2628` | `6.5331` (`+4.3%`) | `6.2981` (`+0.6%`) | `6.4950` (`+3.7%`) | `6.3034` (`+0.6%`) | `6.2630` (`+0.0%`) | **`6.0482` (`-3.4%`, `1.035x`)** |
  | `4096` | `12.3353` | `12.9647` (`+5.1%`) | `12.3654` (`+0.2%`) | `12.6670` (`+2.7%`) | `12.2967` (`-0.3%`) | `12.3262` (`-0.1%`) | **`12.1595` (`-1.4%`, `1.014x`)** |
  | `6144` | `18.2598` | `19.1130` (`+4.7%`) | `18.4717` (`+1.2%`) | `18.8095` (`+3.0%`) | `18.3293` (`+0.4%`) | `18.3633` (`+0.6%`) | **`17.5272` (`-4.0%`, `1.042x`)** |

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
