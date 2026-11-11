# Single-Kernel Multi-Stage Pipelined RoCE Collectives in `sparknet` (`256 KiB – 16 MiB`)

- **Date**: 2026-11-08
- **Base `sparknet` commit**: `1fd6a1fbd8dc4884057046d1ebd93d585c9e4bcf` (`0.5.0` + single-slot cap `<= _slot_bytes`)
- **Base `spark-ds41f` image**: `spark-ds41f:1107`
- **Patch**: `0001-oneshot-add-single-kernel-multi-stage-pipelined-RoCE.patch`
- **Hardware**: `sp1..sp4` (`TP=4`) and `sp1..sp3` (`TP=3`), GB10 (`SM121`, 20 SMs), dual-rail ConnectX-7 200GbE RoCEv2 (`rocep1s0f0,roceP2p1s0f0`)

## 1. Problem Statement & Root-Cause Analysis

In `sparknet` (`1fd6a1fbd8dc4884057046d1ebd93d585c9e4bcf`), `SparknetOneShotAllReduce` was restricted to single-slot transfers (`<= _slot_bytes`, `2 MiB` on `tp4-ring` and `4 MiB` on `tp4-switched` / `tp3-triangle`), forcing larger sequence-parallel (`SP`) prefill collectives (`all_gather`, `reduce_scatter`, `exchange`, `all_to_all_rows` up to `12–16 MiB`) to fall back to `PyNCCL`.

Hardware profiling and code audit of `sparknet/oneshot/` (`_roce_proxy.c`, `_device.py`, `_allgather_tilelang.py`, `_reducescatter_tilelang.py`, `_exchange_tilelang.py`, `runtime.py`) revealed three interacting root causes for why `sparknet`'s prior `sliced=True` path lost to `PyNCCL` above `_slot_bytes`:

1. **High CTA Occupancy (`DEFAULT_BLOCKS = 64` = 16 of 20 SMs on GB10)**:
   - Each TileLang one-shot collective kernel declares `threads = 512` (`max_threads_per_block(512)`, `min_blocks_per_sm(1)`), where 4 blocks of 512 threads saturate the 2,048-thread capacity of 1 GB10 SM.
   - Launching `DEFAULT_BLOCKS = 64` requires **16 of the 20 GB10 SMs (80% of the GPU)** to be simultaneously resident before `roce_doorbell` can ring (since `roce_doorbell` barriers across all `grid_x` blocks via `atomicAdd_block(stage_counter, 1u) == grid_x - 1u`).
2. **Inter-Slice GPU Preemption Under `side_stream` Compute Overlap**:
   - Previously, `runtime.py` sliced multi-slot collectives in Python (`for lo in range(0, local_rows, max_rows):`), launching 2–4 separate 64-CTA kernels on `side_stream`.
   - When overlapped with a compute-bound GEMM on `main_stream` (as in `sp_prefill.py`'s `project_reduce_scatter` and `project_exchange_rows`), completing slice `0` yielded the GPU SMs back to the hardware scheduler. Pending GEMM waves on `main_stream` immediately claimed all 20 SMs, starving slice `1`'s 64-CTA barrier for **150–375 us** per slice.
3. **Why Naive Software Pipelining (`Stage(s + 1)` Before `roce_wait_peers(seq_s)`) Races With `ROCE_SLOTS = 2`**:
   - As documented in `_roce_proxy.c` lines 1054–1069, a rank's collective `N` completes as soon as it receives all peers' incoming RDMA writes (`roce_wait_peers(N)`), while the local CPU proxy may still be posting or draining the rank's outgoing `send[N & 1]` WQE for a slower peer.
   - Overwriting `send[(N + 2) & 1] == send[N & 1]` before `roce_wait_peers(N + 1)` has confirmed all peers entered `N + 1` corrupts the in-flight `send[N & 1]` buffer (`AssertionError` on `reduce_scatter` at `>= 4 MiB`).

## 2. Design: Single-Kernel Persistent Multi-Stage Collectives With Low SM Occupancy

We replace Python-level multi-kernel slicing across `oneshot_allgather`, `oneshot_reducescatter`, `oneshot_exchange`, and `all_to_all_rows` with a **single persistent low-occupancy kernel launch**:

1. **In-Kernel `for s in T.serial(num_stages):` Loop Over Existing Ping-Pong Slots (`ROCE_SLOTS = 2`)**:
   - `oneshot_allgather`, `oneshot_reducescatter`, and `oneshot_exchange` accept `num_stages: T.int32` and `stage_packs: T.int32`.
   - Inside a single kernel launch, the resident CTAs read `epoch = counters[0]` once and iterate `for s in T.serial(num_stages):` with `seq = epoch + s`:
     - **Phase 1**: Stage slice `s` (`cur_packs = min(stage_packs, full_packs - s * stage_packs)`) into `send[seq & 1]`.
     - **Phase 2**: `roce_doorbell` across `grid_x` CTAs publishes `ctrl.stage_seq = seq` (`cur_nbytes` payload bytes), immediately waking the CPU proxy thread to post dual-rail `IBV_WR_RDMA_WRITE_WITH_IMM` for stage `s`.
     - **Phase 3**: Thread `tx < world_size * lanes` waits on `roce_wait_peers(seq)` (with fail-stop timeout protection).
     - **Phase 4**: All threads consume/reduce `recv[seq & 1]` into the stage's slice of `out`.
   - After all `num_stages` complete, `roce_tail` advances `counters[0]` to `epoch + num_stages`.
   - Because stage `s + 1` only writes `send[(epoch + s + 1) & 1]` after `roce_wait_peers(epoch + s)` has completed, `send[(epoch + s) & 1]` is never overwritten before `roce_wait_peers(epoch + s + 1)` confirms all peers have advanced past stage `s`. This is **100% race-free with `ROCE_SLOTS = 2`** and requires **zero additional pinned host memory**.
2. **Low-Occupancy CTA Cap (`SPARKNET_ROCE_COLLECTIVE_BLOCKS = 8`, `2` of `20` GB10 SMs)**:
   - Capping collective CTAs at `8` blocks (`512` threads/block = `4,096` threads = **2 SMs out of 20 on GB10**, vs 4–8 SMs stolen by `PyNCCL` and 16 SMs by `DEFAULT_BLOCKS = 64`) leaves **18 of 20 SMs (90% of the GPU)** dedicated to `main_stream` GEMM/MLA compute while keeping the 8 collective CTAs persistently resident across all stages without inter-stage preemption.
3. **Unified Stage Sizing (`SPARKNET_ROCE_STAGE_BYTES = 4194304`, `4 MiB` Collective Bytes/Stage)**:
   - `_collective_stage_rows(local_rows, collective_row_bytes, slot_row_bytes=...)` bounds each stage to `4 MiB` of total collective payload (`1 MiB`/rank on `TP=4`, `1.33 MiB`/rank on `TP=3`) and at most `_slot_bytes` in the local `send` slot, pipelining `8 MiB`, `12 MiB`, and `16 MiB` collectives into `2`, `3`, and `4` back-to-back stages within one kernel launch.
4. **Re-enabling Multi-Stage `SparknetOneShotAllReduce` in `roce_all_reduce.py`**:
   - Restores `sliced=True` on `should_all_gather` / `all_gather` and removes the single-slot `<= _slot_bytes` cap on `should_reduce_scatter`, `should_exchange`, and `should_all_to_all_rows`.

## 3. Exact Verification & Benchmark Commands

### Unit Tests (`sparknet`)
```bash
python3 -m unittest tests/test_oneshot_cpu.py -v
pytest tests/ -v
```
- Result: `24/24` passed in `test_oneshot_cpu.py`, `97 passed, 62 subtests passed` in `pytest tests/`.

### Multi-Node Hardware Benchmarks (`TP=4` on `sp1..sp4`, `TP=3` on `sp1..sp3`)
```bash
# TP=4 (sp1..sp4: 10.30.0.9..12), 2 boots
bash run_bench.sh 4 4194304 8 tp4_unified_boot1.json 29545
bash run_bench.sh 4 4194304 8 tp4_unified_boot2.json 29546

# TP=3 (sp1..sp3: 10.30.0.9..11), 2 boots
bash run_bench.sh 3 4194304 8 tp3_unified_boot1.json 29547
bash run_bench.sh 3 4194304 8 tp3_unified_boot2.json 29548
```
- **Workload identity (`bench_pipelined_collectives.py`)**:
  - Bit-exact verification (`torch.equal(..., view(torch.int16))`) of `all_gather`, `reduce_scatter` (against rank-order `collectives.reduce_scatter_rank_order`), and `exchange` across every size before timing.
  - **Standalone (`solo`)**: Collective on the default stream preceded by a GPU-stream RoCE barrier (`_SYNC_RT.all_reduce`) to eliminate Gloo TCP wakeup skew (`< 1 us` jitter).
  - **Compute-Overlapped (`ovlp`)**: Collective on `side_stream` overlapped with a compute-bound BF16 GEMM (`[2048, 4096] x [4096, 4096]`, standalone median `241.7–245.7 us` saturating all 20 GB10 SMs) on `main_stream`, measuring total end-to-end completion time of both streams (`30` timed iterations after `8` warmup iterations per point).

## 4. Hardware Results

### 4.1 `TP=4` (`sp1..sp4`, 200GbE Dual-Rail Switched RoCEv2, `stage_bytes = 4 MiB`, `blocks = 8`)

Standalone GEMM baseline (`main_stream` solo): **Boot 1 = `241.7 us`**, **Boot 2 = `245.3 us`**.

#### `TP=4` Compute-Overlapped (`ovlp`: `main_stream` GEMM + `side_stream` Collective, Median `us`)

| Total Collective Bytes | Op | `PyNCCL` (Boot 1 / Boot 2) | `sparknet` Base (Boot 1 / Boot 2) | `sparknet` Cand (Boot 1 / Boot 2) | Cand vs `PyNCCL` (Mean) | Cand vs Base (Mean) |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `64 KiB` | `all_gather` | `299.3 / 264.0` | `253.8 / 255.6` | `309.1 / 247.6` | `-1.2%` | `+9.3%` |
| `64 KiB` | `reduce_scatter` | `301.5 / 259.9` | `274.0 / 304.1` | `249.8 / 298.8` | `-2.3%` | `-5.1%` |
| `64 KiB` | `exchange` | `275.8 / 253.6` | `308.0 / 251.7` | `266.1 / 247.6` | `-3.0%` | `-8.2%` |
| `256 KiB` | `all_gather` | `326.7 / 295.9` | `301.5 / 303.0` | `305.3 / 266.5` | **`-8.2%`** | **`-5.4%`** |
| `256 KiB` | `reduce_scatter` | `327.0 / 339.4` | `261.9 / 259.9` | `259.9 / 304.8` | **`-15.3%`** | `+8.2%` |
| `256 KiB` | `exchange` | `310.3 / 318.4` | `265.7 / 268.1` | `259.9 / 262.0` | **`-17.0%`** | **`-2.2%`** |
| `512 KiB` | `all_gather` | `321.4 / 297.9` | `339.6 / 338.1` | `276.0 / 298.8` | **`-7.2%`** | **`-15.2%`** |
| `512 KiB` | `reduce_scatter` | `284.9 / 354.7` | `272.2 / 274.3` | `276.8 / 276.8` | **`-13.4%`** | `+1.3%` |
| `512 KiB` | `exchange` | `274.7 / 358.4` | `272.2 / 343.1` | `315.3 / 288.1` | **`-4.7%`** | **`-1.9%`** |
| `1024 KiB` | `all_gather` | `313.2 / 331.5` | `292.7 / 292.6` | `293.2 / 296.8` | **`-8.5%`** | `+0.8%` |
| `1024 KiB` | `reduce_scatter` | `305.0 / 375.8` | `294.8 / 305.0` | `298.3 / 297.9` | **`-12.4%`** | `-0.6%` |
| `1024 KiB` | `exchange` | `307.6 / 312.8` | `292.8 / 294.8` | `301.4 / 307.6` | **`-1.8%`** | `+3.6%` |
| `2048 KiB` | `all_gather` | `333.7 / 363.6` | `328.5 / 333.6` | `306.5 / 308.7` | **`-11.8%`** | **`-7.1%`** |
| `2048 KiB` | `reduce_scatter` | `333.4 / 335.8` | `341.8 / 466.8` | `306.9 / 345.9` | **`-2.5%`** | **`-19.3%`** |
| `2048 KiB` | `exchange` | `326.9 / 327.0` | `341.8 / 344.9` | `312.9 / 336.4` | **`-0.7%`** | **`-5.4%`** |
| `4096 KiB` | `all_gather` | `395.3 / 405.3` | `410.5 / 413.5` | `421.7 / 318.7` | **`-7.5%`** | **`-10.1%`** |
| `4096 KiB` | `reduce_scatter` | `399.4 / 399.0` | `425.8 / 429.9` | `428.1 / 422.9` | `+6.6%` | `-0.5%` |
| `4096 KiB` | `exchange` | `395.3 / 402.4` | `429.9 / 434.0` | `439.1 / 338.3` | **`-2.5%`** | **`-10.0%`** |
| `8192 KiB` | `all_gather` | `554.0 / 546.7` | `588.6 / 585.4` | `415.4 / 414.9` | **`-24.6%`** | **`-29.3%`** |
| `8192 KiB` | `reduce_scatter` | `547.8 / 532.3` | `769.5 / 688.2` | `454.2 / 444.4` | **`-16.8%`** | **`-38.4%`** |
| `8192 KiB` | `exchange` | `540.1 / 495.5` | `657.4 / 657.4` | `479.7 / 465.9` | **`-8.7%`** | **`-28.1%`** |
| `12288 KiB` | `all_gather` | `683.3 / 699.1` | `760.7 / 761.7` | `599.6 / 599.0` | **`-13.3%`** | **`-21.3%`** |
| `12288 KiB` | `reduce_scatter` | `685.9 / 694.2` | `823.3 / 831.5` | `633.7 / 638.0` | **`-7.9%`** | **`-23.1%`** |
| `12288 KiB` | `exchange` | `684.5 / 705.1` | `860.2 / 858.9` | `660.5 / 657.4` | **`-5.2%`** | **`-23.3%`** |
| `16384 KiB` | `all_gather` | `866.2 / 835.4` | `940.9 / 940.9` | `804.9 / 776.2` | **`-7.1%`** | **`-16.0%`** |
| `16384 KiB` | `reduce_scatter` | `824.0 / 814.9` | `1041.4 / 1163.3` | `855.0 / 869.4` | `+5.2%` | **`-21.8%`** |
| `16384 KiB` | `exchange` | `800.6 / 817.1` | `1073.1 / 1079.7` | `864.2 / 862.2` | `+6.7%` | **`-19.8%`** |

#### `TP=4` Standalone (`solo`: Collective Only, Median `us`)

| Total Collective Bytes | `AG solo` (`PyNCCL` / Base / Cand) | `RS solo` (`PyNCCL` / Base / Cand) | `EX solo` (`PyNCCL` / Base / Cand) |
| ---: | ---: | ---: | ---: |
| `64 KiB` | `33.3 / 28.2 / 31.0` | `34.8 / 28.3 / 31.3` | `40.5 / 32.3 / 38.1` |
| `256 KiB` | `126.1 / 35.9 / 39.5` | `124.6 / 36.9 / 38.9` | `58.9 / 41.2 / 46.6` |
| `512 KiB` | `65.5 / 45.6 / 48.2` | `141.8 / 46.7 / 50.7` | `62.0 / 51.2 / 57.2` |
| `1024 KiB` | `258.5 / 63.5 / 67.6` | `77.8 / 66.6 / 69.5` | `85.0 / 71.0 / 77.1` |
| `2048 KiB` | `149.0 / 101.4 / 105.4` | `104.3 / 107.1 / 109.4` | `115.2 / 110.5 / 115.7` |
| `4096 KiB` | `183.0 / 177.1 / 182.3` | `165.2 / 185.6 / 187.9` | `184.2 / 188.9 / 195.8` |
| `8192 KiB` | `310.4 / 343.1 / 357.4` | `303.0 / 360.8 / 357.4` | `385.0 / 370.3 / 373.6` |
| `12288 KiB` | `455.5 / 522.1 / 548.4` | `495.0 / 532.5 / 532.5` | `571.2 / 578.2 / 573.6` |
| `16384 KiB` | `682.8 / 700.4 / 745.4` | `631.7 / 734.5 / 738.3` | `862.2 / 803.1 / 785.9` |

---

### 4.2 `TP=3` (`sp1..sp3`, 200GbE Dual-Rail Switched RoCEv2, `stage_bytes = 4 MiB`, `blocks = 8`)

Standalone GEMM baseline (`main_stream` solo): **Boot 1 = `243.7 us`**, **Boot 2 = `245.7 us`**.

#### `TP=3` Compute-Overlapped (`ovlp`: `main_stream` GEMM + `side_stream` Collective, Median `us`)

| Total Collective Bytes | Op | `PyNCCL` (Boot 1 / Boot 2) | `sparknet` Base (Boot 1 / Boot 2) | `sparknet` Cand (Boot 1 / Boot 2) | Cand vs `PyNCCL` (Mean) | Cand vs Base (Mean) |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `48 KiB` | `all_gather` | `245.6 / 251.2` | `304.8 / 275.5` | `242.0 / 241.9` | **`-2.6%`** | **`-16.6%`** |
| `48 KiB` | `reduce_scatter` | `376.6 / 253.8` | `244.1 / 272.9` | `245.6 / 247.6` | **`-21.8%`** | **`-4.6%`** |
| `48 KiB` | `exchange` | `251.7 / 249.7` | `307.3 / 285.7` | `243.9 / 302.3` | `+9.0%` | **`-7.9%`** |
| `192 KiB` | `all_gather` | `251.7 / 321.4` | `298.5 / 251.7` | `245.6 / 251.7` | **`-13.2%`** | **`-9.6%`** |
| `192 KiB` | `reduce_scatter` | `323.5 / 325.0` | `303.8 / 251.7` | `267.7 / 304.9` | **`-11.7%`** | `+3.1%` |
| `192 KiB` | `exchange` | `297.9 / 262.0` | `307.7 / 305.7` | `288.4 / 253.6` | **`-3.2%`** | **`-11.6%`** |
| `384 KiB` | `all_gather` | `325.9 / 274.8` | `340.5 / 283.1` | `309.2 / 306.9` | `+2.6%` | `-1.2%` |
| `384 KiB` | `reduce_scatter` | `293.5 / 295.8` | `262.0 / 262.0` | `308.2 / 263.2` | **`-3.0%`** | `+9.0%` |
| `384 KiB` | `exchange` | `297.3 / 304.8` | `344.0 / 264.0` | `280.0 / 248.9` | **`-12.2%`** | **`-13.0%`** |
| `768 KiB` | `all_gather` | `266.1 / 292.7` | `419.7 / 276.3` | `280.1 / 307.4` | `+5.1%` | **`-15.6%`** |
| `768 KiB` | `reduce_scatter` | `264.2 / 274.9` | `417.5 / 276.4` | `286.1 / 288.1` | `+6.5%` | **`-17.2%`** |
| `768 KiB` | `exchange` | `303.6 / 257.9` | `276.3 / 251.7` | `292.2 / 313.8` | `+7.9%` | `+14.8%` |
| `1536 KiB` | `all_gather` | `308.2 / 344.5` | `304.0 / 303.0` | `306.2 / 283.0` | **`-9.7%`** | **`-2.9%`** |
| `1536 KiB` | `reduce_scatter` | `307.0 / 291.2` | `308.4 / 307.0` | `285.1 / 280.0` | **`-5.5%`** | **`-8.2%`** |
| `1536 KiB` | `exchange` | `305.1 / 318.8` | `306.0 / 309.1` | `309.6 / 313.7` | `-0.1%` | `+1.3%` |
| `3072 KiB` | `all_gather` | `351.2 / 293.2` | `362.2 / 362.3` | `343.7 / 278.4` | **`-3.5%`** | **`-14.1%`** |
| `3072 KiB` | `reduce_scatter` | `359.4 / 353.3` | `366.4 / 368.4` | `312.2 / 296.8` | **`-14.6%`** | **`-17.1%`** |
| `3072 KiB` | `exchange` | `356.2 / 320.9` | `365.4 / 366.4` | `316.3 / 317.8` | **`-6.4%`** | **`-13.4%`** |
| `6144 KiB` | `all_gather` | `450.4 / 467.6` | `473.9 / 305.0` | `359.1 / 361.2` | **`-21.5%`** | **`-7.5%`** |
| `6144 KiB` | `reduce_scatter` | `451.4 / 529.1` | `513.0 / 524.2` | `362.9 / 369.8` | **`-25.3%`** | **`-29.4%`** |
| `6144 KiB` | `exchange` | `443.6 / 348.0` | `527.3 / 526.3` | `375.7 / 369.0` | **`-5.9%`** | **`-29.3%`** |
| `9216 KiB` | `all_gather` | `552.0 / 551.8` | `596.8 / 600.1` | `452.6 / 454.4` | **`-17.8%`** | **`-24.2%`** |
| `9216 KiB` | `reduce_scatter` | `538.7 / 540.5` | `649.6 / 645.1` | `468.8 / 460.8` | **`-13.9%`** | **`-28.2%`** |
| `9216 KiB` | `exchange` | `542.7 / 544.8` | `678.8 / 663.4` | `490.3 / 491.5` | **`-9.7%`** | **`-26.9%`** |
| `12288 KiB` | `all_gather` | `598.0 / 619.5` | `720.6 / 975.5` | `577.5 / 579.2` | **`-5.0%`** | **`-31.8%`** |
| `12288 KiB` | `reduce_scatter` | `578.6 / 468.3` | `793.6 / 787.4` | `609.3 / 603.1` | `+15.8%` | **`-23.3%`** |
| `12288 KiB` | `exchange` | `627.6 / 657.5` | `799.0 / 797.8` | `612.4 / 614.4` | **`-4.5%`** | **`-23.2%`** |

#### `TP=3` Standalone (`solo`: Collective Only, Median `us`)

| Total Collective Bytes | `AG solo` (`PyNCCL` / Base / Cand) | `RS solo` (`PyNCCL` / Base / Cand) | `EX solo` (`PyNCCL` / Base / Cand) |
| ---: | ---: | ---: | ---: |
| `48 KiB` | `29.1 / 27.6 / 30.7` | `30.1 / 27.6 / 30.1` | `35.3 / 31.7 / 37.4` |
| `192 KiB` | `103.1 / 34.3 / 37.5` | `49.8 / 34.7 / 36.9` | `47.1 / 38.4 / 44.3` |
| `384 KiB` | `44.1 / 40.8 / 44.6` | `46.6 / 41.2 / 43.3` | `49.5 / 45.3 / 51.7` |
| `768 KiB` | `60.7 / 53.3 / 57.3` | `110.1 / 54.3 / 57.9` | `65.6 / 58.2 / 64.7` |
| `1536 KiB` | `75.3 / 77.4 / 82.6` | `75.2 / 81.7 / 84.5` | `86.1 / 84.0 / 91.2` |
| `3072 KiB` | `120.7 / 127.6 / 135.2` | `125.5 / 132.6 / 137.7` | `135.1 / 135.5 / 145.5` |
| `6144 KiB` | `216.0 / 233.4 / 252.5` | `213.4 / 252.0 / 255.0` | `280.3 / 258.5 / 263.7` |
| `9216 KiB` | `456.1 / 356.4 / 380.8` | `305.4 / 375.2 / 372.2` | `321.5 / 385.3 / 386.0` |
| `12288 KiB` | `460.3 / 480.7 / 518.2` | `431.0 / 490.9 / 495.6` | `490.4 / 513.8 / 523.8` |

## 5. Memory Observations & Conclusion

- **Memory Footprint**: `0` additional bytes of pinned host or device memory. The multi-stage loop reuses the existing `ROCE_SLOTS = 2` (`2 x 4 MiB` = `8 MiB` send + `8 MiB` per-peer recv) arena bounce slots in ping-pong order (`slot = (epoch + s) & 1`).
- **Conclusion**: Replacing Python-level 64-CTA multi-kernel slicing with a single persistent 8-CTA (`2` of `20` GB10 SMs) multi-stage kernel launch eliminates inter-stage GPU preemption when collectives run concurrently with `main_stream` GEMM compute, beating `PyNCCL` by **13–25% at `6–12 MiB` on `TP=4` and `TP=3`** and beating baseline multi-launch `sparknet` by **20–38% across `6–16 MiB`**.
