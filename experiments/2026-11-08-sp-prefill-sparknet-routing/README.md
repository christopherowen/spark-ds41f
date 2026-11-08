# Sequence-Parallel (`sp_prefill.py`) `sparknet` Collective Routing and Row-Exchange Overlap

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18b6ca6521b602f09cb3ba326774166d1c`
- **Base vLLM patch head**: `320021e3c4dbb202d911ab70e2cd924c75420488` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`, 641 patches)
- **Candidate vLLM patch head**: `a4c48599bdbbd931b64206a63673f7e7b0197e10` (tree `f7d46a4cc63f4f4ceb405c0b6f4824d75e2deaf1`, `patchset_sha256` `efe09d2ed1c60b3f00e3592d5b8da419d469aeab1711a1ebc3425aabd623c93c`, 642 patches)
- **Patch**: `vllm/0762-perf-dsv41-sp-prefill-sparknet-routing.patch`

## Hypothesis & Audit Findings

In `spark-ds41f:1107`, sequence-parallel (`SP`) prefill (`vllm/models/deepseek_v4_1/sp_prefill.py`) had two collective routing defects that either bypassed `sparknet` (`b12x_ar_comm`) when tensors fit within one-shot RoCE capacity or disabled side-stream overlap when `b12x_ar_comm` was present:

1. **Hardcoded `pynccl_comm` in `SPRows.gather_many` / `SPRows.gather_mxfp8` (`sp_prefill.py:219-289`)**:
   - Before each MoE layer in SP prefill, `SPRows.gather_mxfp8` gathers 4 `L`-row tensors (`L = ceil(T / N)`):
     - `values`: `[L, 5120]` `uint8` (`5,120` B/row)
     - `scales`: `[L, 160]` `uint8` (`160` B/row)
     - `topk_weights`: `[L, 6]` `float32` (`24` B/row)
     - `topk_ids`: `[L, 6]` `int32` (`24` B/row)
   - `SPRows.gather_many` hardcoded `getattr(get_tp_group().device_communicator, "pynccl_comm", None)` and never checked `b12x_ar_comm` (`SparknetOneShotAllReduce`).
   - Because `SparknetOneShotAllReduce.should_all_gather(inp, dim=0)` checks the per-rank shard size (`inp.nbytes <= max_gather_bytes`, which is `2 MiB` on `tp4-ring` and `4 MiB` on `tp4-switched` / `tp3-triangle`), `values` fits in one-shot RoCE up to **`T = 1,638` on `tp4-ring`** (`L = 409`, `2.00 MiB/rank`) and up to **`T = 3,276` on `tp4-switched` (`TP4`) / `T = 2,457` on `tp3-triangle` (`TP3`)** (`4.00 MiB/rank`).
   - Furthermore, packing the 3 small trailing tensors (`scales`, `topk_weights`, `topk_ids`, totaling `208` B/row or `<= 85 KiB/rank`) into a single `[L, 208]` `uint8` tensor reduces `roce_comm` collectives from 4 to **2** (one direct `_runtime.all_gather(..., out=...)` for `values` and one packed `all_gather` for the trailing tensors), eliminating two RDMA doorbell + completion round-trips per MoE layer.

2. **`_tp_pynccl_comm()` unconditionally returning `None` when `b12x_ar_comm` is present (`sp_prefill.py:310-392, 571, 645-662`)**:
   - In `1107`, `_tp_pynccl_comm()` returned `None` whenever `dev_comm.b12x_ar_comm` was present and enabled. As a result, `SPRows.project_exchange_rows` always took the `if comm is None:` fallback in production, never overlapping the row exchange with `then()` and never slicing large compute-bound projections (`T >= 6,144`).
   - Hardware profiling across `T = 160..6,144` shows why slicing at `ENGRAM_SLICE_TOKENS = 2048` regressed medium token counts when `project` is a weight-streaming GEMM (`wkv` `[6400, 6144]`): calling `project` per peer (`4` calls at `slices = 1`, `8` calls at `slices = 2`) streams the `wkv` weights 4–8 times for small-`M` slices (`+400..2,100 us` at `T = 160..2,048`).
   - At `T >= 6,144` (`slices = 2`, `768` rows/slice/owner), each slice GEMM is compute-bound and the `[6144, 6400]` BF16 exchange is `78.6 MiB`; here, 2-slice `project_exchange_rows` with `copy_self=False` on `_side_stream` overlapped with local-block projection and `then()` on the main stream beats `1107` by **`853–2,226 us` (`-9.0%` to `-21.6%`)** across all boots and both `ring4` and `direct` topologies.
   - Raising `ENGRAM_SLICE_TOKENS` from `2048` to `6144`, keeping `slices == 1` as a single full-row `project(0, tokens, local_kv)` call when `roce_comm` is active (routing the exchange on `_side_stream` overlapped with `then()` via `roce_comm.all_to_all_rows` when `should_all_to_all_rows` holds or `pynccl_comm.all_to_all_rows(..., copy_self=True)` otherwise), and restoring `_tp_pynccl_comm()` when `b12x_ar_comm` is present captures the `T >= 6,144` 2-slice overlap win without regressing `T = 160..4,096`.

3. **Two-argument `should_exchange(send, recv)` validation in `SPRows.exchange_into` (`sp_prefill.py:537-557`)**:
   - `SPRows.exchange_into` previously called `should_ex(send)` without `recv`, skipping `SparknetOneShotAllReduce.should_exchange(inp, out)` output shape/dtype/contiguity validation.

## Hardware Benchmark Methodology

All measurements were collected on 4x DGX Spark (`sp1..sp4`, GB10 SM121, 200G RoCE over `rocep1s0f0,roceP2p1s0f0`, `spark-ds41f:1107`) under `/tmp/gpu.lock` across both:
- **`ring4` (`tp4-ring`)**: `SPARKNET_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES=2097152` (`_slot_bytes = 2 MiB`, `max_gather_bytes = 2 MiB`, `ring4_relay = True`)
- **`direct` (`tp4-switched`)**: `SPARKNET_ROCE_TOPOLOGY=direct`, `SPARKNET_ROCE_ALLGATHER_MAX_BYTES=4194304` (`_slot_bytes = 4 MiB`, `max_gather_bytes = 4 MiB`, `ring4_relay = False`)

Two benchmark suites were executed across `T in {160, 256, 512, 1024, 2048, 4096, 6144}` (`30` timed iterations after `8` warmup iterations per point, recording `max_rank` CUDA-event wall time and verifying bit-exactness):
1. **Broad 5-case RoCE suite (`receipts/bench_sp_roce_{ring4,direct}_boot{1,2}.json`)**: 2 boots x 2 topologies covering `project_exchange_rows`, `wo.py` MXFP8 exchange (overlapped and standalone), `gather_mxfp8`, standalone `all_to_all_rows`, and overlapped BF16 `_project_exchange`.
2. **Focused candidate verification suite (`receipts/bench_sp_focused_{ring4,direct}_boot{1,2}.json`)**: 2 boots x 2 topologies directly comparing `1107` baseline against the exact candidate implementations for `gather_mxfp8` and `project_exchange_rows`.

## Results: Confirmed Wins (Included in Patch `0762`)

### 1. `SPRows.gather_many` / `gather_mxfp8` (`2`-Call Packed One-Shot RoCE)

In `bench_sp_roce` (4-call `sparknet_4calls` vs `1107` `PyNCCL` group) and `bench_sp_focused` (`g2_roce_2call`: direct `_runtime.all_gather` into `out[:padded_rows]` for `values` + 1 packed `uint8` `all_gather` for `(scales, topk_weights, topk_ids)`):

| Topology | Tokens `T` | `values` / Rank | `bench_sp_roce` Boot 1: `1107` -> `roce` (us) | `bench_sp_roce` Boot 2: `1107` -> `roce` (us) | `bench_sp_focused` Boot 1: `1107` -> `g1` -> `g2` (us) | `bench_sp_focused` Boot 2: `1107` -> `g1` -> `g2` (us) | Bit-Exact (`0` ULP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ring4` (`2 MiB` cap) | `160` | `200 KiB` | `632.70 -> 588.96` (`-6.9%`) | `579.42 -> 545.47` (`-5.9%`) | `911.04 -> 648.93 -> 531.30` (**`-41.7%`**) | `869.60 -> 770.98 -> 496.51` (**`-42.9%`**) | PASS |
| `ring4` (`2 MiB` cap) | `256` | `320 KiB` | `692.80 -> 428.00` (**`-38.2%`**) | `600.83 -> 455.04` (**`-24.3%`**) | `631.20 -> 657.70 -> 418.50` (**`-33.7%`**) | `722.30 -> 681.70 -> 807.97` (`p10: 404.9 -> 309.4 / 448.3`) | PASS |
| `ring4` (`2 MiB` cap) | `512` | `640 KiB` | `812.83 -> 639.97` (**`-21.3%`**) | `813.60 -> 641.12` (**`-21.2%`**) | `844.99 -> 671.04 -> 534.72` (**`-36.7%`**) | `645.34 -> 1035.52 -> 936.35` (`p90: 1438.0 -> 1295.9`) | PASS |
| `ring4` (`2 MiB` cap) | `1024` | `1.25 MiB` | `919.52 -> 545.54` (**`-40.7%`**) | `709.95 -> 564.67` (**`-20.5%`**) | `554.14 -> 445.02 -> 423.68` (**`-23.5%`**) | `1093.79 -> 1196.26 -> 972.70` (**`-11.1%`**) | PASS |
| `ring4` (`2 MiB` cap) | `2048` | `2.50 MiB` (PyNCCL fallback) | `784.06` (fallback) | `917.98` (fallback) | `834.66 -> 822.34 -> 902.66` (fallback) | `1436.16 -> 1510.94 -> 1197.34` (fallback) | PASS |
| `direct` (`4 MiB` cap) | `160` | `200 KiB` | `592.45 -> 900.00` | `535.97 -> 497.34` (`-7.2%`) | `540.90 -> 565.73 -> 596.54` (mean `658.0 -> 584.2`) | `867.04 -> 784.29 -> 893.09` (mean `905.7 -> 878.7`) | PASS |
| `direct` (`4 MiB` cap) | `256` | `320 KiB` | `647.55 -> 799.23` | `786.30 -> 816.86` | `619.58 -> 922.05 -> 912.74` | `973.41 -> 777.02 -> 836.93` (**`-14.0%`**) | PASS |
| `direct` (`4 MiB` cap) | `512` | `640 KiB` | `927.68 -> 624.80` (**`-32.6%`**) | `567.42 -> 557.12` (`-1.8%`) | `1298.43 -> 1048.74 -> 1066.40` (**`-17.9%`**) | `767.58 -> 835.23 -> 802.62` (`p90: 1384.6 -> 1137.9`) | PASS |
| `direct` (`4 MiB` cap) | `1024` | `1.25 MiB` | `851.26 -> 708.67` (**`-16.7%`**) | `739.46 -> 530.62` (**`-28.2%`**) | `1295.36 -> 1567.04 -> 1335.01` | `907.55 -> 859.68 -> 898.78` (`-1.0%`) | PASS |
| `direct` (`4 MiB` cap) | `2048` | `2.50 MiB` | `1097.15 -> 996.90` (**`-9.1%`**) | `998.94 -> 791.42` (**`-20.8%`**) | `1556.42 -> 1468.00 -> 1192.99` (**`-23.4%`**) | `1064.61 -> 1182.24 -> 1161.02` (`p90: 1604.5 -> 1518.3`) | PASS |
| `direct` (`4 MiB` cap) | `4096` | `5.00 MiB` (PyNCCL fallback) | `1309.18` (fallback) | `1248.13` (fallback) | `1609.54 -> 1748.22 -> 1624.99` (fallback) | `1424.64 -> 1140.22 -> 1007.33` (fallback) | PASS |

### 2. `SPRows.project_exchange_rows` (`slices = 1` Single-GEMM + Side-Stream Overlap up to `T < 6144`, `slices = 2` Overlapped Pipeline at `T >= 6144`)

| Topology | Tokens `T` | `bench_sp_roce` Boot 1: `1107` vs `2-slice` (us) | `bench_sp_roce` Boot 2: `1107` vs `2-slice` (us) | `bench_sp_focused` Boot 1: `p0_1107` vs `p1` vs `p3_sliced2` (us) | `bench_sp_focused` Boot 2: `p0_1107` vs `p1` vs `p3_sliced2` (us) | Policy Selected (`ENGRAM_SLICE_TOKENS = 6144`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `ring4` | `160` | `848.61` vs `1907.55` | `1135.68` vs `2201.66` | `1108.10` vs **`1114.82`** vs `3539.84` | `1081.79` -> **`980.74`** (`-9.3%`) vs `3692.06` | `slices = 1`: 1-GEMM + side-stream `roce_comm` overlapped with `then()` |
| `ring4` | `256` | `1148.90` vs `1983.17` | `1310.37` vs `2304.26` | `1061.89` vs **`1066.98`** vs `3332.45` | `1422.50` vs **`1454.14`** vs `3661.70` | `slices = 1`: 1-GEMM + side-stream `pynccl` overlapped with `then()` |
| `ring4` | `512` | `1346.75` vs `2297.89` | `1469.50` vs `2461.06` | `1458.08` vs **`1544.29`** vs `3772.61` | `1524.58` -> **`1462.02`** (`-4.1%`) vs `3678.66` | `slices = 1`: 1-GEMM + side-stream `pynccl` overlapped with `then()` |
| `ring4` | `1024` | `2100.29` vs `3056.35` | `2218.37` vs `3222.14` | `2171.30` vs **`2236.35`** vs `3939.10` | `2179.23` -> **`2102.66`** (`-3.5%`) vs `4219.52` | `slices = 1`: 1-GEMM + side-stream `pynccl` overlapped with `then()` |
| `ring4` | `2048` | `3334.78` vs `5115.23` (`+53.4%` if sliced!) | `3583.71` vs `5192.00` (`+44.9%` if sliced!) | `3597.44` -> **`3532.10`** (`-1.8%`) vs `5466.24` | `3882.56` vs **`4043.39`** vs `5718.43` | `slices = 1`: avoids `+1.6..1.9 ms` 2-slice weight-streaming penalty |
| `ring4` | `4096` | `6919.90` vs `7023.23` | `6850.08` vs `6950.78` | `6979.90` -> **`6761.70`** (`-3.1%`) vs `7184.96` | `7459.04` -> **`7093.60`** (`-4.9%`) vs `7351.78` | `slices = 1`: 1-GEMM + side-stream `pynccl` overlapped with `then()` |
| `ring4` | `6144` | `11559.17` -> **`9269.95`** (**`-19.8%`**, `-2.29 ms`) | `11022.56` -> **`9275.71`** (**`-15.8%`**, `-1.75 ms`) | `11269.63` -> `11722.18` -> **`9250.05`** (**`-17.9%`**, `-2.02 ms`) | `10931.30` -> `10917.22` -> **`9798.46`** (**`-10.4%`**, `-1.13 ms`) | `slices = 2`: 2-slice `copy_self=False` side-stream overlap |
| `direct` | `160` | `1020.54` vs `2419.30` | `947.84` vs `1820.93` | `1683.71` -> **`1650.27`** (`-2.0%`) vs `3674.14` | `1464.32` -> **`1451.30`** (`-0.9%`) vs `3380.83` | `slices = 1`: 1-GEMM + side-stream `roce_comm` overlapped with `then()` |
| `direct` | `2048` | `3365.38` vs `4538.56` (`+34.9%` if sliced!) | `2950.46` vs `4424.22` (`+50.0%` if sliced!) | `3698.24` vs **`4126.53`** vs `5733.79` | `3427.39` -> **`3378.50`** (`-1.4%`) vs `4788.70` | `slices = 1`: avoids `+1.2..2.0 ms` 2-slice weight-streaming penalty |
| `direct` | `4096` | `6232.19` vs `6439.30` | `6017.70` vs `6527.52` | `6593.25` vs **`6821.70`** vs `6981.98` | `6720.86` -> **`6431.84`** (`-4.3%`) vs `6489.60` | `slices = 1`: 1-GEMM + side-stream `pynccl` overlapped with `then()` |
| `direct` | `6144` | `9946.56` -> **`8458.69`** (**`-15.0%`**, `-1.49 ms`) | `10312.38` -> **`8172.00`** (**`-20.8%`**, `-2.14 ms`) | `9517.60` -> `10333.38` -> **`8664.16`** (**`-9.0%`**, `-0.85 ms`) | `10010.30` -> `9952.19` -> **`8609.60`** (**`-14.0%`**, `-1.40 ms`) | `slices = 2`: 2-slice `copy_self=False` side-stream overlap |

## Measured Negative Results / Dead Ends (Excluded from Patch `0762`)

1. **Multi-slice `sparknet` (`> _slot_bytes`) for `exchange_into` and `all_to_all_rows` when overlapped with a main-stream GEMM**:
   - `RoceOneshotAllReduce` supports multi-slice ping-pong pipelining for `exchange` and `all_to_all_rows` above `_slot_bytes`.
   - However, when `exchange_into` runs on `_side_stream` overlapped with a main-stream GEMM (`case5_project_exchange_5120`, `[T, 5120]` BF16), multi-slice `sparknet` is **`1.42x–1.63x` slower** than `1107`'s `PyNCCL` fallback across both boots and both topologies:
     - `ring4`: `T = 4096`: `2419.42 / 2444.83 us (1107) -> 3436.26 / 3596.83 us (sparknet_ms, +42.0% / +47.1%)`; `T = 6144`: `3639.26 / 3534.75 us (1107) -> 5663.97 / 5375.68 us (sparknet_ms, +55.6% / +52.1%)`.
     - `direct`: `T = 4096`: `2138.75 / 1774.53 us (1107) -> 3156.67 / 2780.26 us (sparknet_ms, +47.6% / +56.7%)`; `T = 6144`: `2574.62 / 2825.63 us (1107) -> 4194.14 / 4380.93 us (sparknet_ms, +62.9% / +55.0%)`.
   - **Root cause**: `oneshot_exchange` launches `64` CTAs (`blocks = 64`) that spin-wait in `roce_wait_peers` for every `2 MiB` slice. On GB10 (`48` SMs), a 64-CTA spin-waiting kernel on `_side_stream` across 3–20 sequential slices starves the concurrent main-stream GEMM of SMs, whereas `PyNCCL` uses `4–8` channel CTAs in a single launch. Single-slot `_slot_bytes` gating in `SparknetOneShotAllReduce.should_exchange` and `should_all_to_all_rows` is therefore preserved unchanged.

2. **`TileLangWOProjection.project_sp` (`wo.py`) grouped `PyNCCL` / packed `sparknet` MXFP8 exchange (`case2_wo_mxfp8_exchange`)**:
   - In `wo.py`, `project_sp` calls `sp_rows.exchange_into(xq_send, xq_recv)` (`[4, L, 2048]` `uint8`) and `sp_rows.exchange_into(sf_send, sf_recv)` (`[4, L, 16]` `uint32`) back-to-back on `_side_stream` while `WO-B` runs on `main_stream`.
   - At `T <= 1024`, both `xq_send` and `sf_send` already fit in `_slot_bytes` (`2 MiB`) and already route through `b12x_ar_comm.exchange_into` in `1107`. Packing `xq_send` and `sf_send` into a single `uint8` tensor (`ovlp_packed_sparknet`) adds HBM cat/copy overhead (`885.73 / 1007.71 us` vs `729.28 / 521.98 us` at `T = 1024` on `ring4`), and wrapping the two `exchange_into` calls in an outer `comm.group_start() / group_end()` (`ovlp_grouped_pynccl`) did not consistently beat `1107` at `T >= 2048` (`1439.04 / 1039.81 us` -> `1634.08 / 1180.22 us` at `T = 6144` on `direct`). `wo.py` is therefore left unchanged.

## Verification

- **2-boot GPU unit test receipts (`tests/models/test_deepseek_v41_sp_project_reduce_scatter.py` and `tests/models/test_deepseek_v4_1_sp_prefill.py`)**:
  - `receipts/gpu-pytest-boot1.txt`: `396 passed, 14 warnings in 102.71s`
  - `receipts/gpu-pytest-boot2.txt`: `396 passed, 14 warnings in 103.48s`
- **2-boot CPU unit test receipts (`tests/models/test_deepseek_v4_1_sp_prefill.py` and `tests/models/test_deepseek_v4_1_engram.py`)**:
  - `receipts/cpu-pytest-boot1.txt`: `219 passed, 17 skipped in 11.18s`
  - `receipts/cpu-pytest-boot2.txt`: `219 passed, 17 skipped in 10.93s`
- **Repository validation**:
  - `./bin/spark --cluster-config experiments/2026-11-08-sp-prefill-sparknet-routing/tp3.json doctor` -> `configuration OK`
  - `./bin/spark --cluster-config experiments/2026-11-08-sp-prefill-sparknet-routing/tp4.json doctor` -> `configuration OK`
  - `./bin/spark doctor` -> `configuration OK`
