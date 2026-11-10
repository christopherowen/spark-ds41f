# TP3 6-Channel Bidirectional Ring NCCL Profile Verification and Topology Audit

## Metadata

- **Date**: 2026-11-08
- **Base commit**: `cc20ed18b` (`spark-ds41f-1107`, image `spark-ds41f:1107` / `sha256:3f32a962ce0405ef8c2b3058bf1c8e2a3465d7873f331bd2517bdbfedab78428`)
- **Causal variable**: 3-rank NCCL collective and P2P channel/ring/buffer profile (`control_c8_1m` 8-channel `1 MiB` unidirectional vs. `cand_c6_8m_bidir` 6-channel `8 MiB` bidirectional Ring with `NCCL_P2P_NET_CHUNKSIZE=262144` vs. `cand_c6_8m_bidir_p2p512k` with `NCCL_P2P_NET_CHUNKSIZE=524288`) across the 3-node switchless triangle (`dgx1..dgx3`, 2 dual-root cables/node) and 3-node single-cable switched RoCE (`sp5, sp6, sp8`, 1 dual-root cable/node).
- **Workload identity**: DeepSeek-V4.1-Flash (`spark-ds41f`) TP3 (`world_size = 3`) sequence-parallel prefill collective suite (`engram_sp_ag_bf16`, `moe_sp_gather_mxfp8`, `attn_front_ag_u8`, `wo_local_exchange_mxfp8_a2a`, `wo_local_exchange_slice_a2a`, `mlp_all_to_all_bf16`, `moe_reduce_scatter_bf16`, `mhc_norm_all_reduce_bf16`, `rank_order_all_reduce_bf16`) and overlapped main-stream GEMM + side-stream NCCL collective wall time (`overlap_proj_rs_bf16`, `overlap_engram_proj_a2a_bf16`, `overlap_moe_ag_shared_gemm`) at `T in (16, 256, 1024, 6144)` (`L = ceil(T / 3) in (6, 86, 342, 2048)`).

---

## 1. Configuration & Parameter Mechanics Audit

### 1.1 Promoted State in `spark-ds41f` vs. Upstream `dgx-spark-networking`

1. **`spark-ds41f` (`config/cluster.json`, `releases/spark-ds41f-1107/cluster.json`, `experiments/2026-10-26-spark-ds41f-candidate/1107/tp3.json`)**:
   - Promoted in commit `7b2d6d456` (from `09ba30012` / `experiments/2026-11-06-nccl-sp-channels/`) on `tp3-triangle` and inherited by `tp3-triangle-4k`:
     - `NCCL_ALGO=Ring`
     - `NCCL_MIN_NCHANNELS=6`, `NCCL_MAX_NCHANNELS=6`
     - `NCCL_BUFFSIZE=8388608` (`8 MiB` Simple ring buffer -> `1 MiB` per step across `NCCL_STEPS=8`)
     - `NCCL_P2P_NET_CHUNKSIZE=262144` (`256 KiB` base parameter -> `512 KiB` effective `comm->p2pChunkSize` on GB10 SM121)
     - `NCCL_SWITCHLESS_BIDIRECTIONAL=1`
     - `NCCL_MIN_TRAFFIC_PER_CHANNEL=512`
     - `NCCL_THREAD_THRESHOLDS="-2 -2 -2 1 1 1"`
2. **Contract gap fixed in `spark-ds41f` (`scripts/topology.py` and `tests/test_switched_topology.py`)**:
   - When `7b2d6d456` added `"NCCL_P2P_NET_CHUNKSIZE"` to `SWITCHED_STRIP_RING_KEYS` in `scripts/recipe.py`, `scripts/topology.py` (`problems()` line 199 and `candidate()` line 322) and `tests/test_switched_topology.py` (`fixture()` line 23) were not updated to strip or reject `NCCL_P2P_NET_CHUNKSIZE` on `oneshot-switched` / `nccl-switched` profiles derived from `tp3-triangle`. This branch aligns `scripts/topology.py` and `tests/test_switched_topology.py` with `scripts/recipe.py`.
3. **Upstream `dgx-spark-networking` (`1fd6a1fbd8dc` in `upstreams.lock.json`)**:
   - `sparknet/nccl/profiles.py` (`PROFILES["tp3-triangle"]`), `recipes/tp3-triangle.json`, and `docs/nccl.md` still carry the pre-`1107` `_DIRECT_NCCL` defaults (`NCCL_MAX_NCHANNELS=8, NCCL_BUFFSIZE=1048576`). `spark-ds41f` overrides these in `config/cluster.json` (`shapes.tp3-triangle.environment`).

### 1.2 Why `NCCL_SWITCHLESS_BIDIRECTIONAL=1` (and Not `2`) Is Required for 6 Channels

- In `patches/nccl/0003-balanced-channel-allocation.patch` (`src/graph/connect.cc:balanceSwitchlessRings`) and `sparknet/nccl/profiles.py:250-251`:
  ```c
  if (mode < 1 || mode > 2 || (mode == 2 && channels != 2 && channels % 4 != 0)) {
    WARN("SWITCHLESS_BIDIRECTIONAL mode 2 needs two or a multiple of four channels");
    return ncclInvalidUsage;
  }
  ```
- `mode == 2` is designed for `4k` channels (`tp4-ring` with 4 channels), where ring reversal uses `c % 2 == 1` (`0,1,0,1`) so even-indexed (`NET/0-4`) and odd-indexed (`NET/0-5`) channels split evenly across directions when `channels % 4 == 0`.
- For `channels == 6`, `6 % 4 == 2 != 0`, so `mode == 2` is rejected. Under `mode == 1` (`NCCL_SWITCHLESS_BIDIRECTIONAL=1`), `c >= channels / 2` (`c in {3, 4, 5}`) is reversed:
  - `Ring 00..02 : 2 -> 0 -> 1` (clockwise; channels `0, 2` on `NET/IB/4`, channel `1` on `NET/IB/5`)
  - `Ring 03..05 : 1 -> 0 -> 2` (counter-clockwise; channels `3, 5` on `NET/IB/5`, channel `4` on `NET/IB/4`)
  Because `3` is odd, `c = 3` starts the second half on the opposite NIC root (`NET/IB/5`), so `mode == 1` with 6 channels automatically achieves **exact 3 CW / 3 CCW ring balance AND exact 3 / 3 PCIe root complex balance** (`3` channels on `NET/IB/4` and `3` channels on `NET/IB/5`).

### 1.3 Why `NCCL_P2P_NET_CHUNKSIZE=262144` Produces a `512 KiB` P2P Chunksize on GB10

- In `nccl/src/init.cc:833-834`, on ARM + SM100+ (`comm->cpuArch == NCCL_TOPO_CPU_ARCH_ARM && comm->minCompCap >= 100`, which includes GB10 SM121), NCCL multiplies `NCCL_P2P_NET_CHUNKSIZE` by `ratio = 2` and clamps to `buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS` (`8 MiB / 8 = 1 MiB`):
  - `control_c8_1m` (`NCCL_BUFFSIZE=1048576`, default `NCCL_P2P_NET_CHUNKSIZE=131072`): clamped by `1 MiB / 8 = 131072` (`128 KiB`).
  - `cand_c6_8m_bidir` (`NCCL_BUFFSIZE=8388608, NCCL_P2P_NET_CHUNKSIZE=262144`): `2 * 262144 = 524288` (`512 KiB`), verified in `nccl-init-boot*-cand_c6_8m_bidir-rank0.txt` (`P2P Chunksize set to 524288`).
  - `cand_c6_8m_bidir_p2p512k` (`NCCL_BUFFSIZE=8388608, NCCL_P2P_NET_CHUNKSIZE=524288`): `2 * 524288 = 1048576` (`1 MiB`), verified in `nccl-init-boot*-cand_c6_8m_bidir_p2p512k-rank0.txt` (`P2P Chunksize set to 1048576`).

---

## 2. Exact Commands

```bash
# 2-boot 3-rank RoCE NCCL benchmark on 3 nodes of sp5..sp8 (with flock /tmp/gpu.lock)
python3 experiments/2026-11-08-tp3-bidirectional-ring-nccl/screen_tp3_nccl.py \
  --out-dir experiments/2026-11-08-tp3-bidirectional-ring-nccl/receipts

# Summarize 2-boot receipts into results.json
python3 experiments/2026-11-08-tp3-bidirectional-ring-nccl/analyze_tp3_nccl.py

# Run CPU contract & receipt verification tests
pytest -q experiments/2026-11-08-tp3-bidirectional-ring-nccl/test_tp3_bidirectional_ring_nccl.py \
  tests/test_switched_topology.py tests/test_recipe.py tests/test_topology.py
```

---

## 3. Results: Switchless Triangle (`dgx1..dgx3`) vs. Single-Cable Switched (`sp5, sp6, sp8`)

### 3.1 Switchless 3-Node Triangle (`dgx1..dgx3`, 2 Dual-Root Cables/Node, `vNic 4` + `vNic 5`)

Recorded in `experiments/2026-11-06-nccl-sp-channels/receipts/screen-nccl-tp3-boot{1,2}.jsonl` across 2 boots (`boot1` forward order, `boot2` reverse order):

| Sequence Length `T` (`L = ceil(T/3)`) | Metric | `control_c8_1m` (2-boot mean) | `cand_c6_8m_bidir` (2-boot mean) | Delta (`%`) | Speedup (`x`) |
|---:|---|---:|---:|---:|---:|
| `T = 16` (`L = 6`) | Standalone SP collective bundle | `0.5566 ms` | `0.5061 ms` | `-9.07%` | `1.10x` |
| `T = 16` (`L = 6`) | Overlapped GEMM + collective bundle | `1.1277 ms` | `0.9985 ms` | `-11.46%` | `1.13x` |
| `T = 256` (`L = 86`) | Standalone SP collective bundle | `1.1131 ms` | `0.8588 ms` | `-22.85%` | `1.30x` |
| `T = 256` (`L = 86`) | Overlapped GEMM + collective bundle | `1.0695 ms` | `0.8920 ms` | `-16.60%` | `1.20x` |
| `T = 1024` (`L = 342`) | Standalone SP collective bundle | `3.1128 ms` | `2.3091 ms` | `-25.82%` | `1.35x` |
| `T = 1024` (`L = 342`) | Overlapped GEMM + collective bundle | `1.9693 ms` | `1.5649 ms` | `-20.54%` | `1.26x` |
| `T = 6144` (`L = 2048`) | Standalone SP collective bundle | `16.5973 ms` | `9.6606 ms` | `-41.79%` | `1.72x` |
| `T = 6144` (`L = 2048`) | Overlapped GEMM + collective bundle | `11.0954 ms` | `7.0587 ms` | `-36.38%` | `1.57x` |

**Physical mechanism on `tp3-triangle`**: Each node has two physical ConnectX-7 cables (`rocep1s0f0+roceP2p1s0f0` to peer A and `rocep1s0f1+roceP2p1s0f1` to peer B). Under `control_c8_1m` (`Ring 00..07 : 2 -> 0 -> 1`), every rank transmits only on cable A and receives only on cable B, leaving 50% of the physical wire duplex idle (`12.56 GB/s` busbw at `T=6144`). Under `cand_c6_8m_bidir`, `Ring 00..02` run clockwise and `Ring 03..05` run counter-clockwise across both merged vNICs, activating full duplex on both cables simultaneously (`21.74 GB/s` busbw at `T=6144`, `1.74x`) while using 6 instead of 8 GB10 SMs.

### 3.2 Single-Cable Switched 3-Node Subset (`sp5, sp6, sp8`, 1 Dual-Root Cable/Node, `rocep1s0f0` + `roceP2p1s0f0`)

Measured in `receipts/screen-tp3-nccl-boot{1,2}.jsonl` (`results.json`, 22 bit-identity checks across all collectives and overlapped GEMMs, `0` failures):

#### Bundle Totals (2-Boot Means)

| Sequence Length `T` (`L`) | Bundle | `control_c8_1m` (`128 KiB` P2P) | `cand_c6_8m_bidir` (`512 KiB` P2P) | `cand_c6_8m_bidir` vs. Control | `cand_c6_8m_bidir_p2p512k` (`1 MiB` P2P) | `p2p512k` vs. Control |
|---:|---|---:|---:|---:|---:|---:|
| `T = 16` (`L = 6`) | Standalone SP collective bundle | `0.7592 ms` | `0.8429 ms` | `+11.02%` (`0.90x`) | `0.7363 ms` | `-3.02%` (`1.03x`) |
| `T = 16` (`L = 6`) | Overlapped GEMM + collective bundle | `1.0697 ms` | `1.0763 ms` | `+0.62%` (`0.99x`) | `0.9965 ms` | `-6.84%` (`1.07x`) |
| `T = 256` (`L = 86`) | Standalone SP collective bundle | `1.0421 ms` | `0.9773 ms` | `-6.22%` (`1.07x`) | `1.0101 ms` | `-3.07%` (`1.03x`) |
| `T = 256` (`L = 86`) | Overlapped GEMM + collective bundle | `0.9415 ms` | `0.8601 ms` | `-8.65%` (`1.09x`) | `0.9634 ms` | `+2.33%` (`0.98x`) |
| `T = 1024` (`L = 342`) | Standalone SP collective bundle | `2.8059 ms` | `2.7908 ms` | `-0.54%` (`1.00x`) | `2.8763 ms` | `+2.51%` (`0.98x`) |
| `T = 1024` (`L = 342`) | Overlapped GEMM + collective bundle | `1.5427 ms` | `1.6535 ms` | `+7.18%` (`0.93x`) | `1.5810 ms` | `+2.48%` (`0.98x`) |
| `T = 6144` (`L = 2048`) | Standalone SP collective bundle | `14.6067 ms` | `16.2642 ms` | `+11.35%` (`0.90x`) | `16.7183 ms` | `+14.46%` (`0.87x`) |
| `T = 6144` (`L = 2048`) | Overlapped GEMM + collective bundle | `7.3974 ms` | `7.5137 ms` | `+1.57%` (`0.98x`) | `7.6648 ms` | `+3.61%` (`0.96x`) |

#### Per-Operation 2-Boot Breakdown on `sp5, sp6, sp8`

| `T` | Operation | Payload (B) | `control_c8_1m` (ms, boot1/boot2) | `cand_c6_8m_bidir` (ms, boot1/boot2) | `cand_c6_8m_bidir` Delta | `cand_c6_8m_bidir_p2p512k` (ms, boot1/boot2) | `p2p512k` Delta |
|---:|---|---:|---:|---:|---:|---:|---:|
| `16` | `engram_sp_ag_bf16` | `184,320` | `0.3415` (`0.2142`/`0.4689`) | `0.4494` (`0.4622`/`0.4366`) | `+31.60%` | `0.3463` (`0.2145`/`0.4781`) | `+1.39%` |
| `16` | `moe_sp_gather_mxfp8` | `96,768` | `0.0556` (`0.0552`/`0.0559`) | `0.0559` (`0.0566`/`0.0552`) | `+0.63%` | `0.0508` (`0.0510`/`0.0506`) | `-8.54%` |
| `16` | `attn_front_ag_u8` | `64,512` | `0.0472` (`0.0467`/`0.0477`) | `0.0488` (`0.0493`/`0.0483`) | `+3.39%` | `0.0453` (`0.0456`/`0.0450`) | `-4.03%` |
| `16` | `wo_local_exchange_mxfp8_a2a` | `7,128` | `0.0415` (`0.0416`/`0.0414`) | `0.0370` (`0.0369`/`0.0371`) | `-10.84%` | `0.0376` (`0.0376`/`0.0376`) | `-9.40%` |
| `16` | `wo_local_exchange_slice_a2a` | `92,160` | `0.0401` (`0.0395`/`0.0406`) | `0.0398` (`0.0403`/`0.0392`) | `-0.75%` | `0.0389` (`0.0386`/`0.0392`) | `-2.87%` |
| `16` | `mlp_all_to_all_bf16` | `184,320` | `0.0483` (`0.0479`/`0.0486`) | `0.0469` (`0.0474`/`0.0464`) | `-2.80%` | `0.0551` (`0.0642`/`0.0459`) | `+14.08%` |
| `16` | `moe_reduce_scatter_bf16` | `184,320` | `0.0707` (`0.0704`/`0.0710`) | `0.0598` (`0.0604`/`0.0592`) | `-15.43%` | `0.0590` (`0.0593`/`0.0588`) | `-16.49%` |
| `16` | `mhc_norm_all_reduce_bf16` | `163,840` | `0.1161` (`0.1154`/`0.1168`) | `0.0926` (`0.0933`/`0.0920`) | `-20.21%` | `0.0911` (`0.0906`/`0.0917`) | `-21.50%` |
| `16` | `rank_order_all_reduce_bf16` | `184,320` | `0.1143` (`0.1159`/`0.1126`) | `0.1053` (`0.1066`/`0.1039`) | `-7.88%` | `0.1033` (`0.1038`/`0.1029`) | `-9.54%` |
| `16` | `overlap_proj_rs_bf16` | `184,320` | `0.3450` (`0.2949`/`0.3951`) | `0.4037` (`0.4832`/`0.3242`) | `+17.01%` | `0.3016` (`0.2896`/`0.3136`) | `-12.58%` |
| `16` | `overlap_engram_proj_a2a_bf16` | `184,320` | `0.3465` (`0.3596`/`0.3335`) | `0.2863` (`0.2856`/`0.2870`) | `-17.39%` | `0.3028` (`0.2994`/`0.3063`) | `-12.61%` |
| `16` | `overlap_moe_ag_shared_gemm` | `96,768` | `0.3782` (`0.3123`/`0.4441`) | `0.3863` (`0.3130`/`0.4596`) | `+2.14%` | `0.3921` (`0.3068`/`0.4773`) | `+3.66%` |
| `256` | `engram_sp_ag_bf16` | `2,641,920` | `0.1667` (`0.1368`/`0.1966`) | `0.1217` (`0.1217`/`0.1217`) | `-26.99%` | `0.1275` (`0.1276`/`0.1273`) | `-23.55%` |
| `256` | `moe_sp_gather_mxfp8` | `1,387,008` | `0.0929` (`0.0918`/`0.0940`) | `0.0892` (`0.0898`/`0.0886`) | `-3.98%` | `0.0900` (`0.0899`/`0.0902`) | `-3.07%` |
| `256` | `attn_front_ag_u8` | `924,672` | `0.0779` (`0.0793`/`0.0765`) | `0.0764` (`0.0777`/`0.0750`) | `-1.99%` | `0.0762` (`0.0763`/`0.0760`) | `-2.25%` |
| `256` | `wo_local_exchange_mxfp8_a2a` | `102,168` | `0.0751` (`0.0760`/`0.0743`) | `0.0713` (`0.0749`/`0.0678`) | `-5.06%` | `0.0658` (`0.0663`/`0.0652`) | `-12.51%` |
| `256` | `wo_local_exchange_slice_a2a` | `1,320,960` | `0.0891` (`0.0876`/`0.0906`) | `0.0906` (`0.0911`/`0.0901`) | `+1.68%` | `0.1188` (`0.1190`/`0.1187`) | `+33.39%` |
| `256` | `mlp_all_to_all_bf16` | `2,641,920` | `0.1288` (`0.1292`/`0.1285`) | `0.1224` (`0.1226`/`0.1222`) | `-5.01%` | `0.1235` (`0.1234`/`0.1236`) | `-4.08%` |
| `256` | `moe_reduce_scatter_bf16` | `2,641,920` | `0.1271` (`0.1283`/`0.1260`) | `0.1224` (`0.1233`/`0.1214`) | `-3.78%` | `0.1240` (`0.1241`/`0.1239`) | `-2.48%` |
| `256` | `mhc_norm_all_reduce_bf16` | `2,621,440` | `0.2052` (`0.2061`/`0.2043`) | `0.2038` (`0.2036`/`0.2040`) | `-0.68%` | `0.2044` (`0.2047`/`0.2041`) | `-0.39%` |
| `256` | `rank_order_all_reduce_bf16` | `2,641,920` | `0.2845` (`0.2868`/`0.2823`) | `0.2833` (`0.2847`/`0.2818`) | `-0.46%` | `0.2843` (`0.2856`/`0.2829`) | `-0.11%` |
| `256` | `overlap_proj_rs_bf16` | `2,641,920` | `0.3332` (`0.3412`/`0.3252`) | `0.3240` (`0.3242`/`0.3239`) | `-2.75%` | `0.3861` (`0.4437`/`0.3284`) | `+15.86%` |
| `256` | `overlap_engram_proj_a2a_bf16` | `2,641,920` | `0.3356` (`0.3350`/`0.3361`) | `0.3284` (`0.3316`/`0.3251`) | `-2.15%` | `0.3829` (`0.4366`/`0.3291`) | `+14.08%` |
| `256` | `overlap_moe_ag_shared_gemm` | `1,387,008` | `0.2727` (`0.2382`/`0.3073`) | `0.2077` (`0.2385`/`0.1769`) | `-23.85%` | `0.1944` (`0.1684`/`0.2204`) | `-28.71%` |
| `1024` | `engram_sp_ag_bf16` | `10,506,240` | `0.3755` (`0.3856`/`0.3653`) | `0.3992` (`0.3958`/`0.4026`) | `+6.33%` | `0.3766` (`0.3772`/`0.3760`) | `+0.31%` |
| `1024` | `moe_sp_gather_mxfp8` | `5,515,776` | `0.2080` (`0.2072`/`0.2088`) | `0.2047` (`0.2028`/`0.2066`) | `-1.59%` | `0.2154` (`0.2148`/`0.2160`) | `+3.56%` |
| `1024` | `attn_front_ag_u8` | `3,677,184` | `0.2150` (`0.1543`/`0.2758`) | `0.1499` (`0.1494`/`0.1504`) | `-30.30%` | `0.1549` (`0.1541`/`0.1557`) | `-27.97%` |
| `1024` | `wo_local_exchange_mxfp8_a2a` | `406,296` | `0.1382` (`0.1380`/`0.1385`) | `0.1760` (`0.1846`/`0.1674`) | `+27.31%` | `0.1668` (`0.1674`/`0.1662`) | `+20.65%` |
| `1024` | `wo_local_exchange_slice_a2a` | `5,253,120` | `0.1975` (`0.1975`/`0.1976`) | `0.1933` (`0.1932`/`0.1935`) | `-2.13%` | `0.2190` (`0.2188`/`0.2192`) | `+10.86%` |
| `1024` | `mlp_all_to_all_bf16` | `10,506,240` | `0.3523` (`0.3526`/`0.3520`) | `0.3537` (`0.3538`/`0.3536`) | `+0.40%` | `0.3986` (`0.3976`/`0.3995`) | `+13.13%` |
| `1024` | `moe_reduce_scatter_bf16` | `10,506,240` | `0.3506` (`0.3502`/`0.3510`) | `0.3465` (`0.3462`/`0.3468`) | `-1.17%` | `0.3798` (`0.3792`/`0.3803`) | `+8.31%` |
| `1024` | `mhc_norm_all_reduce_bf16` | `10,485,760` | `0.6699` (`0.6694`/`0.6704`) | `0.6839` (`0.6828`/`0.6850`) | `+2.09%` | `0.7902` (`0.7891`/`0.7913`) | `+17.96%` |
| `1024` | `rank_order_all_reduce_bf16` | `10,506,240` | `0.9688` (`0.9693`/`0.9683`) | `0.9675` (`0.9681`/`0.9668`) | `-0.14%` | `0.9652` (`0.9648`/`0.9655`) | `-0.38%` |
| `1024` | `overlap_proj_rs_bf16` | `10,506,240` | `0.5687` (`0.5698`/`0.5677`) | `0.5545` (`0.5546`/`0.5544`) | `-2.50%` | `0.5893` (`0.5882`/`0.5903`) | `+3.60%` |
| `1024` | `overlap_engram_proj_a2a_bf16` | `10,506,240` | `0.5706` (`0.5714`/`0.5697`) | `0.6212` (`0.6844`/`0.5580`) | `+8.88%` | `0.5844` (`0.5882`/`0.5806`) | `+2.44%` |
| `1024` | `overlap_moe_ag_shared_gemm` | `5,515,776` | `0.4034` (`0.4028`/`0.4040`) | `0.4778` (`0.5499`/`0.4056`) | `+18.43%` | `0.4073` (`0.4063`/`0.4083`) | `+0.97%` |
| `6144` | `engram_sp_ag_bf16` | `62,914,560` | `1.9316` (`1.9222`/`1.9410`) | `2.1028` (`2.0942`/`2.1114`) | `+8.86%` | `2.2370` (`2.2370`/`2.2369`) | `+15.81%` |
| `6144` | `moe_sp_gather_mxfp8` | `33,030,144` | `1.0598` (`1.0604`/`1.0593`) | `1.0768` (`1.0748`/`1.0788`) | `+1.60%` | `1.0883` (`1.0887`/`1.0879`) | `+2.68%` |
| `6144` | `attn_front_ag_u8` | `22,020,096` | `0.7214` (`0.7212`/`0.7216`) | `0.7593` (`0.7592`/`0.7594`) | `+5.26%` | `0.7274` (`0.7277`/`0.7271`) | `+0.83%` |
| `6144` | `wo_local_exchange_mxfp8_a2a` | `2,433,024` | `0.1191` (`0.1187`/`0.1195`) | `0.1260` (`0.1262`/`0.1258`) | `+5.79%` | `0.1244` (`0.1242`/`0.1246`) | `+4.45%` |
| `6144` | `wo_local_exchange_slice_a2a` | `31,457,280` | `0.9546` (`0.9542`/`0.9549`) | `0.9665` (`0.9661`/`0.9668`) | `+1.25%` | `1.2015` (`1.2007`/`1.2022`) | `+25.86%` |
| `6144` | `mlp_all_to_all_bf16` | `62,914,560` | `1.8914` (`1.8916`/`1.8912`) | `2.1115` (`2.1116`/`2.1114`) | `+11.64%` | `2.1939` (`2.1937`/`2.1941`) | `+16.00%` |
| `6144` | `moe_reduce_scatter_bf16` | `62,914,560` | `1.9139` (`1.9136`/`1.9141`) | `2.0510` (`2.0505`/`2.0515`) | `+7.17%` | `2.0538` (`2.0537`/`2.0538`) | `+7.32%` |
| `6144` | `mhc_norm_all_reduce_bf16` | `62,914,560` | `4.1268` (`4.1271`/`4.1265`) | `4.2568` (`4.2562`/`4.2573`) | `+3.15%` | `4.2774` (`4.2768`/`4.2779`) | `+3.65%` |
| `6144` | `rank_order_all_reduce_bf16` | `62,914,560` | `6.0149` (`6.0146`/`6.0152`) | `7.0703` (`7.0694`/`7.0712`) | `+17.55%` | `7.0920` (`7.0923`/`7.0916`) | `+17.91%` |
| `6144` | `overlap_proj_rs_bf16` | `62,914,560` | `2.5913` (`2.5906`/`2.5920`) | `2.6783` (`2.6782`/`2.6784`) | `+3.36%` | `2.7174` (`2.7169`/`2.7178`) | `+4.87%` |
| `6144` | `overlap_engram_proj_a2a_bf16` | `62,914,560` | `2.7099` (`2.7088`/`2.7109`) | `2.7291` (`2.7285`/`2.7297`) | `+0.71%` | `2.8214` (`2.8210`/`2.8218`) | `+4.12%` |
| `6144` | `overlap_moe_ag_shared_gemm` | `33,030,144` | `2.0962` (`2.0956`/`2.0968`) | `2.1063` (`2.1060`/`2.1066`) | `+0.48%` | `2.1260` (`2.1257`/`2.1263`) | `+1.42%` |

---

## 4. Errors and Memory Observations

- **Errors**: `0` runtime errors, `0` timeouts, `0/22` bit-identity check failures across all collectives and overlapped GEMM outputs in both boots.
- **Memory observations**: `NCCL_BUFFSIZE=8388608` (`8 MiB`) across `6` collective channels (`6 * 8 MiB = 48 MiB` per peer ring buffer) vs. `NCCL_BUFFSIZE=1048576` (`1 MiB`) across `8` channels (`8 * 1 MiB = 8 MiB`) adds ~80 MiB of device buffer allocation per rank (`>= 6.03 GiB` cold-boot capture headroom on head, `>= 7.94 GiB` on workers in `1107` TP3 qualification).

## 5. Conclusion

1. **Keep the promoted 6-channel bidirectional `8 MiB` Ring profile (`tp3.json`) on `tp3-triangle` and `tp3-triangle-4k`**: On the 2-cable-per-node switchless triangle (`dgx1..dgx3`), reversing half the rings (`NCCL_SWITCHLESS_BIDIRECTIONAL=1` with 6 channels) unlocks the idle reverse direction of both peer cables, delivering `1.10x–1.72x` standalone collective speedup (`-9.1%` to `-41.8%`) and `1.13x–1.57x` overlapped GEMM + collective speedup (`-11.5%` to `-36.4%`) while saving 2 GB10 SMs.
2. **Keep `NCCL_P2P_NET_CHUNKSIZE=262144` (effective `512 KiB` on GB10) over `524288` (effective `1 MiB`)**: Raising `NCCL_P2P_NET_CHUNKSIZE` to `524288` degrades `wo_local_exchange_slice_a2a` (`+25.9%` at `T=6144`, `+33.4%` at `T=256`) and `T=6144` bundle latency (`16.72 ms` vs `16.26 ms`).
3. **Keep `tp3-switched` on `SWITCHED_DIRECT_ENV` and enforce `NCCL_P2P_NET_CHUNKSIZE` stripping in `scripts/topology.py`**: On 1-cable-per-node switched topologies (`sp5..sp8`), a unidirectional ring already saturates the single switch cable's full duplex (`18.74 GB/s` busbw at `T=6144`), and large P2P chunks / bidirectional rings contend on the shared switch port at `T=6144`. `scripts/topology.py` (`problems()` and `candidate()`) and `tests/test_switched_topology.py` now strip and reject `NCCL_P2P_NET_CHUNKSIZE` on switched profiles, matching `scripts/recipe.py`.
