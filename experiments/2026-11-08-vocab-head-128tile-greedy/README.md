# Vocabulary Head 128-Row Tile Pipeline & Greedy Epilogue (`TILE_CONFIGS[128]`)

- **Date**: 2026-11-08
- **Base commit**: `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` (`spark-ds41f-1107`)
- **vLLM base**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` + `experiments/2026-10-26-spark-ds41f-candidate/1107/vllm/series` (`0001`–`0761`, patch head `2756478b60b90700b9918b457aeb38657da637b5`, tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`)
- **Candidate vLLM patch**: `experiments/2026-11-08-vocab-head-128tile-greedy/vllm/0762-Match-128-row-vocab-head-tile-pipeline-stages.patch` (patch head `7bf1c97e32a8dab9a168c6689b8370fce80234a9`, tree `f1f1ba0f46b2fd7e7f9acfbe7887ee4802c16264`)
- **Node / GPU**: `sp8` (NVIDIA GB10, `sm_121a`, 48 SMs, 99 KiB / 101,376 B dynamic shared memory per CTA, 65,536 32-bit registers per SM)
- **Container image**: `spark-ds41f:1107`

## 1. Root-Cause Diagnosis

In `vllm/models/deepseek_v4_1/tilelang/vocab.py`, `TILE_CONFIGS` previously configured the `128`-row vocabulary head tile as:

```python
TILE_CONFIGS = {
    16: dict(block_N=32, threads=128, step_groups=2, num_stages=4),
    32: dict(block_N=16, threads=128, step_groups=2, num_stages=4),
    64: dict(block_N=32, threads=256, step_groups=1, num_stages=4),
    128: dict(block_N=32, threads=256, step_groups=1, num_stages=4, act_stages=2),
}
```

When `act_stages=2` is paired with `num_stages=4` (`step_groups=1`, `block_K=128`), `_packed_tile_body` emits:

```python
T.Pipelined(steps, order=[0, 1, 2, 3, 4], stage=[2, 0, 0, 3, 3])
```

Inspecting the generated CUDA C++ source and `nvcc -arch=sm_121a --ptxas-options=-v` compilation statistics across `block_M = 16, 32, 64, 128` revealed why `TILE_CONFIGS[128]` with `(num_stages=4, act_stages=2)` regressed at `M = 128`:

1. **Asymmetric stage offset (`act_stage=2` vs weight `cp.async` `stage=0`) unrolls 3 epilogue pipeline drain blocks with live `A_local` register arrays**:
   - For `block_M = 64` (`num_stages=4, act_stages=None`), `stage=[0, 0, 0, 1, 1]` has a 1-stage offset between `cp.async` loads and shared-to-register unpack/MMA, requiring only one drain block after the main `k` loop and keeping `bfloat16_t A_local[16]` to a single register array (**86 registers/thread** in `packed_vocab_projection` and **86 registers/thread** in `packed_vocab_greedy_partials`).
   - For `block_M = 128` with `num_stages=4, act_stages=2`, `stage=[2, 0, 0, 3, 3]` offsets activation staging (`stage=2`) from weight `cp.async` (`stage=0`) by 2 stages and unpack/MMA (`stage=3`) by 3 stages. TileLang unrolls **three full pipeline drain blocks** after the main loop (`tl::cp_async_wait<3>()`, `tl::cp_async_wait<2>()`, `tl::cp_async_wait<1>()`, `tl::cp_async_wait<0>()`) with **three simultaneously live register arrays** (`bfloat16_t A_local_1[32]`, `A_local_2[32]`, `A_local_3[32]`).
   - This inflated per-thread register allocation from **86 registers (`block_M=64`) to 195 registers in `packed_vocab_projection` and 197 registers in `packed_vocab_greedy_partials`** (`256 * 197 = 50,432` registers/CTA out of 65,536 per SM) while consuming `98,304 B` of dynamic shared memory.
   - As a result, at `M = 128`, the `128`-row kernel (`num_stages=4, act_stages=2`) was **slower than running two waves of the `64`-row kernel** on both TP4 (`1809.5` vs `1747.9 us` projection; `1795.4` vs `1753.5 us` greedy epilogue) and TP3 (`2384.9` vs `2378.9 us` projection; `2403.4` vs `2395.6 us` greedy epilogue).

2. **Causal fix (`TILE_CONFIGS[128] = dict(block_N=32, threads=256, step_groups=1, num_stages=2)`)**:
   - Matching activation and weight pipeline depth at `num_stages=2` (`act_stages=None`, `stage=[0, 0, 0, 1, 1]`) keeps the exact same 2 activation buffers (`2 * 128 * 128 * 2 B = 65,536 B`) and uses 2 matched weight buffers (`2 * 32 * 128 * 2 B = 16,384 B`).
   - Dynamic shared memory drops from **`98,304 B` -> `86,016 B`** (`78,336 B` for NVFP4).
   - Register usage drops by **-50%** from **195 -> 97 registers** in `packed_vocab_projection` and from **197 -> 100 registers** in `packed_vocab_greedy_partials` (`0 B` spill stores, `0 B` spill loads), with a single clean epilogue drain stage.

### `ptxas` Kernel Resource Comparison (`sm_121a`, `N=32320, K=5120, out_dtype=float32`)

| Kernel | `block_M` | Config | Dynamic Shared Memory | Registers / Thread | Registers / CTA | Spill Stores / Loads |
|---|---:|---|---:|---:|---:|---:|
| `packed_vocab_projection` | `16` | `s=4, sg=2, t=128` | `94,208 B` | `78` | `9,984` | `0 / 0 B` |
| `packed_vocab_projection` | `32` | `s=4, sg=2, t=128` | `98,304 B` | `72` | `9,216` | `0 / 0 B` |
| `packed_vocab_projection` | `64` | `s=4, sg=1, t=256` | `98,304 B` | `86` | `22,016` | `0 / 0 B` |
| `packed_vocab_projection` (base `1107`) | `128` | `s=4, a=2, sg=1, t=256` | `98,304 B` | **`195`** | **`49,920`** | `0 / 0 B` |
| `packed_vocab_projection` (**candidate**) | `128` | **`s=2, sg=1, t=256`** | **`86,016 B`** | **`97`** | **`24,832`** | `0 / 0 B` |
| `packed_vocab_greedy_partials` (base `1107`) | `128` | `s=4, a=2, sg=1, t=256` | `98,304 B` | **`197`** | **`50,432`** | `0 / 0 B` |
| `packed_vocab_greedy_partials` (**candidate**) | `128` | **`s=2, sg=1, t=256`** | **`86,016 B`** | **`100`** | **`25,600`** | `0 / 0 B` |

## 2. 2-Boot Captured CUDA Graph Benchmark (`sp8`, GB10)

Command run across 2 fresh container boots under `flock /tmp/gpu.lock` (`warmup=20, iters=200` per graph, `torch.equal` verified at every `M` against both `tile64` and `base_1107` and `greedy_shard_pair`):

```bash
flock /tmp/gpu.lock bash -c '
for boot in 1 2; do
  docker run --rm --gpus all --ipc=host --entrypoint python3 \
    -v /tmp/wt-vllm-1107/vllm/models/deepseek_v4_1/tilelang/vocab.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/vocab.py:ro \
    -v /tmp/tilelang_cache:/root/.tilelang/cache \
    -v /tmp:/tmp \
    -w /opt/spark-ds41f/candidate/vllm \
    spark-ds41f:1107 \
    /tmp/bench_vocab_2boot.py /tmp/bench_boot_${boot}.json
done
'
```

Raw per-boot receipts:
- `experiments/2026-11-08-vocab-head-128tile-greedy/receipts/bench-boot1.json`
- `experiments/2026-11-08-vocab-head-128tile-greedy/receipts/bench-boot2.json`

### TP4 Shard (`N = 32320, K = 5120, out_dtype = float32`) — 2-Boot Mean (`Boot 1 / Boot 2`, us)

| `M` | `row_tile` | `proj` `tile64` (us) | `proj` `base_1107` (us) | `proj` `candidate` (us) | `proj` $\Delta$ vs `1107` | `greedy_epilogue` `tile64` (us) | `greedy_epilogue` `base_1107` (us) | `greedy_epilogue` `candidate` (us) | `greedy` $\Delta$ vs `1107` | `proj + shard_pair` `candidate` (us) | `greedy` vs `proj + shard_pair` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `16` | `1197.0` (`1197/1197`) | `1071.6` (`1068/1075`) | `1064.5` (`1061/1068`) | `-0.66%` | `1196.3` (`1192/1201`) | `1072.6` (`1069/1076`) | `1090.2` (`1108/1072`) | `+1.64%` | `1067.8` (`1064/1072`) | `+22.3 us` |
| `6` | `16` | `1187.8` (`1192/1184`) | `1078.4` (`1077/1080`) | `1074.4` (`1071/1078`) | `-0.37%` | `1184.7` (`1183/1187`) | `1076.3` (`1073/1079`) | `1072.2` (`1063/1082`) | `-0.38%` | `1080.5` (`1075/1086`) | `-8.2 us` |
| `16` | `16` | `1233.8` (`1205/1263`) | `1095.5` (`1087/1104`) | `1086.4` (`1084/1088`) | `-0.83%` | `1198.6` (`1193/1204`) | `1075.7` (`1072/1080`) | `1071.8` (`1070/1074`) | `-0.36%` | `1094.3` (`1092/1096`) | `-22.5 us` |
| `32` | `32` | `1229.7` (`1227/1232`) | `1077.0` (`1072/1082`) | `1080.5` (`1079/1082`) | `+0.32%` | `1205.9` (`1199/1213`) | `1061.8` (`1055/1068`) | `1059.9` (`1055/1065`) | `-0.18%` | `1097.2` (`1096/1099`) | `-37.3 us` |
| `48` | `64` | `1220.6` (`1217/1224`) | `1215.1` (`1214/1216`) | `1216.2` (`1215/1217`) | `+0.09%` | `1191.4` (`1184/1199`) | `1189.0` (`1180/1198`) | `1188.3` (`1181/1195`) | `-0.06%` | `1255.8` (`1244/1268`) | `-67.6 us` |
| `64` | `64` | `1208.8` (`1206/1211`) | `1205.5` (`1203/1207`) | `1203.3` (`1199/1208`) | `-0.18%` | `1155.9` (`1150/1162`) | `1188.5` (`1216/1161`) | `1155.8` (`1148/1163`) | `-2.75%` | `1235.3` (`1230/1240`) | `-79.5 us` |
| **`72`** | **`128`** | `1474.5` (`1460/1489`) | `1415.0` (`1410/1420`) | **`1391.2`** (`1383/1399`) | **`-1.68%`** | `1520.1` (`1459/1582`) | `1389.8` (`1380/1399`) | **`1369.2`** (`1357/1381`) | **`-1.48%`** | `1426.0` (`1419/1433`) | **`-56.8 us`** |
| **`96`** | **`128`** | `1580.4` (`1556/1605`) | `1486.4` (`1473/1499`) | **`1435.3`** (`1422/1448`) | **`-3.44%`** | `1614.7` (`1619/1610`) | `1484.2` (`1474/1494`) | **`1416.2`** (`1403/1430`) | **`-4.58%`** | `1482.3` (`1470/1495`) | **`-66.1 us`** |
| **`104`** | **`128`** | `1607.7` (`1584/1631`) | `1535.2` (`1522/1548`) | **`1459.3`** (`1445/1474`) | **`-4.95%`** | `1609.6` (`1580/1639`) | `1532.6` (`1517/1548`) | **`1445.2`** (`1430/1460`) | **`-5.70%`** | `1528.2` (`1497/1560`) | **`-83.0 us`** |
| **`112`** | **`128`** | `1635.8` (`1610/1661`) | `1600.0` (`1585/1615`) | **`1494.4`** (`1480/1509`) | **`-6.60%`** | `1646.1` (`1631/1661`) | `1604.1` (`1592/1617`) | **`1484.5`** (`1471/1498`) | **`-7.46%`** | `1582.7` (`1599/1566`) | **`-98.2 us`** |
| **`128`** | **`128`** | `1747.9` (`1724/1772`) | `1809.5` (`1763/1856`) | **`1600.4`** (`1582/1619`) | **`-11.55%`** | `1753.5` (`1726/1781`) | `1795.4` (`1776/1815`) | **`1595.9`** (`1581/1611`) | **`-11.11%`** | `1666.8` (`1649/1684`) | **`-70.9 us`** |

### TP3 Shard (`N = 43136, org_N = 43072, K = 5120, out_dtype = float32`) — 2-Boot Mean (`Boot 1 / Boot 2`, us)

| `M` | `row_tile` | `proj` `tile64` (us) | `proj` `base_1107` (us) | `proj` `candidate` (us) | `proj` $\Delta$ vs `1107` | `greedy_epilogue` `tile64` (us) | `greedy_epilogue` `base_1107` (us) | `greedy_epilogue` `candidate` (us) | `greedy` $\Delta$ vs `1107` | `proj + shard_pair` `candidate` (us) | `greedy` vs `proj + shard_pair` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `16` | `1594.1` (`1577/1611`) | `1432.9` (`1427/1439`) | `1418.2` (`1417/1419`) | `-1.03%` | `1579.7` (`1580/1580`) | `1425.2` (`1425/1426`) | `1418.6` (`1418/1419`) | `-0.47%` | `1422.9` (`1424/1422`) | `-4.3 us` |
| `6` | `16` | `1556.5` (`1556/1557`) | `1436.1` (`1436/1437`) | `1443.0` (`1456/1430`) | `+0.48%` | `1569.0` (`1571/1567`) | `1428.8` (`1429/1428`) | `1431.9` (`1419/1445`) | `+0.21%` | `1437.6` (`1436/1439`) | `-5.7 us` |
| `16` | `16` | `1602.2` (`1588/1616`) | `1449.4` (`1451/1448`) | `1445.9` (`1449/1443`) | `-0.24%` | `1591.8` (`1592/1592`) | `1433.8` (`1433/1435`) | `1424.2` (`1425/1424`) | `-0.67%` | `1455.0` (`1456/1454`) | `-30.8 us` |
| `32` | `32` | `1611.0` (`1608/1614`) | `1456.6` (`1489/1425`) | `1442.0` (`1451/1433`) | `-1.00%` | `1594.4` (`1588/1601`) | `1446.6` (`1408/1486`) | `1422.1` (`1409/1435`) | `-1.69%` | `1450.6` (`1455/1446`) | `-28.5 us` |
| `48` | `64` | `1619.0` (`1612/1626`) | `1612.0` (`1607/1617`) | `1612.2` (`1606/1618`) | `+0.01%` | `1576.6` (`1567/1586`) | `1575.6` (`1566/1585`) | `1580.0` (`1576/1584`) | `+0.28%` | `1643.2` (`1638/1648`) | `-63.2 us` |
| `64` | `64` | `1619.5` (`1644/1595`) | `1589.1` (`1585/1593`) | `1589.2` (`1586/1593`) | `+0.00%` | `1520.8` (`1518/1523`) | `1519.5` (`1513/1526`) | `1520.3` (`1516/1525`) | `+0.05%` | `1667.2` (`1628/1706`) | `-146.8 us` |
| **`72`** | **`128`** | `1989.8` (`1970/2009`) | `1889.5` (`1884/1895`) | **`1849.6`** (`1835/1864`) | **`-2.11%`** | `2027.3` (`2041/2014`) | `1860.1` (`1856/1864`) | **`1819.4`** (`1812/1827`) | **`-2.19%`** | `1884.2` (`1880/1888`) | **`-64.8 us`** |
| **`96`** | **`128`** | `2193.2` (`2129/2257`) | `2002.0` (`1980/2024`) | **`1912.4`** (`1904/1921`) | **`-4.47%`** | `2154.3` (`2132/2177`) | `1984.9` (`1971/1999`) | **`1893.9`** (`1887/1901`) | **`-4.59%`** | `1974.9` (`1966/1984`) | **`-81.0 us`** |
| **`104`** | **`128`** | `2230.3` (`2253/2208`) | `2058.3` (`2047/2070`) | **`1948.4`** (`1936/1961`) | **`-5.34%`** | `2205.0` (`2169/2242`) | `2060.4` (`2044/2077`) | **`1932.0`** (`1922/1942`) | **`-6.23%`** | `2021.5` (`2005/2038`) | **`-89.6 us`** |
| **`112`** | **`128`** | `2226.6` (`2204/2250`) | `2150.8` (`2132/2169`) | **`2021.3`** (`2032/2011`) | **`-6.02%`** | `2257.9` (`2208/2307`) | `2165.4` (`2138/2193`) | **`1984.8`** (`1970/1999`) | **`-8.34%`** | `2088.2` (`2086/2091`) | **`-103.4 us`** |
| **`128`** | **`128`** | `2378.9` (`2359/2399`) | `2384.9` (`2371/2399`) | **`2138.9`** (`2122/2156`) | **`-10.32%`** | `2395.6` (`2394/2397`) | `2403.4` (`2386/2421`) | **`2184.0`** (`2118/2250`) | **`-9.13%`** | `2255.9` (`2267/2245`) | **`-71.9 us`** |

## 3. Other Investigated Variations (Recorded)

1. **`vec = 8` (`uint2`) and `vec = 4` (`uint32_t`) shared-memory unpacking in `_unpack_weight_stage`**:
   - With `block_N = 32, block_K = 128`, each weight stage holds `4,096` BF16 values. At `VEC = 16` (`uint4` 128-bit shared load), 128 of the 256 threads in `block_M = 64, 128` unpack 32 elements each.
   - Sweeping `vec = 8` (all 256 threads unpack 16 elements via `uint2`) and `vec = 4` on `sp8` was slightly slower at `M = 128` (`1618.2 us` vs `1561.4 us` greedy) because 128-bit `ld.shared.v4.u32` transactions from 4 warps already saturate shared-memory bank bandwidth while overlapping with warp-group MMA on the other warps.
2. **Adding `pdl.chain()` to `packed_vocab_projection` and `packed_vocab_greedy_partials`**:
   - Measured within $\pm 1.5\ \mu\text{s}$ noise (`1568.8` vs `1567.2 us` projection, `1559.0` vs `1561.4 us` greedy at `M = 128`); omitted to preserve single-variable change discipline.
3. **`num_stages = 3` (`act_stages = 2` or `act_stages = None`)**:
   - `num_stages = 3, act_stages = None` requires `122,880 B > 101,376 B` shared memory on `block_M = 128` (exceeds GB10 limit).
   - `num_stages = 3, act_stages = 2` (`90,112 B`) still has a 1-stage offset (`stage=[1, 0, 0, 2, 2]`), using 140 registers and running ~35–40 us slower than matched `num_stages = 2` (`1620.8` vs `1586.0 us` at `M = 128` TP4).

## 4. Verification & Unit Tests (2 Fresh Container Boots)

All 105 unit, PDL, and greedy verification tests pass across 2 fresh container boots on `sp8`:
- `tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py` (73 passed, including `test_tile_config_128_uses_matched_two_stage_pipeline`)
- `tests/kernels/test_deepseek_v41_tilelang_pdl.py` (14 passed)
- `tests/v1/worker/gpu/test_greedy_verification.py` (18 passed)

Receipts:
- `experiments/2026-11-08-vocab-head-128tile-greedy/receipts/pytest-boot1.txt`
- `experiments/2026-11-08-vocab-head-128tile-greedy/receipts/pytest-boot2.txt`
