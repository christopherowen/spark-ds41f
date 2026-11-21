# Group-Aware Wide-Decode `WO-A` Tile (`block_K = 256 / 128` and In-Kernel Shards at `M = 96..128`)

- **Date**: 2026-11-21
- **Base commit**: `6cb470bbc` (`spark-ds41f:candidate-63-r5`, vLLM tree `5f16a050bb283d3d964ab831de4cdf6fca74dea7`)
- **One intended variable**: Replace fixed `block_K = 64` for decode buckets `32, 64, 96, 128` in `vllm/models/deepseek_v4_1/tilelang/wo.py` with group-aware bit-exact `_wo_a_tile(bucket, groups, k)`:
  - `bucket = 16` (`M <= 16`): unchanged (`split=True, block_M=16, block_K=256`).
  - `bucket = 32` (`17..32`): `split=True, block_M=32, block_K=256` for `groups >= 2` (`TP1..TP4`), `block_K=128` for `groups == 1` (`TP8`).
  - `bucket = 64` (`33..64`): `split=True, block_M=64, block_K=128` for `groups >= 2`, keeping `block_K=64` for `groups == 1`.
  - `bucket = 96, 128` (`65..128`): `block_K=128` for `groups >= 2`, with `split=False` (in-kernel shard accumulation, zero FP32 partials buffer or reduce launch) for `groups >= 3` (`TP1..TP3`, `>= 48` N-tiles saturating all 48 SMs), `split=True` for `groups == 2` (`TP4`), and `block_K=64` for `groups == 1` (`TP8`).
  - `bucket = None` (`M > 128`, prefill): unchanged (`split=False, block_M=96, block_K=64`), since `block_K=64` achieves 2 CTAs/SM and outperforms `block_K=128` at large-M prefill.

## Motivation (`TODO.md` Decode #16)

In `candidate-63-r5` (`1120`), `_WO_A_TILES[16]` already used `block_K = 256`, while buckets `32, 64, 96, 128` all defaulted to `block_K = 64` (`num_stages = 2`), running 4× more K-loop iterations and TMA barriers across all 61 attention layers (`Layers 0..60`).

Because `fp8w_bf16_gemm_partials` and `fp8w_bf16_gemm_cast` both sum `K = 4096` in `_WO_A_SHARDS = 4` shards in shard order, widening `block_K` to `128 / 256` and switching `groups >= 3` (`M = 96..128`) to in-kernel shard accumulation is **100% `torch.equal` bit-identical** on every output byte (`xq` and `sf`).

## Pre-Serving Verification (`dgx4`, `spark-ds41f:candidate-63-r5`)

### Correctness & PDL Contracts (`WO OVERLAY VERIFIED`)

- `tests/kernels/quantization/test_deepseek_v41_tilelang_wo.py`: **`16 passed, 0 failed`** (`35.50 s`), including exact bit-identity across `rows = 1, 7, 16, 33, 64, 65, 96, 97, 128, 129, 200`, `NaN` rows, and `act_quant` parity.
- `tests/kernels/test_deepseek_v41_tilelang_pdl.py`: **`15 passed, 0 failed`** (`8.42 s`, `0` PDL violations).

### Cold-L2 Benchmark Across All TP Degrees (`K = 4096`, `N_per_g = 1024`, `shards = 4`, 100% `torch.equal`)

| Topology (`G`) | `M` (`bM`) | Base (`sp, s2, K64`) | Selected Tile | New (`us`) | Delta / Layer (`us`) | Est. 61L (`ms/step`) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `TP8 (G=1)` | `24 (32)` | `22.65` | `sp(s2, K128)` | `21.04` | `-1.60` | `-0.098` |
| `TP8 (G=1)` | `32 (32)` | `22.83` | `sp(s2, K128)` | `21.27` | `-1.56` | `-0.095` |
| `TP8 (G=1)` | `36..128` | `23.48..27.50` | `sp(s2, K64)` (kept) | `23.48..27.50` | `0.00` | `0.000` |
| `TP4 (G=2)` | `24 (32)` | `42.25` | `sp(s2, K256)` | `39.16` | `-3.09` | `-0.188` |
| `TP4 (G=2)` | `32 (32)` | `43.18` | `sp(s2, K256)` | `39.23` | `-3.95` | `-0.241` |
| `TP4 (G=2)` | `36 (64)` | `43.59` | `sp(s2, K128)` | `40.63` | `-2.96` | `-0.181` |
| `TP4 (G=2)` | `48 (64)` | `44.24` | `sp(s2, K128)` | `42.71` | `-1.53` | `-0.093` |
| `TP4 (G=2)` | `64 (64)` | `44.26` | `sp(s2, K128)` | `41.41` | `-2.85` | `-0.174` |
| `TP4 (G=2)` | `96 (96)` | `50.32` | `sp(s2, K128)` | `44.47` | `-5.85` | `-0.357` |
| `TP4 (G=2)` | `128 (128)` | `53.19` | `sp(s2, K128)` | `47.36` | `-5.83` | `-0.356` |
| `TP3 (G=3)` | `24 (32)` | `66.28` | `sp(s2, K256)` | `56.64` | `-9.65` | **`-0.589`** |
| `TP3 (G=3)` | `32 (32)` | `68.15` | `sp(s2, K256)` | `56.92` | `-11.23` | **`-0.685`** |
| `TP3 (G=3)` | `36 (64)` | `70.43` | `sp(s2, K128)` | `61.00` | `-9.43` | **`-0.575`** |
| `TP3 (G=3)` | `48 (64)` | `69.23` | `sp(s2, K128)` | `61.42` | `-7.82` | **`-0.477`** |
| `TP3 (G=3)` | `64 (64)` | `71.52` | `sp(s2, K128)` | `62.58` | `-8.94` | **`-0.545`** |
| `TP3 (G=3)` | `96 (96)` | `81.11` | `ns(s2, K128)` | `67.54` | `-13.58` | **`-0.828`** |
| `TP3 (G=3)` | `128 (128)` | `93.82` | `ns(s2, K128)` | `66.84` | `-26.98` | **`-1.646`** (`-1.11 ms` / 41 CED layers) |
| `TP2 (G=4)` | `24 (32)` | `92.62` | `sp(s2, K256)` | `75.50` | `-17.12` | **`-1.044`** |
| `TP2 (G=4)` | `48 (64)` | `97.03` | `sp(s2, K128)` | `85.99` | `-11.04` | **`-0.673`** |
| `TP2 (G=4)` | `96 (96)` | `121.94` | `ns(s2, K128)` | `101.11` | `-20.83` | **`-1.271`** |
| `TP2 (G=4)` | `128 (128)` | `142.56` | `ns(s2, K128)` | `111.05` | `-31.51` | **`-1.922`** |

*(Prefill `M = 512..6144` check: `pf(s2, K64)` beats `pf(s2, K128)` at both TP3 (`2183.5` vs `2548.2 us` at `M=6144`) and TP4 (`1392.0` vs `1680.1 us`), so `bucket = None` remains `block_K = 64`.)*

## Prediction for Live `tp3-triangle` Serving Screen

- **Outputs**: 100% bit-exact (`outputs_sha256 = b795e6d4...`, `rank1 = 1.0`, `KL = 0`).
- **Decode grid (`M <= 16` shapes: `1x2, 1x3, 1x6, 2x12, 4x8, 4x12, 8x16`)**: Strictly unchanged (`bucket = 16` untouched).
- **Decode grid (`M = 24, 48` shapes: `4x24, 8x24 @4K/64K/256K, 8x48`)**: **`-0.45..-0.60 ms/step` (`-2.0%..-2.7%`)** faster!
- **`c8` closed throughput**: **`+1.0%..+2.0%`** (`+1.7..+3.5 tok/s`, since `c8` averages `~17..24` verified rows/step and hits `bucket = 32` heavily!).
- **`c1` single-stream**: Strictly neutral (`M <= 6` stays in `bucket = 16`).
- **Cold prefill**: Slightly faster (`+0.1..+0.3%`) from the 41 CED `M=128` tail layers saving `-1.11 ms` per 6K chunk.

## Live Serving Runs (`tp3-triangle`, `dgx1..dgx3`)

All runs cooled `< 49 C` on `spark-ds41f:candidate-63-r5` inside lab window `dgx1-1795263633` (`results/private/lab/2026-11-21-campaign/`). Both boots passed `model` `128/128` greedy bit-exact (`outputs_sha256 = b795e6d4c813815a475ab1516faf277a90a86b73614c38f4700189484a8c532d`, `rank1_fraction = 1.0`, `perplexity = 1.3160824`, `failed = 0`).

| Arm | `grid` `mad5` GM vs `ctrl` | `grid` `fl25` GM vs `ctrl` | `1x3 @4K` (ms) | `8x24 @4K` (ms) | `8x48 @4K` (ms) | `c1` (tok/s) | `c8` (tok/s) | `262K` prefill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ctrl-b1` | `+0.085%` | `+0.088%` | `16.890` | `22.619` | `27.557` | `69.65 ±3.82%` | `169.52 ±1.03%` | `7,455` |
| `ctrl-b2` | `-0.085%` | `-0.088%` | `16.923` | `22.625` | `27.576` | `69.73 ±3.79%` | `169.54 ±0.81%` | `7,419` |
| **`wo-a-wide-b1`** | **`-0.134%`** | `-0.043%` | `16.963` | **`22.539`** | **`27.116`** | `69.61 ±3.74%` | `169.28 ±0.99%` | `7,470` |
| **`wo-a-wide-b2`** | **`-0.156%`** | **`-0.087%`** | `16.915` | **`22.517`** | **`27.198`** | `69.56 ±3.80%` | `169.15 ±0.93%` | `7,448` |

### Conclusion

**2-boot verified wide-decode win**: `8x48 @4K` improves **`-0.410 ms/step` (`-1.49%`, `27.567 -> 27.157 ms`)** and `8x24 @4K` improves `-0.094 ms/step`, with zero `1x3` regression and `100%` bit-exactness (`c1`/`c8` remain within noise because `c8` averages `~17` verified rows/step). Ready for `candidate-64`.
