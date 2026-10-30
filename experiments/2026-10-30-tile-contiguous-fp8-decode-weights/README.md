# Tile-Contiguous FP8 Weight Layout for Decode `Q-B` and Indexer `Q-B`

## Summary

In DeepSeek-V4.1-Flash on TP4 (`--linear-backend tilelang`), each of the 61 decoder layers projects its latent query state (`K = 1280`) through two large block-32 FP8 weights:
- Attention `Q-B`: `(N = 8192, K = 1280)` (`10 MB` per layer)
- Indexer `wq_b`: `(N = 4096, K = 1280)` (`5 MB` per layer)

In row-major `[N, K]` layout, each `block_N × block_K` tile (`block_N ∈ {64, 128}`, `block_K ∈ {128, 256}`) strides by `K = 1280` bytes between consecutive output rows, scattering every `block_N × 128` tile across `block_N` non-contiguous cache lines separated by `1280` bytes.

This patch packs `(8192, 1280)` and `(4096, 1280)` weights in-place after checkpoint loading via `pack_weight_tiles` into `[N // 128, K // 128, 128, 128]` viewed as `(N * (K // 128), 128)`:
- Every `128 × 128` prefill/decode tile is a single `16 KB` contiguous slab in DRAM (`T.copy(B[row0, 0], B_shared)`).
- Every `64 × 128` decode half-tile (`bx * 64 % 128 ∈ {0, 64}`) is a single `8 KB` contiguous slab (`row0 = ((bx * block_N // 128) * k_tiles + ko) * 128 + (bx * block_N % 128)`), and pairs of adjacent `64`-row CTAs stream sequentially through the same `16 KB` slab.
- Every output row accumulates over `K` in the exact same order (`ko = 0 .. k_blocks - 1`), maintaining 100% `torch.equal` bit-identity between decode tile heights (`16`, `32`, `64`), prefill (`128`), and row-major `[N, K]`.
- `layer.weight.data` is updated in-place as `packed.view(N, K)` so `layer.weight.shape` remains `(N, K)` with zero additional GPU memory footprint.

## Benchmark on `dgx4` (`spark-ds41f:1030`, GB10 SM121)

Measured with `padded_rows=True` (`Mp = 64` for `M <= 64`) and 64 MB L2 flush for cold measurements on `(8192, 1280)`:

| Rows (`M`) | Row-Major Warm (`µs`) | Tile-Contiguous Warm (`µs`) | Row-Major Cold (`µs`) | Tile-Contiguous Cold (`µs`) | Cold Delta (`µs`) |
|---:|---:|---:|---:|---:|---:|
| `1` | `8.87` | `9.27` | `65.88` | **`62.15`** | **`-3.73 µs`** |
| `6` | `8.95` | `9.44` | `63.23` | **`61.85`** | **`-1.37 µs`** |
| `16` | `9.34` | `9.96` | `65.53` | **`64.44`** | **`-1.09 µs`** |
| `24` | `11.17` | `11.53` | `65.03` | **`62.43`** | **`-2.60 µs`** |
| `32` | `11.27` | `11.91` | `68.46` | **`65.84`** | **`-2.61 µs`** |
| `48` | `17.10` | **`15.91`** (`-1.19`) | `70.32` | **`68.30`** | **`-2.02 µs`** |
| `64` | `17.71` | **`16.66`** (`-1.05`) | `72.27` | **`68.77`** | **`-3.50 µs`** |
| `128` | `34.30` | **`33.36`** (`-0.93`) | `80.62` | **`74.26`** | **`-6.36 µs`** |
| `256` | `49.61` | `51.90` | `100.91` | **`98.71`** | **`-2.21 µs`** |

## Verification on `dgx4` (`spark-ds41f:1030`)

- `pytest -q -p no:cacheprovider --noconftest -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py`: 100% `torch.equal` bit-identity across all decode tile heights (`16`, `32`, `64`) and prefill (`128`), and `0 PDL violations`.
