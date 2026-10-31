# Fused BF16 Markov Greedy Reduction and `step_groups=2` Packed Vocab Tile (`TILE_CONFIGS[64]`)

## Summary

1. **Fused DSpark Markov-bias projection + base-logit addition + per-tile greedy reduction (`bf16_vocab_markov_greedy_partials`):**
   - Previously, each of the `n_spec = 5` sequential DSpark Markov steps in `DSparkSpeculator._sample_sequential` launched `markov_bias_local` (`bf16_gemm`) to materialize a full `[num_reqs, 32320]` BF16 `bias` tensor in global memory, followed by `greedy_markov_shard_pair` (`partial_argmax_kernel` + `finish_argmax_kernel`) to read `base_logits[:, i]` and `bias` back from memory and reduce them.
   - `bf16_vocab_markov_greedy_partials` (`vllm/models/deepseek_v4_1/tilelang/vocab.py`) fuses the BF16 Markov `W2` projection (`A @ B^T`), strided `base_logits[:, i]` addition with exact intermediate BF16 rounding, and per-tile `(max, shard_index)` reduction in shared memory, chained via `pdl.chain()` into `finish_vocab_greedy_pairs`.
   - Eliminates `5` full-shard `[num_reqs, 32320]` BF16 allocations/writes/reads and `5` kernel launches per decode step while keeping the `16.55 MB` Markov `W2` shard resident in L2 across the `5` sequential steps.

2. **`step_groups=2` for `TILE_CONFIGS[64]` and `pdl.chain()` on `finish_vocab_greedy_pairs`:**
   - Updates `TILE_CONFIGS[64]` in `vllm/models/deepseek_v4_1/tilelang/vocab.py` from `step_groups=1, num_stages=4` (`98,304 B` smem) to `step_groups=2, num_stages=3, act_stages=1` (`86,016 B` smem), doubling the contiguous byte/code read length per step to match `16`-row and `32`-row tiles.
   - Adds `pdl.chain()` to `finish_vocab_greedy_pairs` so the final reduction overlaps its launch with `packed_vocab_greedy_partials` and `bf16_vocab_markov_greedy_partials`.

## Microbenchmarks on `dgx4` (GB10, `N = 32,320`)

### 5-Step Sequential DSpark Markov Loop (`K = 256`, `5` steps)
| Requests (`M`) | Baseline (`bf16_gemm` + `greedy_markov_shard_pair`) | Fused (`bf16_vocab_markov_greedy_partials` + `finish`) | Delta | Speedup |
|---:|---:|---:|---:|---:|
| `1` | `289.7 us` (`57.9 us/step`) | `92.3 us` (`18.5 us/step`) | `-197.5 us` | `3.14x` |
| `2` | `179.0 us` (`35.8 us/step`) | `95.2 us` (`19.0 us/step`) | `-83.8 us` | `1.88x` |
| `4` | `167.5 us` (`33.5 us/step`) | `103.4 us` (`20.7 us/step`) | `-64.1 us` | `1.62x` |
| `8` | `153.8 us` (`30.8 us/step`) | `115.1 us` (`23.0 us/step`) | `-38.7 us` | `1.34x` |

### Packed Vocabulary Head at `block_M = 64` (`M = 48, K = 5,120`)
| Configuration | `packed_vocab_projection` | `packed_vocab_greedy_partials + finish` |
|---|---:|---:|
| Baseline (`step_groups=1, num_stages=4`) | `1188.4 us` | `1178.5 us` |
| Candidate (`step_groups=2, num_stages=3, act_stages=1`) | `1154.5 us` (`-33.9 us`) | `1133.7 us` (`-44.8 us`) |

## Verification on `dgx4`

- `pytest -o addopts="" --noconftest tests/v1/worker/test_prepare_kernels.py tests/v1/worker/test_dspark_greedy_markov.py tests/v1/worker/gpu/test_greedy_verification.py tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py`: **100 passed** (`100%` bit-exact across all shapes, strides, tie-breaking, and NaNs).
