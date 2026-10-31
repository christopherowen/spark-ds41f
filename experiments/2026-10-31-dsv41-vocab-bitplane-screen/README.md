# DSV4.1 TileLang Packed Vocab SWAR `__byte_perm` Vectorized 12-bit BF16 Decode + Exception Loop Guard

## Summary

At single-stream (`c1`) and speculative (`c1` MTP / DSpark) decode (`m in {1, 2, 6}`), `packed_vocab_greedy_partials` and `packed_vocab_projection` in `vllm/models/deepseek_v4_1/tilelang/vocab.py` are memory- and instruction-issue-bound on unpacking the 12-bit exponent-table BF16 vocabulary weights (`[N, K // 2]` `uint8` low bytes + `[N, K // 4]` `uint8` nibbles + sparse 4-bit exception runs) and feeding the warp-specialized `T.gemm(..., policy=FullCol)` loop over `K = 5120` (`80` `block_K=64` steps).

In `rc20`, each producer thread per `k` step decoded `2 x 16` BF16 weights (`W_l` and `W_h`) by executing 32 scalar iterations of `vllm_vocab_decode_bf16(low[s], (nib[s >> 1] >> ((s & 1) << 2)) & 0xF, BASE_MINUS_ONE)`:
- 32 scalar `uint8` byte loads for `low` and 16 scalar byte loads for `nib`,
- 32 per-element conditional checks `code == 0 ? 0 : code + BASE_MINUS_ONE` and bit shifts `((uint16_t)low & 0x80u) << 8 | exp << 7 | (low & 0x7Fu)` before writing to `W_s`,
- An unconditional exception scan header in every `(pid_n, k)` tile (`while (idx < end)`), even when a `(block_N=128, block_K=64)` tile has zero out-of-band exponent exceptions (`start == end`, the common case for >99% of vocabulary tiles),
- Fresh `torch.empty` allocations of `(partial_vals, partial_idxs, pairs)` on every `project_head_greedy_pairs` call outside graph capture.

This experiment optimizes `vllm/models/deepseek_v4_1/tilelang/vocab.py`:
1. **Branchless SWAR `__byte_perm` 16-Weight Vectorized Decode (`vllm_vocab_decode_vec16<BASE_MINUS_ONE>`):**
   - Replaces the 32 scalar `vllm_vocab_decode_bf16` calls per thread per `k` step with a single call to `vllm_vocab_decode_vec16<BASE_MINUS_ONE>` operating on `uint4` (`16 B`) of `low_a`, `uint4` (`16 B`) of `low_b`, and `uint4` (`16 B`) of `nib_ab` in registers.
   - Uses PTX `__byte_perm` (`0x4140u` / `0x4342u`) to unpack pairs of 4-bit nibbles into 16-bit halfword lanes (`ca01 = __byte_perm(n01, 0u, 0x4140u); cb01 = (ca01 >> 4) & 0x000F000Fu; ca01 &= 0x000F000Fu`).
   - Detects nonzero codes (`c in 1..15`) across two 16-bit lanes simultaneously without branches via `nza01 = ((ca01 + 0x000F000Fu) >> 4) & 0x00010001u`, and shifts the biased exponent into BF16 bits `7..14` via `expa01 = (ca01 + nza01 * BASE_SCALAR) << 7`.
   - Duplicates each `low` byte into both bytes of a 16-bit halfword via `__byte_perm(la_word, 0u, 0x1100u)` and `0x3322u`, which directly places bit 7 (sign) at bit 15 (and bit 31) and bits `0..6` (mantissa) at bits `0..6` (and bits `16..22`), so `(la01 & 0x807F807Fu) | expa01` constructs two exact BF16 values per `uint32_t` with zero shifts on `low`.
2. **Tile-Level Exception Loop Guard & Vectorized Fragment Rounding:**
   - Guards the sparse exception loop with `if start < end:` so tiles with zero exceptions skip the loop setup entirely.
   - Uses `T.Parallel` vectorized copy into `C_round` (`block_M x block_N` `bfloat16`) in `packed_vocab_greedy_partials` before the row-max reduction.
3. **Cached Greedy Scratch Buffers:**
   - Caches `(partial_vals, partial_idxs, pairs)` on `layer._tilelang_vocab_greedy_scratch` in `project_head_greedy_pairs` when not under CUDA graph capture.

## Verification & Benchmark (`dgx4`, `spark-ds41f:19f2c20ed4d6-next-20`)

### Correctness & PDL Verification
```bash
pytest -q --noconftest -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py
# 35 passed, 14 warnings in 96.44s (0:01:36)
# PDL sources: 69 kernels launched, 0 with PDL, 0 violations
```
Bit-identical outputs (`torch.equal` on `(partial_vals, partial_idxs)`, greedy `pairs`, and full BF16 logits) across zero-exception, single-exception, and dense multi-exception vocabularies.

### CUDA-Graph Replay Latency (`project_head_greedy_pairs`, `K = 5120`, 200 graph replays)

| Shard Shape | Batch `m` | `rc20` Baseline Median (`p10`) | SWAR `__byte_perm` Candidate Median (`p10`) | Delta (Median) |
| :--- | :---: | :---: | :---: | :---: |
| **TP4 (`N = 32320, K = 5120`)** | `m = 1` | `1031.168 µs` (`1026.784 µs`) | **`993.312 µs` (`988.384 µs`)** | **`-37.856 µs` (`-3.7%`)** |
| **TP4 (`N = 32320, K = 5120`)** | `m = 2` | `1027.328 µs` (`1024.000 µs`) | **`992.288 µs` (`989.152 µs`)** | **`-35.040 µs` (`-3.4%`)** |
| **TP4 (`N = 32320, K = 5120`)** | `m = 6` | `1036.512 µs` (`1028.096 µs`) | **`994.208 µs` (`990.240 µs`)** | **`-42.304 µs` (`-4.1%`)** |
| **TP3 (`N = 43096, K = 5120`)** | `m = 1` | `1378.560 µs` (`1367.328 µs`) | **`1313.856 µs` (`1306.848 µs`)** | **`-64.704 µs` (`-4.7%`)** |
| **TP3 (`N = 43096, K = 5120`)** | `m = 2` | `1372.192 µs` (`1365.568 µs`) | **`1311.008 µs` (`1306.848 µs`)** | **`-61.184 µs` (`-4.5%`)** |
| **TP3 (`N = 43096, K = 5120`)** | `m = 6` | `1371.200 µs` (`1366.624 µs`) | **`1313.824 µs` (`1308.896 µs`)** | **`-57.376 µs` (`-4.2%`)** |

With 2 vocabulary projections per speculative step (target `lm_head` + draft/MTP `lm_head`), this saves **`~76–80 µs/step` on TP4** and **`~122–129 µs/step` on TP3**.
