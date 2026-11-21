# Exact Full-Vocabulary Top-2 Reduction in Batched DSpark (`TODO.md` Candidate 61 Follow-up P2 #3)

Date: 2026-11-21. Base commit: `7a7b81982` (`spark-ds41f:candidate-63-r5` / `spark-ds41f-1120` vLLM tree `8fc99ca7546c5c538f0fe64e7c82954f1d2d7ae0`).

## Problem

The batched DSpark dense Markov head (`vllm/models/deepseek_v4_1/tilelang/vocab.py`, `finish_vocab_greedy_pairs(..., top2=True)`) and the materialized fallback (`vllm/v1/worker/gpu/sample/tilelang_sampling.py`, `greedy_markov_shard_pair(..., top2=True)`) previously discarded the winner's entire `2,048`-token partition (`GREEDY_PART = 2048` / `GREEDY_BLOCK = 2048`) when selecting each shard's runner-up for `draft_top2_margins` (`z + 2.0 * log1p(margin)` in `tilelang/confidence.py`).

### Partition-Collision Impact on DeepSeek-V4.1's Actual BPE Vocabulary (`128,000` tokens)

Analyzed on `models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e10.../tokenizer.json`:
- **ASCII punctuation & digits**: **`95/95` (`100.0%`)** live in Partition 0 (`[0..2047]`) — `100.0%` same-partition pair collision rate.
- **Common English prose words (55 tokens)**: **`48/55` (`87.3%`)** live in Partition 0 — `76.7%` same-partition pair collision rate.
- **Common Python/JS code tokens (53 tokens)**: **`32/53` (`60.4%`)** live in Partition 0 — `39.5%` same-partition pair collision rate.
- **Real source-code corpora**:
  - `vocab.py` (`26,472` tokens): **`59.6%`** in Partition 0, `P(same 2048-partition | indep) = 36.7%` (lower bound on correlated top-1/top-2 collisions), `P(same 32-tile | indep) = 2.63%`.
  - `scheduler.py` (`17,891` tokens): **`44.5%`** in Partition 0, `P(same 2048-partition | indep) = 21.8%`, `P(same 32-tile | indep) = 1.22%`.

Whenever both top-1 and top-2 fall inside Partition 0 (or any shared 2,048-token partition), discarding the winner's partition replaces the true runner-up with a distant token outside the partition, artificially inflating `margin = top1 - runner_up` and biasing `adaptive_verification` toward over-predicting draft survival on uncertain common-token branches at `c8` and `c16` (whereas `c1`'s `screened_markov` path already computed exact top-2).

## Design ("WAY 2" — Zero `partials` Overhead + Tensor-Core Single-Tile Exact Rescore)

- **Why not double `partial_vals` in `vocab_*_markov_greedy_partials`?** Measured on `dgx4`: tracking per-tile top-2 inside `partials` added register pressure and doubled global partial writes, regressing `M=48` by `+15..+20 us`.
- **WAY 2 (this patch)**:
  1. **Keep `vocab_packed_markov_greedy_partials` and `vocab_bf16_markov_greedy_partials` 100% untouched** (`+0.00 us` partials overhead, zero extra registers/SMEM/writes).
  2. Replace `finish_vocab_greedy_pairs(..., top2=True)` on Markov heads with **`finish_packed_markov_top2_exact`** and **`finish_bf16_markov_top2_exact`** (1 CTA per row, `threads=128`, `pdl.chain()`):
     - Phase 1 (`vllm_vocab_partials_top2_reduce`): 128-thread warp-shuffled reduction over all `num_tiles` partials yields exact `(best_v, best_i)` and best other-tile `(sec_v, sec_i)` plus `win_bx = best_i // rescore_BN` (`rescore_BN = max(block_N, 32)`).
     - Phase 2: Rescores **only the single winning `rescore_BN` tile** (`32` or `64` columns) using the **exact same `T.gemm` (`mma.sync.aligned.m16n8k16`)** and `vllm_vocab_apply_row_exceptions_clocal` as `partials`, guaranteeing **100% bit-exact (`torch.equal`)** full-vocabulary top-2 in **both `bfloat16` and `float32`** (note: an earlier scalar FMA rescore differed by 1 ULP in `float32` on `bf16 odd` same-tile rows; using `T.gemm` eliminates all ULP discrepancies completely).
     - Phase 3 (`vllm_vocab_tile_second_clocal`): Extracts the winning tile's best column `col != best_i` from row 0 of `C_local`, merges with Phase 1's other-tile runner-up under exact `vllm_vocab_first_argmax_keep` order (NaN > finite, lower index wins ties, `-inf` / `1 << 30` when `vocab < 2`), and writes `pairs[row, 0..3]`.
  3. Wire exact full-vocabulary top-2 into **`vllm/v1/worker/gpu/sample/tilelang_sampling.py`** (activating the existing `partial_argmax_kernel(..., top2=True)` + `_FINISH_LANES_TOP2 = (1, 2, 4)` + `greedy_markov_shard_pair(..., top2=True)`) and **`vllm/v1/worker/gpu/spec_decode/dspark/greedy.py`** (`scratch_shape(..., top2=True)` doubles scratch columns; `speculator.py:275` already passes `top2=self._enable_top2_branch`).

## Verification (`dgx4`, `spark-ds41f:candidate-63-r5`)

### Correctness & PDL (`96/96 passed, 0 failed`)

All 3 patched files mounted at their exact imported container paths (`vllm/models/deepseek_v4_1/tilelang/vocab.py`, `vllm/v1/worker/gpu/sample/tilelang_sampling.py`, `vllm/v1/worker/gpu/spec_decode/dspark/greedy.py`):
- **`test_exact_top2.py`**: **`20/20 passed`** (`packed` & `bf16`, `bfloat16` & `float32`, `M=1, 7, 16, 24, 48`, covering same-32/64-tile, same-2048-partition different-tile, cross-partition, exact same-tile ties, same-partition ties, NaN ties, all `-inf`, and non-zero random `x`, all `torch.equal` bit-identical to CPU full-vocab reference).
- **`tests/v1/spec_decode/test_dspark_vocab_parallel.py`**: **`17/17 passed`**.
- **`tests/v1/spec_decode/test_margin_confidence.py`**: **`8/8 passed`**.
- **`tests/v1/worker/test_dspark_greedy_markov.py`**: **`12/12 passed`** (`tp1/tp3/tp4`, `bf16/fp32`, `top2=False/True`).
- **`tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py`**: **`24/24 passed`** (`packed` & `bf16`, `odd/tp3/tp4`, `bf16/fp32`, `top2=False/True` bit-exact `torch.equal` cross-check between `project_head_markov_greedy_pairs` and `greedy_markov_shard_pair`).
- **`tests/kernels/test_deepseek_v41_tilelang_pdl.py`**: **`15/15 passed`** (including `test_layer_chain_replays_bit_equal[c1/c8/c8x6]` and `test_layer_chain_edges_are_programmatic`).

### True Rotating-Pool Cold-L2 Microbenchmark (`dgx4`, `>= 136 MB` weight pool + 250 ms steady clock burn)

| Shape | `M` | `top2=False` (µs) | `top2=True` exact (µs) | Delta (µs) |
|---|---:|---:|---:|---:|
| TP4 (`32320 x 256`, `136.5 MB`) | 1 | 54.73 | 60.87 | +6.15 |
| TP4 (`32320 x 256`, `136.5 MB`) | 8 | 60.28 | 59.81 | **−0.47** |
| TP4 (`32320 x 256`, `136.5 MB`) | 16 | 59.57 | 68.65 | +9.08 |
| TP4 (`32320 x 256`, `136.5 MB`) | 24 | 68.76 | 66.40 | **−2.36** |
| TP4 (`32320 x 256`, `136.5 MB`) | 48 | 82.52 | 80.52 | **−2.00** |
| TP3 (`43104 x 256`, `165.5 MB`) | 1 | 70.25 | 73.21 | +2.96 |
| TP3 (`43104 x 256`, `165.5 MB`) | 8 | 74.43 | 84.70 | +10.27 |
| TP3 (`43104 x 256`, `165.5 MB`) | 16 | 78.70 | 86.77 | +8.07 |
| TP3 (`43104 x 256`, `165.5 MB`) | 24 | 88.99 | 86.32 | **−2.67** |
| TP3 (`43104 x 256`, `165.5 MB`) | 48 | 100.50 | 102.48 | +1.99 |

Total overhead across all 5 draft steps is `-13 us` to `+51 us` (`<= 0.1%` of a `c8`/`c16` step; `+0.00 us` on `c1` which uses `screened_markov`).

## Live Serving Runs (`tp3-triangle`, `dgx1..dgx3`)

Tested inside lab window `dgx1-1795263633` on `spark-ds41f:candidate-63-r5` (`experiments/2026-11-21-dspark-exact-top2/tp3-dev.json`) against 2-boot `ctrl-b1/b2` (`candidate-63/tp3-dev.json`), `0` failed requests on all runs:

| Arm | `model` (`128` prompts) | `model` `acceptance_length` | Grid `mad5` / `fl25` vs `ctrl` | `c8` tok/s (`tpot_p50`, `tok/step`) | `c1` tok/s (`tpot_p50`) | Prefill `32K` / `262K` |
| --- | --- | ---: | --- | --- | --- | --- |
| `ctrl-b1/b2` mean | `128/128` (`b795e6d4...`, `r1=1.0`) | `2.3035` | `0.000% / 0.000%` | `169.53` (`46.96 ms`) | `69.69` (`14.44 ms`) | `7,259 / 7,437` |
| `top2-b1` | `128/128` (`b795e6d4...`, `r1=1.0`, `ppl=1.3160824`) | **`2.32004`** (`+0.72%`) | `+0.025% / +0.138%` | **`169.76`** (`46.33 ms`, `17.391`) | `69.72` (`14.40 ms`) | `7,257 / 7,455` |
| `top2-b2` | `128/128` (`b795e6d4...`, `r1=1.0`, `ppl=1.3160824`) | **`2.32004`** (`+0.72%`) | `+0.157% / +0.399%` | **`169.60`** (`46.92 ms`, `17.426`) | `69.52` (`14.48 ms`) | `7,195 / 7,389` |
| **2-boot mean** | **100% bit-exact** | **`2.32004` (`+0.72%`)** | **`+0.091% / +0.268%`** | **`169.68` (`46.63 ms`, `+0.15 tok/s`, `-0.33 ms` `tpot_p50`)** | **`69.62` (`14.44 ms`)** | **`7,226 / 7,422`** |

### Conclusion

- **2-boot live verified win & exactness fix:** Removing the Partition-0 margin-inflation bias raised `model` `acceptance_length` identically on both boots from **`2.3035 -> 2.32004` (`+0.72%`)**, improved steady-state `c8` on both boots (**`169.68` vs `169.53 tok/s`**, **`tpot_p50 = 46.63` vs `46.96 ms`**), and kept `c1` (`69.62` vs `69.69`), `grid` (`+0.09%` `mad5`), and `128/128` greedy outputs (`b795e6d4...`, `KL = 0`, `ppl = 1.3160824`) strictly bit-exact. Ready for **`candidate-64`**.
