# DSV4.1 TileLang mHC Pre-Issued `Fn` / `R` / `X` Loads + Fused `RankOrderParts` in `post` & `post_collapse`

## Summary

In `vllm/models/deepseek_v4_1/tilelang/mhc.py`, each decoder sublayer boundary executes lagged mHC stream projection/finalization (`mhc.pre`, `mhc.post_pre`) or standalone stream post-update and collapse (`mhc.post`, `mhc.post_collapse`):
1. **Sequential `R` -> `X` -> `Fn` Load Dependency Chain in `mhc_project_streams` & `mhc_project_finalize`:**
   - In `rc20` (`mode == "post_pre"`), each CTA first loaded `rv` (`R`), then inside `for p in T.serial(world):` interleaved loading `xp[v]` (`X[p, ...]`) with accumulating `av[v] += T.cast(xp[v], "float32")` in FP32, then updated `rv` and wrote `R_out` / `R_s`, and only *after* completing the stream update and shared-memory write did it call `load_split_tf32(Fn, ...)` to load the `[MIXES, BLOCK_H]` FP32 projection weights and decompose them into three exact TF32 parts (`W_hi`, `W_mid`, `W_lo`).
   - Because `Fn`, `R`, and `X[0..world-1]` have zero address or data dependencies on one another, issuing all `rv` (`R`) and `xp[p, v]` (`X[0..world-1]`) global loads upfront before the `av[v]` reduction loop and hoisting `load_split_tf32(Fn, ...)` to the top of the `for n in T.unroll(MULT):` loop body allows the hardware warp scheduler to overlap `Fn`, `R`, and `X` global memory loads in flight simultaneously.
2. **Unfused `RankOrderParts.sum()` Launch Before `mhc.post` & `mhc.post_collapse`:**
   - While `mhc.pre` / `mhc.post_pre` already accepted `RankOrderParts.packed` (`[world, tokens, hidden]`) and reduced across `world` in rank order inside `mhc_project_streams` / `mhc_project_finalize`, both `TileKernelsMHC.post` and `TileKernelsMHC.post_collapse` still executed:
     ```python
     if isinstance(x, RankOrderParts):
         x = x.sum()
     ```
     which launched a standalone `collectives_rank_order_sum` kernel (`~3.5–4.5 µs`), materialized the reduced `[tokens, hidden]` BF16 tensor to global memory, and immediately read it back in `mhc_post` or `mhc_post_collapse`.
   - Extending `post_streams(hidden_size, threads=128, world=1)` and `post_collapse_streams(hidden_size, threads=128, world=1)` with `world > 1` specializations (`mhc_post_world` and `mhc_post_collapse_world`) fuses the rank-order FP32 sum + BF16 rounding directly into registers before the stream post-update (`fma_rn`), eliminating the extra kernel launch and global memory round-trip while preserving exact bit-for-bit outputs (`_assert_bits_equal`).

## Verification & Benchmark (`dgx4`, `spark-ds41f:19f2c20ed4d6-next-20`)

### Correctness & PDL Verification
```bash
pytest -q --noconftest -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/test_deepseek_v41_tilelang_mhc.py
# 118 passed in 66.92s (0:01:06)
# PDL sources: 37 kernels launched, 30 with PDL, 0 violations
```
Bit-identical outputs (`torch.equal` and `_assert_bits_equal` on BF16 bit patterns including signed zeros) across `world in (1, 2, 3, 4)` and `tokens in (1, 6, 9, 64, 96, 300, 2048)`.

### CUDA-Graph Replay Latency (`HIDDEN = 5120`, 500 graph replays)

| Operation & World Size | Batch `m` | `rc20` Baseline Median (`p10`) | Candidate Median (`p10`) | Delta (Median) |
| :--- | :---: | :---: | :---: | :---: |
| **`mhc.post_pre` (`world = 1`)** | `m = 1` | `16.864 µs` (`15.424 µs`) | **`13.536 µs` (`13.440 µs`)** | **`-3.328 µs` (`-19.7%`)** |
| **`mhc.post_pre` (`world = 3`, TP3 `RankOrderParts`)** | `m = 1` | `17.152 µs` (`15.968 µs`) | **`13.536 µs` (`13.440 µs`)** | **`-3.616 µs` (`-21.1%`)** |
| **`mhc.post_collapse` (`world = 3`, TP3 `RankOrderParts`)** | `m = 1` | `7.008 µs` (`6.528 µs`) | **`6.432 µs` (`6.048 µs`)** | **`-0.576 µs` (`-8.2%`)** |
| **`mhc.post_collapse` (`world = 4`, TP4 `RankOrderParts`)** | `m = 1` | `10.944 µs` (`10.496 µs`) | **`6.336 µs` (`6.112 µs`)** | **`-4.608 µs` (`-42.1%`)** |
| **`mhc.post_collapse` (`world = 3`, TP3 `RankOrderParts`)** | `m = 6` | `6.944 µs` (`6.400 µs`) | **`5.920 µs` (`3.232 µs`)** | **`-1.024 µs` (`-14.7%`)** |
| **`mhc.post_collapse` (`world = 4`, TP4 `RankOrderParts`)** | `m = 6` | `7.008 µs` (`6.752 µs`) | **`3.296 µs` (`3.200 µs`)** | **`-3.712 µs` (`-53.0%`)** |
| **`mhc.post_pre` (`world = 1`)** | `m = 96` | `38.112 µs` (`37.920 µs`) | **`34.016 µs` (`33.824 µs`)** | **`-4.096 µs` (`-10.7%`)** |
| **`mhc.post_pre` (`world = 3`, TP3 `RankOrderParts`)** | `m = 96` | `40.128 µs` (`38.496 µs`) | **`34.400 µs` (`34.016 µs`)** | **`-5.728 µs` (`-14.3%`)** |
| **`mhc.post_pre` (`world = 4`, TP4 `RankOrderParts`)** | `m = 96` | `40.320 µs` (`40.064 µs`) | **`36.032 µs` (`35.840 µs`)** | **`-4.288 µs` (`-10.6%`)** |
| **`mhc.post_collapse` (`world = 3`, TP3 `RankOrderParts`)** | `m = 96` | `9.088 µs` (`7.840 µs`) | **`7.424 µs` (`7.296 µs`)** | **`-1.664 µs` (`-18.3%`)** |
| **`mhc.post_collapse` (`world = 4`, TP4 `RankOrderParts`)** | `m = 96` | `9.312 µs` (`8.864 µs`) | **`7.424 µs` (`7.296 µs`)** | **`-1.888 µs` (`-20.3%`)** |
