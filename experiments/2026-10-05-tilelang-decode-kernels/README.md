# TileLang decode kernels against B12X

Goal: before the TileLang family becomes the default, match or beat B12X on
every kernel it owns in the TP4 decode step.

The 1M-recipe decode profiles (rank 0, median six-row step) left these behind
B12X per call: the attention Q-B (24.4 against 21.2 µs), fused Q-A/KV (20.3
against 18.2), indexer Q-B, the DSpark main projection (196 against 177) and
the sparse MLA decode kernel (24.4 against 19.7, though B12X adds a 6.2 µs
page-mapping kernel). Everything else the family owns was already ahead. The
sparse MLA module turned out level or better once its kernels were counted
together (1.07-1.11 against 1.22 ms per step); the gaps were the block-32 FP8
decode projections.

## Findings

**The 64-row decode tile.** Every decode row count up to 64 ran on one 64-row
tile, so one stream's six verified rows paid for 64 rows of tensor-core work.
Decode now runs on the shortest of 16-, 32- and 64-row tiles that holds the
rows, each swept per shape (vLLM 0043).

**A race in TileLang's warp-specialized pipeline.** The 16-row tiles failed
the repeated bit check: identical inputs gave different, wrong results (up to
1e-1 relative error) in up to every run of a 16 × 32 tile, and in 2-7% of runs
of 16 × 64 with 128 threads ([race census](bundles/dump/race2.py)). The
unspecialized pipeline (`TL_DISABLE_WARP_SPECIALIZED`) never failed in any
configuration and is as fast for decode; prefill keeps warp specialization,
which is up to 17% faster there (655 against 609 µs for Q-B at 4096 rows).
Adding `"memory"` clobbers to TileLang's mbarrier inline assembly did not fix
it ([diff](bundles/dump/barrier-memory-clobbers.diff)), so compiler hoisting is
not the cause. The 64-row tiles in the image showed no failure in 18,000 runs.
The DS4.1 expert GEMMs already disabled warp specialization.

**No K split.** Split-K decode routes need the prefill kernel to fold the same
shards. The in-kernel fold made the DSpark projection's 4096-row prefill 2.5
times slower (4,958 against 12,455 µs), so no shape splits K.

**Swizzled prefill for weights larger than L2.** The DSpark main projection
(37.5 MiB per rank at TP4) was re-read from DRAM by every row of tiles; panels
of eight N tiles made a 33 MB projection's 4096-row GEMM three times faster
(5,260 against 1,722 µs) with identical bits. Smaller weights run fastest
unswizzled.

**The serving shapes.** The serving trace showed the DSpark projection's K is
6,144 (not 5,120) and that the shared expert's projections (1152 × 5120,
5120 × 576) take the same path. The final sweep ([final2.py](bundles/dump/final2.py))
times whole calls (activation cast and GEMM) against B12X's `block_fp8_linear`
at all six TP4 shapes and three tile heights, with the weight warm in L2, cold,
and racing its own L2 prefetch, and keeps each tile's best worst-case ratio.
At six rows every projection wins in all three states (the DSpark projection
at parity, 0.997); every chosen tile is bit-identical to prefill over 120
rounds.

**Shared Triton kernels.** `_quantize_attention_inv_rope_to_tdg`, `_pages` and
`_chunk` run slower in TileLang profiles only because the L2 weight prefetch
stream overlaps them; without the prefetch they run at or below B12X's times
(1.31 against 1.38 µs for the attention quantizer).

## Serving windows (TP4 ring, one boot per arm)

| Window | Arm | Decode tiles | prose-c1 step | code-c1 step |
| --- | --- | --- | ---: | ---: |
| 1 | control | image (64-row) | 32.26 ms | 35.31 ms |
| 1 | B12X r5p | — | 33.48 ms | 38.02 ms |
| 2 | v2 | first 16/32/64 table | 31.66 ms | 34.35 ms |
| 3 | v3 | final table (vLLM 0043) | 32.11 ms | 35.31 ms |
| 4 | v3b | v3 with resident-CTA Q-B tiles | 31.53 ms | 34.52 ms |

All arms passed quality 5/5. Step times move with the DSpark cost profile each
arm measures for itself, so the per-kernel profile is the comparison. Rank-0
six-row step, B12X against v3: wall 44.94 against 39.26 ms, GPU busy 36.84
against 29.98 ms.

| Per call (µs) | B12X | v3 |
| --- | ---: | ---: |
| Fused Q-A/KV | 18.0 | 14.2 |
| Q-B | 20.8 | 21.9 |
| Indexer Q-B | 34.3 | 35.2 |
| DSpark main projection | 175.6 | 173.2 |
| Shared expert gate/up, down (side stream) | 52.7, 46.7 | 10.0, 19.1 |

Q-B and indexer Q-B still trail in serving by 5% and 3% although they win in
isolation. Q-B measured 21.8-21.9 µs with three different tiles (v2, v3, v3b),
so the tile is not the limit; B12X reads tile-packed weights, TileLang
row-strided ones. Per module, TileLang is ahead on routed experts (16.50
against 21.05 ms), collectives, mHC, sparse MLA, the router, quantization and
norms, the indexer, Q-A/KV and the DSpark projection.

## Files

- [make_configs.py](make_configs.py): the window arms; [run_window.sh](run_window.sh) runs them.
- [compare_profiles.py](compare_profiles.py): per-module six-row step from rank-0 traces.
- [bundles/sweep](bundles/sweep), [bundles/dump](bundles/dump): kernel-lab sweeps, race census and contention runs.
- [bundles/tests](bundles/tests): the vLLM TileLang linear tests with the decode-tile kernels.
- The vLLM changes are patches 0043 and 0044 in [../2026-10-05-tilelang-r6](../2026-10-05-tilelang-r6).
