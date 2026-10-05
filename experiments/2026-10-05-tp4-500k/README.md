# TP4 prefill at 500K and TP4 serving

The TP4 benchmarks of 2026-10-05 ([B12X and TileLang](../2026-10-05-tp4-validation/README.md))
measured prefill at 32K, 256K and 1M. TP4 is reported at 32K, 256K, 500K and
1M, beside TP3's 32K, 256K and 500K, so this adds 500K alone for each family.
Nothing else is re-measured.

The cluster was re-cabled to the four-node ring the same evening. Both
configurations equal their benchmarked ones
([b12x.json](../2026-10-05-tp4-validation/b12x.json),
[TileLang candidate](../2026-10-04-tilelang-1m/candidate.json)) except the
checkout and cache paths and the repository URL, which follow the rename to
spark-ds41f:

| Family | Config | Image | DSpark cost directory |
| --- | --- | --- | --- |
| TileLang | [tilelang.json](tilelang.json) | `-tilelang-1m-v3` | `ring4-tilelang-1m-v3-20261005` |
| B12X | [b12x.json](b12x.json) | r5p (`-tp4-1m-v1`) | `ring4-tp4-1m-20261005` |

Procedure: TileLang first, then B12X, one boot each. Each runs the quality
gate and real-source-text prefill at 500,000 tokens, two repeats
(`bin/spark bench --suites quality,prefill --prefill-text source
--prefill-sizes 500000 --prefill-repeats 2`). The B12X boot stays up as the
serving configuration.

## Results

Pending.
