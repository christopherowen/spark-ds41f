# 2026-11-16-indexer-prefill-exact

Can the dense indexer's score kernel get faster at long-prefill shapes without changing a bit of
its output? The long-context fall-off of cold prefill is almost all indexer: in candidate-60's
ring sweep (`experiments/2026-11-16-c60-ring-prefill-sweep`) the cost per prompt token rises
from 100.9 µs at 122K tokens to 110.3 µs at 910K. Layers 2, 8 and 14 each score every earlier
key for every prompt row, then select an exact top 512.

- **Base**: candidate-60, `spark-ds41f:candidate-60`.
  - Kernel benchmarks ran on sp8, whose image differs from the ring's only in `attention.py` and
    `sp_prefill.py`; `tilelang/indexer.py` is identical.
  - The baseline is that file, SHA-256 `3825602187077775…` (`bundles/*/indexer_baseline.py`).
- **Base commit**: `8b8c2c3f0`.
- **One variable**: the dense score kernel's code, `__tl_score_rows_dense_fast` in
  `tilelang/indexer.py`. Each variant must reproduce the baseline bit for bit: every score
  over every row's extent, and every selected position.
- **Constraint**: reference arithmetic stays as it is. That means MXFP4 queries and keys, the
  BF16 dot, `bf16(max(dot, 0) * w)`, the FP32 head sum in increasing head order, and the exact
  top 512.

## Bench

[bundles/bench/bench_prefill_indexer.py](bundles/bench/bench_prefill_indexer.py) runs through
`scripts/lab.py kernel-local` (GPU lock, `spark-ds41f:candidate-60`).

- **Shape:** one 256-row index sub-chunk of one request on a ratio-2 cache: 128-row pages, a 1M
  context's 524,288-position planned width, and one shared block-table row, as in prefill.
- **Extents:** each row's extent follows its causal position, at the depths of a 122K, 237K,
  470K and 910K-token prompt.
- **Timing:** CUDA graphs, with the baseline and the variant alternating on 4 random inputs.
  Medians and minimums are recorded.
- **Per-token share:** a sub-chunk's time × 3 layers × 16 sub-chunks ÷ 16,288 rows is its share
  of one prompt token at that depth.

```sh
scripts/lab.py kernel-local experiments/2026-11-16-indexer-prefill-exact/bundles/<variant> --out <dir>
```

Another agent's pytest containers ran on sp8 without the GPU lock during this session, and one
run overlapped them (`receipts/sp8/epi`). Later runs wait until `nvidia-smi` lists no compute
process, and record that the GPU had no other users before and after. Two runs whose variant
edits failed to apply measured the unchanged kernel against itself (`pipe-1`, `dbuf-1`). Those
A/A runs bound the noise at 0.6-1.2%.

## Runs (sp8, one GB10, µs per 256-row call)

The baseline alone (`receipts/sp8/baseline`):

| Context | Keys | Score | Select | Score TFLOPS | Score + select per token at that depth |
|---|---|---|---|---|---|
| 122K | 61,000 | 852 | 199 | 150 | 3.1 µs |
| 237K | 118,500 | 1,577 | 365 | 158 | 5.7 µs |
| 470K | 235,000 | 3,043 | 691 | 162 | 11.1 µs |
| 910K | 455,000 | 5,817 | 1,383 | 164 | 21.4 µs |

The per-token share grows linearly with depth. Averaged over a prompt, it predicts 9.2 µs per
token more at 910K than at 122K; the ring sweep measured 9.4. So the indexer is the long-context
cost, and scoring is 80% of it.

Variants, each against the baseline in the same run:

| Variant | Change | Bit-identical | Score at 237K | Score at 910K |
|---|---|---|---|---|
| `r8` | 8 rows per key tile instead of 4 | yes | 1,802 vs 1,576 (+14%) | 6,685 vs 5,819 (+15%) |
| `epi` (`epi-2`) | bf16x2 ReLU-and-multiply, head sum reads two heads per 32-bit load | yes | 1,357 vs 1,576 (−14%) | 4,947 vs 5,898 (−16%) |
| `pipe` (`pipe-2`) | `epi` plus the next key tile's global reads issued as soon as the current tile is in registers | yes | 1,246 vs 1,576 (−21%) | 4,575 vs 5,894 (−22%) |
| `dbuf` (`dbuf-2`) | `pipe` plus a double-buffered head-sum buffer, dropping one sync per row pair | yes | 1,247 vs 1,579 (−21%) | 4,571 vs 5,922 (−23%) |

- **`r8`:** reproduces candidate-59's finding that more rows per tile raise register pressure.
- **`dbuf`:** adds nothing over `pipe` and doubles the buffer, so it is not kept.
- **`pipe` is the candidate:** about 209 TFLOPS at depth, against 158-164 for the baseline.

### Why the changes are exact

- **`max(bf16(dot), 0) * w` in bf16x2:** `__hmax2` and the FP32 `fmaxf` agree on every input,
  NaN included (both return the non-NaN operand). The product of two BF16 values is exact in
  FP32, so the one rounding of `__hmul2` equals the FP32 path's rounding of the exact product.
  The one case left is a product in the BF16 subnormal range: about 1e-38, beyond any real dot
  times a head weight, and absent from the tests.
- **Head sum:** the swizzle mask is even, so heads 2k and 2k + 1 stay adjacent in shared memory.
  One 32-bit load per pair is added low then high: the same FP32 additions in the same order.
- **Prefetch:** it issues the same addresses with the same predicates, one tile ahead. A tile
  takes the prefetched words only on the fast path, whose rows share row 0's pages; any other
  path discards them.

### Checks of `pipe` (`receipts/sp8/checks-1`, `checks-2`)

- **Prefill depths 122K and 470K:** scores 672 vs 852 µs (−21%) and 2,481 vs 3,065 µs (−19%),
  bit-identical. Selection is unchanged.
- **Decode shapes** (`benchmark_deepseek_v41_indexer_bound.py`'s dense cases, against the
  baseline; 6 and 96 rows, 4K to 1M-key extents, per-row tables): bit-identical. The read-bound
  gate passes with none of 10 cases slower; most are 1-4% faster.
  - The script's candidate cases predate the candidate kernel's packed argument shape and stop
    with a shape error on both kernels, so this copy leaves them out. The candidate kernel is
    unchanged.
- **`tests/kernels/attention/test_deepseek_v41_tilelang.py -k "indexer or topk"`:** 28 passed,
  1 skipped, 1 xfailed.
  - The strict xfail is `test_indexer_nonfinite_query_scale`: the score's ReLU drops a NaN head
    where DeepSeek's `relu_` keeps it. `__hmax2` drops it as `fmaxf` does, so it still xfails.
- **`tests/kernels/test_deepseek_v41_tilelang_pdl.py`:** 15 passed.

Expected end to end, by the per-token arithmetic:
- scoring 22% faster saves about 1.9 µs per token on average over a 910K prompt (110.3 → 108.4,
  +1.8% at the 1M size);
- it saves about 0.5 µs at 237K (+0.5%).

## End to end (tp4-ring)

[tp4-dev.json](tp4-dev.json) is the pinned candidate-60 ring profile with
[overlay/vllm/models/deepseek_v4_1/tilelang/indexer.py](overlay/vllm/models/deepseek_v4_1/tilelang/indexer.py)
(`pipe`, SHA-256 `5a4e224ce53844c2…`) mounted over the image's file. [ring-tp4.json](ring-tp4.json)
runs two jobs:

1. **Sweep:** the candidate-60 ring sweep's exact bench, the same size list and therefore the same
   prompts, 2 boots.
2. **Gate:** the determinism gate (`determinism`, `greedy`, `determinism2`, `greedy2`), 2 boots.
   Its records are compared call for call with candidate-58's final ring gate
   (`experiments/2026-11-14-c58-prefill-10k/receipts/final/tp4-ring/determinism/c58-final-gate-tp4`),
   whose outputs candidate-59 and candidate-60 reproduced.

```sh
bin/spark candidate queue tp4-ring experiments/2026-11-16-indexer-prefill-exact/ring-tp4.json --apply
```

Runs: pending.

## Conclusion

Pending the end-to-end runs.

## Noted, not changed

- **Keeping NaN through the ReLU:** the bf16x2 form makes a vectorizable NaN-keeping ReLU
  possible (`max.NaN.bf16x2`). That is the form the strict xfail above waits for to match
  DeepSeek's `relu_`. It changes outputs for non-finite inputs, so it is a separate change.
- **Fusing selection into scoring:** at 910K, writing the 256 × 455K BF16 score matrix and
  reading it back for selection moves about 0.47 GB per call, roughly 1.9 ms of the 5.9 ms
  score-and-select. Selecting each 4,096-entry segment inside the score kernel would keep that
  traffic on chip, with the same segments and tie rule. It is a larger change, measured
  separately.
