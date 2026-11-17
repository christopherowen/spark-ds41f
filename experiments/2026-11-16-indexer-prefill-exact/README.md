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
- **Variables, one per patch**, each measured against the one before it:
  - 0817: the dense score kernel's code (`__tl_score_rows_dense_fast`).
  - 0818: the score ReLU's NaN handling.
  - 0819: the top-k segment kernel's key load.

  On finite inputs each must reproduce its baseline bit for bit: every score over every row's
  extent, and every selected position.
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

### End-to-end runs

Queue entry `20261116T204637-ring-tp4` ran on dgx1..dgx4 from 20:46Z on 2026-11-16 at
deployment commit `724ef3fbe`, from `spark-ds41f:candidate-60` with the overlay.
`overlays-exact-ring-tp4.txt` records the overlay's SHA-256, and `docker inspect` on dgx1 and
dgx3 showed the mount.

**Sweep** (receipts: [receipts/tp4-ring/bench/](receipts/tp4-ring/bench/)). The prompts are the
same token for token as candidate-60's ring sweep.

| Size label | Tokens | candidate-60 b1 / b2 | 0817 b1 / b2 | Mean change |
|---|---|---|---|---|
| 131,072 | 121,796 | 9,901.5 / 9,927.9 | 9,974.4 / 9,960.6 | +0.53% |
| 262,144 | 236,689 | 9,803.7 / 9,813.5 | 9,888.9 / 9,965.8 | +1.21% |
| 524,288 | 469,970 | 9,553.1 / 9,583.3 | 9,657.8 / 9,677.7 | +1.04% |
| 1,000,000 | 910,469 | 9,055.3 / 9,071.9 | 9,261.7 / 9,240.3 | +2.07% |

- **Above 32K:** the two boots of each build agree within 0.8%.
- **Below 32K:** time to first token dominates and the boots differ by up to 3.6%. Those sizes
  show no change outside that spread: +2.8% at 512, −1.4% at 2K, +0.7% at 32K.
- **Prediction:** +1.8% at 1M; measured +2.1%.

**Gate** (receipts: [receipts/tp4-ring/determinism/exact-ring-tp4/](receipts/tp4-ring/determinism/exact-ring-tp4/),
table: [receipts/tp4-ring/lab/exact-ring-tp4-table.txt](receipts/tp4-ring/lab/exact-ring-tp4-table.txt)).
- **Against candidate-58:** each boot makes 81 calls in each of the four checks. Every one of
  the 648 calls, over both boots, is identical (tokens and log-probabilities) to a call with the
  same prompt in candidate-58's final ring records.
- **Within each boot:** the determinism check reports `identical: true`, with 0 token and 0
  log-probability differences.
- **Distinct-prompt decode:** 271.83 / 273.62 tok/s at 16 streams, 202.66 / 202.80 at 8 and
  156.59 / 157.39 at 4. Candidate-58's ring gate recorded 273.89, 203.39 and 157.98. The
  indexer change targets prefill, and decode is unchanged within noise.

## NaN-keeping ReLU (0818)

DeepSeek's indexer takes `relu_` of each head's dot product, which keeps a NaN. Our kernels'
`max(dot, 0)` drops it, which is the strict xfail `test_indexer_nonfinite_query_scale` from
`experiments/2026-10-14-kernel-contracts`. That experiment fixed the shared and routed SwiGLU
clamps and the Engram gate; it left the indexer out because the NaN-keeping forms TileLang
generated cost 3.4-4.6% of prefill. With the bf16x2 epilogue, the ReLU is one `__hmax2_nan`
(`max.NaN.bf16x2`).

- **Kernels:** both epilogues of the dense kernel use it. The reindex candidate kernel takes the
  same bf16x2 epilogue with it, in place of its float round trip.
- **Finite inputs, against 0817** (`receipts/sp8/nan-1`, `nan-candidates-1`): scores and
  selections are bit-identical at 237K and 910K, and the dense decode shapes are bit-identical
  with none of 10 slower. The candidate kernel at decode shapes is bit-identical: 11.6 vs 12.1 µs
  at 6 rows, 294 vs 299 µs at 96.
- **Prefill score time:** 1,248 vs 1,248 µs at 237K and 4,525 vs 4,556 µs at 910K. No cost.
- **Non-finite inputs:** `test_indexer_nonfinite_query_scale` passes with its xfail removed.
  `test_indexer_candidates_nonfinite_query_scale` is new and checks the same contract on the
  candidate kernel: NaN at every live candidate, `-inf` in a masked block, other rows unchanged.
  `-k "indexer or topk"`: 29 passed and 1 skipped; the new test run on its own: 1 passed.

## Faster selection (0819)

The segment kernel turns each 16,384-entry segment's BF16 scores into order-preserving `uint16`
keys in a swizzled shared buffer, one 2-byte load per thread per step. `__tl_load_keys16` reads
eight scores per 16-byte load and writes the same keys to the same slots. A misaligned row or
the tail falls back to scalar loads, and gathered or grouped selections keep the scalar loop.

Measured against 0817 (`receipts/sp8/selvec-2`; `selvec-1` failed to import because a header was
inserted below a decorator, and is kept):

| Context | Select | Score + select |
|---|---|---|
| 122K | 184 vs 197 µs (−7%) | −1.7% |
| 237K | 330 vs 379 µs (−13%) | −2.0% |
| 470K | 637 vs 720 µs (−11%) | −0.9% |
| 910K | 1,269 vs 1,283 µs (−1%; −7% by minimum) | −0.5% |

- **Exactness:** selections and scores are bit-identical, and the top-k route, extent and
  compaction tests pass.
- **Why the gain is small:** selection is bound by reading the score matrix from DRAM (233 MB
  per call at 910K), not by load instructions.
- **Predicted serving share:** under 0.1% of cold prefill. It is recorded as a micro gain.

## All three together (`receipts/sp8/final-1`)

Against candidate-60's kernel, on finite inputs:

| Context | Score + select |
|---|---|
| 122K | 866 vs 1,040 µs (−17%) |
| 237K | 1,620 vs 1,958 µs (−17%) |
| 470K | 3,114 vs 3,800 µs (−18%) |
| 910K | 5,977 vs 7,341 µs (−19%) |

- **Bits:** bit-identical at every depth. The dense decode shapes are bit-identical with none of
  10 slower, and the candidate kernel is bit-identical.
- **Tests:** `-k "indexer or topk"` 29 passed and 1 skipped, the candidate NaN test passed, and
  the PDL tests passed (15).

## Patches

[vllm/series](vllm/series) is candidate-60's series (689 patches, head `f27a89d6a944`) followed by:

| Patch | Change |
|---|---|
| [0817](vllm/0817-perf-dsv41-bf16x2-epilogue-and-next-tile-key-prefetc.patch) | bf16x2 epilogue, paired head-sum loads and next-tile key prefetch in the dense score kernel |
| [0818](vllm/0818-fix-dsv41-keep-a-NaN-through-the-indexer-score-ReLU-.patch) | NaN-keeping score ReLU in the dense and candidate kernels; the non-finite test un-xfailed, and a candidate-kernel contract test |
| [0819](vllm/0819-perf-dsv41-load-dense-top-k-segment-keys-eight-BF16-.patch) | dense top-k segment keys loaded eight scores per 16-byte read |

- **Lock:** `bin/spark --cluster-config experiments/2026-11-16-indexer-prefill-exact/tp4.json
  build prepare --only vllm --lock` (sp8) applies the series and recorded patch head
  `3eb4081e4457`, tree `b16a066d4252`. That is [source.json](source.json),
  [upstreams.lock.json](upstreams.lock.json) and the `tp4`, `tp4-switched` and `tp3-switched`
  profiles.
- **Tree check:** the prepared tree's `indexer.py` (`34608dfb…`) and test file (`e4562069…`)
  are the files measured in `final-1`.
- **No image yet:** `spark-ds41f:candidate-60-indexer-exact` is not built.
- **Two `tp4` profiles:** [tp4-dev.json](tp4-dev.json) is the end-to-end arm (0817 as an
  overlay on candidate-60's image); `tp4.json` is the image profile for the series.

## Conclusion

Cold prefill at long context gets faster with unchanged output.

- **0817:** scoring is 19-22% faster at prefill depths. End to end at tp4-ring, the gain is
  +1.0-1.2% at 237K-470K and +2.1% at 910K tokens (9,064 → 9,251 tok/s). Outputs are identical:
  648 of 648 gate calls over two boots match candidate-58's records, and the kernel benches are
  bit-identical at every shape.
- **0818:** the indexer keeps a NaN as DeepSeek's reference does, at no cost and with finite
  outputs unchanged. It closes the last open item of the kernel-contracts work.
- **0819:** a micro gain, under 0.1% of serving.

## Not done

- **Fusing selection into scoring** (TODO.md). At 910K, writing the 256 × 455K BF16 score
  matrix and reading it back moves about 0.47 GB per call. The selection's read alone takes most
  of its 1.2-1.3 ms. Scoring a 4-row group's 4,096-position segments into shared memory and
  selecting each there would keep that traffic on chip. Every segmentation gives the same exact
  top 512, since the merges are exact and use the same tie rule. Expected gain: about 0.7-0.9
  ms per call at 910K, roughly +1.4% at 1M and +0.4% at 237K. It needs a new kernel combining
  the CUDA score path with the TileLang selection macros.
