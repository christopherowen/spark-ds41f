# DeepSeek V4.1 TileLang Indexer Score Early-Exit and Top-K Single-Segment / Empty Fast Paths

## Hypothesis & Objective

In DeepSeek V4.1 Flash (`deepseek_v4_1`), every DSA layer runs `indexer_score_v41` followed by `indexer_topk_segment_v41` and `indexer_topk_merge_v41`:

1. **`indexer_score_v41` redundant query/scale/weight loads on inactive CTAs**:
   `indexer_score_v41` launches `(ctas, rows)` CTAs where `ctas = min(ceildiv(width, 64), 128)`. Every CTA unconditionally loads `Q[row, :, :]` (`32 x 128` FP4), `QScale[row, :]` (`32` uint32), and `Weights[row, :]` (`32` BF16) from global memory into shared memory before entering the tile loop `T.serial(ceildiv(max(ceildiv(extent, block_P) - cta, 0), ctas))`. Whenever `cta * block_P >= extent` (for example, `120` of `128` CTAs per row at `extent = 512`, or all `128` CTAs per row on padded CUDA-graph rows where `extent = 0`), the loop executes zero iterations after wasting HBM bandwidth and shared-memory writes. Guarding `score_row` with `if cta * block_P < extent:` after `pdl.chain()` eliminates all HBM loads on inactive CTAs.

2. **`indexer_topk_segment_v41(final=True)` and `indexer_topk_merge_v41` empty-row fast path**:
   When `extent <= 0` (such as padded rows in decode CUDA graphs), `indexer_topk_segment_v41(final=True)` and `indexer_topk_merge_v41(final=True)` previously zero-initialized shared memory and executed two 256-bin radix histogram passes plus four Hillis-Steele block scans (`~30` `__syncthreads()` barriers) only to select `total = 0` entries and write `-1` / `-inf` padding. Adding `write_empty_final` writes `-1` / `-inf` (and `Counts[row] = 0`) directly when `extent <= 0` (`group_live <= 0`).

3. **`indexer_topk_merge_v41` single-live-list (`group_live == 1`) pass-through**:
   Each merge group `g` merges `group_live = min(max(live - g * fan_in, 0), fan_in)` sorted winner lists of length `topk`. Whenever `group_live == 1` — which holds in Level 0 for all rows with `0 < extent <= 4,096 * group` (`<= 4,096` for positions, `<= 32,768` for candidate blocks) and in Level 1 for all rows with `0 < extent <= 65,536 * group` — the single live input list at `base = g * fan_in * topk` is already sorted in ascending entry order with `-1` / `-inf` tail padding. Forwarding that list directly (via `write_final` when `final=True` or direct copy when `final=False`) bypasses loading `fan_in * topk` keys into shared memory and skips both radix histogram passes and all four Hillis-Steele scans.

## Method & Variants

Patch: [`dsv41-indexer-score-and-topk-fastpaths.patch`](./dsv41-indexer-score-and-topk-fastpaths.patch) against `.work/upstreams/vllm` (`d5df88d83`).

Modified files in vLLM:
- `vllm/models/deepseek_v4_1/tilelang/indexer.py`
- `tests/kernels/attention/test_deepseek_v41_tilelang.py`

## Verification Run

Verified on `dgx4` inside `spark-ds41f:1030` with the PDL source checker enabled:

```bash
pytest -p tests.kernels.deepseek_v41_pdl_sources \
  tests/kernels/attention/test_deepseek_v41_tilelang.py \
  -k "(indexer or topk or candidate) and not sanitizer" -v
```

Result: `19 passed, 1 skipped, 1 xfailed; PDL sources: 31 kernels launched, 31 with PDL, 0 violations`.
