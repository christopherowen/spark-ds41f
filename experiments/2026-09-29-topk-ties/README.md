# Deterministic top-k ties

Date: 2026-09-29. Base: the r5l candidate (`experiments/2026-09-29-r5l`).

## Question

B12X's DSA radix top-k gives the slots left at its threshold to whichever
tied candidates reach a shared-memory counter first. On DeepSeek V4.1 Flash
two identical full-row runs selected different sets for 78% of indexer layer
2's prefill rows, about 5 of 512 positions each (the r5l split checks). The
technical report sets no tie rule; DeepSeek's reference selects with
`torch.topk`, which repeats for the same scores, and dgpp
(`docs/inspiration.md`) pins exact ties to the lower index.

`0003-dsa-topk-position-ties.patch` (B12X tree `35299956` over r5k's
`640c8544`) keeps the tied candidates with the lowest logical positions in
both the buffered arm and the exact overflow fallback. Does it cost speed,
does it make selection exact, and does it change draft acceptance?

## Unit tests

On dgx3 inside the r5l image with the patched module on `PYTHONPATH`
(2026-09-29): the new `tests/attention/test_topk_position_ties.py` passes 6/6
(700 and 5,000 tied candidates against a stable sort, 20 repeated runs,
partial rows, distinct scores against `torch.topk`); the unpatched kernel
fails 5 of those 6, including repeating its own selection. B12X's existing
indexer and top-k tests pass 103/103 with the patch.

## Arms

| Arm | What it is |
|---|---|
| `ties` | the r5l candidate with the patched `tiled_topk.py` mounted by `overlay.sh` |
| `ties-check` | `ties` with `SPARK3_DS41_INDEXER_SPLIT_CHECK=1` |

## Method

`run.sh` runs the split check under four background decode streams and
requires zero differences (`check_summary.py --exact`), times one 4,096-token
prefill chunk at 8K-200K of context, then alternates decode screens (prose
and JSON answers at one and eight streams, six samples) as ties, r5k, r5l,
ties, r5k, r5l, and leaves r5k running.
