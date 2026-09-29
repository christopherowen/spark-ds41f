# Indexer prefill split across TP ranks

Date: 2026-09-29. Base: promoted r5k (`config/cluster.json`).

## Question

During sequence-parallel prefill every rank gathers all rows before attention,
and the DeepSeek V4.1 indexer is replicated (`ReplicatedLinear` query and
weight projections, every rank scores every index head). Each of the three
ranks therefore scores and selects the same top-k for every row of a chunk.
Splitting the rows across the ranks and all-gathering the selected indices
would remove two thirds of that work, exactly (tonyd2wild's version measured
+6.3% on a 131K prompt at TP4).

The indexer's cost per row grows with context depth. Before writing code, this
experiment measures its share of one 4,096-token prefill chunk at 8K, 64K, 131K
and 200K of context.

## Arms

| Arm | What it is |
|---|---|
| `baseline-profile` | promoted r5k with the torch profiler (no Python stacks) |
| `split` | the first version of patch 0025 (tree `d82a42e2`) mounted over r5k's attention module by `overlay.sh` at commit `c811b4a` |
| `split-check` | `split` with `SPARK3_DS41_INDEXER_SPLIT_CHECK=1`: every rank also runs the full-row indexer and compares its rows |

## Method

`capture_depth.py` tokenizes one real-text document once. For each depth D, a
request with a fresh cache salt prefills the first D - 4096 tokens; a second
request with the same salt sends the first D tokens, reuses the cached prefix
and prefills exactly one 4,096-token chunk at depth D. Prefill time is read
from `vllm:request_prefill_time_seconds`. Two unprofiled rounds time the
chunks; the third runs under one profiler window that holds only the four
measured chunks. `run_profile.sh` boots the arm, captures, and restores the
promoted configuration.

## Baseline profile

`run_profile.sh`, 2026-09-29. Server-side prefill time of the one 4,096-token
chunk, three rounds (the third under the profiler), and rank 0's kernels in
the profiled round (`analyze_trace.py`; the rank-0 trace is 3.4 MB):

| Depth | Chunk prefill (ms) | Rank-0 wall (ms) | Indexer kernels (ms) | Everything else (ms) |
|---:|---:|---:|---:|---:|
| 8K | 1,035 / 1,034 / 1,033 | 1,032 | 17 | 1,015 |
| 64K | 1,155 / 1,160 / 1,161 | 1,158 | 113 | 1,044 |
| 131K | 1,254 / 1,254 / 1,263 | 1,255 | 218 | 1,037 |
| 200K | 1,369 / 1,368 / 1,367 | 1,355 | 323 | 1,032 |

The indexer (the `dsa_indexer` MXFP4 score, select-prepare, tiled top-k and
sort kernels, plus the page-table kernel) is all of the chunk's growth with
depth: 24% of the chunk at 200K. Its score kernel alone takes 261 ms there,
about 64 TFLOP/s of MXFP4. The rest of the chunk is flat: sparse MLA 256 ms,
routed MoE 355 ms, collectives 162-173 ms, dense GEMMs and mHC 198 ms. Only
index-source layers 2, 8, 14 and 20 score every compressed key; layers 24-36
score layer 20's 16,384 candidates.

Splitting the rows removes two thirds of the indexer on every rank: about
11, 76, 145 and 216 ms per chunk at these depths (1%, 7%, 12% and 16%), less
one all-gather of 2.7 MiB per rank for each of the eight indexer layers.

## Split runs

`run_split.sh`, 2026-09-29, with the first version of patch 0025.

`split-check` (four background decode streams) found differences: on every
sequence-parallel forward, layers 2, 8 and 14 selected different top-k
positions for most rows on all three ranks (for example 1,351 of rank 0's
1,366 rows at layer 2), while layers 20-36 and the candidate lists matched.
Those three are the only indexer layers whose head weights come from
`weights_proj` over the gathered hidden state, and the first version ran that
BF16 projection on each rank's 1,366 rows instead of all 4,096: its GEMV
rounds differently at another row count. Layers 20-36 compute the weights on
local rows in both paths. The FP8 query projection quantizes per row and
matched. The promoted patch 0025 (`patches/vllm/0025-deepseek-v41-indexer-sp-split.patch`,
tree `09f7e369` over r5k) runs the head-weight projection on every row and
slices it; `experiments/2026-09-29-r5l` re-checks it in the built image.

`split` timings, server-side prefill of one 4,096-token chunk (two rounds):

| Depth | r5k (ms) | split (ms) | Change |
|---:|---:|---:|---:|
| 8K | 1,035 | 1,038 / 1,039 | 0% |
| 64K | 1,155 | 1,094 / 1,092 | -5.4% |
| 131K | 1,254 | 1,127 / 1,121 | -10.4% |
| 200K | 1,368 | 1,167 / 1,160 | -14.9% |

The lean screen (`results/private/bench/indexer-split-split`) against r5k's
`candidate-auto` screen: quality 5/5; real-text prefill 32K 3,863 against
3,781 tok/s (+2.2%), 64K 3,825 against 3,696 (+3.5%), 131K 3,783 against
3,569 (+6.0%); decode unchanged (prose 50.4 and 161.7 tok/s at 1 and 8
streams against 50.0 and 164.6, code at eight streams 187.4 against 184.6;
single-stream code 59.0 against 66.1 tok/s came from acceptance, 1.90 against
2.38 drafts per step, while its step time fell from 48.3 to 46.6 ms).
