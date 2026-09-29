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
| `split` | patch 0025 (`0025-deepseek-v41-indexer-sp-split.patch`) mounted over r5k's attention module by `overlay.sh` |
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
