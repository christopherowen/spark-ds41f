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

## Method

`capture_depth.py` tokenizes one real-text document once. For each depth D, a
request with a fresh cache salt prefills the first D - 4096 tokens; a second
request with the same salt sends the first D tokens, reuses the cached prefix
and prefills exactly one 4,096-token chunk at depth D. Prefill time is read
from `vllm:request_prefill_time_seconds`. Two unprofiled rounds time the
chunks; the third runs under one profiler window that holds only the four
measured chunks. `run_profile.sh` boots the arm, captures, and restores the
promoted configuration.
