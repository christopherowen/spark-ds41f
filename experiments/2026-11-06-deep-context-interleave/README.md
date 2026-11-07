# Deep-Context Prefill Step Budgeting (>= 256K) & Elapsed-Stall Decode Interleave

## Base & Candidate Revisions

- Deployment base commit: `239fad997eac7f42bd9a3f001a6958b29080a2cf`
- vLLM base commit (`candidate-1107`): `a75357bba81e31e4f3085700f94b6a0b29aed44a`
- vLLM candidate commit (`feat/deep-context-interleave`): `67f2ff1ba5f97232141b7c225a2152d021e3b25d`
- Exported patch: `experiments/2026-11-06-deep-context-interleave/vllm/0001-sched-gate-2-048-grid-chunk-step-down-to-256K-and-in.patch`
- Verification image: `spark-ds41f:candidate-38` with Candidate `1107` source tree on `sp8` (2 fresh container boots)

## Root Cause of `rc15` Regression & Rework Design

1. **Rework 1 (`#697` Deep-Context Prefill Step Budgeting gated to $c \ge 256\text{K}$)**:
   - In `rc15`, `--prefill-step-target-s 1.0` shrank prefill chunks even at `0..128K` context (`6,096 -> 4,096 -> 2,048` on TP3), turning a 22-chunk 128K prefill into **55 chunks** (`+33` full-model MoE weight-stream passes = `+5.0 s`).
   - **Rework**: `_context_aware_chunk_threshold` gates the 2,048-token grid step-down strictly to **`num_computed_tokens >= 262144` ($\ge 256\text{K}$)** when `self.prefill_yield` (`--prefill-yield short`) is active and `num_runnable_decodes > 0`.
   - Below `256K` (including the entire `0..128K` `grid`, `stall-7`, and `stall-15` suites) and whenever `num_runnable_decodes == 0`, prefill chunks remain at full `long_prefill_token_threshold` (`6,096` on TP3 = **22 chunks at 128K**, `16,288` on TP4 = **9 chunks at 128K**, **0 extra chunks**).
   - At deep contexts (`256K..512K` -> `4,096`; `>= 512K` -> `2,048`), single-step stall is cut by **`50.3%–71.2%`** (`2.583 s -> 1.155 s` at `524K`, `-55.3%`; `3.394 s -> 0.976 s` marginal / `1.688 s` max at `900K`, `-71.2%` / `-50.3%`).

2. **Rework 2 (`#696` Elapsed-Stall-Gated Prefill-Decode Interleave)**:
   - In `rc15`, `#696` interleaved $K=2$ decode steps after *every* prefill chunk unconditionally (`55 * 2 = 110` interleaved decode steps at `128K`).
   - **Rework**: When `--prefill-yield short` is active and runnable decode requests coexist with an in-flight multi-chunk prefill, the scheduler accumulates elapsed prefill stall time since the last decode-only step and interleaves $K=2$ pure decode steps **only when accumulated stall reaches `>= 1.0 s`**.
   - At `128K` on TP3:
     - `grid` (`c1`, 0 active decode peers): **22 chunks, 0 interleaved decode steps, 0.0% prefill change**.
     - `stall-15` (128K prefill + 15 decodes): **22 chunks** (`0` extra chunks), **42 interleaved decode steps** (down from `110` in `rc15`), mean decode ITL **`519.56 ms -> 215.92 ms` (`-58.44%`)**, stall ratio `8.042x -> 3.342x`.
     - `stall-7` (128K prefill + 7 decodes): **22 chunks** (`0` extra chunks), **42 interleaved decode steps**, mean decode ITL **`492.95 ms -> 173.40 ms` (`-64.82%`)**, stall ratio `12.634x -> 4.444x`.

3. **Rework 3 (`#701` Markov Draft Depth Evaluation on GB10 GPU)**:
   - Across 2 container boots on `sp8` (`NVIDIA GB10`), capping `DSparkSpeculator._sample_sequential` from `5` steps to `4/3/2` steps at `c8/c12/c16` saves only **`1.5842 ms`** at `c8`, **`3.1797 ms`** at `c12`, and **`4.8441 ms`** at `c16` (`< 0.4%` of step time, because the 3-layer parallel MTP backbone still runs on all `num_reqs * 5` rows inside the batch-wide FULL CUDA graph).
   - Meanwhile, in a mixed `c16` batch with high-confidence code/structured requests (`p=0.92`) and lower-confidence chat requests (`p=0.55`), `AdaptiveVerificationManager` (`row_cost = 0.035`) + `DeadRows` + `CopyDrafter` (`31412339a`) already assign all 5 draft slots (`k=0..4`) to high-confidence requests while pruning low-confidence rows; capping batch draft depth to `2` steps reduces high-confidence request output from **`4.8414 -> 3.5529` tokens/step (`-43.78%`)**.
   - Conclusion: Keep `#701` unbundled and ship the clean, zero-optionality `#697` + `#696` scheduler rework.
