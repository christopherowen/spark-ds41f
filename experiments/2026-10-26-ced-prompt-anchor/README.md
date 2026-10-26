# CED's decoder replay anchored at the prompt end

The owner approved this correctness fix for the next candidate. CED's decoder now replays
the last 128 rows of the prompt, as the V4.1 paper defines bounded replay, wherever the
prompt's chunks end. Until now, each chunk was anchored at its own end.

- **vLLM branch:** `ced-prompt-anchor`, two commits on `next-candidate` 1bc005688:
  1. e4f46109c: CED replays the prompt's last 128 rows, wherever its chunks end;
  2. aa4c0d1c0: CED replay test, a decoder stack on the TileLang kernels under several
     chunkings.
- **Temperature-0 output changes only for affected prompts:** a prompt longer than one
  chunk whose final chunk has 1-128 rows (about 128/T of such prompts, 0.79% at TP4 and
  1.57% at TP3's 8,192 budget). Every other prompt is bit-identical by construction.
- **Results:** the CPU simulation passes, and the dgx4 runs are [below](#dgx4-runs).
  The full-model gate needs the cluster.

## The bug

Each prompt chunk computed its replay start from its own end:
`replay = max(prefix_start, end - kept if q > kept else 0)` in the requests kernel.

- **A final chunk of 1-128 rows kept every row with no floor.** Its rows then read the
  previous chunk's 128 kept rows, which were computed under that chunk's own floor
  (its end less 128).
- **So the replay depended on where the last chunk boundary fell.** Chunks end at
  multiples of the long-prefill threshold T (budget − max_num_seqs × (1 + drafts)), so
  the result depended on the budget and on TP3 against TP4.
- **The gates never saw it.** Teacher-forced checks run prompt-logprob requests, which
  keep every decoder row (`keep_full`).

## Reference semantics

| Source | Decoder replay | Anchor | Prefix hits | Prompt logprobs |
| --- | --- | --- | --- | --- |
| V4.1 paper | the last n_win = 128 tokens of the prompt, simulated in post-training; a replayed query attends from the replay start s up to i, within the window | prompt end | the replayed decoder SWA KV is "only for decoding, not for prefix caching" | (not discussed) |
| The checkpoint's `inference/model.py` | none: every layer runs every row (CED off) | n/a | n/a | full rows |
| DeepInfer (DeepSeek's engine, 5537415) | `[N − keep, N)` with keep ≥ window + B (README default 140; production 134) | N, the sequence length: the prompt, plus the generated tokens a preempted request recomputes | hits capped at L − 134 | forced off under truncation |
| This branch | the last 128 rows | `prefill_len`: the prompt, plus recomputed outputs after preemption | hits capped at L − 128 (`prefill_replay_window`), so the floor is always L − 128 | full rows (`keep_full`), as the reference computes them |

DeepInfer, in detail (`csrc/engine/input_metadata/prepare_tensor_input_v4.cpp`):

- **Kept rows.** `clip_prefix = clamp(N − keep − q_start, 0, q − 1)`: each chunk keeps
  its rows from `N − keep` on. A chunk wholly before that keeps its last row as a
  placeholder; the placeholder writes no ring slot and sees nothing.
- **Visibility.** `abs_lo = max(pos − window + 1, N − keep, valid_start)`. Its own
  equivalence test (`examples/yoco/yoco_equiv.py`) states the reference as the sliding
  window intersected with `[N − keep, N)`. At keep = 128 that is the paper's definition,
  and ours.
- **Why keep is larger.** keep must be at least window + B. B is
  `DEEPINFER_MAX_BACKTRACKING_TOKENS`, raised to at least drafts + 1 (6 at five drafts).
  It is the rollback reserve of DeepInfer's SWA ring buffer (`get_inference_swa_block_size`
  is window + B): a backtrack of up to B tokens still leaves a full window of valid slots.
- **The effect.** The first replayed rows see up to B more rows than the paper's 128, so
  DeepInfer's production output (keep 134) departs slightly from flat 128.

**What we follow: flat 128.** The paper defines it, and post-training simulated it.
- B belongs to DeepInfer's ring. Our decoder SWA is paged and request-private and never
  rolls back into the prompt, and our prefix-hit cap is already L − 128.
- Following DeepInfer's production would tie the model's semantics to the draft count.
- Matching DeepInfer production instead would mean CED_WINDOW 134 for the replay and the
  hit cap. It is a one-constant change if the owner prefers it.

**How each case behaves on this branch:**
- **Prefill chunks.** The floor is max(prefix start, prefill_len − 128). Every chunk
  keeps exactly its rows from the floor on, and a chunk that ends before the floor keeps
  its last row, which replays only itself (the nonfinal skip's placeholder).
- **Prefix hits.** The KV manager caps a hit at prefill_len − 128, rounded down to the
  cache block, so the floor is prefill_len − 128 and the replay is the paper's in full.
  A first chunk after a hit can be shorter than 128 rows and be cut by the floor; that
  step compacts and runs eagerly (below).
- **Decode steps.** These are unchanged. A decode row's window, pos − 127 and up, is
  already past the floor; decode steps keep the prefix start as their floor value and
  their metadata bits.
- **Prompt logprobs.** These are unchanged: every row, from the prefix start (the
  reference's computation).
- **The drafter.** Its context rows are the decoder's kept rows, and its windows start
  at the replay start (`bind_draft_replay_start`, 0501). Its context is therefore the
  same 128 rows for every chunking. Every step's draft window reads only kept rows
  (checked in the CPU tests and the simulation).
- **Preemption.** The anchor is the prefill length, which includes the outputs a resumed
  request recomputes, as DeepInfer's N does.
- **Streaming input updates.** The anchor is the new prefill length. The live request
  keeps its decoder rows from its original admission, so the new floor's rows exist.

## The change

`vllm/models/deepseek_v4_1/ced.py`:
- `replay_rows(query_lengths, computed, prefill_lens, prefix_start, keep_full)` returns
  each request's floor and kept-row limit:
  - a prefilling request: floor `max(prefix, prefill_len − 128)`, limit
    `clamp(chunk end − floor, 1, 128)`;
  - a decoding request: floor = prefix start, limit 128;
  - `keep_full`: floor = prefix start, every row.
- `plan_ced` compacts whenever a request keeps fewer rows than it schedules. Before, it
  compacted only when a chunk exceeded the window. `max_query_len` stays the window
  bound, so the decoder's mode and plans are those of the unlimited step.

`vllm/models/deepseek_v4_1/tilelang/ced.py`: the requests kernel replays from
`min(floor, first kept row)` in all three of its forms. A placeholder row therefore
replays itself, never an empty window: stale bytes would trip the router's non-finite
assert, as 0501 found.

`nvidia/model_state.py`:
- `_stage_ced` takes the floors and limits from `replay_rows`.
- The nonfinal skip's policy, its alignment and successor contract and its multimodal
  exclusion are gone. The anchor generalizes the skip: any chunk that ends before the
  floor keeps one row, whatever follows it.

**Short chunks.** A chunk of at most 128 rows can now compact, for example the first
chunk after a prefix hit just before a chunk boundary. CED steps must run eagerly
(`prepare_attn` raises otherwise). The runner sent a step eagerly only when a query
exceeded `max_cudagraph_query_len`; now it also asks the model state
(`compacts_short_queries`).
- **Where a graph could take such a step:** DS4.1 is not torch-compiled, so its only
  graphs are FULL decode graphs. The varlen ones, which adaptive verification captures,
  take any step whose queries are all at most 1 + drafts rows (6) and whose tokens fit a
  capture size.
- **When that happens:** a compacting chunk of 2-6 rows. Chunk lengths after a
  block-aligned hit are multiples of 16, so ordinary serving never produces one. A
  streaming input update that ends 2-6 rows before a chunk boundary could, and without
  the check that step would raise.

**Bit-identity.** When the final chunk has more than 128 rows, or the prompt is a single
chunk, the final step's kept rows and replay start are next-candidate's (CPU test, and
the simulation over every case). An earlier chunk may now keep one row where it kept 128
before (the chunk before a final chunk of 129-198 rows, and multimodal prompts), and the
final chunk never read those rows. Batch invariance covers the other
requests in those steps: the 10-19 CED gates found that shrinking one request's decoder
rows changes no other request's bits.

## Affected prompts

A prompt is affected when it is longer than one chunk and its final chunk has 1-128 rows.
That is a prompt length mod T in [1, 128], counted from the prefix hit for cached prompts.

| Shape | Budget | T | Affected share of prompts longer than T |
| --- | ---: | ---: | ---: |
| TP4 ring | 16,384 | 16,288 | 0.79% |
| TP3 triangle | 8,192 | 8,144 | 1.57% |
| TP3 triangle | 6,144 | 6,096 | 2.10% |
| TP3 triangle | 12,288 | 12,240 | 1.05% |

The simulation's cross-shape check measures how the shapes disagree. On next-candidate,
the same prompt gives different decoder replays at two shapes in 256-912 of each pair's
3,391-3,967 boundary prompts. On this branch it never does.

**Speed.**
- The skip now also covers two cases it missed. Prompts whose final chunk has 129-198
  rows used to keep 128 rows in the chunk before it, under the 199-row successor rule;
  that chunk now keeps one. Multimodal prompts were ineligible; they now get the skip.
- In affected prompts, the chunk before the final one keeps 128 − r rows instead of 128.
- Each of these saves at most 127 decoder rows in one step, too little to measure.
- **No step that replayed a graph before runs eagerly now.** A compacting short chunk
  reaches a graph only at 2-6 rows (see Short chunks), which ordinary serving never
  produces.
- **Host cost:** the CPU planning is one loop over the step's requests, as the skip's
  was. The eager check runs only for steps with a prefill whose queries all fit 128 rows.

## CPU simulation

[anchor_sim.py](anchor_sim.py) loads both trees' planning code through AST: `replay_rows`,
`plan_ced`, `nonfinal_limits` and the scheduler's `_aligned_prefill_chunk`. It writes the
requests kernel's replay rule and the window metadata out one line each. A stacked
decoder row depends on its window's rows and, through them, on everything they saw. So
two runs agree when every row in the closure of the last 128 rows (what the first decode
step and the drafter read) has the same window and every row it reaches was kept. The
reference replays the last 128 rows in one pass from the floor.

```sh
python3 -I anchor_sim.py ~/projects/vllm-next-candidate ~/projects/vllm-ced-prompt-anchor --json runs/anchor-sim.json
```

| Threshold | Cases (with hits) | Final chunk ≤ 128 | next-candidate differs from reference | … with a final chunk > 128 | This branch differs | Unwritten reads (decoder / draft), both trees |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 6,096 | 12,783 (10,544) | 2,831 | 2,827 | 0 | 0 | 0 / 0 |
| 8,144 | 12,783 (10,544) | 2,831 | 2,827 | 0 | 0 | 0 / 0 |
| 12,240 | 12,783 (10,544) | 2,831 | 2,827 | 0 | 0 | 0 / 0 |
| 16,288 | 12,927 (10,688) | 2,749 | 2,747 | 0 | 0 | 0 / 0 |

- **Coverage:** prompts of 1-511 tokens, and every length within −256..+319 of the first
  three chunk boundaries. Each runs from the start and from 64-aligned prefix hits near
  the boundaries.
- **The few short final chunks that agree** have a hit exactly 128 rows before the prompt
  end, where the old floor (the hit) was the right one.
- **Final step unchanged:** this branch's final step equals next-candidate's in every
  case whose final chunk has more than 128 rows.

## vLLM tests

`tests/v1/attention/test_b12x_v41_ced_metadata.py`:
- `replay_rows` cases: final chunks of 64 and 129 rows, a hit cut by the floor, short
  prompts, a resumed request, decode rows, prompt logprobs.
- Short-query compaction.
- The anchored replay against the scheduler's aligned chunks at thresholds 300, 4,048,
  8,144 and 16,288, with hits: every replayed row's window, placeholders that see only
  themselves, and every step's draft window.
- The final step for final chunks over the window.
- `compacts_short_queries`.
- On CUDA: the model state's chunks of one prompt with final chunks of 4,096, 129, 128,
  64 and 1 rows, through the requests kernel, and the draft windows of a chunk at the
  floor.

`tests/kernels/test_deepseek_v41_tilelang_ced.py`: the requests kernel against a
reference, with limits of 1, 37, 128 and every row and floors before, at and past the
first kept row. For final chunks and decode rows, it also checks the kernel against the
frozen r6x Triton kernel.

`tests/kernels/attention/test_deepseek_v41_ced_replay.py`, a decoder stack on the
TileLang kernels:
- Three sliding-window layers through the model state's staging, the decoder's token
  metadata, the window metadata and the sparse MLA.
- KV records are written for each step's kept rows into a cache that starts as NaN.
- A 1,024-row prompt start plus a final chunk of 1, 64, 128 or 129 rows, as one chunk,
  in 512- and 384-row chunks, and after a hit at 448. The rows from the floor on must
  equal the single chunk's, bit for bit.

## dgx4 runs

[make_bundles.py](make_bundles.py) writes two `kernel-local` bundles over candidate 6's
image (`spark-ds41f:19f2c20ed4d6-next-6`). Each mounts every file its vLLM ref changes
from the image's tree (26382f5cf).

- `bundles/next-candidate`: the CED test files on next-candidate, and the chunking
  check as a record.
- `bundles/anchor`: the CED test files and the new test, the chunking check (every
  chunking bit-identical), and the CED kernel benchmark.
- `bundles/anchor-wide` and `bundles/next-candidate-wide`: the test files of the paths
  the change reaches beyond CED, on each tree.

[chunking_check.py](chunking_check.py) runs the decoder stack under each tree's own
model state; next-candidate's gets the recipe's skip policy. It covers:
- the small prompts;
- a 64-row first chunk after a hit, which keeps 40 rows;
- prompts at the serving thresholds: 8,144 + r and 16,288 + r for r = 1, 64, 128, 129,
  chunked at TP3's 8,144 and TP4's 16,288.

It reports whether each run's last 128 rows equal the single chunk's, and TP3 against TP4.
For prompts of 8,144 + r, TP4 runs one chunk, so the TP3-chunked case against the single
chunk is TP3 against TP4.

The bundles ran from this branch's experiment folder, checked out in
`~/spark-lab/agents/ced-prompt-anchor/spark-ds41f`. The command was
`scripts/lab.py kernel-local experiments/2026-10-26-ced-prompt-anchor/bundles/<bundle> --out ~/spark-lab/agents/ced-prompt-anchor/out/<bundle>-1`.
Nothing ran on dgx1-dgx3.

**`anchor` (finished 2026-10-26 07:55 CET; [runs/anchor-1](runs/anchor-1)):**
- **Chunking check:** all 25 cases are bit-identical to the single chunk and finite. That
  covers final chunks of 1, 64, 128 and 129 rows; 512- and 384-row chunks; a hit at 448;
  the 64-row hit chunk that keeps 40; and TP3 and TP4 at their thresholds.
- **CED kernel benchmark:** the requests, row-map and copy kernels are exact against the
  frozen r6x Triton kernels (`differs_at: []`), and no gated case is slower twice.
  The requests kernel, with this branch's floors for final chunks and decode rows, is
  2.8-23.6% faster than Triton per call, as the port was.
- **CED test files:** 1,118 passed, 1 skipped, 20 failed, 1 error. Every CED test passes,
  the new ones included. The failures and the error are outside CED:
  - `test_deepseek_v4_1_engram.py`, disk Engram (7): io_uring is not permitted in the
    lab container;
  - `test_gpu_model_runner_v2.py` (13 and the error): hand-built runners and namespaces
    lack attributes (`vllm_config`, `_block_table_max_model_len`, `sink`), and the
    `workspace_init` fixture is absent under `--noconftest`. None reaches
    `compacts_short_queries`.

**`next-candidate` (finished 2026-10-26 08:06 CET; [runs/next-candidate-1](runs/next-candidate-1)):**
- **CED test files:** 1,075 passed, 1 skipped, and the same 20 failures and 1 error as
  on the branch (the verdicts' failure sets are equal). The branch's 43 more passes are
  its new tests.
- **Chunking check:** 17 of the 25 cases differ from the single chunk. They are every run
  with a final chunk of 1, 64 or 128 rows, at the serving thresholds too, plus the 64-row
  hit chunk followed by an 88-row final. In those runs 32-94 of the last 128 rows differ
  (largest difference 1.02 in the stack's units).
  - **Every 129-row final is identical**, at every threshold.
  - **So are chunkings whose final chunk is long**, such as 384 + 384 + 257.
  - **TP3 against TP4.** Prompts of 8,145, 8,208 and 8,272 tokens (8,144 + 1, 64, 128)
    give TP3 output different from TP4's single chunk. At 16,288 + r, TP3 and TP4 agree
    with each other: both put the same short final after the same 128 kept rows. They
    are both wrong against the single chunk.

| Final chunk | next-candidate: chunkings equal to the single chunk | This branch |
| --- | --- | --- |
| 1 row | 1 of 6 (only the 384-row chunking, whose final has 257 rows) | 6 of 6 |
| 64 rows | 1 of 6 (likewise, 320 rows) | 6 of 6 |
| 128 rows | 1 of 6 (likewise, 384 rows) | 6 of 6 |
| 129 rows | 5 of 6 (the 384-row chunking leaves a 1-row final) | 6 of 6 |
| hit at 448, then 88 rows | 0 of 1 | 1 of 1 |

**Unaffected prompts against next-candidate (run 3, finished 14:39 CET;
[runs/anchor-3](runs/anchor-3), [runs/next-candidate-3](runs/next-candidate-3)).** The
chunking check now prints a SHA-256 of every case's rows, so the two trees compare bit
for bit. Run 2 was interrupted when dgx1-dgx4 powered off at about 08:15 CET for
re-cabling, and run 3 replaced it.
- **The single chunk:** identical across trees (25 of 25 digests).
- **Unaffected cases:** all 8 (every 129-row final and the long-final chunkings) give
  next-candidate's rows bit for bit.
- **Affected cases:** all 17 differ from next-candidate, as intended. On the branch each
  equals its single chunk; on next-candidate none does.
- **Everything else repeats run 1:** the branch's chunking check 25 of 25, next-candidate
  17 of 25 differing, the benchmark exact, the same failure sets.

| | Cases | Branch equals its single chunk | Branch equals next-candidate |
| --- | ---: | ---: | ---: |
| Unaffected (next-candidate equals its single chunk) | 8 | 8 | 8 |
| Affected (final chunk of 1-128 rows) | 17 | 17 | 0 |

**Wider tests (`anchor-wide`, finished 14:39 CET; [runs/anchor-wide-1](runs/anchor-wide-1)).**
These are the test files of the paths the change reaches beyond CED:
- the DS4.1 attention layers, DSpark context KV and MLA, SP prefill, the TileLang attention
  kernels;
- the drafter's input preparation and lookahead, adaptive verification, the Markov head;
- the runner's graph profiling, the context limit and the input batch, the KV manager's
  utilities.

428 passed, 2 skipped, 1 xfailed; 43 failed and 5 errors.
- Every failure is in `tests/v1/core/test_kv_cache_utils.py` (40 and the 5 errors) or
  `tests/v1/spec_decode/test_dflash_lookahead.py` (3).
- 43 need a Hugging Face config (`Qwen/Qwen3-0.6B`) that the offline lab container
  lacks; the 5 errors need the `caplog_vllm` fixture, which `--noconftest` removes.
- The branch changes neither file, and the paths they test take no input from it.
- **next-candidate (`next-candidate-wide`, finished 14:43 CET;
  [runs/next-candidate-wide-1](runs/next-candidate-wide-1))** gives the same counts and
  the same failure set exactly.

## What the gate and the release comparison must expect

- **Determinism gate.** The four hashes are unchanged by this branch.
  - The gate's long prompt (700 lines, about 10.5K tokens) is unaffected at every recipe
    threshold: a single chunk at 12,240 and 16,288, final chunks of about 2.4K and 4.4K
    at 8,144 and 6,096.
  - The short and cached prompts are single chunks.
  - `identical: true` must hold in every scenario.
- **Release comparison, prefill sizes** (exact token counts, mod T):

  | Size | TP4 (16,288) | TP3 8,192 (8,144) | TP3 6,144 (6,096) | TP3 12,288 (12,240) |
  | --- | ---: | ---: | ---: | ---: |
  | 32,768 | 192 | 192 | 2,288 | 8,288 |
  | 131,072 | 768 | 768 | 3,056 | 8,672 |
  | 262,144 | 1,536 | 1,536 | **16** | 5,104 |
  | 524,288 (TP4) / 500,000 (TP3) | 3,072 | 3,216 | **128** | 10,400 |
  | 1,000,000 (TP4) | 6,432 | | | |

  Only TP3 at the 6,144 budget has affected sizes, 262,144 and 500,000. Their outputs
  change, which the timing ignores, and their decoder work drops by up to 127 rows in one
  step. Prefill speed must be level everywhere.
- **Model behaviour and the engine grid.** Outputs change only for affected prompts:
  multi-chunk prompts whose length mod T is 1-128, or which come out that way from a
  prefix hit. Decode shapes are unaffected.
- **A new property to gate when the cluster is free:** the same prompt must give identical
  temperature-0 tokens and logprobs at every shape and budget. Use a prompt of exactly
  8,208 tokens (8,144 + 64):
  - at TP4 it is one chunk;
  - at TP3's 8,192 budget its final chunk has 64 rows (affected);
  - at TP3's 6,144 budget its final chunk has 2,112 rows.

  On next-candidate the TP3-8192 output differs from the other two; on this branch all
  three must be identical. The prompt must be token-exact, so send prompt token ids
  through `/v1/completions`.

**For the candidate's release tests:** the manifest already lists the CED test files this
branch changes. Add the new `tests/kernels/attention/test_deepseek_v41_ced_replay.py`.

## Not done here

- **A full-model check of chunking identity.** The full model needs at least three nodes,
  so it waits for the cluster (the property above).
- **The CPU reference runner (#364).** The checkpoint's `inference/model.py` has no
  bounded replay: it computes CED off. To check the anchor, its decoder layers' (20-39)
  sliding windows would start at prompt end − 128, a few lines in the runner. It would
  then be compared with a production run of an affected prompt, which needs the cluster.
  How large the anchored-against-per-chunk difference is next to the FP8 noise floor
  (median 0.056 nats) is unknown.

## Files

- [anchor_sim.py](anchor_sim.py), [runs/anchor-sim.json](runs/anchor-sim.json): the CPU
  simulation and its report.
- [make_bundles.py](make_bundles.py), [chunking_check.py](chunking_check.py),
  `bundles/`: the dgx4 bundles.
- `runs/`: the dgx4 outputs.
