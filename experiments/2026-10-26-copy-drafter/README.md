# Copy drafts beside DSpark: the build of #601's go variant, proven on dgx2

The suffix-drafter record (#601, `experiments/2026-10-26-suffix-drafter`) designed a copy
drafter beside DSpark and gave a go: exact per-request copies, minimum match 16, at most 16
drafts with the cap growing from 8. The owner then dropped any table shared between
requests (#609). This record is that build, for the release after next.

- **Base:** vLLM `next-candidate` c579b3fe1 (candidate 7's tree, frozen; untouched).
- **Branch:** vLLM `copy-drafts`, head 63082e9b9, five commits (local, not pushed;
  [vllm/](vllm/) holds them as patches 0533-0537, numbers provisional).
- **Runs:** dgx2 only, `scripts/lab.py kernel-local`, the branch overlaid on
  spark-ds41f-1025's image. No serving, no cluster time.
- **Rules kept:** temperature-0 text identical with copy drafts on and off (drafts change
  speed only); native weights; every new kernel TileLang, declared through kernel
  preparation and graph-capturable; nothing shared between requests; without
  `copy_drafts` every size, kernel and graph is the release's.

## The short answer

- **Built.** Each step, after DSpark drafts, a request whose context ends with text it has
  seen before (the latest earlier position whose 16, 32, ..., 256 preceding tokens equal
  the context's tail, longest first) takes the tokens that followed it, up to its cap, in
  place of DSpark's drafts. The cap starts at 8 and doubles after a fully accepted copy
  step, to the lookahead (16); a rejection or a step without a copy returns it to 8. The
  copy's confidence is the request's own acceptance estimate. Adaptive verification's
  budget ranks copy and DSpark drafts together; greedy and block verification commit only
  the target's tokens.
- **Strictly per request.** The search reads the request's own row of the token history up
  to its length; the copy state (cap, estimate, pending proposal) lives in the request's
  slot and is reset when a request takes the slot; adaptive verification's planned
  confidences are reset the same way. Tests plant other requests' text in the same batch
  and a slot's previous request's tokens past the new request's length: neither ever
  proposes.
- **Tested on dgx2** (run 4, the final head 63082e9b9, 17:47-17:49 CET, pass): 10
  scheduler and 19 copy-kernel and proposer tests (new), the TileLang verification suite
  at 16 drafts and at 5 (667 and 667), adaptive and greedy verification and input
  preparation (118), the scheduler's aligned chunks and sink rejection (69), and DS4.1's
  decode-row kernels (270: the compressor ring, GEMMs to 200 rows, MoE to 300, sparse
  MLA, mHC). next-candidate's own tree passes the same existing files (663, 117, 69,
  266: the branch adds 4, 1 and 4 tests to them).
- **Temperature 0.** The committed text through the runner's greedy verification kernels
  is the target's own greedy text with DSpark-like drafts alone and with copy drafts
  (a toy target whose rows are batch-invariant, 700 tokens over four prompts that copy a
  lot, a little and not at all). On the real model this rests on DS4.1's rows being
  bit-equal whatever the step's rows and a request's width (the tests above); the
  cluster gate (prepared) checks it end to end.
- **What it costs** (dgx2, gated, run 5): the copy kernels take 31 us a step at one
  stream and a 64K context (125 us at 1M; within 1% of a step everywhere). Greedy
  verification costs the same at either width (4.1 us). Block verification of a step
  without a copy runs at DSpark's width, level with the release (40.7 against 40.8 us at
  c1). A wide step's own costs are its rows and its experts: its routed layer adds 18.7 ms
  at c1 (48 experts a layer against 24), 17.3-18.9 us an added expert-layer, 6% above
  #601's stream rate (251.0 against 266.6 GB/s).
- **Expected on recorded traffic** (#601's agent proxy and copy probe, priced with the
  kernels' measured costs, the measured expert rate and the scheduler's row envelope):

  | Workload | TP4 c1 | TP4 c8 | TP3 c1 | TP3 c8 |
  | --- | --- | --- | --- | --- |
  | agent proxy, visible output | +4.75% (86.6 -> 90.7 tok/s) | +2.49% (207.6 -> 212.8) | +4.72% (69.3 -> 72.6) | +1.40% (162.0 -> 164.3) |
  | with hidden reasoning 1:1 as prose | +2.08% | +0.99% | +2.08% | +0.75% |
  | reproduce a file | +109% | +74% | +107% | +32% |

  Against #601 (+4.9% / +2.8%, TP3 alike): TP3 c8 is lower because with all eight TP3
  slots busy the envelope fits one wide request a step (below).

## The design as built

### The proposal: `copy_match` and `copy_select` (TileLang)

`vllm/v1/worker/gpu/spec_decode/tilelang_copy.py`, owned by `CopyDrafter`
(`copy_drafts.py`), run in `sample_tokens` right after DSpark's `propose()`:

- **`copy_match`**, a grid of (requests, 32) blocks. A block loads the context's last 256
  tokens (the key) into shared memory and scans its share of window ends `e` in
  `[16, total_len)`: a token is loaded only while the run of matching tokens reaches it,
  so most ends cost one coalesced load. Per level (16, 32, 64, 128, 256) a thread keeps its
  latest end; the block takes the maximum (warp shuffles, then the warps) and writes it.
  No atomics, so every launch gives the same sources.
- **`copy_select`**, a warp a request:
  1. the copy state's update from this step's verification of the request's pending copy
     (its admitted drafts, at most the dead-row cut): all accepted doubles the cap (from
     `initial_drafts`, to `max_drafts`), a rejection (a sunk draft zero included) returns
     it to 8; the accepted and judged drafts update decayed counts behind the estimate
     `(hits + 19) / (trials + 20)` (0.95 at a copy's start);
  2. the longest level with a source, its latest source, and the proposal:
     `min(cap, total_len - source)` tokens of the request's own history, each at the
     estimate; DSpark's drafts and confidences where no level has a source (which also
     returns the cap to 8, as #601's walk does after a DSpark step);
  3. the planned confidences and the allocation the scheduler lays out two steps on:
     the next proposal's drafts if this one is accepted in full,
     `min(2 cap, max_drafts, total_len - source)`, at least DSpark's 5.
- The history is `RequestState.all_token_ids` (UVA, as the runner holds it), read only up
  to `total_len`, whose last token is the step's last sampled one (draft zero with sink
  rejection). Nothing is indexed or kept between steps but the per-slot state.
- Both kernels have a fixed grid for their request count and no host synchronization;
  `test_kernels_replay_in_a_cuda_graph` replays them with new inputs bit for bit as eager
  launches. They are declared with the runner's other TileLang calls
  (`CopyDrafter.register_warmups`).

### The verification lookahead apart from DSpark's block

`speculative_config.copy_drafts` (`min_match`, `max_drafts`, `initial_drafts`).
`num_speculative_tokens` stays DSpark's block (5); `num_verification_tokens` is
`max_drafts` (16) and sizes everything a verified row touches:

- **the runner:** the request state's drafts, the rejection sampler and greedy
  verification, adaptive verification's buffers and kernel widths, the sink, the
  sampler's and structured outputs' rows, `post_update`'s sampled width, the kernel
  declarations; the speculator keeps the block;
- **DS4.1:** the reorder threshold (a 17-row request stays on the decode plan), the
  attention's decode capacity, the ratio-2 compressor's state ring (8 slots to 32: a
  17-row block in a ring of 8 overwrites the carry a rejected block's resume reads;
  `test_a_short_ring_loses_a_wide_blocks_carry` is the control), the DSpark context
  graphs and the sequence-parallel prefill floor (both above the envelope);
- **the scheduler:** acceptance statistics and per-position metrics;
- **unchanged without `copy_drafts`:** every one of these is `num_speculative_tokens`, the
  hash factors are unchanged, and the block's kernels and graphs are the ones declared.

### The rows: a step late, inside the envelope

- **The allocation.** The scheduler lays out step t + 2 before step t + 1 runs (the batch
  queue is two deep), so it uses the allocation step t's proposal reported
  (`ModelRunnerOutput.copy_draft_alloc`, copied to the host behind the proposal) and widens
  that request's -1 placeholders to it. This is #601's allocation lag exactly:
  `test_proposer_takes_the_designs_steps` drives the real kernels as the engine lays out
  steps and takes the same proposals, caps, allocations and tokens as variants.py's walk,
  step for step.
- **The envelope** (`VllmConfig.max_decode_rows`): every running request at DSpark's 6
  rows, plus one request's 11 extra. A copying request takes extra rows only from that
  room and from free slots no waiting request may fill this step, so an aligned prefill
  chunk always fits beside the decode rows (the scheduler's startup check). At TP4 c8 the
  room fits five wide requests a step; with every slot busy (TP3 c8, TP4 c16) one.
- **The capture ladder** ends at the envelope (TP4: 107 rows, TP3: 59; two more graphs a
  shape), and the batched-token budget grows by the same 11 tokens, so the long-prefill
  threshold, and with it every prefill chunk, stays the release's.

### Adaptive verification

- The device ranking (`assign_draft_budget`) takes the current proposal's confidences
  (copy and DSpark alike); the dead-row cut applies to copy rows too.
- The host budget reads confidences a step stale. For a copy it reads the planned row (the
  next proposal's width), kept beside the current one in one buffer, so one TP broadcast
  keeps both rank-wide (`test_planned_confidences_reach_the_budget_and_the_current_the_ranking`).
- Block verification compiled at 17 costs three times the work on a 6-row request (+12%:
  4.7 us at c1, 31 at c16). A step where no request was scheduled more than DSpark's
  block runs the TileLang kernels at the block's width into the lookahead's output
  (`kernel_steps`), the same tokens bit for bit
  (`test_a_step_without_a_copy_verifies_at_the_block`). Greedy verification needs no
  dispatch.

### Deviations from #601's design

- **No index and no atomics:** `copy_match` writes per-block partials that `copy_select`
  reduces (#601 sketched an `atomicMax`).
- **The scheduler's allocation** rides the async output (#601's commit 7 "D2H copy-state
  mirror") and is bounded by the envelope, which also counts waiting requests' slots.
- **The recipe keeps the long-prefill threshold** by raising the batched-token budget 11
  tokens (#601 lowered the threshold), so prefill chunking, and temperature-0 text on long
  prompts, stay the release's.
- **Added:** the planned confidences and block verification's block-width dispatch (both
  found while building).

## Tests (dgx2, `kernel-local`)

| Step | Files | Run 4 (63082e9b9) | next-candidate (run 3) |
| --- | --- | --- | --- |
| copy-draft scheduler | `tests/v1/core/test_copy_drafts_scheduler.py` (new) | 10 passed | |
| copy kernels and proposer | `tests/v1/spec_decode/test_copy_drafts.py` (new) | 19 passed | |
| verification at 16 drafts | `test_tilelang_rejection.py` with `VLLM_TEST_VERIFICATION_STEPS=16` | 667 passed | |
| verification at 5 | `test_tilelang_rejection.py` | 667 passed | 663 passed |
| adaptive and greedy verification, dead rows, inputs | 5 files | 118 passed, 1 skipped | 117 passed, 1 skipped |
| scheduler | `test_aligned_prefill_chunks.py`, `test_sink_draft_zero.py` | 69 passed | 69 passed |
| DS4.1 decode-row kernels | compress (4 new tests), linear, MoE, sparse MLA, mHC | 270 passed, 1 skipped, 1 xfailed | 266 passed, 1 skipped, 1 xfailed |

What the new tests hold:

- `copy_match` and `copy_select` equal numpy references written from the definition:
  contexts of 1 to 20,000 tokens across the grid's passes, vocabularies of 3, 64 and
  129,280 tokens, minimum matches 8, 16 and 24, slots in any order; the copy state step by
  step over a reproduced document, with dead-row cuts; what a step judges (only its
  pending copy; a prefill chunk, no verified draft, a sunk draft zero).
- **No leakage:** a slot's stale tokens past its new request's length, another request's
  text in the same batch, a slot's previous copy state, adaptive verification's planned
  row of a reused slot.
- **The design's walk** on four seeds of copy, edit and prose streams (above).
- **Temperature 0** through the real greedy verification kernels (above).
- **Graph replay** of both kernels.
- **The scheduler:** nothing widens without copy drafts; the first wide step is two steps
  after the report; four requests share one wide block, two widen both; waiting requests
  keep their slots' rows; preemption forgets the allocation; only async placeholders are
  laid out, re-laid each step.
- **The compressor ring** of 32 keeps every accepted prefix's carry under a 17-row block
  (and 128 under 65); a ring of 8 loses some (the control).

## Benchmarks (dgx2, gated)

`benchmarks/kernels/benchmark_copy_drafts.py` (CUDA graphs, L2 flushed before each replay):

Run 5 (the final head, 17:50 CET, pass in 21 s). The copy kernels, one step, UVA history
(us; gate: 1% of the release suite's step, 330 us at c1, 857 at c8, 1,714 at c16):

| Requests | 4K | 64K | 256K | 1M |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 20.4 | 30.7 | 51.0 | 124.6 |
| 8 | 28.7 | 45.1 | 83.9 | 227.1 |
| 16 | 32.5 | 55.0 | 112.6 | 449.5 |

Device-memory history is within 15% of these (UVA on GB10's unified memory streams nearly
as fast). The 20 us floor is the two launches and `copy_select`'s serial reduction of 32
chunks by 5 levels.

Verification of a step (us; gate: on DSpark's steps the lookahead is no slower):

| Step | Greedy, lookahead / block | Block verification, lookahead / block |
| --- | --- | --- |
| c1, 6 rows | 4.14 / 4.10 | 40.67 / 40.75 |
| c8, 48 rows | 4.18 / 4.10 | 172.35 / 172.43 |
| c16, 96 rows | 4.13 / 4.10 | 231.10 / 231.62 |
| c1 wide, 17 rows | 4.19 | 86.91 |
| c8 wide, 59 rows | 4.19 | 176.99 |
| c16 wide, 107 rows | 4.19 | 286.98 |

Before the block-width dispatch (run 2) block verification at the lookahead was 44.9 /
189.2 / 261.3 us on DSpark's steps.

[wide_verify_rows.py](wide_verify_rows.py), the routed layer as the model runs it (dispatch
to combine, production plan, TP4 rank shapes, cold weights), a request's rows over its own
24 (6 rows) or 48 (17 rows) experts, the step's experts their union; runs 3, 4 and 5:

| Streams | DSpark step | Copy step | Experts a layer | Added a step (40 layers) |
| ---: | --- | --- | --- | --- |
| 1 | 6 rows | 17 rows | 24 -> 48 | 19.0 / 18.9 / 18.1 ms |
| 8 | 48 rows | 59 rows | 155 -> 171 | 12.4 / 12.4 / 13.9 ms |
| 16 | 96 rows | 107 rows | 248 -> 255 | 6.5 / 4.9 / 6.9 ms |

Less FC1 and FC2's per-row time (2.2 us a row a layer, #601's wide-verify fits), an added
expert-layer costs 17.3-18.9 us (the runs' mean), against 17.1 at #601's 266.6 GB/s:
251.0 GB/s.

## The estimate ([estimate.py](estimate.py), [estimate.json](estimate.json))

#601's step model (`cost.py`), walks and workloads, DSpark alone against the go variant,
with three additions from the build: the copy kernels' measured time on every step (at
64K and 256K contexts; the proxy's streams average 147K tokens), the envelope in the
eight-stream draws (a wide allocation without room is laid out at 6 rows: a copy step
verifies 5 drafts), and the extended ladder. Copy rows are priced by the measured routes
of the edit tool call (the proxy) or each category's own; the bounds are #601's. The
eight-stream draws are 200,000 a point (at #601's 3,000 a gain moves by about 0.4 point
between seeds). Gains at 64K (256K within 0.1 point), the measured expert rate, then
#601's rate:

| Workload | TP4 c1 | TP4 c8 | TP3 c1 | TP3 c8 |
| --- | --- | --- | --- | --- |
| agent proxy, visible output | +4.75% / +4.84% | +2.49% / +2.60% | +4.72% / +4.82% | +1.40% / +1.45% |
| the same, ordinary-text bound | +4.44% | +1.94% | +4.39% | +0.97% |
| the same, independent routes | +2.62% | -0.53% | +2.46% | -0.71% |
| with hidden reasoning 1:1 | +2.08% | +0.99% | +2.08% | +0.75% |
| reproduce a file | +109% | +74% | +107% | +32% |
| the file with comments added | +70% | +55% | +69% | +22% |
| an edit tool call | +27% | +24% | +27% | +13% |

DSpark alone, tok/s at the measured rate: agent proxy 86.6 (TP4 c1), 207.6 (TP4 c8), 69.3
(TP3 c1), 162.0 (TP3 c8); the hybrid 90.7, 212.8, 72.6 and 164.3.

- **Against #601** (+4.9% at c1, +2.8% at c8, TP3 alike, at 3,000 draws): the copy
  kernels cost 0.05-0.1 point, the measured expert rate about 0.1, and the envelope about
  1.2 points at TP3 c8 (one wide request a step with all eight slots busy). On files at c8
  the envelope halves the gain at TP4 (eight requests copying at once) and more at TP3.
  Under independent routes, eight streams lose half a point.
- **The envelope is a recipe question:** two wide blocks a step at TP3 would cost the
  long-prefill budget 11 more tokens. The A/B below measures one.

## The cluster A/B (prepared, not run)

[make_specs.py](make_specs.py) writes every arm and spec from `config/cluster.json`, on
the copy build's image (`make_specs.py IMAGE [VLLM_TREE]`; written here with the
placeholder `spark-ds41f:19f2c20ed4d6-copy-drafts`, to be rerun with the built image and
its tree label):

- **Arms** (`control-*`, `copy-*`, `copy64-tp4`, `agentx-*`): the recipe's tp4-ring and
  tp3-triangle with the copy recipe's ladder (107 / 59 rows; 155 for 64 drafts), the
  budget 11 (59) tokens larger and the long-prefill threshold the release's. The control
  is DSpark alone on the same build; the copy arm adds
  `copy_drafts: {min_match 16, max_drafts 16, initial_drafts 8}`.
- **[hybrid-gate-tp4.json](hybrid-gate-tp4.json), [-tp3](hybrid-gate-tp3.json):** the
  determinism gate's extras (`token_determinism_text.py` with and without logprobs,
  `distinct_streams.py` c8) and [copy_identity.py](copy_identity.py) (the copy probe's
  prompts at temperature 0, alone and eight at once, thinking off and on) on DSpark
  alone, the copy arm and the copy arm after a restart. Pass: every digest of the copy
  arms equals DSpark alone's (and the previous release's, chunking being the release's).
- **[hybrid-measure-tp4.json](hybrid-measure-tp4.json), [-tp3](hybrid-measure-tp3.json):**
  the release, single-stream and sampled suites on copy, DSpark alone, copy at 64 drafts
  (TP4) and copy again; then the copy probe (thinking off) and `copy_identity.py
  --thinking` (the reasoning copies #601 could not see).
- **[agentx-tp4.json](agentx-tp4.json):** the AgentX replay (#574, which must merge first)
  at one session and eight, one hour each, copy and DSpark alone.

## Runs

1. **dgx2, `kernel-local`** (working copy `~/spark-lab/agents/copy-drafter/`, the image
   `spark-ds41f:19f2c20ed4d6-next-6`; bundles from [make_bundles.py](make_bundles.py)):
   - run 1 (17:19 CET): the copy kernels failed to compile (a Python loop TileLang read as
     a TIR loop) and the scheduler tests used a method async scheduling refuses; next-tests
     passed (123 s).
   - run 2 (17:26): copy kernels pass but one threshold; verification passes at both
     widths; the bench found block verification 12% slower at the lookahead and the routed
     script crashed on next-candidate's MoE workspace.
   - run 3 (17:34-17:47): tests pass (410 s), bench gates pass, next-tests pass (123 s).
   - run 4 (17:47-17:50): the final head 63082e9b9, tests pass (104 s, caches warm); the
     bench's gates all pass and both steps exit 0, which `kernel-local` reported as FAIL
     because the bundle read pytest's summary (now `"verdict": "status"`).
   - run 5 (17:50): the bench again, pass in 21 s.
   - The node had no `~/spark-request.json`; nothing else ran on dgx2.

   Outputs: [runs/](runs/) (`run4-copy-tests-dgx2.txt`, `run3-next-tests-dgx2.txt`,
   `copy-bench-dgx2.txt` for run 5, and runs 3 and 4's bench). Each bundle ran as:

   ```sh
   python3 scripts/lab.py kernel-local ~/spark-lab/agents/copy-drafter/bundles/copy-tests \
     --out ~/spark-lab/agents/copy-drafter/out/run4-copy-tests
   ```
2. **The estimate** (laptop, standard library, about 3 minutes; the draws are seeded):

   ```sh
   python3 -I experiments/2026-10-26-copy-drafter/estimate.py
   ```

## Caveats

- **Temperature 0 on the real model** is shown here only through its parts: verification
  commits the target's tokens whatever the drafts (kernel tests and the toy loop), and
  DS4.1's rows are bit-equal across a step's rows and a request's width (kernel tests).
  The cluster gate is the end-to-end check.
- **The estimate is #601's model** with measured inputs; no serving step has run a copy.
  The wide step's rows are priced at the grid's per-row price, its experts at the measured
  rate.
- **The envelope favours the first copying requests in running order** when it binds: the
  others verify 5 copy drafts. Throughput, not correctness.
- **64 drafts** builds and sizes (the compressor ring test covers 65 rows) but its kernels
  were not timed here; the A/B's `copy64-tp4` arm measures it.

## Files

- [vllm/](vllm/): the branch as patches 0533-0537 and [series](vllm/series).
- [make_bundles.py](make_bundles.py), [bundles/](bundles/): the dgx2 bundles (copy-tests,
  next-tests, copy-bench).
- [wide_verify_rows.py](wide_verify_rows.py): the routed layer at c1, c8 and c16 steps.
- [estimate.py](estimate.py), [estimate.json](estimate.json): the expected tok/s.
- [make_specs.py](make_specs.py), the arms and specs, [copy_identity.py](copy_identity.py):
  the cluster A/B.
- [runs/](runs/): the dgx2 outputs.
