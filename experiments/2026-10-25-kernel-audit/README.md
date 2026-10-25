# Kernel audit: unnecessary work in the kernels a DeepSeek V4.1 step launches

Owner, 2026-10-25, for the release after spark-ds41f-1025:

- check each kernel for unnecessary work, especially memory, contiguity and stride work
  left over from when neighbouring kernels ran in different engines;
- optimize the small kernels before fusing (fusion is a separate review);
- ports stay bit-identical, or at DeepSeek's reference semantics, and at least as fast
  as what they replace;
- bound each idea's gain from traces before building it.

This experiment ranks what the traces show, fixes the top findings that are bit-identical
and safe, and holds the lab specs that measure the fixes. **Nothing here has run on a
GPU yet**: a release window held the cluster, and no local torch exists. Every number
below is from existing traces; every fix still needs its window.

## Base and data

- **vLLM:** `tilelang-ports` 26382f5cf, the package files of candidate 6's image
  (`spark-ds41f:19f2c20ed4d6-next-6`, spark-ds41f-1025). The fixes are on the local
  branch `kernel-audit` (worktree `~/projects/vllm-kernel-audit`): 3cd92e872 and
  bb2fb46ee (fixes 1 and 2 below), 1072d662b (fix 1's loop and tests, after run 1), and
  8e596b329 (the mHC projection's fn prefetch), reverted by d9fa7d939 after run 1 measured
  it slower.
- **Traces:** request inventory 5
  ([experiments/2026-10-25-request-inventory](../2026-10-25-request-inventory/README.md)),
  `results/private/determinism/request-inventory-5/ports/`: TP4 ring, rank 0 (dgx1), the
  torch profiler with Python stacks. The profiled build is 9554a406c, 20 commits before
  26382f5cf. None of those 20 changes the kernels fixed here (`tilelang/metadata.py`,
  `tilelang/mhc.py`, `sparse_mla.py` are untouched; `attention.py` changes only its B12X
  helper plans).
- **Scenarios:** greedy (c1: one request, 1 + 5 drafted rows, 11 decode steps), mixed-c8
  (32 steps of one to eight requests), the 15,395-token prompt (one prefill step), and
  the other c1 request types for the sampler's kernels.

## Method ([kernel_shares.py](kernel_shares.py))

```sh
python3 experiments/2026-10-25-kernel-audit/kernel_shares.py \
  results/private/determinism/request-inventory-5 ports \
  --decode greedy,mixed-c8 --prefill long-prefill --top 120
```

- **A step** runs from one `execute_context_*` GPU annotation on the eager stream to the
  next: the target's forward, the head, the sampler and the drafter. The annotation alone
  covers only the target's forward; the head (`vocab_packed_projection`, 1.0 ms a call,
  twice a step), the sampler and DSpark's drafter are about 10% of the GPU time outside it.
- **Busy time** is the union of the step's kernel, copy and fill intervals on every
  stream, `l2_prefetch_kernel` excluded (it runs beside the layers by design).
- **Attribution is critical-path aware.** In every decode layer the routed MoE runs on a
  side stream (router GEMM, `router_partials`, dispatch, cast, grouped GEMMs) beside the
  shared expert on the main stream. The routed branch is the longer one (about 285 µs a
  layer at c1 against about 190 µs), so the shared expert has slack. Time with a
  side-stream kernel active is charged to it; other time is split evenly among the active
  kernels. A plain "exclusive time" measure shows the router at about 0%, though it sits
  on the critical path.
- **A ceiling** is a kernel's charged share of busy time: the most removing it entirely
  could save. **The estimate** is the part that is unnecessary work, from the kernel's
  structure. The traces carry Python stacks: host gaps are inflated, device durations are
  not, so every share is of device time.

Busy time a step: **c1 32.2 ms, c8 50.2 ms, the 15K prefill 2,142 ms.** The large kernels
are 85% of c1, 89% of c8 and 79% of the prefill: the routed GEMMs (46.6% at c1), the
all-reduce (13.3%), the dense MXFP8 GEMMs (10.7%), the head (6.4%) and the sparse MLA.
None of those is a small kernel. Idle time between graph kernels is 7.6% of a c1 step's
span and 6.1% at c8: about 1.5 µs a graph kernel, the host gaps between the step's graphs
included. That idle time is the fusion and launch-overlap review's ceiling, not this
audit's.

## Ranked findings

Ranked by estimated saving, decode first. Ceilings are each kernel's charged share of a
c1 decode step, a c8 step and the 15K prefill; estimates are of what the fix removes.

| # | Kernel (call site) | Unnecessary work | c1 | c8 | prefill | Estimate | Status |
|---|---|---|---:|---:|---:|---|---|
| 1 | `mhc_project_streams` (`tilelang/mhc.py` `_mhc_pre`: every sublayer entry, 86 a step) | Each stream's [24, 128] FP32 `fn` tile is loaded only after the previous stream's GEMM barrier, then split straight into shared memory: four exposed L2 round trips in every decode CTA (12 µs a call at 6 rows, 40 CTAs) | 3.18% | 2.62% | 5.09% | Was c1 0.5-0.8%. Measured: the prefetch made the kernel 0.5-2.9 µs slower at decode (run 1) | **Reverted** (d9fa7d939); diagnostics queued |
| 2 | `metadata_decode` (`attention.py` `_write_decode_metadata`) | At `--max-model-len` 1M and 256-token blocks it writes all 4,096 page-table columns of every row (a 32-step loop a thread in 12 CTAs at decode, 268 MB a layer at a 16K chunk). Readers use ceil(visible / page) of them | 0.70% | 0.41% | 1.19% | c1 ~0.45%, c8 ~0.25%, prefill >2% (measured: 1.2-1.9 µs a decode call, 1.35 ms a 16K prefill call) | **Fixed**, 3cd92e872 + 1072d662b; slower at a full 1M context, diagnostics queued |
| 3 | SP prefill's MoE exchange copy (a device copy: `moe_runner` `write` in `sp_prefill._project_exchange`) | Each sequence-parallel layer's combined MoE output (157 MB) is copied into the exchange's send and parts buffers rather than written there: 80 copies of 39 MB, 345 µs each | — | — | 1.29% | ~1% (three quarters, with the combine writing the peers' blocks in place) | Follow-up |
| 4 | Router chain on the routed side stream: `gemm_bf16_partials` (router gate, split-K), `router_partials`, `moe_dispatch`, `moe_swiglu_cast` (`tilelang/router.py`, `moe.py`) | On the critical path at decode. Each runs 1-4 µs above the launch floor (7.3, 6.2, 3.4, 3.0 µs at c1). Reading found nothing redundant. Candidates to measure: `router_partials`' serial shard loop, dispatch's per-expert padding loops | 2.92% together | 1.98% | 1.14% | Unknown until measured | Lab variant sweep |
| 5 | Metadata builders (`sparse_mla.py`: every KV-cache group, every step, eager) | 26 groups (23 target, 3 drafter), each with a `requests` launch (one 1.2 µs CTA) beside its `tokens` launch: 52 eager launches a step | 0.09% (+0.13% tokens) | 0.07% (+0.11%) | 0% | ~0.1% device; measured 2.8 µs a builder eagerly, ~70 µs a step | **Fixed**, bb2fb46ee |
| 6 | `split_columns` (`attention.py` `_gather_front`: SP prefill, layers without a compressor) | The gathered front is split into contiguous parts. Its q part (1,024 of 1,568 columns) feeds `wq_b`'s per-token cast, which reads strided rows (TileKernels takes a token stride) | — | — | 0.37% | ~0.25% | Follow-up |
| 7 | `metadata_pages` (`attention.py` `forward_mqa`: the indexer's page table) | Same as #2 for the index cache | 0.10% | 0.07% | 0.004% | ~0.04% | **Fixed**, 3cd92e872 |
| 8 | `zero_kv_blocks` (`model_runner.update_requests` → `zero_block_ids`) | Zeroes every newly allocated KV block (4.3 ms for the 15K prompt). The readers checked here (the sparse MLA, the indexer) never use an unwritten record: masked lanes are zero-filled or discarded | ~0 | ~0 | 0.20% | 0.2% | Not changed: vLLM's safety default, the owner's call |
| 9 | `compress_kernel` (`compressor.py`) | Each chunk's loads wait behind the previous chunk's ring-slot stores, and the norm weights behind the output stores: aliasing the compiler cannot rule out. 14.7 µs at 2 CTAs | 0.18% | 0.10% | 0.05% | ~0.1% | Follow-up |
| 10 | Other prefill device copies: `get_inputs_embeds` (79 MB), the SP indexer's gathered top-k copied into its buffer (`forward_mqa`, 32 MB x 3), `all_reduce_rank_order` (158 MB, plus an ATen copy), `all_to_all_rows` (49 MB) | Copies into persistent buffers a producer could write directly | ~0 | ~0 | ~0.2% | ~0.2% | Follow-up |
| 11 | The sampler's logits copy (`sample/sampler.py` `apply_sampling_params`, sampled rows) | Copies the step's logits (7.7 MB at c8) before applying the sampling parameters | 0 | 0.08% | — | ≤0.08% | Follow-up: check the copy's reader |
| 12 | Launch-floor kernels: `rope` (105 a step), `norm_forward` (96), `per_token_cast` (229), `cache_writer` (50), the sparse MLA's merge, the split-K reduces, `moe_combine`, the indexer's quantize and top-k merges, `mhc_post`, `mhc_collapse` | Nothing unnecessary inside: each is 1.5-4 µs at decode, at or near the floor. What they cost is their count | 4.94% together | 3.41% | 8.97% | — | Fusion review |
| 13 | `mhc_finalize` and the `per_token_cast` after it | The cast of `y` for the next GEMM is its own launch. Warp 0's Sinkhorn chain is inherent | 1.70% (+ part of 0.85%) | 0.98% | 0.92% | — | The `moe-overlap` session (folding `per_token_cast` into the finalize) |

The other c1 request types add the sampler's kernels, 0.07-0.61% of a step: most with
logprobs (`topk_log_softmax` 0.32% at 25.7 µs, `token_ranks` 0.10%, `topk_ids` 0.06%) and
with top-p and top-k (`topk_topp` 0.22% at 58.8 µs); 0.07-0.11% for sampled, penalties
and structured output. They were sized here, not audited line by line. Their remaining
ATen launches in these traces (an integer add, an index gather; 1.5-5.7 µs) were not
traced to their source; part of that glue was ported to TileLang after the profiled build
(bc55327b1).

## The fixes (vLLM branch `kernel-audit`)

Each commit is one change, with a bit-exact test and a gated benchmark against the
kernel it replaces. The replaced kernels are kept verbatim: in the benchmark, or in
`tests/kernels/deepseek_v41_baselines.py` when a test compares against them too.

### 1. Page tables over each row's live pages (3cd92e872, 1072d662b)

`decode_metadata` and `pages` write only the columns of a row's visible entries:
ceil(visible / page), at most the table's width, none for a row without a request. The
rest of the buffer keeps whatever it held. The page loop keeps the table width's trip
count (1072d662b): with the live pages as its bound it stopped unrolling, and a full 1M
context's row took 6.7 µs instead of 3.2 (run 1).

**Why this is safe.** The sparse MLA's kept lanes look up logical positions below the
row's indexed length, and the indexer's tiles positions below its extent, all inside the
live pages. Dense indexer tiles never straddle a page. A masked lane's lookup is clamped
into the table and discarded. Every consumer output is unchanged; only the table's dead
columns differ.

**Tests:**

- both kernels equal Triton's on every live column and leave the others untouched;
- new consumer tests poison every dead column with page ids no pool holds (negative ones
  among them) and point the masked lanes into them: every sparse-MLA route keeps its bits
  (`test_sparse_mla_reads_only_live_pages`), and so do the indexer's dense and candidate
  scores and selections (`test_indexer_reads_only_live_pages`).

**Benchmark:** `benchmark_deepseek_v41_metadata_pages.py --gate`. Decode c1 and c16 at 4K,
64K and 1M contexts, the 16K prefill chunk at the start and 48K into a prompt, both main
page sizes; and `pages` at the indexer's chunks.

### 2. Each metadata builder in one launch (bb2fb46ee)

`builder_kernel` writes `requests`' and `tokens`' outputs in one launch, the request
blocks first. The two parts read only the step's inputs, never each other's outputs.
`requests_kernel` and `tokens_kernel` are gone.

**Test:** one launch writes Triton's `_requests` and `_tokens` outputs exactly, over the
pair's packed steps, paged and ring geometries.

**Benchmark:** `benchmark_deepseek_v41_metadata_builder.py --gate`, eagerly (with the
host's launch cost: the builders run eagerly) and under graphs.

### 3. The mHC entry projection's fn tiles loaded ahead (8e596b329): reverted

`project_streams` loaded each stream's `fn` tile into registers first: the first stream's
while the token's columns load, each next stream's under the current stream's GEMMs.

**What run 1 measured.** The outputs kept their bits: every output (updated streams,
partials, both sums of squares, the collapse) matched the 26382f5cf kernel for every
entry, 1-1000 tokens and both token tiles, in 124 mHC tests and 73 model-forward tests.
But the benchmark's gate failed: the change was slower at every decode shape and at the
`pre` entry's prefill.

| Entry | Shape | Loaded ahead | 26382f5cf |
|---|---|---:|---:|
| `post_pre` | decode c1 | 6.75 µs | 5.95 µs |
| `post_pre` | decode c8 | 19.1 µs | 16.8 µs |
| `post_pre` | decode c16 | 32.5 µs | 28.5 µs |
| `pre` | decode c1 | 6.50 µs | 5.96 µs |
| `pre` | prefill 16K | 5,692 µs | 5,355 µs |

`broadcast` and the four-part entry were level. d9fa7d939 reverts it.

**Why, so far a hypothesis.** In isolation `fn` is hot in L2, so there was little load
latency to hide. And the register tile's split stored to shared memory four scalars at a
time, where the `T.Parallel` split stores 16-byte vectors. The diagnostics bundle (Runs)
times four other forms against the 26382f5cf kernel to separate the two effects. A form
returns only if it is no slower anywhere:

- B, this change;
- C, loaded ahead with vector stores;
- D, the scalar split without loading ahead;
- E, loaded ahead into a fragment and split by `T.Parallel`.

### Measured so far (dgx4, kernel benchmarks in isolation)

- **Fix 1:** 1.2-1.9 µs a decode call at 4K-64K contexts. At the inventory's 15 calls a
  step that is 0.06-0.09% of a c1 step in isolation. The traced serving kernel ran at
  15 µs against 3.3 µs here, so serving's saving depends on what slows it there; the
  profile run measures it. A 16K prefill chunk takes 70 against 1,420 µs: at 38 calls a
  prefill, about 51 ms, 2.4% of the 15K prefill. Still slower at a full 1M context (6.7
  against 3.2 µs a call).
- **Fix 2:** 1.3 µs a builder under graphs, 2.8 µs eagerly: about 70 µs a decode step.
- **Fix 3:** reverted.

## Not changed, and why

- **#3, SP prefill's MoE exchange copy.** Removing it means the fused-MoE runner
  allocates its output in the exchange's layout, or hands the combine the destination
  blocks. That reaches beyond the model's kernels into `fused_moe/runner`; it needs its
  own change and test. The fusion survey (draft, `experiments/2026-10-25-fusion-survey`)
  ranks running the exchange under the combine at the same boundary; one change can do
  both.
- **#4, the router chain.** Reading found no redundant work. The next step is a variant
  sweep in the lab (each candidate timed against the current kernel, the
  tilelang-completion "variants" bundles' method): the shard loop unrolled in
  `router_partials`, dispatch's padding stores spread over threads. Each is bit-identical
  by construction (the same adds in the same order).
- **#6 and #10, glue copies.** Each needs its consumers' strides checked one by one.
- **#8, `zero_kv_blocks`.** This is vLLM's default protection against stale cache memory.
  The safety guards stay unless the owner decides otherwise.
- **#9, `compress_kernel`.** Low ceiling (0.18%); a candidate for the same variant sweep.
- **#12 and #13** belong to the fusion review (the survey above ranks the layer's
  small-kernel budget and PDL, which no serving kernel requests at 26382f5cf) and the
  `moe-overlap` session.

## Coordination

The `moe-overlap` session (branch `moe-overlap`) changes the MoE expert weights' cache
hints and folds `per_token_cast` into mHC's finalize. With fix 3 reverted, the audit
changes nothing in `tilelang/mhc.py`, so the two branches no longer meet. A form of fix 3
that returns would edit only `project_streams`.

## Lab plan (next week's windows)

[make_audit_bundles.py](make_audit_bundles.py) writes everything from the vLLM branch:

```sh
python3 experiments/2026-10-25-kernel-audit/make_audit_bundles.py ~/projects/vllm-kernel-audit
```

It writes `bundles/audit-*`, `bundles/serving/overlay`, `audit-tp4.json`,
`prof-candidate6.json` and `prof-audit.json` (BASE 26382f5cf, HEAD `kernel-audit`). Rerun
it after any change to the branch.

1. **[audit-kernels.json](audit-kernels.json)**, run `kernel-audit-kernels`: two bundles
   in parallel on candidate 6's image, with fixes 1 and 2 mounted in each.
   - `audit-attention` on dgx1: the metadata and attention kernel tests, the layer's
     declaration tests, then the live-page and builder benchmarks.
   - `audit-mhc` on dgx2: the mHC tests and the CED and SP model-forward tests.
   - Pass: every test passes and every gate holds.
2. **[audit-serving.json](audit-serving.json)**, run `kernel-audit-serving`: the serving
   check at tp4-ring, arms audit / candidate 6 / audit / candidate 6.
   - The boot's warm-up forwards and graph captures, then the checks' requests, launch
     every changed call; the frozen kernel cache refuses an undeclared one by name.
   - Outputs: temperature-0 determinism with and without log probabilities. The digests
     must equal candidate 6's, and each arm must be identical within itself
     ([tables_arms.py](tables_arms.py)).
   - Rates: c1 (12 distinct prompts), c8 and c16 distinct.
3. **[audit-profile.json](audit-profile.json)**, run `kernel-audit-profile`: the
   inventory's decode and prefill workloads under the torch profiler, candidate 6 first,
   then the audit ([profile_decode.py](profile_decode.py) and
   [profile_prefill.py](profile_prefill.py) are the inventory's). The profiler runs
   without Python stacks, which inflated inventory 5's traces and host time. The costs
   step compares each kernel's time against candidate 6: the fixes' measured saving at
   serving shapes, against the estimates above.

Run them in that order; stop at the first failure.

## Runs

dgx4, single node, opened for GPU testing by the coordinator on 2026-10-25. Commands run
on dgx4 from a checkout of this branch under `~/spark-lab/agents/kernel-audit/`:
`scripts/lab.py kernel-local experiments/2026-10-25-kernel-audit/bundles/<bundle> --out
~/spark-lab/agents/kernel-audit/out/<bundle>`, image `spark-ds41f:19f2c20ed4d6-next-6`
(`sha256:82b071b57485`).

### Run 1: `audit-attention`, 19:00-19:08 CET (branch e88dba66, vLLM 8e596b329): FAILED

Output `~/spark-lab/agents/kernel-audit/out/audit-attention/` on dgx4. Bundle exit 1.

| Step | Exit | Result |
|---|---|---|
| metadata and attention kernel tests | 1 | 134 passed, 10 failed, 1 xfailed. The sparse-MLA and indexer poison tests passed. |
| attention layer tests | 0 | 4 passed |
| live-page benchmark | 1 | Gate failed at c1 with a full 1M context; faster everywhere else |
| one-launch builder benchmark | 0 | Gate passed |

Root causes, both fixed in vLLM 1072d662b:

- **The 1M regression is the kernel's.** The page loop took the live pages as its
  bound, so it stopped unrolling and a long row's 32 table loads ran one after another.
  It keeps the table width's trip count again, with the live pages as the predicate.

  | decode_metadata (µs a call, both page sizes) | live pages | every column |
  |---|---:|---:|
  | c1, 4K-64K context | 1.7-1.9 | 3.3-3.4 |
  | c16, 4K-64K context | 1.4-1.9 | 3.7-4.1 |
  | c1, 1M context | 6.7 | 3.2 |
  | prefill chunk of 16,384 rows at 16K | 37.8 | 1,405-1,415 |
  | same chunk 48K into a prompt | 120-124 | 1,407-1,419 |

  `pages`: 1.5 against 1.5 µs (c1), 2.9 against 3.9 (c16), 5.7 against 7.5 (a 256-row
  prefill chunk). Every benchmark shape's live columns were equal before timing.
- **The 10 test failures are the tests'.**
  - `test_pages[*-lengths0]` (6): my assertion that every step has a live page is wrong
    for a one-token step, whose tested rows (from offset 2) are all padding. The
    assertion is gone.
  - `test_decode_metadata_equals_chunk_and_pages[*-lengths1]` (4, seed-dependent): the
    test built its main rows' requests by flipping a `torch.empty` buffer, so most rows
    held uninitialized request ids and both kernels read past the table; the kernels that
    wrote every column happened to agree on it. The main rows now hold the step's
    requests in reverse, and -1 past them.

### Run 1: `audit-mhc`, 19:08-20:02 CET (branch e88dba66, vLLM 8e596b329): FAILED

Output `~/spark-lab/agents/kernel-audit/out/audit-mhc/`; most of the hour was the
queue for the GPU lock.

| Step | Exit | Result |
|---|---|---|
| mHC tests | 0 | 124 passed, `test_project_streams_bits_match_26382f5cf` among them |
| model-forward tests | 0 | 73 passed |
| mHC projection benchmark | 1 | Gate failed: slower at every decode shape and at `pre`'s prefill (fix 3's table) |

Root cause: fix 3 itself is slower. It is reverted (vLLM d9fa7d939), and the
diagnostics bundle times the other forms.

### Run 2: `audit-attention`, 19:16-20:03 CET (branch cf191082, vLLM 1072d662b): FAILED

Output `~/spark-lab/agents/kernel-audit/out/audit-attention-2/`, from a second checkout
(`spark-ds41f-r2`), so the running bundle's files were untouched.

| Step | Exit | Result |
|---|---|---|
| metadata and attention kernel tests | 0 | 144 passed, 1 xfailed |
| attention layer tests | 0 | 4 passed |
| live-page benchmark | 1 | Gate failed again at c1 on a full 1M context: 6.7 against 3.2 µs |
| one-launch builder benchmark | 0 | 2.0-2.4 against 3.3-3.7 µs under graphs, 9.9-10.0 against 12.5-12.9 µs eagerly |

The page tables are otherwise as in run 1:

- decode at 4K-64K: 2.0-2.4 against 3.3-4.0 µs;
- a 16K prefill chunk: 67-70 against 1,419-1,431 µs;
- the chunk 48K into a prompt: 120-123 against 1,426-1,436 µs.

The fixed trip count did not explain the 1M case: 1072d662b's loop is as slow there as
the dynamic one, and costs about 0.3 µs at short contexts. Each extra iteration costs
about 0.14 µs, as if the iterations ran one after another. The diagnostics bundle times
the loop's forms and writes their CUDA sources.

### Diagnostics: `audit-diagnostics`, 20:20-20:28 CET (branch a2fe7bc6, image's 26382f5cf kernels): PASSED

Output `~/spark-lab/agents/kernel-audit/out/audit-diagnostics/`; CUDA sources in
`~/.cache/spark-lab/compile/kernel-audit-diagnostics/` on dgx4.
[bundles/audit-diagnostics](bundles/audit-diagnostics) runs on the image's own
26382f5cf kernels, and [make_diagnostics.py](make_diagnostics.py) writes its variant
modules from 26382f5cf's sources. Every form's outputs equal the 26382f5cf form's (the
live columns; the projection's every bit).

**decode_metadata's page part**, µs a call (main page 256):

| Form | c1 4K | c1 1M | c16 64K | prefill 16K |
|---|---:|---:|---:|---:|
| old (every column, 26382f5cf) | 2.86 | 2.84 | 3.63 | 1,436 |
| dyn (3cd92e872) | 1.29 | 6.36 | 1.36 | 40.2 |
| static_var (1072d662b) | 1.67 | 6.28 | 1.91 | 70.2 |
| static_expr | 2.32 | 6.91 | 2.59 | 109.5 |
| select | 5.01 | 5.17 | 5.15 | 148.0 |

Root cause of the 1M case, from the generated CUDA: the live-page forms bind
`page_id = main_table[req, col]` as an alias, so `T.if_then_else(page_id > 0, page_id,
-1)` emits the load twice. The second load sits inside the branch on the first's value:
two dependent round trips a column. The old form's load sits inside its select and is
bound once (`condval_10`). vLLM a71368f70 loads each entry into a `T.alloc_var` in both
`decode_metadata` and `pages`. The same aliasing is worth a search through the other
TileLang kernels.

**project_streams**, µs a call:

| Entry and shape | A (26382f5cf) | B (ahead, scalar split) | C (ahead, vector split) | D (scalar split only) | E (ahead into a fragment) |
|---|---:|---:|---:|---:|---:|
| post_pre decode c1 | 5.48 | 6.27 | 5.54 | 6.24 | 5.54 |
| post_pre decode c8 | 16.41 | 18.91 | 16.88 | 19.12 | 16.82 |
| post_pre decode c16 | 28.27 | 32.39 | 28.67 | 32.14 | 28.65 |
| post_pre prefill 8K | 3,799 | 3,850 | 3,790 | 3,829 | 3,818 |
| pre decode c1 | 5.45 | 6.00 | 5.42 | 6.00 | 5.42 |
| pre decode c8 | 15.08 | 16.76 | 15.10 | 16.68 | 15.08 |
| pre decode c16 | 25.93 | 28.04 | 25.40 | 28.00 | 25.40 |
| pre prefill 8K | 2,663 | 2,838 | 2,677 | 2,825 | 2,654 |

- **The scalar split is the whole slowdown.** B and D are equally slow, and the sources
  bear it out: A's `post_pre` source has 33 `float4` accesses, B's 12.
- **Loading ahead gains nothing in isolation.** C and E sit within about ±2.5% of A:
  0.4-0.5 µs slower at `post_pre` c8 and c16, 0.5 µs faster at `pre` c16. With `fn` hot
  in L2 there is no latency to hide.
- Fix 3 stays reverted.
