# vLLM patch stack

Base: `local-inference-lab/vllm@04c30fa98e7917fee0a24c739ea503ce1e22538d`
(`integration/karmic-kraken-beta`).

- `0001-engram-projection-tp-padding.patch` lets `engram_config.projection_tp`
  shard DeepSeek V4.1's Engram WKV projection when its 25600-row output does
  not divide by the TP size (TP3): the output is padded to whole 32-row
  block-FP8 scale blocks per rank, the last rank's missing checkpoint rows are
  zero-filled through `allow_tp_padding`, and the gathered output is sliced
  back. On the r2 base it yielded tree `90fdd043`. Evidence:
  `experiments/2026-09-24-improvement-leads/`. Upstream status: candidate for
  Local Inference Lab, not submitted.

- `0002-engram-async-disk-rows.patch` (`SPARK3_ENGRAM_ASYNC=1`) reads disk
  Engram rows on a reader thread while the forward graph launches, instead of
  finishing both layers' reads before the launch. The main thread queues the
  row decode on a side stream behind a host gate and resets a per-layer ready
  flag; the reader thread only waits for the row IDs, reads the rows, and
  releases the gate (also on failure); each Engram layer waits on its flag,
  captured into the CUDA graph. Graphs are identical with the path on or off.
  Quality: the rows and their decode are unchanged, only their timing; a
  20 ms injected read delay lengthened steps by 27 ms with quality and
  acceptance unchanged, showing the gate holds. Applying 0001-0002 to the base
  yields tree `033fd0cc` on the r3 base. Evidence:
  `experiments/2026-09-24-three-leads/` and
  `experiments/2026-09-25-engram-async-ab/`. On this base the Engram overlap
  (`VLLM_DS41_ENGRAM_OVERLAP`, on by default) is the base's own version of
  the same idea, and saves less on three Sparks
  (`experiments/2026-09-27-lil-head`); the two are exclusive, and the path
  raises unless `VLLM_DS41_ENGRAM_OVERLAP=0`. Upstream status: candidate for
  Local Inference Lab, not submitted.


- `0003-dsml-optional-string-attribute.patch` keeps DSML tool parameters
  that omit `string="true|false"`; the V4 and V4.1 parsers dropped them
  silently. A missing attribute is treated like `string="false"` (JSON,
  falling back to the literal). Port of vllm-project/vllm#56271 (merged as
  `39e33db7`), Python parts only. Quality: tool-call arguments only. Tests:
  the PR's parser tests (128 pass in the r3 image).

- `0004-mm-block-hash-window.patch` hashes every multimodal item that
  overlaps a block completed by generated tokens; the incremental hasher
  started at the last item only, so a block holding two images hashed as if
  it held one. That lost the next turn's prefix-cache hit and let two
  conversations that differ only in an earlier image of that block share
  its KV. Port of vllm-project/vllm#51694 (merged as `e6c07ea5`). Tests: the
  PR's test fails on r3 and passes with the patch.

- `0005-dspark-pinned-cost-curves.patch` (`SPARK3_DSPARK_COST_DIR`) saves TP
  rank 0's adaptive-verification cost curves for a shape set and reuses them
  on later boots with the same shapes, so verified draft counts no longer
  follow one boot's startup timing. `SPARK3_DSPARK_PROFILE_REPLAYS` raises
  the replay count of the pinned profile. Every rank still profiles, so
  collectives stay matched. Output unchanged. Upstream status: candidate
  for Local Inference Lab, not submitted.

- `0006-dspark-marginal-verification-rule.patch`
  (`SPARK3_DSPARK_VERIFY_RULE=marginal`) chooses the draft budget that
  maximizes expected tokens minus the achieved rate times modeled cost,
  instead of each step's expected tokens per millisecond. The rate is an
  exponential average per request count, floored at the no-draft rate, and
  derived identically on every rank. Output unchanged: the target verifies
  every kept draft. Upstream status: candidate, not submitted.

- `0007-dspark-v41-gathered-markov-bias.patch` admits the V4.1 drafter to
  the existing gathered Markov-bias path (`dspark_draft_topk` in the
  speculative config): the bias is added to the top-k base candidates only,
  instead of projecting onto the whole vocabulary at every draft position.
  Only drafts can change, so acceptance can move but output cannot. Off
  unless `dspark_draft_topk` is set. Upstream status: candidate, not
  submitted.

- `0008-dspark-draft-trace.patch` (`SPARK3_DSPARK_TRACE=<path>`) makes TP
  rank 0 append each step's sampled tokens, verified draft counts, drafts
  and raw confidences as JSON lines, for pricing verification policies
  offline at temperature 0. It copies tensors to the host every step: replay
  runs only. Output unchanged.

- `0009-dspark-profile-distinct-tokens.patch`
  (`SPARK3_DSPARK_PROFILE_TOKENS=random`) runs adaptive verification's
  startup cost profile on a fixed pseudo-random token sequence, identical on
  every rank, as real rows rather than padding, and on its embeddings where
  the model takes inputs_embeds. Dummy rows are otherwise padding, which the
  MoE routers skip, so the profile leaves out routed-expert work and prices
  a verified row at about 0.4 ms against several milliseconds in real
  steps. Output unchanged. Upstream status: candidate, not submitted.

- `0010-dspark-dead-verification-rows.patch`
  (`SPARK3_DSPARK_DEAD_ROWS_TAU`) marks verification rows past each
  request's confidence cut (survival of this step's drafts below the
  threshold) as padding inside the draft combine, so routed MoE reads no
  expert weights for them, and commits at most one token past the last live
  row. `SPARK3_DSPARK_VERIFY_RULE=all` schedules every draft so the cut alone
  decides. Off by default; output unchanged. Upstream status: candidate, not
  submitted.

- `0011-dspark-dead-rows-by-ratio.patch` (`SPARK3_DSPARK_DEAD_ROWS=ratio`)
  decides dead rows by cost instead of a fixed cut. A second startup profile
  prices live verification rows on real (routed) rows; each step the host
  stages the cost of every live-draft count inside its budget, and the draft
  combine keeps live the drafts most likely to survive, as many as maximize
  expected tokens per millisecond. Off by default; output unchanged.
  Upstream status: candidate, not submitted.

- `0012-dspark-vocab-parallel-greedy.patch`
  (`SPARK3_DSPARK_VOCAB_PARALLEL`) keeps greedy DSpark drafting sharded by
  vocabulary: the Markov projection is sharded like the LM head, each rank
  reduces its shard to a (max, index) pair, and only the pairs are gathered.
  The drafts are unchanged. Off by default; output unchanged. Upstream
  status: candidate, not submitted.

- `0013-dspark-main-proj-column-parallel.patch`
  (`SPARK3_DSPARK_MAIN_PROJ_TP`) column-shards the drafter's replicated
  `main_proj` with 0001's padded column-parallel linear and gathers the
  output. Off by default; output unchanged. Upstream status: candidate, not
  submitted.

- `0014`-`0017` (`deepseek-v41-prefill-sp-*`) add sequence-parallel
  prefill. For prefill forwards above the threshold,
  each decoder layer's all-reduces become dim-0 reduce-scatters, and each
  rank runs the row-wise work between them (hyper-connection mixes, norms,
  the Engram gate, the residual) on a third of the rows:
  - 0014 holds the per-forward state and reduce-scatters the embedding;
  - 0015 carries the decoder dataflow (attention and MoE gather their
    inputs and reduce-scatter their outputs);
  - 0016 covers the CED encoder layers and gathers the carried rows before
    the boundary layer;
  - 0017 runs the attention front (`fused_wqa_wkv`, norms, index weights) on
    local rows and gathers its narrower product.
  Decode never engages. With 0014-0017 alone the threshold is
  `SPARK3_DS41_PREFILL_SP_MIN_ROWS` (unset is off); output differs only by
  the reduce-scatter's summation order. Upstream status: candidate, not
  submitted.

- `0018-deepseek-v41-prefill-sp-transport-threshold.patch` removes that
  setting. A forward runs sequence-parallel when it is prompt processing
  (larger than every decode forward) and its padded hidden-state message
  exceeds what the one-shot RoCE all-reduce takes, so that the
  reduce-scatter really scatters: below that SP only adds two all-gathers
  per layer. With the 2 MiB RoCE limit and 5120 bf16 hidden rows that is
  205 tokens at TP 3. The RoCE communicator exposes its limit as
  `all_reduce_max_bytes`. Upstream status: candidate, not submitted.

- `0019-custom-op-fill-defaults-one-read.patch` makes `vllm/env_override.py`
  install a `torch._library.utils.fill_defaults` that reads
  `schema.arguments` once. torch's version reads it once per argument
  (quadratic), from the ADInplaceOrView kernel of every mutating
  `torch.library.custom_op`; B12X's 64-argument dynamic MoE launch spent
  about 0.8 ms of host time per call there (873 us against 48 us). Same
  arguments returned; output unchanged. Upstream status: the bug is still in
  PyTorch main; not submitted.

- `0020-spec-decode-dead-row-kernel-warmup.patch` compiles the DEAD_ROWS variants of the draft combine and the
  sampled/rejected-count Triton kernels at warmup. Kernel warmup runs with
  adaptive verification off, so the first served request compiled them (568 ms
  to first token against about 210 ms afterwards). Output unchanged. Upstream
  status: candidate, not submitted.

- `0021-worker-display-carveout-kv.patch` backs the KV cache with the GB10 display
  carve-out when `SPARK3_KV_DISPLAY_CARVEOUT=1`: a DRM dumb buffer from the
  firmware scanout reserve, exported as a dma-buf and imported with
  `cuImportExternalMemory` (about 95% of ordinary memory bandwidth). The
  worker refuses to start if the carve-out cannot hold the backing; startup
  KV caches under 256 MiB (profiling, B12X preparation) stay in ordinary
  memory. Output
  unchanged; inert without the variable. Tests:
  `tests/v1/worker/test_display_carveout.py`. Upstream status: candidate, not
  submitted.

Applying 0001-0021 to the base yields patch head `d0d429a6` and tree
`2f8e61c6`.
Applying 0001-0020 to the base yields patch head `58bff2b1` and tree
`bf8910a6`. 0001-0019, the r5h and r5i images, give patch head `c42e75cf`
and tree `17f5431d`. 0001-0017, the r5g image, give patch head `6151f609` and tree
`5e088694`. 0001-0013, the r5f image, give patch head `0a00c7b3` and tree
`73a843bb`. 0001-0011, the r5e image, give patch head `138b562f`
and tree `7b839dc1`. The r5c image carried an earlier 0009 that moved
`_dummy_run`'s decorators onto its new helper; its 0001-0009 gave patch
head `d7234353`, tree `f250542a` (on the r4 base `01f1b874`: patch head
`84f0b5dc`, tree `ee3a0fd4`).
