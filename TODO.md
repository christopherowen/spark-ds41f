# TODO

Hints for the next improvement runs. These notes are not configuration authority:
`config/cluster.json`, the current baseline manifest and `docs/current-state.md`
describe what runs. Every figure here was measured on this stack; each idea
enters as an ordinary experiment under `experiments/` and is promoted only by the
owner.

## Goal

Step time and the bench matrix are proxies. The outcome that matters is
end-to-end throughput of the production agent workload: findings per hour on a
fixed scope and tool policy. Record the server side of every agent run with
`bin/spark3 workload --json <path> -- <agent command>` (read-only; it sends no
requests) and judge candidates on that whenever a change could shift the
result.

A first 10-second window of live agent traffic on r5j (2026-09-28):

| Measure | Value |
|---|---|
| Concurrent requests | 2 |
| Prompt length | 64-100K tokens, 98-99% served from the prefix cache |
| Output per request | 350-500 tokens |
| Drafts | 1.89 accepted of 3.40 verified per step; by position 0.72, 0.55, 0.48, 0.46, 0.42 |
| Time to first token | ~0.9 s |
| Inter-token latency | p50 63 ms, p99 ~0.1 s |
| Engine steps carrying prefill | 0.7% |

Acceptance on this traffic is far below the bench's code answers (3.1 accepted
per step), so the bench overstates what long drafts buy. Capture whole scans
before tuning anything on this basis.

## Current reference

`manifests/benchmarks/2026-10-02-karmic-kraken-r5o-64k.json`: the 64 KiB kernel
with the memory saver, 512K limit, 3.5 GiB of KV, 2,845,543 tokens. Aggregate
tok/s across all streams, temperature 0, 256 output tokens, reasoning on:

| Workload | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Code, reasoning on | 64.8 | 90.7 | 135.6 | 195.8 |
| Prose, reasoning on | 52.8 | 79.4 | 117.9 | 171.2 |

Single-stream steps are 40.8 ms on prose and 45.4 ms on code. This reference
has no answer-only rows; `2026-09-29-karmic-kraken-r5l.json` has the last ones.

- **Time to first token (short prompts):** 0.20-0.21 s at one stream and
  0.54-0.60 s at eight.
- **Source-text prefill:** 2.2k tok/s at 1K, 3.8k at 32K and 64K, 3.7k at
  256K. A 32K prefix takes 6.55 s cold and 0.27 s warm (99% hits).
- **Long context:** four ~485K-token prompts coexisted with zero preemptions
  (`experiments/2026-10-02-memory-saver-capacity`).
- **Quality:** `experiments/2026-09-29-r5k/consistency.py` measures the
  decode-versus-prefill logprob gap on greedy generations (r5k: 0.0506 mean,
  2.84% argmax disagreement); use it for any change that touches decode-only
  state. The r5k figure predates the r5n/r5o fence fixes; re-measure it on r5o
  before using it as the reference.

A step-time saving is a fixed cost per step. At one stream it converts almost
fully into tok/s. At eight streams the step is longer (up to 48 verified rows,
more distinct experts read), so the same saving is a smaller share. Report one
and eight streams for every decode change.

## Running a round

- **Screen lean, promote thorough.** One boot per arm:
  `bin/spark3 bench --suites quality,decode --decode-cases prose,code,prose-nothink,code-nothink --concurrency 1,8 --min-samples 3 --max-samples 3`
  (about 8 minutes with the boot). Run the full matrix only for a promotion
  candidate.
- **Pin the verification cost table.** Same-TP arms share one pinned
  adaptive-verification table (`SPARK3_DSPARK_COST_DIR`, vLLM patch 0005), with
  its hash recorded, so verification policy is fixed. TP3 and TP4 need their
  own tables.
- **Start cool.** `bench` and lab jobs bring every node below 55 °C before
  measuring (`--cool-below`, `--cool-timeout`). Long prefills still reach
  83 °C on dgx1 and dgx2 (see Four nodes).
- **Separate stable measures from output-dependent ones.** Temperature-0
  outputs drift within one boot, even at one stream, so anything that depends
  on the generated text varies between samples: accepted drafts per step,
  prefix-cache hits, output length. Time per verification step is stable to
  about ±0.5 ms with three samples; acceptance is not. For changes that move
  acceptance, replay a recorded draft trace (patch 0008 `SPARK3_DSPARK_TRACE`
  with `experiments/2026-09-26-dspark-policy/replay.py`) or raise the sample
  count. Eight-stream aggregates swing 3-5% between samples. Across 231
  recorded runs, 41% of three-sample one-stream A/A pairs differ by more than
  3% in tok/s (24% at eight streams); gate on step time.
- **Boot-to-boot noise can look significant.** The 2026-09-27 `base` and
  `nol2` arms ran identical code (the L2 prefetch was not running) and read
  1.2 ms per step (-2.4%) apart, "significant" with two boots per arm. When a
  decision hinges on 2.5% or less, add a null arm: the same configuration on
  a separate boot.
- **Control on the same day, with the same protocol.** A lean run and a full
  matrix of the same image differ by about 1 ms per step at one stream and 3-7%
  at eight streams.
- **Python-only vLLM changes need no image build to screen.** Bind-mount the
  changed files from the checked patch branch over the last candidate image via
  `container.mounts` (the image imports vLLM from `/opt/spark3/candidate/vllm`).
  Build only for promotion.
- **Queue arms unattended** with a `setsid -f` chain, `ssh -n` and logs in
  `~/tl-logs`. Keep other GPU work off the nodes while an arm is timing, and
  keep bench load off the service while the agent workload is running.
- **Check every candidate's boot log for silent fallbacks** (`disabled`,
  `fallback`, `compile failed`) and for any JIT compilation after readiness.
  The L2 weight prefetch was advertised from r5c but failed to compile until
  r5i, and only the boot log said so. Comparisons made within r5c-r5h are valid
  relative to each other; with/without-prefetch conclusions from that range are
  void (corrected in `experiments/2026-09-27-lil-head` and
  `experiments/2026-09-27-r5-engram-async`). Settings chosen there were tuned
  without the prefetch and never re-tuned with it: dead-row survival cut 0.2,
  five drafts, adaptive cost scale 2.0.
- **Shared packages move forward only.** When components disagree on a shared
  runtime package, patch the lagging pin forward. The image build checks every
  CuTe DSL consumer's requirement; do the same for any new shared package.

## Memory and safety

- dgx1 hosts rank 0 and the API server, so it is always the tightest node:
  5.83 GiB minimum MemAvailable with four ~485K-token contexts on the 64 KiB
  profile. The memory saver's gain went into 1.3 GiB more KV and the 512K
  limit. Check its headroom before anything else, and never lower the guards
  (5 GiB at startup, 3 GiB steady).
- Known to cross the startup guard on dgx1: 8,192 batched tokens, draft TP 1,
  full in-engine B12X autotune. That was measured on 2026-09-24, before the
  display reserve and the 64 KiB memory saver; re-test on TP4, or on TP3 by
  giving back some KV.
- Profiling: never use the torch profiler with `with_stack=True` on dgx1;
  stopping it dropped MemAvailable to 1.5 GiB and the guard stopped the
  service. `py-spy record` hangs on the workers; a loop of
  `py-spy dump --nonblocking` works as a sampler. Workers rename themselves
  (`VLLM::Worker_TP0`), so select processes by container, not by name.
- Kernel, sysctl, boot, firmware, package, display and network changes are the
  owner's decisions, and any host change goes to every node. Stop the
  service before rebooting a node, and don't restart inference while it is in
  use.
- **Carve-out integrity check is a debug mode.** Patch 0026's
  `SPARK3_DISPLAY_CARVEOUT_CHECK_SECONDS` stays unset in production; set it
  (for example to 60) only to chase a suspected carve-out fault. It costs
  about 20 ms per check and nothing when unset. Serving-path changes for
  quality must not add computation.

## Mixed prefill and decode

The agent workload interleaves long prefill bursts with decode phases, and this
is the gap between the bench and production. The bench measures decode and prefill
separately.

- **What happens today.** The batched-token budget is 4,096. A step that
  carries a full prefill chunk takes about 1.1 s, and every decoding stream
  waits for it. A cold 61K prompt is about 15 such steps, so other requests
  decode at roughly one step per second for 16 s.
- **Where it bites.** Cached prompts avoid most of it: in the live window above,
  98-99% of prompt tokens came from the prefix cache and 0.7% of steps carried
  prefill. The cost falls on cold prompts: the first request of a scan, cache
  evictions, large new tool outputs.
- **Bench suite to add:** run N decode streams, inject a cold long prompt, and
  report the decode streams' inter-token p50, p99 and longest stall, the long
  prompt's time to first token, and total tokens in the window.
  `experiments/2026-09-29-determinism/mixed_latency.py` and
  `experiments/2026-10-01-scheduler-lanes/hol_latency.py` already do parts of
  this; fold them into `bin/spark3 bench`. In production,
  watch `inter_token` p99 and `carrying_prefill_share` from `workload`.
- **Levers (restart required):** a smaller batched-token budget while decodes
  are running (1,024-2,048), trading prefill rate for shorter stalls;
  `--long-prefill-token-threshold` (4,096 today); a prefill interleave
  interval. Keep `--max-parallel-prefills 1`: 8 raised the four-64K mean time
  to first token from 41 s to 67 s. Two lanes and prefill/decode compute
  sharing (`--prefill-compute-share auto`) are prepared in
  `experiments/2026-10-01-scheduler-lanes` with two upstream liveness fixes,
  and not yet measured.

## Decode

Where a step goes on r5o (union of kernel intervals per rank,
`experiments/2026-09-29-determinism`; compute excludes the collectives, whose
kernels spin while waiting for the slowest rank):

| Workload, per step | Compute | Collective wait | Idle |
|---|---:|---:|---:|
| One stream, dgx1 | 47.1 ms | 1.0 ms | 3.7 ms |
| Eight streams, dgx1 / dgx2 / dgx3 | 99.4 / 98.9 / 99.6 ms | 3.9 / 3.0 / 2.5 ms | 5.3 / 6.7 / 6.5 ms |

Routed MoE is half of single-stream compute (24 of 48 ms) and 62% at eight
streams (67 of 108 ms): the target for faster decoding in general. The older
r5e split of the rest (`experiments/2026-09-27-single-stream-profile`): dense
FP8 GEMMs ~15 ms at ~160 GB/s (floor ~10 ms); latency-bound work (81 RoCE
all-reduces, mHC, norms, quantization, top-k, attention) ~6 ms; drafter layers,
draft and target heads and Markov ~7-9 ms (drafter head and Markov now NVFP4).
On that profile the step was about 70% bandwidth-efficient against a ~30 ms
floor.

1. **Dense FP8 GEMM efficiency** is the largest lever. Per-shape plan sweeps
   found only 0-4% (2026-09-24), so the loss is between kernels rather than in
   tile choice. Try keeping weights streaming across kernel boundaries:
   programmatic dependent launch with the weight prefetch issued before the
   dependency sync, fusing GEMMs that share an input, and prefetching during the
   latency-bound phases. The L2 prefetch runs since r5i; its per-phase budgets
   were never retuned for TP3. r5o idles 3.7 ms per single-stream step and
   3.6 ms per eight-stream step on average (7.5 ms gaps in half the steps);
   find what the GPU waits on in those gaps.
2. **CUDA graph launch cost.** One `cudaGraphLaunch` costs about 0.86 ms of host
   time. Check how much of it is exposed, and whether splitting the graph with a
   short lead chunk lets the GPU start sooner.
3. **All-reduce skew.** Median all-reduce cost is ~17 µs; the tail comes from
   ranks arriving late. Collective wait is 2.5-3.9 ms per eight-stream step on
   r5o, and dgx2 is the slowest host. Measure per-rank arrival times. Rank 0
   also runs the API server, and the RoCEnante proxy threads are created without
   CPU affinity (`b12x/comm/roce/_roce_proxy.c`) on GB10's mixed core types;
   pinning them to a fast core is untested.
4. **Per-step confidence broadcast.** Adaptive verification broadcasts draft
   confidences from rank 0 every step, because replicated projections can
   differ in floating-point reduction order, and ranks that choose different
   verify lengths would issue mismatched collectives and hang. Removing it
   needs a confidence computation that is bitwise identical on every rank.
   Measure the broadcast's cost before spending effort on it.
5. **Exact LM-head compression.** The target head is 441 MB per rank, read on
   every step. Fifteen exponent values cover 99.984% of its weights, so an
   exact coded form would be ~331 MB, worth ~0.45 ms per step (~1%). It needs
   a custom exact GEMV for M ≤ 48 that is safe inside CUDA graphs. Low priority.
6. **A second stream inside the decode graph.** Overlap the shared expert with
   the routed MoE, and the attention key path (cache and indexer-key writes)
   with the query path, on a second CUDA stream captured into the graph. First
   measure how much of the ~6 ms latency-bound bucket is serial dependency
   rather than kernel time.

## Speculative decoding

- **Draft depth by load** is the most promising untested lever.
  `num_speculative_tokens_per_batch_size` exists in the scheduler
  (`vllm/v1/core/sched/scheduler.py`) and is unused. Five drafts pay off for
  code at one stream (+14-18% for code answers when adopted). They likely cost
  low-acceptance text at eight streams: reasoning prose at eight streams fell
  6.8% in the r5c promotion that adopted them, before dead rows existed (r5c
  also changed sources and NCCL). Batch size is only a proxy for content,
  though: acceptance is set by what is generated (3.1 accepted per step on
  code answers, 1.89 on live agent traffic). Adaptive verification already
  trims the verify length each step, and dead rows skip routed MoE for
  low-survival rows, so long drafts now cost drafter work plus dense and
  attention work on the rows still verified. Screen depth schedules (for
  example 5 drafts at one or two streams and 3 above) against recorded agent
  traffic as well as the bench. A per-request depth from recent acceptance is
  the content-aware alternative. Include a deeper arm (7 drafts) for code at
  one stream.
- **Drafter reuse of the target's token selection.** Check whether the DSpark
  draft layers run their own indexer scoring and top-k every step. If they do,
  try reusing the target's selection from the anchor position instead. It
  changes only the drafts, so output stays exact; measure drafter time and
  acceptance with the replay trace.
- **Re-check verification pricing on a replayed trace.** The production table
  prices verification on padding rows: about 0.5 ms per extra row (17.6 to
  20.1 ms over six rows, `experiments/2026-10-03-prose-speculation/cost-audit.json`),
  and its cumulative maximum makes the third draft free. Priced on real rows a
  verified row cost about 6.5 ms, and 0009 was not adopted because accepted
  drafts fell in proportion to the shorter steps (`experiments/2026-09-27-dspark-depth5`).
  Those acceptance comparisons came from three samples of non-repeating text;
  a replayed draft trace removes that variation. Re-check 0009, the dead-row
  cut (0.2) and the cost scale (2.0) that way.
- **Verification objective.** The current rule picks the verify length that
  maximizes expected accepted tokens per unit of profiled cost at each step.
  Maximizing expected tokens minus a running-rate-weighted cost may be closer
  to the long-run optimum. Neither the marginal rule (0006) nor the cost-aware
  dead-row count (0011) beat the current configuration.
- **Cost-profile noise.** The startup cost profile takes the median of five
  replays and varies between boots. One pinned table across a fresh boot and
  two restarts gave 51.8, 51.2 and 51.8 tok/s on single-stream prose
  (`experiments/2026-10-03-prose-speculation`): pinning removes the table as a
  variable, not the output-dependent sample noise. Boot patch 0046 (profile
  once when pinned curves match) is written and not promoted.
- **Determinism.** `experiments/2026-09-29-determinism` found three sources: the
  atomic routed-MoE combine, four-way split-K turbo, and a dense GEMM race that
  also gave wrong shared-expert outputs (fixed in r5n). The batch-invariant
  reference ref4f (`VLLM_DS41_BATCH_INVARIANT=1`) keeps every lean workload
  within about 3% of r5o and passed the full validation; it still needs the
  full measurement matrix and the owner's decision. Its traces reach 8,093
  tokens, and its drafter is not batch-invariant, so acceptance still varies
  with batch composition. A scheduling refinement of the sequential mHC kernel
  is bit-identical and 3-4% faster at prefill sizes, and not in `mhc-seq` yet.
  Outside batch invariance the same kernel (vllm-0050) read -4.6% on 16K
  prefill once; repeat that before any production claim.

## Prefill and first token

- **First request after a restart** is still ~0.37 s slower to its first token
  (579 ms against ~210 ms), and nothing JIT-compiles after readiness on r5j.
  Untested suspects: CUDA lazy module loading (screen `CUDA_MODULE_LOADING=EAGER`
  and watch the boot time), first-use allocations, and cold Engram rows.
  Sample the first request with py-spy dumps.
- **Tiny prefill is host-bound:** a 60-token step took 236 ms wall time with
  157 ms of GPU work. Patch 0019 removed most of it. CuTe DSL's argument
  rectification compared every tensor argument with `Tensor.__eq__`, costing
  ~200 µs per 20-tensor launch on 4.6.2. The image now runs 4.7.1
  (`experiments/2026-09-28-cute-dsl-471` never recorded results): re-measure
  there and patch forward if it remains. On r5o an eager step that admits a
  prompt still takes about 180 ms whatever its token count (about 1.6 ms of
  host time per routed-MoE call and 2.0-2.5 ms per attention call).
- **Prefill chunk profile (r5k, 2026-09-29, one 4,096-token chunk under
  sequence parallelism; `experiments/2026-09-29-indexer-split`):** routed MoE
  355 ms, sparse MLA 256, dense GEMMs and mHC 198, collectives 162-173 (the
  NCCL ring reduce-scatters and all-gathers of SP, ~126 ms), indexer 17 ms at
  8K of context and 323 ms at 200K. Patch 0025 split the indexer's rows across
  the ranks (-5% at 64K to -15% at 200K per chunk). What remains of it is the
  MXFP4 score kernel at about 64 TFLOP/s, a kernel lever. On r5o a whole
  chunk takes 1,032 / 1,086 / 1,115 / 1,156 ms at 8K / 64K / 131K / 200K of
  context.
- **NCCL channels for the SP collectives.** They exceed the RoCE one-shot limits
  and run on NCCL with its default channel choice (`NCCL_MAX_NCHANNELS=8`). On
  TP4, four Ring channels instead of one halved NCCL kernel time in a 16K
  prefill trace and made 4K-64K prefill about 23% faster
  (`experiments/2026-10-03-collective-serving`). Read TP3's actual channel
  count from the rank-0 NCCL init log, then screen four Ring channels on
  cooled 32K and 64K prefill. NCCL reports `GDR 0` (no GPUDirect RDMA) on
  every launch.
- **Indexer selection at long context.** Split the indexer's 323 ms per chunk at
  200K into scoring and top-k selection. If selection is a measurable share,
  try a radix top-k over bounded blocks of candidate rows.
- **Sparse-MLA prefill occupancy with few heads per rank.** Each rank holds 24
  query heads at TP3 (16 at TP4). Check the sparse-MLA prefill kernel's SM
  occupancy at long context; if it is under-filled, split each sequence across
  SMs (context parallelism inside the GPU).
- **NVMe interrupt coalescing is off** on every node by the owner's decision
  (2026-10-02). Prefill measured 0.7-1.6% slower that way
  (`experiments/2026-10-02-nvme-coalescing`); don't change it without the owner.
- **Prefix-cache retention:** agent prompts are 64-100K tokens and live on
  cache hits. Add a probe for retention around block boundaries and under
  eviction pressure.

## Boot (~121 s)

Container start to API: engine spawn 20 s, distributed init 11, model build 5,
B12X shard routing 7, weight read 15 (NVMe rate for 101 GB), post-load and B12X
preparation 24, KV setup and warmup 16, graph capture ~8 s of real work.

1. **Post-load preparation (24 s)** is the easiest large win: cache B12X
   routing metadata and prepared state between boots. Boot patches 0046 (DSpark
   costs profiled once when pinned curves match) and 0047 (no temporary-pool
   B12X state stage without autotuning) are written (commit 2b918e1) and not
   yet measured for promotion. Sample it first with
   `experiments/2026-09-28-boot-time/pyspy_sampler.py` to see what the 24 s
   holds.
2. **API server setup:** about 8 s before the engine starts, mostly tokenizer
   `from_pretrained` copies and conversion plus module imports. Audit import
   time.
3. **Per-rank processed-weight image:** the largest and riskiest boot lever. It
   must preserve the MLA absorbed views and the `e_score_correction_bias`
   reference.

- **Already measured, no boot gain:** persisted driver and TileLang JIT caches
  (kept anyway), `NCCL_GIN_ENABLE=0` (communicator init is 0.5-0.6 s),
  `--skip-mm-profiling` (also costs dgx1 0.1 GiB).
- **Pitfalls:** a persisted autotune cache holding per-rank entries can deadlock
  a TP > 1 boot; an interrupted boot can leave corrupt CuTe cache files; warming
  the page cache takes MemFree that the GB10 allocator needs.

## Context capacity

- **Decode context parallelism.** DS4.1 has one shared KV head
  (`num_key_value_heads: 1`), so under TP3 every rank stores the whole KV cache.
  Sharding tokens across the ranks would hold up to three times as many tokens
  per KV byte, at the cost of a per-layer exchange of attention partials. Our
  vLLM has decode-context-parallel infrastructure, including a sparse-attention
  indexer path, but DS4.1's attention does not use it yet. Large; scope it
  against the 512K limit and long agent contexts first.

## dgx1 headroom

- **Vision tower (926 MiB of BF16 weights on every rank).** Image input is
  enabled (`--limit-mm-per-prompt {"image":4}`), so the tower loads. The served
  `deepseek_v4_1/nvidia/vl_model.py` sets `use_data_parallel = True` whatever
  the TP size, so the tower stays replicated even at TP4
  (`experiments/2026-10-03-tp3-tp4-comparison/padding.md`): full weights on
  every rank, images split across ranks, embeddings all-gathered. Options, each
  needing a restart:
  - Switch the tower to tensor parallel, and at TP3 also pad it to 18 heads and
    its MLP from 2,816 to 2,820 with zero weights. That saves ~0.6 GiB per
    rank, including dgx1.
  - If a deployment needs no images, `--limit-mm-per-prompt {"image":0}` or
    `--language-model-only` stubs the tower and skips its weights. That's the
    owner's decision, since image input is a promoted feature.
- **Display reserve:** in use since r5k for the embedding and output head
  (842.5 MiB per rank of the 2,032 MiB the firmware reserves; dgx3 keeps its
  8 MiB console framebuffer). The GPU mapping streams at full speed but is
  uncached with small pages, so only data read once per step belongs there;
  about 1.2 GiB is still free. `experiments/2026-09-29-display-carveout-kv` has
  the access-speed measurements.
- **Uneven head split instead of TP3 padding.** TP3 pads DS4.1's 64 heads and 8
  output groups to 72 and 9 with zero weights, so each rank holds 24 heads in 3
  groups. A balanced uneven split (3/3/2 groups) removes the zero group's
  weights and work from the rank that carries it; measure which rank that is
  and what it costs. The busiest rank keeps 24 heads, so speed only improves
  if heads can split inside a group (22/21/21), which needs an extra reduction
  in the grouped output projection. TP4 needs neither (16 heads per rank).
  `experiments/2026-10-03-tp3-tp4-comparison/padding.md` lists every TP3
  padding (heads, groups, Engram width, vocabulary) and where it is applied.
- **Do not raise `min_free_kbytes`;** it eats the margin the guards protect.

## Quality

- **TMA stage releases outside DS4.1's serving set** (`experiments/2026-09-30-proxy-fence-audit`):
  r5o fences every serving kernel the audit found; still open upstream are the
  NVFP4/W6A8 MoE releases, a SASS pass over kernels DS4.1 does not run, two
  single-stage write-after-read races in raw paged kernels, and mbarrier init fences.

- **Temperature-0 outputs still vary between identical requests** on r5n: the
  atomic MoE combine and split-K turbo change summation order run to run
  (see the determinism item under decode).

- **Reasoning loops in long, compacted agent sessions.** Untested here, and
  reported on other DS4.1 stacks: after several context compactions, hidden
  reasoning degrades into short repeated lines and grows each turn. The
  DS4.1 encoder keeps reasoning for every assistant step since the last user
  message, which is the feedback path. Replay a long compacted transcript at
  reasoning effort 75 and 100, with DSpark on and off, and scan the reasoning
  for short repeated lines. Add that scan as a debug-only eval check, and
  record finish reasons in `workload` windows so a loop shows up as
  length-capped requests.

- **BF16 sparse attention** (patch 0023, off in configuration): measure its
  fidelity with a teacher-forced comparison against
  `VLLM_DS41_ATTENTION_COMPUTE=reference` on long agent transcripts before
  enabling it. It cost 0.5-0.76 GiB of headroom and 8% of decode with four
  180K contexts for a 12% smaller decode/prefill logprob gap. TP4's headroom
  would remove the memory objection; the fidelity check is cheap, so run it
  first.

## Four nodes (TP4)

TP4 on the four-node ring is a measured candidate, not production
(`experiments/2026-10-03-tp3-tp4-comparison`): decode +14 to +27% from one to
eight streams (single-stream steps -19%), source-text prefill +21 to +22%,
acceptance unchanged. It needs no model-dimension padding (16 heads per rank).

- **Long prefill runs hot.** dgx1 and dgx2 reach 83 °C with 3-4 °C of reported
  headroom. The thermal guard aborted a 256K TP3 prefill on the restored
  triangle with dgx2's fans already at maximum
  (`experiments/2026-10-03-tp3-revalidation`), so it is not specific to TP4 or
  the relay; the cause is open. dgx3 stays near 67 °C.
- **Collective policy:** the RoCEnante neighbour relay for small collectives and
  NCCL Ring with four channels for large transfers
  (`experiments/2026-10-03-collective-serving/decision.md`). The explicit-policy
  image (`experiments/2026-10-03-collective-contract`) is source-qualified,
  unbuilt and launch-disabled.
- **Open:** sustained-load cooling and full-context qualification.
- **Cabling decides the profile.** TP3 needs the dgx1-dgx2-dgx3 triangle and
  TP4 the four-node ring, each with a matching node map (the current site
  `config/nodes.json` is a triangle map). The cabling has changed back and
  forth since 2026-10-02:
  check the live links before starting either.
- **Not selected, kept as experiments:** NIC-forwarded mesh4 and four fixed
  paths (`experiments/2026-10-02-rocenante-mesh4`, `-mesh4-fourpaths`).

## Upstream and hygiene

- The one-read `fill_defaults` fix (0019) is still needed on PyTorch main.
- The B12X tiny-decode clamp is still unfixed upstream; keep
  `B12X_W4A8_TINY_DECODE=0`.
- The NCCL IB send-path fence patch should be dropped once upstream releases the
  fix.
- A newer CuTe DSL (4.8.0) exists; qualify the move forward.
- The B12X integration-branch tests fail from API drift in the tests themselves
  (`hc.plan(policy=)`, `TableLayout`, `Plan.state_capacity`), identically on CuTe
  DSL 4.6.2 and 4.7.1. Keep a list of expected failures so new ones stand out.

## Tried and not adopted

Don't repeat these without a new reason:

- **Verification and drafting:** marginal verify rule (0006); gathered Markov
  bias (0007); verifying every draft (`VERIFY_RULE=all`, −5 to −14% at eight
  streams); cost-aware dead-row count (0011); `main_proj` split (0013).
- **Prefill:** M32 MoE tile (+1%); fused M32 MoE (no gain, less headroom);
  `NCCL_PROTO=Simple` (no gain); `--max-parallel-prefills 8` (four-64K mean TTFT
  41 → 67 s); sequence parallelism below 205 tokens (+12 ms TTFT on short
  prompts).
- **Collectives:** NCCL for decode all-reduces (75-86 µs against the relay's
  17-19 µs, `experiments/2026-10-03-collective-policy`); 8 or 16 NCCL channels
  for serving prefill (0.7-1.2% over four, intervals spanning zero);
  bidirectional NCCL rings (no general gain); streaming the relay in 32-64 KiB
  chunks (a consistent bulk penalty, `experiments/2026-10-03-relay-progress`);
  GPUNetIO direct mode (dgx1 rebooted; the runner now rejects it).
- **Drivers:** R610 and CUDA 13.3 libraries
  (`experiments/2026-10-02-driver-cuda-refresh`): no consistent gain, and stock
  R610 faults on large copies with 64 KiB pages.
