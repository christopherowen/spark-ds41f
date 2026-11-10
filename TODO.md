# TODO

Hints for the next improvement runs. These notes are not configuration authority:
`config/cluster.json`, the current baseline manifest and `docs/current-state.md`
describe what runs. Every figure here was measured on this stack; each idea
enters as an ordinary experiment under `experiments/` and is promoted only by the
owner.

## Recent candidate decisions

- **Candidate `1107`** ([`experiments/2026-10-26-spark-ds41f-candidate/1107`](experiments/2026-10-26-spark-ds41f-candidate/1107/README.md), PR [#937](https://github.com/christopherowen/spark-ds41f/pull/937)) consolidates the C45 TileLang compiler (`tilelang/0001..0013` with explicit wrapper-cache and disk-streamed IKET controls), full TileLang runtime closure (`0690..0713`, `0717`, `0752..0754`), CopyDrafter continuation priority, warp early-exit pruning and expected-yield gating against DSpark (`0744..0745`), native-depth CUDA graph retention under copy verification (`0728`), online confidence calibration (`0746`), per-layer SP WO local MXFP8 exchange (`0737`), SP Engram prefill column exchange (`0738`), TP3 6-channel NCCL Ring and sparse-MLA `block_H=16` occupancy (`0740`), 2-launch mHC PDL (`0739` + `0755`), selective pack12 KV compressor (`0749`), fused indexer `wq_b` RoPE + scale (`0748`), packed 264-byte Engram reader with multi-CTA step, hot-row cache and negative `col_delta` guard (`0750`), deep-context scheduler interleave (`0751`), tensor-parallel vision tower (`0758`), and native `TP2..TP8` cross-topology parity (`0747`). Ready for consolidated image build and full servebench qualification on `tp4-ring` and `tp3-triangle`.
- C39 delivered default CopyDrafter behavior, but its optional preparation-overlap
  performance claim showed no proven serving-throughput gain against either
  saved C38 recipe. Preserve the adverse engine-grid cost and original collector
  failures with their separate revalidation; do not repeat the image.
- C40 retains the independent scratch-read correctness fix with neutral aggregate
  serving performance. It is the best validated corrected default-enabled base,
  not a new serving-speedup claim or a promotion.
- C41 attempted to remove optional overlap complexity, declared as simplification
  before measurement. Correctness passed, serving throughput was neutral, but
  grid cost rose 0.206–0.409% and cold 524K prefill fell 0.181–0.667% against
  C40. The focused review found no defensible new correction; reject C41 and
  retain C40. Attribution is unresolved because those pure workloads bypass
  the changed preparation calls. No extra rebuild or old-image remeasurement.
  See the [full measured decision](experiments/2026-11-04-copy-overlap-rework/decision.md).
- The minimum-match-8 policy hypothesis was dropped at source review, not counted
  as a failed serving benchmark. A future copy-policy idea needs a bounded
  expected benefit and budget-binding mixed copy/edit/tool correctness evidence.
- C42's built TileLang refresh plus existing #879–#883 fixes failed its native gate:
  8073 tests passed, 360 skipped and the required IKET prepare/freeze check failed.
  #880 ignores an explicitly prepared kernel when cache is disabled. A focused
  review confirmed that VERSION-only C44 cannot fix this branch; withdraw C44
  unmeasured. The separate C45 correction repairs explicit preparation with
  caching disabled and gives the new compiler its own source-bound cache
  namespace. It continues toward native qualification; C40's compiler remains
  the validated reference meanwhile. No serving/performance point was run,
  and this is not a performance rejection of the compiler idea.
  Preserve the separate cache-provenance gap and all failed native evidence in
  `experiments/2026-11-05-compiler-live-qualification`. Subsequent release gates use
  isolated fresh cache directories; no shared cache is deleted or baseline rerun.

## Goal

Step time and the bench matrix are proxies. The outcome that matters is
end-to-end throughput of the production agent workload: findings per hour on a
fixed scope and tool policy. Record the server side of every agent run with
`bin/spark workload --json <path> -- <agent command>` (read-only; it sends no
requests) and judge candidates on that whenever a change could shift the
result.

A first 10-second window of live agent traffic (2026-09-28):

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

## Next release

<!-- BEGIN GENERATED: next-release-status -->
Production is **spark-ds41f-1107** (`manifests/baselines/2026-11-07-spark-ds41f-1107.json`, image `spark-ds41f:1107`), promoted 2026-11-07 across `tp4-ring` ([suite reference](manifests/suites/2026-11-07-spark-ds41f-1107-tp4/)), `tp3-triangle` ([suite reference](manifests/suites/2026-11-07-spark-ds41f-1107-tp3/)), and `tp3-switched`..`tp8-switched` ([release record](experiments/2026-10-26-spark-ds41f-candidate/README.md)).
<!-- END GENERATED: next-release-status -->

Before promoting `1107`:
1. **Build and qualify `spark-ds41f:1107`** across the full servebench release,
   single-stream, and sampled suites on `tp4-ring` and `tp3-triangle` against
   `spark-ds41f-1030`.
2. **Verify `dgx1` memory headroom and native-depth graph capture (`0728`)**:
   record startup and steady-state `MemAvailable` during 1M cold prefill and
   under the widened `max_decode_rows` (`59` on TP3, `107` on TP4) plus 69
   target CUDA graphs on TP3 (`60 -> 69`), confirming headroom stays comfortably
   above the 5 GiB startup and 3 GiB steady-state guards.
3. **Retire legacy `bin/spark bench` suites** once no active workflow references
   them; servebench suites (`docs/release-benchmark.md`) are the release
   benchmark authority.

## Current reference

<!-- BEGIN GENERATED: current-reference -->
Active baseline manifest: [`manifests/baselines/2026-11-07-spark-ds41f-1107.json`](manifests/baselines/2026-11-07-spark-ds41f-1107.json) (`spark-ds41f-1107`). Releases are judged by the servebench release, single-stream and sampled suites (`docs/release-benchmark.md`, `scripts/servebench_suite_table.py`). Current benchmark references are [`manifests/suites/2026-11-07-spark-ds41f-1107-tp4`](manifests/suites/2026-11-07-spark-ds41f-1107-tp4/) for `tp4-ring` and [`manifests/suites/2026-11-07-spark-ds41f-1107-tp3`](manifests/suites/2026-11-07-spark-ds41f-1107-tp3/) for `tp3-triangle`:

| Measure | `tp4-ring` (`spark-ds41f-1107`, 1M, 16 seqs, 16K batch) | `tp3-triangle` (`spark-ds41f-1107`, 1M, 8 seqs, 8K/6K batch) |
| --- | ---: | ---: |
| Decode grid geometric mean (20 TP4 / 15 TP3 shapes) | 21.05 ms | 21.27 ms |
| Decode `c16`, 16 streams (output tok/s) | 272.9 | — |
| Decode `c8`, 8 streams (output tok/s) | 207.4 | 160.8 |
| Decode `c4`, 4 streams (output tok/s) | — | 123.7 |
| Single-stream `c1` (output tok/s) | 82.6 | 64.3 |
| Sampled `c16` / `c8` / `c1` (`temp=1.0`, output tok/s) | 268.3 / 203.8 / 80.0 | 156.6 / 157.6 / 63.3 |
| Poisson arrivals (0.7 TP4 / 0.42 TP3 req/s, output tok/s) | 180.2 | 119.2 |
| Source-text prefill 32K (tok/s) | 8,508 | 6,452 |
| Prefill 131K (tok/s) | 8,657 | 6,480 |
| Prefill 262K (tok/s) | 8,261 | 6,135 |
| Prefill 524K (TP4) / 500K (TP3) (tok/s) | 7,460 | 5,514 |
| Prefill 1M (tok/s) | 6,012 | 4,434 |
| Short-request TTFT p50 during 131K prefill (3 / 11 TP4 or 3 / 6 TP3 streams) | 1.02 s / 1.44 s | 1.07 s / 1.18 s |
<!-- END GENERATED: current-reference -->

A step-time saving is a fixed cost per step. At one stream it converts almost
fully into tok/s. At eight or sixteen streams the step is longer (more verified
rows and more distinct experts read), so the same saving is a smaller share.
Report one and eight streams for every decode change, and sixteen on TP4.

## Running a round

- **Screen what the release judges.** Development screens use the fixed-row
  decode grid (adaptive verification off) plus distinct prompts at the recipe's
  concurrency on TP4 and TP3, measured cumulatively against the previous
  release with at least two boots per candidate arm. Compare against recorded
  baseline runs rather than re-benchmarking a released build.
- **Confirm a new path engages** (log it once) before trusting an A/B.
- **Start cool.** `bench` and lab jobs bring every node below 55 °C before
  measuring (`--cool-below`, `--cool-timeout`). Long prefills still reach
  83 °C on dgx1 and dgx2 (see Four nodes).
- **Separate stable measures from output-dependent ones.** Anything that depends
  on generated text varies between samples: accepted drafts per step,
  prefix-cache hits, output length. Fixed-row step time is stable to about
  ±0.5 ms with three samples; acceptance is not. For changes that move
  acceptance, use paired multi-prompt suites (`servebench`) across two boots.
- **When a decision hinges on 2.5% or less**, watch boot-to-boot variance across
  repeat boots before attributing a small delta to a kernel or policy change.
- **Python-only vLLM changes need no image build to screen.** Bind-mount the
  changed files from the checked patch branch over the last candidate image via
  `container.mounts` (the image imports vLLM from `/opt/spark-ds41f/candidate/vllm`).
  Build only for promotion or when compiled extensions/TileLang sources change.
- **Check every candidate's boot log for silent fallbacks** (`disabled`,
  `fallback`, `compile failed`) and for any JIT compilation after readiness.
- **Shared packages move forward only.** When components disagree on a shared
  runtime package, patch the lagging pin forward.

## Memory and safety

- `dgx1` hosts rank 0 and the API server, so it is always the tightest node.
  Check its headroom before anything else, and never lower the guards
  (5 GiB at startup, 3 GiB steady).
- Profiling: never use the torch profiler with `with_stack=True` on `dgx1`;
  stopping it dropped `MemAvailable` to 1.5 GiB and the guard stopped the
  service. `py-spy record` hangs on the workers; a loop of
  `py-spy dump --nonblocking` works as a sampler. Workers rename themselves
  (`VLLM::Worker_TP0`), so select processes by container, not by name.
- Kernel, sysctl, boot, firmware, package, display and network changes are the
  owner's decisions, and any host change goes to every node. Stop the
  service before rebooting a node, and don't restart inference while it is in
  use.
- **Carve-out integrity check is a debug mode.**
  `SPARK_DS41F_DISPLAY_CARVEOUT_CHECK_SECONDS` stays unset in production; set it
  (for example to 60) only to chase a suspected carve-out fault. It costs
  about 20 ms per check and nothing when unset. Serving-path changes for
  quality must not add computation.

## Mixed prefill and decode

The agent workload interleaves long prefill bursts with decode phases. Cached
prompts avoid most of it (98–99% prefix-cache hit rate in live agent traffic),
so the cost falls on cold prompts: the first request of a scan, cache
evictions, or large new tool outputs.

- **In `1107` (`0751`)**: deep-context scheduler interleave (`VLLM_DS41_DEEP_PREFILL_INTERLEAVE=1`
  above 16K context) and `block_M=32` long-context sparse-MLA staging reduce
  32K–64K mixed-decode ITL stalls by 23–27% while keeping pure 131K prefill
  throughput neutral (`experiments/2026-11-06-deep-context-scheduler`).
- **Idle streams during long cold prefills.** A 1M cold prefill takes ~3 minutes
  and sends nothing before its first token. Check whether clients or gateways
  cut idle streams during 1M prefills; if so, send SSE keep-alives.

## Decode

Routed MoE is the largest share of decode compute (~50% at one stream, ~62% at
eight streams), followed by dense FP8 projections, MLA attention/indexer,
collectives, and DSpark drafting.

1. **Routed MoE continuous `FC1 -> SwiGLU -> FC2 -> Combine` persistent kernel**
   (~2.4–3.5% decode ceiling). TileLang's `swiglu_forward_and_per_token_cast`
   takes 23–30 µs per layer (~3% of a decode step) and runs serially between the
   two grouped GEMMs. Keeping routed tiles resident across `FC1 -> SwiGLU -> FC2`
   (or fusing the SwiGLU + MXFP8 requantization epilogue over live tiles) avoids
   the global intermediate round-trip and extra kernel launch per layer.
2. **Direct producer writes (`WO-B` / `moe_combine`) into registered `sparknet`
   RoCE send buffers** (~1.3–1.6% decode ceiling). `1107` overlaps collective
   launches via PDL (`0753`, `0755`, `0756`), but `roce_all_reduce` still stages
   an explicit `cudaMemcpyAsync` from the producer output tensor into
   `comm_buf` (116 launches per step). Writing final BF16 outputs directly into
   the registered RoCE send buffer removes the staging copy.
3. **Q-B and indexer Q-B tile-contiguous weight layout.** Both projections win
   in isolation but trail by 3–5% per call in serving when reading row-major
   weights. Screen a tile-contiguous weight layout for the decode projections,
   kept bit-identical with prefill.
4. **Vocabulary head (`128`-row token tile and greedy argmax epilogue).**
   - **A 128-row token tile:** a head call with 65–128 rows on TP4 `c16` runs a
     second 64-row tile that re-reads and unpacks the 248 MB shard from DRAM.
     A 128-row tile fits in ~96 KB of shared memory with two activation stages
     and four weight stages, keeping the same K accumulation order (~1% ceiling
     at sixteen streams on TP4).
   - **Greedy argmax in the head epilogue:** write each tile's `(max, index)`
     pair instead of full FP32 logits when all active requests are greedy (~12 MB
     less written and read per TP4 call).
5. **A second stream inside the decode graph.** Overlap the shared expert with
   routed MoE, and attention key/indexer-key writes with the query path, on a
   second CUDA stream captured into the graph. First bound how much of the
   latency-bound bucket is serial dependency rather than SM occupancy.
6. **All-reduce arrival skew and RoCE proxy core affinity.** Rank 0 also runs
   the API server, and `sparknet` proxy threads run without explicit CPU core
   affinity on GB10's mixed core types; test pinning them to fast cores.
7. **A RoCE one-shot broadcast for adaptive verification.** Adaptive
   verification broadcasts rank 0's confidence snapshot every decode step
   through NCCL (`adaptive_verification.py`). Bound its share of the step from
   decode traces and screen `sparknet`'s one-shot broadcast path if material.

## Speculative decoding

In `1107`, DSpark online confidence calibration (`0746`), CopyDrafter
continuation-priority tie-breaking (`0744`), descending-priority warp early
exit (`0744`), expected-yield gating against DSpark (`0745`), native-depth CUDA
graph retention (`0728`), fused single-rank Markov drafting (`0717`), and
device-side `prepare_inputs` folding (`0754`) are all active.

Open levers:
- **Bounded or indexed CopyDrafter history match at 100K–1M contexts.** Even
  with warp early-exit (`0744`), a decode step that does not hit a 256-token
  match scans the full `[min_match, total_len)` history in UVA (`all_token_ids.gpu`)
  on `output_copy_stream` (~0.36–0.47 ms/step at `c16`, 1M random tokens, and up
  to ~2 ms on repeated 16/32-token boilerplate matches), contending with DSpark
  on `main_stream` for unified memory bandwidth. Maintain a lightweight
  incremental per-slot rolling-hash index of 16-token suffix positions updated
  only for newly committed tokens, or bound the scan to a recency + prefix
  window so `copy_match` is $O(1)$ in context length.
- **Draft depth by load or recent acceptance.** Five DSpark drafts pay off on
  high-acceptance code at one stream (`+14–18%`), whereas live agent traffic and
  8–16 stream prose accept fewer tokens per step. Screen load-aware
  (`num_speculative_tokens_per_batch_size`) or per-request acceptance-aware
  draft depth schedules on recorded agent traffic and the paired servebench
  suites.
- **Drafter reuse of the target's token selection.** Check whether reusing the
  target model's indexer top-k selection from the anchor position in DSpark
  draft layers preserves acceptance while cutting drafter attention work.
- **Cross-request verify-row allocation at high concurrency.** Compare our
  per-slot survival budget (`assign_draft_budget`) plus per-step verify length
  against ranking candidate draft slots across the whole batch by marginal
  expected accepted tokens per marginal millisecond of step cost at 8–16 streams.

## Prefill and first token

In `1107`, per-layer SP WO local MXFP8 exchange (`0737`), sequence-parallel
Engram prefill column exchange and sliced projection (`0738`), TP3 6-channel
NCCL Ring and sparse-MLA `block_H=16` (`2` CTAs/head, `0740`), unconditional
`M=64` MoE `prefill_swizzle_panel=2` (`0736`), retained CED encoder endpoints
(`0727`), CED local boundary reduction (`0742`), and SP context-KV projection
(`0743`) are all active.

Open levers:
- **Sequence-parallel MoE input (`experiments/2026-10-14-sp-moe-gather`).** Each
  encoder layer all-gathers BF16 rows, then casts them to MXFP8 twice (shared
  and routed experts) and routes them on all $T$ rows on every rank (~4.4 ms a
  layer at an 8,096-row chunk). Routing and casting each rank's own shard before
  all-gathering MXFP8 rows with their scales and routes preserves exact bits
  with a ~3–3.9% prefill ceiling.
- **First request after a restart** is ~0.37 s slower to its first token than
  subsequent requests. Screen `CUDA_MODULE_LOADING=EAGER` (checking boot time
  impact) and sample the first request with `py-spy dump --nonblocking`.
- **Indexer selection at 200K–1M context.** Split long-context indexer time into
  MXFP4 scoring vs top-k selection; if top-k selection is a measurable share at
  500K–1M, screen a bounded radix top-k.
- **NVMe interrupt coalescing stays off** on every node by the owner's decision
  (2026-10-02; prefill measured 0.7–1.6% slower with it on in
  `experiments/2026-10-02-nvme-coalescing`).
- **Prefix-cache retention:** agent prompts are 64–100K tokens and depend on
  98–99% prefix-cache hit rates ([#902](https://github.com/christopherowen/spark-ds41f/issues/902)).
  Add a stress probe for retention around block boundaries and under eviction
  pressure.

## Boot

- **Post-load preparation and routing metadata:** audit remaining post-load
  preparation time with `py-spy` now that B12X is retired and 264-byte packed
  Engram shards are reused across boots (`0750` logs `bytes`, `written_bytes`,
  `reused`, and `pack_s` per rank/layer at startup).
- **API server setup (~8 s):** mostly tokenizer `from_pretrained` copies and
  module imports before engine spawn.
- **Per-rank pre-processed weight image:** largest boot lever, if needed; must
  preserve MLA absorbed views and `e_score_correction_bias`.
- **Pitfalls:** warming the host page cache consumes `MemFree` that the GB10
  unified memory allocator needs; keep startup caches bounded.

## Context capacity

- **Decode context parallelism.** DS4.1 has one shared KV head
  (`num_key_value_heads: 1`), so under TP3/TP4 every rank stores a full copy of
  the KV cache. Sharding KV tokens across ranks would multiply KV capacity by
  the TP degree at the cost of a per-layer exchange of attention partials.

## dgx1 headroom

- **Vision tower (926 MiB of BF16 weights, tensor-parallelized in `1107` via
  `0758`).** `deepseek_v4_1/nvidia/vl_model.py` and `tilelang_vision.py` shard
  the ViT (`wqkv`, `wo`, `mlp.w1`, `mlp.w2`) and aligner (`w1`, `w2`) across TP
  ranks (`--mm-encoder-tp-mode weights`), saving ~0.68 GiB per rank at TP4 and
  ~0.59 GiB per rank at TP3 (with zero-weight tail padding to 18 heads / 2,880
  MLP / 5,184 aligner on TP3).
- **Display reserve (`0762`, `experiments/2026-11-08-display-carveout-weight-expansion`):**
  12-bit packed `embed_tokens` and `lm_head` dumb buffers are compacted in place
  (`864 -> 640 MiB` at TP3, `640 -> 480 MiB` at TP4), and the remaining budget up
  to `1,968 MiB` (leaving `64 MiB` of the `2,032 MiB` scanout reserve on `dgx3`
  for the `8 MiB` console framebuffer) holds DSpark's `markov_w1.weight` (`64 MiB`)
  plus decode-only DSpark routed-expert `w13_weight`/`w2_weight` slabs (`1,904 MiB`
  buffers / `1,895.0 MiB` weights at TP3, saving `+1.23 GiB` allocated / `+1.29 GiB`
  reserved UMA per rank; `1,968 MiB` buffers / `1,936.6 MiB` weights at TP4, saving
  `+1.43 GiB` per rank). See `experiments/2026-09-29-display-carveout-kv`.
- **Uneven head split instead of TP3 padding.** TP3 pads DS4.1's 64 heads and 8
  output groups to 72 and 9 with zero weights (24 heads in 3 groups per rank).
  `experiments/2026-10-03-tp3-tp4-comparison/padding.md` documents all TP3
  paddings.
- **Do not raise `min_free_kbytes`;** it eats the margin the guards protect.

## Quality

- **Temperature 0 is deterministic** alone and in mixed batches
  (`token_determinism.py --wide`). Run it on every candidate.
- **Loop containment ([#864](https://github.com/christopherowen/spark-ds41f/issues/864)):**
  `1107` carries the v2 loop detector (`0679`/`0684`). Complete the long-context
  83K max-effort canary soak on `1107` during release qualification.
- **DSML token provenance ([#876](https://github.com/christopherowen/spark-ds41f/issues/876),
  [#948](https://github.com/christopherowen/spark-ds41f/issues/948), PR
  [#949](https://github.com/christopherowen/spark-ds41f/pull/949)):** ensure
  quoted DSML markup inside normal reply text or tool-call string arguments is
  distinguished from genuine DSML structural tokens on top of `0643`'s streaming
  string-argument extender.
- **Opt-in Latent Sparse Attention (`VLLM_DS41_LSA`, `0741` in `1107`):** ships
  disabled by default (`VLLM_DS41_LSA=0`) with full runtime plumbing and unit
  tests; any future enablement requires an independent long-context answer
  quality evaluation.
- **BF16 sparse attention** (patch `0023`, off in configuration): measure its
  fidelity with a teacher-forced comparison against
  `VLLM_DS41_ATTENTION_COMPUTE=reference` on long agent transcripts before
  considering it on TP4.

## Topologies (`tp4-ring`, `tp3-triangle`, and `TP2..TP8`)

- **Cabling decides the profile.** `tp3-triangle` needs the `dgx1-dgx2-dgx3`
  triangle and `tp4-ring` the four-node ring (`bin/spark site shape`). `cluster
  start` refuses a node map whose cable subnets do not match the live links, and
  any change of node count or shape without `--change-shape`.
- **Cross-topology parity (`0747` in `1107`):** `TP2..TP8` head/KV/expert
  divisibility, padding, and collective routing are unified in source. Any new
  physical topology still requires its own hardware qualification before
  promotion.
- **Long prefill runs hot.** `dgx1` and `dgx2` reach 83 °C during sustained
  256K–1M prefills with fans at maximum (`dgx3` stays near 67 °C). Keep
  `--cool-below 55` between benchmark arms.
- **Collective policy (`0320`):** `sparknet`'s one-shot RoCE collectives for
  small decode/prefill transfers and NCCL Ring (4 channels on TP4, 6 channels on
  TP3 in `1107`) for large sequence-parallel collectives, with no silent fallback
  to PyTorch process-group collectives.

## TileLang kernel family

The TileLang runtime migration is complete in `1030` and `1107`: no B12X kernel
runs in serving (`--load-format direct` and all model, drafter, sampler,
rank-reduction, gather-store, scoring, vision-tower, L2-prefetch, and collective
paths run on TileLang + `sparknet`; retiring the remaining `b12x` build input
from `upstreams.lock.json` and `docker/Dockerfile` can follow after `1107`).

Remaining kernel-level investigations:
1. **`mHC: fn` in three bytes.** `project_streams` reads `fn` (`24 x 20,480`
   FP32, 2 MB per sublayer) near memory bandwidth, and a TF32 MMA consumes only
   the top 19 bits of each value. Storing the top 24 bits would cut that traffic
   by 25% with bit-identical results if SM121's TF32 MMA truncates rather than
   rounds its inputs; verify on hardware first.
2. **TMA stage releases outside DS4.1's active kernel set**
   (`experiments/2026-09-30-proxy-fence-audit`): all serving kernels are fenced
   (and enforced in the compiler via `tilelang/0001..0013` in `1107`); keep the
   compiler barrier tests in the release gate whenever TileLang is refreshed.

## Upstream and hygiene

- **No fallbacks** (owner, 2026-10-14): every collective or kernel dispatch path
  must either be the tested production path or raise with a traceback.
- The one-read `fill_defaults` fix (`0019`) is still needed on PyTorch main.
- The NCCL IB send-path fence patch should be dropped once upstream releases the
  fix.

## Tried and not adopted

Don't repeat these without a new reason:

- **Verification and drafting:** marginal verify rule (`0006`); gathered Markov
  bias (`0007`); verifying every draft (`VERIFY_RULE=all`, −5 to −14% at eight
  streams); cost-aware dead-row count (`0011`); `main_proj` split (`0013`);
  lowering `VLLM_DS41_VERIFY_ROW_COST` to `0.010` or `0.020`
  (`experiments/2026-11-06-verification-budget-hardware`, regressed throughput
  vs `0.030`); minimum-match-8 CopyDrafter policy (dropped at source review);
  C39/C41 CopyDrafter preparation-overlap variations
  (`experiments/2026-11-04-copy-overlap-rework`).
- **Decode kernels:** 1-launch atomic last-CTA `mHC project_and_finalize_streams`
  for `M <= 16` (`experiments/2026-11-06-mhc-project-finalize`, #946: 12–34%
  slower than 2-launch PDL due to atomic/fence serialization and register
  pressure); fusing decode `WO-A` split-K reduction into `WO-A` (`0737` applies
  only to prefill SP WO; decode `WO-A` uses `split_k=4` across `K=2048` and
  `WO-B` reads all `1280` reduced columns per CTA); bounded L2 access-policy
  retention windows on GB10 (#928, no gain / regression on unified memory);
  incremental GPU block-table metadata gathers (#932, negligible step impact vs
  state complexity); `0714`/`0716` score-ID host transfer and `0718` TP1-only
  embedding ID prep (#913, #914: unmeasured / TP1-only).
- **Prefill:** M32 MoE tile (+1%); fused M32 MoE (no gain, less headroom);
  `NCCL_PROTO=Simple` (no gain); `--max-parallel-prefills 8` (four-64K mean TTFT
  41 → 67 s); sequence parallelism below 205 tokens (+12 ms TTFT on short
  prompts); `8` or `12` NCCL Ring channels on TP3 SP prefill
  (`experiments/2026-11-06-nccl-sp-channels`, `6` channels won while `8` and
  `12` regressed).
- **Collectives:** NCCL for decode all-reduces (75–86 µs against one-shot RoCE's
  13–19 µs, `experiments/2026-10-03-collective-policy`); 8 or 16 NCCL channels
  on TP4 serving prefill (0.7–1.2% over four, intervals spanning zero);
  bidirectional NCCL rings (no general gain); streaming the relay in 32–64 KiB
  chunks (a consistent bulk penalty, `experiments/2026-10-03-relay-progress`);
  GPUNetIO direct mode (`dgx1` rebooted; the runner now rejects it).
- **Drivers:** R610 and CUDA 13.3 libraries
  (`experiments/2026-10-02-driver-cuda-refresh`): no consistent gain, and stock
  R610 faults on large copies with 64 KiB pages.
