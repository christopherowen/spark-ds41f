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

`manifests/benchmarks/2026-09-29-karmic-kraken-r5l.json` (256K limit, 2.2 GiB
of KV, 1,348,708 tokens). Aggregate tok/s across all streams, temperature 0,
256 output tokens:

| Workload | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| JSON, answer only | 77.0 | 109.7 | 165.2 | 234.3 |
| Code, answer only | 80.6 | 118.2 | 165.1 | 233.0 |
| Code, reasoning on | 59.7 | 90.4 | 131.4 | 187.4 |
| Prose, answer only | 59.1 | 84.8 | 124.9 | 174.5 |
| Prose, reasoning on | 50.3 | 76.1 | 111.8 | 162.0 |

- **Time to first token (short prompts):** 0.21-0.23 s at one stream and
  about 0.52 s at eight.
- **Real-text prefill:** 3.9k tok/s at 4K, 3.8k at 64K, 3.8k at 200K.
- **Quality:** `experiments/2026-09-29-r5k/consistency.py` measures the
  decode-versus-prefill logprob gap on greedy generations (r5k: 0.0506 mean,
  2.84% argmax disagreement); use it for any change that touches decode-only
  state.

A step-time saving is a fixed cost per step. At one stream it converts almost
fully into tok/s. At eight streams the step is longer (up to 48 verified rows,
more distinct experts read), so the same saving is a smaller share. Report one
and eight streams for every decode change.

## Running a round

- **Screen lean, promote thorough.** One boot per arm:
  `bin/spark3 bench --suites quality,decode --decode-cases prose,code,prose-nothink,code-nothink --concurrency 1,8 --min-samples 3 --max-samples 3`
  (about 8 minutes with the boot). Run the full matrix only for a promotion
  candidate.
- **Separate stable measures from output-dependent ones.** Temperature-0
  outputs drift within one boot, even at one stream, so anything that depends
  on the generated text varies between samples: accepted drafts per step,
  prefix-cache hits, output length. Time per verification step is stable to
  about ±0.5 ms with three samples; acceptance is not. For changes that move
  acceptance, replay a recorded draft trace (patch 0008 `SPARK3_DSPARK_TRACE`
  with `experiments/2026-09-26-dspark-policy/replay.py`) or raise the sample
  count. Eight-stream aggregates swing 3-5% between samples.
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
  6.03 GiB minimum MemAvailable under load on r5k (the 256K limit costs about
  0.4 GiB). Check its headroom before
  anything else, and never lower the guards (5 GiB at startup, 3 GiB steady).
- Known to cross the startup guard on dgx1: 8,192 batched tokens, draft TP 1,
  full in-engine B12X autotune.
- Profiling: never use the torch profiler with `with_stack=True` on dgx1;
  stopping it dropped MemAvailable to 1.5 GiB and the guard stopped the
  service. `py-spy record` hangs on the workers; a loop of
  `py-spy dump --nonblocking` works as a sampler. Workers rename themselves
  (`VLLM::Worker_TP0`), so select processes by container, not by name.
- Kernel, sysctl, boot, firmware, package, display and network changes are the
  owner's decisions, and any host change goes to all three nodes. Stop the
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
  prompt's time to first token, and total tokens in the window. In production,
  watch `inter_token` p99 and `carrying_prefill_share` from `workload`.
- **Levers (restart required):** a smaller batched-token budget while decodes
  are running (1,024-2,048), trading prefill rate for shorter stalls;
  `--long-prefill-token-threshold` (4,096 today); a prefill interleave
  interval. Keep `--max-parallel-prefills 1`: 8 raised the four-64K mean time
  to first token from 41 s to 67 s.

## Decode

Where a single-stream step goes (steps are now about 42 ms on prose and 48 ms
on code; breakdown from the r5e trace in
`experiments/2026-09-27-single-stream-profile`):

| Part | Time | State |
|---|---:|---|
| Routed MoE (40 calls) | ~17 ms | at the ~236 GB/s streaming ceiling |
| Dense FP8 GEMMs (248 per step, 40-80 µs each) | ~15 ms | ~160 GB/s average; floor ~10 ms |
| Latency-bound work (81 RoCE all-reduces, mHC, norms, quantization, top-k, attention) | ~6 ms | |
| Outside verification (drafter layers, draft and target heads, Markov) | ~7-9 ms | drafter head and Markov now NVFP4 |

The whole step is about 70% bandwidth-efficient against a ~30 ms floor.

1. **Dense FP8 GEMM efficiency** is the largest lever. Per-shape plan sweeps
   found only 0-4% (2026-09-24), so the loss is between kernels rather than in
   tile choice. Try keeping weights streaming across kernel boundaries:
   programmatic dependent launch with the weight prefetch issued before the
   dependency sync, fusing GEMMs that share an input, and prefetching during the
   latency-bound phases. The L2 prefetch runs since r5i; its per-phase budgets
   were never retuned for TP3. Measure GPU idle time as the union of kernel
   intervals.
2. **CUDA graph launch cost.** One `cudaGraphLaunch` costs about 0.86 ms of host
   time. Check how much of it is exposed, and whether splitting the graph with a
   short lead chunk lets the GPU start sooner.
3. **All-reduce skew.** Median all-reduce cost is ~17 µs; the tail comes from
   ranks arriving late. Measure per-rank arrival times. Rank 0 also runs the API
   server, and the RoCEnante proxy threads are created without CPU affinity
   (`b12x/comm/roce/_roce_proxy.c`) on GB10's mixed core types; pinning them
   to a fast core is untested.
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
  the content-aware alternative.
- **Verification objective.** The current rule picks the verify length that
  maximizes expected accepted tokens per unit of profiled cost at each step.
  Maximizing expected tokens minus a running-rate-weighted cost may be closer
  to the long-run optimum. Neither the marginal rule (0006) nor the cost-aware
  dead-row count (0011) beat the current configuration.
- **Cost-profile noise.** The startup cost profile takes the median of five
  replays, and is a likely source of the ~3% boot-to-boot variation. Pinned cost
  curves (0005) did not remove the sample noise.
- **Determinism.** `experiments/2026-09-29-determinism` found three sources: the
  atomic routed-MoE combine, four-way split-K turbo, and a dense GEMM race that
  also gave wrong shared-expert outputs (fixed in r5n). An experimental
  deterministic mode (its 0004-0006, split-K through the FP32 reducer) repeats
  exactly at one stream with no single-stream cost; one unpinned eight-stream
  screen read JSON 3.7% slower, with verification work not held fixed. Its
  0008 (masked top-k sum, no dead-route clearing) matches the atomic combine
  within 0.4% at fixed shapes. Before proposing it: profile against r5n with
  one pinned cost table, and check repeatability across batch compositions.

## Prefill and first token

- **First request after a restart** is still ~0.37 s slower to its first token
  (579 ms against ~210 ms), and nothing JIT-compiles after readiness on r5j.
  Untested suspects: CUDA lazy module loading (screen `CUDA_MODULE_LOADING=EAGER`
  and watch the boot time), first-use allocations, and cold Engram rows.
  Sample the first request with py-spy dumps.
- **Tiny prefill is host-bound:** a 60-token step took 236 ms wall time with
  157 ms of GPU work. Patch 0019 removed most of it. CuTe DSL's argument
  rectification compared every tensor argument with `Tensor.__eq__`, costing
  ~200 µs per 20-tensor launch on 4.6.2; re-measure on 4.7.1 and patch forward
  if it remains.
- **Prefill chunk profile (r5k, 2026-09-29, one 4,096-token chunk under
  sequence parallelism; `experiments/2026-09-29-indexer-split`):** routed MoE
  355 ms, sparse MLA 256, dense GEMMs and mHC 198, collectives 162-173 (the
  NCCL ring reduce-scatters and all-gathers of SP, ~126 ms), indexer 17 ms at
  8K of context and 323 ms at 200K. Patch 0025 split the indexer's rows across
  the ranks (-5% at 64K to -15% at 200K per chunk). What remains of it is the
  MXFP4 score kernel at about 64 TFLOP/s, a kernel lever. The SP collectives
  exceed the RoCE one-shot limits and run on NCCL: check whether large
  transfers use every available port and PCIe path.
- **Prefix-cache retention:** agent prompts are 64-100K tokens and live on
  cache hits. Add a probe for retention around block boundaries and under
  eviction pressure.

## Boot (123 s)

Container start to API: engine spawn 20 s, distributed init 11, model build 5,
B12X shard routing 7, weight read 15 (NVMe rate for 101 GB), post-load and B12X
preparation 24, KV setup and warmup 16, graph capture ~8 s of real work.

1. **Post-load preparation (24 s)** is the easiest large win: cache B12X
   routing metadata and prepared state between boots. Sample it first with
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

## dgx1 headroom

- **Vision tower (926 MiB of BF16 weights on every rank).** Image input is
  enabled (`--limit-mm-per-prompt {"image":4}`), so the tower loads. Its 16
  attention heads don't divide by 3, so vLLM falls back to a data-parallel ViT
  (`is_vit_use_data_parallel`): full weights on every rank, images split across
  ranks, embeddings all-gathered. Options, each needing a restart:
  - Pad the ViT to 18 heads and its MLP from 2,816 to 2,820 with zero weights,
    so TP3 can shard it exactly. That saves ~0.6 GiB per rank, including dgx1.
  - If a deployment needs no images, `--limit-mm-per-prompt {"image":0}` or
    `--language-model-only` stubs the tower and skips its weights. That's the
    owner's decision, since image input is a promoted feature.
- **Display reserve:** in use since r5k for the embedding and output head
  (842.5 MiB per rank of the 2,032 MiB the firmware reserves; dgx3 keeps its
  8 MiB console framebuffer). The GPU mapping streams at full speed but is
  uncached with small pages, so only data read once per step belongs there;
  about 1.2 GiB is still free. `experiments/2026-09-29-display-carveout-kv` has
  the access-speed measurements.
- **Do not raise `min_free_kbytes`;** it eats the margin the guards protect.

## Quality

- **Two more TMA stage-release races in r5n** (`experiments/2026-09-30-proxy-fence-audit`):
  the mHC TF32 prefill projection (609/12000 wrong beside the routed MoE, 59 beside
  a copy) and the BF16 GEMV prefill (up to 45/12000) release TMA-filled stages
  with shared loads pending, like the dense GEMM before r5n; fences remove them.
  mHC runs in every layer's prefill: qualify and promote the fix next.

- **Temperature-0 outputs still vary between identical requests** on r5n: the
  atomic MoE combine and split-K turbo change summation order run to run
  (see the determinism item under decode).

- **BF16 sparse attention** (patch 0023, off in configuration): measure its
  fidelity with a teacher-forced comparison against
  `VLLM_DS41_ATTENTION_COMPUTE=reference` on long agent transcripts before
  enabling it. It cost 0.5-0.76 GiB of headroom and 8% of decode with four
  180K contexts for a 12% smaller decode/prefill logprob gap.

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
- **Memory:** 512K maximum model length (−6% at four and eight streams, less
  dgx1 margin; opt-in only).
