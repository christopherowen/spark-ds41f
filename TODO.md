# TODO

Hints for the next improvement runs. These notes are not configuration authority:
`config/cluster.json`, the current baseline manifest and `docs/current-state.md`
describe what runs. Every figure here was measured on this stack; each idea
enters as an ordinary experiment under `experiments/` and is promoted only by the
owner.

## Running a round

- **Screen lean, promote thorough.** One boot per arm:
  `bin/spark3 bench --suites quality,decode --decode-cases prose,code,prose-nothink,code-nothink --concurrency 1,8 --min-samples 3 --max-samples 3`
  (about 8 minutes with the boot). Run the full matrix only for a promotion
  candidate.
- **Decide on step time.** Single-stream step time is
  `(1 + accepted drafts per step) / tok/s` and is precise to about ±0.5 ms with
  three samples. Eight-stream aggregates swing 3-5% between samples, because
  temperature-0 outputs differ from run to run even at one stream. Treat
  single-arm differences under 5% as noise.
- **Control on the same day, with the same protocol.** A lean run and a full
  matrix of the same image differ by about 1 ms per step at one stream and 3-7%
  at eight streams.
- **Python-only vLLM changes need no image build to screen.** Bind-mount the
  changed files from the checked patch branch over the last candidate image via
  `container.mounts` (the image imports vLLM from `/opt/spark3/candidate/vllm`).
  Build only for promotion.
- **Queue arms unattended** with a `setsid -f` chain, `ssh -n` and logs in
  `~/tl-logs`. Keep other GPU work off the nodes while an arm is timing.
- **Check every candidate's boot log for silent fallbacks** (`disabled`,
  `fallback`, `compile failed`) and for any JIT compilation after readiness.
  The L2 weight prefetch was advertised from r5c but failed to compile until
  r5i, and only the boot log said so.
- **Shared packages move forward only.** When components disagree on a shared
  runtime package, patch the lagging pin forward. The image build checks every
  CuTe DSL consumer's requirement; do the same for any new shared package.

## Memory and safety

- dgx1 hosts rank 0 and the API server, so it is always the tightest node:
  6.44 GiB minimum MemAvailable under load on r5j. Check its headroom before
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
  service before rebooting a node.

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

- **Draft length by load.** Five drafts pay off for code and JSON but cost
  low-acceptance prose at eight streams. `num_speculative_tokens_per_batch_size`
  exists in our vLLM tree and is untested.
- **Verification objective.** The current rule picks the verify length that
  maximizes expected accepted tokens per unit of profiled cost at each step.
  Maximizing expected tokens minus a running-rate-weighted cost may be closer
  to the long-run optimum. Neither the marginal rule (0006) nor the cost-aware
  dead-row count (0011) beat the current configuration.
- **Cost-profile noise.** The startup cost profile takes the median of five
  replays, and is a likely source of the ~3% boot-to-boot variation. Pinned cost
  curves (0005) did not remove the sample noise.
- **Determinism.** Temperature-0 outputs differ within one boot even at one
  stream. Suspects: the B12X atomic-scatter MoE combine and atomic split-K.
  `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` fails B12X preparation. A deterministic
  mode would make every A/B cheaper.

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
- **Real-text prefill is flat at 3.8k tok/s.** The last chunk profile
  (4K: MoE 353 ms, attention 272, mHC 152, NCCL 142, dense 101) predates
  sequence-parallel prefill; profile it again before choosing a target. Prefill
  all-reduces are limited by the RoCE link at ~145 Gb/s. Check whether large
  transfers use every available port and PCIe path.
- **Missing bench suites:** decode throughput while another request runs a long
  cold prefill (and whether smaller prefill chunks help it), and a probe for
  prefix-cache retention around block boundaries.

## Boot (123 s)

Container start to API: engine spawn 20 s, distributed init 11, model build 5,
B12X shard routing 7, weight read 15 (NVMe rate for 101 GB), post-load and B12X
preparation 24, KV setup and warmup 16, graph capture ~8 s of real work.

- **API server setup:** about 8 s before the engine starts, mostly tokenizer
  `from_pretrained` copies and conversion plus module imports. Audit import
  time.
- **Post-load preparation (24 s):** cache B12X routing metadata and prepared
  state between boots.
- **Per-rank processed-weight image:** the largest and riskiest boot lever. It
  must preserve the MLA absorbed views and the `e_score_correction_bias`
  reference.
- **Already measured, no boot gain:** persisted driver and TileLang JIT caches
  (kept anyway), `NCCL_GIN_ENABLE=0` (communicator init is 0.5-0.6 s),
  `--skip-mm-profiling` (also costs dgx1 0.1 GiB).
- **Pitfalls:** a persisted autotune cache holding per-rank entries can deadlock
  a TP > 1 boot; an interrupted boot can leave corrupt CuTe cache files; warming
  the page cache takes MemFree that the GB10 allocator needs.

## dgx1 headroom

- **Display reserve:** each node's firmware sets aside memory for a display. On
  dgx1 and dgx2 nothing is connected; dgx3 has an HDMI output connected. Using
  that reserve for KV needs `nvidia-drm` fbdev off, so it is the owner's
  decision. Verify MemAvailable neutrality and bandwidth on all three nodes
  first.
- **Vision tower:** it is replicated on every rank, including dgx1.
- **Do not raise `min_free_kbytes`;** it eats the margin the guards protect.

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
