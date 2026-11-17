# TODO

Hints for the next improvement runs. These notes are not configuration authority:
`config/cluster.json`, the current baseline manifest and `docs/current-state.md`
describe what runs. Every figure here was measured on this stack; each idea
enters as an ordinary experiment under `experiments/` and is promoted only by the
owner.

## Recent candidate decisions

- **Candidate `61` (in progress)** ([`experiments/2026-11-16-c61-decode`](experiments/2026-11-16-c61-decode/README.md)) stacks on `candidate-60` with decode and infrastructure improvements verified through `c61-r7` (`7bec4cd0b`, 2-boot `c1` mean `83.48 tok/s`) on `tp4-ring` (`dgx1..dgx4`), with combined `c61-r8 + r9` staged on `origin/main` (`c61-r9-screen-tp4`):
  - Integrated certified 2-pass `W_q` Markov vocab screening + per-row $\ell_2$ scale search + extreme-outlier sign-clamping (`Rank 2`), BF16 compressor replication up to `TP<=4` (`Rank 5`), `PR #1225` (`0817..0819`) + `pipe` + `<2>` dense `indexer_score_v41` + gated `_decode_index_split` (`Rank 6`; also supersedes `PR #1086`), grouped 3-layer DSpark `context_kv` + strided `front.py` (`Rank 10`), `PR #995` CopyDrafter guards/gentle ramp (`Rank 13`), fixed `PR #1072` `shm_broadcast.py` 2 ms post-wakeup spin anchor (`Rank 14`: eliminated 5s stalls, `ttft_p99 = 0.133s`), `PR #1081` shard-local sampled verification + `PR #988` sparse top-k Markov (`Rank 15`), `PR #1078` 48-row decode projection tile (`Rank 16A`), `PR #984` UMA allocator hygiene (`Rank 17`), `PR #987` 128-row `WO-B` / `_FULL_DEPTH_VARLEN_DECODE_MAX_REQS = 16` / `all_reduce_rank_order`, `Decode #3` (`wo_b` and `shared_up` `tile_contiguous=True`), `Decode #5` (`aux_stream` priority `-3` + `_PriorityCUDAGraphExec` `UseNodePriority=0x08`), `Decode #6` (`moe_down` PDL prologue weight preload + `LayerChain` fix), `tilelang_rejection.py` `initcheck` uninitialized-read guard (`705 passed, 0 errors`), `qwen3_dspark.py` `apply_bias_topk` deduplication, and `PR #1222`/`#1223` lab fixes (`124 passed` in `test_lab.py` + `test_source_locks.py`). (`PR #967` `sparse_mla.py` TIRx compiler fixes verified byte-identical in `candidate-60` across all 6 routes.)
  - Screened through `c61-r7` (`7bec4cd0b` boot 1: `c1 = 83.34 ± 5.91%`, `c16 = 271.47`; `c61-r8-screen-tp4` synced to `origin/main` `6b4fdf6be` = repeat boot of `c61-r7`: `c1 = 83.63 ± 5.93% tok/s`, `27.88 ms/step`, `2.340 tok/step`, `ttft_p50 = 0.1136 s`, paired `prose 86.95 dec`, `code 89.34 dec`; `c16 = 270.58 ± 1.31% tok/s`; `0` diffs across all 6 greedy determinism scenarios vs `c60-base` `0a2dacdd...` / `12fd8257...`, 2-boot `c61-r7` mean `83.48 tok/s`). Combined `c61-r8 + r9` (`wo_b` + `shared_up` `tile_contiguous=True`, `moe_down` PDL weight preload + `LayerChain` fix, and `vocab.py` finite extreme-outlier `W_q` sign-clamping; `140 + 16 + 38 + 6` pytest tests passed on `sp8`, `0` PDL violations) is queued in `c61-r9-screen-tp4`.
- **Candidate `60`** ([`experiments/2026-11-16-c60-prefill`](experiments/2026-11-16-c60-prefill/README.md)) builds on `candidate-59` (via `#1218` / `#1219`) in three gated steps across switched TP4 (`sp1..sp4`) and switched TP3 (`sp5..sp7`):
  - Integrated PR `#992` `bf16x2` producer casts, `indexer.py` `bfloat162` SMEM + warp-shuffle `block_scan` top-$k$, `attention.py` SP prefill compressor/front async overlap + deferred `finish_front` + `cacheable = index_rows <= chunk_rows`, PR `#983` Clock-Pro 2-bit Engram prefill shield + `SPARK_DS41F_ENGRAM_PIN_SCALES`, PR `#1010` hybrid KV retention/eviction, PR `#978` multimodal prefix-cache fastpath, PR `#1006` uncached GPU post-load vectorizations / pressure-gated `release_device_memory_under_pressure` / lazy imports / first-prefill `_warm_model_runtime` warmup (dropping `postload_cache` after measuring slower reload than uncached GPU vectorization on GB10), piece-level parallel prompt tokenizer caching (`vllm/tokenizers/deepseek_v41.py`), PR `#1212` (rebased `#1047`) literal image-token escape, `defer_loading: true` + `tool_reference` inline tool-schema resolution (`serving.py`, `deepseek_v41.py`, `deepseek_v41_encoding.py`), and stream/lifecycle/parser hardening (`sp_prefill.py`, `engram_rows/rows.py`, `deepseek_v41.py`, `deepseek_v41_encoding.py`; evaluated and removed opt-in `SPARK_DS41F_PREFILL_MIX_CHUNK` from PR `#1150` during optionality cleanup).
  - Qualified with zero regressions on `tp4-switched` (`model`: `128/128` greedy / `100.00%` identical, `KL mean 0.000e+00`, `8/8` gate determinism/greedy `0` diffs; `grid` `-0.07% ±0.13%` `same`; `c16` `+0.17%` / `+0.15%` `same`; `c1-decisions` `82.91 tok/s`; `prefill 262K` `default-text` **`+2.10% ±0.64%` (`better`, `9,833.2 tok/s`)**, `+4.50%` over `c58-final`; `source` prefill `32K/131K/262K` `9,521 / 9,924 / 9,830 tok/s`, with Step 1 hitting `10,000.1 tok/s` mean at `131K`; `bench` `quality` `5/5` pass) and `tp3-switched` (`model`: `100.00%` identical, `8/8` gate `0` diffs, `grid` `-0.03% ±0.12%` `same`, `c8` `-0.25%` / `-0.23%` `same`, `sp5` startup `min_available_gib` `5.90–5.97 GiB`; `tp4-ring` remained held and untested).
- **Candidate `59`** ([`experiments/2026-11-15-c59-prefill`](experiments/2026-11-15-c59-prefill/README.md)) builds on `candidate-58` in three gated steps across switched TP4 (`sp1..sp4`) and switched TP3 (`sp5..sp7`):
  - Integrated bounded UTF-8 suffix detokenizer priming (`detokenizer.py`), zero-copy `int32` `PickleBuffer` `NewRequestData` admission IPC (`output.py`, `request.py`, `buffer_utils.py`), `bf16x2` / `cvt.rn.bf16x2.e2m1x2` dequant (`sparse_mla.py`), dense `indexer_score_v41` `rows >= 64` `R=4` BF16 dual-row SMEM reduction (`indexer.py`), fused `prefill_gate_up -> moe_swiglu_cast` with 64-col `(gate, up)` slab interleaving (`moe.py`), and PR `#1005` MoE warp-shuffle dispatch scan, sync/PDL tightening, and cascaded `moe_swiglu_cast` preloads (keeping `c58` BF16 SwiGLU intermediate rounding).
  - Qualified with zero regressions on `tp4-switched` (`model`: `128/128` greedy / `100.00%` identical, `KL mean 0.000e+00`, `8/8` gate determinism/greedy `0` diffs; `grid` `+0.24%` `same`; `c16` `+0.11%`/`+0.13%` `same`; `c1-decisions` `82.94 tok/s`; `prefill 262K` `default-text` `+2.35%` `better`; `source` prefill `32K/131K/262K` `9,548 / 9,886 / 9,757 tok/s`, `+1.28–1.53%` over `c58-final`) and `tp3-switched` (`model`: `100.00%` identical, `8/8` gate `0` diffs, `grid` `+0.02%` `same`, `c8` `+0.12%`/`+0.14%` `same`; `tp4-ring` remained held and untested).
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
   the global intermediate round-trip and extra kernel launch per layer. On
   `sp8` (`H=5120, I=576`, `E=384, topk=6`), `fuse_swiglu=True` at
   `decode_block_M=16` is 100% bit-identical (`diff_q=0, diff_sf=0` across
   distinct and shared-expert multi-row tiles) and cuts Cold-L2 `routed` MoE at
   `live=1` (`182.5 -> 171.6 us`), with smaller overlapped `Shared + Routed` gain
   (`-1.1` to `-0.7 us` at `live=1..3`, `+4.4 us` at `live=4` vs `pm3+08`) because
   `fuse_swiglu` mandates `trigger_after_loop=True`. Full FC1->SwiGLU->FC2
   persistence remains open.
2. **Direct producer writes (`WO-B` / `moe_combine`) into registered `sparknet`
   RoCE send buffers** (~1.3–1.6% decode ceiling). `1107` overlaps collective
   launches via PDL (`0753`, `0755`, `0756`), but `roce_all_reduce` still stages
   an explicit `cudaMemcpyAsync` from the producer output tensor into
   `comm_buf` (116 launches per step). Writing final BF16 outputs directly into
   the registered RoCE send buffer removes the staging copy.
3. **Q-B, indexer Q-B, WO-B, and SHARED-UP tile-contiguous weight layout.** Q-B and indexer
   Q-B already pack `[N // 128, K // 128, 128, 128]` tiles (`TILE_CONTIGUOUS_FP8`).
   On `sp8` (`M=3`, `block_M=16`, Cold-L2 across 10 copies), `wo_b` `(5120, 2048)`
   is bit-identical with `tile_contiguous=True` and speeds up from `51.93` to
   `48.90 us/call` (`-3.03 us/call`, `x40 = -0.12 ms/step`); wired in `c61-r8`
   for `(5120, 3072)`, `(5120, 2048)`, `(5120, 1024)` (`gemm.py` + `wo.py`
   `_PeerWeight`, `_sp_prefill_calls`, and `bind` unpacking/repacking for
   heterogeneous TP ranks; `69 passed` across `test_deepseek_v41_tilelang_wo`,
   `test_tile_contiguous_fp8_matches_row_major`, and `test_deepseek_v4_1_attention`).
   In `c61-r9`, shared-expert `gate_up` shapes `(1536, 5120)`, `(1152, 5120)`,
   `(1024, 5120)`, `(768, 5120)`, `(640, 5120)` (`TP3..TP8`) were added to
   `TILE_CONTIGUOUS_FP8` (`gemm.py`; `sp8` Cold-L2 `SHARED_UP(1152x5120)`,
   `M=6, Mp=16`, `COPIES=8` + `40 MiB` flush: `36.15 -> 34.94 us/L`,
   `-1.20 us/L`, `40L = -0.048 ms/step`, bit-exact; `16 passed`, `0` PDL violations).
4. **Vocabulary head (`128`-row token tile and per-tile epilogue summary).**
   - **A 128-row token tile:** a head call with 65–128 rows on TP4 `c16` runs a
     second 64-row tile that re-reads and unpacks the 248 MB shard from DRAM.
     A 128-row tile fits in ~96 KB of shared memory with two activation stages
     and four weight stages, keeping the same K accumulation order (~1% ceiling
     at sixteen streams on TP4).
   - **Sampled top-$k$/LSE summary in the head epilogue:** greedy argmax in the
     head epilogue is active; emit a compact per-tile summary (top-$k$
     values/indices plus tile max and log-sum-exp) so sampled (`temp=1.0`)
     decode and verification also reduce compact tile summaries rather than
     scanning full FP32 `[M, vocab_shard]` logits.
5. **A second stream inside the decode graph (and incremental `old + new`
   indexer split).** Overlap the shared expert with routed MoE, DSpark context-KV
   preparation on tap layers (`1, 30, 60`) with that layer's MoE block, and
   attention compressor / indexer-key writes with the query path on a second CUDA
   stream captured into the graph via `cudaEventRecord` / `cudaStreamWaitEvent`.
   On KV-source layers, decouple the compressor from the indexer critical path
   by scoring the existing compressed prefix (`n_old`) and running radix top-$k$
   on the main stream immediately after `wq_b` while the side stream finishes
   the compressor and scores only the `n_new` ($\le M$) newly written blocks,
   merging the `n_new` scores into the sorted `k` prefix keys in a lightweight
   merge kernel before MLA. Passing `cudaGraphInstantiateFlagUseNodePriority`
   (`0x08`) via `_PriorityCUDAGraphExec` (`cudagraph_utils.py`, retaining `self._graph`
   until `reset()` so `CUDACachingAllocator` pool refcounts stay nonzero across
   shared-pool captures) with highest-priority `aux_stream` (`priority=-3` on
   SM121 `(0, -3)`, `torch_utils.py`) landed in `c61-r7` (`7bec4cd0b`): on `sp8`
   Cold-L2 40-layer `Shared + Routed MoE` (`H=5120, I=576`), `pm3 + 0x08` beats
   `p0` by `-2.5` to `-5.7 us/call` across `live=1..4` (`live=1`: `225.7 -> 223.2`,
   `live=2`: `342.0 -> 337.8`, `live=3`: `455.7 -> 451.6`, `live=4`: `570.8 -> 565.1 us`;
   `-0.10` to `-0.23 ms/step`), and `c61-r7-screen-tp4` measured `c1 = 83.34 tok/s`
   (`35.81 steps/s`, `27.93 ms/step`, `2.336 tok/step`; paired `prose dec = 86.64`,
   `code dec = 89.40`), `c16 = 271.47 tok/s`, and `greedy = identical`. Still open:
   overlapping DSpark context-KV preparation and compressor/indexer-key writes on
   `aux_stream`, plus incremental `old + new` indexer scoring. First bound how
   much of the latency-bound bucket is serial dependency rather than SM
   occupancy; if GB10's 48 SMs or 99 KiB per-CTA shared memory serialize
   concurrent streams during tail waves, screen co-tuning pipeline stages /
   dynamic SMEM on the overlapping kernel pair or carving out a small CUDA Green
   Context SM partition for the latency-critical side stream.
6. **Two-phase sparse-MLA decode KV and PDL prologue weight prefetch before
   `griddepcontrol.wait`.** In decode sparse-MLA, the prior sliding-window KV
   rows (`pos < pos0`, i.e., all but the current step's rows) and—on layers that
   reuse `topk` indices from an earlier `candidate_source_layer`—all `ktop`
   compressed KV rows are already committed in memory before the upstream
   `wq_b` / `wkv` / `topk` kernel finishes. Issuing global-to-shared loads for
   those already-valid KV slots before `griddepcontrol.wait` overlaps LPDDR5X KV
   fetch latency with the upstream tail, waiting on `griddepcontrol.wait` only
   for the newly written KV rows and `q`. Likewise, in PDL-chained decode
   projections and GEMMs (`0739`, `0755`, `0756`, `MOE DOWN`, `WO-B`), ensure
   stage-0 immutable weight and scale tile loads issue before
   `griddepcontrol.wait` so LPDDR5X weight reads overlap the producer's final
   wave before the consumer waits on the activation tile. In `c61-r8`,
   `grouped_fp8_fp4_gemm(..., gather=False)` (`moe.py`) stages stage-0 TMA
   weight tiles (`tl::tma_load`) and scale codes (`SFB`/`SFB8`) before
   `cudaGridDependencySynchronize()` while keeping producer-written `A`/`SFA`
   strictly after (`140 passed` in `test_deepseek_v41_tilelang_moe.py`; `sp8`
   Cold-L2 Routed MoE full `M=4: 497.15 -> 465.46 us/L [-1.27 ms/40L]`, `M=6:
   736.27 -> 699.36 us/L [-1.48 ms/40L]`; `M=1..3` unchanged at `134.8/243.8/353.4 us/L`),
   and `c61-r9` fixed `grouped_fp8_fp4_gemm` empty-expert entry (`if expert < 0: T.thread_return()`
   before `rows_shared`, without `TRIGGER` before `WAIT`) + `EARLY_PARAMETERS` (`LayerChain`
   `4/4` passed, `0` PDL violations). On `sp8`, `_mxfp8_gemm` stage-0 weight preload (`WO-B`,
   `WQ-B`, `WQA_WKV`, `SHARED_UP`) saved only `-0.01..-0.02 ms/40L` (`pf_l1_k01`) while
   SMEM peeling regressed `+0.05 ms/40L` (`b0_shared`), and `sparse_mla.py` `SwaIndices`/
   `SwaLengths` preload before `pdl.chain()` races on Layer 0 (`_write_decode_metadata` in
   `attention.py` writes them immediately prior); neither was adopted.
7. **Forward-fusing `mHC` stream finalize and MXFP8 quantization into consumer
   projections (plus fused `embed` + Layer-0 `project_streams`).** Unlike
   1-launch atomic `project_and_finalize_streams` (which regressed in `#946`),
   keep `project_streams` writing the unnormalized collapsed stream `C` and the
   partial sums-of-squares `Q`, and have the consumer decode projections
   (`fused_wqa_wkv` and MoE gate/shared-expert input) reduce the partial sums
   `Q`, multiply by `rsqrt * gamma`, and quantize to MXFP8 inside their
   prologue—removing the global BF16 normalized activation round-trip and
   standalone MXFP8 cast launches. At model entry, also screen fusing token
   embedding lookup, 4-stream broadcast, Engram `token_map` lookup, and Layer-0
   `project_streams` into one launch.
8. **Piggybacking KV-RoPE onto `wq_b` via extra tail CTAs.** `0748` and `0749`
   fuse `wq_b + indexer.wq_b` with query RoPE and MXFP4 quantization; screen
   appending tail CTAs to `wq_b`'s grid that branch to apply RoPE and cache
   writes to `wkv`'s newly projected KV rows concurrently with `wq_b`,
   eliminating 61 separate KV-RoPE launches per step.
9. **Intra-request multi-row `KPool` reuse in decode `indexer_score_v41` (and
   deep-context TP row split).** Under speculative verification, all $V$
   verification rows of a request share the identical `PageTable` and `KPool`
   prefix up to `pos0`, yet `indexer_score_v41` (`0029`) launches an independent
   CTA per query row that streams the full 64K–1M `KPool` prefix from LPDDR5X $V$
   times per rank (and $V \times \text{TP}$ times across the cluster, since
   `0025` splits indexer rows across TP ranks only in SP prefill). Loading each
   shared prefix tile `K_shared [64, 128]` and `SFK_shared` once per request and
   scoring all $V$ verification rows of that request against it before advancing
   `p0` cuts decode `KPool` traffic by $V\times$. Inside the fused multi-row
   CTA, keep all $V$ verification rows' query slices (`q` across heads)
   permanently in registers across the `KPool` key-block loop (avoiding repeated
   shared-memory `LDS.128` reloads per key block and query row) and reduce
   per-lane dot products across 4 indexer heads simultaneously with a 6-shuffle
   butterfly scatter-reduce (`xor 16, 8` leaving one head per lane, then
   `4, 2, 1`, preserving exact pairwise FP32 addition order for bit-identity)
   instead of 20 shuffles per `(key_block, query_row)`. Above ~64K context this
   can pair with a TP row split and one-shot `sparknet` RoCE gather of
   `topk_indices_buffer[:rows]` (`2 KiB` per row).
10. **Same-request multi-row SWA KV reuse in decode `sparse_mla_fwd_v41`.** On
    SWA-only layers (`idx_width = 0`, `swa_width = 128`, `segments = 1`) and in
    the SWA chunk of sparse-MLA layers (`0740`), each CTA currently handles a
    single query row even though all $V$ verification rows of a request attend to
    the identical 128-token SWA prefix (`pos < pos0`) from the single shared KV
    head (`num_key_value_heads = 1`), differing only in their causal mask over
    `[pos0, pos0 + V)`. Grouping a request's verification rows inside one CTA
    amortizes the 576 B/token SWA KV loads from LPDDR5X across all $V$ rows.
11. **All-reduce arrival skew and RoCE proxy core affinity.** Rank 0 also runs
    the API server, and `sparknet` proxy threads run without explicit CPU core
    affinity on GB10's mixed core types; test pinning them to fast cores.
12. **A RoCE one-shot broadcast for adaptive verification.** Adaptive
    verification broadcasts rank 0's confidence snapshot every decode step
    through NCCL (`adaptive_verification.py`). Bound its share of the step from
    decode traces and screen `sparknet`'s one-shot broadcast path if material.
13. **Mid-network (Layer 16/20) dead-row routed-MoE pruning at single stream
    (`c1`).** At `c1`, no other requests in the batch amortize routed-expert
    LPDDR5X reads (4.5 MiB per expert per rank across 40 routed layers), so
    every rejected draft row surviving `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU = 0.2`
    before Layer 0 incurs unamortized routed-expert DRAM reads. Because target
    layers are strictly causal and row-independent, pruning diverging tail draft
    rows from routed MoE at mid-network (Layer 16/20) while masking pruned rows
    as rejected in `rejection_sampler` cuts half of those wasted routed-expert
    DRAM reads while keeping all accepted target tokens 100% bit-exact.

## Speculative decoding

In `1107`, DSpark online confidence calibration (`0746`), CopyDrafter
continuation-priority tie-breaking (`0744`), descending-priority warp early
exit (`0744`), expected-yield gating against DSpark (`0745`), native-depth CUDA
graph retention (`0728`), fused single-rank Markov drafting (`0717`), and
device-side `prepare_inputs` folding (`0754`) are all active.

Open levers:
- **Single-stream (`c1`) cold-L2 routed-MoE marginal row cost and `DEAD_ROWS_TAU` break-even calibration.** On `sp8` across distinct layer weights (`>2 GiB`, cold L2 matching real serving), 39-layer `MoeExpertsV41Plan` at `M_pad=6` scales with `live_rows = 1..6` (`6 -> 12 -> 18 -> 24 -> 29 -> 34` unique experts/layer) as `4.66 -> 8.76 (+4.10) -> 13.08 (+4.32) -> 17.02 (+3.94) -> 20.57 (+3.55) -> 24.08 (+3.51) ms` (`s3/2` beats `s2/2` across warmed `live=3, 4, 16, 38, 64` at `159.1 vs 160.8`, `192.7 vs 194.6`, `622.8 vs 623.9`, `1423.5 vs 1427.4`, `2369.4 vs 2377.9 us`). Live `dgx1` `c61-r5` `prose-c1` telemetry (`4,701` steady decode steps) shows `36.3%` commit 1 token (`med 25.30 ms`), `30.0%` commit 2 (`25.60 ms`), `21.1%` commit 3 (`26.70 ms`), `8.1%` commit 4 (`28.70 ms`), `3.2%` commit 5 (`30.90 ms`), `1.2%` commit 6 (`33.50 ms`). `c61-r6` (`e0436c584`) tested IRLS warm-start priors (`_WARM_CONFIDENCE_HEAD_INTERCEPTS` + zero-preservation) + `SINGLE_REQ_CALIBRATED_ROW_COST_SCALE = 2.5` (`tau_c1 = 0.1386`) on `dgx1..dgx4`: warmup req `#00` improved `71.66 -> 73.13 dec tok/s` (`ttft` `0.283s -> 0.163s`), `code-c1` improved `114.25 -> 115.04 dec tok/s` (`med_dt 31.50 -> 29.90 ms`), and `c16` (`271.38`) + determinism (`0` diffs) held, but `prose-c1` shifted `-1.83 ms/step` (`26.58 -> 24.75 ms` mean) against `-0.163 commits/step` (`2.144 -> 1.981`, `81.38 -> 80.72 dec tok/s`), showing the pruned `pos 2..4` tail tokens earned `89.1 tok/s` marginal rate (conditional acceptance > independent confidence product). Keep warm-start priors and drop `scale = 2.5` back to `1.0` in `c61-r7`. (`PR #1002` `VLLM_DSPARK_ONLINE_ADAPT=1` measured `+0.03%` on held-out `c1` (`87.11 -> 87.14`) and `+0.10%` on coding (`160.91 -> 161.07`) in `c57-adapt` and remains opt-in.)
- **Bounded or indexed CopyDrafter history match, forward-locality cursor, and
  payoff-gated wide copy passes at 100K–1M contexts.** Even with warp
  early-exit (`0744`), a decode step that does not hit a 256-token match scans
  the full `[min_match, total_len)` history in UVA (`all_token_ids.gpu`) on
  `output_copy_stream` (~0.36–0.47 ms/step at `c16`, 1M random tokens, and up
  to ~2 ms on repeated 16/32-token boilerplate matches), contending with DSpark
  on `main_stream` for unified memory bandwidth.
  1. Maintain an incremental per-slot hash-chain index (`head[hash]`,
     `prev[pos]` over 4- to 16-token windows) updated only for newly committed
     tokens so `copy_match` is $O(1)$ in context length.
  2. Track a per-request copy cursor (`last_src`): while output follows the
     matched span, advance the cursor directly without re-searching; when the
     output edits a few tokens and re-enters the source, prefer the nearest
     candidate match at or after `last_src` (so long as its backward match is
     at least `min(best_len, 8)`) so look-alike lines or JSONL records do not
     jump to the wrong record.
  3. After consecutive fully accepted 16-row copy passes, test widening single-
     stream copy verification to 32/48/64 rows (`M=32/48/64` TileLang GEMM
     tiles with identical K-reduction order), gated by wall-clock payoff back-
     off (`confirmed_tokens / ms_wide >= 16 / ms_16`, doubling the required
     full-pass streak up to 16 whenever a wide streak underperforms 16-row
     passes).
- **Draft depth by load or recent acceptance.** Five DSpark drafts pay off on
  high-acceptance code at one stream (`+14–18%`), whereas live agent traffic and
  8–16 stream prose accept fewer tokens per step. Screen load-aware
  (`num_speculative_tokens_per_batch_size`) or per-request acceptance-aware
  draft depth schedules on recorded agent traffic and the paired servebench
  suites.
- **Coupled `(seed, pos, vocab_id)` Gumbel-Max sampling and block-wise prefix
  verification for `temp > 0`.** When DSpark drafts greedily ($q(x_t) = 1$) and
  `tilelang_rejection.py` verifies at `temp = 1.0`, exact rejection sampling
  must reject even the target model's own argmax with probability $1 - p(x_t)$
  on near-ties, and block verification ($w_i = \prod_{t \le i} p(x_t)/q(x_t)$)
  cannot lift $p(x_t)/q(x_t) > 1$ for a one-hot draft. Screen two lossless remedies:
  1. *Coupled Gumbel-Max draft and target sampling:* have both DSpark's Markov
     head and `tilelang_rejection.py` draw deterministic per-`(seed, position,
     vocab_id)` Gumbel noise $G(\text{seed}, \text{pos}, v) = -\ln(-\ln u)$ and
     select $\arg\max_v (z_v / T + G(\text{seed}, \text{pos}, v))$. By the
     Gumbel-Max trick this preserves the exact target `temp > 0` distribution
     and makes multi-row verification bit-identical to 1-token-per-pass target
     decoding while naturally accepting near-ties whenever DSpark and the
     40-layer target assign similar relative logits.
  2. *Non-degenerate draft probabilities with block verification and two-level
     quad Gumbel-Max residual sampling:* pass DSpark's draft probabilities
     $q(x_t) < 1$ into cumulative prefix importance-ratio verification
     ($h_i = w_i / (w_i + 1 - w_{i-1})$), paired with a two-level Gumbel-Max
     residual/target draw over 64-token vocabulary quads.
- **Grouped 3-layer DSpark context-KV projection.** Concatenate
  `mtp.{0,1,2}.attn.wkv` so all three DSpark layers' context-KV projections and
  RoPE execute in a single grouped launch over the tapped hidden states from
  target layers `{1, 30, 60}` rather than three separate per-layer launches.
- **Drafter reuse of the target's token selection.** Check whether reusing the
  target model's indexer top-k selection from the anchor position in DSpark
  draft layers preserves acceptance while cutting drafter attention work.
- **Certified `W_q` Cauchy-Schwarz screening for DSpark (`project_head_screened_markov_greedy_pairs`) and $\ell_2$-optimal `build_screen_weights`.** Commit `0f2eb0a85` (`fuses_screened_markov_greedy`, active in `c61-r5`+) screens single-request DSpark Markov drafting via 8-bit `W_q` (`cudagraph screened` `639.7 µs` vs `cudagraph dense` `678.6 µs` on `sp8`, `N=32320, K=2560, R=256, steps=5`). `c61-r6` added load-time per-row $\ell_2$-optimal `alpha` grid search `(1.0, 0.93, ..., 0.55)` in `build_screen_weights()` (`86 passed, 6 skipped`), and `c61-r9` fixed finite extreme-outlier rows (`full_amax > 8.0 * in_amax`) by quantizing `torch.where(torch.isfinite(w_full_f), w_full_f, w_in_f)` (clamping outliers to `u7 = 127` with their true sign rather than zeroing them to `0`) over a wider 14-point `alpha` grid down to `0.25` plus `full_base * f_alpha` candidates (`f_alpha in (1.0, 0.75, 0.50, 0.30)`), dropping extreme-outlier `row_bound` max `1.5019 -> 0.2481` (`mean 0.4973 -> 0.2442`) and improving `M=1` `screened+markov` `964.02 -> 962.56 us (-1.46 us)` bit-exactly (`38 passed`, `0` PDL violations). On `sp8` Cold-L2 at Flash TP4 head geometry (`V=32320, H=5120, K_m=512`), `lm_head_greedy` runs in `844.4 / 825.1 / 828.9 us/step` (`M=1/3/6`), `screen_head_base(M=5)` runs in `801.3 us/step`, and screened+5×Markov (`M=2/3/5`) saves `-240.1 / -221.7 / -111.0 us/step` over dense (`1338.3 -> 1098.3`, `1413.1 -> 1191.4`, `1563.5 -> 1452.5 us/step`).
- **Load-time scale-code search for online NVFP4 draft projections.** If
  revisiting `Nvfp4OnlineLinearMethod` / `prepare_nvfp4_head` (`0647`) for
  DSpark draft or Markov projections (where target verification keeps outputs
  exact), replace naive `amax` scaling (`scaled_fp4_quant`) with load-time search
  over legal E4M3 block-scale codes around the default code under a
  Hessian-weighted layer reconstruction loss and blockwise error compensation,
  improving draft acceptance with zero search overhead on the serving path.
- **Tail-slot (`survival < DEAD_ROWS_TAU`) short-suffix `CopyDrafter` splice.**
  Currently `CopyDrafter` (`0744..0745`) switches all-or-nothing per request
  against DSpark's full 5-token chain, leaving trailing slots as dead rows
  whenever DSpark's cumulative survival drops below `DEAD_ROWS_TAU = 0.2`
  after draft position 1 or 2. Keep DSpark's high-confidence prefix tokens
  (`0..k-1` where `survival >= tau`) and query `CopyDrafter` anchored on the
  extended prefix ending at `d_{k-1}` to fill remaining tail slots `k..4` when
  a short context match exists.
- **Fixed-width tree side-candidate leaves in low-confidence tail verify slots.**
  When a request's linear DSpark chain drops below `DEAD_ROWS_TAU` after depth
  $L < 5$ and `CopyDrafter` has no suffix match, instead of leaving the
  remaining $5 - L$ slots as dead rows or extending a low-survival linear chain,
  populate those tail slots with DSpark's 2nd/3rd candidate tokens as sibling
  branches at the first low-confidence depth using the verify attention mask,
  promoting the matching sibling leaf if the primary chain misses at that depth.
- **Opt-in per-request near-tie acceptance threshold ([#1211](https://github.com/christopherowen/spark-ds41f/issues/1211)).**
  Evaluate an opt-in per-request sampling parameter (e.g. `vllm_xargs`
  `spec_near_tie_k` $\in [0, 1]$, default `0` = exact verification untouched and
  bit-identical) in `tilelang_rejection.py` (`vllm_shifted(logit, row_max) >=
  ln(k) -> log_prob = 0`), scoped to `thinking` requests and tested against
  loop-containment gates ([#864](https://github.com/christopherowen/spark-ds41f/issues/864))
  since collapsing near-ties toward greedy with thinking off increases
  repetition-loop risk.
- **Cross-request verify-row allocation at high concurrency.** Compare our
  per-slot survival budget (`assign_draft_budget`) plus per-step verify length
  against ranking candidate draft slots across the whole batch by marginal
  expected accepted tokens per marginal millisecond of step cost at 8–16 streams.

## Prefill and first token

In `1107`, per-layer SP WO local MXFP8 exchange (`0737`), sequence-parallel
Engram prefill column exchange and sliced projection (`0738`), TP3 6-channel
NCCL Ring and sparse-MLA `block_H=16` (`2` CTAs/head, `0740`), unconditional
`M=64` MoE `prefill_swizzle_panel=2` (`0736`), retained CED encoder endpoints
(`0727`), CED local boundary reduction (`0742`), SP context-KV projection
(`0743`), and sequence-parallel MoE input local routing/MXFP8 gather are all
active; `candidate-59` added `bf16x2`/`cvt.rn.bf16x2.e2m1x2` dequant (`sparse_mla.py`),
dense `indexer_score_v41` `rows >= 64` `R=4` BF16 SMEM reduction, fused
`prefill_gate_up -> moe_swiglu_cast`, and PR `#1005` MoE warp-shuffle dispatch scan;
and `candidate-60` added PR `#992` `bf16x2` producer casts, `bfloat162` SMEM +
warp-shuffle `block_scan` top-$k$ (`indexer.py`), SP compressor/front async overlap +
deferred `finish_front` (`attention.py`), PR `#983` Clock-Pro 2-bit Engram prefill
shield + pinned two-plane scales (`SPARK_DS41F_ENGRAM_PIN_SCALES`), PR `#1010`
hybrid KV retention/eviction, PR `#978` multimodal prefix-cache fastpath, PR `#1006`
uncached GPU post-load vectorizations + first-prefill `_warm_model_runtime` warmup,
piece-level parallel prompt tokenizer caching (`_BoundedPieceCache` + cut-after-`\n`
`encode_batch` in `vllm/tokenizers/deepseek_v41.py`), PR `#1212` literal image-token
escape, `defer_loading: true` + `tool_reference` inline tool-schema resolution
(`serving.py`, `deepseek_v41.py`, `deepseek_v41_encoding.py`), and stream/lifecycle
hardening (`sp_prefill.py`, `engram_rows/rows.py`).

Open levers:
- **Fuse the dense indexer's top-k into its score kernel** (long-context prefill).
  - **The cost:** on candidate-60 the dense indexer (layers 2, 8 and 14) is nearly all of
    cold prefill's long-context cost. Per prompt token it is 100.9 µs at 122K and 110.3 µs at
    910K tokens on tp4-ring.
  - **What it would remove:** each 256-row sub-chunk writes its BF16 score matrix
    (256 × 455K at 910K, 233 MB) and the segment top-k reads it back. That read takes most of
    selection's 1.2-1.3 ms per call. Scoring a 4-row group's 4,096-position segments into
    shared memory and selecting each segment there would keep that traffic on chip.
  - **Exactness:** every segmentation gives the same exact top 512, since the merges are exact
    and use the same tie rule.
  - **Expected gain:** about 0.7-0.9 ms per call at 910K, roughly +1.4% cold prefill at 1M
    and +0.4% at 237K.
  - **The work:** a new kernel combining the CUDA score path with the TileLang selection
    macros (`threshold`, `compact`).
  - **Already measured:** vectorized key loads (vLLM 0819) took only 0.5-2% off score and
    select, because the read itself bounds it.
  - Evidence: `experiments/2026-11-16-indexer-prefill-exact`.
- **Rank-offset CTA remapping for sequence-parallel all-gather overlap.** `_sp_front` /
  `finish_front` (`defer_q`) already hides `q_all_gather` in attention; if new fused
  SP consumers are introduced later, remap consumer row-tile coordinates by rank offset
  (`pid_m = (raw_pid_m + rank * tiles_per_rank) % total_tiles`) so every rank
  starts computing on its local shard at $t = 0$ while peer shards arrive over RoCE.
- **NVMe interrupt coalescing stays off** on every node by the owner's decision
  (2026-10-02; prefill measured 0.7–1.6% slower with it on in
  `experiments/2026-10-02-nvme-coalescing`).
- **Live validation of prefix-cache retention and `defer_loading` tool schemas:**
  [#902](https://github.com/christopherowen/spark-ds41f/issues/902) hybrid retention/eviction
  (`test_prefix_cache_retention_probe.py`) and inline `tool_reference` schema injection
  (`defer_loading: true`) shipped in `candidate-60`; confirm live 64–100K prefix-cache
  retention on production agent traffic.

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
- **Display reserve:** in use for the embedding and output head (842.5 MiB per
  rank of the 2,032 MiB firmware carve-out; `dgx3` keeps its 8 MiB console
  framebuffer). About 1.2 GiB remains free on TP3 (and ~2.0 GiB on TP4) for
  read-once-per-step data (`experiments/2026-09-29-display-carveout-kv`).
  - **GPU-L2-cached RM scanout carve-out (`NV01_MEMORY_SYSTEM` with
    `ISO_YES | GPU_CACHEABLE_YES`) and CUDA VMM contiguous stitching:**
    `vllm/v1/worker/display_carveout.py` (`0021`/`0030`/`0804`) currently
    allocates via `/dev/dri/card0` (`DRM_IOCTL_MODE_CREATE_DUMB` +
    `cuImportExternalMemory`), which inherits `nvidia-drm`'s
    `GPU_CACHEABLE_DEFAULT -> GPU_CACHEABLE_NO` (`mem.c`) and therefore suffers
    7 GB/s random 576-byte reads (vs. 221 GB/s in `cudaMalloc`) and 2.3×
    slower large-$M$ GEMMs (`experiments/2026-09-29-display-carveout-kv`). Screen
    allocating directly through `/dev/nvidiactl` (`NV_ESC_RM_ALLOC`
    `NV01_MEMORY_SYSTEM` with `NVOS32_ATTR2_ISO_YES |
    NVOS32_ATTR2_GPU_CACHEABLE_YES` and 2 MiB huge pages, exported via
    `NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD` and imported via
    `cuMemImportFromShareableHandle` + `cuMemAddressReserve` + `cuMemMap`):
    first run `reserve_bw.cu` and `gemm_place.py` on `sp8` to verify whether
    `GPU_CACHEABLE_YES` restores full L2 cache reuse (~221 GB/s random reads,
    1.0× large-$M$ GEMM); if confirmed, use the remaining ~1.2 GiB/rank on TP3
    (and ~2.0 GiB/rank on TP4) to hold `lm_head` `W_q` screening /
    `vision_model` / resident `Engram` tables or VMM-stitch (`cuMemMap`) the
    carve-out onto the KV cache to reclaim 1.2–2.0 GiB of host `MemAvailable`
    per node while avoiding `/dev/dri/card0` DRM master contention and
    `cuImportExternalMemory` `cuda-compat` pointer quirks (`0804`).
  - **DMA-BUF VMM range export (`CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD`) and Arm64 Write-Combining / mixed-mapping barrier discipline:**
    (1) when exporting `cuMemCreate` VMM ranges for CPU `mmap` access or
    ConnectX-7 RDMA registration (`ibv_reg_dmabuf_mr`), use
    `cuMemGetHandleForAddressRange(..., CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0)` +
    `mmap(..., MAP_SHARED, dma_buf_fd, page_offset)` (CUDA 13.3+, no external
    kernel module required; yields a coherent cached mapping on GB10 unless
    `CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE` is set); (2) whenever the CPU
    copies to or from a Write-Combining (`pgprot_writecombine` / `Normal-NC`)
    or `Device-nGnRE` mapping on GB10's Armv9.2-A cores (such as the current DRM
    dumb-buffer scanout carve-out in `display_carveout.py` or ConnectX-7
    BlueFlame/MMIO doorbells in `sparknet`), avoid `glibc` `memcpy` (which can
    emit unaligned accesses or partial-line WC flushes) and instead align the
    device pointer with power-of-two `WRITE_ONCE`/`READ_ONCE` steps followed by
    64-byte `FEAT_LS64` (`__arm_st64b`/`__arm_ld64b` when
    `AT_HWCAP3 & HWCAP3_LS64` is set) or 4×-unrolled `vst1q_u8_x4`/`vld1q_u8_x4`
    NEON bursts and `dmb st`; (3) whenever a code path mixes Write-Combining
    (`Normal-NC`) and cached (`Normal-WB`) mappings to the same buffer or across
    payload + completion flags, `dmb st` only orders store-before-store—enforce
    a full `dmb sy` on the CPU paired with `cuda::thread_scope_system` (`.sys`
    acquire/release) on the GPU.
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
  83K max-effort canary soak on `1107` during release qualification, and verify
  that loop detection and thinking-budget cuts evaluate every committed token
  position across multi-token verify passes (`1..5` DSpark or `16..64`
  CopyDrafter rows) and truncate at the exact first token index where the rule
  holds so truncation is invariant to speculative verify width.
- **DSML token provenance ([#876](https://github.com/christopherowen/spark-ds41f/issues/876),
  [#948](https://github.com/christopherowen/spark-ds41f/issues/948), PR
  [#949](https://github.com/christopherowen/spark-ds41f/pull/949)):** ensure
  quoted DSML markup inside normal reply text or tool-call string arguments is
  distinguished from genuine DSML structural tokens on top of `0643`'s streaming
  string-argument extender, and handle stray quoted `</think>` tags in the
  answer body when the model quotes `</think>` inside reasoning (closing the
  thinking span early and emitting a second `</think>` before a tool call).
- **Answer-only `repetition_penalty` scope (`output_bin_counts` without
  `prompt_bin_counts`):** vLLM's `repetition_penalty` kernel currently penalizes
  tokens appearing in `prompt_bin_counts` as well as `output_bin_counts` (unlike
  `presence_penalty` and `frequency_penalty`), which penalizes identifiers,
  numbers, and lines copied from the prompt during code edits and lowers
  `CopyDrafter` / DSpark acceptance when callers pass `repetition_penalty > 1.0`.
  Screen restricting `repetition_penalty` to generated output tokens (or a
  bounded trailing answer window).
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
- **TMA stage releases outside DS4.1's active kernel set**
  (`experiments/2026-09-30-proxy-fence-audit`): all serving kernels are fenced
  (and enforced in the compiler via `tilelang/0001..0013` in `1107`); keep the
  compiler barrier tests in the release gate whenever TileLang is refreshed.

## Upstream and hygiene

- **No fallbacks** (owner, 2026-10-14): every collective or kernel dispatch path
  must either be the tested production path or raise with a traceback.
- **No silent test-gated defaults:** best production paths must be active by
  default without requiring environment opt-ins or disabling guards.
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
  (`experiments/2026-11-04-copy-overlap-rework`); lowering `effective_dead_rows_tau`
  ~3.7× at `c1` (`PR #1003` sunk-cost rule, `c61-r1-2`: pushed low-survival tail
  rows into 384-expert MoE, adding `+1.78 ms/step` and regressing `c1` `-3.25%`);
  cohort-atomic budgeting + `COALESCED_ROW_COST_SCALE=0.28` (`PR #953`/`#986`,
  `c61-r3`: pushed `c16` `sched/step` to `40.05` and regressed `c16` `-3.3 tok/s`).
- **Decode kernels:** 1-launch atomic last-CTA `mHC project_and_finalize_streams`
  for `M <= 16` (`experiments/2026-11-06-mhc-project-finalize`, #946: 12–34%
  slower than 2-launch PDL due to atomic/fence serialization and register
  pressure); fusing decode `WO-A` split-K reduction into `WO-A` (`0737` applies
  only to prefill SP WO; decode `WO-A` uses `split_k=4` across `K=2048` and
  `WO-B` reads all `1280` reduced columns per CTA); bounded L2 access-policy
  retention windows on GB10 (#928, no gain / regression on unified memory);
  incremental GPU block-table metadata gathers (#932, negligible step impact vs
  state complexity); `0714`/`0716` score-ID host transfer and `0718` TP1-only
  embedding ID prep (#913, #914: unmeasured / TP1-only); 24-bit (top-3-byte)
  `mHC: fn` weights (`experiments/2026-11-08-mhc-fn24-and-green-ctx-audit`:
  superseded by `0092`'s 3x-TF32 exact FP32 split; `99.61%` of checkpoint
  `hc_*_fn` weights have non-zero low 8 bits so 24-bit storage is not bit-exact,
  and unpacking `uint16+uint8` regresses L2-hot `project_streams` latency by
  `+2.4–6.1%` across `M=1..64`); 2-split IEEE RN TF32 in `mhc.py` (`c61-r2`:
  shifted `reference_sha256` from `0a2dacdd...` to `81a029f1...`); CUDA Green
  Context (`CUgreenCtx`, `2/46` SM split) isolation for side-stream L2 prefetch
  (`experiments/2026-11-08-mhc-fn24-and-green-ctx-audit`: `+1.71 µs/fork`
  cross-context event wait inside `CUDAGraph` replay vs `+0.155 µs/fork` on a
  same-context stream, regressing per-layer decode latency by `+5.3` to
  `+8.1 µs/layer` [`+1.4–3.5%`] against `VLLM_L2_PREFETCH_GRID=2`); Decode #10
  multi-row SWA KV sharing in `sparse_mla.py` (`sp8`: `rows=1..6` adds only
  `+0.14..0.21 µs/layer` because L2 already serves shared SWA KV); Decode #6
  speculative pre-sync reads of producer-written dynamic buffers (`PDL CHAIN`
  forbids reading upstream outputs before `T.pdl_sync()`, and `sparse_mla.py`
  cannot preload `SwaIndices`/`SwaLengths` before `pdl.chain()` because
  `_write_decode_metadata` in `attention.py` writes them immediately prior on
  Layer 0 / cache-group roots; immutable checkpoint weights like `moe_down`
  `B`/`SFB` preloaded in `c61-r8` are safe per `pdl.py`, whereas
  `_mxfp8_gemm` stage-0 weight preload on `sp8` saved only `-0.01..-0.02 ms/40L`
  with `pf_l1_k01` and regressed `+0.05 ms/40L` with `b0_shared`, not adopted);
  Decode #8 `q` RoPE piggyback (`0.59 µs/layer` on `aux_stream`, KV-RoPE already
  fused in `attention_front`); and Decode #13 mid-network Layer 16/20 routed-MoE
  pruning (DS4.1 safetensors carry no intermediate Layer 16/20 LM head/probe
  weights).
- **Prefill:** M32 MoE tile (+1%); fused M32 MoE (no gain, less headroom);
  `NCCL_PROTO=Simple` (no gain); `--max-parallel-prefills 8` (four-64K mean TTFT
  41 → 67 s); sequence parallelism below 205 tokens (+12 ms TTFT on short
  prompts); `8` or `12` NCCL Ring channels on TP3 SP prefill
  (`experiments/2026-11-06-nccl-sp-channels`, `6` channels won while `8` and
  `12` regressed); dense `indexer_score_v41` `<2>` (`2 <= rows < 64`) with
  `16 KiB` `contrib` SMEM (`+0.34%` `@4K` decode regression vs `<1>` + `8 KiB`
  `contrib` kept for `rows < 64`) and `<8>` (`rows >= 128`, `B_reg[8]` register
  spill slowed `rows=1/64/4096`, `experiments/2026-11-15-c59-prefill`); on-disk
  `postload_cache` serialization of packed BF16 weights and expert scale codes in
  `candidate-60` Step 3 (reload from `/cache` was slower on GB10 unified memory
  than PR `#1006`'s uncached vectorized GPU `pack_bf16_weight` and
  `encode_expert_scale_codes` kernels, so `postload_cache` was removed before
  qualification, `experiments/2026-11-16-c60-prefill`); opt-in
  `SPARK_DS41F_PREFILL_MIX_CHUNK` (PR `#1150`, removed during `c60` optionality
  cleanup); `wq_b` prefill split (`1.132 ms` single vs `1.134 ms` `4096+12288`
  at `M=16384` BF16 on `sp8`); head-subset Cauchy-Schwarz indexer scoring (`sp8`
  `IndexerV41Plan` `rows=4096`, `page_size=256`, `topk=512` measured `score`
  `30.18/58.20/113.70/277.43 ms` and `select_blocks` `4.62/9.87/19.83/39.32 ms`
  [`gather_select` `1.68..1.71 ms`] across `131K/262K/524K/1M`, only encoder layers
  `[2, 8, 14]` run full-chunk dense scoring, and `weights_proj + wq_b` head
  importance is spread evenly at `0.341..0.363` top-8/32 and `0.613..0.621`
  top-16/32); runtime BPE boundary token rewrite (mutating prefix tokens changes
  deterministic sampling/logprobs; `_BoundedPieceCache` retained instead); and
  global `CUDA_MODULE_LOADING=EAGER` (boot/headroom cost vs `_warm_model_runtime`).
- **Collectives:** NCCL for decode all-reduces (75–86 µs against one-shot RoCE's
  13–19 µs, `experiments/2026-10-03-collective-policy`); 8 or 16 NCCL channels
  on TP4 serving prefill (0.7–1.2% over four, intervals spanning zero);
  bidirectional NCCL rings (no general gain); streaming the relay in 32–64 KiB
  chunks (a consistent bulk penalty, `experiments/2026-10-03-relay-progress`);
  GPUNetIO direct mode (`dgx1` rebooted; the runner now rejects it).
- **Drivers:** R610 and CUDA 13.3 libraries
  (`experiments/2026-10-02-driver-cuda-refresh`): no consistent gain, and stock
  R610 faults on large copies with 64 KiB pages.
