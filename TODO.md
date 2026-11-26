# TODO

Hints for the next improvement runs. These notes are not configuration authority:
`config/cluster.json`, the current baseline manifest and `docs/current-state.md`
describe what runs. Every figure here was measured on this stack; each idea
enters as an ordinary experiment under `experiments/` and is promoted only by the
owner.

## Recent candidate decisions

- **Candidate `64` (built & 2-boot qualified 2026-11-21 on `tp3-triangle`, unpromoted)**
  ([`candidate-64`](experiments/2026-10-26-spark-ds41f-candidate/candidate-64/README.md),
  image `spark-ds41f:candidate-64`, `sha256:50b29406...`, vLLM `0001..0843`, tree
  `7464b04c3cc7`): builds on `candidate-63-r5` (`0830..0832` universal auto-sized
  contiguous CUDA pool without `expandable_segments`, 2-pass `512 MiB` `hugetlb`
  pageblock reset, and `engine_grid` TP rank sync; `3`-boot fresh-reboot drift
  `+1.21% -> +0.02%`, `O13 = 189..209`) plus 11 verified patches (`0833..0843`):
  `mhc` `async_sinkhorn` (`0833`, `#1264`: `-1.00%` grid, `c1 tpot_p50 -0.34 ms`),
  exact BlockMax-32 wide prefill (`0834`, `#1265`: `+2.11%` at `1M`),
  group-aware wide-decode `WO-A` (`0835`, `#1267`: `8x48 -0.410 ms`), real-row
  profiler KV isolation (`0836`, `#1270`), defensive `T_decode >= 50 ms` async
  prefill stall attribution (`0837`, `#1273`), exact full-vocab batched DSpark
  top-2 (`0838`, `#1274`: `model` `acceptance_length +0.72%`, `c8` faster),
  content-addressed warm-cache `prepare_declared()` index (`0839`, `#1275`:
  `-13.8 s` warm boot), single-segment SWA `extend` route (`0840`, `#1277`),
  per-request mixed-batch sampling split (`0841`, `#1278`: `-47.3%` / `-43.7%`
  compute + `4x-16x` less RoCE), `#864` loop/thinking-budget verify invariance
  (`0842`, `#1279`: 3 bugs fixed), and dedicated `R=3` decode dense
  `indexer_score_v41` (`0843`, `#1276`: Cold-DRAM `-17.1%` at `8x6 @256K`,
  `-27.4%` at `8x6 @1M`, `-24.2%` at `16x6 @1M`). Also accompanied on `main` by
  zero-fork E-core `memguard` (`#1269`) and Sonar `engine_grid` prefix-cache
  reset in `finally` (`#1266`). Measured neutral and dropped: `VLLM_SHARE_PYNCCL_COMMS=1`
  (`#1264`), `WO_BANDS=""` & `NEXT_MB=15` on `tp3-triangle` (`#1268`). Audited &
  bounded without patches: `M=128` CED tail fast-path (already routed), tail-slot
  `CopyDrafter` splice & tree verify leaves, `#876`/`#948` (already solved in
  `0824`/`0829`), sampled draft margin (already greedy DSpark), answer-only
  `repetition_penalty` (canonical vLLM semantics). Built-image tests on `dgx4`:
  **`2,315 passed, 40 skipped, 0 failed`** (`15/15` PDL). Live 2-boot
  `tp3-triangle` qualification (unrebooted `1120`-aged nodes, `O13 = 3/13/7`,
  `0` failed): **100% bit-exact** (`128/128` greedy & top-20 logprobs vs `1120`,
  `256/256` across boots, `KL = 0`, `perplexity = 1.313727` [`+0.000%`],
  `determinism --wide` `0/0` diffs), warm boot `108.7 s` (`-25.9 s` vs `1120`),
  multi-stream grid `-1.72%` vs `1120` (`-1.44%` vs `c63` across all 15 shapes),
  `1M` prefill `7,000 tok/s` (`+1.13%`), `stall-3 probe_p50` `-15.2%`.
- **Candidate `62`, promoted 2026-11-20 as `spark-ds41f-1120`**
  ([`candidate-62`](experiments/2026-10-26-spark-ds41f-candidate/candidate-62/README.md)):
  `candidate-61-r2`'s series (vLLM `0001..0822`) plus the Python fixes
  `0823..0827` and `0829` and the opt-in LSA upgrade `0828` (`VLLM_DS41_LSA`
  unset by default), `--max-num-seqs 32` on `TP4..TP8`, and four new shapes
  (`tp1`, `tp2-direct`, `tp2-switched`, `tp3-triangle-4k`). Its benchmarks are
  carried over from the two-boot `candidate-61` / `candidate-61-r2` runs; the
  `candidate-62` image itself ran the release tests (`12,023 passed, 356
  skipped, 0 failed`) and two-boot checks on `tp3-triangle` and `tp1`. What it
  leaves open is under [Next release](#next-release).
- **Candidate `61` (shipped in `spark-ds41f-1120` as `candidate-62`'s base)** ([`experiments/2026-11-16-c61-decode`](experiments/2026-11-16-c61-decode/README.md)) stacks on `candidate-60` with decode and infrastructure improvements verified through integrated margin confidence (`9649db2dc` / `45f67d1b4`: 2-boot **`c1 = 86.07 / 86.39` [`86.23 mean`]**, **`c16 = 273.96 / 274.30` [`274.13 mean`]**, `0` diffs) and `c61-r20` (`ad23a0e17`: `mhc.py` `_sinkhorn_stream` shares `aux_stream()`, **`c1 = 86.27 ± 5.69% tok/s`**, **`tpot_p50 = 11.24 ms`**, **`c16 = 273.39 ± 1.13% tok/s`**, `0` greedy diffs, `0` `/dev/dri/card0`) on `tp4-ring` (`dgx1..dgx4`). (`c61-r21` `_WO_A_TILES[16]` `block_K=64` test: **`c1 = 85.78 ± 5.80%`**, `c16 = 274.61 ± 0.91%`, `0` diffs, but engine grid regressed `+0.111..+0.238 ms/step` on `1x` and `+0.297 ms` on `16x16`, so reverted to `c61-r20` `95f23ef91` / `block_K=256`; its gain at `M=36/48/96` was not isolated, see Decode #16):
  - Integrated certified 2-pass `W_q` Markov vocab screening + per-row $\ell_2$ scale search + extreme-outlier sign-clamping (`Rank 2`), BF16 compressor replication up to `TP<=4` (`Rank 5`), `PR #1225` (`0817..0819`) + `pipe` + `<2>` dense `indexer_score_v41` + gated `_decode_index_split` (`Rank 6`; also supersedes `PR #1086`), grouped 3-layer DSpark `context_kv` + strided `front.py` (`Rank 10`), `PR #995` CopyDrafter guards/gentle ramp (`Rank 13`), fixed `PR #1072` `shm_broadcast.py` 2 ms post-wakeup spin anchor (`Rank 14`: eliminated 5s stalls, `ttft_p99 = 0.133s`), `PR #1081` shard-local sampled verification + `PR #988` sparse top-k Markov (`Rank 15`), `PR #1078` 48-row decode projection tile (`Rank 16A`), `PR #984` UMA allocator hygiene (`Rank 17`), `PR #987` 128-row `WO-B` / `_FULL_DEPTH_VARLEN_DECODE_MAX_REQS = 16` / `all_reduce_rank_order`, `Decode #3` (`wo_b`, `shared_up`, and `dspark.main_proj` `tile_contiguous=True` + `linear.py` `split_override` unpacking guard), `Decode #5` (`aux_stream` priority `-3` + `_PriorityCUDAGraphExec` `UseNodePriority=0x08` + `MoERunner._forward_impl` `M<=16` `_post_gate_hook` deferral fix), `Decode #6` (`moe_down` PDL prologue weight preload + `LayerChain` fix), `c61-r10` host-vs-GPU $p_k^2$ coverage-squaring elimination (`adaptive_verification.py`), `c61-r12` `SparseMlaV41Plan` `"split"` `skip_inactive=True` (`sparse_mla.py`), `c61-r13` device-side `is_padding` dead-row masking (`metadata.py` + `attention.py`), `c61-r15` hybrid `WQ_ONLY` + 1-warp/ping-pong `vocab.py` refine + `context_kv_front_3layer` + `gemm.py` `M=32/48` retune, `c61-r16`/`r17` `mhc` strict `pdl.chain()`-first + split `finalize_norm`/`finalize_sinkhorn` (`-0.20..-0.23 ms/step` on the `sp8` kernel screen; the split kernels ship in `0818` but nothing on the serving path calls them, see [Next release](#next-release) item 4), `c61-r20` `_sinkhorn_stream` sharing `aux_stream()`, `wo.py` `_WO_A_TILES` (`16: split=True, block_K=256`; `64/96/128` restored to `split=True` in `c61-r17`), `/dev/nvidiactl` RM scanout carve-out replacing `/dev/dri/card0`, `c61-r18` `grid_x=40` persistent loop in `refine_vocab_greedy_partials` (`vocab.py`, `4,040 -> 40` CTAs for Target `lm_head` + DSpark Markov), top-2 margin confidence calibration (`confidence.py`, `9649db2dc`), `tilelang_rejection.py` `initcheck` uninitialized-read guard (`705 passed, 0 errors`), `qwen3_dspark.py` `apply_bias_topk` deduplication, and `PR #1222`/`#1223` lab fixes (`124 passed` in `test_lab.py` + `test_source_locks.py`). (`PR #967` `sparse_mla.py` TIRx compiler fixes verified byte-identical in `candidate-60` across all 6 routes.)
  - Screened `c61-r13` (`179fd3b75`: **`c1 = 84.78 ± 6.11% tok/s`**, `28.00 ms/step`, **`c16 = 271.66 ± 1.05% tok/s`**, `0` diffs), `c61-r14` (`00306d1ef`: `c1 = 83.62 tok/s`, `c16 = 269.91 tok/s`, `0` diffs), `c61-r15` (`d0a111fa2`: `c1 = 84.11 tok/s`, `c16 = 271.23 tok/s`, `0` diffs), `c61-r16` (`99808dbd5`: `697 passed, 6 skipped`, `984` kernels / `0` PDL viol with `-v /dev/null:/dev/dri/card0`; ready `+132.1 s`, `MemAvailable 117.2–118.0 GiB`, `0` `/dev/dri/card0` mounts; **`c1 = 85.16 ± 5.85% tok/s`** [`27.89 ms/step`, `tpot_p50 = 11.68 ms`, `ttft_p50 = 0.1116 s`, `prose 81.15 / 83.79 dec`, `code 115.56 / 121.51 dec`, `#00 = 76.23 dec`], `c16 = 269.88 tok/s`, `0` diffs), `c61-r17` (`8bcf2b7ea`: `wo.py` `64/96/128` restored to `split=True`; ready `+114.6 s`, `jit_new_files = 0`, **`c1 = 85.45 ± 5.94% tok/s`** [`27.83 ms/step`, **`tpot_p50 = 11.63 ms`**, `ttft_p50 = 0.1117 s`, `prose 81.37 / 84.00 dec`, `code 115.50 / 121.44 dec`, `#00 = 76.22 dec`; **`2-boot mean = 85.30 tok/s`**, `+2.07` vs `c60-base`], **`c16 = 270.80 ± 1.38% tok/s`**, `0` greedy diffs), `c61-r18` (`8a51b5ac0`: `grid_x=40` persistent `refine_vocab_greedy_partials`; `97 passed, 6 skipped`, `0` PDL viol; ready `+129.0 s`, `+34` JIT files, **`c1 = 85.03 ± 5.87% tok/s`** [`27.91 ms/step`, `tpot_p50 = 11.68 ms`, `prose 81.03 / 83.66 dec`, `code 115.54 / 121.51 dec`], `c16 = 269.84 ± 1.33% tok/s`, `0` greedy diffs; `c61_r18_engine_grid.json` `1x` `dft_ms` **`-13.2 us` mean** vs `r17`, `gap_ms` `1.314–1.338 ms`), `c61-r19` (`777822915`: `MoERunner._forward_impl` `M<=16` `_post_gate_hook` deferral fix; `114 passed`, `0` PDL viol; ready `+116.7 s`, `jit_new_files = 0`, **`c1 = 85.28 ± 5.86% tok/s`** [`27.84 ms/step`, `tpot_p50 = 11.62 ms`, `ttft_p50 = 0.1116 s`, `prose 81.32 / 83.95 dec`, `code 115.22 / 121.13 dec`], **`c16 = 270.31 ± 1.14% tok/s`** [`tpot_p50 = 56.82 ms`], `0` greedy diffs; `c61_r19_engine_grid.json` `1x` `-301.8 us` mean vs `r13`, `1x2` `-16.9..-39.4 us` & `multi M<=16` `-10.1 us` vs `r18`), integrated `c61-margin` (`9649db2dc` / `45f67d1b4`: 2-boot **`c1 = 86.07 / 86.39` [`86.23 mean`]**, `tpot_p50 = 11.23 / 11.21 ms`, **`c16 = 273.96 / 274.30` [`274.13 mean`]**, `0` greedy diffs), and `c61-r20` (`ad23a0e17`: `mhc.py` `_sinkhorn_stream` shares `aux_stream()`; `197 passed`, `0` PDL viol; ready `+115.8 s`, `jit_new_files = 0`, `MemAvailable 117.3–118.0 GiB`, **`c1 = 86.27 ± 5.69% tok/s`** [`27.56 ms/step`, `tpot_p50 = 11.24 ms`, `ttft_p50 = 0.1109 s`, `prose 81.80 / 84.48 dec`, `code 116.10 / 122.03 dec`], **`c16 = 273.39 ± 1.13% tok/s`** [`tpot_p50 = 57.04 ms`], `0` greedy diffs; `c61_r20_engine_grid.json` vs `c61-margin`: `1x` `-13.3 us` step / `-19.5 us` fwd, `multi M<=16` `-33.5 us` step / `-72.8 us` fwd, `multi M>16` `+106.7 us` step). Rejected both the `+2,045/-325`-line `step_gate` flat-graph early-exit prototype (`/tmp/rework_dspark/final.patch`) as a bad complexity/trade-off (`+1,118` net production lines across 7 entangled files, 6 duplicated `@T.prim_func` paths, all 4 RoCE `all_gather`s still run, and `+11.33 us/step` `ALL-ALIVE` regression for only `-80.04 us/step` weighted `c1` / `-55.02 us/step` beyond `grid_x=40`) and the `dspark_context` fused MXFP8 cast prototype (`+0.19..+1.15 us` standalone and `+1.23..+3.95 us` end-to-end with `main_proj` across `cap=1..96`).
- **Candidate `60`** ([`experiments/2026-11-16-c60-prefill`](experiments/2026-11-16-c60-prefill/README.md)) builds on `candidate-59` (via `#1218` / `#1219`) in three gated steps across switched TP4 (`sp1..sp4`) and switched TP3 (`sp5..sp7`):
  - Integrated PR `#992` `bf16x2` producer casts, `indexer.py` `bfloat162` SMEM + warp-shuffle `block_scan` top-$k$, `attention.py` SP prefill compressor/front async overlap + deferred `finish_front` + `cacheable = index_rows <= chunk_rows`, PR `#983` Clock-Pro 2-bit Engram prefill shield + pinned two-plane scales (the reader's `pin_scales` argument, on by default; no environment switch of that name is in the series), PR `#1010` hybrid KV retention/eviction, PR `#978` multimodal prefix-cache fastpath, PR `#1006` uncached GPU post-load vectorizations / pressure-gated `release_device_memory_under_pressure` / lazy imports / first-prefill `_warm_model_runtime` warmup (dropping `postload_cache` after measuring slower reload than uncached GPU vectorization on GB10), piece-level parallel prompt tokenizer caching (`vllm/tokenizers/deepseek_v41.py`), PR `#1212` (rebased `#1047`) literal image-token escape, `defer_loading: true` + `tool_reference` inline tool-schema resolution (`serving.py`, `deepseek_v41.py`, `deepseek_v41_encoding.py`), and stream/lifecycle/parser hardening (`sp_prefill.py`, `engram_rows/rows.py`, `deepseek_v41.py`, `deepseek_v41_encoding.py`; evaluated and removed opt-in `SPARK_DS41F_PREFILL_MIX_CHUNK` from PR `#1150` during optionality cleanup).
  - Qualified with zero regressions on `tp4-switched` (`model`: `128/128` greedy / `100.00%` identical, `KL mean 0.000e+00`, `8/8` gate determinism/greedy `0` diffs; `grid` `-0.07% ±0.13%` `same`; `c16` `+0.17%` / `+0.15%` `same`; `c1-decisions` `82.91 tok/s`; `prefill 262K` `default-text` **`+2.10% ±0.64%` (`better`, `9,833.2 tok/s`)**, `+4.50%` over `c58-final`; `source` prefill `32K/131K/262K` `9,521 / 9,924 / 9,830 tok/s`, with Step 1 hitting `10,000.1 tok/s` mean at `131K`; `bench` `quality` `5/5` pass) and `tp3-switched` (`model`: `100.00%` identical, `8/8` gate `0` diffs, `grid` `-0.03% ±0.12%` `same`, `c8` `-0.25%` / `-0.23%` `same`, `sp5` startup `min_available_gib` `5.90–5.97 GiB`; `tp4-ring` remained held and untested).
- **Candidate `59`** ([`experiments/2026-11-15-c59-prefill`](experiments/2026-11-15-c59-prefill/README.md)) builds on `candidate-58` in three gated steps across switched TP4 (`sp1..sp4`) and switched TP3 (`sp5..sp7`):
  - Integrated bounded UTF-8 suffix detokenizer priming (`detokenizer.py`), zero-copy `int32` `PickleBuffer` `NewRequestData` admission IPC (`output.py`, `request.py`, `buffer_utils.py`), `bf16x2` / `cvt.rn.bf16x2.e2m1x2` dequant (`sparse_mla.py`), dense `indexer_score_v41` `rows >= 64` `R=4` BF16 dual-row SMEM reduction (`indexer.py`), fused `prefill_gate_up -> moe_swiglu_cast` with 64-col `(gate, up)` slab interleaving (`moe.py`), and PR `#1005` MoE warp-shuffle dispatch scan, sync/PDL tightening, and cascaded `moe_swiglu_cast` preloads (keeping `c58` BF16 SwiGLU intermediate rounding).
  - Qualified with zero regressions on `tp4-switched` (`model`: `128/128` greedy / `100.00%` identical, `KL mean 0.000e+00`, `8/8` gate determinism/greedy `0` diffs; `grid` `+0.24%` `same`; `c16` `+0.11%`/`+0.13%` `same`; `c1-decisions` `82.94 tok/s`; `prefill 262K` `default-text` `+2.35%` `better`; `source` prefill `32K/131K/262K` `9,548 / 9,886 / 9,757 tok/s`, `+1.28–1.53%` over `c58-final`) and `tp3-switched` (`model`: `100.00%` identical, `8/8` gate `0` diffs, `grid` `+0.02%` `same`, `c8` `+0.12%`/`+0.14%` `same`; `tp4-ring` remained held and untested).
- **Candidate `1107`** ([`experiments/2026-10-26-spark-ds41f-candidate/1107`](experiments/2026-10-26-spark-ds41f-candidate/1107/README.md), PR [#937](https://github.com/christopherowen/spark-ds41f/pull/937)) consolidates the C45 TileLang compiler (`tilelang/0001..0013` with explicit wrapper-cache and disk-streamed IKET controls), full TileLang runtime closure (`0690..0713`, `0717`, `0752..0754`), CopyDrafter continuation priority, warp early-exit pruning and expected-yield gating against DSpark (`0744..0745`), native-depth CUDA graph retention under copy verification (`0728`), online confidence calibration (`0746`), per-layer SP WO local MXFP8 exchange (`0737`), SP Engram prefill column exchange (`0738`), TP3 6-channel NCCL Ring and sparse-MLA `block_H=16` occupancy (`0740`), 2-launch mHC PDL (`0739` + `0755`), selective pack12 KV compressor (`0749`), fused indexer `wq_b` RoPE + scale (`0748`), packed 264-byte Engram reader with multi-CTA step, hot-row cache and negative `col_delta` guard (`0750`), deep-context scheduler interleave (`0751`), tensor-parallel vision tower (`0758`), and native `TP2..TP8` cross-topology parity (`0747`). Ready for consolidated image build and full suite qualification on `tp4-ring` and `tp3-triangle`.
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
Production is **spark-ds41f-1126** (`manifests/baselines/2026-11-26-spark-ds41f-1126.json`, image `spark-ds41f:1126`), promoted 2026-11-26 across all 16 shapes: `tp4-ring` ([suite reference](manifests/suites/2026-11-26-spark-ds41f-1126-tp4/)), `tp3-triangle` ([suite reference](manifests/suites/2026-11-26-spark-ds41f-1126-tp3/)), `tp3-triangle-4k`, `tp2-direct`, `tp1`, and `tp2-switched`..`tp8-switched` ([release record](experiments/2026-10-26-spark-ds41f-candidate/README.md)).
<!-- END GENERATED: next-release-status -->

Open after `1120`:
1. **XID 43 engine deaths ([#1036](https://github.com/christopherowen/spark-ds41f/issues/1036)),
   site-reported and not reproduced here.** Sites report the whole engine dying
   with XID 43 and `CUDA error: unspecified launch failure` on
   `spark-ds41f:1107`: on `tp4-ring` with 4 KiB pages and on the qualified
   64 KiB kernel, on a `1030`-derived stack, on frozen `candidate-54`, and
   (2026-11-18) on an eight-node TP8 deployment in the Engram row gather, with
   tool-heavy prompts and with a 1,024-token generation. One site has a
   repro that dies in 60–70 s: two concurrent chat requests, each about 2K
   text tokens plus 10 images. Its bisection: still dies with speculative
   decoding off, with a 4,096 batched-token budget, with `--enforce-eager` and
   with PDL disabled; no crash with `CUDA_LAUNCH_BLOCKING=1` and
   `--enforce-eager`. `candidate-62`'s record lists a live smoke pass for it on
   `tp3-triangle`; no experiment here reproduces the fault or shows `1120`
   clear of it. Run the two-request ten-image repro and the TP8 triggers on
   `1120` (`tp4-ring` or `tp4-switched`, and `tp8-switched`), record the
   outcome under `experiments/`, and answer the issue.
2. **`tp3-triangle`'s decode grid reads `worse` in the `1120` reference**
   ([release table](manifests/suites/2026-11-20-spark-ds41f-1120-tp3/release-table.txt)):
   geometric mean `+20.20%` (`21.27 -> 25.52 ms`). The six one-request shapes
   take `33.2–33.7 ms` against `18.5–19.7 ms` on `1107` (`+71%..+79%`), while
   the multi-request shapes improved `3.10–12.69%` and live single-stream `c1`
   improved `10.39%`. The release notes attribute it to the one-request shapes'
   dummy padding rows; `tp4-ring`'s grid improved on all 20 shapes. Confirm the
   cause on the triangle, then decide whether the release grid moves to rows
   that route and read context
   ([`experiments/2026-11-17-indexer-decode-share`](experiments/2026-11-17-indexer-decode-share/README.md):
   the fixed step's dummy rows are flagged as padding, skip the routed MoE and
   see no compressed context). A changed suite is a new suite; the recorded
   references stay.
   - **The padding-row account does not cover `tp3-switched`.** The same image
     booted twice there and the boots disagree: `1x2 rows @4K` reads `33.82 ms`
     on one boot and `16.49 ms` on the other
     ([release table](manifests/suites/2026-11-20-spark-ds41f-1120-tp3-switched/release-table.txt)),
     so that grid's verdict is `same` only because its repeat spread is
     `17.75%`. `candidate-61`'s
     [over/under](experiments/2026-11-16-c61-decode/over-under-1107-vs-candidate-61.md)
     puts the cost in the drafter at one request on padding rows (target
     `forward_ms` improved `7.3–7.8%`); a boot that reads `16.49 ms` on the
     same rows needs a second cause. Find what differs between the two boots
     before treating the one-request shapes as a grid artefact.
   - **The grid can already run its rows as real ones and the release does not
     use it.** `0820` and Sonar carry `real_rows`; both `1120` grids ran with
     `"real_rows": false` and no suite in `config/suites/` sets it. The
     real-rows profile has one boot
     (`experiments/2026-11-17-indexer-decode-share/receipts/tp4-switched/profile-real-rows`);
     its repeat (`profile-real-rows-again-tp4-switched.json`) has no receipts
     and the README no write-up of either.
   - **What the grid leaves out was measured on `candidate-57`**
     ([`experiments/2026-11-10-grid-diverse-tokens`](experiments/2026-11-10-grid-diverse-tokens/README.md)):
     its rows route 0 of 384 experts in every target layer and the drafter's
     three at every shape, where real text reaches `9.7` a layer at `1x2` and
     `114.8` at `16x96`. With real text's routes the step is `2.50x` the grid's
     at the geometric mean of 14 TP4 shapes (`1.23x` with real text through
     dummy gates). Its version-2 grid was a `servebench` change and was never
     merged.
3. **`tp3-triangle` short-request TTFT beside six streams.** In the same
   reference, a short request's TTFT p50 during a 131K prefill beside six
   streams reads `1.76 s` against `1.18 s` on `1107` (`+48.88% ±16.13%`,
   `worse`) and `1.44 s` on the repeat boot (`+21.58% ±52.47%`, `same`), while
   beside three streams it improved (`1.07 -> 0.80 s`) and the prefill beside
   six streams is `13.19%` faster. **Diagnosed (2026-11-21, `PR #1273` / `#1281`):**
   `stall-7`'s exact `long_share = 0.3432` (`22/64` steps) proves `K = 2`
   interleave already fires on every chunk (`T_decode ~ 25 ms < 50 ms`). The
   bimodal `1.2–2.1 s` `stall-6` TTFT is a `max_num_seqs = 8` slot-8 occupancy
   cascade: 6 decode streams + 1 long prefill leave 1 slot for `3.0 s`-spaced
   probes, and `1120`'s higher `adaptive_verification_row_cost = 0.049` trims
   drafts during prefill-interleaved decode steps, pushing 16-token probe
   lifetime past `3.0 s` so Probe `i` holds slot 8 when Probe `i+1` arrives.
   (`candidate-64` patch `0837` fixes the separate async `T_decode >= 50 ms`
   undercounting bug for `tp4-ring` `stall-11` / 32-stream shapes; TP3 `stall-6`
   interleaved-draft budgeting remains open.)
4. **The mHC finalize split is in the image and never runs.** **Resolved in
   `candidate-64` (`0833`, `PR #1264` / `#1280`):** wired `async_sinkhorn` across
   `nvidia/model.py` and `nvidia/dspark.py`, verified `0` `mhc_finalize_kernel`
   calls via CUPTI, and 2-boot verified `-1.00%` `tp3-triangle` engine grid /
   `c1 tpot_p50 -0.34 ms (-2.35%)` with `128/128` bit-exactness.
5. **GLP-27 did not become the default vector at promotion.** `candidate-55`'s
   [record](experiments/2026-10-26-spark-ds41f-candidate/candidate-55/README.md)
   carries the owner's decision of 2026-11-13 ("Yes, at promotion") for the
   release whose image reads Spec v3: its promotion commit makes
   `GLP-27-L11-37-a3.5` the default, qualified at three testbeds
   ([receipts](experiments/2026-10-26-spark-ds41f-candidate/candidate-55/receipts/glp/README.md)).
   `1120`'s image reads Spec v3 (`0805`), yet
   `releases/spark-ds41f-1120/cluster.json` and `config/cluster.json` serve
   `GLP-39-L1-39-a0.75`, and #985, which made the switch, was closed on
   2026-11-15 as integrated. The owner decides whether the decision stands; the
   switch changes the promoted configuration and needs its own baseline.
6. **TP1..TP3 ship a prefetch row band that its sweep did not confirm.**
   **Measured on `tp3-triangle` (2026-11-21, `PR #1268` / `#1280`):** both
   `WO_BANDS=""` (`mad5 -0.045%`, `fl25 -0.017%`, `c1 69.58`, `c8 169.39`) and
   `NEXT_MB=15` (`mad5 -0.012%`, `fl25 -0.035%`, `c1 69.66`, `c8 169.45`) were
   statistically flat within `±0.09%` noise on Boot 1; existing TP3 defaults
   kept.
   `tp3-triangle`, `tp2-direct` and `tp1` set
   `VLLM_DS41_L2_PREFETCH_WO_BANDS=17-32=12`, and `tp3-switched`,
   `tp3-triangle-4k` and `tp2-switched` extend them. The sweep's two-boot
   confirmation at `tp3-switched` missed its mark (grid `0.06%` faster against
   a `0.30%` margin) and concluded that the shape keeps no band
   ([sweep](experiments/2026-11-13-prefetch-window-sweep/README.md)). The value
   entered the recipe with #1218, and `docs/doctor.md` names it the qualified
   setting for TP3. Confirm it over two boots on `tp3-triangle` against the
   same image with no band, or take it out. Two smaller things from the same
   sweep: `tp5..tp8-switched` inherit TP4's `17-32=10` without a measurement
   of their own, and `NEXT 15` (`0.26%` and `0.49%` faster at the two TP4
   testbeds, under the rule's margin) was left as a lead and not followed.
7. **Torch for the toolkit is not in `1120`.** The owner asked on 2026-11-13
   for torch to match the CUDA build. `candidate-55`'s seventh build carried
   torch `2.16.0.dev20261113+cu135` with NCCL `2.32.3` (which that torch
   needs) and vLLM `0808`
   ([record](experiments/2026-10-26-spark-ds41f-candidate/candidate-55/README.md),
   "Seventh build"). Its grid read `2.18%` faster at `tp4-ring` and `4.11%` at
   `tp4-switched`, and `c16` `1.58%` faster at `tp4-switched`, with outputs
   identical to `candidate-54`'s. It was not promoted:
   - 172 release tests failed: torch's Triton `3.9` against the TileLang
     sampling, top-k/top-p and rejection kernels at signed zeros, infinities,
     NaN and top-p ties;
   - the base image's DeepGEMM extension did not load, and no NIXL wheel was
     built for it;
   - `tp4-ring` booted only with `TORCH_DIST_USE_NCCL2=0`;
   - `tp3-switched` fell below its startup and bench memory floors;
   - 262K prefill at `tp4-ring` was `0.92%` slower.

   The eighth build went back to torch `2.13.0+cu130` and NCCL `2.30.7`, which
   `1120` ships. Retry it on a newer torch:
   1. settle those edge values in the tests or the kernels;
   2. rebuild DeepGEMM and NIXL against it;
   3. measure the grid gain again.
8. **Retire the measurement matrix** (`bin/spark bench` without `--suite`, now
   Sonar's `matrix`) once no active workflow references it; the suites in
   `config/suites/`, run by Sonar (`docs/release-benchmark.md`), are the release
   benchmark authority.

### Candidate 61 review follow-up

1. **Reset the prefix cache after a failed Sonar grid run (P1).** A worker can
   write profiling KV before the RPC raises or returns an empty result. Attempt
   cleanup on every exit after dispatch. Fail the run if the reset fails and
   preserve the original error. Test a worker failure after a write, an empty
   reply, and a failed reset. Verify a subsequent request with the same prefix
   produces the same token IDs as a cold request after successful cleanup.
   [Sonar wrapper](spark_ds41f/sonar.py)
2. **Isolate writable KV in the real-row profiler (P2).** Give each concurrent
   request its own writable ring blocks. Initialize the history for each
   requested depth and preserve it when a repeat sets `fill_real_context=False`.
   Compare two requests with distinct histories at different depths against
   their individual runs, then reverse the order and repeat.
   [Grid implementation](experiments/2026-11-16-c61-decode/vllm/0820-perf-spec-decode-optimize-DSpark-margin-confidence-h.patch)
3. **Check the batched DSpark runner-up against the full vocabulary (P2).**
   The dense path discards the winner's entire 2,048-token partition. Test
   logits with the top two in the same partition, in different partitions, and
   tied. Measure how often the approximation changes verification decisions on
   held-out traffic. Screen an exact top-two reduction at one and 16 streams
   against the shipped calibration, which already gained 1.2% and 1.3%.
   [Reduction](experiments/2026-11-16-c61-decode/vllm/0819-perf-dsv41-tune-vocab-head-grid-width-and-packed-BF1.patch) · [Calibration A/B](experiments/2026-11-17-draft-calibration-sizing/README.md)

After the two profiler fixes, prioritize the [eight-session replay](#mixed-prefill-and-decode).

## Current reference

<!-- BEGIN GENERATED: current-reference -->
Active baseline manifest: [`manifests/baselines/2026-11-26-spark-ds41f-1126.json`](manifests/baselines/2026-11-26-spark-ds41f-1126.json) (`spark-ds41f-1126`). Releases are judged by the release, single-stream and sampled suites, measured with Sonar (`docs/release-benchmark.md`, `scripts/sonar_table.py`). Current benchmark references are [`manifests/suites/2026-11-26-spark-ds41f-1126-tp4`](manifests/suites/2026-11-26-spark-ds41f-1126-tp4/) for `tp4-ring` and [`manifests/suites/2026-11-26-spark-ds41f-1126-tp3`](manifests/suites/2026-11-26-spark-ds41f-1126-tp3/) for `tp3-triangle`:

| Measure | `tp4-ring` (`spark-ds41f-1126`, 1M, 32 seqs, 16K batch) | `tp3-triangle` (`spark-ds41f-1126`, 1M, 16 seqs, 6K batch) |
| --- | ---: | ---: |
| Decode grid geometric mean (20 TP4 / 15 TP3 shapes) | 19.21 ms | 19.10 ms |
| Decode `c16`, 16 streams (output tok/s) | 283.3 | — |
| Decode `c8`, 8 streams (output tok/s) | 215.3 | 172.3 |
| Decode `c4`, 4 streams (output tok/s) | — | 133.9 |
| Single-stream `c1` (output tok/s) | 89.5 | 71.3 |
| Sampled `c16` / `c8` / `c1` (`temp=1.0`, output tok/s) | 277.1 / 211.2 / 87.7 | 219.9 / 167.1 / 69.1 |
| Poisson arrivals (0.7 TP4 / 0.42 TP3 req/s, output tok/s) | 180.1 | 119.6 |
| Source-text prefill 32K (tok/s) | 9,729 | 7,327 |
| Prefill 131K (tok/s) | 10,272 | 7,582 |
| Prefill 262K (tok/s) | 10,253 | 7,544 |
| Prefill 524K (TP4) / 500K (TP3) (tok/s) | 10,067 | 7,398 |
| Prefill 1M (tok/s) | 9,686 | 7,120 |
| Short-request TTFT p50 during 131K prefill (3 / 11 TP4 or 3 / 6 TP3 streams) | 1.65 s / 1.42 s | 0.84 s / 0.91 s |
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
  acceptance, use paired multi-prompt suites (Sonar's) across two boots.
- **When a decision hinges on 2.5% or less**, watch boot-to-boot variance across
  repeat boots before attributing a small delta to a kernel or policy change.
- **Record each node's uptime with a run.** Runs do not store how long each node
  has been up; keep it with the run's evidence so time since reboot can be
  checked as a source of boot-to-boot variance.
- **Python-only vLLM changes need no image build to screen.** Bind-mount the
  changed files from the checked patch branch over the last candidate image via
  `container.mounts` (the image imports vLLM from `/opt/spark-ds41f/candidate/vllm`).
  Build only for promotion or when compiled extensions/TileLang sources change.
- **Check every candidate's boot log for silent fallbacks** (`disabled`,
  `fallback`, `compile failed`) and for any JIT compilation after readiness.
- **Shared packages move forward only.** When components disagree on a shared
  runtime package, patch the lagging pin forward.
- **The bench's multi-stream decode points are bimodal.** `prose-cN` and
  `code-cN` send N copies of one prompt.
  - A sample is fast only while the copies stay in step, with every request's
    `last_s` equal. Once they drift it is 25–45% slower, so a point's mean is
    mostly how many of its samples stayed in step. This held for every `c4` and
    `c8` sample of `1107` (both boots), `c58-final` and `candidate-61` at
    `tp3-switched`, checked 2026-11-18.
  - Classify samples by `last_s` spread and compare `step_ms` within a mode.
  - When `output_sha256` differs between builds, tok/s and tokens per step on
    that point are not comparable.
  - Judge `c8` and `c1` on the distinct-prompt suites.

  `docs/methodology.md` does not say this yet.

## Attempt ledger and idle-time runs

Two process changes proposed 2026-11-18. Neither is started; the second depends
on the first.

- **Keep a ledger of attempts, grouped by problem and read at the start of every
  round.** The experiment folders are ordered by date (339 on 2026-11-20) and
  this file is 72 KB of results, so a round begins by surveying history: sizing
  draft accuracy needed eighteen drafter records read first
  ([`experiments/2026-11-18-draft-accuracy-sizing`](experiments/2026-11-18-draft-accuracy-sizing/README.md)).
  One file per area. Under each recurring problem, every attempt records:
  1. the change and why it was expected to help;
  2. the predicted result, written before the run (22 of the 339 experiment
     READMEs mention a prediction at all);
  3. the measured result, losses included, with a link to the experiment;
  4. the conditions it was measured under and those it depends on (image and
     calibration, shape and kernel page size, workload), so a new base, shape,
     workload or checkpoint names the entries to measure again;
  5. checks prepared and not run.
     [`experiments/2026-10-26-dspark-acceptance`](experiments/2026-10-26-dspark-acceptance/README.md)
     has held two since 2026-11-04 (a chat-format probe of the drafter and a CPU
     reference parity run), found again only by that survey.

  Start with speculative decoding, from the records that survey already covers,
  and judge the format there before backfilling another area.
- **Run the ledger's open entries when the machines are idle.** A runner takes
  the top open entry, writes its prediction, builds, screens through the lab
  queue behind the hold file and the idle guard, records the outcome, and
  proposes a pass for the open candidate as a pull request. Promotion stays the
  owner's. Before a runner may count anything as a pass:
  - **The screen must be shown to track the goal.** One `c1` boot reads ±5.7%
    (`c61-r20`), and most open levers are worth 0.5 to 1.5%, so many unattended
    attempts at that noise collect false passes. An AgentX A/B of a known gain
    at the production session count is the check, and it is not run.
  - **Fixed gates:** zero greedy diffs, two boots, and a result that matches the
    prediction written beforehand. A gain that was not predicted is noise until
    a repeat confirms it.
  - **The owner names the machines.** A lab window stops the promoted service
    for its duration; start on the Spaceports, not the ring.

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
- **#1134's third part was not carried.** Its first two ship:
  - `cluster.py`'s pre-boot defragmentation with `0800`'s buddy sponge;
  - `0801`'s L3-aware core pinning.

  The third is not on `main`:
  - a `memguard` that caches the container's PID and polls `/proc` with no
    forks (`scripts/memguard.sh` still runs `docker inspect` on every poll);
  - `memguard` and the lab's telemetry confined to the efficiency cores 0–4
    (`AllowedCPUs`).

  #1134 measured the three parts together (the host gap's boot-to-boot SD
  `16.4x` lower on `sp8`). Measure the third alone on the host gap before
  carrying it.

## Mixed prefill and decode

The agent workload interleaves long prefill bursts with decode phases. Cached
prompts avoid most of it (98–99% prefix-cache hit rate in live agent traffic),
so the cost falls on cold prompts: the first request of a scan, cache
evictions, or large new tool outputs.

- **Replay eight real AgentX sessions while a cold 131K prompt arrives.** Fix
  the arrival trace, tool turns and cache state across candidate arms. Timestamp
  arrival, tokenization, queue admission, prefill, first token and completion.
  Report short-turn p50/p99 TTFT, active-stream p99 inter-token gaps, and the
  long request's completion time. Use this replay to compare the chunk policy
  below with scheduling by deadline at chunk boundaries, increasing priority
  with wait time so long requests finish. On TP4 also run the 32-seat suite
  on the next candidate and report per-session latency and SLO goodput at
  saturation. The published TP4 throughput comparison reaches 16 streams.
  [Mixed-load reference](manifests/suites/2026-11-20-spark-ds41f-1120-tp4/release-table.txt) · [32-seat suite](config/suites/c32.json)
- **In `1107` (`0751`)**: deep-context scheduler interleave (`VLLM_DS41_DEEP_PREFILL_INTERLEAVE=1`
  above 16K context) and `block_M=32` long-context sparse-MLA staging reduce
  32K–64K mixed-decode ITL stalls by 23–27% while keeping pure 131K prefill
  throughput neutral (`experiments/2026-11-06-deep-context-scheduler`).
- **Idle streams during long cold prefills.** A 1M cold prefill takes ~3 minutes
  and sends nothing before its first token. Check whether clients or gateways
  cut idle streams during 1M prefills; if so, send SSE keep-alives.
- **Concurrency-aware dynamic prefill chunk stepping under active decode streams.**
  Static `SPARK_DS41F_PREFILL_MIX_CHUNK` (`#1150`) was retired, and the full
  `16,384` (`tp4`) / `32,768` (`tp5..tp8`) batch budget maximizes solo cold
  prefill throughput (`10,108–12,429 tok/s` at `131K`), but holds sibling
  decoding streams for `~1.8–2.1 s` per mixed step (`stall-3 / stall-11`).
  Screen dynamically stepping the scheduler's per-step prefill chunk cap down to
  `2,048–4,096` tokens whenever `num_running_decode_reqs > 0` (keeping the full
  `16K/32K` chunk when `active_decodes == 0`), and letting queued short delta
  prefills (`<= 512–1,024` uncached tokens) preempt at the next sub-chunk
  boundary rather than waiting behind `--prefill-yield-limit-s 10`.
- **Token-budget-gated parallel short-delta prefills (`<= 512–1,024` combined
  uncached tokens).** Global `--max-parallel-prefills 8` regressed four-`64K`
  cold prefills because batching multiple long prompts defeats CED's
  single-request `128`-row window. Screen keeping `--max-parallel-prefills 1`
  for any request with `> 1,024` uncached tokens while allowing the scheduler to
  coalesce multiple short cached-delta turns (`<= 256–512` uncached tokens each,
  combined uncached budget `<= 1,024` tokens) into one step when concurrent
  subagents return tool outputs or fork simultaneously.
- **Atomic admission for the buffered chat batch endpoint
  ([#1141](https://github.com/christopherowen/spark-ds41f/pull/1141), open,
  contributor).** An opt-in header sends a batch's prepared prompts to the
  engine in one message, so an idle engine does not start decoding before the
  batch is complete. Ranked last in `candidate-61`'s backlog (multi-stream
  waves only, no single-stream effect) and not evaluated in a round; `1120`
  does not carry it.

## Decode

Routed MoE is the largest share of decode compute (~50% at one stream, ~62% at
eight streams), followed by dense FP8 projections, MLA attention/indexer,
collectives, and DSpark drafting.

**Target 100 tok/s on single-stream prose** from the two-boot 84.91 tok/s
reference. The measured bandwidth accounts for 24.42 ms of a 27.53 ms step.
Trace actual expert IDs and weight bytes read per committed token across
successive steps. Use the observed reuse to screen keeping frequently reused
routed weights resident with identical target outputs. Report bytes per
committed token, step time and serving tok/s alongside the kernel timings.
[Bandwidth measurements](experiments/2026-11-17-c1-floor-now/README.md) · [Two-boot prose results](experiments/2026-11-16-c61-decode/over-under-1107-vs-candidate-61.md)

1. **Routed MoE continuous `FC1 -> SwiGLU -> FC2 -> Combine` persistent kernel**
   (about 0.6% decode ceiling, down from the 2.4–3.5% first written here). That
   estimate priced TileLang's `swiglu_forward_and_per_token_cast` at 23–30 µs
   per layer; on `candidate-61`'s base `moe_swiglu_cast` takes `4.1 us` of a
   `564.3 us` routed-MoE graph at `M=6` on `sp8` (about 0.16 ms over 40 layers
   of a 27.8 ms step), three pipeline stages beat two stages with
   `fuse_swiglu`, and `c61-r16` reverted `c61-r14`'s `fuse_swiglu` change
   ([`experiments/2026-11-16-c61-decode`](experiments/2026-11-16-c61-decode/README.md)).
   The cast runs serially between the two grouped GEMMs. Keeping routed tiles resident across `FC1 -> SwiGLU -> FC2`
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
   RoCE send buffers** — **superseded / measured negative on GB10** (live
   4-node CUPTI confirms `sparknet 0.5.0` already has `0` pre-AR
   `cudaMemcpyAsync` calls [`0.0006 ms/step` total `gpu_memcpy` at `c1`],
   instead staging L2-hot `inp` in-kernel via 16-byte coalesced
   `roce_copy_pack`; `sp8` Cold-L2 prototype writing `wo_b` / `moe_combine`
   directly into `pin_memory=True` `send_slot` with `fence.sc.sys` /
   `st.release.sys.v4` regressed `wo_b` by `+2.42..+51.06 us/L` at `M=6..48`
   and `moe_chain` at `M=3..48`).
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
   `TILE_CONTIGUOUS_FP8` (`gemm.py`), with `_block32_scale_words(..., split_override=None)`
   in `linear.py` keeping `TileLangGroupedBlock32Rows` `(1536, 5120)` unpacked under
   `SPLIT_FP8[(512, 5120)]` (`sp8` Cold-L2 `SHARED_UP(1152x5120)`, `M=6, Mp=16`,
   `COPIES=8` + `40 MiB` flush: `36.15 -> 34.94 us/L`, `-1.20 us/L`, `40L = -0.048 ms/step`,
   bit-exact; `17 passed`, `0` PDL violations across `test_deepseek_v41_tilelang_dspark_context.py`
   and `test_deepseek_v41_tilelang_linear.py`).
4. **Vocabulary head (`128`-row token tile and per-tile epilogue summary).**
   - **A 128-row token tile: shipped.** `TILE_CONFIGS[128]` and the certified
     screen's `SCREEN_TILE_CONFIGS[128]` serve wide head calls on a 128-row
     tile, on a matched two-stage pipeline since `0784` (#1001, in `1120`;
     86,016 B of shared memory, bit-identical at every `M=1..128`).
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
   `c61-r16` adds the kernels for the intermediate split, which no serving
   path calls yet ([Next release](#next-release) item 4): `mhc_finalize` (`80×/step`) splits
   into a 5-CTA critical-path `mhc_finalize_norm` (`1.9 us`) plus off-critical-path
   `mhc_finalize_sinkhorn` (`threads=32`) on `aux_stream()` (reused from
   `vllm.utils.torch_utils.aux_stream()` in `c61-r20`; `_finalize_sinkhorn`
   stays launched before `_finalize_norm` so the `_finalize_norm -> GEMM` PDL
   edge stays unbroken) via `async_sinkhorn()` / `wait_sinkhorn(...)` (`model.py`),
   while keeping `pdl.chain()` strictly first across all `mhc` kernels so
   `mhc_project_streams` does not contend with `oneshot_allreduce`'s early
   `roce_doorbell` `T.pdl_trigger()` (verified `-2.23..-2.69 us/call` =
   `-0.20..-0.23 ms/step` on `sp8` Cold-L2, `72/72` `torch.equal`, `0` PDL
   violations; fusing MXFP8 cast into `dspark_context.context_inputs` was also
   rejected after `sp8` Cold-L2 measured `+1.23..+3.95 us` slower with `main_proj`).
   Still open: keep `project_streams`
   writing the unnormalized collapsed stream `C` and the partial sums-of-squares
   `Q`, and have the consumer decode projections (`fused_wqa_wkv` and MoE
   gate/shared-expert input) reduce the partial sums `Q`, multiply by `rsqrt * gamma`,
   and quantize to MXFP8 inside their prologue—removing the global BF16
   normalized activation round-trip and standalone MXFP8 cast launches. At model
   entry, also screen fusing token embedding lookup, 4-stream broadcast, Engram
   `token_map` lookup, and Layer-0 `project_streams` into one launch.
8. **Piggybacking KV-RoPE onto `wq_b` via extra tail CTAs.** `0748` and `0749`
   fuse `wq_b + indexer.wq_b` with query RoPE and MXFP4 quantization; screen
   appending tail CTAs to `wq_b`'s grid that branch to apply RoPE and cache
   writes to `wkv`'s newly projected KV rows concurrently with `wq_b`,
   eliminating 61 separate KV-RoPE launches per step.
9. **Intra-request multi-row `KPool` reuse in decode `indexer_score_v41` (and
   deep-context TP row split).** Implemented and kernel-verified in
   [`experiments/2026-11-21-decode-indexer-multirow-kpool`](experiments/2026-11-21-decode-indexer-multirow-kpool/README.md)
   (pending Round 4 two-boot live screen on `tp3-triangle`): replaces the
   merged `<2>/<4>` kernel in `IndexerV41Plan` with dedicated decode
   (`row_group=3`, `rows <= 96`, `158` regs, `0` spills, `3 CTAs/SM`,
   wave-aligned `max_ctas = 144` through `256K` and `288` above `256K`) and
   prefill (`row_group=4`, `rows > 96`) kernels, removes dead `w_reg`, enables
   odd-`R` software-pipelined `same_k`, and adds `loaded_page` reuse in
   `!same_k` (`107 passed, 1 skipped`, `15/15` PDL incl. `layer_chain`;
   `dgx4` disjoint Cold-L2 across `6L` dense layers: `8x6 @ 256K` `-17.1%`
   [`-0.98 ms/step`], `16x6 @ 256K` `-9.7%` [`-0.99 ms/step`], `1x6 @ 1M`
   `-19.1%` [`-0.68 ms/step`], `4x6 @ 1M` `-24.4%` [`-3.16 ms/step`],
   `8x6 @ 1M` `-27.4%` [`-7.04 ms/step`], `16x6 @ 1M` `-24.2%`
   [`-12.11 ms/step`], `adapt8 @ 1M` `-11.6%` [`-2.43 ms/step`]; `SMEM-Q`
   `R=6/12/16` and register `R=6` rejected as slower). Still open for a
   separate follow-up: multi-request TP row split above `64K` with one-shot
   `sparknet` RoCE gather of `topk_indices_buffer[:rows]` (`2 KiB/row`).
10. **Same-request multi-row SWA KV reuse in decode `sparse_mla_fwd_v41`.**
    **Audited & bounded; single-segment `extend` route landed** in
    [`experiments/2026-11-21-swa-single-segment-extend`](experiments/2026-11-21-swa-single-segment-extend/README.md).
    Cold-L2 profiling on `dgx4` (`198 MiB` rotating SWA pools) showed unique
    SWA KV per layer is only `67 KiB` (`1x3`) to `1.07 MiB` (`16x6`) — `0.3%–4.5%`
    of GB10's `24 MiB` L2 — and single-kernel `extend` runs at a flat
    `14.59–15.62 us/L` across `rows = 1..48` (`0.00 us` DRAM bandwidth scaling),
    bounding multi-row SWA KV DRAM reuse. The audit caught that `SparseMlaV41Plan.routes()`
    still routed 1-segment SWA-only layers (`segments == 1`) through the
    two-kernel `split`/`decode` + `merge` path (writing FP32 `Partial` to DRAM
    for 1 segment); routing `segments <= 1` directly to `"extend"` is `100%`
    `uint16` bit-exact (`38/38` passed incl. `15/15` PDL), wins `16/16` shapes
    (`-1.06..-4.06 us/L` across `rows = 1..96` on `H=16/24`), removes `2`
    compiled kernels per SWA-only shape, and drops SWA `Partial` scratch to `0`.
11. **One-shot all-reduce latency per call (proxy core affinity shipped).**
    `1120` pins the `sparknet` proxy, the worker main thread, the engine core
    and rank 0's API server to disjoint fast cores by L3 cluster (vLLM `0801`,
    `sparknet` `0001`; #1134, #1008). Still open: the per-call time itself. The
    last isolated timing is 2026-10-03, before `sparknet` 0.5.0 and its three
    patches (`experiments/2026-10-03-collective-policy`: 16–19 µs at 10 KiB and
    about 29 µs at 60 KiB), and every decode layer pays it. Time one call on
    `1120` at the one-row and six-row sizes on `tp4-switched` and `tp4-ring`,
    split into wire, proxy and GPU-side signalling, and measure rank arrival
    skew, before sizing any change to how shares and flags reach the peers.
12. **A RoCE one-shot broadcast for adaptive verification.** Adaptive
    verification broadcasts rank 0's confidence snapshot every decode step
    through NCCL (`adaptive_verification.py`). Bound its share of the step from
    decode traces and screen `sparknet`'s one-shot broadcast path if material.
    Also test whether the broadcast is needed at all: count on device how often
    a rank's own snapshot differs from rank 0's, and drop the broadcast if the
    count stays at zero across the release suites.
13. **Mid-network (Layer 16/20) dead-row routed-MoE pruning at single stream
    (`c1`).** At `c1`, no other requests in the batch amortize routed-expert
    LPDDR5X reads (4.5 MiB per expert per rank across 40 routed layers), so
    every rejected draft row surviving `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU = 0.2`
    before Layer 0 incurs unamortized routed-expert DRAM reads. Because target
    layers are strictly causal and row-independent, pruning diverging tail draft
    rows from routed MoE at mid-network (Layer 16/20) while masking pruned rows
    as rejected in `rejection_sampler` cuts half of those wasted routed-expert
    DRAM reads while keeping all accepted target tokens 100% bit-exact.
14. **Engram row-cache capacity on live traffic.** Each disk Engram reader
    (layers 1 and 14) keeps a 65,536-row 4-way row cache (17.3 MiB);
    `SPARK_DS41F_ENGRAM_ROW_CACHE_ROWS` resizes it without a rebuild. The size
    was set on synthetic hot, Zipfian and cold lookups
    ([`experiments/2026-11-06-engram-step-and-cache`](experiments/2026-11-06-engram-step-and-cache/README.md))
    and has had no capacity sweep. First record hit rate and evictions per layer
    on recorded agent traffic (`rows.stats()` reports `cache_hits`; add an
    eviction count if it has none), because a bench point of identical streams reuses rows and
    hides misses. If evictions are material, raise the size until they reach
    about zero, and weigh the gain against 16-stream decode and dgx1 headroom at
    1M context. Confirm at the first boot that the switch reaches the reader.
15. **Router share of a single-stream step.** No record sizes the per-layer
    router (the 384-expert gate projection, scoring and top-6 selection) on its
    own. Profile it on `1120` at one stream: its time per layer, its achieved
    read rate, and whether the gate is replicated or sharded across ranks.
    Only if it is material, screen one launch for projection, scoring and
    selection with the same order and tie rule.
16. **A narrower `WO-A` reduction for wide steps only.** `c61-r21` set
    `_WO_A_TILES[16]` to `block_K=64, num_stages=2` for every step and was
    reverted whole: the one-request shapes slowed by `+0.111..+0.238 ms/step`
    and `16x16` by `+0.297 ms`, but the wide shapes gained (`8x36 -0.244`,
    `8x48 -0.434`, `16x48 -0.489`, `16x96 -0.556 ms`) and the one `c16` boot
    read `274.61` against `c61-r20`'s `273.39 tok/s`
    ([`experiments/2026-11-16-c61-decode`](experiments/2026-11-16-c61-decode/README.md)).
    The record does not isolate why the wide shapes gained. Find it, then
    screen it for steps above 32 rows only, with the `block_K=256` tile kept
    below.
17. **A second NCCL communicator made at load slows the decode grid about 1%,
    for a reason not known.** The turn-learning experiment's grids at
    `tp4-switched`
    ([record](experiments/2026-11-17-dspark-turn-learning/README.md), "What slows
    the grid") measured it against the pooled learner-off boots (sd `0.15%`):
    - a communicator made at load and left idle: `+1.13%` (4 boots);
    - one made at load and destroyed after use: `+1.23%` (1 boot);
    - one made after the graphs are captured: nothing measurable.

    Memory held from load, bare streams and NCCL operations on the step's own
    communicator each cost nothing. No kernel takes longer: the step's critical
    path lengthens after its all-reduces (`mhc_project_streams_kernel` ends
    later). Two things follow:
    - a feature that needs a communicator of its own makes it after startup;
    - count the NCCL communicators serving already makes at load (vLLM's
      groups, the prefill path's). If one beyond the step's own is made before
      capture, moving it after capture is a lever to screen. This is unmeasured.
18. **Requests that set a sampling control (`top_k`, `top_p`, `min_p`, a
    penalty, `logit_bias`).** Unmeasured. Every sampled point of
    `config/suites/sampled.json` and `c32.json` sends `temperature` with
    `top_p: 1.0` and no other control, and no record says how often clients
    send one. The two paths differ:
    - a step whose rows only scale by temperature samples and verifies from
      the raw BF16 logits in one kernel (`0623`);
    - a step with any other control takes the `[rows, vocab]` FP32 copy and
      launches each control in turn. Top-k/top-p alone is four kernels, one of
      them a five-step pivot search over the row (`0436`).

    Size it before anything else, then screen the two levers only if the gap
    is material:
    - **Sizing (not run).** Count the controls that recorded agent traffic
      sends. Read sampled `c1`, `c8` and `c16` with `top_p=0.95`, then with
      `top_k` and a penalty added, against the `1120` sampled reference. This
      needs a new suite: the recorded ones are not edited.
    - **Choose the fused path per row.** **Implemented & verified** in
      [`experiments/2026-11-21-per-row-mixed-sampling-split`](experiments/2026-11-21-per-row-mixed-sampling-split/README.md)
      (`model_runner.py` `_samples_split_from_shards` / `_sample_split_from_shards`,
      zero D2H sync, guarded by `_cu_num_logits_np_exact`): partitions mixed
      batches per request so greedy/temperature-only requests stay on
      shard-local verification/sampling (zero full-vocab RoCE all-gather, zero
      FP32 copy) while only non-temperature control rows all-gather and run
      `apply_sampling_params` / `apply_top_k_top_p` (also giving `top_p` full SM
      split parallelism sized to active rows). Verified `5/5` passed on `dgx4`
      (`100% torch.equal` fixed-seed parity across interleaved greedy/temp/top_p/
      top_k/rep_penalty/dead_row_cut + `TP=3` selective gather check); GPU
      compute `-418.0 us` (`-47.3%`) on `8x6 (1/8 ctrl)` and `-539.6 us`
      (`-43.7%`) on `16x6 (1/16 ctrl)` plus `4×–16×` less RoCE all-gather.
    - **One tile pass for the controls.** Apply the per-token controls
      (penalties, bias, temperature) in the tile kernel's prologue and reduce
      each vocabulary tile to its top-$k$ candidates with the tile max and
      log-sum-exp, the summary item 4 proposes for uncontrolled sampling.
      Top-k and top-p then merge candidates instead of searching the full row
      for a pivot. The log-sum-exp bounds the mass outside the candidates;
      when top-p needs more than they hold, the row falls back to today's
      path, so the result stays exact. The top-p sum's order changes, so fix
      its tie rule and compare token for token at fixed seeds.

## Speculative decoding

In `1107`, DSpark online confidence calibration (`0746`), CopyDrafter
continuation-priority tie-breaking (`0744`), descending-priority warp early
exit (`0744`), expected-yield gating against DSpark (`0745`), native-depth CUDA
graph retention (`0728`), fused single-rank Markov drafting (`0717`), and
device-side `prepare_inputs` folding (`0754`) are all active.

**Improve the first two DSpark draft positions (without training).** Check
drafter parity on identically formatted prompts; we are not training a custom
drafter. Explore input-driven drafting improvements only where they do not hurt
decode step time or throughput. Measure acceptance by position, verification
cost, tokens committed per step and serving tok/s on prose, code and agent turns
with exact target verification.
[First-miss measurements](experiments/2026-11-18-draft-accuracy-sizing/README.md)

Open levers:
- **Single-stream (`c1`) cold-L2 routed-MoE marginal row cost and `DEAD_ROWS_TAU` break-even calibration.** On `sp8` across distinct layer weights (`>2 GiB`, cold L2 matching real serving), 39-layer `MoeExpertsV41Plan` at `M_pad=6` scales with `live_rows = 1..6` (`6 -> 12 -> 18 -> 24 -> 29 -> 34` unique experts/layer) as `4.66 -> 8.76 (+4.10) -> 13.08 (+4.32) -> 17.02 (+3.94) -> 20.57 (+3.55) -> 24.08 (+3.51) ms` (`s3/2` beats `s2/2` across warmed `live=3, 4, 16, 38, 64` at `159.1 vs 160.8`, `192.7 vs 194.6`, `622.8 vs 623.9`, `1423.5 vs 1427.4`, `2369.4 vs 2377.9 us`). Live `dgx1` `c61-r5` `prose-c1` telemetry (`4,701` steady decode steps) shows `36.3%` commit 1 token (`med 25.30 ms`), `30.0%` commit 2 (`25.60 ms`), `21.1%` commit 3 (`26.70 ms`), `8.1%` commit 4 (`28.70 ms`), `3.2%` commit 5 (`30.90 ms`), `1.2%` commit 6 (`33.50 ms`). `c61-r6` (`e0436c584`) tested IRLS warm-start priors (`_WARM_CONFIDENCE_HEAD_INTERCEPTS` + zero-preservation) + `SINGLE_REQ_CALIBRATED_ROW_COST_SCALE = 2.5` (`tau_c1 = 0.1386`) on `dgx1..dgx4`: warmup req `#00` improved `71.66 -> 73.13 dec tok/s` (`ttft` `0.283s -> 0.163s`), `code-c1` improved `114.25 -> 115.04 dec tok/s` (`med_dt 31.50 -> 29.90 ms`), and `c16` (`271.38`) + determinism (`0` diffs) held, but `prose-c1` shifted `-1.83 ms/step` (`26.58 -> 24.75 ms` mean) against `-0.163 commits/step` (`2.144 -> 1.981`, `81.38 -> 80.72 dec tok/s`), showing the pruned `pos 2..4` tail tokens earned `89.1 tok/s` marginal rate (conditional acceptance > independent confidence product). Keep warm-start priors and drop `scale = 2.5` back to `1.0` in `c61-r7`. In `c61-r9` (`60caa95be`, `tau = 0.06335`), `cov = [1.0, 0.861, 0.482, 0.225, 0.100]` (`2.667`) vs un-squared `sqrt_cov = [1.0, 0.928, 0.694, 0.474, 0.316]` (`3.412`), whereas `c61-r6` (`tau = 0.1386`) raised conditional acceptance `surv/cov` to `[0.662, 0.507, 0.509, 0.543, 0.557]` while squaring `cov` down to `[1.0, 0.619, 0.262, 0.111, 0.056]` (`2.048`) vs un-squared `2.869`: root cause was async `_stale_confidences` ($C_{t-1}$) on the host (`get_num_tokens` / `reallocate_drafts`) intersecting current `_confidence_probs` ($C_t$) on the device (`DeadRows`), squaring per-step survival coverage to $p_k^2$. `c61-r10` restores `SINGLE_REQ_CALIBRATED_ROW_COST_SCALE = 2.5` (`tau_c1 = 0.1386`) for device `DeadRows` while admitting all `<=5` single-tile DSpark slots on the host (`64 passed, 1 skipped` in `test_adaptive_verification.py` + `test_acceptance_estimator.py`). On `tp4-ring` (`c61-r10`, `2f9eaee88`), `c1` reached **`84.03 ± 6.13% tok/s`** (`code = 112.99 e2e / 118.71 dec`, `prose = 80.06 e2e / 82.64 dec`, `acc/draft = 1.4042` vs `1.3443` in `r9`, `c16 = 271.23 ± 0.87%`, `0` greedy diffs). `c61-r11` (`4cd801eb2`) kept `dspark.main_proj` `TILE_CONTIGUOUS_FP8` (`c16 = 271.31 tok/s`, `tpot_p50 = 11.74 ms`, `0` diffs) while `scale = 1.75` (`tau ~ 0.1035`) measured `c1 = 83.90 tok/s` (`+0.205` rows/step at `+0.42 ms/step` = `2.03 ms/row` for only `+0.0181` accepted tokens/draft = `8.84%` marginal survival, `43.5 tok/s` < `84.03`), so `c61-r12` restores `scale = 2.5`). `c61-r12` (`fcced2855`, `SparseMlaV41Plan` `"split"` `skip_inactive=True`, `-8.10..-13.66 us/L` at `idx_len=16`, `-5.87..-11.81 us/L` at `idx_len=512`, `0` new JIT files) measured **`c1 = 84.75 tok/s`** (`35.707 steps/s`, `prose = 80.78/83.39`, `code = 113.59/119.36`, `c16 = 271.60`, `0` greedy diffs). `c61-r13` (`179fd3b75`, `metadata.py` + `attention.py` `is_padding` dead-row masking, `153 passed, 0 PDL violations`, `IndexerCand -23.09..-28.11 us/L`) measured **`c1 = 84.78 tok/s`** (`35.720 steps/s`, `c16 = 271.66 tok/s`, `0` greedy diffs). (`PR #1002` `VLLM_DSPARK_ONLINE_ADAPT=1` measured `+0.03%` on held-out `c1` (`87.11 -> 87.14`) and `+0.10%` on coding (`160.91 -> 161.07`) in `c57-adapt`; it is not in `1120`: `VLLM_DSPARK_ONLINE_ADAPT` exists only in the patches of `experiments/2026-11-08-online-dspark-adaptation`, merged as a record.)
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
  draft depth schedules on recorded agent traffic and the paired Sonar
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
  Shipped in `1120` (`context_kv_front_3layer`, `0817` and `0820`;
  `candidate-61` Rank 10).
- **Drafter reuse of the target's token selection (`experiments/2026-11-08-dspark-topk-selection-reuse`, that experiment's own vLLM `0762`).**
  Not in `1120` (#1004 is merged as a record; the series' `0762` is a different
  patch), and it would change nothing there while the drafter's attention is
  window-only.
  Measured across 2 boots on `sp5` (`4K..256K`, `TP=4` and `TP=3`, `c1` and `c8`).
  In `1107`, `DeepSeek-V4.1-Flash` configures `compress_ratios = [0, 0, 0]` on all
  3 DSpark draft layers (`layers.40..42`, `mtp.0..2`), so production DSpark draft
  attention is already window-only (`swa_width = 136`, `idx_width = 0`, `0.00 ms`
  indexer work). When indexed draft attention (`compress_ratio = 1`, `idx_width = 512`)
  is enabled, reusing the target model's accepted anchor row top-$k$ selection
  (`gather_dspark_anchor_topk_indices`, `"target"` mode) or step-0 anchor top-$k$
  (`broadcast_dspark_step0_topk_indices`, `"anchor"` mode) requires zero new
  checkpoint weights, keeps 3-layer DSpark indexed attention $O(1)$ (`0.080–0.085 ms`
  at `c1`, `0.193–0.198 ms` at `c8` on `TP=4`) across `4K..256K` (saving up to
  `2.69 ms/step` / `14.58x` vs 1 shared draft indexer and `8.10 ms/step` / `41.86x`
  vs 3 per-layer draft indexers at `256K c8`), and preserves `97.1%–98.8%` top-$k$
  index overlap, `0.999975–0.999986` 3-layer cosine similarity, and `95.8%–100.0%`
  (`98.5%` mean) 5-step greedy Markov draft token agreement.
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
- **End-of-turn drafter learning with the learned table drafting (stage 4, not
  run).** [`experiments/2026-11-17-dspark-turn-learning`](experiments/2026-11-17-dspark-turn-learning/README.md)
  ran its first three stages: observe, the offline study and shadow. Offline,
  an hour of AgentX traffic raises teacher-forced accepted tokens by about
  `1.5%` with the full-vocabulary rule and `0.9%` with the rule serving can
  run, both still rising; that is an upper bound on what verification
  realizes. Shadow costs the grid's decode forward `0.43%` with its
  communicator made at the first exchange. Stage 4, drafts read from the
  learned table while it wins over a window and reset when it loses, was not
  run, so no serving speed has been measured with learned drafts and the
  entry under "Tried and not adopted" records the shadow's cost only. Run it
  on AgentX traffic over two boots and weigh the result against that cost.
- **Per-position calibration alone (the arm #1002 did not run).**
  [`experiments/2026-11-08-online-dspark-adaptation`](experiments/2026-11-08-online-dspark-adaptation/README.md)
  measured calibration, W1 and confidence-head updates together and lost. Its
  record names the arm that would separate them:
  `VLLM_DSPARK_ONLINE_CALIBRATION=1` against off, to say whether the
  per-position calibrator alone shortens `0504`'s budgets. The code is in that
  experiment's patch only.
- **`log p_max` beside the top-2 margin, and a margin for sampled requests.**
  `1120` ships the margin, sized at `+1.3` to `+1.8%` at single-stream row
  costs; adding `log p_max` sized at `+1.7` to `+2.3%` but needs a sum over
  the vocabulary that the certified screen avoids
  ([`experiments/2026-11-17-draft-calibration-sizing`](experiments/2026-11-17-draft-calibration-sizing/README.md)).
  Sampled requests' draft steps measure no margin and calibrate as before.
  The screen ran two workloads on `tp4-ring` only: no AgentX run, no other
  shape and no soak. Open: a `log p_max` the screened head can return without
  the full sum, a margin on sampled draft steps, and the AgentX run.
- **#953 at the sizes it was written for.**
  [`experiments/2026-11-08-cohort-sync-verification`](experiments/2026-11-08-cohort-sync-verification/README.md)
  starts from identical-prompt prose `c16` falling from `1030` to `1107` at
  `TP6` (`-9.6%`), `TP7` (`-9.6%`) and `TP8` (`-13.5%`) while `TP4` and `TP5`
  rose. Its change was served only on `tp4-ring`, together with #986, and
  reverted there (under "Tried and not adopted"). The drop has not been read
  again on `1120`. Read prose `c16` on `tp6..tp8-switched` first; if it is
  still there, screen the burst coalescing alone on `tp8-switched`.

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
shield + pinned two-plane scales (`pin_scales`, on by default), PR `#1010`
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
  - Compare the fused candidate's selected indices and tie handling against
    the exact reference. Measure cold 131K and 1M prefill on two boots against
    the published 1120 ring runs (about 10.1K and 9.39K tok/s respectively).
    [Cold-prefill reference](manifests/suites/2026-11-20-spark-ds41f-1120-tp4/prefill.json)
- **Rank-offset CTA remapping for sequence-parallel all-gather overlap.** `_sp_front` /
  `finish_front` (`defer_q`) already hides `q_all_gather` in attention; if new fused
  SP consumers are introduced later, remap consumer row-tile coordinates by rank offset
  (`pid_m = (raw_pid_m + rank * tiles_per_rank) % total_tiles`) so every rank
  starts computing on its local shard at $t = 0$ while peer shards arrive over RoCE.
- **Fast-pathing or graph-capturing the constant `128`-row CED decoder tail
  (`Layers 20..60`) on short cached-delta prefills (`128..1,024` uncached
  tokens).** With `--enable-prefix-caching` and `--block-size 256`, interactive
  chat and multi-turn agent steps typically prefill a `256–512` token uncached
  delta (`c1` `TTFT p50 = 109 ms` on `tp4-ring`). Because CED trims every
  prefill `>= 128` tokens to `M = 128` rows at Layer 20 (`ced_keep_tokens =
  128`), all 41 decoder layers (`Layers 20..60`) execute at an invariant
  `M = 128` shape where single-wave `128`-row decode tiles (`WO-A`, `WO-B` in
  `0762`, `fused_wqa_wkv`, `shared_expert`, `vocab`) and `sparknet` one-shot
  RoCE all-reduce (`128` rows `= 1.84 MiB < 2 MiB`) apply. Screen capturing or
  piecewise-graphing the fixed `128`-row non-attention blocks across Layers
  `20..60` (and routing `M = 128` CED tails through the `128`-row decode tile
  set) to remove eager launch overhead on short prefills.
- **`256`-token block-boundary prefix sealing for multi-agent forks.** Under
  `--block-size 256`, when a coordinator forks $N$ subagents from a shared
  prompt or transcript of $256k + r$ tokens ($0 < r < 256$), only the $k$
  complete `256`-token blocks hit the prefix cache; the trailing $r$ shared
  tokens sit in an unsealed block and are re-prefilled and stored $N$ times.
  Screen sealing the shared trunk across all $N$ children (either by warming
  the longest common prefix once before fanning out the $N$ short suffixes or
  by aligning the shared system/context boundary to a `256`-token block
  multiple).
- **NVMe interrupt coalescing stays off** on every node by the owner's decision
  (2026-10-02; prefill measured 0.7–1.6% slower with it on in
  `experiments/2026-10-02-nvme-coalescing`).
- **Live validation of prefix-cache retention and `defer_loading` tool schemas:**
  [#902](https://github.com/christopherowen/spark-ds41f/issues/902) hybrid retention/eviction
  (`test_prefix_cache_retention_probe.py`) and inline `tool_reference` schema injection
  (`defer_loading: true`) shipped in `candidate-60`; confirm live 64–100K prefix-cache
  retention on production agent traffic.
  Screen warming the exact token prefixes shared by repository context,
  documents and tool schemas. Compare first use, warm reuse and reuse after
  cache pressure. Record warmup cost, retained KV bytes, cache-hit tokens,
  TTFT and token parity with cold execution.
- **Follow-up-turn TTFT is not measured.** The suites time cold prefill and
  short requests beside one; none times the second turn of a conversation whose
  prefix is cached (256-token blocks). Time a follow-up turn on a 4K and a 900K
  conversation on `1120` and record it beside the release references.

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
    `ISO_YES | GPU_CACHEABLE_YES` / `NO`) and CUDA VMM contiguous stitching:**
    In `c61-r16`, `vllm/v1/worker/display_carveout.py` replaces `/dev/dri/card0`
    (`DRM_IOCTL_MODE_CREATE_DUMB` + `cuImportExternalMemory`) with direct
    `/dev/nvidiactl` calls (`NV_ESC_RM_ALLOC` `NV01_MEMORY_SYSTEM` with
    `NVOS32_ATTR2_ISO_YES`, `2 MiB` huge pages, and selectable
    `GPU_CACHEABLE_YES` / `GPU_CACHEABLE_NO`, exported via
    `NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD` and imported via CUDA VMM
    `cuMemImportFromShareableHandle` + `cuMemAddressReserve` + `cuMemMap`,
    immediately closing all RM/FD handles and enforcing `_MAX_TOTAL_BYTES = 1968 << 20`).
    Verified on `sp8` (`candidate-60`, `-v /dev/null:/dev/dri/card0`, `22 passed`):
    both RM modes stream at `251–257 GB/s` sequential and **`230.6–234.4 GB/s`
    random-576B** (vs `10.2 GB/s` on `card0`), `GPU_CACHEABLE_YES` matches `cudaMalloc`
    (`1.00x`) on large-M ViT GEMMs, `GPU_CACHEABLE_NO` avoids L2 allocation on
    single-pass `W_q` (`rm_HYBRID`: `lm_head(M=6) = 718.02 us` vs `card0_WQ_ONLY = 716.46 us`
    and `cudaMalloc = 767.20 us`), and `rm_hybrid` (`776 MiB` carve-out) reclaims
    **`+868.92 MiB`** host `MemAvailable` over `cudaMalloc` (`rm_wq_only`: `+128.67 MiB`).
    The `/dev/dri/card0` bind-mount and `c 226:* rw` rule have been removed from
    `config/cluster.json` and `spark_ds41f/lab.py` (with `doctor` downgrading
    missing `modeset`/`drm_card` to warnings when `/dev/dri/card0` is not
    mounted). Still open: test VMM-stitching (`cuMemMap`) any remaining scanout
    capacity onto the KV cache or other read-once/cached tables.
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
- **Qualify sampled schemas, tool calls and agent task completion.** The 1120
  gate covers grammar-bitmask sizing and quoted DSML closers in tool arguments.
  Its two-boot TP4 reference matched 256/256 greedy token sequences. Retain
  the native token-ID check with logprobs disabled. Add fixed-seed sampled
  cases for schema validity and streamed tool names and arguments, including
  quoted closers. Score held-out agent tasks through natural completion under
  a fixed tool policy. Report correct completions per wall-clock hour, with
  failures, truncations and timeouts counted in the workload.
  [Parser and grammar fixes](experiments/2026-10-26-spark-ds41f-candidate/candidate-62/README.md) · [Greedy token comparison](manifests/suites/2026-11-20-spark-ds41f-1120-tp4/release-table.txt)
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
- **Opt-in Latent Sparse Attention (`VLLM_DS41_LSA`, `0741` in `1107`, `0828`
  in `1120`):** ships disabled by default (`VLLM_DS41_LSA` unset) with full
  runtime plumbing and unit tests; `0828` moves the opt-in path to FP8 group-32
  dominance summaries and declares no LSA kernel when unset. Any future
  enablement requires an independent long-context answer quality evaluation.
  Evidence so far is for `0741` on `1107`: a second site's `tp4-ring` read 1M
  source-text prefill at `8,722` against `6,342 tok/s` with it on (#1017,
  [`experiments/2026-11-09-site-d-lsa`](experiments/2026-11-09-site-d-lsa/README.md)).
  `0828` has its unit tests and the default-off check only: no boot has served
  with `VLLM_DS41_LSA=1` on `1120`. Run prefill, decode and that record's
  quality cases with it on before the question of enabling it is put. An
  offline study of `0741`'s selection on three documents' captured operands
  ([`experiments/2026-11-17-lsa-recall-breakdown`](experiments/2026-11-17-lsa-recall-breakdown/README.md))
  found:
  - the native keys it drops are lone picks in their 8-key block (91–100% of
    misses);
  - a better block mean recovers only 1.2–2.1 points, and a 512-key recent
    window 0.0–0.9;
  - only a larger budget moves recall: dense ×4 reaches 98.1–99.3% mean.

  `0828`'s summaries were not in that study.
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
paths run on TileLang + `sparknet`). `1120` retires B12X completely (`0802`,
#1096).

Remaining kernel-level investigations:
- **TMA stage releases outside DS4.1's active kernel set**
  (`experiments/2026-09-30-proxy-fence-audit`): all serving kernels are fenced
  (and enforced in the compiler via `tilelang/0001..0013` in `1107`); keep the
  compiler barrier tests in the release gate whenever TileLang is refreshed.
- **Two TileLang pull requests are open since 2026-11-08 and not in `1120`:**
  [#1015](https://github.com/christopherowen/spark-ds41f/pull/1015) (audit of
  upstream TileLang commits since the `1107` pin, staging six patches under
  its own numbering) and
  [#967](https://github.com/christopherowen/spark-ds41f/pull/967) (vectorized
  `reinterpret` and merged read-state branches in `sparse_mla`;
  `candidate-61`'s record notes its `sparse_mla.py` fixes as verified
  byte-identical in `candidate-60` across all 6 routes). #1015's 2-split
  round-to-nearest TF32 for `mhc.project_streams` moved `reference_sha256` in
  `c61-r2` and was reverted, so that part needs the owner's decision; the rest
  of both has not been screened in a candidate.

## Upstream and hygiene

- **No fallbacks** (owner, 2026-10-14): every collective or kernel dispatch path
  must either be the tested production path or raise with a traceback.
- **No silent test-gated defaults:** best production paths must be active by
  default without requiring environment opt-ins or disabling guards.
- The one-read `fill_defaults` fix (`0019`) is still needed on PyTorch main.
- The NCCL IB send-path fence patch should be dropped once upstream releases the
  fix.
- **Records of incorporated pull requests live only on their branches.** 58
  experiment folders (`experiments/2026-11-06-*` to `2026-11-12-*`, most of
  them the records of pull requests closed as incorporated into
  `candidate-50..61`) are on unmerged pull-request branches and not on `main`
  (counted 2026-11-20). Bring them to `main`, or keep those branches: pruning
  one drops its record.
- **Closing notes of #956 and #957 cite patches `0792`, `0795` and `0796`,
  which no series on `main` has.** Their changes ship in `1120` inside the
  consolidated `0813`, `0814` and `0816`.
- **The RoCE trace tooling of #962 is in no series.**
  `experiments/2026-11-08-network-iket-roce-trace` is merged as a record; its
  `sparknet` and vLLM patches were written before `sparknet` `0002`/`0003`;
  check that they still apply before the next collective timing (Decode #11)
  relies on them.
- **`_c1_calibrated_row_cost` (`0820`, `adaptive_verification.py`) is defined
  and never called.** Remove it or wire it in with the next change to that
  file.
- **Site reports without an answer (2026-11-20):** #1036 (above), #1211
  (near-tie acceptance, under Speculative decoding), #1230 (32 sequences on
  `tp4-ring` measured on `1107`; `1120` ships 32), #1233 (cross-request recall
  proposal), #1237 (whether the `tp4-ring` minimum free RAM is a measured
  floor) and #1243 (`1120` at TP1 on a DGX Station GB300, `sm_103a`: a new
  hardware report).
- **Four records misname their sources or have no outcome:**
  - `candidate-62`'s README lists issues #1224 and #1049 as pull requests it
    took;
  - the `1120` baseline's notes credit `0821` (the `nvidiactl` carve-out) to
    #1225, whose patches are `0817..0819`;
  - `experiments/2026-11-14-c56-batch1` and
    `experiments/2026-11-14-c56-host-and-adapt` end without a result.

  The baseline is immutable, so correct its attribution in the next release's
  notes. Fix the README, and write the two outcomes from their receipts.

## Tried and not adopted

Don't repeat these without a new reason:

- **Verification and drafting:** marginal verify rule (`0006`); gathered Markov
  bias (`0007`); verifying every draft (`VERIFY_RULE=all`, −5 to −14% at eight
  streams); cost-aware dead-row count (`0011`); `main_proj` split (`0013`);
  lowering `VLLM_DS41_VERIFY_ROW_COST` to `0.010` or `0.020`
  (`experiments/2026-11-06-verification-budget-hardware`, regressed throughput
  vs `0.030`); minimum-match-8 CopyDrafter policy (dropped at source review);
  C39/C41 CopyDrafter preparation-overlap variations
  (`experiments/2026-11-04-copy-overlap-rework`); widening what CopyDrafter
  reaches (`experiments/2026-11-17-copy-drafter-reach`: with the prompts in the
  history every rule change is within ±0.1% or a loss, scanning the whole
  history gains `+0.04%`, minimum matches below 16 lose, and 86.6% of steps
  have no 4-token match anywhere in the request); lowering `effective_dead_rows_tau`
  ~3.7× at `c1` (`PR #1003` sunk-cost rule, `c61-r1-2`: pushed low-survival tail
  rows into 384-expert MoE, adding `+1.78 ms/step` and regressing `c1` `-3.25%`);
  cohort-atomic budgeting + `COALESCED_ROW_COST_SCALE=0.28` (`PR #953`/`#986`,
  `c61-r3`: pushed `c16` `sched/step` to `40.05` and regressed `c16` `-3.3 tok/s`; the two were served only together, on `tp4-ring`, and reverted together, so neither has a result of its own and #953's burst coalescing has none at `TP6..TP8`);
  online verification-driven DSpark adaptation, per-position calibration plus W1
  and confidence-head updates after every step
  (`experiments/2026-11-08-online-dspark-adaptation`, #1002: c1 decode −3.8% coding
  and −5.5% held-out at `tp4-switched`; shorter `0504` budgets and ~0.8 ms a step;
  the end-of-turn shadow learner, `experiments/2026-11-17-dspark-turn-learning`,
  in no patch and off by default: drafts never read its learned table, so no
  serving gain is measured. It costs the grid `+0.37..+0.48%` from its per-step
  capture, and its end of turn costs nothing measurable once its communicator
  is made after startup; made at load, `+1.06%`, see Decode #17).
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
  fused in `attention_front`); intra-request `KPool` register-tile prefix reuse
  with `<1>`/`<6>`/`<4>` row-block specialization in `indexer.py`
  (`experiments/2026-11-15-decode-prefix-reuse`: bit-exact and faster alone at
  `rows=6..32`, `0.121` against `0.154 ms` at 32, but no decode gain over two
  boots and source prefill `0.73–1.88%` behind `c58` Step 3); and Decode #13 mid-network Layer 16/20 routed-MoE
  pruning (DS4.1 safetensors carry no intermediate Layer 16/20 LM head/probe
  weights). Other `sp8` Cold-L2 decode screens: `dspark.main_proj` (`1280x15360`,
  `M=6`) split-K `shards=2/3/4` regressed `104.71 -> 109.71 / 104.88 / 109.63 us`
  while `tile_contiguous` `tc64` (`block_N=64`) improved to `100.72 us` bit-exact
  (`-4.0 us/step`); and reduced-SMEM `mhc_project_streams` (`two_buf` `40 KiB` /
  `one_buf` `24 KiB`) regressed `15.45 -> 17.73 / 18.90 us` due to extra
  `__syncthreads` barriers across the K loop. Live 4-node CUPTI (`dgx1..dgx4`) +
  36-shape `engine_grid` on `c61-r14` (`00306d1ef`) also ruled out three `r14`
  changes: (a) 2-CTA burst-reset `l2_prefetch` + 256-CTA `wo_a` (`block_N=32,
  block_K=128`) — un-paced 2-CTA `l2_prefetch` (`53–56 us`) collides with
  short-context Target `wo_a` (`sparse_mla 3.6–10.0 us`), stretching `wo_a` from
  `29.8 -> 65.6 us/L` (`+1.43 ms/40L`) and slowing `16x96` by `+1.061 ms` (the
  original 1-CTA `200 ns`/chunk pacing in `l2_prefetch` is intentional bandwidth
  throttling); (b) pre-`pdl.chain()` `Fn` copy & L2 prefetch in `mhc.py` —
  pre-FFN `mhc_project_streams` (`grid=(1, 20)` across all 20 SMs) launches
  during `oneshot_allreduce` (`41.9 us` RoCE ring), stretching pre-FFN `mhc`
  from `6.5–6.6 us` to `49.3–74.0 us` (`3.09–3.81 ms/step` vs `1.54–1.65 ms`);
  and (c) CUDA Graph conditional `IF` Markov halting — adds `10–14 us` per
  conditional stream boundary (`+0.34 ms` `gap_ms` across all 36 `engine_grid`
  shapes and `+0.05..+0.15 ms` `dft_ms` at `8x..16x`). Direct producer writes
  into `sparknet` `pin_memory=True` `send_slot` (`TODO` Decode #2) were also
  retired after live 4-node CUPTI confirmed `0` pre-AR `cudaMemcpyAsync`
  staging copies (`0.0006 ms/step` total `gpu_memcpy` at `c1`; `roce_copy_pack`
  stages 16B-coalesced L2-hot `inp` in-kernel) and `sp8` Cold-L2 `COPIES=12`
  measurements showed `wo_b` `pin+fence` `+2.30..+51.06 us/L` slower (`M=1..48`)
  and `moe_chain` `+0.14..+1.00 us/L` slower (`M=3, 6, 48`, vs `-0.30..-2.51 us/L`
  for `skip_stage` alone). A faster pace for the FFN prefetch window alone (`100`,
  `50`, `0 ns` against `200 ns`, candidate-61 at `tp4-ring`) was `0.48–0.64%`
  slower on the grid's geometric mean than both reference boots, and about `2%`
  slower on the one-request shapes
  (`experiments/2026-11-18-ffn-prefetch-pace`).
- **Prefill:** M32 MoE tile (+1%); fused M32 MoE (no gain, less headroom);
  `NCCL_PROTO=Simple` (no gain); `--max-parallel-prefills 8` (four-64K mean TTFT
  41 → 67 s); sequence parallelism below 205 tokens (+12 ms TTFT on short
  prompts); `8` or `12` NCCL Ring channels on TP3 SP prefill
  (`experiments/2026-11-06-nccl-sp-channels`, `6` channels won while `8` and
  `12` regressed); dense `indexer_score_v41` `<2>` (`2 <= rows < 64`) with
  `16 KiB` `contrib` SMEM (`+0.34%` `@4K` decode regression vs `<1>` + `8 KiB`
  `contrib` kept for `rows < 64`; `0817` in `1120` calls
  `__tl_score_rows_dense_fast<2>` there all the same, as `candidate-61`'s
  Rank 6, so this entry no longer describes what ships) and `<8>` (`rows >= 128`, `B_reg[8]` register
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
