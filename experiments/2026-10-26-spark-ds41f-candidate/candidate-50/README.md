# candidate-50

Unpromoted release candidate for release qualification
([release-benchmark](../../../docs/release-benchmark.md#9-release-qualification-final-benchmarking), step 9):
spark-ds41f-1107 plus every PR that passed PR validation in candidates 47-49, each reworked PR as its
corrected patch. 18 PRs, 19 patches. It is qualified on `tp4-ring` (dgx1..dgx4), `tp4-switched`
(sp1..sp4), `tp3-switched` (sp5..sp7), `tp8-switched` (sp1..sp8) and `tp3-triangle` (dgx1..dgx3),
the shapes 1107 recorded.

## Source

- vLLM: 1107's series (`0001..0761`, 641 active) plus the patches below, renumbered from `0762` in
  application order ([vllm/series](vllm/series)): candidate-47's, then candidate-48's, then
  candidate-49's, each in its candidate's order. Each patch is copied from `main`: the PR's patch
  as its candidate carried it, or the rework's corrected patch. The provenance and SHA-256 of each
  are in [receipts/selection.json](receipts/selection.json), and the stack log is
  [receipts/stack.log](receipts/stack.log).
- `patch_count` 660, `patch_head` `ba7400d9`, `expected_tree` `d5778ab5`,
  `patchset_sha256` `38d230e9`, recorded by `bin/spark build prepare --lock`
  ([source.json](source.json), [upstreams.lock.json](upstreams.lock.json)).
- Every other source is 1107's.

| Patch | PR | Title | Source | Verdict |
| --- | --- | --- | --- | --- |
| `0762-perf-dsv41-evict-first-l2-policy-on-routed-moe-tma-weights.patch` | #966 | perf(dsv41): default unswizzled routed MoE TMA weight loads to L2 evict_first | candidate-47, PR head `06695dd7b829` | [pass](../candidate-47/receipts/own-marks/pr-966/result.json) |
| `0763-perf-dsv41-certified-two-pass-lm-head-vocab-screen.patch` | #997 | perf(dsv41): certified two-pass Cauchy-Schwarz lm_head greedy vocabulary screening | candidate-47, PR head `32151eb3cfed` | [pass](../candidate-47/receipts/own-marks/pr-997/result.json) |
| `0764-perf-worker-eliminate-decode-step-h2d-and-aten-overhead.patch` | #965 | perf(worker): eliminate decode-step redundant H2D transfers, UVA copies, and eager ATen ops | [#965 round 1](../candidate-47/receipts/rework/pr-965-r1) (candidate-47) | [corrected-meets-mark](../candidate-47/receipts/rework/pr-965-r1/result.json) |
| `0765-perf-dsv41-overlap-decode-kv-index-writes-with-query.patch` | #1012 | perf(dsv41): overlap decode KV/indexer-key writes with query/index-query on aux_stream | [#1012 round 1](../candidate-47/receipts/rework/pr-1012-r1) (candidate-47) | [corrected-meets-mark](../candidate-47/receipts/rework/pr-1012-r1/result.json) |
| `0766-perf-dsv41-align-qb-and-indexer-qb-tile-contiguous-decode.patch` | #1016 | perf(dsv41): align Q-B and indexer Q-B decode tiles with 128x128 tile-contiguous weights | [#1016 round 1](../candidate-47/receipts/rework/pr-1016-r1) (candidate-47) | [corrected-meets-mark](../candidate-47/receipts/rework/pr-1016-r1/result.json) |
| `0767-perf-dsv41-sparse-mla-stage1-kv-gather-and-barrier-opt.patch` | #964 | perf(dsv41): sparse MLA stage-1 KV gather and staging barrier opt | candidate-47, PR head `d6d69b92afb9` | [pass](../candidate-47/receipts/own-marks/pr-964/result.json) |
| `0768-deepseek-v41-dsml-token-provenance.patch` | #955 | Require the DSML token for DeepSeek V4.1 tool call markup (#949) | candidate-48, PR head `3a854ba440e1` | [pass](../candidate-48/receipts/own-marks/pr-955/result.json) |
| `0769-perf-spec-decode-elide-tp-confidence-broadcast-and-lock-sampling-seeds.patch` | #980 | perf(spec_decode): elide TP confidence broadcasts and lockstep TP sampling seeds | [#980 round 1](../candidate-48/receipts/rework/pr-980-r1) (candidate-48) | [corrected-meets-mark](../candidate-48/receipts/rework/pr-980-r1/result.json) |
| `0770-fix-structured-output-enforce-completions-grammar-un.patch` | #1013 | fix(vllm): enforce completions structured-output grammar and add size-classed prefill lanes | candidate-48, PR head `aeba4fc4ac40` | [pass](../candidate-48/receipts/own-marks/pr-1013/result.json) |
| `0771-feat-serve-sse-keepalive-heartbeat-and-abort-cleanup.patch` | #972 | feat(serve): enable periodic SSE keep-alive heartbeats and shield streaming abort cleanup | candidate-48, PR head `3827857dc3dd` | [pass](../candidate-48/receipts/own-marks/pr-972/result.json) |
| `0772-fix-entrypoints-move-model-inspection-RPC-out-of-LLM.patch` | #1011 | experiments(vllm): audit Nov 2026 mainline PRs and backport 0762..0764 | candidate-48, PR head `8904a1fec170` | [pass](../candidate-48/receipts/own-marks/pr-1011/result.json) |
| `0773-fix-attention-keep-FA-sliding-window-active-under-ba.patch` | #1011 | experiments(vllm): audit Nov 2026 mainline PRs and backport 0762..0764 | candidate-48, PR head `8904a1fec170` | [pass](../candidate-48/receipts/own-marks/pr-1011/result.json) |
| `0774-perf-dsv41-fast-path-multi-turn-chat-rendering-and-s.patch` | #973 | perf(dsv41): fast-path multi-turn chat rendering and segment tokenization | candidate-48, PR head `fed91173461f` | [pass](../candidate-48/receipts/own-marks/pr-973/result.json) |
| `0775-perf-dsv41-widen-long-context-indexer-top-k-segments.patch` | #999 | perf(dsv41): widen long-context indexer top-k segments and reuse source blocks | [#999 round 1](../candidate-49/receipts/rework/pr-999-r1) (candidate-49) | [corrected-meets-mark](../candidate-49/receipts/rework/pr-999-r1/result.json) |
| `0776-perf-dsv41-vectorize-indexer-KPool-loads-and-co-sche.patch` (rebased, context only) | #993 | perf(dsv41): vectorize indexer_score_v41 KPool loads and co-schedule dense CTA columns | candidate-49, PR head `a3067d857fda` | [pass](../candidate-49/receipts/own-marks/pr-993/result.json) |
| `0777-perf-dsv41-hw-e2m1x2-indexer-and-gemm-mxfp4-quantize.patch` | #991 | perf(dsv41): use hardware cvt.rn.satfinite.e2m1x2.f32 in indexer and GEMM MXFP4 quantizers | [#991 round 1](../candidate-49/receipts/rework/pr-991-r1) (candidate-49) | [meets its mark, coordinator](../candidate-49/receipts/rework/pr-991-r1/analysis.md) (sub-agent: [corrected-misses-mark](../candidate-49/receipts/rework/pr-991-r1/result.json)) |
| `0778-perf-dsv41-sp-prefill-local-attention-front.patch` (rebased, context only) | #1009 | perf(dsv41): fuse SP prefill local attention front and keep index_weights local | candidate-49, PR head `78e6f933d270` | [pass](../candidate-49/receipts/own-marks/pr-1009/result.json) |
| `0779-perf-dsv41-pipeline-the-sliced-SP-Engram-row-exchang.patch` | #979 | perf(dsv41): route SP prefill gather_many and row exchange through sparknet | [#979 round 1](../candidate-49/receipts/rework/pr-979-r1) (candidate-49) | [corrected-meets-mark](../candidate-49/receipts/rework/pr-979-r1/result.json) |
| `0780-perf-dsv41-cache-SP-local-routing-masks-and-skip-SP-.patch` | #996 | perf(sp-prefill): cache SP local routing masks and skip SP tail pad rows | [#996 round 2](../candidate-49/receipts/rework/pr-996-r2) (candidate-49) | [meets its mark, coordinator](../candidate-49/receipts/rework/pr-996-r2/analysis.md) (sub-agent: [corrected-misses-mark](../candidate-49/receipts/rework/pr-996-r2/result.json)) |

**Carried in part.** #1013 is carried as its grammar fix (`0770`) alone, and #1011 as candidate-48's
`0768` and `0769` alone (`0772` and `0773` here). Two patches were left out at selection, so the
series has no gap where they were:

| Patch (candidate-48) | PR | Why it is left out |
| --- | --- | --- |
| [`0766-feat-sched-size-classed-prefill-lanes-and-aligned-ch.patch`](../candidate-48/vllm/0766-feat-sched-size-classed-prefill-lanes-and-aligned-ch.patch) | #1013 | left out at selection: does nothing on this deployment (owner, 2026-11-09) |
| [`0770-fix-rocm-reserve-kpool-decode-indexer-logits-in-work.patch`](../candidate-48/vllm/0770-fix-rocm-reserve-kpool-decode-indexer-logits-in-work.patch) | #1011 | left out at selection: does nothing on this deployment (owner, 2026-11-09) |

The lanes patch's test files (`tests/v1/core/test_prefill_lane_livelock.py`,
`test_prefill_qos_lanes.py`, `test_aligned_prefill_lanes.py`) are not in the tree and not in
[release-tests.json](release-tests.json).

**Dropped** after rework, with no defensible correction: #986
([round 1](../candidate-47/receipts/rework/pr-986-r1/result.json)), #1005
([round 1](../candidate-47/receipts/rework/pr-1005-r1/result.json)) and #1010
([round 2](../candidate-48/receipts/rework/pr-1010-r2/result.json)).

**Two carried reworks record a sub-agent miss that the coordinator's verdict overrules** (each folder's
`analysis.md`), and the release qualification measures them in place:
- #991 round 1: GEMM M=6 and M=8 read slower in the eager-launch bench. By kernel time (CUPTI) and
  CUDA-graph replay, the path decode serving takes, every cell is faster beyond its interval
  ([step 7](../../../docs/release-benchmark.md#7-own-marks): judge on the path serving takes).
- #996 round 2: 2 non-tail per-call cells and 3 pass cells read slower than 1107 boots recorded in
  another session. The receipt does not attribute them to the patch: unchanged round 1 code
  re-run in the same session is slower in a different set of cells, and the in-process A/B finds all
  three within ([step 7](../../../docs/release-benchmark.md#7-own-marks): run the baseline in the same session).

### Stacking

The candidates were selected to overlap at most in test files; the reworks changed some patches.
On a fresh tree ([receipts/stack.log](receipts/stack.log)), 17 of the 19 apply cleanly in this order:
- `0776` (#993) does not apply after `0767` (#964), and `0778` (#1009) does not apply after `0765`
  (#1012 round 1). In each pair, both patches append a test after the last line of the same test
  file (`tests/kernels/attention/test_deepseek_v41_tilelang.py`,
  `tests/models/test_deepseek_v4_1_attention.py`).
- No other order applies: all six candidate orders and #964 moved after #993 fail
  ([stack.log](receipts/stack.log), section 2).
- `0776` and `0778` are rebased with a context-only change: the hunk's index line, header and
  context lines. Every added line is unchanged. The SHA-256 before and after and the exact diffs
  are in [receipts/rebases](receipts/rebases/rebases.json). The union keeps both tests.

Patches of two candidates also share non-test files, and apply cleanly there:
- `attention.py`: #1012 round 1 and #1009;
- `tilelang/gemm.py`: #1016 round 1 and #991 round 1;
- `model_runner.py`, `acceptance_estimator.py`, `speculator.py`: #965 round 1 and #980 round 1.

Each pair edits different functions, and no semantic conflict was found
([rebases.json](receipts/rebases/rebases.json), `semantic_review`). PR validation never measured
these combinations; the release qualification is their first measurement.

## Cluster configurations

A cluster configuration for every shape the recipe serves, so the candidate can be run on any setup: `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-50/<file> …` with your site's node map. Each one extends `releases/spark-ds41f-1107/cluster.json` at its shape and sets only the image `spark-ds41f:candidate-50`, its tree labels and this lock; no PR changes the recipe. The `-dev` file adds the dev-mode collective RPC that the release suite's engine grid needs. This candidate is qualified at the five shapes marked; the other files are configurations, not part of its runs.

| Shape | Configuration | With dev RPC | |
| --- | --- | --- | --- |
| `tp4-ring` | [tp4.json](tp4.json) | [tp4-dev.json](tp4-dev.json) | qualified here |
| `tp3-triangle` | [tp3.json](tp3.json) | [tp3-dev.json](tp3-dev.json) | qualified here |
| `tp3-triangle-4k` | [tp3-triangle-4k.json](tp3-triangle-4k.json) | [tp3-triangle-4k-dev.json](tp3-triangle-4k-dev.json) |  |
| `tp3-switched` | [tp3-switched.json](tp3-switched.json) | [tp3-switched-dev.json](tp3-switched-dev.json) | qualified here |
| `tp4-switched` | [tp4-switched.json](tp4-switched.json) | [tp4-switched-dev.json](tp4-switched-dev.json) | qualified here |
| `tp5-switched` | [tp5-switched.json](tp5-switched.json) | [tp5-switched-dev.json](tp5-switched-dev.json) |  |
| `tp6-switched` | [tp6-switched.json](tp6-switched.json) | [tp6-switched-dev.json](tp6-switched-dev.json) |  |
| `tp7-switched` | [tp7-switched.json](tp7-switched.json) | [tp7-switched-dev.json](tp7-switched-dev.json) |  |
| `tp8-switched` | [tp8-switched.json](tp8-switched.json) | [tp8-switched-dev.json](tp8-switched-dev.json) | qualified here |

- [image-dgx4.json](image-dgx4.json): build-only profile.

## Release tests

[release-tests.json](release-tests.json) is 1107's classification, plus the test files the carried
PRs add or change (candidate-47's `tests/v1/worker/test_gpu_block_table.py` for #965, and
candidate-48's three #955 parser files), plus 1107's unclassified
`tests/v1/core/test_prefix_group_validity.py`, each gated with its reason. No other file needed a new
classification. `scripts/release_tests.py check` reports no problems on dgx4's prepared tree: 215
selected files, 153 gate and 62 excluded
([receipts/release-tests-check.txt](receipts/release-tests-check.txt)). The gate manifest is
[release-tests-gate-manifest.json](release-tests-gate-manifest.json) (154 files), and the bundle is
[bundles/release-tests](bundles/release-tests).

#1013's own unit tests (`tests/v1/structured_output/test_structured_output_manager.py`) stay excluded,
as in 1107: they need a Hugging Face model that is not on the nodes. Its evidence is the live check
below.

## Determinism gate: declared output changes

Temperature-0 output hashes must equal the 1017 checkpoint's, except where a PR declares a change:

| PR | Declared output change |
| --- | --- |
| #955 | parser output only: quoted DSML stays text |
| #980 | sampled (temperature > 0) realizations |
| #1013 | grammar-constrained completions only |
| #972 | SSE comment frames (keep-alive) |

Every other PR declares none.

## Qualification plan

Step 9 measures this candidate once, completely, against 1107's recorded runs; 1107 is not measured
again. Each testbed opens its window with `scripts/lab.py`, after the hold and idle guards.

1. [build.json](build.json) (`candidate-50-image`), in the dgx window. It builds the image on dgx4,
   copies it along the ring, and runs the release tests ([bundles/release-tests](bundles/release-tests))
   on dgx4. The image is then streamed to sp1 and sp5
   (`ssh dgx1 docker save spark-ds41f:candidate-50 | ssh <head> docker load`) and distributed from each
   head with `build distribute --apply`. Every rank shows the same image ID.
2. **Determinism gate**, first on each testbed:
   - [candidate-tp4.json](candidate-tp4.json) at `tp4-ring`;
   - [candidate-tp4-switched.json](candidate-tp4-switched.json) at `tp4-switched`;
   - [candidate-tp3-switched.json](candidate-tp3-switched.json) at `tp3-switched`;
   - [candidate-tp3.json](candidate-tp3.json) at `tp3-triangle`, once the dgx fabric is re-cabled.
   - `tp8-switched` boots after the `tp4-switched` and `tp3-switched` runs, on sp1..sp8; its release
     arms boot first, so a startup failure shows there.
3. **Cold boots** ([testing](../../../docs/testing.md)): one rank's, every rank's and every JIT cache
   empty, with nothing compiled after preparation.
4. **Release set**: the `release-<shape>.json` specs run `candidate-50` and `candidate-50-again`, each
   with startup memory, against 1107's recorded runs. The promoted runs become each shape's
   `benchmark_reference`.

   | Shape | Spec | Suites | 1107 reference |
   | --- | --- | --- | --- |
   | `tp4-ring` | [release-tp4.json](release-tp4.json) | `release`, `single-stream`, `sampled` | `"release"`: `manifests/suites/2026-11-07-spark-ds41f-1107-tp4` |
   | `tp3-triangle` | [release-tp3.json](release-tp3.json) | `release-tp3`, `single-stream`, `sampled` | `"release"`: `manifests/suites/2026-11-07-spark-ds41f-1107-tp3` |
   | `tp4-switched` | [release-tp4-switched.json](release-tp4-switched.json) | `release`, `single-stream`, `sampled` | `manifests/suites/2026-11-07-spark-ds41f-1107-tp4-switched` |
   | `tp3-switched` | [release-tp3-switched.json](release-tp3-switched.json) | `release-tp3`, `single-stream`, `sampled` | `manifests/suites/2026-11-07-spark-ds41f-1107-tp3-switched` |
   | `tp8-switched` | [release-tp8-switched.json](release-tp8-switched.json) | `release`, `single-stream`, `sampled` | `manifests/suites/2026-11-07-spark-ds41f-1107-tp8-switched` |

   Each companion's reference is the same path with `-single-stream` or `-sampled` appended.
   - The switched specs name their reference path. At those shapes, the recipe's
     `benchmark_reference` is a `bin/spark bench` report, which is not a suite run.
   - 1107's `tp3-switched` record has no prefill step (no `prefill.json`, no `finished_utc`), so
     prefill at `tp3-switched` has no 1107 row.
5. **Quality evals** that 1107 recorded, on the shapes where it ran them: LongBench v2, RULER,
   EvalPlus HumanEval, Vision Eval, tool-eval-bench and the 960K needle
   ([1107/receipts](../1107/receipts/README.md)).
6. **AgentX**: the lab's `agentx` job, 1 hour, `c8`, seed 42, on every shape 1107 reported:
   `tp4-ring`, `tp4-switched`, `tp3-switched` and `tp8-switched`.
7. **Live functional checks** of the carried fixes, as candidate-48 declared them:
   - #955: a live tool call quoting DSML returns complete arguments.
   - #1013: a live `/v1/completions` request with a JSON grammar under the `deepseek_v41` reasoning
     parser returns valid output.
   - #972: a live client disconnect aborts the engine request, keep-alive frames arrive every 15 s,
     and servebench's streams parse.
   - #980: the 1,000-step TP lockstep screen shows 0 rank mismatches, and the sampled suite completes.
   - #1011: its tests pass in the release tests.
8. **Startup**: each rank's lowest `MemAvailable` (the startup memory floor) and the time to ready,
   from the release arms' startup memory, with no JIT after readiness.

The release notes then report against 1107, in the format of
[release-notes.md](../release-notes.md): paired decode at each concurrency, the engine grid,
prefill from 32K to 1M, sampled decode, short-request TTFT during a 131K prefill, greedy identity,
the quality scores, AgentX throughput and prefix-cache hit rate, and the startup memory floor.

## Status

Prepared and locked. Not built, not measured.
