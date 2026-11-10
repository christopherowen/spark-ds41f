# candidate-58

The merged base of round 3 (candidates 55-57): candidate-54 plus every PR that passed PR validation in candidates
55-57, each reworked PR as its corrected patch, and one recipe variable, TP3's L2 prefetch caps. 4 PRs, 5 vLLM
patches and 1 sparknet patch. It is not a release candidate
([release-benchmark](../../../docs/release-benchmark.md#9-release-qualification-final-benchmarking), step 9, "Between
releases"). So it is verified, not release-qualified. Verified on all three testbeds, it becomes the first frozen
candidate ([Frozen candidates](../../../docs/release-benchmark.md#frozen-candidates-pre-releases)), and the next round's
three candidates build on it.

Verification: the determinism gate, the screen and the carried fixes' live checks, on the three testbeds (`tp4-ring`
dgx1..dgx4, `tp4-switched` sp1..sp4, `tp3-switched` sp5..sp7).

## Source

- vLLM ([vllm/series](vllm/series)):
  - candidate-54's series by relative path to its files, as candidate-54 lists them: 1107's `0001..0761` (641
    active), candidate-50's `0762..0780` with #1009 round 1, and candidate-54's `0781..0791`; 671 patches;
  - round 3: 5 patches copied from `main` and renumbered from `0792` in this order: #1008, #988's dead-row part, #956
    r4, #957 part 2, #957 part 4. Each is the patch its candidate carried, or the rework's corrected patch.
- sparknet ([sparknet/series](sparknet/series)): 1107's pin `1fd6a1fb` plus #1008's `0001`, copied from candidate-55's
  and carried in the lock as candidate-55's is: `patch_head` `d50a1b4c` and `expected_tree` `dab8c457`, candidate-55's.
  `patchset_sha256` is `55c0ea53`: the series file differs from candidate-55's only in its comment line. The
  configurations' `local.spark-ds41f.sparknet.tree` label is `dab8c457`.
- The provenance, SHA-256 and author of each patch are in [receipts/selection.json](receipts/selection.json), and the
  stack log is [receipts/stack.log](receipts/stack.log).
- `patch_count` 676, `patch_head` `e5911bdf`, `expected_tree` `4815b7a0`, `patchset_sha256` `e22e7270`, recorded by
  `bin/spark build prepare --lock` ([source.json](source.json), [upstreams.lock.json](upstreams.lock.json)).
- Every other source is 1107's.
- **Authors:** every carried PR is christopherowen's (`gh pr view`, 2026-11-10). Every patch since 1107 (`0762..0796`)
  has Christopher Owen's `From:`, except `0768` (#955), which is koldfrontier's: an owner-approved exception, carried
  since candidate-50. The sparknet `0001` is Christopher Owen's.

| Patch | PR | Title | Source | Verdict |
| --- | --- | --- | --- | --- |
| `0762-perf-dsv41-evict-first-l2-policy-on-routed-moe-tma-weights.patch` | #966 | perf(dsv41): default unswizzled routed MoE TMA weight loads to L2 evict_first | [candidate-50's](../candidate-50/vllm/0762-perf-dsv41-evict-first-l2-policy-on-routed-moe-tma-weights.patch) | [pass](../candidate-47/receipts/own-marks/pr-966/result.json) |
| `0763-perf-dsv41-certified-two-pass-lm-head-vocab-screen.patch` | #997 | perf(dsv41): certified two-pass Cauchy-Schwarz lm_head greedy vocabulary screening | [candidate-50's](../candidate-50/vllm/0763-perf-dsv41-certified-two-pass-lm-head-vocab-screen.patch) | [pass](../candidate-47/receipts/own-marks/pr-997/result.json) |
| `0764-perf-worker-eliminate-decode-step-h2d-and-aten-overhead.patch` | #965 | perf(worker): eliminate decode-step redundant H2D transfers, UVA copies, and eager ATen ops | [candidate-50's](../candidate-50/vllm/0764-perf-worker-eliminate-decode-step-h2d-and-aten-overhead.patch) | [corrected-meets-mark](../candidate-47/receipts/rework/pr-965-r1/result.json) |
| `0765-perf-dsv41-overlap-decode-kv-index-writes-with-query.patch` | #1012 | perf(dsv41): overlap decode KV/indexer-key writes with query/index-query on aux_stream | [candidate-50's](../candidate-50/vllm/0765-perf-dsv41-overlap-decode-kv-index-writes-with-query.patch) | [corrected-meets-mark](../candidate-47/receipts/rework/pr-1012-r1/result.json) |
| `0766-perf-dsv41-align-qb-and-indexer-qb-tile-contiguous-decode.patch` | #1016 | perf(dsv41): align Q-B and indexer Q-B decode tiles with 128x128 tile-contiguous weights | [candidate-50's](../candidate-50/vllm/0766-perf-dsv41-align-qb-and-indexer-qb-tile-contiguous-decode.patch) | [corrected-meets-mark](../candidate-47/receipts/rework/pr-1016-r1/result.json) |
| `0767-perf-dsv41-sparse-mla-stage1-kv-gather-and-barrier-opt.patch` | #964 | perf(dsv41): sparse MLA stage-1 KV gather and staging barrier opt | [candidate-50's](../candidate-50/vllm/0767-perf-dsv41-sparse-mla-stage1-kv-gather-and-barrier-opt.patch) | [pass](../candidate-47/receipts/own-marks/pr-964/result.json) |
| `0768-deepseek-v41-dsml-token-provenance.patch` | #955 | Require the DSML token for DeepSeek V4.1 tool call markup (#949) | [candidate-50's](../candidate-50/vllm/0768-deepseek-v41-dsml-token-provenance.patch) | [pass](../candidate-48/receipts/own-marks/pr-955/result.json) |
| `0769-perf-spec-decode-elide-tp-confidence-broadcast-and-lock-sampling-seeds.patch` | #980 | perf(spec_decode): elide TP confidence broadcasts and lockstep TP sampling seeds | [candidate-50's](../candidate-50/vllm/0769-perf-spec-decode-elide-tp-confidence-broadcast-and-lock-sampling-seeds.patch) | [corrected-meets-mark](../candidate-48/receipts/rework/pr-980-r1/result.json) |
| `0770-fix-structured-output-enforce-completions-grammar-un.patch` | #1013 | fix(vllm): enforce completions structured-output grammar and add size-classed prefill lanes | [candidate-50's](../candidate-50/vllm/0770-fix-structured-output-enforce-completions-grammar-un.patch) | [pass](../candidate-48/receipts/own-marks/pr-1013/result.json) |
| `0771-feat-serve-sse-keepalive-heartbeat-and-abort-cleanup.patch` | #972 | feat(serve): enable periodic SSE keep-alive heartbeats and shield streaming abort cleanup | [candidate-50's](../candidate-50/vllm/0771-feat-serve-sse-keepalive-heartbeat-and-abort-cleanup.patch) | [pass](../candidate-48/receipts/own-marks/pr-972/result.json) |
| `0772-fix-entrypoints-move-model-inspection-RPC-out-of-LLM.patch` | #1011 | experiments(vllm): audit Nov 2026 mainline PRs and backport 0762..0764 | [candidate-50's](../candidate-50/vllm/0772-fix-entrypoints-move-model-inspection-RPC-out-of-LLM.patch) | [pass](../candidate-48/receipts/own-marks/pr-1011/result.json) |
| `0773-fix-attention-keep-FA-sliding-window-active-under-ba.patch` | #1011 | experiments(vllm): audit Nov 2026 mainline PRs and backport 0762..0764 | [candidate-50's](../candidate-50/vllm/0773-fix-attention-keep-FA-sliding-window-active-under-ba.patch) | [pass](../candidate-48/receipts/own-marks/pr-1011/result.json) |
| `0774-perf-dsv41-fast-path-multi-turn-chat-rendering-and-s.patch` | #973 | perf(dsv41): fast-path multi-turn chat rendering and segment tokenization | [candidate-50's](../candidate-50/vllm/0774-perf-dsv41-fast-path-multi-turn-chat-rendering-and-s.patch) | [pass](../candidate-48/receipts/own-marks/pr-973/result.json) |
| `0775-perf-dsv41-widen-long-context-indexer-top-k-segments.patch` | #999 | perf(dsv41): widen long-context indexer top-k segments and reuse source blocks | [candidate-50's](../candidate-50/vllm/0775-perf-dsv41-widen-long-context-indexer-top-k-segments.patch) | [corrected-meets-mark](../candidate-49/receipts/rework/pr-999-r1/result.json) |
| `0776-perf-dsv41-vectorize-indexer-KPool-loads-and-co-sche.patch` | #993 | perf(dsv41): vectorize indexer_score_v41 KPool loads and co-schedule dense CTA columns | [candidate-50's](../candidate-50/vllm/0776-perf-dsv41-vectorize-indexer-KPool-loads-and-co-sche.patch) (rebased there, context only) | [pass](../candidate-49/receipts/own-marks/pr-993/result.json) |
| `0777-perf-dsv41-hw-e2m1x2-indexer-and-gemm-mxfp4-quantize.patch` | #991 | perf(dsv41): use hardware cvt.rn.satfinite.e2m1x2.f32 in indexer and GEMM MXFP4 quantizers | [candidate-50's](../candidate-50/vllm/0777-perf-dsv41-hw-e2m1x2-indexer-and-gemm-mxfp4-quantize.patch) | [meets its mark, coordinator](../candidate-49/receipts/rework/pr-991-r1/analysis.md) |
| `0778-perf-dsv41-sp-prefill-local-attention-front.patch` | #1009 | perf(dsv41): fuse SP prefill local attention front and keep index_weights local | [#1009 round 1](../candidate-50/receipts/rework/pr-1009-r1) (candidate-50), in place of candidate-50's `0778` | [corrected-meets-mark](../candidate-50/receipts/rework/pr-1009-r1/result.json); [booted and screened](../candidate-50/receipts/tp4-switched/README.md) at tp4-switched and [tp3-switched](../candidate-50/receipts/tp3-switched/README.md) |
| `0779-perf-dsv41-pipeline-the-sliced-SP-Engram-row-exchang.patch` | #979 | perf(dsv41): route SP prefill gather_many and row exchange through sparknet | [candidate-50's](../candidate-50/vllm/0779-perf-dsv41-pipeline-the-sliced-SP-Engram-row-exchang.patch) | [corrected-meets-mark](../candidate-49/receipts/rework/pr-979-r1/result.json) |
| `0780-perf-dsv41-cache-SP-local-routing-masks-and-skip-SP-.patch` | #996 | perf(sp-prefill): cache SP local routing masks and skip SP tail pad rows | [candidate-50's](../candidate-50/vllm/0780-perf-dsv41-cache-SP-local-routing-masks-and-skip-SP-.patch) | [meets its mark, coordinator](../candidate-49/receipts/rework/pr-996-r2/analysis.md) |
| `0781-perf-dsv41-64k-chunked-globaltimer-paced-l2-prefetch.patch` | #961 | perf(dsv41): 64 KiB chunked, %globaltimer-paced side-stream L2 prefetch | [candidate-54's](../candidate-54/vllm/0781-perf-dsv41-64k-chunked-globaltimer-paced-l2-prefetch.patch), from candidate-51, PR head `9362f944201a` | [pass](../candidate-51/receipts/own-marks/pr-961/result.json) |
| `0782-perf-dsv41-unconditional-pack12-plane-loads-and-swar.patch` | #963 | perf(dsv41): unconditional pack12 plane loads, byte-perm vocab SWAR, and per-column exception lookup | [candidate-54's](../candidate-54/vllm/0782-perf-dsv41-unconditional-pack12-plane-loads-and-swar.patch), from [#963 round 1](../candidate-51/receipts/rework/pr-963-r1) (candidate-51) | [corrected-meets-mark](../candidate-51/receipts/rework/pr-963-r1/result.json) |
| `0783-perf-dsv41-extend-WO-B-WIDE_DECODE-to-128-rows-and-g.patch` | #987 | perf(dsv41): extend WO-B WIDE_DECODE to 128 rows and guard high-concurrency decode cliffs | [candidate-54's](../candidate-54/vllm/0783-perf-dsv41-extend-WO-B-WIDE_DECODE-to-128-rows-and-g.patch), from candidate-51 (its WO-B part), PR head `9be94ed5310c` | [pass](../candidate-51/receipts/own-marks/pr-987/result.json) |
| `0784-Match-128-row-vocab-head-tile-pipeline-stages.patch` | #1001 | perf(dsv41): match 128-row vocab head tile pipeline stages on GB10 | [candidate-54's](../candidate-54/vllm/0784-Match-128-row-vocab-head-tile-pipeline-stages.patch), from [#1001 round 1](../candidate-51/receipts/rework/pr-1001-r1) (candidate-51) | [corrected-meets-mark](../candidate-51/receipts/rework/pr-1001-r1/result.json) |
| `0785-fix-dsv41-align-moe_swiglu_cast-with-DeepSeek-Expert.patch` | #1030 | fix(dsv41): align moe_swiglu_cast with DeepSeek Expert.forward BF16 rounding | [candidate-54's](../candidate-54/vllm/0785-fix-dsv41-align-moe_swiglu_cast-with-DeepSeek-Expert.patch), from candidate-52, PR head `a7edd33143aa` | [pass](../candidate-52/receipts/own-marks/pr-1030/result.json) |
| `0786-perf-spec-decode-remaining-token-draft-cap.patch` | #968 | perf(spec_decode): clamp scheduled decode verify rows to remaining max_tokens | [candidate-54's](../candidate-54/vllm/0786-perf-spec-decode-remaining-token-draft-cap.patch), from candidate-52, PR head `1445e800b172` | [pass](../candidate-52/receipts/own-marks/pr-968/result.json) |
| `0787-perf-dsv41-direct-insert-packed-engram-cache-and-ali.patch` | #969 | perf(dsv41): direct-insert packed Engram cache rows and align split cache arrays | [candidate-54's](../candidate-54/vllm/0787-perf-dsv41-direct-insert-packed-engram-cache-and-ali.patch), from candidate-52, PR head `aad967cbe897` | [pass](../candidate-52/receipts/own-marks/pr-969/result.json) |
| `0788-perf-cudagraph-target-full-replay-and-post-gate-fork.patch` | #960 | Target FULL CUDA graph metadata/CED replay and MoE post-gate fork | [candidate-54's](../candidate-54/vllm/0788-perf-cudagraph-target-full-replay-and-post-gate-fork.patch), from [#960 round 1](../candidate-53/receipts/rework/pr-960-r1) (candidate-53) | [corrected-meets-mark](../candidate-53/receipts/rework/pr-960-r1/result.json) |
| `0789-perf-spec-decode-copy-drafter-survival-and-tile-cap.patch` | #971 | perf(spec-decode): cumulative survival in copy_apply, single-req 16-row tile cap, and wide-step dead-row mask | [candidate-54's](../candidate-54/vllm/0789-perf-spec-decode-copy-drafter-survival-and-tile-cap.patch), from [#971 round 1](../candidate-53/receipts/rework/pr-971-r1) (candidate-53) | [corrected-meets-mark](../candidate-53/receipts/rework/pr-971-r1/result.json) |
| `0790-perf-spec-decode-copy-drafter-long-context-scan.patch` | #977 | perf(spec_decode): bound CopyDrafter long-context history scan in copy_match | [candidate-54's](../candidate-54/vllm/0790-perf-spec-decode-copy-drafter-long-context-scan.patch), from candidate-53, PR head `4e0244deb7d5` | [pass](../candidate-53/receipts/own-marks/pr-977/result.json) |
| `0791-perf-worker-compact-packed-vocab-tables-and-expand-d.patch` | #998 | perf(worker): compact packed vocab tables and expand display carve-out weights | [candidate-54's](../candidate-54/vllm/0791-perf-worker-compact-packed-vocab-tables-and-expand-d.patch), from [#998 round 1](../candidate-53/receipts/rework/pr-998-r1) (candidate-53) | [corrected-meets-mark](../candidate-53/receipts/rework/pr-998-r1/result.json) |
| `0792-perf-worker-roce-proxy-and-rank0-fast-core-affinity.patch` | #1008 | perf(worker): roce proxy and rank0 fast-core affinity partition | [candidate-55's `0792`](../candidate-55/vllm/0792-perf-worker-roce-proxy-and-rank0-fast-core-affinity.patch), PR head `7f29f8934f00` | [pass](../candidate-55/receipts/mark-1008/README.md): rank 0's step p99 lower beyond the interval, nothing worse |
| `0793-perf-spec_decode-sample-skip-dead-verification-rows-.patch` | #988 | experiments: add dead-row verification sampler skip and sparse top-k draft sampling | [candidate-55's `0794`](../candidate-55/vllm/0794-perf-spec_decode-sample-skip-dead-verification-rows-.patch), PR head `cfb7069e6882`: its dead-row part, split at selection, retitled | [meets its mark](../candidate-55/receipts/own-marks/pr-988/result.json) |
| `0794-perf-dsv41-sp-ratio2-compressor-and-ced-encoder-only-chunks.patch` | #956 | perf(dsv41): SP ratio-2 compressor projection and CED encoder-only chunks | [#956 r4](../candidate-56/receipts/rework/pr-956-r4) (candidate-56): the round-3 rebase re-derived on candidate-54 alone, in place of candidate-56's `0796` | [pass](../candidate-56/receipts/tp4-switched-956r4/README.md) ([rework](../candidate-56/receipts/rework/pr-956-r4/result.json)) |
| `0795-perf-dsv41-branchless-sparse-MLA-stage-2-online-soft.patch` | #957 | TP3/TP4 decode compressor, sparse MLA, and spec-decode fusions | [candidate-57's `0797`](../candidate-57/vllm/0797-perf-dsv41-branchless-sparse-MLA-stage-2-online-soft.patch), PR head `e464d364431a`: its part 2, split at selection, retitled | [meets its mark](../candidate-57/receipts/own-marks/pr-957/result.json) (part 2) |
| `0796-perf-dsv41-fuse-DSpark-confidence-calibration-and-ke.patch` | #957 | TP3/TP4 decode compressor, sparse MLA, and spec-decode fusions | [candidate-57's `0798`](../candidate-57/vllm/0798-perf-dsv41-fuse-DSpark-confidence-calibration-and-ke.patch), PR head `e464d364431a`: its part 4, split at selection, retitled | [meets its mark](../candidate-57/receipts/own-marks/pr-957/result.json) (part 4) |
| sparknet `0001-perf-oneshot-pin-RoCE-proxy-thread-to-dedicated-fast.patch` | #1008 | perf(oneshot): pin RoCE proxy thread to dedicated fast core | [candidate-55's](../candidate-55/sparknet/0001-perf-oneshot-pin-RoCE-proxy-thread-to-dedicated-fast.patch), PR head `7f29f8934f00` | with #1008's `0792` |

**Round 3** (candidates 55-57, each tested on candidate-54):
- **#1008** ([candidate-55](../candidate-55/README.md)) with its sparknet patch: the rank-0 and RoCE-proxy fast-core
  partition. It passed at `tp4-ring` on candidate-55 without #978: rank 0's step p99 lower beyond the interval in all
  four pairings, nothing worse ([mark-1008](../candidate-55/receipts/mark-1008/README.md)). It acts on every shape;
  this candidate's screens are its first runs at `tp4-switched` and `tp3-switched`.
- **#988's dead-row part** (candidate-55's `0794`), as candidate-55 carries it. Its sparse top-k drafting is left out.
- **#956 r4** ([candidate-56/receipts/rework/pr-956-r4](../candidate-56/receipts/rework/pr-956-r4/result.json)), in
  place of candidate-56's `0796`. That patch, #956's round-3 rebase, builds on #957 part 1, which is dropped. r4
  re-derives it on candidate-54 alone: at every TP > 1, `bind()` gathers the unpacked compressor shards once into one
  [1024, 5120] BF16 weight for the ratio-2 SP projection (+10 MiB per ratio-2 layer per rank, +30 MiB in all), and
  decode keeps candidate-54's packed shard. At `tp4-switched`, in one session against candidate-54's code, prefill is
  3.4%, 2.8% and 3.8% faster at 64K, 128K and 256K, 32K and the grid the same
  ([tp4-switched-956r4](../candidate-56/receipts/tp4-switched-956r4/README.md)).
- **#957 parts 2 and 4**, as candidate-57 carries them: the branchless sparse-MLA stage 2, and the fused DSpark
  confidence calibration with the dummy run's active `num_reqs`. Both met their marks
  ([own marks](../candidate-57/receipts/own-marks/README.md)).

**Carried in part:**
- #988: its dead-row part (`0793`), without the sparse top-k drafting;
- #957: parts 2 and 4 (`0795`, `0796`); part 1 is dropped, part 3 superseded, part 5 not defensible (below);
- #956: r4 (`0794`), the round-3 rebase re-derived without #957 part 1;
- from candidate-54: #1013's grammar fix (`0770`) alone, #1011's `0772` and `0773` alone, #987's WO-B part (`0783`),
  and #971 round 1 without the dead-row floor.

**Left out**, so the series has no gap where they were. Each reason and source path is in
[receipts/selection.json](receipts/selection.json) (`left_out`: round 3's first, then candidate-54's record).

Round 3:

| PR | Left out | Why |
| --- | --- | --- |
| #978 | the whole PR, at its round 2 ([patch](../candidate-55/receipts/rework/pr-978-r2/0793-perf-multimodal-prefix-cache-fastpath-under-mm-proce.patch)) | fails the gate after its second rework: the gate's `again` scenario (prompts re-run on prefix-cache hits) differs in one request ([tp4-ring-978r2](../candidate-55/receipts/tp4-ring-978r2/README.md)). A third rework is under way by owner exception; if it passes, it is judged in the next round |
| #957 | part 1, the replicated BF16 compressor ([round 1's patch](../candidate-56/receipts/rework/pr-957p1-r1/0795-perf-dsv41-replicate-the-BF16-compressor-projection-.patch)) | dropped after its rework: round 1 keeps the cold read of the replicated weight, and the design that removes the harm removes part 1's benefit ([candidate-56](../candidate-56/README.md#957-part-1-round-1)) |
| #957 | part 5, the TP3 adaptive-verification constants ([patch](../candidate-57/receipts/rework/pr-957-p5-r1/57-03-957-p5-perf-dsv41-tp3-decode-compressor-mla-and-spec-fusions.patch)) | not defensible: its replay predicts a TP3 loss at every concurrency ([result.json](../candidate-57/receipts/rework/pr-957-p5-r1/result.json)) |
| #984 | the whole PR ([candidate-53's `0784`](../candidate-53/vllm/0784-perf-worker-reclaim-uma-heap-and-allocator-slack-acr.patch); [round 1](../candidate-53/receipts/rework/pr-984-r1/result.json)) | dropped after its round 1: no startup-floor gain beyond the boot spread ([candidate-53](../candidate-53/README.md#984-round-1)) |
| #995 | its negative-slot guard in `copy_alloc` and `copy_apply` ([round-3 rework](../candidate-55/receipts/rework/pr-995-guard-r1/result.json)) | inert on this deployment: no serving path passes a negative slot to the kernels it guards ([candidate-55](../candidate-55/README.md)) |
| #988 | sparse top-k DSpark drafting | does nothing on this deployment: it runs only with `dspark_draft_topk`, which neither the recipe nor the checkpoint sets |
| #995 | the 31/63-draft ramp | does nothing on this deployment: the recipe keeps 16 drafts (17 rows) |
| #995 | the cap decay and `copy_match`'s anchor pruning | separate variables; queued as their own experiments |
| #1006 | `postload_cache.py` and its hooks; `deepseek_v41/nvidia/vl_model.py` | does nothing on this deployment: the cache needs a directory the recipe does not set, and GB10 routes to `deepseek_v4_1` |
| #957 | part 3 | superseded by the base's `0764` (#965 round 1) and `0769` (#980 round 1) |

The last five are the round-3 selection's ([selection-proposal.json](../candidate-55/receipts/selection-proposal.json),
`left_out`, and [candidate-55's selection](../candidate-55/receipts/selection.json) for the cap decay and pruning).
#978's test file `tests/multimodal/test_prefix_cache_fastpath.py` is not in the tree and not in
[release-tests.json](release-tests.json).

Carried from candidate-54's record:

| PR | Left out | Why |
| --- | --- | --- |
| #1013 | [`0766-feat-sched-size-classed-prefill-lanes-and-aligned-ch.patch`](../candidate-48/vllm/0766-feat-sched-size-classed-prefill-lanes-and-aligned-ch.patch) (candidate-48) | left out at selection: does nothing on this deployment (owner, 2026-11-09), as candidate-50 records it |
| #1011 | [`0770-fix-rocm-reserve-kpool-decode-indexer-logits-in-work.patch`](../candidate-48/vllm/0770-fix-rocm-reserve-kpool-decode-indexer-logits-in-work.patch) (candidate-48) | left out at selection: does nothing on this deployment (owner, 2026-11-09), as candidate-50 records it |
| #971 | the zero-confidence dead-row floor ([round 1](../candidate-53/receipts/rework/pr-971-r1/result.json)) | does nothing on this deployment: it acts only at `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU` <= 0, and the recipe sets 0.2 on every shape; dropped in round 1 |
| #987 | `sp_prefill.py` (`resolve_min_rows`) and `cudagraph_utils.py` (`_FULL_DEPTH_VARLEN_DECODE_MAX_REQS`), with their tests | does nothing on this deployment (candidates 51-53's `left_out`) |
| #987 | `tilelang/collectives.py` and `cuda_communicator.py` (`all_reduce_rank_order` through RoCE) | not inert: a second causal variable, split out to be its own experiment (candidates 51-53's `left_out`) |
| #988, #995, #1006, #983 | the opt-in parts named in candidates 51-53's `left_out` | does nothing on this deployment |
| #954, #1014, #981, #1004 | the whole PR | does nothing on this deployment |

**Dropped** after rework, with no defensible correction: #986, #1005 and #1010 (round 1); #984 (round 2); #978 after its
second rework, #957 part 1 and #957 part 5 (round 3, above).

### Stacking

All 35 patches after 1107's 641 apply cleanly in series order on a fresh tree, with no rebase. `git diff --check` is
clean ([receipts/stack.log](receipts/stack.log), section 1). The first 30 reproduce candidate-54's head `4e8133ff`
(tree `5f021d9a`). `bin/spark build prepare --lock` gives the same head and tree (section 4). The sparknet `0001`
applies on 1107's pin to candidate-55's head and tree.

The checks behind that (sections 2 and 3):
- **Each round-3 pass's files are its own.** Every file a round-3 patch touches is byte for byte the file of that
  pass's own verified stack on candidate-54:
  - candidate-55 without #978 (#1008, #988's dead-row part): tree `3f534a18`, the code candidate-55's gate, screen
    and #1008's mark measured. candidate-55's stack with #978 reproduces its lock (`410463b3`, tree `e39b835c`), and
    its `no978/overlay` files equal this stack's;
  - #956 r4 alone: tree `703f24a3`, the rework's;
  - candidate-57: its lock, tree `72b40d71`.
- **The five are disjoint.** No two touch a common file, test files included. Each applies alone on candidate-54,
  every ordered pair applies, and each pair's two orders give one tree.

So no pair of round-3 patches shares a non-test file, and each one's files against candidate-54's patches were
reviewed in its own candidate, on the same base and the same file contents (candidates
[55](../candidate-55/receipts/selection.json), [56](../candidate-56/receipts/rework/pr-956-r4/result.json) and
[57](../candidate-57/receipts/selection.json), `semantic_review`). Combinations no candidate measured:
- #1008's thread placement at `tp4-switched` and `tp3-switched`;
- #956 r4 with #957 part 4 and with #1008;
- every round-3 patch under TP3's L2 prefetch caps (below).

The interfaces they could meet through, in different files, were read at candidate-58's commits
([selection.json](receipts/selection.json), `semantic_review`):
- **#988 and #957 part 4:** #988's new sampler arguments (`dead_row_cut`, `cu_num_logits`, `copy_logits`'
  `live_idx_mapping`) are optional and passed only by its own rejection-sampler hunk. Its dead-row mask runs only on
  real verification steps with non-temperature sampling. #957 part 4 changes dummy runs and draft confidences, and
  calls neither.
- **#956 r4 and #957 part 4:** #956's CED encoder-only flag is set from a step's real requests and cleared on capture,
  and the model ignores it while a graph is captured. An encoder-only chunk is a non-final prefill chunk, which drafts
  nothing.
- **#957 part 2** changes only the body of decode sparse MLA's stage 2, which #956 does not touch.
- **#1008** changes host thread placement only.

No semantic conflict was found.

## Recipe change: TP3's L2 prefetch caps

Its own variable, the only change to the recipe:
- [tp3.json](tp3.json) (`tp3-triangle`), [tp3-triangle-4k.json](tp3-triangle-4k.json) and
  [tp3-switched.json](tp3-switched.json) set `VLLM_DS41_L2_PREFETCH_WO_MB=15`, `VLLM_DS41_L2_PREFETCH_FFN_MB=12` and
  `VLLM_DS41_L2_PREFETCH_NEXT_MB=12` (the defaults are 20/16/20 MB). Their `-dev` files extend them, so they run with
  the caps too, and add only `VLLM_SERVER_DEV_MODE` ([Arms](../../../docs/release-benchmark.md#arms)).
- TP4 ([tp4.json](tp4.json), [tp4-switched.json](tp4-switched.json)) and TP5..TP8 are unchanged, so #961's TP4 gain
  stays.
- **Evidence:** candidate-54's cap sweep ([tp3-grid-attribution](../candidate-54/receipts/tp3-grid-attribution/README.md))
  on candidate-54's image at `tp3-switched`. #961 causes candidate-54's TP3 small-batch grid loss (2.87% slower than
  1107, 1-request shapes 5.5-7.2%). Of the caps measured, 15/12/12 MB is the best: as fast as candidate-54 without #961
  at 1 request (1x2 rows at 4K: 18.49 against 18.70 ms), and within about 0.5% of it in the grid geometric mean, the
  gap mostly at 4 requests.
- **Measured on `tp3-switched`; applied to the TP3 triangle** (`tp3-triangle`, `tp3-triangle-4k`) by the same TP
  size, unmeasured there.
- **Output:** the caps change only how much of each next weight the side stream prefetches into L2, so they declare
  no output change. The TP3 gate checks that.
- **Confirmation here:** the grid-only job of [screen-tp3-switched.json](screen-tp3-switched.json) (Verification).

## Cluster configurations

A cluster configuration for every shape the recipe serves, so the candidate can be run on any setup: `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-58/<file> …` with your site's node map. Each one extends `releases/spark-ds41f-1107/cluster.json` at its shape and sets only the image `spark-ds41f:candidate-58`, its tree labels and this lock, and at TP3 the L2 prefetch caps (above). The `-dev` file adds the dev-mode collective RPC that the screen's engine grid needs. This candidate is verified at the three testbeds marked; the other files are configurations, not part of its runs.

| Shape | Configuration | With dev RPC | |
| --- | --- | --- | --- |
| `tp4-ring` | [tp4.json](tp4.json) | [tp4-dev.json](tp4-dev.json) | verified here |
| `tp3-triangle` | [tp3.json](tp3.json) | [tp3-dev.json](tp3-dev.json) | TP3 caps |
| `tp3-triangle-4k` | [tp3-triangle-4k.json](tp3-triangle-4k.json) | [tp3-triangle-4k-dev.json](tp3-triangle-4k-dev.json) | TP3 caps |
| `tp3-switched` | [tp3-switched.json](tp3-switched.json) | [tp3-switched-dev.json](tp3-switched-dev.json) | verified here; TP3 caps |
| `tp4-switched` | [tp4-switched.json](tp4-switched.json) | [tp4-switched-dev.json](tp4-switched-dev.json) | verified here |
| `tp5-switched` | [tp5-switched.json](tp5-switched.json) | [tp5-switched-dev.json](tp5-switched-dev.json) |  |
| `tp6-switched` | [tp6-switched.json](tp6-switched.json) | [tp6-switched-dev.json](tp6-switched-dev.json) |  |
| `tp7-switched` | [tp7-switched.json](tp7-switched.json) | [tp7-switched-dev.json](tp7-switched-dev.json) |  |
| `tp8-switched` | [tp8-switched.json](tp8-switched.json) | [tp8-switched-dev.json](tp8-switched-dev.json) |  |

- [image-dgx4.json](image-dgx4.json): build-only profile.

## Release tests

[release-tests.json](release-tests.json) is candidate-54's classification plus #1008's new test file,
`tests/v1/worker/test_roce_cpu_affinity.py`, gated with candidate-55's reason. The other round-3 patches change only
test files it already gates: `tests/v1/worker/test_dead_rows.py` (#988), `tests/models/test_deepseek_v4_1_compressor.py`,
`_compressor_prepare.py` and `_ced_model.py` (#956 r4), and `tests/kernels/test_deepseek_v41_tilelang_confidence.py`
(#957 part 4).

`scripts/release_tests.py check` reports no problems on dgx4's prepared tree: 216 selected files, 154 gate and 62
excluded ([receipts/release-tests-check.txt](receipts/release-tests-check.txt)). The gate manifest is
[release-tests-gate-manifest.json](release-tests-gate-manifest.json): 155 files, candidate-54's 154 with the same
deselections, plus #1008's. The bundle, written with main's tooling, is [bundles/release-tests](bundles/release-tests).
It carries the DeepSeek test plugin (`deepseek_models.py`).

## Determinism gate: declared output changes

Temperature-0 output hashes must equal candidate-54's: TP4 `18d61961`/`d6568649` with logprobs (text
`0a2dacdd`/`12fd8257`) at `tp4-ring` and `tp4-switched`, and TP3 `5c22ce41`/`bce6092e` (text `26a739a9`/`12cfee9b`)
at `tp3-switched`. No round-3 pass declares a greedy change, and neither do the TP3 caps, so the gates are identity
checks.

| PR | Declared output change |
| --- | --- |
| #955 | parser output only: quoted DSML stays text |
| #980 | sampled (temperature > 0) realizations |
| #1013 | grammar-constrained completions only |
| #972 | SSE comment frames (keep-alive) |
| #1030 | temperature-0 text changes on every shape (against the 1017 checkpoint's hashes): routed MoE activations are rounded to BF16 before MXFP8 quantization; candidate-54's hashes carry it |
| #957 (part 4) | draft confidences within FP32 rounding of the two-step path, so verification rows may differ; temperature-0 text must not change; sampled realizations may change |

#971 and #977 change drafting, as candidate-54 declares: temperature-0 text must not change, and sampled
realizations may. #1008, #988's dead-row part (live verification rows bit-exact), #956 r4 (bit-exact), #957 part 2 and
the TP3 caps declare none. So the screens' greedy identity against candidate-54's references must hold.

## Verification

Per step 9, "Between releases": the gate, the screen on each testbed and the live checks of the carried fixes. No
release set, quality evals, AgentX or cold boots; those belong to release qualification.

1. [build.json](build.json) (`candidate-58-image`), in a dgx window
   ([step 4](../../../docs/release-benchmark.md#4-build-and-release-tests-dgx-window)). It builds the image on dgx4
   and copies it along the ring, then runs one `kernel` job with the release-test bundle on dgx4. The image then
   crosses to sp1 once and fans out over the fabric, to sp5 for `tp3-switched`. On each head it is distributed with
   `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-58/<shape>.json build distribute --apply`.
2. On each testbed, the gate first, then the screen:

| Testbed | Gate | Screen: two candidate-58 boots | Live checks |
| --- | --- | --- | --- |
| `tp4-ring` | [candidate-tp4.json](candidate-tp4.json) | [screen-tp4.json](screen-tp4.json) | [live-checks-tp4.json](live-checks-tp4.json) |
| `tp4-switched` | [candidate-tp4-switched.json](candidate-tp4-switched.json) | [screen-tp4-switched.json](screen-tp4-switched.json) | |
| `tp3-switched` | [candidate-tp3-switched.json](candidate-tp3-switched.json) | [screen-tp3-switched.json](screen-tp3-switched.json), with `single-stream`, `sampled` and the grid-only job | |

- **Gates:** candidate-54's gates at each testbed (the same extras), on this candidate's arm: identity checks against
  candidate-54's hashes (above).
- **References:** each screen's is candidate-54's recorded screen at that testbed, in
  [candidate-54/receipts/references](../candidate-54/receipts/references) (`candidate-54-<shape>-screen`), the
  accumulated base's.
- **`tp3-switched`** runs `screen-tp3`, `single-stream` and `sampled` in one job on two boots, as candidate-57's screen
  does. The lab reads the companions beside the reference, at `candidate-54-tp3-switched-screen-single-stream` and
  `-sampled`. `sampled` runs because #957 part 4 changes draft confidences.
- **The grid-only TP3 job** confirms the caps: [grid-tp3-suite.json](../candidate-54/grid-tp3-suite.json)
  (`screen-tp3`'s engine grid step alone), ABBA in one session: candidate-54's
  [tp3-switched-dev.json](../candidate-54/tp3-switched-dev.json) (`candidate-54-grid`), candidate-58 twice
  (`candidate-58-grid`, `candidate-58-again-grid`), then candidate-54 again (`candidate-54-again-grid`).
  - Declared before measuring: the caps are confirmed when candidate-58's 1-request cells (1x2 and 1x3 rows at 4K) are
    faster than candidate-54's beyond the interval, and no cell is slower beyond it.
  - It measures the caps with round 3. Round 3's own TP3 grid effect is candidate-57's: the grid the same, 4-request
    shapes 1.7-1.9% faster ([candidate-57](../candidate-57/README.md#status)).
  - Its runs keep their own directories beside the screen's. Its table replaces the screen's in the run's one table
    file (`results/private/lab/candidate-58-screen-tp3-switched-table.txt`), as candidate-54's jobs did. Each table is
    also printed to the window log, and the dry run lists each job's `scripts/servebench_suite_table.py` command, which
    rebuilds it from the runs.
- **Live checks** on [tp4.json](tp4.json):
  - candidate-54's three, candidate-50's scripts by repository path:
    [check_955_quoted_dsml.py](../candidate-50/check_955_quoted_dsml.py),
    [check_1013_completions_grammar.py](../candidate-50/check_1013_completions_grammar.py) and
    [check_972_stream.py](../candidate-50/check_972_stream.py);
  - #1008's [check_1008_affinity.py](../candidate-55/check_1008_affinity.py): 16 streams for 120 s against rank 0's API
    server, and snapshots of every rank's serving threads meanwhile. #1008's partition should show on every rank
    (`partitioned`).
- **The screens are the next round's references:** later PRs are judged against this accumulated base.

## Frozen candidate

Per [Frozen candidates](../../../docs/release-benchmark.md#frozen-candidates-pre-releases): once its rework cycle is
done and it has passed verification on all three testbeds (the gate, the screen, the live checks of the fixes it
carries, and its release tests), candidate-58 is the first frozen candidate.
- Its folder is then frozen: its series, lock, profiles and verification receipts never change again. Later findings
  about it go into the next candidate's README.
- The coordinator prepares the freeze manifest (`manifests/frozen/candidate-58.json`) and the pre-release notes, and
  asks the owner before creating the tag `spark-ds41f-candidate-58` and the GitHub pre-release. Its base is
  spark-ds41f-1107, through candidate-54.
- Freezing is not promotion: `config/cluster.json` and the release's baseline do not change.
- The next round selects its PRs against it, builds its three candidates on its series, and judges them against its
  recorded screens.

## Status

Prepared and locked; not built, not measured.

**Added during verification (2026-11-10):** [single-stream-tp4.json](single-stream-tp4.json), c1 at `tp4-ring` in one
session against candidate-54 (candidate-58, candidate-54, candidate-58 again), as candidate-55's was. c1 decode on
`tp4-ring` is one of the release targets, and the merged-base screen does not measure it.

**c1 attribution (2026-11-10).** The same-session `single-stream` run at `tp4-ring` finds candidate-58's c1 paired
decode 0.35% below candidate-54's (81.93/81.90 against 82.24 tok/s; margin 0.31%). #1008 and #988's dead-row part
were the same as candidate-54 at c1 (candidate-55 without #978), so [attribute-c1-tp4.json](attribute-c1-tp4.json)
runs c1 in one session on candidate-58, candidate-58 without each of #957 part 4, #957 part 2 and #956 r4 (their
files at their pre-patch versions, `no957p4/`, `no957p2/`, `no956/`; the three share no file), candidate-54, and
candidate-58 again.

**Determinism investigation (2026-11-10).** [receipts/determinism-investigation](receipts/determinism-investigation/result.json):
no kernel is convicted. #961, #963 (both versions), #987 and #1001 (both versions), the WO-B and lm_head paths, sparse
MLA, the indexer and the rest of a TP4 decode layer give a fixed row the same bits at every M from 1 to 128 and with any
neighbours, on one GPU. The `again` failures are tp4-ring only (2 of 7 boots in the 1107 era; tp4-switched 0 of 8,
three with candidate-51's patches; tp3 0 of 7), and the online acceptance estimator's state across requests changes
`again`'s rows and drafts, which alone changes no row's numbers. Not covered on one GPU: the tp4-ring collectives.
[gate-repeat-tp4.json](gate-repeat-tp4.json) runs the gate twice on each of 12 boots in one session, 4 each of
candidate-58, spark-ds41f-1107 and candidate-58 without candidate-51's patches (`noC51/`), with a wrapper
(`token_determinism_record.py`) that keeps both outputs of any differing request.
