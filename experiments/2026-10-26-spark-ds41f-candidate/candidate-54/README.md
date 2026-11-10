# candidate-54

The merged base of round 2 (candidates 51-53): candidate-50 with #1009 round 1, plus every PR that passed PR
validation in candidates 51-53, each reworked PR as its corrected patch. 29 PRs, 30 patches. It is not a release
candidate ([release-benchmark](../../../docs/release-benchmark.md#9-release-qualification-final-benchmarking), step 9,
"Between releases"). So it is verified, not release-qualified, and the next round's candidates build on it.

Verification: the determinism gate, the screen and the carried fixes' live checks, on the three testbeds (`tp4-ring`
dgx1..dgx4, `tp4-switched` sp1..sp4, `tp3-switched` sp5..sp7).

## Source

- vLLM ([vllm/series](vllm/series)):
  - 1107's series (`0001..0761`, 641 active);
  - round 1: candidate-50's `0762..0780`, listed by relative path to its files, in its order, with #1009 round 1
    ([candidate-50/receipts/rework/pr-1009-r1](../candidate-50/receipts/rework/pr-1009-r1)) in place of
    candidate-50's `0778`;
  - round 2: 11 patches copied from `main` and renumbered from `0781` in this order: candidate-51's, candidate-52's,
    then candidate-53's, ending with #998 round 1. Each is the PR's patch as its candidate carried it, or the
    rework's corrected patch. #984 missed and is left out (below).
- The provenance, SHA-256 and author of each patch are in [receipts/selection.json](receipts/selection.json), and the
  stack log is [receipts/stack.log](receipts/stack.log).
- `patch_count` 671, `patch_head` `4e8133ff`, `expected_tree` `5f021d9a`, `patchset_sha256` `fd767123`, recorded by
  `bin/spark build prepare --lock` ([source.json](source.json), [upstreams.lock.json](upstreams.lock.json)).
- Every other source is 1107's.
- **Authors:** every carried PR is christopherowen's. Every patch's `From:` is Christopher Owen's, except
  `0768` (#955), which is koldfrontier's: an owner-approved exception.

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
| `0781-perf-dsv41-64k-chunked-globaltimer-paced-l2-prefetch.patch` | #961 | perf(dsv41): 64 KiB chunked, %globaltimer-paced side-stream L2 prefetch | candidate-51, PR head `9362f944201a` | [pass](../candidate-51/receipts/own-marks/pr-961/result.json) |
| `0782-perf-dsv41-unconditional-pack12-plane-loads-and-swar.patch` | #963 | perf(dsv41): unconditional pack12 plane loads, byte-perm vocab SWAR, and per-column exception lookup | [#963 round 1](../candidate-51/receipts/rework/pr-963-r1) (candidate-51) | [corrected-meets-mark](../candidate-51/receipts/rework/pr-963-r1/result.json) |
| `0783-perf-dsv41-extend-WO-B-WIDE_DECODE-to-128-rows-and-g.patch` | #987 | perf(dsv41): extend WO-B WIDE_DECODE to 128 rows and guard high-concurrency decode cliffs | candidate-51 (its WO-B part), PR head `9be94ed5310c` | [pass](../candidate-51/receipts/own-marks/pr-987/result.json) |
| `0784-Match-128-row-vocab-head-tile-pipeline-stages.patch` | #1001 | perf(dsv41): match 128-row vocab head tile pipeline stages on GB10 | [#1001 round 1](../candidate-51/receipts/rework/pr-1001-r1) (candidate-51) | [corrected-meets-mark](../candidate-51/receipts/rework/pr-1001-r1/result.json) |
| `0785-fix-dsv41-align-moe_swiglu_cast-with-DeepSeek-Expert.patch` | #1030 | fix(dsv41): align moe_swiglu_cast with DeepSeek Expert.forward BF16 rounding | candidate-52, PR head `a7edd33143aa` | [pass](../candidate-52/receipts/own-marks/pr-1030/result.json) |
| `0786-perf-spec-decode-remaining-token-draft-cap.patch` | #968 | perf(spec_decode): clamp scheduled decode verify rows to remaining max_tokens | candidate-52, PR head `1445e800b172` | [pass](../candidate-52/receipts/own-marks/pr-968/result.json) |
| `0787-perf-dsv41-direct-insert-packed-engram-cache-and-ali.patch` | #969 | perf(dsv41): direct-insert packed Engram cache rows and align split cache arrays | candidate-52, PR head `aad967cbe897` | [pass](../candidate-52/receipts/own-marks/pr-969/result.json) |
| `0788-perf-cudagraph-target-full-replay-and-post-gate-fork.patch` | #960 | Target FULL CUDA graph metadata/CED replay and MoE post-gate fork | [#960 round 1](../candidate-53/receipts/rework/pr-960-r1) (candidate-53) | [corrected-meets-mark](../candidate-53/receipts/rework/pr-960-r1/result.json) |
| `0789-perf-spec-decode-copy-drafter-survival-and-tile-cap.patch` | #971 | perf(spec-decode): cumulative survival in copy_apply, single-req 16-row tile cap, and wide-step dead-row mask | [#971 round 1](../candidate-53/receipts/rework/pr-971-r1) (candidate-53) | [corrected-meets-mark](../candidate-53/receipts/rework/pr-971-r1/result.json) |
| `0790-perf-spec-decode-copy-drafter-long-context-scan.patch` | #977 | perf(spec_decode): bound CopyDrafter long-context history scan in copy_match | candidate-53, PR head `4e0244deb7d5` | [pass](../candidate-53/receipts/own-marks/pr-977/result.json) |
| `0791-perf-worker-compact-packed-vocab-tables-and-expand-d.patch` | #998 | perf(worker): compact packed vocab tables and expand display carve-out weights | [#998 round 1](../candidate-53/receipts/rework/pr-998-r1) (candidate-53) | [corrected-meets-mark](../candidate-53/receipts/rework/pr-998-r1/result.json) |

**#1009 round 1 returns.** It fixes candidate-50's `tp4-ring` boot failure
([candidate-50/receipts/tp4-boot-failure](../candidate-50/receipts/tp4-boot-failure)). Its coordinator's verdict
([analysis.md](../candidate-50/receipts/rework/pr-1009-r1/analysis.md)) waited for a boot at TP4 sizing. On the
candidate-50 image with its overlay, it then passed its boot, gate and screen at `tp4-switched` (tp4-ring's
sizing) and at `tp3-switched`. candidate-54 is its first build, and this gate its first `tp4-ring` boot.

**#984 missed, and is in rework.** candidate-54 leaves out its patch
([candidate-53's `0784`](../candidate-53/vllm/0784-perf-worker-reclaim-uma-heap-and-allocator-slack-acr.patch)). It joins
a later base when its rework passes. Its attribution
([candidate-53/attribute-984-tp3-switched.json](../candidate-53/attribute-984-tp3-switched.json)) ran ABBA, with and
without #984, on the same image at `tp3-switched`:
- **Time to ready:** #984 adds about 77 s a boot: 203 and 199 s with it, against 121.5 and 126.9 s without.
- **Startup `MemAvailable` floor:** it improves only 0.03-0.12 GiB a rank. On the first boots that is
  5.83/7.65/7.57 against 5.80/7.60/7.45 GiB.
- **Prefill:** unchanged. **Stall-3:** worse with #984. Prefill under decode is 2.37% slower (margin 1.27%),
  and decode p90 during it is 0.77% higher. *(Corrected 2026-11-10: this line first said #984 did not move
  stall-3.)*

Round 1 ([candidate-53/receipts/rework/pr-984-r1](../candidate-53/receipts/rework/pr-984-r1/result.json)) keeps
one heap trim at the end of startup and drops the rest. Its TP3 check,
[candidate-53/attribute-984r1-tp3-switched.json](../candidate-53/attribute-984r1-tp3-switched.json), removes round 0's
startup and stall-3 costs but finds no floor gain beyond the boot spread. #984 is dropped from this release
([candidate-53](../candidate-53/README.md#984-round-1)).

Its own mark on sp8 was [cannot-evaluate](../candidate-53/receipts/own-marks/pr-984/result.json). #998 round 1
applies without #984 before it ([stack.log](receipts/stack.log), section 2), so round 2 ends with #998 round 1
(`0791`). The record is in [selection.json](receipts/selection.json), `base_left_out`. The first assembly, with #984
as `0792`, is kept in [stack.log](receipts/stack.log), section 6.

**Carried in part:**
- #1013: its grammar fix (`0770`) alone, as candidate-50 carries it;
- #1011: candidate-48's `0768` and `0769` alone (`0772` and `0773` here);
- #987: its WO-B part (`0783`), as candidate-51 carries it;
- #971: round 1, the `copy_apply` survival gate and the single-request tile cap, without the dead-row floor.

**Left out**, so the series has no gap where they were. Each reason and source path is in
[receipts/selection.json](receipts/selection.json) (`left_out`):

| PR | Left out | Why |
| --- | --- | --- |
| #1013 | [`0766-feat-sched-size-classed-prefill-lanes-and-aligned-ch.patch`](../candidate-48/vllm/0766-feat-sched-size-classed-prefill-lanes-and-aligned-ch.patch) (candidate-48) | left out at selection: does nothing on this deployment (owner, 2026-11-09), as candidate-50 records it |
| #1011 | [`0770-fix-rocm-reserve-kpool-decode-indexer-logits-in-work.patch`](../candidate-48/vllm/0770-fix-rocm-reserve-kpool-decode-indexer-logits-in-work.patch) (candidate-48) | left out at selection: does nothing on this deployment (owner, 2026-11-09), as candidate-50 records it |
| #971 | the zero-confidence dead-row floor: `_ZERO_CONFIDENCE_LOG_TAU`, `dead_row_args`' copy-tail branch, `declare_step_kernels`' copy-drafter flag, and their tests ([round 1](../candidate-53/receipts/rework/pr-971-r1/result.json)) | does nothing on this deployment: it acts only at `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU` <= 0, and the recipe sets 0.2 on every shape; dropped in round 1 |
| #987 | `sp_prefill.py` (`resolve_min_rows`) and `cudagraph_utils.py` (`_FULL_DEPTH_VARLEN_DECODE_MAX_REQS`), with their tests | does nothing on this deployment (candidates 51-53's `left_out`) |
| #987 | `tilelang/collectives.py` and `cuda_communicator.py` (`all_reduce_rank_order` through RoCE) | not inert: a second causal variable, split out to be its own experiment (candidates 51-53's `left_out`) |
| #988, #995, #1006, #983 | the opt-in parts named in candidates 51-53's `left_out` | does nothing on this deployment |
| #957 | part 3 | superseded by the base's #965 and #980 |
| #954, #1014, #981, #1004 | the whole PR | does nothing on this deployment |

The left-out lanes patch's test files (`tests/v1/core/test_prefill_lane_livelock.py`, `test_prefill_qos_lanes.py`,
`test_aligned_prefill_lanes.py`) are not in the tree and not in [release-tests.json](release-tests.json), as in
candidate-50.

**Dropped** after rework, with no defensible correction: #986, #1005 and #1010 (round 1; candidate-50's README).
Round 2 dropped none.

### Stacking

All 30 patches apply cleanly in series order on a fresh tree, with no rebase. `git diff --check` is clean
([receipts/stack.log](receipts/stack.log), section 1). `bin/spark build prepare --lock` gives the same head and
tree (section 4).

The checks behind that order (sections 2 and 3):
- **Each candidate's files are its own.** Every file a round-2 patch touches is byte for byte the file of that
  patch's own candidate stack on the round-2 base. That base is candidate-50 without #1009, tree `14c5a2dc`. The
  stacks are:
  - candidate-51 with #963 and #1001 round 1: tree `35cd2a0e`, the tree those reworks measured;
  - candidate-52: its lock, tree `1116740c`;
  - candidate-53 with its three reworks and without #984: tree `8b643f4b`.
- **The candidates are disjoint.** Candidates 51, 52 and 53 touch no common file. No round-2 patch touches #1009
  round 1's six files.
- **#998 round 1 does not need #984.** It applies without it.

So two combinations are new here, and PR validation never measured them:
- #1009 round 1 with round 2, which touch disjoint files;
- candidate-53's three reworks together, without #984. Each was verified beside the others' original patches and
  #984.

Shared non-test files: patches of two PRs share these, and apply cleanly there. Each pair was read at candidate-54's
commits ([selection.json](receipts/selection.json), `semantic_review`):
- `model_runner.py`: #965 round 1, #980 round 1, #960 round 1 and #998 round 1;
- `dflash/speculator.py`: #980 round 1 and #960 round 1;
- `nvidia/model_state.py`: #1013 and #960 round 1;
- `packed_bf16.py`: #997 and #998 round 1;
- `tilelang/moe.py`: #966 and #1030;
- `tilelang/gemm.py`: #1016 round 1, #991 round 1, #963 round 1 and #987;
- `tilelang/vocab.py`: #997, #963 round 1 and #1001 round 1;
- `spec_decode/tilelang_copy.py`: #971 round 1 and #977.

Each pair edits different functions, or different keys of one table. The exception is `packed_bf16.py`:
#998's first version compacted the carve-out buffer that also holds #997's screening table, which made candidate-53's
release tests crash. Round 1's guard is that fix. No semantic conflict was found.

## Cluster configurations

A cluster configuration for every shape the recipe serves, so the candidate can be run on any setup: `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-54/<file> …` with your site's node map. Each one extends `releases/spark-ds41f-1107/cluster.json` at its shape and sets only the image `spark-ds41f:candidate-54`, its tree labels and this lock; no PR changes the recipe. The `-dev` file adds the dev-mode collective RPC that the screen's engine grid needs. This candidate is verified at the three testbeds marked; the other files are configurations, not part of its runs.

| Shape | Configuration | With dev RPC | |
| --- | --- | --- | --- |
| `tp4-ring` | [tp4.json](tp4.json) | [tp4-dev.json](tp4-dev.json) | verified here |
| `tp3-triangle` | [tp3.json](tp3.json) | [tp3-dev.json](tp3-dev.json) |  |
| `tp3-triangle-4k` | [tp3-triangle-4k.json](tp3-triangle-4k.json) | [tp3-triangle-4k-dev.json](tp3-triangle-4k-dev.json) |  |
| `tp3-switched` | [tp3-switched.json](tp3-switched.json) | [tp3-switched-dev.json](tp3-switched-dev.json) | verified here |
| `tp4-switched` | [tp4-switched.json](tp4-switched.json) | [tp4-switched-dev.json](tp4-switched-dev.json) | verified here |
| `tp5-switched` | [tp5-switched.json](tp5-switched.json) | [tp5-switched-dev.json](tp5-switched-dev.json) |  |
| `tp6-switched` | [tp6-switched.json](tp6-switched.json) | [tp6-switched-dev.json](tp6-switched-dev.json) |  |
| `tp7-switched` | [tp7-switched.json](tp7-switched.json) | [tp7-switched-dev.json](tp7-switched-dev.json) |  |
| `tp8-switched` | [tp8-switched.json](tp8-switched.json) | [tp8-switched-dev.json](tp8-switched-dev.json) |  |

- [image-dgx4.json](image-dgx4.json): build-only profile.

## Release tests

[release-tests.json](release-tests.json) is candidate-53's classification without #984's two `tests/utils_` entries
(`test_gc_utils.py` gated, `test_mem_utils.py` excluded). Only #984 changes those files, so without it the series
no longer selects them, and their reasons name #984. What remains is candidate-50's classification plus main's
#1038 change, the same as candidates 51 and 52 carry. #1009 round 1 and the round-2 patches change only test files it
already classifies, so no file needed a new classification.

`scripts/release_tests.py check` reports no problems on dgx4's prepared tree: 215 selected files, 153 gate and 62
excluded ([receipts/release-tests-check.txt](receipts/release-tests-check.txt)). The gate manifest is
[release-tests-gate-manifest.json](release-tests-gate-manifest.json) (154 files, the same files and deselections as
candidate-51's). The bundle, written with main's tooling, is [bundles/release-tests](bundles/release-tests). It
carries the DeepSeek test plugin (`deepseek_models.py`).

## Determinism gate: declared output changes

Temperature-0 output hashes must equal the 1017 checkpoint's, except where a PR declares a change:

| PR | Declared output change |
| --- | --- |
| #955 | parser output only: quoted DSML stays text |
| #980 | sampled (temperature > 0) realizations |
| #1013 | grammar-constrained completions only |
| #972 | SSE comment frames (keep-alive) |
| #1030 | temperature-0 text changes on every shape: routed MoE activations are rounded to BF16 before MXFP8 quantization, as in DeepSeek's reference |

#971 and #977 change drafting (#971: the copy continuation yield and the single-request 16-row cap; #977: copy
drafts past 32K), as candidate-53 declared. Temperature-0 text must not change, and sampled realizations may.
Every other PR declares none.

**#1030 changes the gate's hashes by construction.**
- Every other carried PR declares temperature-0 text unchanged. So candidate-54's TP4 hashes are expected to equal
  candidate-52's, 18d61961/d6568649: that is #1030 on the round-2 base, at tp4-switched
  ([candidate-52/receipts/tp4-switched](../candidate-52/receipts/tp4-switched/README.md)). `tp4-ring` and
  `tp4-switched` share TP4's hashes (candidate-50).
- At TP3 no hash with #1030 is recorded yet; this gate records the first.
- The screens' greedy identity against 1107's reference fails by construction, as candidate-52's did against
  candidate-50's. Speed is judged on distinct prompts: the engine grid, paired decode and prefill.

## Verification

Per step 9, "Between releases": the gate, the screen on each testbed and the live checks of the carried fixes. No
release set, quality evals, AgentX or cold boots; those belong to release qualification.

1. [build.json](build.json) (`candidate-54-image`), in a dgx window
   ([step 4](../../../docs/release-benchmark.md#4-build-and-release-tests-dgx-window)). It builds the image on dgx4
   and copies it along the ring, then runs one `kernel` job with the release-test bundle on dgx4. The image then
   crosses to sp1 once and fans out over the fabric, to sp5 for `tp3-switched`. On each head it is distributed with
   `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-54/<shape>.json build distribute --apply`.
2. On each testbed, the gate first, then the screen:

| Testbed | Gate | Screen: two candidate-54 boots | Live checks |
| --- | --- | --- | --- |
| `tp4-ring` | [candidate-tp4.json](candidate-tp4.json) | [screen-tp4.json](screen-tp4.json) | [live-checks-tp4.json](live-checks-tp4.json) |
| `tp4-switched` | [candidate-tp4-switched.json](candidate-tp4-switched.json) | [screen-tp4-switched.json](screen-tp4-switched.json) | [live-checks-tp4-switched.json](live-checks-tp4-switched.json) |
| `tp3-switched` | [candidate-tp3-switched.json](candidate-tp3-switched.json) | [screen-tp3-switched.json](screen-tp3-switched.json) | |

- **Gates:** candidate-50's gates at each testbed (the same extras), on this candidate's arm.
- **References:** each screen's is 1107's recorded screen at that shape, in
  [candidate-50/receipts/references](../candidate-50/receipts/references)
  (`spark-ds41f-1107-<shape>-screen`).
- **`tp3-switched` also runs the round's companions** against candidate-50's runs, as candidate-53's screen does:
  - `single-stream` and `sampled`, against `candidate-50-no1009-tp3-switched-screen-<suite>`;
  - the stall steps alone ([candidate-50/stall-tp3-suite.json](../candidate-50/stall-tp3-suite.json)), against
    candidate-50's run on sp5 (`results/private/suites/candidate-50-stall-tp3-switched/candidate-50-no1009`).
- **Each companion is its own job.** The lab gives a measure job one reference and reads its companions at
  `<reference>-<suite>`. 1107 has no `tp3-switched` companion records, so `screen-tp3`, `single-stream`, `sampled`
  and the stall steps run as four jobs of two boots each. Their labels carry the suite (`candidate-54-single-stream`,
  `candidate-54-again-single-stream`, ...), as candidate-53's stall job does.
  - The cost is four more boots than one job would take.
  - Each job's suite runs keep their own directories. Their tables share the run's one table file
    (`results/private/lab/candidate-54-screen-tp3-switched-table.txt`), as candidate-53's two jobs do, so each
    later table replaces the earlier one there. Each table is also printed to the window log, and the dry run
    lists each job's `scripts/servebench_suite_table.py` command, which rebuilds it from the runs.
- **The screens are the next round's references:** later PRs are judged against this accumulated base.
- **Live checks:** candidate-50's three, its scripts by repository path, on [tp4.json](tp4.json):
  - [check_955_quoted_dsml.py](../candidate-50/check_955_quoted_dsml.py): a tool call quoting DSML keeps its
    arguments, and quoted DSML in a reply stays text;
  - [check_1013_completions_grammar.py](../candidate-50/check_1013_completions_grammar.py): `/v1/completions` with a
    JSON-schema grammar under the `deepseek_v41` reasoning parser returns schema-valid output;
  - [check_972_stream.py](../candidate-50/check_972_stream.py): a client disconnect aborts the engine request,
    keep-alive frames arrive every 15 s, and servebench's stream parsing ignores them.

## Status

Built (`spark-ds41f:candidate-54`, image ID `e1de71be`); its release tests passed on dgx4 (10,171 passed, 311
skipped).

| Testbed | Gate | Screen against 1107 | Live checks |
| --- | --- | --- | --- |
| `tp4-ring` | identical outputs; #1030's TP4 hashes 18d61961/d6568649 | faster on every core metric: grid 4.78%, c16 decode 1.61%, prefill 3.5% at 32K and 7.8% at 256K | stopped before running ([receipts/tp4-ring](receipts/tp4-ring/README.md)); run at `tp4-switched` instead |
| `tp4-switched` | identical outputs; the same TP4 hashes | grid 4.29% faster, c16 output 1.6% faster, c16 decode the same, prefill the same at 32K and 7.5% faster at 256K ([receipts/tp4-switched](receipts/tp4-switched/README.md)) | #955, #1013 and #972 pass |
| `tp3-switched` | identical outputs; new TP3 hashes 5c22ce41/bce6092e (#1030's declared change) | **grid 2.87% slower**, at small batches (1-request shapes 5.5-7.2%); c8, `single-stream`, `sampled` and the stall steps the same; grid and c8 from one boot and prefill not measured (sp5's 4 GiB bench floor) ([receipts/tp3-switched](receipts/tp3-switched/README.md)) | |

The screen references the next round uses are in [receipts/references](receipts/references). Their greedy corpora
stay on the nodes, listed with their SHA-256 in `large-files-on-nodes.tsv`.
