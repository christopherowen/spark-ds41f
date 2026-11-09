# candidate-48

Unpromoted release candidate: spark-ds41f-1107 plus 7 PRs, correctness fixes, plus the multi-turn chat-render fast path. It is tested at
`tp4-switched` (sp1..sp4).

## Source

- vLLM: 1107's series (`0001..0761`, 641 active) plus the patches below, renumbered from `0762` in
  application order ([vllm/series](vllm/series)). Each patch is copied from its PR's head. The
  provenance and SHA-256 of each are in [receipts/selection.json](receipts/selection.json), and the
  stack log is [receipts/stack.log](receipts/stack.log).
- `patch_count` 651, `patch_head` `bd28dbf3`, `expected_tree` `fe5bddc4`,
  `patchset_sha256` `95a7270c`, recorded by `bin/spark build prepare --lock`
  ([source.json](source.json), [upstreams.lock.json](upstreams.lock.json)).
- Every other source is 1107's.

| Patch | PR | Title | PR head |
| --- | --- | --- | --- |
| `0762-deepseek-v41-dsml-token-provenance.patch` | #955 | Require the DSML token for DeepSeek V4.1 tool call markup (#949) | `3a854ba440e1` |
| `0763-fix-prefix-cache-retain-decode-frontier-swa-and-orde.patch` | #1010 | fix(core): retain decode-frontier SWA checkpoints and order hybrid KV eviction tail-to-root | `ad613458e559` |
| `0764-perf-spec-decode-elide-tp-confidence-broadcast-and-lock-sampling-seeds.patch` | #980 | perf(spec_decode): elide TP confidence broadcasts and lockstep TP sampling seeds | `11b169b0dd07` |
| `0765-fix-structured-output-enforce-completions-grammar-un.patch` | #1013 | fix(vllm): enforce completions structured-output grammar and add size-classed prefill lanes | `aeba4fc4ac40` |
| `0766-feat-sched-size-classed-prefill-lanes-and-aligned-ch.patch` | #1013 | fix(vllm): enforce completions structured-output grammar and add size-classed prefill lanes | `aeba4fc4ac40` |
| `0767-feat-serve-sse-keepalive-heartbeat-and-abort-cleanup.patch` | #972 | feat(serve): enable periodic SSE keep-alive heartbeats and shield streaming abort cleanup | `3827857dc3dd` |
| `0768-fix-entrypoints-move-model-inspection-RPC-out-of-LLM.patch` | #1011 | experiments(vllm): audit Nov 2026 mainline PRs and backport 0762..0764 | `8904a1fec170` |
| `0769-fix-attention-keep-FA-sliding-window-active-under-ba.patch` | #1011 | experiments(vllm): audit Nov 2026 mainline PRs and backport 0762..0764 | `8904a1fec170` |
| `0770-fix-rocm-reserve-kpool-decode-indexer-logits-in-work.patch` | #1011 | experiments(vllm): audit Nov 2026 mainline PRs and backport 0762..0764 | `8904a1fec170` |
| `0771-perf-dsv41-fast-path-multi-turn-chat-rendering-and-s.patch` | #973 | perf(dsv41): fast-path multi-turn chat rendering and segment tokenization | `fed91173461f` |

## Cluster configurations

A cluster configuration for every shape the recipe serves, so the candidate can be run on any setup: `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-48/<file> …` with your site's node map. Each one extends `releases/spark-ds41f-1107/cluster.json` at its shape and sets only the image `spark-ds41f:candidate-48`, its tree labels and this lock; no PR changes the recipe. The `-dev` file adds the dev-mode collective RPC that the release suite's engine grid needs. This candidate is measured at `tp4-switched` only; the other files are configurations, not part of its runs.

| Shape | Configuration | With dev RPC | |
| --- | --- | --- | --- |
| `tp4-ring` | [tp4.json](tp4.json) | [tp4-dev.json](tp4-dev.json) |  |
| `tp3-triangle` | [tp3.json](tp3.json) | [tp3-dev.json](tp3-dev.json) |  |
| `tp3-triangle-4k` | [tp3-triangle-4k.json](tp3-triangle-4k.json) | [tp3-triangle-4k-dev.json](tp3-triangle-4k-dev.json) |  |
| `tp3-switched` | [tp3-switched.json](tp3-switched.json) | [tp3-switched-dev.json](tp3-switched-dev.json) |  |
| `tp4-switched` | [tp4-switched.json](tp4-switched.json) | [tp4-switched-dev.json](tp4-switched-dev.json) | tested here |
| `tp5-switched` | [tp5-switched.json](tp5-switched.json) | [tp5-switched-dev.json](tp5-switched-dev.json) |  |
| `tp6-switched` | [tp6-switched.json](tp6-switched.json) | [tp6-switched-dev.json](tp6-switched-dev.json) |  |
| `tp7-switched` | [tp7-switched.json](tp7-switched.json) | [tp7-switched-dev.json](tp7-switched-dev.json) |  |
| `tp8-switched` | [tp8-switched.json](tp8-switched.json) | [tp8-switched-dev.json](tp8-switched-dev.json) |  |

- [image-dgx4.json](image-dgx4.json): build-only profile.

## Release tests

[release-tests.json](release-tests.json) is 1107's classification, plus the test files these PRs add or
change, plus 1107's unclassified `tests/v1/core/test_prefix_group_validity.py`, each gated with its reason.
`scripts/release_tests.py check` reports no problems on dgx4's prepared tree
([receipts/release-tests-check.txt](receipts/release-tests-check.txt) and
[candidate-47/receipts/dgx4-logs](../candidate-47/receipts/dgx4-logs)). The gate manifest is
[release-tests-gate-manifest.json](release-tests-gate-manifest.json).

## Runs

Each testbed opens its window with `scripts/lab.py`, after the hold and idle guards.

1. [build.json](build.json) (`candidate-48-image`), in the dgx window: build the image on dgx4 (owner, 2026-11-09: builds on dgx, not the laptop), copy it along the ring, and run the release tests ([bundles/release-tests](bundles/release-tests)) on dgx4. The image is then copied to sp1..sp4, the same image ID on every rank.
2. [candidate-tp4-switched.json](candidate-tp4-switched.json) (`candidate-48-tp4-switched`): the determinism gate at `tp4-switched`.
3. [release-tp4-switched.json](release-tp4-switched.json) (`candidate-48-release-tp4-switched`): `candidate-48`, `spark-ds41f-1107`, `candidate-48-again`.

Suites: `release`, `single-stream`, `sampled`. Reference: 1107 boots as the middle arm. Its switched record is a `bin/spark bench` report without request tables, so it cannot serve ([release-benchmark](../../../docs/release-benchmark.md#3-release-comparison)).

## Marks

Declared before building or measuring ([methodology](../../../docs/methodology.md#round-purpose-report-and-drop-rule)).
A PR earns its keep in one of two ways:

- **Correctness fix:** its functional evidence passes on the candidate image, and no core metric is worse.
- **Performance change:** it improves its own metric, and no unrelated core metric is worse.

"Worse" and "better" mean beyond the suite table's intervals: the candidate's two boots, and the
reference's repeat where recorded. The paired decode rows are judged against the recorded A/A floor.

**Core metrics:**
- paired decode on distinct prompts;
- the engine decode grid;
- prefill at every suite size;
- short-request TTFT during a long prefill;
- the `single-stream` and `sampled` companions;
- the determinism gate (temperature-0 output hashes equal to the 1017 checkpoint's, unless a PR declares a change below);
- startup (lowest `MemAvailable`, time to ready, no JIT after readiness) and steady memory (KV capacity, swap, OOMs);
- zero failed requests, and no silent fallback in any rank's boot log;
- the release tests (`check` reports no problems, and the bundle exits 0).

A kernel PR's own mark is measured on the candidate image, the integrated stack, by the PR's own
benchmark: two boots on sp8. When a core metric is worse, the PRs whose path can move it are
removed one at a time until the cause is found.

| PR | Kind | Own mark | Declared output change |
| --- | --- | --- | --- |
| #955 | fix | Its DSML fixture replay (36/36) on the image, and a live tool call quoting DSML returns complete arguments | parser output only (quoted DSML stays text) |
| #1010 | fix | Its retention probe passes on the image; the live prefix-cache hit share on multi-turn traffic is not below 1107's | none |
| #980 | fix | Its 1,000-step TP lockstep screen shows 0 rank mismatches on the image; the sampled suite completes | sampled (temperature > 0) realizations |
| #1013 | fix | A live `/v1/completions` request with a JSON grammar under the `deepseek_v41` reasoning parser returns valid output; 0763's defaults leave scheduling unchanged | grammar-constrained completions only |
| #972 | fix | A live client disconnect aborts the engine request (its 32/32 test); keep-alive frames every 15 s; servebench's streams parse | SSE comment frames |
| #1011 | fix | Its tests pass on the image | none |
| #973 | perf | Its render-and-tokenize bench on the image: gain beyond the interval, identical `tokens_sha256` | none |

**Rework (owner, 2026-11-09).** A PR that misses its mark goes to a sub-agent for one focused
rework, with the miss, the receipts and the mark. A second rework is allowed when the first
shows promise. A PR that still misses is dropped from this release, and its record stays here.
A reworked PR joins the next candidate with its own source identity and fresh qualification.

## Selection

One of candidates 47, 48 and 49, tested at the same time. The selection receipts are in [candidate-47/receipts](../candidate-47/receipts).

## Status

Prepared and locked. Not built, not measured.
