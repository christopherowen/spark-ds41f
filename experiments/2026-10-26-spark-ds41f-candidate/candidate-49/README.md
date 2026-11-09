# candidate-49

Unpromoted release candidate: spark-ds41f-1107 plus 6 PRs, prefill and long-context kernels that leave output bit-identical. It is tested at
`tp3-switched` (sp5..sp7).

## Source

- vLLM: 1107's series (`0001..0761`, 641 active) plus the patches below, renumbered from `0762` in
  application order ([vllm/series](vllm/series)). Each patch is copied from its PR's head. The
  provenance and SHA-256 of each are in [receipts/selection.json](receipts/selection.json), and the
  stack log is [receipts/stack.log](receipts/stack.log).
- `patch_count` 647, `patch_head` `d0ee539e`, `expected_tree` `2eaad40f`,
  `patchset_sha256` `ee85de7e`, recorded by `bin/spark build prepare --lock`
  ([source.json](source.json), [upstreams.lock.json](upstreams.lock.json)).
- Every other source is 1107's.

| Patch | PR | Title | PR head |
| --- | --- | --- | --- |
| `0762-perf-dsv41-widen-long-context-indexer-top-k-segments.patch` | #999 | perf(dsv41): widen long-context indexer top-k segments and reuse source blocks | `5f50df40e090` |
| `0763-perf-dsv41-vectorize-indexer-KPool-loads-and-co-sche.patch` | #993 | perf(dsv41): vectorize indexer_score_v41 KPool loads and co-schedule dense CTA columns | `a3067d857fda` |
| `0764-perf-dsv41-hw-e2m1x2-indexer-and-gemm-mxfp4-quantize.patch` | #991 | perf(dsv41): use hardware cvt.rn.satfinite.e2m1x2.f32 in indexer and GEMM MXFP4 quantizers | `c2953fb0dd49` |
| `0765-perf-dsv41-sp-prefill-local-attention-front.patch` | #1009 | perf(dsv41): fuse SP prefill local attention front and keep index_weights local | `78e6f933d270` |
| `0766-perf-dsv41-sp-prefill-sparknet-routing.patch` | #979 | perf(dsv41): route SP prefill gather_many and row exchange through sparknet | `c1d1bf3b76bc` |
| `0767-perf-dsv41-cache-SP-local-routing-masks-and-skip-SP-.patch` | #996 | perf(sp-prefill): cache SP local routing masks and skip SP tail pad rows | `86d4a427f46c` |

## Cluster configurations

A cluster configuration for every shape the recipe serves, so the candidate can be run on any setup: `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-49/<file> …` with your site's node map. Each one extends `releases/spark-ds41f-1107/cluster.json` at its shape and sets only the image `spark-ds41f:candidate-49`, its tree labels and this lock; no PR changes the recipe. The `-dev` file adds the dev-mode collective RPC that the release suite's engine grid needs. This candidate is measured at `tp3-switched` only; the other files are configurations, not part of its runs.

| Shape | Configuration | With dev RPC | |
| --- | --- | --- | --- |
| `tp4-ring` | [tp4.json](tp4.json) | [tp4-dev.json](tp4-dev.json) |  |
| `tp3-triangle` | [tp3.json](tp3.json) | [tp3-dev.json](tp3-dev.json) |  |
| `tp3-triangle-4k` | [tp3-triangle-4k.json](tp3-triangle-4k.json) | [tp3-triangle-4k-dev.json](tp3-triangle-4k-dev.json) |  |
| `tp3-switched` | [tp3-switched.json](tp3-switched.json) | [tp3-switched-dev.json](tp3-switched-dev.json) | tested here |
| `tp4-switched` | [tp4-switched.json](tp4-switched.json) | [tp4-switched-dev.json](tp4-switched-dev.json) |  |
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

1. [build.json](build.json) (`candidate-49-image`), in the dgx window: build the image on dgx4 (owner, 2026-11-09: builds on dgx, not the laptop), copy it along the ring, and run the release tests ([bundles/release-tests](bundles/release-tests)) on dgx4. The image is then copied to sp5..sp7, the same image ID on every rank.
2. [candidate-tp3-switched.json](candidate-tp3-switched.json) (`candidate-49-tp3-switched`): the determinism gate at `tp3-switched`.
3. [release-tp3-switched.json](release-tp3-switched.json) (`candidate-49-release-tp3-switched`): `candidate-49`, `spark-ds41f-1107`, `candidate-49-again`.

Suites: `release-tp3`, `single-stream`, `sampled`. Reference: 1107 boots as the middle arm, as for candidate 48.

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
| #999 | perf | Its indexer top-k bench on the image: gain beyond the interval, bit-exact; long prefill not worse | none |
| #993 | perf | Its indexer score bench: gain beyond the interval; bit-exactness shown on the image (the PR does not state it) | none |
| #991 | perf | Its quantizer bench: gain beyond the interval, bit-exact | none |
| #1009 | perf | Its SP attention-front bench: gain beyond the interval, bit-exact; prefill not worse | none |
| #979 | perf | Its SP gather bench at T>=6144: gain beyond the interval, bit-exact | none |
| #996 | perf | Its route_local bench: gain beyond the interval, bit-exact | none |

**Rework (owner, 2026-11-09).** A PR that misses its mark goes to a sub-agent for one focused
rework, with the miss, the receipts and the mark. A second rework is allowed when the first
shows promise. A PR that still misses is dropped from this release, and its record stays here.
A reworked PR joins the next candidate with its own source identity and fresh qualification.

## Selection

One of candidates 47, 48 and 49, tested at the same time. The selection receipts are in [candidate-47/receipts](../candidate-47/receipts).

## Status

Prepared and locked. Not built, not measured.
