# candidate-47

Unpromoted release candidate: spark-ds41f-1107 plus 8 PRs, #986 (the owner's priority) and decode kernels that leave temperature-0 output bit-identical. It is tested at
`tp4-ring` (dgx1..dgx4).

## Source

- vLLM: 1107's series (`0001..0761`, 641 active) plus the patches below, renumbered from `0762` in
  application order ([vllm/series](vllm/series)). Each patch is copied from its PR's head. The
  provenance and SHA-256 of each are in [receipts/selection.json](receipts/selection.json), and the
  stack log is [receipts/stack.log](receipts/stack.log).
- `patch_count` 649, `patch_head` `fd03994c`, `expected_tree` `eb6f1070`,
  `patchset_sha256` `d5160a07`, recorded by `bin/spark build prepare --lock`
  ([source.json](source.json), [upstreams.lock.json](upstreams.lock.json)).
- Every other source is 1107's.

| Patch | PR | Title | PR head |
| --- | --- | --- | --- |
| `0762-perf-spec-decode-adaptive-verify-graph-context-cost.patch` | #986 | perf(spec-decode): price MMA tile knees, 64-row wave saturation, and context bands in _step_cost | `c00819eeb06b` |
| `0763-perf-dsv41-evict-first-l2-policy-on-routed-moe-tma-weights.patch` | #966 | perf(dsv41): default unswizzled routed MoE TMA weight loads to L2 evict_first | `06695dd7b829` |
| `0764-perf-dsv41-certified-two-pass-lm-head-vocab-screen.patch` | #997 | perf(dsv41): certified two-pass Cauchy-Schwarz lm_head greedy vocabulary screening | `32151eb3cfed` |
| `0765-perf-dsv41-skip-empty-moe-swiglu-slabs-and-restore-p.patch` | #1005 | perf(dsv41): skip empty moe_swiglu_cast slabs and restore post-gate shared_experts hook | `9fc2775b2577` |
| `0766-perf-worker-eliminate-decode-step-h2d-and-aten-overhead.patch` | #965 | perf(worker): eliminate decode-step redundant H2D transfers, UVA copies, and eager ATen ops | `52c00805b36e` |
| `0767-perf-dsv41-overlap-decode-kv-index-writes-with-query.patch` | #1012 | perf(dsv41): overlap decode KV/indexer-key writes with query/index-query on aux_stream | `9cae52eb684a` |
| `0768-perf-dsv41-align-qb-and-indexer-qb-tile-contiguous-decode.patch` | #1016 | perf(dsv41): align Q-B and indexer Q-B decode tiles with 128x128 tile-contiguous weights | `a272efff662e` |
| `0769-perf-dsv41-sparse-mla-stage1-kv-gather-and-barrier-opt.patch` | #964 | perf(dsv41): sparse MLA stage-1 KV gather and staging barrier opt | `d6d69b92afb9` |

## Cluster configurations

A cluster configuration for every shape the recipe serves, so the candidate can be run on any setup: `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-47/<file> …` with your site's node map. Each one extends `releases/spark-ds41f-1107/cluster.json` at its shape and sets only the image `spark-ds41f:candidate-47`, its tree labels and this lock; no PR changes the recipe. The `-dev` file adds the dev-mode collective RPC that the release suite's engine grid needs. This candidate is measured at `tp4-ring` only; the other files are configurations, not part of its runs.

| Shape | Configuration | With dev RPC | |
| --- | --- | --- | --- |
| `tp4-ring` | [tp4.json](tp4.json) | [tp4-dev.json](tp4-dev.json) | tested here |
| `tp3-triangle` | [tp3.json](tp3.json) | [tp3-dev.json](tp3-dev.json) |  |
| `tp3-triangle-4k` | [tp3-triangle-4k.json](tp3-triangle-4k.json) | [tp3-triangle-4k-dev.json](tp3-triangle-4k-dev.json) |  |
| `tp3-switched` | [tp3-switched.json](tp3-switched.json) | [tp3-switched-dev.json](tp3-switched-dev.json) |  |
| `tp4-switched` | [tp4-switched.json](tp4-switched.json) | [tp4-switched-dev.json](tp4-switched-dev.json) |  |
| `tp5-switched` | [tp5-switched.json](tp5-switched.json) | [tp5-switched-dev.json](tp5-switched-dev.json) |  |
| `tp6-switched` | [tp6-switched.json](tp6-switched.json) | [tp6-switched-dev.json](tp6-switched-dev.json) |  |
| `tp7-switched` | [tp7-switched.json](tp7-switched.json) | [tp7-switched-dev.json](tp7-switched-dev.json) |  |
| `tp8-switched` | [tp8-switched.json](tp8-switched.json) | [tp8-switched-dev.json](tp8-switched-dev.json) |  |

- [image-dgx4.json](image-dgx4.json): build-only profile.
- [tp4-no986.json](tp4-no986.json), [tp4-no986-dev.json](tp4-no986-dev.json): candidate-47 without #986. They use the same image, with 1107's `adaptive_verification.py` mounted from [no986/overlay](no986/overlay). #986 is pure Python and no later patch here touches that file, so the arm is candidate-47 minus exactly #986 (SHA-256 in [receipts/selection.json](receipts/selection.json)).

## Release tests

[release-tests.json](release-tests.json) is 1107's classification, plus the test files these PRs add or
change, plus 1107's unclassified `tests/v1/core/test_prefix_group_validity.py`, each gated with its reason.
`scripts/release_tests.py check` reports no problems on dgx4's prepared tree
([receipts/release-tests-check.txt](receipts/release-tests-check.txt) and
[candidate-47/receipts/dgx4-logs](../candidate-47/receipts/dgx4-logs)). The gate manifest is
[release-tests-gate-manifest.json](release-tests-gate-manifest.json).

## Runs

Each testbed opens its window with `scripts/lab.py`, after the hold and idle guards.

1. [candidate-tp4.json](candidate-tp4.json) (`candidate-47-tp4`): sync; build the image on dgx4 and copy it along the ring; the release tests ([bundles/release-tests](bundles/release-tests)) on dgx4; the determinism gate.
2. [release-tp4.json](release-tp4.json) (`candidate-47-release-tp4`): `candidate-47` and `candidate-47-again` against 1107's recorded runs.
3. [release-tp4-no986.json](release-tp4-no986.json) (`candidate-47-no986-release-tp4`): `candidate-47-no986` and `candidate-47-no986-again`, the same against 1107. Compared with run 2 it shows #986's effect.

These run in one dgx window. Production (1107 at `tp4-ring`) is restored when the window closes.

Suites: `release`, `single-stream`, `sampled`. Reference: 1107's recorded runs (`manifests/suites/2026-11-07-spark-ds41f-1107-tp4` and its companions), `"reference": "release"`.

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
| #986 | perf (policy) | Paired decode at c8 or c16 better than `candidate-47-no986` (same image, 1107's `adaptive_verification.py` mounted), and c1 not worse | drafting and verification rows; temperature-0 text must not change |
| #966 | perf | Its FFN-layer bench (`bench_moe_l2_evict.py`) on the candidate image, 2 boots: gain at M=1 beyond the interval, `torch.equal` | none |
| #997 | perf | Its lm_head pair bench on the candidate image: gain at M=1..48, 0 mismatches; KV capacity falls by no more than its +158..211 MiB per rank | none |
| #1005 | perf | Its FFN sublayer bench against the 1107 image: gain beyond the interval, bit-exact | none |
| #965 | perf | Its worker-step harness on the candidate image: step time lower beyond the interval, bit-exact | none |
| #1012 | perf | Its indexed-layer replay bench: gain beyond the interval, bit-exact | none |
| #1016 | perf | Its chained Q-B bench: gain beyond the interval, bit-exact | none |
| #964 | perf | Its sparse-MLA stage-1 bench: gain beyond the interval, bit-exact | none |

**Rework (owner, 2026-11-09).** A PR that misses its mark goes to a sub-agent for one focused
rework, with the miss, the receipts and the mark. A second rework is allowed when the first
shows promise. A PR that still misses is dropped from this release, and its record stays here.
A reworked PR joins the next candidate with its own source identity and fresh qualification.

## Selection

Candidates 47, 48 and 49 are tested at the same time, one per testbed. The PRs that earn their keep
are merged into the next candidate, which is qualified on all shapes. The selection receipts are here:

- [pr-triage-2026-11-09.md](receipts/pr-triage-2026-11-09.md): every open PR on 2026-11-09.
- [pairwise-conflicts.json](receipts/pairwise-conflicts.json): of the 143 pairs that touch a common file, 49 conflict.
- [pr-patch-targets.json](receipts/pr-patch-targets.json): the files each PR's patches touch.
- [scripts-as-run](receipts/scripts-as-run).

Each candidate is conflict-free within itself. Across the three, the only overlaps are test-file
unions: #980 with #986, #1009 with #1012, and #964 with #993.

## Status

Prepared and locked. Not built, not measured.
