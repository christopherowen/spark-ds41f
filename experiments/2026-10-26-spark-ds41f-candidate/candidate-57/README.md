# candidate-57

Unpromoted release candidate: candidate-54 plus #957's TP3 parts 2 and 4 (the branchless sparse-MLA stage 2, measured
at TP3's 24 heads, and the fused DSpark confidence calibration with the dummy-run fix). It is tested at
`tp3-switched` (sp5..sp7, head sp5).

## Source

- vLLM: candidate-54's series by relative path to its files ([candidate-54/vllm/series](../candidate-54/vllm/series):
  1107's `0001..0761`, 641 active, candidate-50's `0762..0780` with #1009 round 1, and round 2's `0781..0791`; 671
  patches). Then the two patches below as `0797` and `0798` ([vllm/series](vllm/series)). `0795` and `0796` are
  candidate-56's (957-p1, then #956), so this series has no `0795..0796`. The provenance, SHA-256 and author of each
  patch are in [receipts/selection.json](receipts/selection.json), and the stack log is
  [receipts/stack.log](receipts/stack.log).
- `patch_count` 673, `patch_head` `d74e38ee`, `expected_tree` `72b40d71`, `patchset_sha256` `5131e020`, recorded by
  `bin/spark build prepare --lock` ([source.json](source.json), [upstreams.lock.json](upstreams.lock.json)).
- Every other source is 1107's.
- **Authors:** #957 is christopherowen's, and both patches' `From:` is Christopher Owen's.

| Patch | PR | Title | PR head | Source |
| --- | --- | --- | --- | --- |
| `0797-perf-dsv41-branchless-sparse-MLA-stage-2-online-soft.patch` | #957 | TP3/TP4 decode compressor, sparse MLA, and spec-decode fusions | `e464d364431a` | its part 2, split at selection, retitled |
| `0798-perf-dsv41-fuse-DSpark-confidence-calibration-and-ke.patch` | #957 | TP3/TP4 decode compressor, sparse MLA, and spec-decode fusions | `e464d364431a` | its part 4, split at selection, retitled |

- **Each split is retitled for the part it carries:** only the Subject changed, and one body line names the PR's
  subject. Every kept hunk is the PR's.
  - `0797` (part 2): `tilelang/sparse_mla.py`, stage 2's online-softmax update without the segment guard.
  - `0798` (part 4): the fused `probability_and_calibrate`, the preallocated vocabulary-bias scratch, and the dummy
    run's active `num_reqs` (`input_batch.py`, two `model_runner.py` hunks, `dspark/speculator.py`, `nvidia/dspark.py`,
    `tilelang/confidence.py` and its test).
- #957's part 1 is candidate-56's `0795`. Part 3 is left out at selection: it is superseded by the base's `0764`
  (#965 round 1) and `0769` (#980 round 1).

**#957 part 5 is dropped** after its round-3 rework
([receipts/rework/pr-957-p5-r1](receipts/rework/pr-957-p5-r1/result.json), verdict `not-defensible`). Part 5 is the
TP3 adaptive-verification constants (`HIGH_ROW_COST_SCALE_TP3`, `MIN_CALIBRATED_DEAD_ROWS_TAU_TP3`).
- Its replay of the real budget policy predicts a TP3 throughput loss at every concurrency. By served step time that
  is about -1% at c1 and -3% at c8, beyond the 0.35-0.4% A/A floor.
- Its premise, that TP3 rows past the knee cost more than TP4's, does not hold on our grid. TP3-switched rows cost
  0.170 and 0.220 ms, and TP4-switched rows 0.255 and 0.272 ms.
- The rework's deliverables are kept: the patch, `overlay/`, `no957p5/`, `result.json`, `commands.txt` and `raw/`.
  The drop is recorded in [receipts/selection.json](receipts/selection.json) (`left_out`).

### Stacking

`0797` and `0798` apply cleanly with `git am` on candidate-54's head (`4e8133ff`, tree `5f021d9a`), without `0795` and
`0796`, to tree `72b40d71`, the selection's tree for 57 ([stack.log](receipts/stack.log), section 3). `git diff --check`
is clean. Each applies alone on the base, and in both orders (section 4). They also apply after candidate-56's 957-p1
(section 5).

**957-p4 and #960 round 1.** The selection asked for 957-p4's `model_runner.py` hunks to be read with `0788` (#960
round 1), whose FULL-graph capture builds metadata for a step's request count.
- **#960 round 1** stages the real batch on non-dummy FULL steps only. It never reads the capture's request count.
- **At capture,** FULL-mode `prepare_attn` sizes the metadata by `num_reqs_after_padding`. 957-p4 keeps that at the
  graph's request count, so the captured metadata keeps its padded extent.
- **What 957-p4 changes:** the dummy batch's tail rows become empty requests, as in a real padded step, and the
  speculator drafts for the active requests. So the `(4, 24)` target capture captures the `(4, 20)` DSpark draft
  graph.
- **What reading does not settle:** a captured kernel whose launch size follows the active count. The boot (no JIT
  after readiness), the gate and 957-p4's own capture check settle it.

The full review is in [selection.json](receipts/selection.json) (`semantic_review`). Candidates 55, 56 and 57 touch no
common file.

## Cluster configurations

A cluster configuration for every shape the recipe serves, so the candidate can be run on any setup: `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-57/<file> …` with your site's node map. Each one extends `releases/spark-ds41f-1107/cluster.json` at its shape and sets only the image `spark-ds41f:candidate-57`, its tree labels and this lock; no PR changes the recipe. The `-dev` file adds the dev-mode collective RPC that the screen's engine grid needs. This candidate is measured at `tp3-switched` only; the other files are configurations, not part of its runs.

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
- No overlay control: both parts edit files no other patch here touches, so `no<part>` overlay controls are
  possible if attribution needs one (part 4's files are Python; part 2's is a TileLang kernel source).

## Release tests

[release-tests.json](release-tests.json) is candidate-54's classification. #957's parts 2 and 4 change only
`tests/kernels/test_deepseek_v41_tilelang_confidence.py`, which it already gates. `scripts/release_tests.py check`
reports no problems on the prepared tree (sp8): 215 selected files, 153 gate and 62 excluded
([receipts/release-tests-check.txt](receipts/release-tests-check.txt)). The gate manifest is
[release-tests-gate-manifest.json](release-tests-gate-manifest.json) (154 files), and the bundle, written with main's
tooling, is [bundles/release-tests](bundles/release-tests).

## The build: sp8, outside a window

The lab's `image` and `kernel` jobs cannot target sp8: a window runs them only on its production profile's nodes,
the `image` job builds in the deployment checkout, and no site map pairs sp8 with sp5 alone. So sp8 builds and
tests the image by hand, with the same repository commands the jobs run, from a clean clone at the locked commit
(`~/spark-lab/agents/candidates-55-57/spark-ds41f`), outside sp8's deployment checkout:
- `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-57/image-sp8.json build prepare`,
  then `build check`, then `build image --apply` (the build under the memory guard, then the GPU smoke);
- the release-test bundle on sp8's GPU:
  `scripts/lab.py kernel-local experiments/2026-10-26-spark-ds41f-candidate/candidate-57/bundles/release-tests`;
- the image over the spaceport fabric, `ssh sp8 'docker save spark-ds41f:candidate-57 | ssh sp5 docker load'`,
  once the testbed is idle, then from sp5
  `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-57/tp3-switched.json build distribute --apply`
  to sp6 and sp7. Every rank shows the same image ID.

This skips the laptop stream of a dgx build (16-50 min an image). [image-sp8.json](image-sp8.json) is
[image-dgx4.json](image-dgx4.json) at `tp3-switched`; the lock and labels are the same. [build.json](build.json), the
dgx4 route, is kept as the fallback. The script as run and its log are in `receipts/build-candidate-57-image/`
once the build finishes.

## Runs

Each testbed opens its window with `scripts/lab.py`, after the hold and idle guards.

1. The image, built and tested on sp8 and copied to sp5 (above).
2. [candidate-tp3-switched.json](candidate-tp3-switched.json) (`candidate-57-tp3-switched`): the determinism gate at
   `tp3-switched`, first ([step 5](../../../docs/release-benchmark.md#5-gate-first)).
3. [screen-tp3-switched.json](screen-tp3-switched.json) (`candidate-57-screen-tp3-switched`): `screen-tp3`,
   `single-stream` and `sampled` on two boots, `candidate-57` and `candidate-57-again`, with startup memory
   ([step 6](../../../docs/release-benchmark.md#6-screen)).
   - **Reference:** candidate-54's `tp3-switched` screen at
     `experiments/2026-10-26-spark-ds41f-candidate/candidate-54/receipts/references/candidate-54-tp3-switched-screen`.
     The lab reads the companions at `-single-stream` and `-sampled` beside it.
   - The coordinator records them. Until they are on `main`, the lab's dry run reports them as missing.
   - `sampled` runs because part 4 changes draft confidences within FP32 rounding, and so sampled realizations may
     change.
   - No stall job: neither part touches prefill scheduling.
4. **Own marks** on sp8 (below), two boots each, ABBA against candidate-54.

## Marks

Declared before building or measuring ([methodology](../../../docs/methodology.md#round-purpose-report-and-drop-rule)).
A PR earns its keep in one of two ways:

- **Correctness fix:** its functional evidence passes on the candidate image, and no core metric is worse.
- **Performance change:** it improves its own metric, and no unrelated core metric is worse.

"Worse" and "better" mean beyond the screen table's intervals: the candidate's two boots, and the reference's repeat
(candidate-54's two boots). The paired decode row is judged against the recorded A/A floor.

**Core metrics** (the screen, its companions, the gate, the release tests):
- greedy identity against the reference (the screen's model step);
- the engine decode grid (up to eight requests);
- paired decode in the 8-stream closed loop, one stream (`single-stream`) and sampled (`sampled`);
- prefill at 32K and 256K;
- the determinism gate (below);
- startup (lowest `MemAvailable`, time to ready, no JIT after readiness) and steady memory (KV capacity, swap, OOMs);
- zero failed requests, and no silent fallback in any rank's boot log;
- the release tests (`check` reports no problems, and the bundle exits 0).

| PR | Kind | Own mark | Declared output change |
| --- | --- | --- | --- |
| #957 (part 2) | perf | Its `screen_tp3_decode.py` sparse-MLA cells on the candidate image, 2 boots, ABBA against candidate-54's image, by CUDA-graph replay: stage 2 at TP3 (24 heads, 12 and 16 rows) faster beyond the interval (the PR: -49..-62 us a step at 12 rows and -46..-99 us at 16, over 30 indexed layers), outputs `torch.equal`; TP4 cells (16 heads) not worse. On the testbed: engine-grid decode not worse | none: stage 2 starts M at the sink logit, so an inactive segment's (-inf, 0) partial leaves M, L and o as the skipped update did |
| #957 (part 4) | perf (spec-decode bookkeeping) | Its `screen_tp3_decode.py` confidence cells and `tests/kernels/test_deepseek_v41_tilelang_confidence.py` on the candidate image, 2 boots, ABBA: `probability_and_calibrate` faster than `probability` plus `calibrate_confidence_probs` beyond the interval (the PR: 17.2-19.0 -> 8.5-9.2 us), calibrated confidences within FP32 rounding of the two-step path; the `(4, 24)` dummy run captures the `(4, 20)` DSpark draft graph (functional). On the testbed: paired decode at 4 streams (the grid's 4-request cells) and 8 (the closed loop) not worse; `single-stream` and `sampled` not worse | draft confidences within FP32 rounding of the two-step path, so verification rows may differ; temperature-0 text must not change; sampled realizations may change |

### Own-mark commands

On sp8, from a checkout of this commit, each command under `flock /tmp/spark-lab-gpu.lock`, receipts in
`receipts/own-marks/pr-<N>/` with a `result.json` ([step 7](../../../docs/release-benchmark.md#7-own-marks)):

- **Images.** The candidate is `spark-ds41f:candidate-57`, the base `spark-ds41f:candidate-54`. Stream both to sp8
  first.
- **ABBA in one session:** base, candidate, candidate, base. Two boots of each arm.
- **Judge on the path serving takes:** these are decode kernels, so CUDA-graph replay or kernel time decides. Record
  the eager timings beside them.
- **TP3 settings.** sp8 is one GPU: run the cells at the TP3 shard shapes the PR used (24 heads, TP3 rows).

Per part:

- **#957 part 2.** Script: `experiments/2026-11-08-tp3-decode-recovery/screen_tp3_decode.py` at `e464d364`,
  `python3 screen_tp3_decode.py <out.json>`, with `PYTHONPATH` at the image's vLLM and a TileLang cache directory, as
  its README runs it. Judge its sparse-MLA cells at TP3 and TP4. Test:
  `pytest --noconftest tests/kernels/attention/test_deepseek_v41_tilelang.py -k test_sparse_mla_plan_routes`.
- **#957 part 4.** The same script's confidence cells. Tests:
  `pytest --noconftest tests/kernels/test_deepseek_v41_tilelang_confidence.py`, and a check that the `(4, 24)` dummy
  run captures the `(4, 20)` DSpark draft graph (the candidate's boot log on the testbed, or a one-GPU capture at TP3
  sizing).

**Rework (owner, 2026-11-09).** A PR that misses its mark goes to a sub-agent for one focused rework, with the miss,
the receipts and the mark. A second rework is allowed when the first shows promise. A PR that still misses is dropped
from this release, and its record stays here. A reworked PR joins the next candidate with its own source identity and
fresh qualification.

## Determinism gate: declared output changes

| PR | Declared output change |
| --- | --- |
| #957 (part 4) | draft confidences within FP32 rounding, so verification rows may differ; temperature-0 text must not change; sampled realizations may change |

The gate decodes greedily, so its hashes must equal candidate-54's TP3 hashes: `5c22ce41`/`bce6092e` with logprobs
(text `26a739a9`/`12cfee9b`), from candidate-54's `tp3-switched` gate on sp5 (2026-11-10). They differ from TP3's
pre-#1030 hashes by construction. Part 2 declares none.

## Selection

One of candidates 55, 56 and 57, tested at the same time, one per testbed. The PRs that pass are carried into all
later candidates. The round's selection receipts are in [candidate-55/receipts](../candidate-55/receipts).

## Status

Built on sp8 (`spark-ds41f:candidate-57`, image ID `ba512cb2`). Its release tests run on sp8 (`kernel-local`).

| Testbed | Gate | Screen against candidate-54 |
| --- | --- | --- |
| `tp3-switched` | identical outputs; candidate-54's TP3 hashes 5c22ce41/bce6092e | grid the same (4-request shapes 1.7-1.9% faster), c8 decode 0.56% faster (one boot), `single-stream` and `sampled` the same, greedy identical; prefill not measured (sp5's 4 GiB bench floor) ([receipts/tp3-switched](receipts/tp3-switched/README.md)) |

Its release tests pass on sp8 (10,174 passed, 311 skipped, 0 failed). Both own marks are met
([receipts/own-marks](receipts/own-marks/README.md)):
- **#957 part 2:** confirmed on 4+4 boots in one session: TP3 12- and 16-row cells 2.7-3.8% faster a call beyond the
  interval, no TP4 cell slower (the c1 shape +0.2%, the same), outputs bit-identical. Predicted serving share about
  -0.05% at TP3 c1: a micro gain below serving resolution, carried forward.
- **#957 part 4:** the fused calibration faster in 5/5 cells, within 1-2 FP32 ULP; the dummy run now drafts the
  active requests through the (4, 20) graph.

**Verdict: candidate-57 passes.** #957 parts 2 and 4 join the next merged base.
