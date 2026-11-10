# candidate-56

Unpromoted release candidate: candidate-54 plus #957's part 1 (the replicated BF16 compressor at TP <= 4) and #956
round 3 (the SP ratio-2 compressor projection and CED encoder-only chunks), which builds on it. It is tested at
`tp4-switched` (sp1..sp4, head sp1).

## Source

- vLLM: candidate-54's series by relative path to its files ([candidate-54/vllm/series](../candidate-54/vllm/series):
  1107's `0001..0761`, 641 active, candidate-50's `0762..0780` with #1009 round 1, and round 2's `0781..0791`; 671
  patches). Then the two patches below as `0795` and `0796` ([vllm/series](vllm/series)). The provenance, SHA-256 and
  author of each are in [receipts/selection.json](receipts/selection.json), and the stack log is
  [receipts/stack.log](receipts/stack.log).
- `patch_count` 673, `patch_head` `295c2ced`, `expected_tree` `0c848613`, `patchset_sha256` `0a6aa2ea`, recorded by
  `bin/spark build prepare --lock` ([source.json](source.json), [upstreams.lock.json](upstreams.lock.json)).
- Every other source is 1107's.
- **Authors:** every PR is christopherowen's, and every patch's `From:` is Christopher Owen's.

| Patch | PR | Title | PR head | Source |
| --- | --- | --- | --- | --- |
| `0795-perf-dsv41-replicate-the-BF16-compressor-projection-.patch` | #957 | TP3/TP4 decode compressor, sparse MLA, and spec-decode fusions | `e464d364431a` | its part 1, split at selection, retitled |
| `0796-perf-dsv41-sp-ratio2-compressor-and-ced-encoder-only-chunks.patch` | #956 | perf(dsv41): SP ratio-2 compressor projection and CED encoder-only chunks | `ecb3812cb8fa` | [round 3 rebase](receipts/rework/pr-956-r3-rebase) onto candidate-54 + 957-p1 |

- **#957 is carried as its part 1** (`compressor.py`, `tilelang/linear.py` and the compressor test). `0795` is retitled
  for that part: only the Subject changed, and one body line names the PR's subject. Parts 2 and 4 are candidate-57's;
  part 3 is left out at selection (superseded by the base's `0764` and `0769`); part 5 is dropped (candidate-57).
- **#956 round 3** ([result.json](receipts/rework/pr-956-r3-rebase/result.json), verdict `ready`) re-derives the PR on
  candidate-54 + 957-p1: the ratio-2 layers take #1009 round 1's SP local front, their SP projection reads 957-p1's
  replicated weight (no all-gather at TP <= 4), and CED's encoder-only exit sits beside #960 round 1's pinned rings.
  Every output its harness compares is bit-identical to candidate-54 + 957-p1. Its deliverables are copied into
  [receipts/rework/pr-956-r3-rebase](receipts/rework/pr-956-r3-rebase): both patches, the overlay, `no956/`,
  `result.json`, `commands.txt`, `raw/` and `shim/` (its `work/` scratch is not).
- **Retitling changes the commits, not the tree.** The rework's stack gave patch head `abb26187` with 957-p1 as
  selected. With `0795` retitled the head is `295c2ced`; the tree is the same, `0c848613`.

### Stacking

`0795` then `0796` apply cleanly with `git am` on candidate-54's head (`4e8133ff`, tree `5f021d9a`) to tree `0c848613`,
#956 round 3's tree ([stack.log](receipts/stack.log), section 3a). `git diff --check` is clean. `0795` alone gives
`3959b211`, the selection's tree for 957-p1. `0796` does not apply without `0795`, by design: it builds on 957-p1's
`compressor.py` (section 4). The files these patches share with base patches were read
([selection.json](receipts/selection.json), `semantic_review`). Candidates 55, 56 and 57 touch no common file.

## Cluster configurations

A cluster configuration for every shape the recipe serves, so the candidate can be run on any setup: `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-56/<file> …` with your site's node map. Each one extends `releases/spark-ds41f-1107/cluster.json` at its shape and sets only the image `spark-ds41f:candidate-56`, its tree labels and this lock; no PR changes the recipe. The `-dev` file adds the dev-mode collective RPC that the screen's engine grid needs. This candidate is measured at `tp4-switched` only; the other files are configurations, not part of its runs.

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
- **No956 control:** [tp4-switched-no956.json](tp4-switched-no956.json) (and its [-dev](tp4-switched-no956-dev.json))
  mounts [no956/overlay](no956/overlay) over the candidate-56 image, as candidate-53's `no960-no998` controls do.
  - The overlay is #956's six source files as they are on candidate-54 + `0795`. Each equals that tree's blob, and
    `0796` applies to that tree ([stack.log](receipts/stack.log), section 3a).
  - 957-p1 is the only other patch that touches them (`compressor.py`), and its change stays in. So the control
    isolates #956 from 957-p1.
  - #956's three test files are not mounted.

## Release tests

[release-tests.json](release-tests.json) is candidate-54's classification. 957-p1 and #956 round 3 change only test
files it already gates (`tests/models/test_deepseek_v4_1_compressor.py`, `_compressor_prepare.py`,
`_ced_model.py`), so no file needed a new classification. `scripts/release_tests.py check` reports no problems on the
prepared tree (sp8): 215 selected files, 153 gate and 62 excluded
([receipts/release-tests-check.txt](receipts/release-tests-check.txt)). The gate manifest is
[release-tests-gate-manifest.json](release-tests-gate-manifest.json) (154 files), and the bundle, written with main's
tooling, is [bundles/release-tests](bundles/release-tests).

## The build: dgx4, not sp8

The coordinator asked for an sp8 build copied to sp1 over the spaceport fabric. The repository has no sound route
for it:
- **The kernel job.** A lab window runs a `kernel` job only on its production profile's nodes. No live production
  profile contains sp8: `tp4-switched` is sp1..sp4 and `tp3-switched` sp5..sp7, and `tp8-switched` is not served this
  round.
- **The checkout.** The `image` job builds in the builder's deployment checkout, and a window's `sync` moves only its
  production nodes' checkouts. sp8's deployment checkout is on another agent's working branch (2026-11-10).
- **The node map.** The `image` job then requires the same layers on every node of the build profile's node map. No
  site map names sp8 with sp1 (or sp5) alone, so it would need a new site map.
- **The precedents on sp8 are other routes.** `2026-11-06-ced-local-projection` assembled a Python-only image on sp8
  with its own scripts and node map (`sp567-nodes.json`). `2026-11-01-dspark-default-copy-drafts` built thin `FROM`
  images by hand and never distributed them.
- **The node's use.** sp8 is the round's one-GPU own-mark node, and a full build there loads its CPU under the
  benches.

So [build.json](build.json) builds on dgx4, as candidate-53's did, and the image crosses to spaceport through the
laptop once (step 4).

## Runs

Each testbed opens its window with `scripts/lab.py`, after the hold and idle guards.

1. [build.json](build.json) (`candidate-56-image`), in a dgx window
   ([step 4](../../../docs/release-benchmark.md#4-build-and-release-tests-dgx-window)): build the image on dgx4, copy it
   along the ring, then one `kernel` job with this candidate's release-test bundle on dgx4. Then stream it once to sp1
   (`ssh dgx1 docker save spark-ds41f:candidate-56 | ssh sp1 docker load`) and distribute it from there with
   `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-56/tp4-switched.json build distribute --apply`.
   Every rank shows the same image ID.
2. [candidate-tp4-switched.json](candidate-tp4-switched.json) (`candidate-56-tp4-switched`): the determinism gate at
   `tp4-switched`, first ([step 5](../../../docs/release-benchmark.md#5-gate-first)).
3. [screen-tp4-switched.json](screen-tp4-switched.json) (`candidate-56-screen-tp4-switched`): the `screen` suite on
   `candidate-56` and `candidate-56-again`, with startup memory ([step 6](../../../docs/release-benchmark.md#6-screen)).
   **Reference:** candidate-54's `tp4-switched` screen at
   `experiments/2026-10-26-spark-ds41f-candidate/candidate-54/receipts/references/candidate-54-tp4-switched-screen`.
   The coordinator records it; until it is on `main`, the lab's dry run reports it as missing.
4. [attribute-956-tp4-switched.json](attribute-956-tp4-switched.json) (`candidate-56-attribute-956-tp4-switched`):
   #956's own serving mark. [prefill-suite.json](prefill-suite.json) is the screen's prefill step at #956's contexts
   (32K, 64K, 128K and 256K, three repeats each, as its servebench prefill ran). It runs ABBA in one session:
   `candidate-56`, `candidate-56-no956`, `candidate-56-no956-again`, `candidate-56-again`.
5. **Own marks** on sp8 (below), two boots each, ABBA against the base.

## Marks

Declared before building or measuring ([methodology](../../../docs/methodology.md#round-purpose-report-and-drop-rule)).
A PR earns its keep in one of two ways:

- **Correctness fix:** its functional evidence passes on the candidate image, and no core metric is worse.
- **Performance change:** it improves its own metric, and no unrelated core metric is worse.

"Worse" and "better" mean beyond the screen table's intervals: the candidate's two boots, and the reference's repeat
(candidate-54's two boots). The paired decode row is judged against the recorded A/A floor.

**Core metrics** (the screen, the gate, the release tests):
- greedy identity against the reference (the screen's model step);
- the engine decode grid;
- paired decode in the 16-stream closed loop;
- prefill at 32K and 256K;
- the determinism gate (below);
- startup (lowest `MemAvailable`, time to ready, no JIT after readiness) and steady memory (KV capacity, swap, OOMs);
- zero failed requests, and no silent fallback in any rank's boot log;
- the release tests (`check` reports no problems, and the bundle exits 0).

| PR | Kind | Own mark | Declared output change |
| --- | --- | --- | --- |
| #957 (part 1) | perf | Its `screen_tp3_decode.py` compressor cells and `tests/models/test_deepseek_v4_1_compressor.py` on the candidate image, 2 boots, ABBA against candidate-54's image, by CUDA-graph replay and kernel time: the 4 compressor projections' local time lower beyond the interval (the PR: 103.6-107.6 -> 51.6-53.1 us a step), the decode step lower by the 4 removed all-gathers (the PR: about 135-155 us a step), compressor outputs `torch.equal` to TP1 at TP2..TP4. On the testbed: engine-grid decode cells and SP prefill at 32K and 256K not worse; the static memory delta recorded (the PR: +21.9..24.4 MiB a rank). It acts at TP3 and TP4 only | none (the replicated path is TP1's, which the test already required) |
| #956 (round 3) | perf (prefill) | Its servebench prefill on `tp4-switched` ([attribute-956-tp4-switched.json](attribute-956-tp4-switched.json)): candidate-56 against its no956 control, ABBA in one session, prefill throughput better at 32K..256K and not worse at any context (the PR on 1107: +2.31%, +0.44%, +1.67%, +1.71% at 32K, 64K, 128K, 256K). The screen's prefill steps and greedy identity cover bit-exactness end to end. On sp8: its `bf16_gemm_two` bit-exactness test (`test_sp_ratio2_projection_matches_full_rows`) and the round-3 tests on the candidate image. Watch the zero hidden state an encoder-only step hands the greedy vocabulary screen (about one head GEMV a non-final chunk, estimated in the rework) | none (bit-exact) |

### Own-mark commands

On sp8, from a checkout of this commit, each command under `flock /tmp/spark-lab-gpu.lock`, receipts in
`receipts/own-marks/pr-<N>/` with a `result.json` ([step 7](../../../docs/release-benchmark.md#7-own-marks)):

- **Images.** The candidate is `spark-ds41f:candidate-56`, the base `spark-ds41f:candidate-54`. Stream both to sp8
  first. #956's base is the candidate image with [no956/overlay](no956/overlay) mounted, as
  [tp4-switched-no956.json](tp4-switched-no956.json) mounts it.
- **ABBA in one session:** base, candidate, candidate, base. Two boots of each arm.
- **Judge on the path serving takes.** Record CUDA-graph replay or kernel time, and the eager timings beside them.

Per PR:

- **#957 part 1.** Script: `experiments/2026-11-08-tp3-decode-recovery/screen_tp3_decode.py` at `e464d364`,
  `python3 screen_tp3_decode.py <out.json>`, with `PYTHONPATH` at the image's vLLM and a TileLang cache directory, as
  its README runs it. Judge its compressor cells; the sparse-MLA and confidence cells are candidate-57's parts. Tests:
  `pytest --noconftest tests/models/test_deepseek_v4_1_compressor.py`.
- **#956.** The rework's harness and layer bench ([shim](receipts/rework/pr-956-r3-rebase/shim): `attn_harness.py`,
  `bench_layers.py`, as its `commands.txt` runs them) on the candidate image against the same image with the no956
  overlay, ABBA. Tests: the rework's list (`result.json`, `tests.files`), including
  `test_deepseek_v4_1_compressor.py::test_sp_ratio2_projection_matches_full_rows`.

**Rework (owner, 2026-11-09).** A PR that misses its mark goes to a sub-agent for one focused rework, with the miss,
the receipts and the mark. A second rework is allowed when the first shows promise. A PR that still misses is dropped
from this release, and its record stays here. A reworked PR joins the next candidate with its own source identity and
fresh qualification.

## Determinism gate: declared output changes

None. Both patches declare bit-identical output, so the gate is a pure identity check against candidate-54's TP4
hashes: `18d61961`/`d6568649` with logprobs (text `0a2dacdd`/`12fd8257`). `tp4-ring` and `tp4-switched` share TP4's
hashes.

## Selection

One of candidates 55, 56 and 57, tested at the same time, one per testbed. The PRs that pass are carried into all
later candidates. The round's selection receipts are in [candidate-55/receipts](../candidate-55/receipts).

## Status

Prepared and locked. Not built, not measured.
