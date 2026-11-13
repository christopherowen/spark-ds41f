# candidate-51

Unpromoted release candidate: candidate-50 without #1009 plus 4 PRs, bit-identical TP4 decode kernels
(GEMM and vocabulary tiles at 97..128 rows, L2 prefetch pacing). It is tested at `tp4-ring` (dgx1..dgx4).

## Source

- vLLM: candidate-50's series without `0778-perf-dsv41-sp-prefill-local-attention-front.patch` (#1009):
  1107's `0001..0761` (641 active) plus candidate-50's `0762..0777`, `0779` and `0780`, 659 patches listed
  by relative path to their files. Then the patches below, renumbered from `0781` in the selection's
  order ([vllm/series](vllm/series)). Each is copied from its PR's head, or is the selection's rebase or
  split of it. The provenance and SHA-256 of each are in [receipts/selection.json](receipts/selection.json),
  and the stack log is [receipts/stack.log](receipts/stack.log).
- `patch_count` 663, `patch_head` `cc1115c5`, `expected_tree` `987ba058`,
  `patchset_sha256` `9fae6dc3`, recorded by `bin/spark build prepare --lock`
  ([source.json](source.json), [upstreams.lock.json](upstreams.lock.json)).
- Every other source is 1107's.

| Patch | PR | Title | PR head | Source |
| --- | --- | --- | --- | --- |
| `0781-perf-dsv41-64k-chunked-globaltimer-paced-l2-prefetch.patch` | #961 | perf(dsv41): 64 KiB chunked, %globaltimer-paced side-stream L2 prefetch | `9362f944201a` | the PR's patch |
| `0782-perf-dsv41-unconditional-pack12-plane-loads-and-swar.patch` | #963 | perf(dsv41): unconditional pack12 plane loads, byte-perm vocab SWAR, and per-column exception lookup | `e5db47f172cf` | the PR's patch |
| `0783-perf-dsv41-extend-WO-B-WIDE_DECODE-to-128-rows-and-g.patch` | #987 | perf(dsv41): extend WO-B WIDE_DECODE to 128 rows and guard high-concurrency decode cliffs | `9be94ed5310c` | the WO-B part, split at selection ([rebases](receipts/rebases/rebases.json)) |
| `0784-Match-128-row-vocab-head-tile-pipeline-stages.patch` | #1001 | perf(dsv41): match 128-row vocab head tile pipeline stages on GB10 | `03e2d903daf9` | context-only rebase ([rebases](receipts/rebases/rebases.json)) |

**The base leaves out #1009.** It causes candidate-50's `tp4-ring` boot failure
([candidate-50/receipts/tp4-boot-failure](../candidate-50/receipts/tp4-boot-failure)) and is in rework;
when its rework passes it joins a later base. Without `0778` the base's tree is `14c5a2dc`, the same as
candidate-50's prepared tree with `0778` reverted, which the selection used ([stack.log](receipts/stack.log),
section 1).

**#987 is carried as its WO-B part.** `0783` keeps `tilelang/gemm.py`, `tilelang/wo.py` and their test
(`tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py`) unchanged; its message is the PR's, plus one
line naming the kept files. The rest of the PR's patch was left out at selection:

| Part of #987 | Why it is left out |
| --- | --- |
| `sp_prefill.py` and its test (`resolve_min_rows`) | does nothing on this deployment: at `max_num_seqs` <= 16 the SP floor stays at the 205-row RoCE limit; it acts only past 32 streams |
| `cudagraph_utils.py` and its test (`_FULL_DEPTH_VARLEN_DECODE_MAX_REQS = 16`) | does nothing on this deployment: no change to captured graphs at `max_num_seqs` <= 16, and the recipe serves 16 (TP4) and 8 (TP3) |
| `tilelang/collectives.py` and `cuda_communicator.py` (`all_reduce_rank_order` through RoCE reduce-scatter + all-gather) | not inert: it moves CED decoder steps of 205..408 rows from the PyNCCL rank-order path to RoCE, a second causal variable unrelated to the WO-B mark. Split out; to be re-submitted as its own experiment |

Every patch the round left out, with its reason, is in [receipts/selection.json](receipts/selection.json)
(`left_out`).

### Stacking

All four apply cleanly with `git am` on the base, in this order ([stack.log](receipts/stack.log), section 2),
and `git diff --check` is clean. #963 shares `tilelang/gemm.py` with #987 and `tilelang/vocab.py` with #1001;
each pair edits different code, and the selection found no conflict
([pairwise-conflicts.json](receipts/pairwise-conflicts.json)). The two rebases change context only: every added
and removed line is the PR's ([rebases.json](receipts/rebases/rebases.json)). Candidates 51, 52 and 53 touch no
common file, test files included.

## Cluster configurations

A cluster configuration for every shape the recipe serves, so the candidate can be run on any setup: `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-51/<file> …` with your site's node map. Each one extends `releases/spark-ds41f-1107/cluster.json` at its shape and sets only the image `spark-ds41f:candidate-51`, its tree labels and this lock; no PR changes the recipe. The `-dev` file adds the dev-mode collective RPC that the screen's engine grid needs. This candidate is measured at `tp4-ring` only; the other files are configurations, not part of its runs.

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
- No overlay control: #961's two `l2_prefetch.py` files are touched by no other patch here, so a `tp4-no961`
  control is possible if attribution needs one; #963, #987 and #1001 share `gemm.py` and `vocab.py`, so none of
  them has one.

## Release tests

[release-tests.json](release-tests.json) is candidate-50's classification. These PRs change only test files it
already gates (`tests/models/test_glm5next_tilelang_l2_prefetch.py`,
`tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py`,
`tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py`), so no file needed a new classification.
It also carries main's release-test change that landed after candidate-50 (#1038): the bundle resolves
upstream tests' stand-in hub models to the served checkpoint, and two `tests/v1/core/test_scheduler.py` tests
that then run are deselected with their reason, as in `config/release-tests.json`.
`scripts/release_tests.py check` reports no problems on dgx4's prepared tree: 215 selected files, 153 gate and 62 excluded
([receipts/release-tests-check.txt](receipts/release-tests-check.txt)). The gate manifest is
[release-tests-gate-manifest.json](release-tests-gate-manifest.json) (154 files), and the bundle is
[bundles/release-tests](bundles/release-tests).

## Runs

Each testbed opens its window with `scripts/lab.py`, after the hold and idle guards.

1. [build.json](build.json) (`candidate-51-image`), in the round's one dgx window
   ([release-benchmark](../../../docs/release-benchmark.md#4-build-and-release-tests-dgx-window), step 4):
   build the image on dgx4 and copy it along the ring. Queue candidate-51's, candidate-52's and then
   candidate-53's `build.json`; candidate-53's ends with the round's one `kernel` job, which runs the three
   release-test bundles at once, each on its own node: this candidate's
   ([bundles/release-tests](bundles/release-tests)) on dgx4.
2. [candidate-tp4.json](candidate-tp4.json) (`candidate-51-tp4`): the determinism gate at `tp4-ring`, first
   ([step 5](../../../docs/release-benchmark.md#5-gate-first)).
3. [screen-tp4.json](screen-tp4.json) (`candidate-51-screen-tp4`): the `screen` suite on `candidate-51` and
   `candidate-51-again`, with startup memory ([step 6](../../../docs/release-benchmark.md#6-screen)).
   - **Reference:** candidate-50's screen at `tp4-ring`, the accumulated base, recorded at
     `experiments/2026-10-26-spark-ds41f-candidate/candidate-50/receipts/references/candidate-50-no1009-tp4-ring-screen`.
   - The record is on `main` (candidate-50's `tp4-ring` receipts), a run of the current `screen` suite: the
     lab's dry run accepts it. candidate-50's `tp4-switched` and `tp3-switched` references, which candidates 52
     and 53 name, are still pending.
   - No companion suite: every mark below is measured by the screen or by the PR's own bench.
4. **Own marks** on sp8 (below), two boots each, ABBA against the base.
5. [single-stream-tp4.json](single-stream-tp4.json) (`candidate-51-single-stream-tp4`): the `single-stream`
   suite at `tp4-ring`, added after the screen. The screen's one-request grid rows are 4.4-8.4% faster than
   candidate-50's, and its 16-stream paired decode 0.68%; the screen has no one-stream closed loop, so the
   effect on one-stream decode is not measured.
   - **Arms:** `candidate-51`, `candidate-50-no1009` (the accumulated base, as the middle arm) and
     `candidate-51-again`, on the same development arms as the screen.
   - **The base is measured here** because no `single-stream` record of candidate-50 exists at `tp4-ring`
     ([step 6](../../../docs/release-benchmark.md#6-screen): a reference is recorded by the run that first
     needs it). 1107's recorded run (`manifests/suites/2026-11-07-spark-ds41f-1107-tp4-single-stream`) is
     from another session, so it is printed beside the table, not used for the verdict.
   - **Result** (dgx1 window 2026-11-09 23:52Z to 2026-11-10 00:21Z,
     [table](receipts/tp4-ring/lab/candidate-51-single-stream-tp4-table.txt)): against candidate-50 without
     #1009, one-stream paired decode is 1.17% better (margin 0.39%), on both boots (+1.14% and +1.20%).
     - Output: 81.82 tok/s for the base, 82.94 and 82.97 tok/s for candidate-51 (window row +1.36% ±5.85%, same).
     - Engine steps a second +0.96% and +1.01%, tokens a step +0.43% and +0.40%: each the same within its interval.
     - The candidate's repeat: +0.04% paired.
     - So the grid's 4.4-8.4% on one-request rows reaches one-stream serving as about 1.2%.
   - **Against 1107's recorded run** (another session,
     [table](receipts/tp4-ring/lab/candidate-51-single-stream-tp4-vs-1107-table.txt)): paired decode +0.42%
     (margin 0.57%), the same; output 82.62 against 82.94 tok/s; paired TTFT 1.11% worse (±0.27%). The base
     measured here reads 1.0% below 1107's record. This run cannot tell a session difference from an effect
     of candidate-50's PRs at one stream: no 1107 arm ran in this window.
   - **Receipts:** `receipts/tp4-ring/suites/candidate-51-single-stream-tp4/`,
     `receipts/tp4-ring/determinism/candidate-51-single-stream-tp4/` (startup memory, telemetry) and
     `receipts/tp4-ring/lab/candidate-51-single-stream-tp4*` (the run's log, tables and mounted-file hashes).
   - **How it was started:** `queue add` on dgx1, then
     [candidate-51-single-stream-tp4-idle_then_queue.sh](receipts/tp4-ring/lab/candidate-51-single-stream-tp4-idle_then_queue.sh):
     `scripts/lab_idle_start.sh`'s rule of 10 idle minutes, with the runner started as
     `queue run --idle-minutes 2` so the window closed and production was restored once the queue emptied
     ([its log](receipts/tp4-ring/lab/candidate-51-single-stream-tp4-idle.log)).

## Marks

Declared before building or measuring ([methodology](../../../docs/methodology.md#round-purpose-report-and-drop-rule)).
A PR earns its keep in one of two ways:

- **Correctness fix:** its functional evidence passes on the candidate image, and no core metric is worse.
- **Performance change:** it improves its own metric, and no unrelated core metric is worse.

"Worse" and "better" mean beyond the screen table's intervals: the candidate's two boots, and the
reference's repeat (candidate-50's two boots without #1009). The paired decode row is judged against the
recorded A/A floor.

**Core metrics** (the screen, the gate, the release tests):
- greedy identity against the reference (the screen's model step);
- the engine decode grid;
- paired decode in the 16-stream closed loop;
- prefill at 32K and 256K;
- the determinism gate (temperature-0 output hashes equal to the 1017 checkpoint's, unless a PR declares a change below);
- startup (lowest `MemAvailable`, time to ready, no JIT after readiness) and steady memory (KV capacity, swap, OOMs);
- zero failed requests, and no silent fallback in any rank's boot log;
- the release tests (`check` reports no problems, and the bundle exits 0).

A kernel PR's own mark is measured on the candidate image, the integrated stack, by the PR's own
benchmark: two boots on sp8. When a core metric is worse, the PRs whose path can move it are
removed one at a time until the cause is found.

| PR | Kind | Own mark | Declared output change |
| --- | --- | --- | --- |
| #961 | perf | Its `bench_l2_prefetch_ab.py` (16 MiB cold prefetch, then the consumer read) on the image, 2 boots, ABBA against the base image, with the recipe's `VLLM_L2_PREFETCH_GRID=2`: the consumer read faster beyond the interval and the L2 hit share higher; its pytest passes; no decode cell of the engine grid worse | none |
| #963 | perf | Its `screen_pack12_kernels.py` on the image, 2 boots, ABBA against the base image, by kernel time and CUDA-graph replay: packed BF16 GEMM (`compressor_up`, M=1..256) and packed vocab projection and greedy partials faster beyond the interval, 0 ULP, including through #997's certified screen (in the base) | none |
| #987 | perf | WO-B part only. Its WO-B bench (`mxfp8_gemm` and `TileLangWOProjection` at M=104..128, K=3072/2048/1024) on the image, 2 boots, ABBA: faster beyond the interval at M=97..128, `torch.equal` against the prefill GEMM; the engine grid's 16-stream cell not worse. It acts only at TP4: rows past 96 need 16 streams with copy widening (107 rows at `max_num_seqs` 16; TP3's 8 streams reach 59) | none |
| #1001 | perf | Its `bench_vocab_2boot.py` (captured CUDA graphs) on the image, 2 boots, ABBA: projection and greedy epilogue at M=97..128 (the 128-row tile) faster beyond the interval, bit-identical logits and (max, index) pairs, M<=96 not worse. TP4 only, as #987 | none |

### Own-mark commands

On sp8, from a checkout of this commit, each command under `flock /tmp/spark-lab-gpu.lock`, receipts in
`receipts/own-marks/pr-<N>/` with a `result.json`
([step 7](../../../docs/release-benchmark.md#7-own-marks)):

- **Images.** The candidate is `spark-ds41f:candidate-51`. The base is `spark-ds41f:candidate-50` with
  candidate-50's four `no1009/overlay` files mounted over it, as [candidate-50's tp4-no1009.json](../candidate-50/tp4-no1009.json)
  mounts them. Stream both to sp8 first (`ssh dgx1 docker save spark-ds41f:candidate-<N> | ssh sp8 docker load`,
  for candidate-51 and candidate-50).
- **ABBA in one session:** base, candidate, candidate, base. Two boots of each arm. The base arm is not a
  release record, so measuring it here is allowed.
- **Judge on the path serving takes:** these are decode kernels, so CUDA-graph replay or kernel time decides.
  Record the eager timings beside them.
- The PRs' commands patched or mounted their change into `spark-ds41f:1107`. On the candidate image the change
  is already in, so run the bench scripts from the PR heads without the patch step.
- Several scripts compare in-process against an embedded 1107 copy of the kernel. Keep that comparison as
  recorded, and add the image A/B: the base image's kernel against the candidate image's.

Per PR:

- **#961.** Script: `experiments/2026-11-08-l2-prefetch-pacing/bench_l2_prefetch_ab.py` at `9362f944`. It embeds
  the 1107 prefetch (`_ORIG_HEADER`) and times off, the 1107 grids and the 64 KiB paced candidates in one pass.
  Run it on both images with `VLLM_L2_PREFETCH_GRID=2`:
  ```bash
  docker run --rm --gpus all --ipc=host -e VLLM_L2_PREFETCH_GRID=2 -v "$PWD/<pr-961 dir>:/work" \
    --entrypoint python3 <image> /work/bench_l2_prefetch_ab.py --boot <k> --replays 35 --out /work/receipts/<arm>-boot<k>.json
  ```
  Then `pytest tests/models/test_glm5next_tilelang_l2_prefetch.py` in the candidate image (the PR: 34 passed,
  1 skipped).
- **#963.** Script: `experiments/2026-11-08-pack12-unconditional-loads-swar/screen_pack12_kernels.py` at
  `e5db47f1`, `python3 screen_pack12_kernels.py boot<k>`. It builds its candidate by text replacement on the
  image's `gemm.py` and by overwriting the image's `vocab.py` with a copy from
  `/tmp/pack12_exp/vllm-1107-exact` (line 331). On the candidate image the change is already in, and
  `gemm.py` and `vocab.py` also carry #987 and #1001. So run only its timing half (the image's own kernels,
  without the patch steps) on each image, ABBA. Then the PR's tests in the candidate image:
  `test_deepseek_v41_tilelang_vocab.py`, `test_deepseek_v41_tilelang_linear.py`,
  `test_deepseek_v41_tilelang_embedding.py` and `test_deepseek_v41_tilelang_pdl.py` (the PR: 274 passed).
- **#987.** The PR's WO-B bench (`/tmp/bench_wo_b.py`, run as `python3 /tmp/bench_wo_b.py <out.json>`) is not
  committed in the PR; only its receipts are (`receipts/wo_b_boot1.json`, `wo_b_boot2.json`). Recover it from the
  PR's author, or write a bench of `mxfp8_gemm` and `TileLangWOProjection` at M=97..128 and K=3072/2048/1024
  whose rows match the receipts, and commit it with the own-mark receipts. Its high-concurrency bench covers the
  parts left out, so it is not run. Tests: `test_deepseek_v41_tilelang_linear.py::test_wo_b_wide_decode_128_rows_matches_prefill`,
  `::test_mxfp8_decode_tile_repeats` and `tests/kernels/quantization/test_deepseek_v41_block_scale_words.py`.
- **#1001.** Script: `experiments/2026-11-08-vocab-head-128tile-greedy/bench_vocab_2boot.py` at `03e2d903`,
  `python3 bench_vocab_2boot.py <out.json>` with `-w /opt/spark-ds41f/candidate/vllm` and a TileLang cache
  mount, as the PR ran it. It times its embedded `base_1107` tile configuration against the image's in captured
  CUDA graphs. Run it on both images, ABBA. Tests: `test_deepseek_v41_tilelang_vocab.py`,
  `tests/kernels/test_deepseek_v41_tilelang_pdl.py`, `tests/v1/worker/gpu/test_greedy_verification.py`
  (the PR: 105 passed).

**Rework (owner, 2026-11-09).** A PR that misses its mark goes to a sub-agent for one focused
rework, with the miss, the receipts and the mark. A second rework is allowed when the first
shows promise. A PR that still misses is dropped from this release, and its record stays here.
A reworked PR joins the next candidate with its own source identity and fresh qualification.

## Determinism gate: declared output changes

None. Every PR here declares bit-identical output, so the gate is a pure identity check against the 1017
checkpoint's hashes.

## Selection

One of candidates 51, 52 and 53, tested at the same time, one per testbed. The PRs that pass are carried into
all later candidates. The selection receipts are here, in [receipts](receipts):

- [pr-triage-2026-11-09.md](receipts/pr-triage-2026-11-09.md): every open PR on 2026-11-09, against this base.
- [selection-proposal.json](receipts/selection-proposal.json): the order, SHA-256s, marks, placement and `left_out`.
- [pairwise-conflicts.json](receipts/pairwise-conflicts.json) and [pr-patch-targets.json](receipts/pr-patch-targets.json).
- [rebases](receipts/rebases): the three patches that differ from the PR's own (candidate-51's `0783` and
  `0784`, candidate-53's `0782`).
- [scripts-as-run](receipts/scripts-as-run).

## Status

Prepared and locked. Not built, not measured.
