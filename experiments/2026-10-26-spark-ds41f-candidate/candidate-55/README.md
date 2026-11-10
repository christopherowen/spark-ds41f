# candidate-55

Unpromoted release candidate: candidate-54 plus 3 PRs, host-side and sampler changes that declare no output change
(the rank-0 and RoCE-proxy CPU affinity, the multimodal prefix-cache fast path, the dead-row sampler skip). It is tested
at `tp4-ring` (dgx1..dgx4).

## Source

- vLLM: candidate-54's series by relative path to its files ([candidate-54/vllm/series](../candidate-54/vllm/series):
  1107's `0001..0761`, 641 active, candidate-50's `0762..0780` with #1009 round 1, and round 2's `0781..0791`; 671
  patches). Then the patches below, renumbered from `0792` in the selection's order ([vllm/series](vllm/series)). Each
  is the PR's patch or the selection's rebase or split of it. The provenance, SHA-256 and author of each are in
  [receipts/selection.json](receipts/selection.json), and the stack log is [receipts/stack.log](receipts/stack.log).
- `patch_count` 674, `patch_head` `410463b3`, `expected_tree` `e39b835c`, `patchset_sha256` `6a1f3da3`, recorded by
  `bin/spark build prepare --lock` ([source.json](source.json), [upstreams.lock.json](upstreams.lock.json)).
- sparknet: 1107's pin `1fd6a1fb` plus #1008's `0001` ([sparknet/series](sparknet/series)), as rc18's lock carries its
  sparknet series: `patch_head` `d50a1b4c`, `expected_tree` `dab8c457`, `patchset_sha256` `d58b3be5`. The configurations'
  `local.spark-ds41f.sparknet.tree` label is `dab8c457`.
- Every other source is 1107's.
- **Authors:** every PR is christopherowen's, and every patch's `From:` is Christopher Owen's.

| Patch | PR | Title | PR head | Source |
| --- | --- | --- | --- | --- |
| `0792-perf-worker-roce-proxy-and-rank0-fast-core-affinity.patch` | #1008 | perf(worker): roce proxy and rank0 fast-core affinity partition | `7f29f8934f00` | the PR's patch |
| `0793-perf-multimodal-prefix-cache-fastpath-under-mm-proce.patch` | #978 | perf(multimodal): prefix-cache host fastpath under --mm-processor-cache-gb 0 | `67792978d16d` | context-only rebase ([rebases](receipts/rebases/rebases.json)) |
| `0794-perf-spec_decode-sample-skip-dead-verification-rows-.patch` | #988 | experiments: add dead-row verification sampler skip and sparse top-k draft sampling | `cfb7069e6882` | its dead-row part, split at selection, retitled |
| sparknet `0001-perf-oneshot-pin-RoCE-proxy-thread-to-dedicated-fast.patch` | #1008 | perf(oneshot): pin RoCE proxy thread to dedicated fast core | `7f29f8934f00` | the PR's sparknet patch |

- **#978** is rebased with a context-only change: its two retry-map pops in `AsyncLLM.abort` go before #972's
  shielded abort block (`0771`). Every added and removed line is the PR's.
- **#988 is carried as its dead-row part.** `0794` keeps the dead-row skip (`mask_dead_rows`, the live-row mapping
  through the sampling kernels, `test_dead_rows.py`). It is retitled for that part: only the Subject changed, and one
  body line names the PR's subject. Its sparse top-k drafting is left out (below).

**Left out**, with each reason and source path in [receipts/selection.json](receipts/selection.json) (`left_out`):

| PR | Left out | Why |
| --- | --- | --- |
| #995 | its negative-slot guard in `copy_alloc` and `copy_apply`, as reworked in round 3 ([receipts/rework/pr-995-guard-r1](receipts/rework/pr-995-guard-r1)) | does nothing on this deployment (below) |
| #995 | the 31/63-draft ramp (`SPARK_DS41F_COPY_MAX_DRAFTS`, `greedy_rejection_kernel` widths above 32) | does nothing on this deployment: the recipe keeps 16 drafts (17 rows) |
| #995 | the cap decay and `copy_match`'s anchor pruning | separate variables; queued as their own experiments on the new base |
| #988 | sparse top-k DSpark drafting (`topk_*` kernels, `_sample_sequential_topk`, `apply_markov_bias_topk`) | does nothing on this deployment: it runs only with `dspark_draft_topk`, which neither the recipe nor the checkpoint sets |

**#995's guard is left out.** Its round-3 rework ([result.json](receipts/rework/pr-995-guard-r1/result.json)) is
bit-exact and costs nothing measurable at the copy-step bench (about 40 ns of device time a step at kernel level).
Its caller trace (`serving_effect.caller_trace` there) shows that no serving path passes a negative slot to
`copy_alloc` or `copy_apply`: both get `input_batch.idx_mapping`, real slots from `RequestState`, and only
`copy_match` receives `-1` rows, which it already guards. So on this deployment it guards a path nothing takes, and
under step 1's deployment-only rule it does not belong in a candidate. The rework's deliverables are kept: the patch,
overlay, `result.json`, `commands.txt`, `raw/` and `scripts/`.

### Stacking

All three apply cleanly with `git am` on candidate-54's head (`4e8133ff`, tree `5f021d9a`), in this order, to tree
`e39b835c`, the selection's tree for 55 ([stack.log](receipts/stack.log), section 2). `git diff --check` is clean.
Each applies alone on the base, and every ordered pair applies (section 4). #1008 and #978 share
`vllm/v1/engine/core.py` and edit different functions. Every shared file was read
([selection.json](receipts/selection.json), `semantic_review`); no semantic conflict was found. Candidates 55, 56 and
57 touch no common file.

## Cluster configurations

A cluster configuration for every shape the recipe serves, so the candidate can be run on any setup: `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-55/<file> …` with your site's node map. Each one extends `releases/spark-ds41f-1107/cluster.json` at its shape and sets only the image `spark-ds41f:candidate-55`, its tree labels and this lock; no PR changes the recipe. The `-dev` file adds the dev-mode collective RPC that the screen's engine grid needs. This candidate is measured at `tp4-ring` only; the other files are configurations, not part of its runs.

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
- No overlay control is written. #988's dead-row files are touched by no other patch here, so a `tp4-no988` control is
  possible if attribution needs one. #978 and #1008 share `core.py`, and #1008's proxy pinning is C in sparknet, so
  neither has a pure overlay control.

## Release tests

[release-tests.json](release-tests.json) is candidate-54's classification, plus two new test files, gated with their
reasons:
- `tests/v1/worker/test_roce_cpu_affinity.py` (#1008): the GB10 fast-core discovery, the disjoint role partition and
  the thread pinning.
- `tests/multimodal/test_prefix_cache_fastpath.py` (#978): the parallel hasher, the image header cache, the
  metadata-only processor cache and the strip and self-heal of P1-resident items.

#988's dead-row part changes `tests/v1/worker/test_dead_rows.py`, which is already gated. `scripts/release_tests.py
check` reports no problems on the prepared tree (sp8): 217 selected files, 155 gate and 62 excluded
([receipts/release-tests-check.txt](receipts/release-tests-check.txt)). The gate manifest is
[release-tests-gate-manifest.json](release-tests-gate-manifest.json) (156 files), and the bundle, written with main's
tooling, is [bundles/release-tests](bundles/release-tests).

## Runs

Each testbed opens its window with `scripts/lab.py`, after the hold and idle guards.

1. [build.json](build.json) (`candidate-55-image`), in a dgx window
   ([step 4](../../../docs/release-benchmark.md#4-build-and-release-tests-dgx-window)): build the image on dgx4, copy it
   along the ring, then one `kernel` job with this candidate's release-test bundle on dgx4. candidates 56 and 57
   build and run their release tests on sp8 (their READMEs, "The build").
2. [candidate-tp4.json](candidate-tp4.json) (`candidate-55-tp4`): the determinism gate at `tp4-ring`, first
   ([step 5](../../../docs/release-benchmark.md#5-gate-first)).
3. [screen-tp4.json](screen-tp4.json) (`candidate-55-screen-tp4`), two jobs
   ([step 6](../../../docs/release-benchmark.md#6-screen)):
   - the `screen` suite on `candidate-55` and `candidate-55-again`, with startup memory. **Reference:** candidate-54's
     `tp4-ring` screen, the accumulated base, at
     `experiments/2026-10-26-spark-ds41f-candidate/candidate-54/receipts/references/candidate-54-tp4-ring-screen`. The
     coordinator records it; until it is on `main`, the lab's dry run reports it as missing.
   - **the own marks that need the testbed**, on one boot of candidate-54 ([its tp4.json](../candidate-54/tp4.json),
     label `candidate-54-checks`) and then one of candidate-55 (`candidate-55-checks`):
     - [check_1008_affinity.py](check_1008_affinity.py): 16 streams for 120 s against rank 0's API server, and six
       snapshots of every rank's serving threads meanwhile: where the sparknet proxy, each worker's main thread, the
       engine core and the API server may run and where they ran. It records the client's chunk gaps under that
       load. The base shows the shared big-core regime; the candidate should show #1008's partition on every rank
       (`partitioned`).
     - [check_978_image_chat.py](check_978_image_chat.py): a two-turn image chat, twice, at temperature 0. The
       candidate's turns are compared with the base's, run just before it: text and prompt token count equal.
4. [single-stream-tp4.json](single-stream-tp4.json) (`candidate-55-single-stream-tp4`): `single-stream` on
   `candidate-55`, candidate-54 ([its tp4-dev.json](../candidate-54/tp4-dev.json)) and `candidate-55-again`, in one
   session, as candidate-51 ran it. candidate-54's `tp4-ring` screen ran the `screen` suite alone, so no single-stream
   reference is recorded at `tp4-ring`.
5. **Own marks** on sp8 (below), two boots each, ABBA against candidate-54.

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
| #1008 | perf (host) | Its `bench_roce_core_affinity.py`, its pytest (`test_roce_cpu_affinity.py`, `test_deepseek_v41_engram_host_placement.py`) and the sparknet unit tests on the candidate image on sp8, 2 boots, ABBA against candidate-54's image in one session. Against the base's placement (the recipe's shared big-core regime, `SPARKNET_ROCE_PROXY_CPU=big`, `--cpuset-cpus=5-9,15-19`) the PR shows the mean 116-collective step 0.106 and 0.084 ms -> 0.081 and 0.082 ms, and no rank-0 preemption spike (step p99 0.939 ms on one shared boot). Pass: step p99 under rank-0 API load lower beyond the interval, the mean not worse. On `tp4-ring`: the partition on every rank under load (`check_1008_affinity.py`), and the screen's 16-stream closed loop (paired decode, TPOT percentiles) and time to ready not worse | none |
| #978 | perf (host) | Its `bench_mm_prefix_fastpath.py` and `tests/multimodal/test_prefix_cache_fastpath.py` on the candidate image, 2 boots, ABBA against candidate-54's image: turn-2+ host processing of repeated images faster beyond the interval (the PR: 1302x1302 JPEG 18.1 -> 1.4 ms, 4 x 3840x2160 PNG 705 -> 40 ms), prompt token ids and mm hashes identical. On `tp4-ring`: the two-turn image chat returns the base's text at temperature 0 (`check_978_image_chat.py`). The screen sends no images, so there it shows only that no core metric is worse | none |
| #988 | perf (sampler) | Dead-row part only. Its `screen_dead_row_and_sparse_topk.py`, dead-row cells only, and `tests/v1/worker/test_dead_rows.py` on the candidate image, 2 boots, ABBA: `Sampler.apply_sampling_params` with `dead_row_cut` faster beyond the interval at 8 and 16 requests with top_p < 1, top_k, min_p or penalties; live rows bit-exact. Kernel time and CUDA-graph replay where the path is captured, both recorded. It acts only on verification steps with non-temperature sampling parameters, which no suite sends: the screen sees its startup and "no core metric worse" | none (live verification rows bit-exact; dead rows are never read) |

### Own-mark commands

On sp8, from a checkout of this commit, each command under `flock /tmp/spark-lab-gpu.lock`, receipts in
`receipts/own-marks/pr-<N>/` with a `result.json` ([step 7](../../../docs/release-benchmark.md#7-own-marks)):

- **Images.** The candidate is `spark-ds41f:candidate-55`, the base `spark-ds41f:candidate-54`. Stream both to sp8
  first (`ssh dgx1 docker save spark-ds41f:candidate-<N> | ssh sp8 docker load`).
- **ABBA in one session:** base, candidate, candidate, base. Two boots of each arm.
- **Judge on the path serving takes.** Record CUDA-graph replay or kernel time where the path is captured, and the
  eager timings beside them.
- The PRs' commands mounted their files over `spark-ds41f:1107`. On the candidate image the change is already in, so
  run the scripts from the PR heads without the mounts.

Per PR:

- **#1008.** Script: `experiments/2026-11-08-roce-proxy-fast-core-affinity/bench_roce_core_affinity.py` at `7f29f893`,
  `python3 /opt/exp/bench_roce_core_affinity.py --boot-label boot<k> --warmup-steps 60 --bench-steps 500 --output
  /opt/exp/receipts/<arm>-boot<k>.json`, with `--security-opt seccomp=unconfined --ulimit memlock=-1`. It times three
  placements in one process (unpinned, the shared big-core regime, the partition) under simulated rank-0 load. It
  imports `vllm.utils.cpu_affinity`, which candidate-54's image lacks: on the base image record the shared regime's
  rows with the base's own placement code, and keep the in-process comparison as the PR ran it. Tests, as the PR ran
  them: `pytest --noconftest -v tests/v1/worker/test_roce_cpu_affinity.py
  tests/kernels/test_deepseek_v41_engram_host_placement.py`, and `PYTHONPATH=<sparknet>/src python3 -m unittest
  discover -s <sparknet>/tests -v` with candidate-55's prepared sparknet tree (`dab8c457`) mounted at `<sparknet>`.
- **#978.** Script: `experiments/2026-11-08-multimodal-prefix-cache-fastpath/bench_mm_prefix_fastpath.py` at
  `67792978`, `python3 bench_mm_prefix_fastpath.py <arm> <out.json>`, on both images, ABBA. Tests: the PR's pytest
  command (its README) on the candidate image: `test_prefix_cache_fastpath.py`, `test_hasher.py`,
  `media/test_image.py`, `test_cache.py` and `test_deepseek_v4_1_image_span.py`, with the PR's `-k` exclusions.
- **#988.** Script: `experiments/2026-11-08-sampler-dead-row-skip-and-sparse-topk/screen_dead_row_and_sparse_topk.py`
  at `cfb7069e`, `python3 screen_dead_row_and_sparse_topk.py --boot <k> --out <file>`. Judge its dead-row cells; its
  top-k cells exercise the part left out. Tests: `tests/v1/worker/test_dead_rows.py`,
  `tests/v1/worker/test_gpu_sampling_states_gather.py`, `tests/kernels/test_deepseek_v41_copy_logits.py`,
  `tests/v1/worker/test_prepare_kernels.py`.

**Rework (owner, 2026-11-09).** A PR that misses its mark goes to a sub-agent for one focused rework, with the miss,
the receipts and the mark. A second rework is allowed when the first shows promise. A PR that still misses is dropped
from this release, and its record stays here. A reworked PR joins the next candidate with its own source identity and
fresh qualification.

## Determinism gate: declared output changes

None. Every PR here declares bit-identical output, so the gate is a pure identity check against candidate-54's TP4
hashes: `18d61961`/`d6568649` with logprobs (text `0a2dacdd`/`12fd8257`). candidate-54's `tp4-ring` gate matches
candidate-52's, the first record of #1030's text.

## Selection

One of candidates 55, 56 and 57, tested at the same time, one per testbed. The PRs that pass are carried into all
later candidates. The round's selection receipts are here, in [receipts](receipts):

- [pr-triage-2026-11-10.md](receipts/pr-triage-2026-11-10.md): every open PR on 2026-11-10, against candidate-54.
- [selection-proposal.json](receipts/selection-proposal.json): the order, SHA-256s, marks, placement and `left_out`.
- [pairwise-conflicts.json](receipts/pairwise-conflicts.json) and [pr-patch-targets.json](receipts/pr-patch-targets.json).
- [rebases](receipts/rebases): the patches that differ from the PR's own.
- [scripts-as-run](receipts/scripts-as-run).
- [rework/pr-995-guard-r1](receipts/rework/pr-995-guard-r1): #995's guard rework, left out.

## Status

Built on dgx4 (`spark-ds41f:candidate-55`, image ID `41d961dc`). **It does not boot**, and its release tests fail
(13 tests); both come from #978 (0793), which is in rework ([receipts/tp4-ring](receipts/tp4-ring/README.md)).
Without #978 ([receipts/tp4-ring-no978](receipts/tp4-ring-no978/README.md)): the gate keeps candidate-54's TP4
hashes, and the screen and `single-stream` are the same as candidate-54's. **#988's dead-row part passes.**
**#1008 passes** ([receipts/mark-1008](receipts/mark-1008/README.md)): rank 0's step p99 is lower beyond the interval and
nothing is worse; its ring-level p99 reads the same.

## #978 round 1, and candidate-55 without #978

**#978's round 1** ([receipts/rework/pr-978-r1](receipts/rework/pr-978-r1/result.json)) fixes the three defects in 0793:
- the boot failure: `Callable` was never imported in `async_llm.py` (not a rebase loss; the PR's base lacks it too);
- the 13 release-test failures: text-only requests read multimodal state (`renderer.mm_processor_cache`,
  `_pending_mm_retries`); the fast path is now decided once and text-only requests never reach it;
- a latent engine crash: `get_computed_blocks` returns three values, and the stripped-item check unpacked two.
The two release-gate files pass (62), #978's tests pass on two boots, and the wider engine and serve tests fail only
where candidate-54's fail (environmental).

**It misses its mark on the serving path.** The PR's bench meets it (7.5-23x on all 8 workloads), but only because the
bench builds a processing context that carries `vllm_config`. The server's context does not, so the metadata cache
never engages: items are never lazy and the full image payload crosses IPC every turn. Through a real `AsyncLLM`
frontend against candidate-54, 1302x1302 JPEG requests are 57% slower with one image and 35% with four, and PNG 5-7%
faster; the original #978 behaves the same. #978 goes to its second and last rework: wire the setting through to the
server's processor, add a schedule-time retryable miss for stripped items, find the JPEG slowdown, and measure on the
serving path with `check_978_image_chat.py`.

**candidate-55 without #978.** So #1008 and #988's dead-row part are measured now, on the candidate-55 image with
0793 reverted: [no978/overlay](no978/overlay) holds the 8 non-test files 0793 changes, at their reverted versions (tree
3f534a18), mounted by [tp4-no978.json](tp4-no978.json). Its gate, screen (with #1008's affinity check) and
single-stream run as [candidate-tp4-no978.json](candidate-tp4-no978.json),
[screen-tp4-no978.json](screen-tp4-no978.json) and [single-stream-tp4-no978.json](single-stream-tp4-no978.json).

## #978 round 2

[receipts/rework/pr-978-r2](receipts/rework/pr-978-r2/result.json), its second and last rework.
- **The fast path engages in the server.** A new `mm_prefix_metadata_cache` flag on the processing context, which the
  renderer sets only with no processor cache, prefix caching on, one engine per frontend and no KV or EC transfer.
- **The JPEG slowdown was #978's own:** lazy loading moved the decode into the processor's thread (about 3x the page
  faults), and multithreaded blake3 woke that thread on the slow cores. The header cache now runs only on the fast
  path, and `hasher.py` is candidate-54's again.
- **The four risks are closed:** a schedule-time retryable miss for stripped items, every sampled child keeps its
  retry materializers, identifier tracking is bounded (LRU 4096), and retries run off the output handler.
- **Its mark, round 1's real-`AsyncLLM` bench on sp8, ABBA against candidate-54, two boots:** all 12 image cells are
  faster beyond the interval: turn 2 onward 4.2-9.5x, turn 1 2-15% faster. Prompt token ids and mm hashes are
  candidate-54's, and text-only requests are the same. One caveat: under the recipe's process environment (not part
  of the mark), 1920 PNG turn 1 was slower on one of two boots.
- **On the testbed:** [validate-978r2-tp4.json](validate-978r2-tp4.json) boots candidate-55 with the round-2 files
  (`tp4-978r2.json`, from [978r2/overlay](978r2/overlay)): the gate, then `check_978_image_chat.py` against
  candidate-54 under #1008's real thread placement.

**#978 round 2 on the testbed** ([receipts/tp4-ring-978r2](receipts/tp4-ring-978r2/README.md)): the live image chat
passes, but the gate's `again` scenario (prompts re-run on prefix-cache hits) differs in one request in both runs,
where candidate-55 without #978 is identical. **#978 is dropped from this release** after its second rework.

**Verdict for candidate-55:** #1008 and #988's dead-row part pass and join the next merged base; #995's guard is left
out; #978 is dropped.

**Correction (2026-11-10).** The `again` difference that dropped #978 also occurs without it (candidate-51's gate,
one run in about 500), so it is not shown to be #978's ([receipts/tp4-ring-978r2](receipts/tp4-ring-978r2/README.md)).
#978's round 3 (owner exception; [receipts/rework/pr-978-r3](receipts/rework/pr-978-r3/result.json)) puts text-only
requests on candidate-54's own statements and fixes a hidden admission-control test regression from round 2; its mark
holds (image turns 2 onward 4.25-9.59x faster, no text cell slower). It is judged in the next round on candidate-58,
with a two-boot gate ([validate-978r3-tp4.json](validate-978r3-tp4.json) is the candidate-55 form).
