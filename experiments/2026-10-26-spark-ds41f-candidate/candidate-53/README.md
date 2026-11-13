# candidate-53

Unpromoted release candidate: candidate-50 without #1009 plus 5 PRs: every PR that edits `model_runner.py`
(one candidate must hold them all), with the TP3-only display carve-out and the copy-drafter pair. It is tested
at `tp3-switched` (sp5..sp7, head sp5).

## Source

- vLLM: candidate-50's series without `0778-perf-dsv41-sp-prefill-local-attention-front.patch` (#1009):
  1107's `0001..0761` (641 active) plus candidate-50's `0762..0777`, `0779` and `0780`, 659 patches listed
  by relative path to their files. Then the patches below, renumbered from `0781` in the selection's
  order ([vllm/series](vllm/series)). Each is copied from its PR's head, or is the selection's rebase of it.
  The provenance and SHA-256 of each are in [receipts/selection.json](receipts/selection.json), and the stack
  log is [receipts/stack.log](receipts/stack.log).
- `patch_count` 664, `patch_head` `62df12dc`, `expected_tree` `c0e25ff1`,
  `patchset_sha256` `f7f2d13c`, recorded by `bin/spark build prepare --lock`
  ([source.json](source.json), [upstreams.lock.json](upstreams.lock.json)).
- Every other source is 1107's.

| Patch | PR | Title | PR head | Source |
| --- | --- | --- | --- | --- |
| `0781-perf-cudagraph-target-full-replay-and-post-gate-fork.patch` | #960 | Target FULL CUDA graph metadata/CED replay and MoE post-gate fork | `6cfd0bd33413` | the PR's patch |
| `0782-perf-spec-decode-copy-drafter-survival-and-tile-cap.patch` | #971 | perf(spec-decode): cumulative survival in copy_apply, single-req 16-row tile cap, and wide-step dead-row mask | `9ef22e1fcdf4` | context-only rebase ([rebases](../candidate-51/receipts/rebases/rebases.json)) |
| `0783-perf-spec-decode-copy-drafter-long-context-scan.patch` | #977 | perf(spec_decode): bound CopyDrafter long-context history scan in copy_match | `4e0244deb7d5` | the PR's patch |
| `0784-perf-worker-reclaim-uma-heap-and-allocator-slack-acr.patch` | #984 | perf(worker): reclaim UMA host heap and CUDA allocator slack across lifecycle | `737aec88ed62` | the PR's patch |
| `0785-perf-worker-compact-packed-vocab-tables-and-expand-d.patch` | #998 | perf(worker): compact packed vocab tables and expand display carve-out weights | `e89ab5f9a31b` | the PR's patch |

**The base leaves out #1009.** It causes candidate-50's `tp4-ring` boot failure
([candidate-50/receipts/tp4-boot-failure](../candidate-50/receipts/tp4-boot-failure)) and is in rework;
when its rework passes it joins a later base. Without `0778` the base's tree is `14c5a2dc`, the same as
candidate-50's prepared tree with `0778` reverted, which the selection used ([stack.log](receipts/stack.log),
section 1).

### Stacking

All five apply cleanly with `git am` on the base, in this order ([stack.log](receipts/stack.log), section 2),
and `git diff --check` is clean. They share non-test files, and the selection found no conflicting pair
([pairwise-conflicts.json](../candidate-51/receipts/pairwise-conflicts.json)):
- `vllm/v1/worker/gpu/model_runner.py`: #960, #971, #984 and #998;
- `vllm/v1/worker/gpu/cudagraph_utils.py`: #960 and #984;
- `vllm/v1/worker/gpu/spec_decode/tilelang_copy.py`: #971 and #977 (they apply together);
- `tests/v1/spec_decode/test_copy_drafts.py`: #971 and #977.

`0782` (#971) is rebased with a context-only change: its test appends where a base patch appended one, and
every added and removed line is the PR's ([rebases.json](../candidate-51/receipts/rebases/rebases.json)).
Candidates 51, 52 and 53 touch no common file, test files included. Every patch the round left out, with its
reason, is in [receipts/selection.json](receipts/selection.json) (`left_out`).

## Cluster configurations

A cluster configuration for every shape the recipe serves, so the candidate can be run on any setup: `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-53/<file> …` with your site's node map. Each one extends `releases/spark-ds41f-1107/cluster.json` at its shape and sets only the image `spark-ds41f:candidate-53`, its tree labels and this lock; no PR changes the recipe. The `-dev` file adds the dev-mode collective RPC that the screen's engine grid needs. This candidate is measured at `tp3-switched` only; the other files are configurations, not part of its runs.

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
- No overlay control: #960, #971, #984 and #998 all edit `model_runner.py`, and #977 shares `tilelang_copy.py`
  with #971, so no PR here can be removed by mounting files. Attribution here needs per-PR image variants.
  #984 and #998 both move the startup floor; their own benches separate them.

## Release tests

[release-tests.json](release-tests.json) is candidate-50's classification, plus #984's two changed
`tests/utils_` files, each with its reason:
- `tests/utils_/test_gc_utils.py`, **gate**: the GC debug configuration and the host-heap trim before freezing
  the GC around CUDA graph capture. It imports only `vllm.utils.gc_utils`.
- `tests/utils_/test_mem_utils.py`, **excluded (environment)**: it imports `vllm_test_utils`
  (`tests/vllm_test_utils`) at module level, and the serving image does not install that package (the
  Dockerfile and the image history name it nowhere; #984's own run put it on `PYTHONPATH`). #984's four tests in
  it run in its own mark, with the package on `PYTHONPATH`.

The other test files these PRs change are already gated. The classification also carries main's
release-test change that landed after candidate-50 (#1038): the bundle resolves
upstream tests' stand-in hub models to the served checkpoint, and two `tests/v1/core/test_scheduler.py` tests
that then run are deselected with their reason, as in `config/release-tests.json`.

`scripts/release_tests.py check` reports no problems on dgx4's prepared tree: 217 selected files, 154 gate
and 63 excluded ([receipts/release-tests-check.txt](receipts/release-tests-check.txt)). The gate manifest is [release-tests-gate-manifest.json](release-tests-gate-manifest.json) (155 files), and the bundle is
[bundles/release-tests](bundles/release-tests).

## Runs

Each testbed opens its window with `scripts/lab.py`, after the hold and idle guards.

1. [build.json](build.json) (`candidate-53-image`), in the round's one dgx window
   ([release-benchmark](../../../docs/release-benchmark.md#4-build-and-release-tests-dgx-window), step 4):
   build the image on dgx4 and copy it along the ring. Queue it after candidate-51's and candidate-52's
   `build.json`: it ends with **the round's one `kernel` job**, which runs the three candidates' release-test
   bundles at once, each on its own node (candidate-51's on dgx4, candidate-52's on dgx3, this one's on dgx2).
   The image is then streamed to sp5 (`ssh dgx1 docker save spark-ds41f:candidate-53 | ssh sp5 docker load`) and
   distributed from there with
   `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-53/tp3-switched.json build distribute --apply`.
   Every rank shows the same image ID.
2. [candidate-tp3-switched.json](candidate-tp3-switched.json) (`candidate-53-tp3-switched`): the determinism gate
   at `tp3-switched`, first ([step 5](../../../docs/release-benchmark.md#5-gate-first)).
3. [screen-tp3-switched.json](screen-tp3-switched.json) (`candidate-53-screen-tp3-switched`): the `screen-tp3`
   suite plus the companions this round's marks need, on `candidate-53` and `candidate-53-again`, with startup
   memory ([step 6](../../../docs/release-benchmark.md#6-screen)):
   - `single-stream` for #971's paired decode at 1 stream;
   - `sampled` for the sampled realizations #971 and #977 may change;
   - the stall steps alone ([../candidate-50/stall-tp3-suite.json](../candidate-50/stall-tp3-suite.json), release-tp3's three stall steps) for #984's short-request TTFT during a long prefill. It runs as a second measure job, against candidate-50's run of the same suite in the sp5 window (`stall-tp3-switched.json`). The full `release-tp3` suite halts at its model step on sp5, at the 4 GiB bench memory floor.
   - **Reference:** candidate-50's runs at `tp3-switched`, the accumulated base. They are recorded at
     `experiments/2026-10-26-spark-ds41f-candidate/candidate-50/receipts/references/candidate-50-no1009-tp3-switched-screen`,
     with the companions beside it as `-single-stream` and `-sampled`.
     candidate-50's [screen-tp3-switched.json](../candidate-50/screen-tp3-switched.json) and
     [companions-tp3-switched.json](../candidate-50/companions-tp3-switched.json) record them.
   - **The references are pending** until the coordinator commits them; until then the lab's dry run reports them missing.
4. **Own marks** on sp8 (below), two boots each, ABBA against the base.

## Release tests: #998 fails; boot: #960 fails; measured without both

**Release tests** (dgx2, 2026-11-09): they passed on candidates 51 and 52 and failed on candidate-53, in two files, both #998's:
- `tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py` crashes with a segmentation fault;
- `tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py::test_process_weights_replaces_w13_before_preparing_w2` fails with `'_Plan' object has no attribute 'max_rows'`, at `fused_moe/tilelang.py:224`, a line #998 adds.

With candidate-50's copies of `fused_moe/tilelang.py`, `packed_bf16.py` and `embedding.py` mounted over the candidate-53 image, both files pass (104 passed).

**Boot at tp3-switched** (sp5, 2026-11-09): it fails in the warm-up decode step on every rank, with `'KVConnector' object has no attribute 'has_connector'` at `model_runner.py:3204`.
- #960 (0781) adds that call. This deployment's `KVConnector` has no such method; serving uses `NO_OP_KV_CONNECTOR`.
- #960's own mark on one GPU never ran a decode step through `execute_model`.

**Measured without both:**
- [tp3-switched-no960-no998.json](tp3-switched-no960-no998.json) mounts the 10 non-test files that differ when 0781 and 0785 are reverted from candidate-53's tree (`62df12dc` to tree `94015950`), from [no960-no998/overlay](no960-no998/overlay).
- Shared files such as `model_runner.py` keep the other PRs' changes. So the gate and screen measure #971 (as built), #977 and #984.
- #960 and #998 are each in rework round 1.

**Recorded error.** The first control, `no998` (#1048), mounted only those three files. #998 also changes `qwen3_dspark.py`, `display_carveout.py` and `model_runner.py`, so that control left part of #998 in place. It never measured anything, because the boot failed first on #960. The release-test result above stands: those tests touch only the three files. The `no998` control is removed.

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
- paired decode in the 8-stream closed loop;
- prefill at 32K and 256K;
- the determinism gate (temperature-0 output hashes equal to the 1017 checkpoint's, unless a PR declares a change below);
- startup (lowest `MemAvailable`, time to ready, no JIT after readiness) and steady memory (KV capacity, swap, OOMs);
- zero failed requests, and no silent fallback in any rank's boot log;
- the release tests (`check` reports no problems, and the bundle exits 0).

A kernel PR's own mark is measured on the candidate image, the integrated stack, by the PR's own
benchmark: two boots on sp8. When a core metric is worse, the PRs whose path can move it are
removed one at a time until the cause is found; here that needs per-PR image variants (no overlay control).

| PR | Kind | Own mark | Declared output change |
| --- | --- | --- | --- |
| #960 | perf | Its `screen_cudagraph_wins.py` on the image, 2 boots, ABBA against the base: per-step pre-launch metadata and CED staging on FULL decode steps lower beyond the interval (the PR: 438-451 us to 33-41 us from (1,1) to (16,64)), bit-exact; the engine grid and paired decode not worse. Its receipts predate its head commit (`6cfd0bd3` dropped the post-gate MoE fork), so they must be re-run on the current patch | none |
| #971 | perf (policy) | Its `screen_copy_drafter_smooth.py` 58-layer step screen on the image, 2 boots: the single-stream initial copy step and the post-reject shrunk step faster beyond the interval at TP3; paired decode at 1 stream better and at 8 streams not worse; temperature-0 text unchanged | drafting and verification rows (copy continuation yield, single-request 16-row cap, wide-step padding mask); temperature-0 text must not change; sampled realizations may change |
| #977 | perf (policy) | Its `bench_copy_match_long_context.py` on the image, 2 boots: `copy_match` at 64K..1M faster beyond the interval, unchanged at <=32K (still a dense scan); copy drafts identical at <=32K | copy drafts at contexts past 32K (the bounded scan can miss a far match); temperature-0 text must not change; sampled realizations may change |
| #984 | perf (memory) | Its `bench_uma_allocator_hygiene.py` on the image, 2 boots, and the screen's startup record: each rank's lowest `MemAvailable` at startup higher than the base's beyond the interval, time to ready not worse, no JIT after readiness; short-request TTFT during a long prefill not worse (its post-prefill cache flush runs on the serving path); KV capacity unchanged (the recipe fixes `--kv-cache-memory-bytes`) | none |
| #998 | perf (memory) | Its `bench_carveout_expansion.py` on the image, 2 boots (the PR commits no receipts: run it fresh): regular UMA saved per rank (the PR: 1,263 MiB at TP3), the moved DSpark MoE layers not slower (the PR: 8.5-12.3% faster), `torch.equal`; the startup floor higher. It acts only where `SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1`, the TP3 shapes | none |

### Own-mark commands

On sp8, from a checkout of this commit, each command under `flock /tmp/spark-lab-gpu.lock`, receipts in
`receipts/own-marks/pr-<N>/` with a `result.json`
([step 7](../../../docs/release-benchmark.md#7-own-marks)):

- **Images.** The candidate is `spark-ds41f:candidate-53`. The base is `spark-ds41f:candidate-50` with
  candidate-50's four `no1009/overlay` files mounted over it, as [candidate-50's tp4-no1009.json](../candidate-50/tp4-no1009.json)
  mounts them. Stream both to sp8 first (`ssh dgx1 docker save spark-ds41f:candidate-<N> | ssh sp8 docker load`,
  for candidate-53 and candidate-50).
- **ABBA in one session:** base, candidate, candidate, base. Two boots of each arm. The base arm is not a
  release record, so measuring it here is allowed.
- The PRs' commands mounted their changed files over `spark-ds41f:1107`. On the candidate image the change is
  already in, so run the bench scripts from the PR heads without those mounts.
- Several scripts compare in-process (a baseline and a candidate path in one process). Keep that comparison as
  recorded, and add the image A/B: the same script on the base image and on the candidate image.
- **TP3 settings.** sp8 is one GPU: run the benches at the TP3 shard shapes the PRs used, and with the TP3
  shape's environment where a PR's path depends on it (#998: `SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1`).

Per PR:

- **#960.** Script: `experiments/2026-11-08-cudagraph-decode-optimizations/screen_cudagraph_wins.py` at
  `6cfd0bd3`, `python3 screen_cudagraph_wins.py --boot <k>`. Its README gives no run command; it runs a
  baseline and a candidate path in-process. **Its receipts (`receipts/boot-1.json`, `boot-2.json`) predate its
  head**, which dropped the post-gate MoE fork, so the MoE fork rows in them do not describe this patch: run it
  fresh on both images. Tests: `tests/v1/attention/test_b12x_v41_ced_metadata.py`,
  `tests/v1/cudagraph/test_cudagraph_manager.py`.
- **#971.** Script: `experiments/2026-11-08-copy-drafter-survival-and-tile-cap/screen_copy_drafter_smooth.py` at
  `9ef22e1f`, `python3 /tmp/screen_copy_drafter_smooth.py --boot boot<k>` with `-w /opt/spark-ds41f/candidate/vllm`.
  Tests, `-v --noconftest`: `tests/v1/spec_decode/test_copy_drafts.py`, `tests/v1/core/test_copy_drafts_scheduler.py`,
  `tests/v1/spec_decode/test_adaptive_verification.py`, `tests/v1/worker/test_prepare_kernels.py`. Its
  1-stream paired decode is not a screen row (see Runs).
- **#977.** Script: `experiments/2026-11-08-copy-drafter-long-context-scan/bench_copy_match_long_context.py` at
  `4e0244de`, `python3 /exp/bench_copy_match_long_context.py --label <base|candidate> --out /exp/receipts/<label>-boot<k>.jsonl`.
  The PR already ran it as an image A/B (plain 1107 against 1107 with its file mounted): here the two arms are
  the base and candidate images. Then, in the candidate image,
  `pytest -o addopts= --noconftest -v tests/v1/spec_decode/test_copy_drafts.py` and
  `python3 benchmarks/kernels/benchmark_copy_drafts.py --gate --cases copy`.
- **#984.** Script: `experiments/2026-11-08-uma-allocator-hygiene/bench_uma_allocator_hygiene.py` at `737aec88`,
  `python3 /tmp/bench_uma_allocator_hygiene.py --boot-label boot<k>` with `--security-opt seccomp=unconfined
  --ulimit memlock=-1:-1` and the Hugging Face cache mounted read-only. Its `--mode both` default runs a baseline
  and a candidate lifecycle in-process. Tests, as the PR ran them: `pytest -v -k "not test_memory_profiling"`
  on `tests/utils_/test_mem_utils.py`, `tests/utils_/test_gc_utils.py`, `tests/v1/worker/test_gpu_model_runner_v2.py`,
  `tests/v1/worker/test_gpu_worker.py` and the two `tests/model_executor/model_loader/test_direct_loader*.py`,
  with `PYTHONPATH` including `tests/vllm_test_utils`. The PR also put `/tmp/stub_pkgs` on `PYTHONPATH`; those
  stubs are not committed, so record what the run needs in their place. The startup floor and time to ready come
  from the screen's startup record; the TTFT during a long prefill is not a screen row (see Runs).
- **#998.** Script: `experiments/2026-11-08-display-carveout-weight-expansion/bench_carveout_expansion.py` at
  `e89ab5f9` (the README runs it as `/tmp/run_carveout_bench.py <boot>`), `python3 bench_carveout_expansion.py <k>`
  with `--device /dev/dri/card0`, `-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and a TileLang cache
  mount. It runs its baseline and candidate layouts in-process. **The PR commits no receipts**, so both boots
  are new. Tests, with the release bundle's `tblib` stub: `tests/v1/worker/test_display_carveout.py`,
  `tests/models/test_dspark_mla.py`, `tests/model_executor/model_loader/test_direct_loader.py`,
  `tests/v1/worker/test_gpu_model_runner_v2.py`, `tests/kernels/test_deepseek_v41_tilelang_embedding.py`.

**Rework (owner, 2026-11-09).** A PR that misses its mark goes to a sub-agent for one focused
rework, with the miss, the receipts and the mark. A second rework is allowed when the first
shows promise. A PR that still misses is dropped from this release, and its record stays here.
A reworked PR joins the next candidate with its own source identity and fresh qualification.

## Determinism gate: declared output changes

Temperature-0 output hashes must equal the 1017 checkpoint's, except where a PR declares a change:

| PR | Declared output change |
| --- | --- |
| #971 | drafting and verification rows; temperature-0 text must not change; sampled realizations may change |
| #977 | copy drafts at contexts past 32K; temperature-0 text must not change; sampled realizations may change |

The gate decodes greedily, so its hashes must still equal the 1017 checkpoint's. #960, #984 and #998 declare none.

## Selection

One of candidates 51, 52 and 53, tested at the same time, one per testbed. The PRs that pass are carried into
all later candidates. The selection receipts are in [candidate-51/receipts](../candidate-51/receipts).

## #984 round 1

The rework's record is [receipts/rework/pr-984-r1](receipts/rework/pr-984-r1/result.json). Its patch is
`0784-perf-worker-trim-the-host-heap-once-when-startup-freezes.patch`.
- **Where round 0's 77 s went:** round 0 added `gc.collect()` and a heap trim to the per-module UMA release that
  `process_weights_after_loading()` calls. At TP3 it fires 277 times, and each full collection of the loaded heap
  takes about 0.25 s. The trims cost almost nothing.
- **What round 1 keeps:** `trim_host_heap()` (mimalloc's `mi_collect(true)` and glibc's `malloc_trim(0)`), called
  once in `freeze_gc_heap()` at the end of startup. It drops every mid-startup collection and the post-prefill
  flush, which blocked the host for 32–35 ms on the decode path.
- **sp8, one rank at tp3-switched rank shapes, same session:**
  - Time to ready is 88.4/89.3 s, against the base's 89.0/87.8 s (round 0: 157.7/156.1 s).
  - The trim returns 66 MiB in 5 ms.
  - The `MemAvailable` floor difference is inside the boot spread (about 0.15 GiB).
- **Deciding check:** [attribute-984r1-tp3-switched.json](attribute-984r1-tp3-switched.json) runs ABBA on this image
  at `tp3-switched` against `tp3-switched-no960-no998-no984.json`, with startup memory and the stall-3 suite.
  [984r1/overlay](984r1/overlay) differs from the control's overlay in three files: `gc_utils.py`, `mem_utils.py` and
  `gpu_worker.py`.
- **Its mark stays #984's:** the startup floor higher beyond the interval, time to ready and stall-3 not worse, KV
  capacity unchanged. If the floor is the same, #984 has no metric left and is dropped.
- **Result (sp5 window, 2026-11-10, [receipts/tp3-switched](receipts/tp3-switched/README.md), item 3):**
  - Time to ready: 126.8/127.8 s, against the control's 124.4/124.9 s. Round 0's 77 s is gone.
  - Startup `MemAvailable` floor: +0.14, +0.01 and +0.02 GiB at sp5, sp6 and sp7. sp6 and sp7 are inside their boot
    spread; sp5's control repeat had 6.1 GiB of page cache at its low point, against 3.9 GiB on the other boots.
  - Stall-3 and prefill: the same. Round 0's 2.37% stall-3 loss is gone.
- **Verdict: #984 misses and is dropped from this release.** The floor does not rise beyond the interval, and what
  remains of the PR is a 66 MiB trim at the end of startup, so a second rework has nothing to build on.

## Status

Built (`spark-ds41f:candidate-53`). The full candidate does not boot (#960; its round 1 fixes it). The results
without #960 and #998 are in [receipts/tp3-switched](receipts/tp3-switched/README.md). Its PRs, reworked where needed
and without #984, are in [candidate-54](../candidate-54/README.md). #984 is dropped after its round 1 (above).
