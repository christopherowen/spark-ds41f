# candidate-38

Base: `candidate-37` (`0001..0676`, vLLM patch head `5b7cfa37f`, tree `af0fb152`; `sparknet` `1fd6a1f`).

Candidate-38 carries the selected CopyDrafter solution: shared verification budgets
(`0677`) and removal of the unused per-slot limit machinery (`0678`). The host-wait
and device-clamp implementations from #855/#856/#859 are superseded; neither is in
this series. Their experiment records remain historical evidence. #860's two
patches, #848/#853/#854's runtime changes and #863's test repairs are carried forward
here; #842's default-enablement change remains separate.

This is an unpromoted candidate. The final image's release-test gate passes, and
focused loop and CopyDrafter reproductions completed on TP3 switched, each with
a fresh boot and repeat boot. The
[TP3 triangle qualification](../../2026-11-03-candidate-38-tp3-qualification/README.md)
now records two functional and two full performance boots per recipe, plus two
standard tail-latency confirmation boots. Greedy model outputs and recorded
probabilities match 1030; headline throughput improves. Tail latency and small
copy-recipe costs remain under review, so this is not a no-regression promotion.
Broader topology and long-session qualification remains separate. The [fault investigation](../../2026-11-03-copy-fault-forensics/README.md)
demonstrates the unequal-layout defect and a stale-receive-buffer corruption
mechanism. Its last host preparation records do not prove which GPU forward
produced the original NaN.

## Loop containment

The selected loop work is the server-side containment path in `0656`, `0674`
and `0679`. These are successive changes to one implementation. Keep the tested
candidate-38 source and its regression tests as the basis for qualification.

| State | Action |
| --- | --- |
| Repetition inside reasoning | Force reasoning to end once through the existing thinking-budget path; stop if the end has not landed within 64 output tokens. |
| Repetition after a reasoning-loop intervention | Stop the request with the repetition finish reason. |
| Repetition in an otherwise healthy visible answer | Require sustained degeneration before stopping; retain the persistence and diversity checks that protect arrays, tables, repeated text and tool payloads. |

The table describes default detection. Explicit `repetition_detection` settings
retain their existing semantics, including the opt-out used for controlled
comparisons. Detection reads committed output tokens and follows the reasoning
parser's marker transitions. Keep the detection/intervention/completion receipts.
A forced reasoning end is containment; successful task completion still needs
its own assessment.

Preserve the requested effort and existing alias semantics. The separate
[`#862`](https://github.com/christopherowen/spark-ds41f/pull/862) proposal to map
`max`/`ultra` to 85 is retired from the candidate plan. It was never included here.
Its branch and experiment evidence remain historical records; neither an 85
ceiling nor another client retry or rewrite is part of this containment path.

The [two-boot reproduction](../../2026-11-03-candidate-38-reproduction/README.md#loop-result)
shows seeded sustainment at efforts 100 and 75 with detection off, and containment
with the default detector. The clean fixture's tokens are unchanged by the
detector. This supports the containment behavior without establishing the cause
of spontaneous onset or its absence in longer sessions.

Remaining work is focused:

1. Check the reported long-context canary in
   [`#864`](https://github.com/christopherowen/spark-ds41f/issues/864) and a long
   compacted tool session on this candidate, preserving exact requests, model
   identity, sampling settings and detector receipts. The report's 83K-token
   canary is from an older rc10 build; it is not a candidate-38 reproduction.
   Separate onset, containment, and the usefulness of the resulting answer or
   tool call. GPU runs still need an authorized window and a repeat boot.
2. Keep the existing legitimate-repetition and marker-transition regressions,
   plus the focused seeded/default/off/clean checks, when changing this path.
   The answer safeguards address observed false positives and are not cleanup
   candidates merely because the policy can be described simply.
3. Review the completed TP3 triangle qualification's latency findings and
   complete the remaining topology and long-session gates against recorded
   references. The diagnostic runs below are not a no-regression performance verdict.

The onset/root-cause question stays open under #864. Broader effort, precision,
activation or DSpark sweeps are investigations to justify from a concrete
reproduction, not additional fixes required by the containment design. Earlier
experiments and unpublished work remain preserved; they are not deployment inputs.

## Patch series

Added patches (`0677..0687`). Each is the `git format-patch` commit its pull request made on
`5b7cfa37f` (`0677..0681`, the candidate's changes) or on `753354cf4`, 0681's head
(`0682..0687`, the release gate's test fixes), taken from the pull request's branch byte for
byte apart from the re-sequenced `[PATCH n/687]` subject prefix (`make_series.py`), and applied
in this order:

- `0677-fix-copy-drafts-budget-verification-from-what-every-.patch` (`#860`, vLLM `72abb31ff`):
  the fix for #852. The adaptive verification budget no longer reads CopyDrafter's per-rank host
  mirror of the slot draft limits (one argument in `model_runner.py`), so every rank budgets the
  same step from inputs the ranks hold alike. Measured on sp5–sp7 with the patch's file mounted
  over candidate-37's image: #852 reproduced on candidate-37 (`route_partials: gate logits
  contain nan or inf` in the single-stream suite after the copy probe, min match 8, dead rows
  off) and the fix ran the same sequence clean with the ranks' layout CRCs equal; paired
  single-stream −0.08% ±0.09% against candidate-37 and tool tokens per step 6.232 against 6.083
  (`lab-cost.json`'s run, as the session that queued it reports it; the experiment's README at
  `0c9ae9fa` still lists that run as queued). Unit tests in the image: 79 passed, 1 skipped.
- `0678-cleanup-copy-drafts-remove-the-per-slot-draft-limits.patch` (`#860`, vLLM `88982f34e`):
  removes what 0677 left unread: the host mirror, its staging ring and event,
  `sync_slot_draft_limits`, the limits' copy in `attach_copy_draft_alloc`, the device buffer and
  the manager's `slot_draft_limits` argument. Unit-tested only (the same 79 passed, 1 skipped
  with both patches); not performance-measured and in no arm of #860. The final
  combined source is covered by this candidate's own unit gate below; its serving
  cost and behavior must be qualified with the rest of the candidate.
- `0679-feat-v1-act-on-a-loop-by-where-it-is-and-follow-the-.patch` (`#848`, vLLM `050e8b1cc`, v2):
  `check_stop` acts on a detected loop by where it is. Inside reasoning it ends the reasoning (a
  forced end that has not landed within 64 tokens stops the request); in the answer after a
  reasoning loop it stops at once; otherwise token soup (one token over 60% of the last 512)
  stops after 256 tokens and any other loop only after 2,048 tokens with fewer than one distinct
  token in ten. Markers follow `DeepSeekV4Parser`. Measured: 40 unit tests and the offline replays
  of the known triggers on candidate-37's image (sp8); the sp5–sp7 probe soak passed 15 of 15
  (v1 had cut two legitimate periodic answers, which v2's second horizon lets run).
- `0680-perf-dspark-capture-context-graphs-at-the-target-s-d.patch` (`#853`, vLLM `5340c9e2f`):
  `DSparkContextCudaGraphs` captures the row counts of the target's own FULL decode graphs (23
  graphs at TP3 instead of 7, 35 at TP4 instead of 8), so a full-depth decode step pads the
  context projection no further than the target does. Measured on one GB10 (dgx4), L2 flushed
  before each replay: −0.76 ±0.59 µs (TP3) and −1.65 ±0.49 µs (TP4) per context replay, at most
  −7.1 µs (TP4 c11); outputs bit-identical at every row count, no new TileLang kernels, +5–8 MiB
  driver memory per rank, +6/+10 ms capture; a 3,000-step poisoned-buffer soak per shape (sp8)
  matched bit for bit. The RoCE all_gather bytes it saves were not measured.
- `0681-perf-dspark-reduce-vocabulary-parallel-greedy-drafts.patch` (`#854`, vLLM `3b7884972`):
  the vocabulary-parallel greedy draft steps (TP > 1) reduce base logits + Markov bias in the
  projection's epilogue (the fused kernels 0671 already compiles) instead of materializing the
  bias shard; top-2 is reproduced bit for bit; no fallback (kernel preparation raises for a
  Markov head that does not fuse). Measured on sp8 with the patch's files mounted over
  candidate-37's image: 234 tests passed with the PDL sources plugin (0 violations); the
  CUDA-graph-captured draft step (`chain`, cold) is −2.0 to −12.7 µs at TP3 and −2.9 to
  −16.4 µs at TP4 from 3 requests, even at 2, and +0.4–0.6 µs per position at one request
  on TP3; with the shard hot in L2 it is 1–8 µs slower per step at the recipe's request
  counts, about 0.03% of a decode step at one request; a
  9,000-step functional soak under the draft workspace lane had 0 mismatches. No serving soak
  with real traffic has run.
- `0682-test-tilelang-wo-call-_block32_linear-by-layer-as-th.patch` (`#863`, vLLM `a1a4f086d`):
  `test_grouped_gemm_equals_groups` called `_block32_linear(x, out, layer.tilelang_key,
  scratch)`; the series' custom-op cleanup (`1f3dc7c14`) removed the custom op and its layer
  registry, so the function takes the layer itself. The test now calls it that way; its eleven
  row counts check again that one grouped launch gives each group's ordinary GEMM bit for bit.
  Measured in candidate-38's first image on dgx4: 16 passed (was 5 passed, 11 failed).
- `0683-test-engram-read-the-truncated-row-file-past-the-rea.patch` (`#863`, vLLM `f6fe2e8fe`):
  `test_disk_engram_preparation_refreshes_graph_and_rejects_stale_rows` truncates a row file and
  expects a short-read failure, but the reader's row cache (`56873181ff`, #759 and #754) served
  the step's rows without reading the file. The test builds its reader with `cache_rows=0`, so
  every step reads the files; the cache's own tests are `test_deepseek_v41_engram_rows.py`.
  Measured: 43 passed (was 42 passed, 1 failed).
- `0684-test-ced-metadata-give-the-DFlash-speculator-stand-i.patch` (`#863`, vLLM `e5987aaf4`):
  the `DFlashSpeculator` stand-in of
  `test_dspark_compact_context_preserves_query_anchor_and_rejections` carries the two host
  staging buffers #763 (`6efba0add`) made the speculator fill, `_query_start_loc_cpu` and
  `_draft_seq_lens_cpu_upper_bound`. Measured: 39 passed (was 38 passed, 1 failed).
- `0685-test-gpu-model-runner-the-stand-in-runners-carry-wha.patch` (`#863`, vLLM `89f7cbdbb`):
  six stand-in runners of `test_gpu_model_runner_v2.py` carry what the series made the runner
  read: `num_draft_steps` and `copy_drafter` (copy drafts' wide verification), the block-table
  geometry methods (`e57a0d844`), `uniform_decode_query_len`, the DP sync's `skip_drafts`
  (`7d3263647`), a config and KV-cache spec for the attention profile (which now binds one block
  per KV-cache group, so the test expects three blocks), and `sink` (`2d832d876`). Measured: 40
  passed (was 20 passed, 20 failed).
- `0686-test-gpu-worker-the-stand-in-workers-carry-the-start.patch` (`#863`, vLLM `2e975048a`):
  four stand-in workers of `test_gpu_worker.py` carry the startup and collective hooks the worker
  reads around the code under test: `_roce_failure` (the stopped-collective naming, `dfb93679e`
  and `aaebbf88d`), `_prepare_kernels` and `_startup_rendezvous` before the memory profile
  (`63f390bc7`), `_startup_heartbeat`, and the bound `_determine_available_memory`. Measured: 96
  passed (was 34 passed, 62 failed).
- `0687-test-cudagraph-profiling-model-the-driver-s-free-mem.patch` (`#863`, vLLM `913c74e99`):
  `test_profile_cudagraph_memory_frees_throwaway_pool` asserted `native_after - native_before
  >= 0` on the driver's free-memory reading, which on GB10 (unified memory) drifts with the host
  by 41–64 MiB over 3 s while the process allocates nothing (two probes on an idle dgx4). The test
  models the driver's free memory as total − Torch's reservations − its own `cudaMalloc`, so the
  native delta is exact for both parametrizations; the throwaway-pool release is checked as
  before. Measured: 36 passed in three runs in a row (the case had passed once and failed twice
  in this candidate's first gate records).

#853 and #854 both change `vllm/models/deepseek_v4_1/nvidia/dspark.py`; on `5b7cfa37f` they
apply in either order and give the same tree (#853's record). Here #854 follows #853, and the
whole series applies to a fresh `19f2c20ed` tree without a conflict.

`0682..0687` change six test files and no runtime module (76 insertions, 8 deletions). Each
header states its base (`753354cf4`, this folder's series), rationale, quality implications (none
for serving), the file's counts in candidate-38's first image, and its upstream status (the
TileLang WO, Engram and CED tests are this deployment's; the runner, worker and profiling tests
are Local Inference Lab's and follow the series' own changes). #863 ran the full gate with the
six files mounted over candidate-38's first image: 8,069 passed, 360 skipped, 0 failed
(`experiments/2026-11-03-release-gate-tests/runs/2026-11-03-dgx4/`). That run is #863's
evidence for the fixes; this candidate's gate is the run in its own image, below.

Not included: `#842` (CopyDrafter on by default; its latest patch already incorporates
#860's fix and cleanup, but its additional changes still need their own qualification
and are staged for candidate-39), `#856` and `#859` (superseded by #860).

vLLM tree: `c061e5f6931306fa4655ba531428f639c365bf10` (patch head
`1f33102547e4da5c64712390604695d1abd577c4`; 572 patches; patch set `55370020…`). `git diff
--check` from `5b7cfa37f` is clean. The intermediate trees are 0677 `58882ab9`, 0678
`f473423c`, 0679 `8542285f`, 0680 `77b534d1`, 0681 `37b34b8f`, 0682 `2369ddc0`, 0683
`e7d7eab2`, 0684 `7cc2c394`, 0685 `58427352`, 0686 `d0158098`, 0687 `c061e5f6`. Before
`0682..0687` the candidate was `0001..0681`, head `753354cf4`, tree `37b34b8f`
(commit `8ad0a4c7`); its first image and gate are recorded below.

## Arms

`tp3.json` and `tp4.json` extend rc14's `rc-tp3.json` and `rc-tp4.json` (spark-ds41f-1030's
frozen recipes) with the image, its expected trees and this folder's lock, as candidate-37's
do; `tp3-dev.json`, `tp3-copy.json`, `tp3-copy-dev.json` and `tp3-8192.json` are
candidate-37's with the name changed. `candidate-tp3.json`, `release-tp3.json` and
`screen-tp3.json` are candidate-37's gates with the labels renamed. `image-dgx4.json` is the
build profile. These broader qualification suites remain pending; the focused
TP3-switched reproduction below uses its own tracked arms.

## Checks

- `bin/spark --cluster-config <arm> doctor` (configuration, not `--live`; rerun after `0687`):
  configuration OK for `tp3.json`, `tp3-dev.json`, `tp3-copy.json`, `tp3-copy-dev.json` and
  `tp3-8192.json`. The consolidation check also passes `tp4.json` and
  `image-dgx4.json`: their inherited baseline now names 1030, matching this candidate's
  lock, and the redundant `NCCL_IB_HCA` input is removed. The node-map renderer already
  supplied that variable; the environment rendered for every ring rank is unchanged.
  [Before/after checks](runs/2026-11-03-consolidation/profile-check.json) retain the
  two original doctor failures and verify the correction changes no image, source,
  serving arguments or rendered environment. Frozen releases and candidate-37 are unchanged.
- `git diff --check`: clean, for the repository change and for the vLLM series (`5b7cfa37f` to
  `753354cf4`, and `753354cf4` to `1f3310254`).
- The repository's unit suite (`python3 -m unittest discover -s tests -t .`): 626 tests OK, 5 skipped (38.740 s, consolidation working tree; [receipt](runs/2026-11-03-consolidation/repository-tests.json)).
- Consolidation replayed all 572 patches from the pinned upstream base and reproduced
  the locked head, tree and patch-set digest. The 23 changed Python files compile;
  the removed per-slot limit identifiers are absent from runtime and tests. The
  existing final build inputs also pass a fresh read-only `build check` on dgx4.
  [Consolidation receipts](runs/2026-11-03-consolidation/README.md) separate these
  checks from the final image's recorded GPU gate and pending serving qualification.
- `scripts/release_tests.py check` for `19f2c20ed..1f3310254`: 197 selected test files, 135
  gate, 62 excluded, 0 problems (the same at `753354cf4`; the six files `0682..0687` change
  were gate files already). It was not clean before: 22 files the series changes were
  unclassified, all of them already at candidate-37's head (one, the loop detector's
  `tests/v1/core/test_repetition_detection.py`, also changed by 0679). They are triaged in
  `config/release-tests.json` from a run of the 21 inherited ones in candidate-37's image on
  dgx4 (`c38-triage-probe`, 173 s): 15 join the gate (each passes there, or its tests needing
  hub models skip offline), the Anthropic Messages conversion test joins the gate with its 8
  `async def` tests deselected (the image has no pytest-asyncio or anyio, so pytest fails them
  unrun), and 7 are excluded for the environment: three import a module the image does not
  install (`multiprocess`, `pytest_asyncio`, `scipy`), four have every test skip for a Hugging
  Face download.
- `scripts/release_tests.py gate` at `1f3310254`: 136 gate files, 12 deselected tests
  (`release-tests-gate-manifest.json`); the bundle is `bundles/release-tests`.

## Image

Built on dgx4 from this folder's commit (`bin/spark --cluster-config
experiments/2026-10-26-spark-ds41f-candidate/candidate-38/image-dgx4.json build prepare`,
`build check`, `build image --apply`), tag `spark-ds41f:candidate-38`. The final image
was subsequently copied unchanged to sp5–sp8 for qualification, as recorded below.

The candidate was built twice. The arm's `container.expected_labels` must match the image's
`local.spark-ds41f.vllm.tree` label (`bin/spark cluster start` and `doctor` refuse an image
whose labels differ, `spark_ds41f/cluster.py`), `prepare --lock` writes the series' tree into
that label, and the lab's image step removes a same-named image whose tree differs from the
recipe's as a stale build and builds again (`spark_ds41f/lab.py`, `run_image`). So extending
the series to `0687`, tests only, still changes the image the arms name, and the gate has to be
the new image's own run (`docker/README.md`, "Measured evidence": a source change needs new
test receipts for that build; the earlier run keeps its identity).

- First image (vLLM `0001..0681`, tree `37b34b8f`): fresh tree on dgx4 (2026-11-03 03:17Z,
  `build prepare --lock` from commit `ef78db56`) in
  `.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-8d0ef291`, patch head `753354cf4ed0`,
  tree `37b34b8f9e36`, `--lock` changed no file, `build check` OK; `build image --apply` from
  commit `f3100c53` (03:21–03:29Z): ID
  `sha256:4ce47c40be9ec06719ace4e7d98b3dd9837f7e2ee2ab4ba1a6c43efb90fb8b83` (10.5 GB, 470 s,
  8 compile jobs, lowest MemAvailable 78.7 GiB, receipt
  `…-8d0ef291/images/4ce47c40be9ec067.json`), the GPU, TileLang and sparknet import smokes
  passed. Its gate failed (below). Removed from dgx4 as the stale build under the candidate's tag
  before the second build; it was never copied anywhere.
- **Image** (vLLM `0001..0687`, tree `c061e5f6`): fresh tree on dgx4 (2026-11-03 09:17–09:19Z,
  `build prepare --lock` from commit `df50aa94`) in `.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-05dd5db6`, patch head
  `1f33102547e4`, tree `c061e5f69313`; `--lock` changed no file (the lock, manifest and arms
  already carried these identifiers, recorded with the build's own sync); `build check`: build
  inputs OK. `build image --apply` from commit `df50aa94` (09:19–09:27Z): tag
  `spark-ds41f:candidate-38`, **ID `sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`**
  (10.5 GB), built in 466 s with 8 compile jobs, lowest
  MemAvailable 78.1 GiB (receipt `…-05dd5db6/images/213ba4df68d06665.json`). The GPU import smoke, the TileLang
  import smoke and the sparknet import smoke passed. Labels:
  `local.spark-ds41f.vllm.tree=c061e5f69313…`, `.vllm.patch-head=1f33102547e4…`,
  `.sparknet.tree=8cc28a5cd3e6…`, `.tilelang.tree=222b895d04ee…`,
  `.tile_kernels.tree=cb2fa41c8af2…`, `.nccl.tree=6af10aa730a6…`, `.b12x.tree=1a647a311ca5…`.
  The image has no registry digest; its content-addressed image ID above identifies
  the build. It is now also present unchanged on sp5–sp8, as recorded below.

The other sources are candidate-37's in both builds: NCCL `eeacf1c6` (tree `6af10aa7`), B12X
`69e5795e` (`1a647a31`), TileLang `a0e66d71` (`222b895d`), TileKernels `edfa64d0` (`cb2fa41c`),
sparknet `1fd6a1fb` (`8cc28a5c`), CUTLASS v4.7.1.

## Release tests

The gate bundle ran on dgx4 in the image (`scripts/lab.py kernel-local bundles/release-tests`,
no serving container there), once in each image.

**In the image (tree `c061e5f6`; 09:28–10:04Z; `runs/2026-11-03-dgx4-0687/release-tests/`):
136 files, 8,069 passed, 360 skipped, 0 failed; the verdict is pass and kernel-local exits
0.** The six files the first gate failed pass whole: `test_deepseek_v41_tilelang_wo.py` 16, `test_deepseek_v4_1_engram.py` 43, `test_b12x_v41_ced_metadata.py` 39, `test_gpu_model_runner_v2.py` 40, `test_gpu_worker.py` 96, `test_gpu_model_runner_v2_cudagraph_profiling.py` 36. Per-test outcomes are in `results.jsonl` (one line per file). Skips: 360, the offline classes (Hugging Face models the image lacks, ROCm-only, two GPUs), as in the first gate.

In the first image (tree `37b34b8f`; 03:30–04:20Z; `runs/2026-11-03-dgx4/release-tests/`,
kept as that run's record): 136 files, 7,973 passed, 360 skipped, 96 failed in 6 files; the
verdict was FAIL and kernel-local exited 1. None of the six files was one `0677..0681` change,
and each failure was candidate-37's: the same six files rerun in `spark-ds41f:candidate-37` on
dgx4 right after the gate (`runs/2026-11-03-dgx4/failing-files-on-candidate-37/`) failed with
the same 96 test ids and messages, and rerun in the first image
(`runs/2026-11-03-dgx4/failing-files-rerun/`) with 95, the odd one a measurement that passed or
failed on either image. The cause of each, and the patch that fixes it:

| File | Failed / passed | Why, and the fix |
|---|---|---|
| `tests/kernels/quantization/test_deepseek_v41_tilelang_wo.py` | 11 / 5 | `test_grouped_gemm_equals_groups[*]` read `tilelang_key` from the WO modules; the series' custom-op cleanup (`1f3dc7c14`, 2026-10-31) removed it and the test was not updated (passed 16/16 at candidate-12). `0682` calls `_block32_linear` by layer |
| `tests/models/test_deepseek_v4_1_engram.py` | 1 / 42 | `test_disk_engram_preparation_refreshes_graph_and_rejects_stale_rows`: after a row file is truncated, `disk_rows.failure()` stayed `None` for 30 s (the reader's row cache served the rows), so the test's `in` check raised `TypeError`; the same failure was reported on candidate-36's image in #846. `0683` reads past the cache |
| `tests/v1/attention/test_b12x_v41_ced_metadata.py` | 1 / 38 | `test_dspark_compact_context_preserves_query_anchor_and_rejections`: its `DFlashSpeculator` stand-in lacked `_query_start_loc_cpu`, which #763 (`6efba0add`) made the speculator read. `0684` gives the stand-in both host buffers |
| `tests/v1/worker/test_gpu_model_runner_v2.py` | 20 / 20 | the stand-in runners lacked `num_draft_steps`, `copy_drafter` (the copy-drafts patches), `vllm_config`, `_block_table_max_model_len`, `uniform_decode_query_len` and `sink`; #860's record lists the same 20 on candidate-37. `0685` |
| `tests/v1/worker/test_gpu_worker.py` | 62 / 34 | the stand-in workers lacked `_roce_failure`, which `execute_model` and the warm-up read since the stopped-collective naming (`dfb93679e`, `aaebbf88d`) and the memory breakdown (`111c5be7e`). `0686` |
| `tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py` | 1 / 35 | `test_profile_cudagraph_memory_frees_throwaway_pool[0]`: a subprocess asserted `native_after - native_before >= 0` on the driver's free-memory reading around a fake capture, which drifts on GB10; it passed in the triage probe on candidate-37's image (03:12Z), failed in this gate and in the candidate-37 rerun, and passed in the first image's rerun. `0687` models the reading |

So the first image failed no test that candidate-37's image passed: the 95 deterministic
failures were test stand-ins and one test left behind by candidate-37's series (and earlier
candidates'), not runtime defects, and `0682..0687` are those tests brought up to the series.

Skips: 360, the offline classes (Hugging Face models the image lacks, ROCm-only, two GPUs).

## Unit evidence

Both bundles ran on dgx4 in the image right after its gate (`runs/2026-11-03-dgx4-0687/`):

- `bundles/unit-evidence` (the runner over the files the `0677..0681` experiments name;
  10:04–10:06Z): **395 passed, 1 skipped, 0 failed** in 131 s. Per file: `test_repetition_detection.py` 40 (#848), `test_copy_drafts.py` 24 and
  `test_copy_drafts_scheduler.py` 11 and `test_adaptive_verification.py` 36 + 1 skipped (the
  two-CUDA-device test) and `test_prepare_kernels.py` 8 (#860), `test_deepseek_v41_tilelang_dspark_context.py`
  24 and `test_cudagraph_manager.py` 26 (#853), `test_deepseek_v41_tilelang_vocab.py` 72,
  `test_dspark_greedy_markov.py` 48, `test_dspark_vocab_parallel.py` 17,
  `test_greedy_verification.py` 18 and `test_deepseek_v4_1_sp_prefill.py` 71 (#854).
- `bundles/unit-evidence-pdl` (#854's own command: its six files under
  `tests.kernels.deepseek_v41_pdl_sources`, `--noconftest`; 10:06Z): **234 passed**
  in 25 s, as on candidate-37 with the patch mounted; `PDL sources: 230 kernels launched, 62 with PDL, 0 violations`.

The same bundles in the first image (`runs/2026-11-03-dgx4/`, 04:20–04:22Z): 395 passed,
1 skipped (the two-CUDA-device test), 0 failed in 130 s, and 234 passed with `230 kernels
launched, 62 with PDL, 0 violations`.

## Qualification at tp3-switched (sp5–sp7)

The three spaceport nodes serve candidate-37 at the recipe's `tp3-switched` shape from a
node-local profile (`config/candidate-37-tp3-switched-dev.local.json`). Candidate-38's is
`config/candidate-38-tp3-switched-dev.local.json` on sp5, sp6 and sp7 (sha256 `461cfbe2…`):
that profile with this candidate's image, expected trees and lock. The arms here extend it, as
#860's arms extended candidate-37's: `tp3-switched-dev.json`, and `tp3-switched-copy-dev.json`
with `tp3-copy.json`'s serve options. Both carry the dev-mode collective RPC, as the serving
profile does, so this shape's gate digests are measured in dev mode (the triangle's `tp3.json`
is not).

- `candidate-tp3-switched.json` (run `candidate-38-tp3-switched`): `candidate-tp3.json`'s
  extras on the two arms, with the ring's node names replaced (`probes.py --nodes
  sp5,sp6,sp7`; `mem_snapshot.py` gains a `--nodes` option, the ring by default).
- `release-tp3-switched.json` (run `candidate-38-release-tp3-switched`): the `release-tp3`,
  `single-stream` and `sampled` suites on four boots, `candidate-38`, `candidate-38-copy`,
  `candidate-38-copy-again`, `candidate-38-again`, so each measured arm has a boot and a repeat
  boot and the paired verdict pairs the first and the last. The reference is
  spark-ds41f-1030's recorded runs at this shape on sp5
  (`results/private/suites/1030-tp3-switched-{release-tp3,single-stream,sampled}/1030`,
  2026-11-01 05:52–07:19Z; suite digests `4f54f4b5…`, `7c59b27e…`, `ab647d38…`, the current
  definitions), linked as `results/private/suites/references/1030-tp3-switched{,-single-stream,-sampled}`,
  because the shape's `benchmark_reference` is a bench manifest, not a suite run. Candidates
  27–37's runs beside them were made outside the lab, one boot each.

**Image on the nodes.** `spark-ds41f:candidate-38` on sp5, sp6 and sp7 is dgx4's build, not a
rebuild. dgx4 reaches the spaceport hosts only through a relay at about 1.7 MB/s, and the nodes
already held 39 of the image's 57 layers (candidate-37's base; every host runs the containerd
image store, where `docker save` writes compressed blobs by digest), so what shipped was `docker
save`'s OCI layout with the 18 layer blobs the nodes lacked, the config and the manifests: 24
blobs, 831,180,800 bytes, sha256 `436d8b7b…`, dgx4 to sp5 through the Mac (12:32–12:42Z), then
sp5 to sp8, sp6 and sp7 over their LAN. `docker load` took the shared layers from each store.
sp8 was loaded first as the trial. On dgx4, sp5, sp6, sp7 and sp8 the image is ID
`sha256:213ba4df68d0…`, labels `local.spark-ds41f.vllm.tree=c061e5f6…` and
`.sparknet.tree=8cc28a5c…`, and the layer list's digest `a5f28f19…` is the same.

The broader candidate/release suites above remain pending. The separate
[focused reproduction](../../2026-11-03-candidate-38-reproduction/README.md)
completed on 2026-11-03 at deployment commit `806c5797`, using the unchanged final
image and no source overlays. Both loop boots reproduced seeded sustainment at
effort 100 and 75 with detection off; the default detector forced a reasoning end.
The bounded clean fixture did not develop a new loop. Both CopyDrafter boots
completed the original copy-probe/single-stream sequence and eight-/sixteen-client
stress without a captured NaN assertion, worker failure or request error.
No per-step rank-layout recorder was present, and these diagnostic rates do not
establish a release performance comparison. Candidate-37 was restored on all
three nodes, verified, and the test window closed.
