# candidate-38

Base: `candidate-37` (`0001..0676`, vLLM patch head `5b7cfa37f`, tree `af0fb152`; `sparknet` `1fd6a1f`).

Added patches (`0677..0681`). Each is the `git format-patch` commit its pull request made on
`5b7cfa37f`, taken from the pull request's branch byte for byte apart from the re-sequenced
`[PATCH n/681]` subject prefix (`make_series.py`), and applied in this order:

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
  with both patches); not measured and in no arm of #860. Included as a harmless architectural
  cleanup, under the owner's rule.
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

#853 and #854 both change `vllm/models/deepseek_v4_1/nvidia/dspark.py`; on `5b7cfa37f` they
apply in either order and give the same tree (#853's record). Here #854 follows #853, and the
whole series applies to a fresh `19f2c20ed` tree without a conflict.

Not included: `#842` (CopyDrafter on by default; it needs rebasing onto #860 and its own
qualification, and is staged for candidate-39), `#856` and `#859` (superseded by #860).

vLLM tree: `37b34b8f9e365cf9194b198008abf650581936a0` (patch head
`753354cf4ed0970f31ba4ef8c754a1c80a55e4e5`; 566 patches; patch set `49f728a8…`). `git diff
--check` from `5b7cfa37f` is clean. The intermediate trees are 0677 `58882ab9`, 0678
`f473423c`, 0679 `8542285f`, 0680 `77b534d1`, 0681 `37b34b8f`.

## Arms

`tp3.json` and `tp4.json` extend rc14's `rc-tp3.json` and `rc-tp4.json` (spark-ds41f-1030's
frozen recipes) with the image, its expected trees and this folder's lock, as candidate-37's
do; `tp3-dev.json`, `tp3-copy.json`, `tp3-copy-dev.json` and `tp3-8192.json` are
candidate-37's with the name changed. `candidate-tp3.json`, `release-tp3.json` and
`screen-tp3.json` are candidate-37's gates with the labels renamed. `image-dgx4.json` is the
build profile. Qualification is queued by the parent session; nothing here was started on a
serving node.

## Checks

- `bin/spark --cluster-config <arm> doctor`: configuration OK for `tp3.json`, `tp3-dev.json`,
  `tp3-copy.json`, `tp3-copy-dev.json` and `tp3-8192.json`. `tp4.json` and `image-dgx4.json`
  report `cluster and upstream locks name different baselines`, as candidate-37's do: rc14's
  ring recipe names spark-ds41f-1025 (the ring's last qualified baseline) and the lock is
  generated from spark-ds41f-1030. Setting the ring arm's baseline to 1030 instead fails the
  1030 recipe's `NCCL_IB_HCA` rule, so the arm stays as candidate-37's. The build does not read
  this check.
- `git diff --check`: clean, for the repository change and for the vLLM series (`5b7cfa37f` to
  `753354cf4`).
- The repository's unit suite (`python3 -m unittest discover -s tests -t .`): 626 tests OK,
  5 skipped.
- `scripts/release_tests.py check` for `19f2c20ed..753354cf4`: 197 selected test files, 135
  gate, 62 excluded, 0 problems. It was not clean before: 22 files the series changes were
  unclassified, all of them already at candidate-37's head (one, the loop detector's
  `tests/v1/core/test_repetition_detection.py`, also changed by 0679). They are triaged in
  `config/release-tests.json` from a run of the 21 inherited ones in candidate-37's image on
  dgx4 (`c38-triage-probe`, 173 s): 15 join the gate (each passes there, or its tests needing
  hub models skip offline), the Anthropic Messages conversion test joins the gate with its 8
  `async def` tests deselected (the image has no pytest-asyncio or anyio, so pytest fails them
  unrun), and 7 are excluded for the environment: three import a module the image does not
  install (`multiprocess`, `pytest_asyncio`, `scipy`), four have every test skip for a Hugging
  Face download.
- `scripts/release_tests.py gate`: 136 gate files, 12 deselected tests
  (`release-tests-gate-manifest.json`); the bundle is `bundles/release-tests`.

## Image

Built on dgx4 from this commit (`bin/spark --cluster-config
experiments/2026-10-26-spark-ds41f-candidate/candidate-38/image-dgx4.json build prepare`,
`build check`, `build image --apply`), tag `spark-ds41f:candidate-38`. Not copied anywhere.

- Fresh tree (dgx4, 2026-11-03 03:17Z, `build prepare --lock` from commit `ef78db56`): the
  series applied to `19f2c20ed` in `.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-8d0ef291`
  and reproduced patch head `753354cf4ed0` and tree `37b34b8f9e36`; `--lock` changed no file
  (the lock, manifest and arms already carried these identifiers); `build check`: build inputs
  OK. The other sources are candidate-37's: NCCL `eeacf1c6` (tree `6af10aa7`), B12X `69e5795e`
  (`1a647a31`), TileLang `a0e66d71` (`222b895d`), TileKernels `edfa64d0` (`cb2fa41c`), sparknet
  `1fd6a1fb` (`8cc28a5c`), CUTLASS v4.7.1.
- Image (dgx4, `build image --apply` from commit `f3100c53`, 03:21–03:29Z): tag
  `spark-ds41f:candidate-38`, **ID `sha256:4ce47c40be9ec06719ace4e7d98b3dd9837f7e2ee2ab4ba1a6c43efb90fb8b83`**
  (10.5 GB), built in 470 s with 8 compile jobs, lowest MemAvailable 78.7 GiB (receipt
  `.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-8d0ef291/images/4ce47c40be9ec067.json`).
  The GPU import smoke, the TileLang import smoke and the sparknet import smoke passed. Labels:
  `local.spark-ds41f.vllm.tree=37b34b8f9e36…`, `.vllm.patch-head=753354cf4ed0…`,
  `.sparknet.tree=8cc28a5cd3e6…`, `.tilelang.tree=222b895d04ee…`,
  `.tile_kernels.tree=cb2fa41c8af2…`, `.nccl.tree=6af10aa730a6…`, `.b12x.tree=1a647a311ca5…`.
  The image has no registry digest: it is on dgx4 only, under no other tag.

## Release tests

The gate bundle ran on dgx4 in the image (`scripts/lab.py kernel-local bundles/release-tests`,
03:30–04:20Z, no serving container there; `runs/2026-11-03-dgx4/release-tests/`): **136 files,
7,973 passed, 360 skipped, 96 failed in 6 files; the verdict is FAIL and kernel-local exits 1.**
None of the six files is one the five new patches change, and each failure is candidate-37's:
the same six files rerun in `spark-ds41f:candidate-37` on dgx4 right after the gate (the same
runner and test files, `runs/2026-11-03-dgx4/failing-files-on-candidate-37/`) fail with the
same 96 test ids and messages. Rerun in candidate-38's image
(`runs/2026-11-03-dgx4/failing-files-rerun/`) they fail with 95: the one that differs is a
measurement test that passes or fails on either image.

| File | Failed / passed | Why |
|---|---|---|
| `tests/kernels/quantization/test_deepseek_v41_tilelang_wo.py` | 11 / 5 | `test_grouped_gemm_equals_groups[*]` reads `tilelang_key` from the WO modules; the series' custom-op cleanup (`1f3dc7c14`, 2026-10-31) removed it and the test was not updated (passed 16/16 at candidate-12) |
| `tests/models/test_deepseek_v4_1_engram.py` | 1 / 42 | `test_disk_engram_preparation_refreshes_graph_and_rejects_stale_rows`: after a row file is truncated, `disk_rows.failure()` stays `None` for 30 s, so the test's `in` check raises `TypeError`; the same failure was reported on candidate-36's image in #846 |
| `tests/v1/attention/test_b12x_v41_ced_metadata.py` | 1 / 38 | `test_dspark_compact_context_preserves_query_anchor_and_rejections`: its `DFlashSpeculator` stand-in lacks `_query_start_loc_cpu`, which #763 (`6efba0add`) made the speculator read |
| `tests/v1/worker/test_gpu_model_runner_v2.py` | 20 / 20 | the stand-in runners lack `num_draft_steps`, `copy_drafter` (the copy-drafts patches), `vllm_config`, `_block_table_max_model_len`, `uniform_decode_query_len` and `sink`; #860's record lists the same 20 on candidate-37 |
| `tests/v1/worker/test_gpu_worker.py` | 62 / 34 | the stand-in workers lack `_roce_failure`, which `execute_model` and the warm-up read since the stopped-collective naming (`dfb93679e`, `aaebbf88d`) and the memory breakdown (`111c5be7e`) |
| `tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py` | 1 / 35 | `test_profile_cudagraph_memory_frees_throwaway_pool[0]`: a subprocess asserts `native_after - native_before >= 0` on the driver's free-memory reading around a fake capture; it passed in the triage probe on candidate-37's image (03:12Z), failed in this gate and in the candidate-37 rerun, and passed in the candidate-38 rerun |

So the image fails no test that candidate-37's image passes. The 95 deterministic failures are
test stand-ins and one test left behind by candidate-37's series (and earlier candidates'), not
runtime defects; fixing those tests is its own change. Until then the gate does not exit 0 on
this series.

Skips: 360, the offline classes (Hugging Face models the image lacks, ROCm-only, two GPUs).

## Unit evidence

Both bundles ran on dgx4 in the image right after the gate (`runs/2026-11-03-dgx4/`):

- `bundles/unit-evidence` (the runner over the files the five patches' experiments name;
  04:20–04:22Z): **395 passed, 1 skipped, 0 failed** in 130 s. Per file:
  `test_repetition_detection.py` 40 (#848), `test_copy_drafts.py` 24 and
  `test_copy_drafts_scheduler.py` 11 and `test_adaptive_verification.py` 36 + 1 skipped (the
  two-CUDA-device test) and `test_prepare_kernels.py` 8 (#860), `test_deepseek_v41_tilelang_dspark_context.py`
  24 and `test_cudagraph_manager.py` 26 (#853), `test_deepseek_v41_tilelang_vocab.py` 72,
  `test_dspark_greedy_markov.py` 48, `test_dspark_vocab_parallel.py` 17,
  `test_greedy_verification.py` 18 and `test_deepseek_v4_1_sp_prefill.py` 71 (#854).
- `bundles/unit-evidence-pdl` (#854's own command: its six files under
  `tests.kernels.deepseek_v41_pdl_sources`, `--noconftest`; 04:22Z): **234 passed** in 22 s, as
  on candidate-37 with the patch mounted; `PDL sources: 230 kernels launched, 62 with PDL, 0
  violations`.
