# Release-gate test fixes for candidate-38's series

Candidate-38 ([`#861`](https://github.com/christopherowen/spark-ds41f/pull/861)) is built and
gated; its gate exits 1 on 96 tests in six files that candidate-37's series left behind, every
one reproduced with the same ids and messages on candidate-37's image (its record,
`experiments/2026-10-26-spark-ds41f-candidate/candidate-38/README.md`, "Release tests").
This experiment fixes those tests, so that the gate exits 0 on the series, as a promotion
requires ([methodology](../../docs/methodology.md#promotion), item 8). Nothing but the six test
files changes: no runtime module, no configuration, no image.

## Base

- Deployment: `main` at `c26ed0a4`; candidate-38's record at `8ad0a4c7` (branch `candidate-38`).
- vLLM: candidate-38's patch head `753354cf4ed0970f31ba4ef8c754a1c80a55e4e5` (tree `37b34b8f`),
  566 patches on `19f2c20ed`. The six patches here are `git format-patch` commits on that head.
- Image: `spark-ds41f:candidate-38`, ID `sha256:4ce47c40be9ec06719ace4e7d98b3dd9837f7e2ee2ab4ba1a6c43efb90fb8b83`,
  on dgx4 only; built from `f3100c53`. Nothing was rebuilt: the fixed files are mounted over
  the image's copies, which is how the gate reads a test file anyway (the image's tests are the
  patch head's).
- Gate bundle: candidate-38's `bundles/release-tests` (136 files, 12 deselected tests, the
  runner, `tblib_stub.py`, `offline_skip.py`), copied byte for byte, plus the six mounts.

## Intended variable

The six test files, and nothing else. Each failure is a stand-in or a fixture that a series
patch changed the code under without updating the test; none is a defect of the model or the
runtime, and none of the fixes touches a runtime module.

| File | Failed | Cause | Fix (patch) |
|---|---:|---|---|
| `tests/kernels/quantization/test_deepseek_v41_tilelang_wo.py` | 11 | `test_grouped_gemm_equals_groups` passed `layer.tilelang_key` to `_block32_linear`; the custom-op cleanup (`1f3dc7c14`) removed the key and the registry, and `_block32_linear` takes the layer | call it by layer (`0001`) |
| `tests/models/test_deepseek_v4_1_engram.py` | 1 | the test truncates a row file and waits for the next step's read to fail; since the reader's row cache (`56873181ff`), a step whose rows were read before reads no file, and the test's last steps reuse the ids of the step before them, so `failure()` stayed `None` (reported on candidate-36 too, `#846`) | the test's reader has `cache_rows=0`, so every step reads the files (`0002`) |
| `tests/v1/attention/test_b12x_v41_ced_metadata.py` | 1 | the `DFlashSpeculator` stand-in lacked the host staging buffers `#763` (`6efba0add`) added, `_query_start_loc_cpu` and `_draft_seq_lens_cpu_upper_bound` | the stand-in carries both (`0003`) |
| `tests/v1/worker/test_gpu_model_runner_v2.py` | 20 | six stand-in runners lacked `num_draft_steps`, `copy_drafter`, `uniform_decode_query_len` (copy drafts, `e8c696282`), the DP sync's `skip_drafts` (`7d3263647`), `sink` (`2d832d876`), the block-table geometry methods (`e57a0d844`), and the DeepSeek V4 profile's `vllm_config` and KV cache spec (it binds one block per KV cache group now) | each stand-in carries what the runner holds; the profile test expects one block per group plus the null block (`0004`) |
| `tests/v1/worker/test_gpu_worker.py` | 62 | four stand-in workers lacked `_roce_failure` (the stopped-collective cause, `dfb93679e`, `aaebbf88d`), the kernel preparation and startup rendezvous (`63f390bc7`), the heartbeat, and `_determine_available_memory` | each stand-in carries them, as the memory-breakdown test (`111c5be7e`) already did (`0005`) |
| `tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py` | 1 (flapping) | `test_profile_cudagraph_memory_frees_throwaway_pool[0]` asserted a non-negative native delta read from the driver; on GB10 the device's memory is the host's and the driver's free reading drifts with nothing allocated (see "Memory observations") | the test models the driver's free memory as Torch's reservations plus its own `cudaMalloc`, so the delta is exact (`0006`) |

## Patches

`vllm/0001..0006`, one per file, in the table's order. Each applies with `git am` on
`753354cf4`; together they give patch head `913c74e99a` here (tree `c061e5f6931306fa4655ba531428f639c365bf10`; a fresh `git am` on `753354cf4` reproduces the tree).
`git diff --check` is clean. The candidate's packager appends them to its series (`0682..0687`
after candidate-38's `0681`); they change no runtime module, so the image they are gated in
needs no rebuild for them (a rebuild that includes them gives the same runtime bits).

Each message states its base, the rationale, the quality implications (none: test-only), the
tests and the upstream status.

## Runs (dgx4, 2026-11-03, UTC)

All in `spark-ds41f:candidate-38` on dgx4, the GPU to themselves (no serving container; the
lab's `/tmp/spark-lab-gpu.lock` held), the six files mounted over the image's copies from a
clone of the vLLM tree at the patches' head (`/tmp/gate-tests-fix/vllm`), the same
`-p tblib_stub -p offline_skip`, environment, seccomp allowance and compile cache as
`scripts/lab.py kernel-local` gives the gate (`runs/2026-11-03-dgx4/run_tests.sh`). Receipts
are the pytest output and JUnit report of each run.

| Run | When (UTC) | What | Result | Receipts |
|---|---|---|---|---|
| six files, first | 07:46:06–07:47:52 | the six files with the fixes as first written | 251 passed, 2 failed, 17 errors; exit 1 | `six-files-first/` |
| six files | 07:49:07–07:50:55 | the six files as patched | **270 passed**, exit 0: WO 16, Engram 43, CED metadata 39, model runner 40, worker 96, cudagraph profiling 36 | `six-files/` |
| cudagraph profiling, repeats | 07:51:39–07:52:19, 07:52:19–07:52:58 | `test_gpu_model_runner_v2_cudagraph_profiling.py` alone, twice | 36 passed each time; with the six-file run, `test_profile_cudagraph_memory_frees_throwaway_pool[0]` passed three times in a row (it passed once and failed twice in the record's three runs) | `cudagraph-profiling-repeat-1/`, `-2/` |
| GB10 free-memory probe | 07:52:58–07:53:02 | 16 readings of the driver's free memory, 0.2 s apart, nothing allocated | 64.1 MiB of spread (41.3 MiB in a first, unsaved probe at 07:36) | `gb10-free-memory-probe/` |
| release gate, first start | 07:56:09–07:57 | the same bundle; stopped after a minute (it was attached to a session that could have cut it off at ten minutes) and relaunched detached; its partial outputs were removed | stopped, no result | none |
| release gate | 07:57:55–08:33:26 | `scripts/lab.py kernel-local bundles/release-tests`: 136 files, 12 deselected tests, the six fixed files mounted over the image's copies | **8,069 passed, 360 skipped, 0 failed; verdict pass, kernel-local exit 0**, 2,131 s (candidate-38's gate: 7,973 passed, 360 skipped, 96 failed, 2,998 s); every file `ok`; `verdict.json` hashes the bundle's inputs, the six files among them | `release-tests/` |

The first run's two failures and 17 errors were this experiment's own, not the series':
`test_boundary_logits_only_dispatches_pending_cache_tasks` read `skip_drafts` from its DP sync
sentinel, which the fix's next iteration gives it (patch `0004` has it), and the 17 errors are
pytest's `--basetemp` whose parent directory the ad hoc script had not created (the gate's
runner creates it; `run_tests.sh` is the corrected script). The run is kept as the record asks.


## Commands

On the Mac (this repository at `c26ed0a4`, a clone of the vLLM tree at `753354cf4`):

```sh
git -C <vllm> checkout -b release-gate-tests 753354cf4ed0970f31ba4ef8c754a1c80a55e4e5
# edit the six test files, one commit each
git -C <vllm> format-patch -o experiments/2026-11-03-release-gate-tests/vllm 753354cf4..HEAD
git -C <vllm> diff --check 753354cf4..HEAD
# the fresh-apply check: a detached worktree at 753354cf4, `git am vllm/000*.patch`, tree c061e5f6
```

On dgx4 (`/tmp/gate-tests-fix/vllm`: a clone of `/tmp/c38-stack` at `753354cf4` with the
patches applied; `run_tests.sh` is `runs/2026-11-03-dgx4/run_tests.sh`, the lab's container
command for an ad hoc pytest):

```sh
run_tests.sh six-2 tests/kernels/quantization/test_deepseek_v41_tilelang_wo.py \
  tests/models/test_deepseek_v4_1_engram.py tests/v1/attention/test_b12x_v41_ced_metadata.py \
  tests/v1/worker/test_gpu_model_runner_v2.py tests/v1/worker/test_gpu_worker.py \
  tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py
run_tests.sh cg-2 tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py   # and cg-3
docker run --rm --gpus all --entrypoint python3 -v .../probe:/p:ro spark-ds41f:candidate-38 /p/mem.py
# the gate, from a clone of this branch at 5105781e:
scripts/lab.py kernel-local experiments/2026-11-03-release-gate-tests/bundles/release-tests \
  --out /tmp/gate-tests-fix/gate-out
```

The gate's `results.jsonl` comes from the compile cache the bundle mounts at `/c`
(`~/.cache/spark-lab/compile/release-tests/candidate-38-gate-tests/`), `output.txt` and
`verdict.json` from `--out`.

## Memory observations

GB10's device memory is the host's (`torch.cuda.get_device_properties(0).is_integrated` is 1,
126,756 MiB total). In the image, with nothing allocated, 16 readings of
`torch.cuda.mem_get_info()` free memory over 3 s spread 64.1 MiB
(`runs/2026-11-03-dgx4/gb10-free-memory-probe/`; 41.3 MiB in a first, unsaved probe). A 64 MiB
Torch allocation dropped the free reading by 54.2 MiB (90.4 MiB in the first probe). So `total - free - torch reserved`, the native measurement
`profile_cudagraph_memory` records, moves by tens of MiB between two readings, which is the
flapping test's whole signal in its `native_bytes=0` case. Production reads the same
measurement into `cudagraph_native_memory_profile`; `_determine_available_memory` clamps the
retained native capture at 0 and bounds it by the measured and late-persistent memory, so the
drift moves the KV budget by tens of MiB at most. Not a defect; noted for the record.

## Checks

- `bin/spark doctor`: configuration OK (run in the worktree with the main checkout's git-ignored
  site maps copied in; the change touches nothing under `config/`).
- The repository's unit suite (`python3 -m unittest discover -s tests -t .`): 626 tests OK,
  5 skipped (with the site maps; without `config/nodes-ring4.local.json`, five tests error on
  the missing map, which is the worktree's state and not the change's).
- `git diff --check`: clean, for the repository change and for the vLLM patches
  (`753354cf4` to the patched head).

## Conclusion

The release-test gate exits 0 on candidate-38's series with these six test-only patches:
136 files, 8,069 passed, 360 skipped, 0 failed (the 96 that failed now pass, the same 360
offline skips as before, no other file changed its counts). Candidate-38's promotion therefore
needs the six appended to its series (`0682..0687`), nothing else: no runtime module changes,
so the gated image is the same runtime whether or not it is rebuilt with them.

Each failure was a test standing still while a series patch moved the code under it: four
files of stand-in objects missing attributes, one test defeated by a reader's cache, and one
test asserting a sign on a driver reading that drifts on unified memory. None exposed a defect
of the model or the runtime, so none was reported as one. The LIL-owned files
(`test_gpu_worker.py`, `test_gpu_model_runner_v2.py`,
`test_gpu_model_runner_v2_cudagraph_profiling.py`) get the smallest diffs that follow the
series' changes; the unified-memory drift in the native capture measurement would be worth a
word to LIL.

Not touched: any serving cluster, lab window, the hold file, the `candidate-38` branch, or
dgx4's deployment checkout (`~/projects/spark-ds41f`, still detached at `f3100c53`). dgx4 was
free again at 08:33:26Z.
