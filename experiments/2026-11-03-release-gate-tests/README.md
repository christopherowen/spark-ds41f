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
| `tests/models/test_deepseek_v4_1_engram.py` | 1 | the test truncates a row file and waits for the next step's read to fail; since the reader's row cache (`56873181ff`), a step whose rows were read before reads no file, and the test's last steps reuse the step before's ids, so `failure()` stayed `None` (reported on candidate-36 too, `#846`) | the test's reader has `cache_rows=0`, so every step reads the files (`0002`) |
| `tests/v1/attention/test_b12x_v41_ced_metadata.py` | 1 | the `DFlashSpeculator` stand-in lacked the host staging buffers `#763` (`6efba0add`) added, `_query_start_loc_cpu` and `_draft_seq_lens_cpu_upper_bound` | the stand-in carries both (`0003`) |
| `tests/v1/worker/test_gpu_model_runner_v2.py` | 20 | six stand-in runners lacked `num_draft_steps`, `copy_drafter`, `uniform_decode_query_len` (copy drafts, `e8c696282`), the DP sync's `skip_drafts` (`7d3263647`), `sink` (`2d832d876`), the block-table geometry methods (`e57a0d844`), and the DeepSeek V4 profile's `vllm_config` and KV cache spec (it binds one block per KV cache group now) | each stand-in carries what the runner holds; the profile test expects one block per group plus the null block (`0004`) |
| `tests/v1/worker/test_gpu_worker.py` | 62 | four stand-in workers lacked `_roce_failure` (the stopped-collective cause, `dfb93679e`, `aaebbf88d`), the kernel preparation and startup rendezvous (`63f390bc7`), the heartbeat, and `_determine_available_memory` | each stand-in carries them, as the memory-breakdown test (`111c5be7e`) already did (`0005`) |
| `tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py` | 1 (flapping) | `test_profile_cudagraph_memory_frees_throwaway_pool[0]` asserted a non-negative native delta read from the driver; on GB10 the device's memory is the host's and the driver's free reading drifts with nothing allocated (see "Memory observations") | the test models the driver's free memory as Torch's reservations plus its own `cudaMalloc`, so the delta is exact (`0006`) |

## Patches

`vllm/0001..0006`, one per file, in the table's order. Each applies with `git am` on
`753354cf4`; together they give patch head `HEAD_PLACEHOLDER` (tree `TREE_PLACEHOLDER`).
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

RUNS_PLACEHOLDER

## Memory observations

GB10's device memory is the host's (`torch.cuda.get_device_properties(0).is_integrated` is 1,
126,756 MiB total). In the image, with nothing allocated, 16 readings of
`torch.cuda.mem_get_info()` free memory over 3 s spread 41.3 MiB
(`runs/2026-11-03-dgx4/gb10-free-memory-probe/`). A 64 MiB Torch allocation dropped the free
reading by 90.4 MiB. So `total - free - torch reserved`, the native measurement
`profile_cudagraph_memory` records, moves by tens of MiB between two readings, which is the
flapping test's whole signal in its `native_bytes=0` case. Production reads the same
measurement into `cudagraph_native_memory_profile`; `_determine_available_memory` clamps the
retained native capture at 0 and bounds it by the measured and late-persistent memory, so the
drift moves the KV budget by tens of MiB at most. Not a defect; noted for the record.

## Conclusion

CONCLUSION_PLACEHOLDER
