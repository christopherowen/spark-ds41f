# candidate-55

The first candidate on frozen [candidate-54](../candidate-54/README.md) (tag `candidate-54`): five of the owner's PRs
that change the platform under the model rather than one kernel. Chosen by the owner on 2026-11-11 ("check 1134 and
include the cuda 13.5 setup for candidate-55", "also add the b12x removal", "also add the sparknet rework PR": "both
#989 and #982"). It is a candidate, not a merged base: verified against candidate-54's recorded runs, not
release-qualified.

| PR | What | Carried as |
| --- | --- | --- |
| #1134 | Weight arenas and MoE slabs take whole free blocks of 2 MiB or more at load (both kernel page sizes), and the RoCE proxy, TP worker and EngineCore are placed by L3 size | vLLM `0800`, `0801`; sparknet `0001` |
| #1096 | B12X removed: no B12X source in the lock, no B12X import on DS4.1 runtime or loader paths, its startup rounds skipped | vLLM `0802` (rebased on candidate-54 and #1134) |
| #989 | sparknet: one persistent multi-stage kernel per collective, 8 thread blocks, no single-slot cap, so SP prefill collectives up to 16 MiB stay on sparknet | sparknet `0002` |
| #982 | sparknet: PDL chaining of the four one-shot collectives (Part A) | sparknet `0003` (re-placed in #989's stage loop); vLLM `0803` (its PDL-chain test entry) |
| #974 | CUDA 13.5.37 toolchain (nvcc/ptxas, cudart) and cuBLASLt 13.10.0.20 | `sources.cuda_toolkit` in this lock (hash-locked debs) |

## Selection

- **Authors:** every PR is christopherowen's, and every patch's `From:` is Christopher Owen's.
- **#1134 is carried in part:** its vLLM and sparknet patches only, at head `d9b04765` (byte-identical to the
  earlier head `2d0c6c0b` they were first taken from). Its `scripts/memguard.sh`, `scripts/memguard-start.sh` and
  `spark_ds41f/lab.py` telemetry pinning are deployment tooling, not image content: memguard runs on production nodes.
  They go to `main` on their own. Its servebench changes (`trim_stall_tail`, an untimed prefill warm-up), which would
  alter the measurement this candidate is judged by, have since left #1134.
- **#1134 contains #1008:** its `0801` adds `vllm/utils/cpu_affinity.py` with #1008's partition, and its sparknet
  `0001` is #1008's sparknet patch (the same tree, `dab8c457`). #1008 is not carried separately.
- **#982 is carried in part:** Part A (PDL chaining). Part B (`next_send_buffer` zero-copy staging) has no caller in
  vLLM, so it does nothing in this deployment and is left out.
- **#974 is selected through the lock:** its toolchain stage builds only for a lock that lists `sources.cuda_toolkit`,
  so `spark-ds41f-1107`, candidate-54 and every other lock build as before on the base image's CUDA 13.0.2. The debs
  are hash-locked in `requirements/`.
- **#1096's tooling** (`scripts/kernel_backend.py`, `scripts/recipe.py`: a lock without B12X drops its tree label) is
  merged to `main` with #974's selection; its experiment folder is not carried.
- **Round 3's PRs** (#978, #988, #957, #956) are not in this candidate; they re-enter later candidates on frozen
  candidate-54 with fresh validation.

## Source

- vLLM ([vllm/series](vllm/series)): candidate-54's 672 patches by relative path, then `0800`, `0801` (#1134's files,
  unchanged), `0802` (#1096 rebased, [receipts/b12x-rebase](receipts/b12x-rebase/README.md)) and `0803` (#982's test
  entry). 676 patches, `patch_head` `a64a804c`, `expected_tree` `980e3812`, `patchset_sha256` `cff45647`.
- sparknet ([sparknet/series](sparknet/series)) on `1fd6a1fb`: `0001` (#1134), `0002` (#989), `0003` (#982 Part A,
  [receipts/sparknet-stack](receipts/sparknet-stack/README.md)). `patch_head` `1fdf8dc7`, `expected_tree` `e06f6a06`,
  `patchset_sha256` `60bb1942`.
- No B12X source. CUDA 13.5.37 through `sources.cuda_toolkit` ([requirements/cuda-13-5-sbsa.txt](../../../requirements/cuda-13-5-sbsa.txt), 23
  debs, 1.51 GB). Every other source is candidate-54's.
- Recorded by `bin/spark build prepare --lock` on dgx4 for both locks (the same vLLM and sparknet trees), the heads
  copied into each `upstreams.lock.json`; the 23 debs downloaded and verified against the lockfile there. The
  provenance, SHA-256 and author of each patch are in [receipts/selection.json](receipts/selection.json).
- **What #1096 changes on the serving path:** one TileLang kernel, `engram_resident_lookup_kernel`, whose row bound
  is now `0 <= n_tok <= tokens` (bit-identical inside it, recompiled); the rest is renames, dead fields, the deleted
  B12X loader and startup rounds skipped without B12X ([receipts/b12x-rebase](receipts/b12x-rebase/README.md)).

## The control: candidate-55-no974

[no974/](no974) is the same series and sources without `sources.cuda_toolkit`: image
`spark-ds41f:candidate-55-no974`, on CUDA 13.0.2. #974 very likely changes temperature-0 output (new SASS from ptxas
13.5, other cuBLASLt algorithms), so candidate-55's gate alone cannot show that the other four PRs leave outputs
unchanged. The control can: it must give candidate-54's TP4 hashes `18d61961`/`d6568649`.

## Marks, declared before the build

| PR | Own mark | Serving |
| --- | --- | --- |
| #974 | Its sp6 study (`experiments/2026-11-08-cuda-13-5-compiler-cublaslt`) on sp8, candidate-55 against the control, 2 boots each: the `sm_121a` `LDGSTS.E.64` encoding fix (outputs bit-equal on `down_384x576_evict_first`), and the dense MXFP8 kernels and cuBLASLt FP8/BF16 matmuls faster beyond the interval | candidate-55 against the control at `tp4-ring`: grid, c16 and prefill not worse |
| #1134 | Its 8-boot `ABBAABBA` single-GPU screen on sp8 (`bench_one_boot.py`): boot-to-boot spread of the target forward lower and its mean not slower, the control against candidate-54's image | Every boot reaches ready (no memory-floor halt from the 10 GiB `MemAvailable` floor); the screens' two boots no further apart than candidate-54's |
| #1096 | Its release-test selection passes; no B12X module is imported at startup | Inert: the control's gate gives candidate-54's hashes; time to ready not worse (three startup RPC rounds skipped) |
| #989 | Its `bench_pipelined_collectives.py` on sp1..sp4 (TP4) and sp1..sp3 (TP3): collectives of 4-16 MiB faster than PyNCCL beyond the interval, outputs bit-exact | Prefill at 32K and 256K not worse at `tp4-ring` and `tp4-switched` |
| #982 | Its `bench_pdl_direct_stage.py` chains on sp1..sp4, Part A only: chain time at M=1..32 faster beyond the interval; the PDL-chain source test passes | The gate on 12 boots at `tp4-ring` and the c1 decision check: every boot identical. #982 makes more graph edges programmatic, the class of hazard `0799` fixed |

## Verification

- **Build:** one dgx job builds both images on dgx4 and copies them along the ring, then runs the release tests on
  candidate-55's image.
- **Gate:** candidate-55 on 12 boots at `tp4-ring` and 4 at each switched testbed, recording its own hashes (#974's
  declared change); the control on 2 boots at `tp4-ring`.
- **c1 decision check:** 4 boots at `tp4-ring` and `tp4-switched`.
- **Screens:** two boots against candidate-54's recorded screens
  ([candidate-54/receipts/references](../candidate-54/receipts/references), `candidate-54-0799-<testbed>-…`).

## Release tests

[release-tests.json](release-tests.json) is candidate-54's classification plus three gate files the series changes or
adds (`tests/quantization/test_mxfp4_csf.py` and `tests/v1/core/test_contiguous_kv_packing.py` by #1096,
`tests/v1/worker/test_roce_cpu_affinity.py` by #1134), less the exclusion of the deleted
`tests/model_executor/model_loader/test_b12x_loader.py`. `scripts/release_tests.py check` on the prepared tree: 217
selected files (156 gate, 61 excluded), 0 problems; the gate manifest has 157 files, bundled in
[bundles/release-tests](bundles/release-tests). `0803`'s `test_sparknet_oneshot_collectives_pdl_sources` is a gate
test that passes only with sparknet `0003` in the image.

## Status

Assembled (2026-11-11). `bin/spark doctor` passes on every configuration (the switched shapes with the example site
maps), and the unit suite passes in full on dgx4 (889 tests). Next: the build job ([build.json](build.json)) in
dgx1's window.
