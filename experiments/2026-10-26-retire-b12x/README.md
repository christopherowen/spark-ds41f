# Retire B12X from DS4.1 serving (2026-10-26)

**Goal:** port what DS4.1 serving still took from B12X into vLLM, so that B12X can leave the
image and the lock, from the release after next. The next candidate (vLLM `next-candidate`,
1bc005688) is frozen and untouched; this work is on top of it.

**Status (2026-10-26, 15:30 CET): done, every dgx2 check passes.** On dgx2 (kernel-local;
the coordinator moved the checks there from dgx4): the packer and the head's logits, and the
Engram helpers, are bit-identical to B12X's on the 1017 checkpoint; with B12X uninstalled,
one TP4 rank's direct load has today's weight digest and the release-test gate passes; and
the import sweeps show that nothing in vLLM needs B12X at import any more. See
[Results](#results).

## Inventory (next-candidate 1bc005688)

Under the recipe: `--load-format direct`, `VLLM_PLUGINS=""`, TileLang for attention,
linear and MoE, Engram tables on disk (`table_memory: disk`), the packed BF16 head
(`VLLM_DS41_PACKED_BF16_LM_HEAD=1`), the drafter sharing the target's head, PCIe all-reduce
off, sparknet collectives.

### What imports B12X at all

Three files import the `b12x` package at module level; every other vLLM import of it is
inside a function.

| File | What | When |
|---|---|---|
| `vllm/models/deepseek_v4_1/common/engram.py` | `b12x.preparation.PreparedCall`, `b12x.sequence.engram` | import: the DS4.1 model imports it, so **DS4.1 does not import without B12X** |
| `vllm/model_executor/model_loader/b12x_loader.py` | `b12x.loader` (capabilities, session, progress, shared read group) | only when `--load-format b12x` asks for it (lazy registration) |
| `vllm/utils/b12x.py` | `find_spec("b12x")`, then `import_module` of 25 B12X submodules | import, only when B12X is installed; without it every accessor returns `None`. The DS4.1 path imports it (the logits processor), so with B12X installed every serving process imports those 25 modules |

### Reached by DS4.1 serving under the recipe

| Stage | Call | Port |
|---|---|---|
| import | `engram.py`'s module-level `b12x` imports | gone (0521) |
| load | `PackedBf16LmHeadMethod.process_weights_after_loading` → `b12x.gemm.packed_bf16_vocab_projection.pack(in_place=True)` (three Triton kernels; B12X patch 0012) | `tilelang/vocab_pack.py`, three TileLang kernels (0520) |
| load | `EngramLayout`: `build_compressed_token_map`, `build_geometry`, `Caps` | `common/engram_geometry.py` (0521) |
| run | none: the head's projection, Engram's hash, gate and disk gather are TileLang's; the B12X preparation pass runs with no unit (only its control exchanges) | - |

The background's list was complete for the recipe. Reached only outside the recipe, and
handled here too:

| What | B12X use | Here |
|---|---|---|
| resident Engram tables (`table_memory` device or ram) | B12X lookup plan, `_bind_lookup_state` and `PreparedCall` through the B12X preparation pass, `run_lookup` | a TileLang gather of a resident shard, declared like the disk table's kernels (0521) |
| packed head with `--linear-backend b12x` | B12X's packed projection (patch 0012 only) | gone: the packed head is TileLang's only (0520) |
| `--load-format b12x` / `VLLM_PLUGINS=b12x_loader` | B12X's loader | gone (0523); `direct` (0499) is its port |
| DS4.1 MXFP4-CSF checkpoints | B12X's fused MoE (`mxfp4_csf.py`) | the method moves to the quantization package for Kimi and DS4-Flash; DS4.1 CSF is refused (0522) |
| `VLLM_DS41_DRAFT_NVFP4_HEAD`, `VLLM_DS41_MARKOV_NVFP4` | B12X's NVFP4 linear kernel | gone (0525); already off and out of 1025's recipe |
| names | `DeepseekV41B12x*`, backend `"B12X"`, `b12x_dsv41`, `vllm::dsv41_b12x_*` | renamed (0524) |

Left alone (generic code, importable without B12X):
- other models' B12X kernel families (Qwen4-exp, GLM-5-next, Kimi-K3, DS4, DS3.2), B12X
  attention backends, the B12X PCIe all-reduce and DCP: all import B12X inside functions;
- the B12X preparation driver and coordinator: DS4.1 declares no unit, so they only run
  their control exchanges and never import B12X;
- naming in generic code: sparknet's communicator is held in `b12x_ar_comm`
  (`cuda_communicator.py`, read by `sp_prefill.py` and the worker's `_b12x_roce_*`
  health checks); `LogitsProcessor.prepare_b12x_vocab_projection` registers TileLang heads
  too.

Which vLLM modules fail to import without B12X is measured by the import sweeps below
(next-candidate and this branch, each with B12X installed and then uninstalled).

## vLLM branch `retire-b12x` / `runtime-vision-b12x-cleanup`

Rebased onto `d5df88d83` (`engram-drafted-prefill`, candidate 14) as `runtime-vision-b12x-cleanup`,
head **190d0d46d** (originally `4672ecc3d` on `1bc005688`); patches in [vllm/](vllm/series),
numbered 0520-0528 provisionally:

1. **0520 f0ca194c6** The packed BF16 head packs with TileLang in vLLM; B12X's packed
   projection goes. `vocab_pack.py`: `PackedBf16Weight`, `pack`, `unpack`, `is_supported`
   (B12X's contract). TileLang passes for B12X's three Triton ones: the exponent histogram
   (shared-memory counts), per-row exception counts (a warp a row), and the row pack (a
   warp a row, each 128-column group as four 32-column parts in column order with a warp
   prefix sum for the exception slots, every value decoded and compared). B12X's host code
   (window choice, offsets, the in-place 1,024-row staging, the round-trip refusal) is
   unchanged. Tests: B12X's packer tests ported, B12X's packed-head integration test
   ported to TileLang; the vocabulary tests no longer `importorskip` B12X.
2. **0521 f4c005e17** Engram's token map, geometry and caps in vLLM; resident tables
   gather with TileLang (`engram_table_gather`, B12X's lookup arithmetic and zeros).
3. **0522 86775e0cc** MXFP4-CSF leaves the DS4.1 package.
4. **0523 c002e7639** The `b12x` load format and its in-tree adapter go.
5. **0524 6b6d35a9f** Names: `DeepseekV41Metadata`, `DeepseekV41MetadataBuilder`,
   `DeepseekV41Backend` (`"DSV41"`), `"DSV41_COMPRESSOR"`, `cache_dtype_str="dsv41"`,
   `vllm::dsv41_attention`, `vllm::dsv41_prepare_global_kv`.
6. **0525 aaf510365** The DSpark drafter's NVFP4 head switches go.
7. **0526 e51cff338** Tests: B12X's own preparation tests (including `test_b12x_output.py`'s
   subprocess replay) skip without B12X, as vLLM's other B12X tests do; the coordinator's
   other tests run.
8. **0527 308b87a6d** Pre-build `direct_io` (`_storage.c`) and `engram_rows` (`_reader.c`)
   `io_uring` C99 helpers into `VLLM_NATIVE_CACHE_DIR` (`/opt/spark-ds41f/native/{direct_io,engram_rows}/<source_digest>`)
   during `docker build` so first boot loads the pre-compiled `.so` directly without invoking
   `cc` into `$XDG_CACHE_HOME`. Custom `CC` / `CUDA_HOME` overrides fall back to the runtime
   cache build path.
9. **0528 190d0d46d** Tensor-parallelize the TileLang vision tower (`DeepseekV4ViT`) and
   aligner (`DeepseekV4Aligner`) across TP ranks unless `is_vit_use_data_parallel()` is set:
   - Column-parallel `wqkv`, `mlp.w1`, and `aligner.w1` with zero-padded tail columns when
     `tp_size` does not evenly divide `16` heads (`6` heads/rank at TP3, `4` at TP4),
     `2,816` MLP width (`960`/rank at TP3, `704` at TP4), or `5,120` aligner width
     (`1,728`/rank at TP3, `1,280` at TP4) so every TileLang GEMM/GEMV dimension stays a
     multiple of 64.
   - Row-parallel `wo`, `mlp.w2`, and `aligner.w2` followed by `tensor_model_parallel_all_reduce`
     (with bias added after all-reduce so it is applied once), recovering ~0.61 GiB/rank at TP3
     and ~0.69 GiB/rank at TP4.

## spark-ds41f: image, lock and tooling

The promoted lock (spark-ds41f-1025's) still lists B12X and builds exactly as before (same
build directory, contexts, stage chain contents, labels and smoke). A lock without a `b12x`
entry builds an image without B12X:
- `docker/Dockerfile`: `runtime-core` (everything but B12X) and `runtime` (B12X on it,
  with its labels and checks); the TileLang stage builds on `TILELANG_BASE_STAGE`.
- `bin/spark`: B12X is a project built only when the lock lists it
  (`LEGACY_BUILD_PROJECTS`); without it there is no `src/b12x` or `b12x-source`, the build
  directory is `vllm-<12>-<hash>`, `build image` passes `TILELANG_BASE_STAGE=runtime-core`,
  a lock without B12X must list TileLang, and the GPU smoke imports neither
  `b12x_mla_sparse` nor `b12x_roce_all_reduce` and asserts `find_spec("b12x") is None`.
- Tests: `tests/test_build.py` (`LockWithoutB12xTest`) and `tests/test_kernel_backend.py`.
- Docs: `docs/upstreams.md` ("Locks without B12X") and `docker/README.md`.

**The lock change for the release after next:** in that release's candidate lock, drop
`sources.b12x`; in its profile, drop `local.spark-ds41f.b12x.tree` from
`container.expected_labels`. Its vLLM series is the next candidate's plus 0520-0526. The
recipe keeps `--load-format direct` and `VLLM_PLUGINS=""`.

## Verification

### Done (CPU)

- The ported geometry and prime test against B12X's source (loaded by path from the locked
  tree, 236ddff + `experiments/2026-10-08-lil-rebase/b12x/series`, tree 1a647a31): equal for
  the checkpoint's Engram config (layers 1 and 14, base 16,000,000, compressed vocabulary
  99,092; table rows 384,006,168 and 384,016,682, the checkpoint's
  `engram_num_embeddings`), the defaults and two other parameter sets; `is_prime_64` equal
  on 5,008 values; the same errors for invalid arguments.
- spark-ds41f unit suite (below).

### Bundles (`scripts/lab.py kernel-local`)

[make_bundles.py](make_bundles.py) writes them for a vLLM ref over candidate 6's image
(spark-ds41f:19f2c20ed4d6-next-6, tree 6f8e062f = 3bdf9558c): every package file the ref
changes is mounted, every module it deletes is mounted as one that fails to import.

| Bundle | B12X | Checks |
|---|---|---|
| `compare-ea60f8afe` | installed | [pack_compare.py](checks/pack_compare.py): the TileLang packer against B12X's on B12X's test weights (zeros, subnormals, infinities, NaN, out-of-window values, many exceptions), partial in-place blocks, and the 1017 `head.weight` sharded as vLLM shards it at TP1, TP3 (padded rows zero) and TP4, in place and out of place: every field byte for byte, the round trip, the in-place storage; then the TileLang projection's BF16 and FP32 logits from both packed weights, bit for bit. [engram_compare.py](checks/engram_compare.py): the token map of the checkpoint's tokenizer, the geometry of its config, the caps of every TP3 and TP4 rank, and the resident gather against B12X's lookup kernel on synthetic shards and on 65,536 real rows of layer 1's table, bit for bit |
| `checkpoint-absent-ea60f8afe` | uninstalled | the packer alone on the 1017 head (its packed bytes' and logits' SHA-256, against `compare`'s), then TP4 rank 0's direct load and weight digest against 2026-10-25's `a4893a5e...` ([rank_digest.py](checks/rank_digest.py) over `final-names`' rank_load.py) |
| `sweep-next-candidate-1bc005688`, `sweep-retire-b12x-ea60f8afe` | installed, then uninstalled | [import_sweep.py](checks/import_sweep.py): every `vllm` module imported; the failures that appear only without B12X are what needs B12X at import; a meta-path hook records which vLLM module imported each B12X module ([sweep_report.py](checks/sweep_report.py) compares the two runs) |
| `gate-absent-ea60f8afe` | uninstalled | the release-test gate (101 files of `config/release-tests.json`) with the host's compute-sanitizer |
| `targeted-absent-4672ecc3d` | uninstalled | the gate's entries for the files the gate run failed, at the final head |

"Uninstalled" is [without_b12x.py](checks/without_b12x.py): in the throwaway container,
`pip uninstall b12x`, a `PYTHONPATH` without the B12X tree, and a check in a fresh
interpreter that `find_spec("b12x")` is `None`, no `b12x` distribution exists and no
`sys.path` entry holds a `b12x` package. (Earlier bundles over next-candidate dropped the
B12X tree from `PYTHONPATH` only; the site-packages install stayed importable.)

A real DS4.1 serve needs at least the triangle; on one node the strongest checks are these
(model layers and tests, one rank's load, the head on the real checkpoint, every module's
import).

### Results

Images: candidate 6's `spark-ds41f:19f2c20ed4d6-next-6` (`sha256:82b071b57485`) with the
branch mounted. Outputs in [runs/](runs/).

**compare (dgx2, `compare-ea60f8afe-r2`, 14:31-14:32 CET, pass, B12X installed).**
- The 1017 `head.weight` ([129280, 5120] BF16, SHA-256 `12f45f83...`) as vLLM shards it,
  packed by vLLM's TileLang packer and by B12X's: every field byte for byte, the round trip
  exact, the in-place pack in the weight's own storage. Exponent window 112..126 on every
  shard.

  | Shard | Rows | Exceptions | B12X pack | TileLang pack | Logits (BF16 and FP32, 1/6/64 rows) |
  |---|---|---|---|---|---|
  | TP1 | 129,280 | 104,397 | 150 ms | 33 ms | 6/6 bit-identical |
  | TP3 rank 0, 1, 2 (rank 2: 128 zero padding rows) | 43,136 | 34,800 / 33,875 / 35,722 | 16 ms | 11 ms | 6/6 each |
  | TP4 rank 0-3, in place | 32,320 | 26,148 / 25,456 / 25,500 / 27,293 | 12 ms | 8 ms | 6/6 each |
  | TP4 rank 0-3, out of place | 32,320 | (same) | 10 ms | 6 ms | - |

  (Warm kernels; a cold first call compiles the three TileLang kernels in about 5.5 s.)
- B12X's own test weights (zeros, subnormals, values far outside the window, infinities, a
  NaN), a partial in-place block (2,500 rows), 1,000 rows at K = 384 with many exceptions, no
  exceptions, all zeros, one exception: byte-identical; the three refusals raise the same
  exception types.
- Engram: the token map of the checkpoint's tokenizer (129,280 entries, compressed size
  99,092), the geometry of its config (table rows 384,006,168 and 384,016,682), the caps of
  every TP3 and TP4 rank and five refusals: equal. The resident gather against B12X's lookup
  kernel: 24 synthetic cases (4 shards, the padded last shard among them; live counts 0, 1,
  9, 16, 17 and -1; NaN and the scale edges) and 65,536 checkpoint rows of layer 1's table:
  bit for bit.

**Import sweeps (dgx2, pass).**

| | modules | fail with B12X | fail without B12X | fail only without B12X | import B12X |
|---|---|---|---|---|---|
| next-candidate 1bc005688 | 2,785 | 49 | 54 | `b12x_loader`, `deepseek_v4_1.common.engram`, and through it `deepseek_v4_1.nvidia.{model,dspark,vl_model}` | `vllm.utils.b12x` (34 modules, at import), `engram` (`b12x.sequence.engram`), `b12x_loader` (19) |
| retire-b12x ea60f8afe | 2,788 | 51 | 51 | none | `vllm.utils.b12x` only (33 modules, at import, when installed) |

The two trees' failures with B12X differ by exactly the two modules the branch deletes. Of
the branch's 51, six are deletion stubs (four for next-candidate's own deletions, two for
this branch's) and 45 are unrelated and the same in both trees: optional packages the image
lacks (Ray, XPU and ROCm kernels, Cohere, gRPC, LMCache, Helion and others), a transformers
version mismatch (11 Pixtral modules) and duplicate custom-op registrations (12). Without
B12X, the branch's only B12X lookup is `vllm.utils.b12x`'s `find_spec`.

**checkpoint-absent (dgx2, `checkpoint-absent-ea60f8afe-r2`, 15:19-15:20 CET, pass, B12X
uninstalled).**
- The packer alone, on the same synthetic weights and 1017 head shards: all 20 cases give
  the same packed values, exception lists and 48 logits hashes as `compare-r2` (with B12X).
- TP4 rank 0 through the direct loader (final-names' `rank_load.py`): weight digest
  `a4893a5ed2f95f53...`, equal to 2026-10-25's from both loaders; 96,081 tensors (71.6 GiB)
  in 23.6 s, 3.26 GB/s; 89/89 sampled tensors equal to safetensors' reads; loader overhead
  0.90 GiB.

**gate-absent (dgx2, `gate-absent-ea60f8afe`, 14:32-15:19 CET, B12X uninstalled, dgx2's
compute-sanitizer mounted).** The release-test gate (101 files): **6,091 passed, 189
skipped, 4 failed.**
- Three tests of B12X's tuning coordinator (`test_b12x_startup.py`:
  `test_all_local_winners_consolidate_once_with_empty_world_rank`,
  `test_cancelled_consolidation_discards_every_partial_winner`,
  `test_cancel_drains_pending_winners_then_orders_collective_warmup`) build B12X's
  `TuningRequirement` through the `_tuning_progress` helper, which my scan of test bodies
  missed: fixed in 0526 (4672ecc3d), the helper skips without B12X.
- `test_deepseek_v41_tilelang.py::test_indexer_reads_only_visible_rows`: not B12X. Its
  compute-sanitizer child (`deepseek_v41_indexer_reads.py`) could not import the candidate
  tree: the release-test bundle sets `PYTHONPATH=/b`, and only pytest's own interpreter adds
  its working directory. Candidate 6's gate bundle mounted no sanitizer, so the test
  skipped there. The bundles now put the vLLM tree on `PYTHONPATH` (still without the B12X
  tree). `scripts/release_tests.py bundle` has the same gap if a sanitizer is ever mounted.
- Targeted rerun of both files at 4672ecc3d (`targeted-absent-4672ecc3d`, 15:21-15:22 CET,
  B12X uninstalled, the fixed `PYTHONPATH`): **pass, 51 passed, 24 skipped.**
  `test_indexer_reads_only_visible_rows` ran compute-sanitizer's initcheck and passed
  (24.7 s); the three coordinator tests skip ("could not import 'b12x'"). With the 6,091
  passes above, the gate passes without B12X at 4672ecc3d.

**Uninstall check.** In every B12X-absent bundle, `without_b12x.py` reported
`Successfully uninstalled b12x-1.3.0`, `find_spec("b12x")` `None`, no `b12x` distribution and
no B12X tree on `sys.path`. The B12X-absent packer (`checkpoint-absent`, synthetic cases)
gave the same packed hashes as the run with B12X.

**Rebased series + prebuilt `io_uring` helpers + TP vision tower (dgx4, 2026-10-30, pass, B12X uninstalled).**
- Image: `spark-ds41f:19f2c20ed4d6-next-14` (`sha256:7918fd6145b5`) with `0520..0528` (`190d0d46d`) mounted over `d5df88d83` and `b12x-1.3.0` uninstalled by `without_b12x.py`.
- Both `direct_io._native.prebuild()` and `engram_rows._native.prebuild()` built into `VLLM_NATIVE_CACHE_DIR`, verified `liburing.so` linkage via `ldd`, and resolved via `load()` with an empty `XDG_CACHE_HOME` (`xdg_compiled_so: []`).
- 8 gate test files (`test_b12x_startup.py`, `test_b12x_output.py`, `test_deepseek_v4_1_engram.py`, `test_deepseek_v41_engram_rows.py`, `test_deepseek_v41_tilelang_engram.py`, `test_direct_loader.py`, `test_deepseek_v41_tilelang_vision.py`, `test_deepseek_v41_tilelang_vocab.py`): **276 passed, 25 skipped, 0 failed, 0 errors** in 154.8 s (`exit: 0`).

**Interrupted (recorded, no result).** dgx4 08:03: `pack-dev` stopped by the planned
power-off. dgx4 13:50 `compare-ea60f8afe` and dgx2 14:29 `checkpoint-absent-ea60f8afe`: my
check script moved safetensors' mmapped `head.weight` straight to the GPU, which hangs on
GB10 (R580, 64K pages: large copies from untouched mmap pages never complete; the host spins,
the GPU idles); I killed both containers. The `-r2` bundles clone the tensors first, as
vLLM's `copy_weight` does. (The synthetic cases before the hang had all passed.)

### Needs the cluster

The release after next's own candidate (its image built without B12X from a lock without
it): start-up, the determinism gate, and the serving digests against the previous release.
B12X's absence changes no kernel: the packed bytes, the head's logits and Engram's tables
are compared bit for bit above, and the disk path is unchanged.

## What still cannot drop B12X

Nothing on the DS4.1 serving path (the end-to-end serve without B12X is the release after
next's own candidate on the cluster). Outside it: other models'
B12X kernel families and their tests (they need B12X when used, and import without it),
and the Kimi-K3 and DeepSeek-V4-Flash MXFP4-CSF experts (B12X's fused MoE).
