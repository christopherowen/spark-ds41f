# Retire B12X from DS4.1 serving (2026-10-26)

**Goal:** port what DS4.1 serving still took from B12X into vLLM, so that B12X can leave the
image and the lock, from the release after next. The next candidate (vLLM `next-candidate`,
1bc005688) is frozen and untouched; this work is on top of it.

**Status (2026-10-26, 09:00 CET):** the vLLM branch and the spark-ds41f tooling are done and
pass the CPU checks below. **No GPU check has run yet:** dgx4 went into the production ring
(the coordinator's re-cabling, 08:10 CET) before any bundle started. The bundles are ready
(`bundles/*-ea60f8afe`); see [Verification](#verification).

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

## vLLM branch `retire-b12x`

Local branch on next-candidate 1bc005688 (worktree `~/projects/vllm-retire-b12x`, shared
object store), head **ea60f8afe**; patches in [vllm/](vllm/series), numbered 0520-0526
provisionally.

1. **0520 6c05a5b01** The packed BF16 head packs with TileLang in vLLM; B12X's packed
   projection goes. `vocab_pack.py`: `PackedBf16Weight`, `pack`, `unpack`, `is_supported`
   (B12X's contract). TileLang passes for B12X's three Triton ones: the exponent histogram
   (shared-memory counts), per-row exception counts (a warp a row), and the row pack (a
   warp a row, each 128-column group as four 32-column parts in column order with a warp
   prefix sum for the exception slots, every value decoded and compared). B12X's host code
   (window choice, offsets, the in-place 1,024-row staging, the round-trip refusal) is
   unchanged. Tests: B12X's packer tests ported, B12X's packed-head integration test
   ported to TileLang; the vocabulary tests no longer `importorskip` B12X.
2. **0521 7370718b2** Engram's token map, geometry and caps in vLLM; resident tables
   gather with TileLang (`engram_table_gather`, B12X's lookup arithmetic and zeros).
3. **0522 0b0ddfed3** MXFP4-CSF leaves the DS4.1 package.
4. **0523 54643f76c** The `b12x` load format and its in-tree adapter go.
5. **0524 b748cec19** Names: `DeepseekV41Metadata`, `DeepseekV41MetadataBuilder`,
   `DeepseekV41Backend` (`"DSV41"`), `"DSV41_COMPRESSOR"`, `cache_dtype_str="dsv41"`,
   `vllm::dsv41_attention`, `vllm::dsv41_prepare_global_kv`.
6. **0525 9ff3d4adb** The DSpark drafter's NVFP4 head switches go.
7. **0526 ea60f8afe** Tests: fifteen tests of B12X's own preparation objects skip without
   B12X (as vLLM's other B12X tests do); the coordinator's other tests run.

41 files, +1181 / -1766. ruff (0.16.6) is clean on every line changed.

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

### Bundles (dgx4, `scripts/lab.py kernel-local`; not yet run)

[make_bundles.py](make_bundles.py) writes them for a vLLM ref over candidate 6's image
(spark-ds41f:19f2c20ed4d6-next-6, tree 6f8e062f = 3bdf9558c): every package file the ref
changes is mounted, every module it deletes is mounted as one that fails to import.

| Bundle | B12X | Checks |
|---|---|---|
| `compare-ea60f8afe` | installed | [pack_compare.py](checks/pack_compare.py): the TileLang packer against B12X's on B12X's test weights (zeros, subnormals, infinities, NaN, out-of-window values, many exceptions), partial in-place blocks, and the 1017 `head.weight` sharded as vLLM shards it at TP1, TP3 (padded rows zero) and TP4, in place and out of place: every field byte for byte, the round trip, the in-place storage; then the TileLang projection's BF16 and FP32 logits from both packed weights, bit for bit. [engram_compare.py](checks/engram_compare.py): the token map of the checkpoint's tokenizer, the geometry of its config, the caps of every TP3 and TP4 rank, and the resident gather against B12X's lookup kernel on synthetic shards and on 65,536 real rows of layer 1's table, bit for bit |
| `checkpoint-absent-ea60f8afe` | uninstalled | the packer alone on the 1017 head (its packed bytes' and logits' SHA-256, against `compare`'s), then TP4 rank 0's direct load and weight digest against 2026-10-25's `a4893a5e...` ([rank_digest.py](checks/rank_digest.py) over `final-names`' rank_load.py) |
| `sweep-next-candidate-1bc005688`, `sweep-retire-b12x-ea60f8afe` | installed, then uninstalled | [import_sweep.py](checks/import_sweep.py): every `vllm` module imported; the failures that appear only without B12X are what needs B12X at import; a meta-path hook records which vLLM module imported each B12X module |
| `gate-absent-ea60f8afe` | uninstalled | the release-test gate (101 files of `config/release-tests.json`) with dgx4's compute-sanitizer |

"Uninstalled" is [without_b12x.py](checks/without_b12x.py): in the throwaway container,
`pip uninstall b12x`, a `PYTHONPATH` without the B12X tree, and a check in a fresh
interpreter that `find_spec("b12x")` is `None`, no `b12x` distribution exists and no
`sys.path` entry holds a `b12x` package. (Earlier bundles over next-candidate dropped the
B12X tree from `PYTHONPATH` only; the site-packages install stayed importable.)

Run order: `compare`, `checkpoint-absent`, the two sweeps, then `gate-absent`, each with
`run-bundles` style queuing on the GPU lock. A real DS4.1 serve needs at least the
triangle; on one node the strongest checks are these (model layers and tests, one rank's
load, the head on the real checkpoint, every module's import).

### Needs the cluster

The release after next's own candidate (its image built without B12X from a lock without
it): start-up, the determinism gate, and the serving digests against the previous release.
B12X's absence changes no kernel: the packed bytes, the head's logits and Engram's tables
are compared bit for bit above, and the disk path is unchanged.

## What still cannot drop B12X

Nothing on the DS4.1 serving path, once the bundles confirm it. Outside it: other models'
B12X kernel families and their tests (they need B12X when used, and import without it),
and the Kimi-K3 and DeepSeek-V4-Flash MXFP4-CSF experts (B12X's fused MoE).
