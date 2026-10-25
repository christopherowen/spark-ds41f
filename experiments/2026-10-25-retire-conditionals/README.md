# Retire the conditionals: TileLang is the DS4.1 model's only path

Owner, 2026-10-25: with the TileLang migration complete, remove the dead conditionals and
fallback paths from the DS4.1 serving path, for the release after candidate 6. "No fallback
paths that never run." Nothing here ran on a GPU, and nothing touches the frozen candidate
(its series, image, arms and recipe are read, never written).

## What it is

- **vLLM:** branch `retire-conditionals` (local, `~/projects/vllm-retire-conditionals`),
  four commits on candidate 6's vLLM (`3bdf9558c`: `tilelang-ports` 26382f5cf plus rc6's
  0489-0490, tree `6f8e062f`). Head `b9d7e37b6`, tree `4377c31d`. 56 files, +1020 / -9953
  lines. The patches below are the branch, and they replay onto `3bdf9558c` to the same tree.

  | patch | commit | what goes |
  |-------|--------|-----------|
  | [0493](vllm/0493-DS4.1-TileLang-is-the-model-s-only-kernel-family-the.patch) | a9b9addd2 | the B12X model family |
  | [0494](vllm/0494-DS4.1-No-kernel-family-switch-VLLM_DS41_KERNEL_BACKE.patch) | 231959e4c | `VLLM_DS41_KERNEL_BACKEND` and every engine branch it gated |
  | [0495](vllm/0495-DS4.1-The-ported-model-kernels-Triton-originals-go.patch) | 96d7d0b2f | the Triton originals of the ported model kernels |
  | [0496](vllm/0496-DS4.1-B12X-remnants-the-L2-prefetch-packed-weight-wa.patch) | b9d7e37b6 | B12X remnants: the L2 prefetch packed-weight walk |

  [vllm/series](vllm/series) is candidate 6's series (375 patches, through 0490) followed by
  0493-0496 (379). 0491 (KV capacity) and 0492 (drafter window) are taken by other drafts and
  0497 by final-names; this series does not depend on them.
- **spark-ds41f:**
  - [scripts/kernel_backend.py](../../scripts/kernel_backend.py): a new vLLM capability,
    `tilelang-only`, for a series without the switch. Such a series refuses
    `VLLM_DS41_KERNEL_BACKEND` in the environment. A series without the capability
    (candidate 6 and earlier) still requires it to be `tilelang`.
  - [bin/spark](../../bin/spark): the image build's GPU import smoke skips
    `b12x_roce_all_reduce` for a `tilelang-only` series, since that series deletes the
    module.
  - [tests/test_kernel_backend.py](../../tests/test_kernel_backend.py): both cases, and
    the smoke.
  - [docs/doctor.md](../../docs/doctor.md), [docs/testing.md](../../docs/testing.md):
    the contract.
  - [config/release-tests.json](../../config/release-tests.json):
    - the three deleted test files and the six deselected B12X tests leave;
    - `tests/quantization/test_register_quantization_config.py` (changed by the series)
      joins the gate;
    - `release_tests.py check` at the branch: 152 selected, 97 gate, 55 excluded,
      0 problems;
    - against candidate 6's own tree, the three deleted files show as unclassified, as
      they should: candidate 6 keeps its frozen manifest
      ([rc6/release-tests-gate-manifest.json](../2026-10-25-spark-ds41f-candidate/rc6/release-tests-gate-manifest.json)).

## What goes

Request inventory 5 (`request-inventory-5`, every request type profiled on the ports
build) found no Triton, CuTe or FlashInfer kernel in any request type
([tilelang-completion README](../2026-10-24-tilelang-completion/README.md), "The JIT
monitor's cost" and "Startup without unused kernels"). Every recipe shape sets
`VLLM_DS41_KERNEL_BACKEND=tilelang` with the TileLang attention, linear and MoE backends.
Candidate 5's window (run `candidate-5-tp4`, candidate 6's package files) logged no B12X
preparation, no temporary KV blocks and no JIT monitor.

| site | why it is dead | evidence |
|------|----------------|----------|
| `VLLM_DS41_KERNEL_BACKEND`: `envs.py`, `kernels.uses_tilelang()`, the DS4.1 config check, `CudaCommunicator`, the scheduler's `ALIGN_PREFILL_CHUNKS`, `gpu_worker`'s preparation and startup progress | serving always sets `tilelang`; the `b12x` default is never served, so each conditional collapses to its TileLang branch | the recipe's environment and `kernel_backend`; `test_every_recipe_shape_runs_tilelang` |
| `b12x_layers.py` (976 lines: B12X linear, FP8, embedding methods, `B12xMHC`, `B12xRMSNorm`), `nvidia/b12x_vision.py` (809), `nvidia/b12x_attention.py` | the TileLang family implements every component; nothing imports these under TileLang | inventory 5: no CuTe kernel; the GPU check's import-failing stubs (below) test that nothing imports them |
| `attention.py`: helper plans and `_AttentionHelpers`, the B12X sparse MLA, indexer and WO plans, preparation units and calls, B12X RoPE, cache writes, index-weight scaling, the WO pack and projection | `DeepseekV41TileLangAttention` overrides each hook; B12X preparation declared no unit under TileLang (vLLM 1bd5f0c59) | candidate 5's window: no B12X preparation line |
| `compressor.py`: B12X projection and compression plans, the primer | the TileLang compression applies the norm weight itself; the norm is a weight holder | same |
| Engram (`common/engram.py`): the B12X hash (metadata kernel, bindings, hash plans), the B12X mix (plans, boolean masks), `SPRows.pack_columns` | TileLang's hash and gate run on the step's token ids | inventory 5 |
| `dspark.py` `_ContextKVProjection`; `model.py` B12X layer registry and mHC providers; `vl_model.py` B12X tower | `TileLangBlock32Rows` is the projection; only `B12xMHC`'s custom ops read the registry; `tilelang_vision` is the tower | inventory 5 |
| `kernels.MOE_FP32_COMBINE`, `kernels.SHARED_SWIGLU_FP32` | always-on constants; their off arms never ran | code |
| `VLLM_DS41_ATTENTION_COMPUTE` | read only by the B12X sparse MLA | candidate recipe already dropped it |
| `b12x_roce_all_reduce.py`, PyNCCL's all-reduce past the one-shot limit, `VLLM_ROCE_ALLREDUCE_MAX_SIZE`, `VLLM_ROCE_ALLGATHER_MAX_SIZE` | selected only by `VLLM_ENABLE_ROCE_ALLREDUCE=1` outside TileLang; TileLang groups always build sparknet's one-shot collectives with rank-order all-reduce and reduce-scatter | sparknet transports set the variable to 0; it is now refused when set |
| Triton originals: `sparse_mla.py` `_requests`, `_tokens`, `_chunk`; `attention.py` `_pages` (`triton.cdiv` becomes vLLM's `cdiv`); `compressor.py` `_load_partial_state`, `_save_partial_states`; `model_state.py` `_gather_lookback_kernel` | ported to `tilelang.metadata`, `tilelang.compress`, `tilelang.lookback`; dispatch was TileLang-only | inventory 5: no Triton kernel |
| `l2_prefetch.object_segments` and `linear_segments`' `b12x_weight` | only B12X linear methods and the B12X WO pack set `b12x_weight` | code |

Tests:
- **Deleted** (only the removed paths):
  - `test_deepseek_v4_1_rmsnorm.py` (B12X RMSNorm);
  - `test_deepseek_v4_1_vision.py` (B12X tower);
  - `tests/v1/attention/test_b12x_v41_workspace.py`;
  - `benchmark_tilelang_vision.py` (timed against B12X);
  - B12X DS4.1 tests in `test_b12x_linear.py`, `test_dspark_mla.py` and
    `test_deepseek_v4_1_prepared_projections.py`. The generic B12X tests there stay;
    the files stay excluded.
- **Ported to TileLang:**
  - the compressor's rejection, reorder, page-boundary and high-offset scenarios (two of
    the six former deselects; the other four were B12X attention-plan tests, which go).
    Their padding is the runner's: rows past the step's requests, which must write
    neither latents nor the ring of a stale table row's pages. The B12X form, a request
    owning rows without slots, is not a step serving builds: every row a request owns
    has a slot (candidate 6's ring-writes test, and the runner's block tables);
  - the Engram model tests (token ids, lookup-only preparation, a gate graph with the
    image sentinel), on a 1024-wide model: the TileLang gate's smallest width;
  - the SP Engram gate, the attention staging tests and the compressor-preparation tests.
- **Vision bounds:** candidate 6's vision test bounded each FP64 comparison by B12X's
  error on the same inputs. B12X's tower is gone, so its errors are numbers in the test
  (`B12X_ERROR`, 38 of them), measured with candidate 6's own test on its image
  ([bundles/vision-b12x-error](bundles/vision-b12x-error)). Each bound is the tighter of
  candidate 6's B12X-relative one and a fixed one. The fixed ones alone were the base
  test's never-run no-B12X branch: four BF16 steps for attention, where B12X itself
  measured 10.7-14.8.
- **Oracles moved to torch:** the metadata kernels (requests, tokens, chunk, pages,
  decode metadata), the state-ring gather and the lookback window. Before the Triton
  went, each torch reference was checked against a transliteration of the deleted
  Triton on the tests' inputs (70 metadata cases, 40 ring cases).
- **No B12X oracle is left for a TileLang kernel**, except for the kept Engram pieces
  and the packed-head format below.

## Kept: serving reaches them

- **b12x's Engram geometry, token map and caps.** The TileLang Engram reads their layout.
- **The generic B12X preparation pass** (`b12x_prepare.collect_b12x_units`). It binds
  the TileLang vocabulary heads.
- **The shared V4.1 cache layout and metadata classes** (`DeepseekV41B12x*` in
  `sparse_mla.py`). TileLang subclasses them, and the names are upstream's.
- **The packed BF16 head's three load-time Triton kernels** (`TRITON_CACHE_DIR` stays).
- **The B12X checkpoint loader.** Production loads through B12X's own `B12xModelLoader`
  plugin (final-names' finding), not vLLM's in-tree `b12x_loader.py`.

## Upstream files left alone

These hold Triton that serves other features or models, including the four `HAS_TRITON`
sites in the brief. The churn column counts upstream commits in the 30 days before the base
(19f2c20ed, 2026-10-08).

| file | what the guard or Triton serves | churn |
|------|---------------------------------|------:|
| `v1/worker/gpu/sample/gumbel.py` (lines 16, 22) | Module constants of Triton device functions (`tl_rand32`, `spec_step_seed`, `gumbel_noised_argmax`, `gumbel_block_argmax`). The watermarker, DFlash2's speculator, vocabulary-parallel drafting and the rejection sampler's Triton references still call them. The sampler itself runs TileLang: `gumbel_sample` and `apply_temperature` dispatch to `tilelang_sampling`. `_temperature_kernel` and `_gumbel_sample_kernel` launch nowhere; they are the TileLang sampling tests' bit-exact oracles (judgment call 5). | 1 (series: 5) |
| `v1/worker/gpu/boundary_checkpoint.py` (lines 31-43) | the V2 runner's boundary-checkpoint copies (Mamba and hybrid state); not launched for DS4.1 (inventory 5) | 4 |
| `utils/jit_monitor.py` (line 102) | the monitor's Triton autotune hook; the candidate recipe runs `--jit-monitor-mode off`, which installs no hook (judgment call 6) | 0 |
| `distributed/parallel_state.py:2585` | shutdown clears the V1 Triton top-k/top-p buffer cache when Triton is present: the V1 sampler of other models | 8 |

Upstream rebase cost of this series: future upstream edits to these files conflict.

| change | churn |
|--------|------:|
| `b12x_layers.py` deleted (modify/delete on every upstream edit) | 4 |
| `nvidia/b12x_vision.py` deleted | 3 |
| `nvidia/b12x_attention.py` deleted | 1 |
| `b12x_roce_all_reduce.py` deleted | 2 |
| `VLLM_DS41_ATTENTION_COMPUTE`, `VLLM_DS41_KERNEL_BACKEND`, the RoCE sizes (`envs.py`) | 37 |
| `v1/core/sched/scheduler.py` (three sites read `align_prefill_chunks`) | 26 |
| `v1/worker/gpu_worker.py` | 15 |
| `model_executor/models/config.py` | 11 |
| `nvidia/model.py`, `attention.py`, `common/engram.py` | 10, 10, 9 |
| `nvidia/dspark.py`, `vl_model.py`, `compressor.py`, `cuda_communicator.py` | 6, 5, 5, 5 |
| `nvidia/model_state.py`, `sparse_mla.py`, `config/scheduler.py`, `registry.py` | 3 each |
| `l2_prefetch.py` | 1 |

## Judgment calls

1. **`b12x_roce_all_reduce.py` deleted, not kept for other models.**
   - Upstream's own launch scripts (`scripts/serve-glm53-flash-nvfp4-tp2-rdma.sh`,
     `serve-ds4-flash-dspark-tp{2,4}-rdma.sh`) set `VLLM_ENABLE_ROCE_ALLREDUCE=1`. On
     this series they stop at startup with an error that names the cause.
   - Recommendation: keep it deleted. This fork serves DS4.1 only, and keeping the
     module would keep a model-dependent communicator branch that never runs here.
2. **`VLLM_ENABLE_ROCE_ALLREDUCE=1` is refused, not ignored.** Silently swapping
   collectives under a setting that asks for others is worse than a startup error.
   Recommendation: keep. The RoCEnante transports in `scripts/topology.py` set it; their
   removal is the RoCEnante session's work.
3. **Aligned prefill chunks became `SchedulerConfig.align_prefill_chunks`**, which the
   DS4.1 config sets. They are not unconditional because generic schedulers (other
   models, and test schedulers without `long_prefill_token_threshold`) break when they
   are on. Recommendation: keep.
4. **The B12X lookup of resident Engram tables (`table_memory` device or ram) is kept.**
   - Production uses `disk`, so this path never serves, but `device` is vLLM's default
     for the model.
   - Recommendation: port the lookup to TileLang, then delete the B12X call. The lookup
     is a row gather times its E8M0 scale, rounded to BF16: the disk reader's TileLang
     rows compute the same thing, and `test_deepseek_v41_engram_rows.py` already holds a
     torch model of it. The alternative is to refuse device and ram for DS4.1.
5. **The V2 runner's Triton that is left** (`gumbel.py`'s references and their users).
   - Every model's sampler already runs TileLang.
   - Recommendation: one follow-up, "V2 runner without Triton":
     - port or delete the watermark, DFlash2 and vocabulary-parallel Triton kernels;
     - give the TileLang sampling tests a torch Philox reference, checked on the nodes
       against the Triton kernels while they still exist;
     - then delete `gumbel.py`'s Triton and its guards.
   - The cost is mostly ours: upstream changed gumbel.py once in the 30 days before the
     base, and the series changed it five times.
6. **The JIT monitor stays** as a debug option. Production runs it off, which installs
   no hooks, as other checks run as debug flags that are off in production. Its upstream
   default (`warn`) is left alone. Recommendation: keep.
7. **The `DeepseekV41B12x*` class names stay.** Renaming them is cosmetic and costs
   rebase conflicts. Recommendation: rename only if upstream does.
8. **The packed head's load-time Triton stays.** It is the last Triton in the
   deployment, and it runs only at load. Recommendation: port it with the vocabulary-head
   work.

## Overlap with the other drafts

A trial merge (`git merge-tree`) against the open vLLM branches:

- **final-names** (8ee55f901): merges cleanly.
- **kernel-audit** (8e596b329): conflicts in four files.
  - Files: `attention.py`, `sparse_mla.py`, `tilelang/metadata.py` and the metadata test.
  - Cause: its new `pages`, decode-metadata and builder signatures sit next to the
    removed Triton branches, and its tests still use the deleted Triton
    `_requests` / `_tokens` / `_pages` as oracles.
  - Fix for whichever lands second: keep kernel-audit's kernels, drop the Triton branches,
    and move its oracles onto this branch's torch references (`_ref_requests`,
    `_ref_tokens`, `_ref_pages`, which `pages`' live-page contract may need to follow).
- **moe-overlap** (1c458d618): modify/delete on `b12x_layers.py` and
  `test_b12x_v41_workspace.py`.
  - Its `B12xMHC` changes are moot once the family is gone.
  - Its auto-merged `kernels.cast_mxfp8` and two `nvidia/model.py` sites still call
    `kernels.uses_tilelang()`, which this branch removes. Whichever lands second
    collapses them to the TileLang branch.

## Recipe change for the release after candidate 6

When a series includes 0493-0496:
- **Recipe:** drop `VLLM_DS41_KERNEL_BACKEND` from the environment.
- **Source manifest:** add `tilelang-only` to `vllm.capabilities`. Doctor and preflight
  then refuse the variable. Before the capability is added, they still require it, so a
  half-applied change fails either way.
- **Image build:** its import smoke drops `b12x_roce_all_reduce` for such a series.
- **Not done here:** `config/cluster.json` and the promotion branch are frozen.

## Verification done (laptop, CPU only)

- **vLLM tests** (at 3082ef9bf, before dgx4's test fixes): the touched and related test files on candidate 6's tree and on the
  branch, CPU venv (no CUDA, Triton or TileLang; b12x 236ddff on the path, TileLang
  stubbed):
  - 27 files. Candidate 6's tree: 458 passed, 61 failed, 161 skipped, 2 files that do not
    collect (the config and Engram tests import cuda-python through the B12X family).
    The branch: 489 passed, 49 failed, 83 skipped, every file collects.
  - No test that passes on candidate 6's tree fails on the branch. The tests that leave
    are the deleted B12X ones and the renamed or re-parametrized ones, whose successors
    pass. 36 new or ported tests pass. 9 that failed on the laptop now pass.
  - The 46 failures common to both trees are the laptop's. 39 are `test_scheduler.py`'s
    CPU platform config. The rest are CUDA-only setups in the excluded B12X files and a
    Hugging Face download, which the gate's offline plugin reports as a skip.
  - The branch's 3 other failures are `test_deepseek_v4_1_config.py`, whose hook applies
    only on SM12x. With a per-test SM12x platform shim all 3 pass: the config hook, the
    attention class and aligned chunks. On candidate 6's tree that file does not collect
    on the laptop.
- **Patches:** 0493-0496 replay onto `3bdf9558c` to tree `4377c31d`, the branch's.
- **spark-ds41f:** `python3 -m unittest discover -s tests`: 485 tests OK (5 skipped).

## GPU checks on dgx4

Owner, 2026-10-25: sub-agents test on dgx4 (uncabled, serving nothing) while dgx1-3
serve. Every run used `scripts/lab.py kernel-local BUNDLE --out OUT` on dgx4 itself.
- **Checkout:** this branch, under `~/spark-lab/agents/retire-conditionals/spark-ds41f`.
- **Outputs:** `~/spark-lab/agents/retire-conditionals/out/<run>`. Per-test JUnit files
  are copied there from `~/.cache/spark-lab/compile/release-tests/<label>`.
- **Image:** `spark-ds41f:19f2c20ed4d6-next-6` (`sha256:82b071b57485`), as it is.
- **Request file:** `~/spark-request.json` was absent before each run.
- **Targeted runs:** [make_targeted.py](make_targeted.py) copies the checkout's release-test
  bundle with only the named gate files, under its own label.

| run (CET) | commit | bundle | verdict |
|-----------|--------|--------|---------|
| release-tests, 19:01-19:54 | 289f38f4 (vLLM 3082ef9bf) | [bundles/release-tests](bundles/release-tests), 97 files | **FAIL**: 7 failed, 2 errors, 5712 passed, 161 skipped |
| vision-b12x-error, 20:20 | 174c5d98 | [bundles/vision-b12x-error](bundles/vision-b12x-error) | FAIL in 1 s: the image has no `tblib`, which the conftest imports (bundle fixed to load the release tests' stand-in) |
| targeted-1, 20:28-20:35 | 174c5d98 (vLLM 47a849319) | the metadata, compressor and Engram files | FAIL: 1 failed (compressor padding), 124 passed |
| vision-b12x-error, 20:36-20:40 | 7c3f538c | [bundles/vision-b12x-error](bundles/vision-b12x-error) | pass: 23 passed, 38 B12X error records |
| targeted-2, 20:50-20:51 | 354066f8 (vLLM b9d7e37b6) | the vision, metadata, compressor and Engram files | pass: 206 passed |
| release-tests-2, 20:52-21:17 | 354066f8 (vLLM b9d7e37b6) | [bundles/release-tests](bundles/release-tests), 97 files | **pass**: 5721 passed, 161 skipped, 0 failed (1476 s) |

The first run's failures, all in test files this branch changed. Serving code was not
involved: the same TileLang kernels run in candidate 6.
- **Metadata graph refresh, compressor setup (1 failed, 2 errors):** the tests'
  `SimpleNamespace` metadata lacked `query_start_loc_cpu`, which the builder's
  decode/prefill split reads. The graph-refresh test was in an excluded file before this
  branch, and the compressor tests were deselected. The namespaces now carry the host
  copy.
- **Vision attention (3 failed):** the fixed four-step bound (see "Vision bounds"). With
  B12X's numbers TileLang's worst elements are 8.8 / 8.8 / 12.7 steps against B12X's
  10.7 / 14.8 / 12.5, inside candidate 6's rule (B12X's + 1).
- **Engram model tests (3 failed):** the tests' 64-wide model reaches the TileLang gate,
  which asserts a multiple of 1024. In candidate 6 they ran B12X's gate. The model is now
  1024 wide.
- **Compressor padding (targeted-1):** the TileLang compression saves the state ring for
  every row its ring metadata assigns to a request. The B12X-era scenario gave a request
  rows without slots, and the TileLang ring ignores the input slots (as the deleted
  Triton `_tokens` did for ring groups). That step does not occur in serving. The
  scenario now pads as the runner does, and the pages stay untouched.

The release-test gate passes on the branch with its deleted modules failing to import:
no gate test reaches them.

## Serving check (queued, not run)

[retire-check.json](retire-check.json), written by [make_bundles.py](make_bundles.py). Its
kernel job repeats the release-test gate above on dgx2; its measure job needs the cluster.
- **Image:** `spark-ds41f:19f2c20ed4d6-next-6` as it is, with nothing built.
- **When it can run:** the lab runs origin/main's specs, so it runs after this PR merges.
  The owner can also run it outside the repository, as candidate 6's test preview ran.

1. **sync.**
2. **kernel:** [bundles/release-tests](bundles/release-tests) on dgx2.
   - Runs the release-test gate at the branch head
     ([release-tests-gate-manifest.json](release-tests-gate-manifest.json), 97 files).
   - The branch's 27 package files are under `overlay/vllm`.
   - Each of its 4 deleted modules is replaced by a stub that raises
     `ModuleNotFoundError`, so anything still importing one fails as on the branch.
   - Its 21 test and benchmark files are mounted over the image's copies.
   - **Pass:** no failed test.
3. **measure** (lean, no bench): [retire-tp4.json](retire-tp4.json) then candidate 6
   ([rc6/rc-tp4.json](../2026-10-25-spark-ds41f-candidate/rc6/rc-tp4.json)), both at
   tp4-ring.
   - The retire arm mounts the same overlay over candidate 6. It keeps
     `VLLM_DS41_KERNEL_BACKEND`, which candidate 6's manifest requires and the branch's
     vLLM no longer reads (unknown `VLLM_` variables only warn).
   - Extras: temperature-0 determinism with log probabilities and without (greedy),
     seeded temperature 1, the generation controls, prompt logprobs, images, startup
     lines.
   - [tables_arms.py](tables_arms.py) passes only when:
     - each arm is identical within itself at temperature 0;
     - against candidate 6, the retire arm's digests are equal: temperature 0 (the eight
       prompts and the long prompt, with log probabilities and as text), the seeded
       temperature-1 reference, every control digest except `penalties-sampled/c8`
       (which differs between boots of one build, so it is only recorded), prompt
       logprobs and images;
     - the retire arm's startup has no B12X preparation, temporary KV block or JIT
       monitor line, and the same TileLang call count.
   - Candidate 5's window digests (candidate 6's package files), for the record:
     `3eab5112` / `db25fde3`, text `c9cf9dbe` / `c6bb4f4e`, seeded `80b3b3f3`, images
     `92679390`.
   - **TP3:** an arm over `rc6/rc-tp3.json` can join once #582 merges; that arm is not on
     main yet.
