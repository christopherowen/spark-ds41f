# #1096 rebased onto candidate-54 and #1134 (`0802`), and #982's test entry (`0803`), 2026-11-11

Prepared on dgx3 (`~/spark-lab/agents/candidate-55-vllm/vllm`, branch `candidate-55`) from candidate-54's prepared tree;
tested on sp8 under `flock /tmp/spark-lab-gpu.lock` in throwaway `spark-ds41f:candidate-54` containers.

| State | Commit | Tree |
| --- | --- | --- |
| candidate-54 | `15833ed1` | `ad2dc11f` (the lock's) |
| + `0800` (#1134) | `e3548734` | `5edb3933` |
| + `0801` (#1134) | `bec51821` | `1d71d7fb` (#1134's recorded tree) |
| + `0802` (#1096) | `bb71f454` | `89341a98` |
| + `0803` (#982) | `ef670e33` | `980e3812` |

`0800` and `0801` apply with plain `git am`. Replaying the four files on a fresh checkout of `15833ed1` reproduces every
tree. `0802` keeps #1096's author date and message, with its base line changed and one paragraph naming the
resolutions below.

## 0802: the rebase

`git am -3` merged 11 files three ways; one had a textual conflict. Every other hunk landed as #1096 wrote it,
including those next to candidate-54's and #1134's changes in `direct_loader.py` and `gpu_worker.py`.

1. **`sp_prefill.py`, a textual conflict:** candidate-54 (`0778`/`0779`) had already replaced the inline
   `b12x_ar_comm` lookups with `_tp_roce_comm()` and `_roce_should_exchange()`. candidate-54's code is kept, and
   #1096's intent goes inside the helper: `_tp_roce_comm()` prefers `roce_ar_comm` and falls back to `b12x_ar_comm`.
   No runtime change: `cuda_communicator.py:215` sets both to the same `SparknetOneShotAllReduce`.
2. **`attention.py:1155-1157`, merged cleanly but broken:** candidate-54's `_forward_decode_overlapped` (#1012)
   annotates with `DeepseekV41B12xMetadata`, which #1096 renames to `DeepseekV41Metadata`; without postponed
   annotations the import raises NameError. The three annotations are renamed.
3. **`logits_processor.py:205`, a missing alias:** #1096's description keeps `prepare_b12x_vocab_projection` as an
   alias, but its patch only renames the method, and callers remain (`deepseek_v4/nvidia/dspark.py:546`,
   `glm5next/nvidia/mtp.py:165`, `tests/models/test_deepseek_v4_1_prepared_projections.py:45`). The alias is added.

Checks: `git diff --check` clean for both patches; `py_compile` of the 38 changed Python files; pyflakes shows the same
9 pre-existing unused imports before and after and no undefined names; no code reference to `DeepseekV41B12x*`,
`b12x_dsv41`, `B12X_DSV41_COMPRESSOR`, `lookup_plans`, `deepseek_v4_1.mxfp4_csf`, `DeepseekV41Mxfp4CsfConfig` or
`b12x_loader` remains.

## What changes at run time

No B12X kernel on candidate-54's TileLang serving path is replaced. Beyond renames, comments and dead fields:

- **`tilelang/engram_rows.py:524-531`** (`engram_resident_lookup_kernel`, on the serving path): a row is owned when
  `0 <= n_tok <= tokens` and `t < n_tok`, where it was `t < num_tokens[0]`. Bit-identical inside that range; outside
  it every row is now zero (before, all rows were looked up when `n_tok > tokens`). The disk gather kernel already
  used this bound. The kernel recompiles.
- **Startup:** the B12X preparation phases run only when `has_b12x()` is true on SM12x
  (`v1/executor/abstract.py:186,213,413`, `warmup/b12x_prepare.py:551,680`, `v1/worker/gpu_worker.py:640`). They
  matter in candidate-55, whose lock has no B12X.
- **`tilelang/prepare.py:165-171`:** `declare_modules` binds `lm_head` to its logits processor (idempotent with B12X;
  the only declaration without it).
- **`utils/b12x.py`:** its 26 submodules load on first use, not at import.
- **Strings:** the DS4.1 backend name "B12X" → "DSV41", the compressor's "DSV41_COMPRESSOR", `cache_dtype_str`
  "b12x_dsv41" → "dsv41"; nothing in `vllm/` compares them.
- **Refused:** `--load-format b12x` and MXFP4-CSF DS4.1 checkpoints (`quantization/mxfp4_csf.py:31-37`); neither is
  served.

**Remaining references:** in DS4.1 only comments, docstrings, the C include guard `B12X_ROW_PLAN_H` and the two
fallbacks above; in `model_loader` the direct-I/O port's Apache-2.0 provenance headers (license notices, kept) and a
function-local import in the Kimi/DS4-Flash `mxfp4_csf_loader.py`. An AST scan of 1,110 files finds module-scope B12X
imports only under `if TYPE_CHECKING:`.

## 0803

#982's raw diff applies with `git apply` (hunk 1 lands after candidate-54's `gemm_packed_bf16_*` entries). It adds the
four one-shot collectives to `CHAIN` in `tests/kernels/deepseek_v41_pdl_sources.py` and
`test_sparknet_oneshot_collectives_pdl_sources` to `tests/kernels/test_deepseek_v41_tilelang_pdl.py`, a gate file:
it passes only on an image whose sparknet carries sparknet `0003`'s PDL calls, as candidate-55's and the control's do.

## Tests (sp8)

The image's tracked `vllm/` and `tests/` files (5,657) match candidate-54's blobs; the tree's 51 changed files were
copied over `/opt/spark-ds41f/candidate/vllm` and its 3 deleted files removed. #1096's 19-module selection, rebuilt
from its README's categories, plus `tests/v1/worker/test_roce_cpu_affinity.py`:

| Run | Result | Time | Log |
| --- | --- | --- | --- |
| With B12X (as candidate-54's image ships it) | 1,108 passed, 0 failed, 8 skipped (offline Qwen, 2- and 4-GPU tests, libvterm) | 860 s | [c55-with-b12x.log](c55-with-b12x.log) |
| Without B12X (`has_b12x()` false) | 1,060 passed, 0 failed, 56 skipped (the 48 more need B12X) | 376 s | [c55-no-b12x.log](c55-no-b12x.log) |

All 81 modules these patches touch import in both modes. A first "without" attempt left a second B12X copy importable
and was in effect another with-B12X run (1,108 passed); its log is kept
([c55-without-b12x-attempt1-b12x-still-present.log](c55-without-b12x-attempt1-b12x-still-present.log)).

Not run here: `0803`'s test, the release gate and the image build; they run on candidate-55's image.
