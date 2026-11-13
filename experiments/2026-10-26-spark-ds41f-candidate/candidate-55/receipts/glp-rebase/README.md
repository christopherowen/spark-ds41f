# #738 rebased onto candidate-55 (`0805`), 2026-11-12

#738's patch ([0001-dsv41-glp-specv3-fold-steering.patch](../../../../2026-10-31-glp-specv3-fold-steering/0001-dsv41-glp-specv3-fold-steering.patch),
PR head `13d55574`) carried as vLLM `0805`, after `0804`. Prepared on dgx3 (`~/spark-lab/agents/candidate-55-vllm/vllm`,
branch `candidate-55-r4` from `candidate-55-r3`, the third build's tree):

| State | Commit | Tree |
| --- | --- | --- |
| candidate-55's third build (`0800`..`0804`) | `6517d177` | `e8c20f54` |
| + `0805` (#738) | `0925deae` | `8aaae3df` (the lock's, recorded on dgx4) |

A plain `git am` of the exported `0805` on a fresh branch from `6517d177` gives the same tree. `0805` keeps #738's
author and date (Matt Suiche, 2026-11-01, co-authored by Christopher Owen). Its message adds the upstream base,
rationale, quality implications, tests and upstream status, and names the resolution below.

## The resolution

Five of the six files apply unchanged (`glp.py`, `tilelang/glp.py` and the three test files). `nvidia/model.py`
rejects two hunks, and both rejections come from context only: candidate-54 added an `aux_out` keyword where #738
adds its own.

1. **`DeepseekV4DecoderLayer.forward`:** `glp=None`, `layer_idx: int = 0` and `steer_rows=None` follow `aux_out`
   among the keyword-only parameters.
2. **`DeepseekV4Model.forward`, the layer call:** `glp=self.glp`, `layer_idx=idx` and `steer_rows=steer_rows` follow
   `aux_out=pending_aux_out`. The re-compaction of `steer_rows` after the CED boundary layer is removed, as #738
   removes it. Its two other `nvidia/model.py` hunks apply as written: `steer_attn` after attention, and
   compaction to `boundary_indices` before the boundary layer.

The resolved `0805` changes the same lines as #738's patch (6 files, 636 insertions, 68 deletions).

## What changes at run time

- **A Spec v1 vector (GLP-39, every measured arm):** `steer_attn` returns in Python before any launch, because
  `mean_streams` is false. The fold kernel `glp_steer_kernel` takes a new `mean_streams` argument. Its Spec v1 body
  sits under a trace-time `else:` and is the same code as before. The kernel is compiled again because its JIT
  signature changes.
- **The CED boundary:** `steer_rows` is now compacted to `boundary_indices` before the boundary layer, not after
  it. Under sequence parallelism, the boundary layer gathers its rows to the replicated compact set
  (`gather_sp_rows`) before attention, so `steer_rows` matches the rows of both the attention write and the fold.
  For a Spec v1 vector, only the fold reads `steer_rows`, after the layer, as before.
- **A Spec v3 vector (GLP-27, the qualification arm only):** the fold projects the mean of the four streams, and
  `glp_steer_write` (PDL-chained) projects each covered layer's attention write. Unmarked rows exit both kernels
  before writing.
- **The reader** (`glp.py`) accepts Spec v1 and Spec v3 and refuses any other pairing. It checks
  `glp.content_sha256` when the file carries one.

## Checks

- `git diff --check` is clean, and `py_compile` passes on the six files.
- No other caller of `tilelang_glp.declare`, `glp_steer` or `steer_attn` exists outside `glp.py` and `nvidia/model.py`.
- The tests the patch changes are release-gate files: `tests/kernels/test_deepseek_v41_tilelang_glp.py` and
  `tests/models/test_deepseek_v4_1_glp.py`. So is `tests/kernels/test_deepseek_v41_tilelang_pdl.py`, which reads
  `glp_steer_write`'s entry in `deepseek_v41_pdl_sources.py`'s `CHAIN`. So are the other files of #738's suite:
  `tests/test_weightless.py`, `tests/models/test_deepseek_v4_1_ced_model.py` and
  `tests/models/test_deepseek_v4_1_sp_prefill.py`.
- `scripts/release_tests.py check` at `d6d4319e`: 217 files (156 gate, 61 excluded), 0 problems. The gate
  manifest is regenerated there, and only its head and repo change.
- Not run here: the release tests, the image build and #738's suite. They run on candidate-55's next image.
