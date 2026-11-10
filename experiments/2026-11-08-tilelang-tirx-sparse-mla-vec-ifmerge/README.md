# TileLang TIRx `reinterpret` Vectorization and Read-State `IfThenElse` Merging in `sparse_mla`

- **Date**: 2026-11-08
- **Base deployment commit**: `cc20ed18b6ca6521b602f09cb3ba326774166d1c`
- **Base vLLM patch head**: `320021e3c4dbb202d911ab70e2cd924c75420488` (tree `ed1bdb7a7684da33d403a055521ac0c4dd7ca670`, 641 patches)
- **Candidate vLLM patch head**: `66b5b3e10231603235ae6e204e69dad123d7dc9b` (tree `955b2da8ff2309d4f3aef6a7af3eca010856d61a`, `patchset_sha256` `ef51c3a289db2a701376f5f8f01ccd826fcedff2be71ee27107bc454aed56eda`, 642 patches)
- **vLLM runtime pass & test patch**: `vllm/0762-perf-dsv41-tilelang-vectorize-reinterpret-and-merge-readstate-ifs.patch`
- **TileLang C++ compiler patch**: `tilelang/0014-transform-vectorize-reinterpret-and-merge-readstate-ifs.patch`

## Problem & Root-Cause Analysis

Auditing the TileLang TIRx compiler pipeline against `vllm/models/deepseek_v4_1/tilelang/sparse_mla.py` on `sp8` (`spark-ds41f:1107`) uncovered two compiler pass bugs affecting `sparse_mla`:

1. **`dequant_swa` scalarization due to missing `TVectorizable = true` on `tirx.reinterpret` (`builtin::reinterpret()`)**:
   - Every compilation of `sparse_mla` emitted:
     ```
     Warning: T.vectorized loop over `vec` with extent 8 is lowered as a serial loop because TileLang could not find a valid vectorization plan.
     ```
   - In `dequant_swa`, the UE8M0 scale is reconstructed via `T.reinterpret(T.shift_left(T.cast(staged[i, D + d // SWA_GROUP], T.uint32), 23), accum_dtype)`.
   - After `LayoutInference` creates `for vec in T.vectorized(8):`, `DecoupleTypeCast` stages `staged[...]` into a local array `uchar staged_local_cast_2[8]` indexed by `vec`.
   - Next, `LegalizeVectorizedLoop` invokes `VectorizePlanner::VisitExpr_(const CallNode*)` (`src/transform/loop_vectorize.cc`), which checks `Op::GetAttrMap<TVectorizable>("TVectorizable")`. Even though `TLVectorizer::MutateReinterpretExpr_` in `src/transform/vectorize_loop.cc` and `src/transform/common/loop_vectorization_utils.h` already vectorizes `builtin::reinterpret()` across lanes (`uint4` -> `float4`), `tirx.reinterpret` is registered without `.set_attr<TVectorizable>("TVectorizable", true)`. Consequently, `VectorizePlanner` rejects the loop-variant argument `staged_local_cast_2[vec]` inside `reinterpret`, sets `vectorize_size = 1`, and degrades `dequant_swa` to a scalar 8-iteration loop (`for (int vec = 0; vec < 8; ++vec)`).

2. **30-way `if (seg_active)` branch split in `sparse_mla` Stage 2 and 10-way split in Stage 1 due to `IfStmtBinding` / `MergeIfStmt` `CallEffectKind::kReadState` mismatch**:
   - In `decode_body` Stage 1 and Stage 2 (`sparse_mla_kernel` and `sparse_mla_kernel_1`), statements are guarded by `if not skip_inactive or seg_active(seg, SwaLengths, IdxLengths, row):` (without an `else` branch).
   - `IfStmtBinding` (`src/transform/if_stmt_binding.cc`) distributes `IfThenElse` (without `else`) across each child statement in `SeqStmt` so `ThreadSync` can insert `__syncthreads()` barriers at the outer `SeqStmt` level. Because `seg_active(...)` reads `SwaLengths[row]` and `IdxLengths[row]`, `SideEffect(condition)` is `CallEffectKind::kReadState` (`2`). `IfStmtBinding` checks `IsReplayableScalarBind` (`SideEffect <= CallEffectKind::kReadState` and read/write buffer disjointness), sees that the body never writes to `SwaLengths` or `IdxLengths`, and duplicates the condition onto every split `IfThenElse`.
   - Later, `MergeIfStmt` (`src/transform/merge_if_stmt.cc`) runs after `ThreadSync` to re-merge adjacent `IfThenElse` blocks that were not separated by a barrier, but checks `SideEffect(current_condition) <= CallEffectKind::kPure` (`1`). Because `kReadState > kPure`, `MergeIfStmt` refuses to merge any condition containing a `BufferLoad`, leaving **10 separate `if (swa_active | ...)` blocks in Stage 1** and **6 separate `if` blocks per segment inside `#pragma unroll for (int seg = 0; seg < 5; ++seg)` (30 unrolled branches) in Stage 2**.

## Changes

1. **Runtime compiler pass activation in `vllm` (`0762-perf-dsv41-tilelang-vectorize-reinterpret-and-merge-readstate-ifs.patch`)**:
   - In `vllm/models/deepseek_v4_1/tilelang/sparse_mla.py`, `_install_tirx_compiler_fixes()` idempotently sets `TVectorizable = True` on `Op.get("tirx.reinterpret")` and wraps `tilelang.transform.MergeIfStmt` with `tl.MergeReadStateIfStmt` so consecutive `IfThenElse` blocks (without `else`) sharing an identical `CallEffectKind::kReadState` condition are merged whenever intermediate statements do not write to any buffer (`BufferStore` or writable `tvm_access_ptr`/`tl.access_ptr`) or redefine any `Var` (`Bind`) read by the condition.
   - Renames the macro local `swa_active` -> `swa_seg_active` in `seg_active` so `sparse_mla`'s `func.script(show_meta=True)` cache key is distinct from any pre-existing unpatched disk-cache entry without altering the cache namespace or invalidating other TileLang kernels.
   - Adds `test_sparse_mla_vectorizes_swa_dequant_and_merges_readstate_segment_ifs` in `tests/kernels/attention/test_deepseek_v41_tilelang.py`.

2. **Companion TileLang C++ patch (`tilelang/0014-transform-vectorize-reinterpret-and-merge-readstate-ifs.patch`)**:
   - Updates `VectorizePlanner::VisitExpr_(const CallNode*)` in `src/transform/loop_vectorize.cc` to accept `builtin::reinterpret()` alongside `op_vectorizable_`, matching `TLVectorizer`.
   - Updates `MergeIfStmtRewriter` in `src/transform/merge_if_stmt.cc` to merge consecutive `IfThenElse` statements with `CallEffectKind::kReadState` conditions when read buffers and variables are disjoint from body writes and bindings.

## Verification & Benchmark Results (`sp8`, `spark-ds41f:1107`)

- **Two-boot GPU `pytest` verification on `sp8` (`spark-ds41f:1107`)**:
  - `receipts/gpu-pytest-boot1.txt`: `24 passed in 126.00s` (`PDL sources: 27 kernels launched, 18 with PDL, 0 violations`)
  - `receipts/gpu-pytest-boot2.txt`: `24 passed in 126.79s` (`PDL sources: 27 kernels launched, 18 with PDL, 0 violations`)
- **Interleaved `sparse_mla` GPU benchmark on `sp8` (`H=40, SWA=128, IDX=512, page=64`, 10 interleaved rounds x 200 iterations, warmed 50x, `receipts/sparse-mla-bench.json`)**:
  - Bitwise-identical `uint16` outputs (`bitwise_equal = True`) across all routes (`split`, `decode`, `extend`):

| Route & Shape | Mode | `LegalizeVectorizedLoop` Warnings | `seg_active` Branches (Stage 1 + Unrolled Stage 2) | Interleaved Median (`us`) | Interleaved Min (`us`) | Interleaved p90 (`us`) | Delta vs Baseline (Median / Min) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`split` (`rows=1`, `H=40`)** | `baseline` | 2 | `10 + 30` | `11.31` | `10.88` | `11.36` | — |
| | `merge_if_only` | 2 | `2 + 5` | `11.31` | `10.87` | `12.54` | `0.0%` / `-0.1%` |
| | `reinterpret_vec_only` | **0** | `10 + 30` | `11.31` | `10.80` | `11.44` | `0.0%` / `-0.7%` |
| | **`candidate_both`** | **0** | **`2 + 5`** | **`11.29`** | **`10.79`** | **`11.42`** | **`-0.2%` / `-0.8%`** |
| **`decode` (`rows=8`, `H=40`)** | `baseline` | 2 | `10 + 30` | `17.28` | `16.24` | `17.37` | — |
| | `reinterpret_vec_only` | **0** | `10 + 30` | `17.18` | `15.96` | `18.70` | `-0.6%` / `-1.7%` |
| | `merge_if_only` | 2 | `2 + 5` | `16.68` | `15.66` | `16.82` | `-3.5%` / `-3.6%` |
| | **`candidate_both`** | **0** | **`2 + 5`** | **`16.39`** | **`15.63`** | **`16.53`** | **`-5.2%` / `-3.8%`** |
| **`extend` (`rows=32`, `H=40`)** | `baseline` | 2 | `10` (Stage 1) | `93.62` | `92.52` | `94.69` | — |
| | `reinterpret_vec_only` | **0** | `10` (Stage 1) | `92.85` | `92.08` | `92.99` | `-0.8%` / `-0.5%` |
| | **`candidate_both`** | **0** | **`2` (Stage 1)** | **`92.72`** | **`91.84`** | **`93.53`** | **`-1.0%` / `-0.7%`** |
