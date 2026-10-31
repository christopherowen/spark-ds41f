# Weak-Ref Nested Outputs of Captured CUDA Graphs (`#63427`)

## Upstream Source

- Upstream commit: `vllm-project/vllm@f81f0f2509` (`[Bugfix][CUDA Graph] Weak-ref nested outputs of captured graphs (#63427)`)

## Problem

`weak_ref_tensors()` previously only applied `weak_ref_tensor` one level deep inside a `list` or `tuple` (`[weak_ref_tensor(t) for t in tensors]`). When a captured graph or compilation partition returns nested containers such as `(hidden_states, [aux_hidden_states, ...])` (which occurs when EAGLE3/MTP `aux_hidden_states` are returned alongside `hidden_states`) or hoisted `int`/`LayerName` values, the inner list's tensors were returned as strong `torch.Tensor` references instead of weak tensor references, pinning their underlying memory blocks across graph captures.

## Changes

- **`vllm/utils/torch_utils.py` (`weak_ref_tensors`)**:
  - Recurse via `weak_ref_tensors(t)` on `list` and `tuple` elements instead of calling `weak_ref_tensor(t)` non-recursively.
  - Pass through `int` and `LayerName` values unchanged, and raise `ValueError` for unknown nested types so unhandled types cannot silently bypass weak-referencing.
- **`tests/utils_/test_torch_utils.py`**:
  - Add `test_weak_ref_tensors_does_not_keep_nested_tensors_alive` and `test_weak_ref_tensors_rejects_unknown_nested_types`.

## Verification

Verified in `spark-ds41f:19f2c20ed4d6-next-21` on `dgx4`:
- `pytest -q tests/utils_/test_torch_utils.py` (`38 passed`).
