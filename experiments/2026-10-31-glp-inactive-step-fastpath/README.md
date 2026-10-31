# Skip Per-Step GLP `copy_to_uva` + `mark_rows` and Eager `glp_steer` Launches When No Steered Request Is Active

## Problem

In `rc12..rc21`, enabling GLP dual-serving in the recipe (`WEIGHTLESS_STEER_PATH` + `WEIGHTLESS_STEER_MODEL`) causes `DeepseekV41ModelState.prepare_inputs` (`vllm/models/deepseek_v4_1/nvidia/model_state.py`) to unconditionally execute on every step:
1. `self.steered_requests.copy_to_uva()` (host-to-UVA synchronization and tensor view creation),
2. `mark_rows(...)` (`vllm/models/deepseek_v4_1/tilelang/glp_rows.py`, 1 ungraphed TileLang kernel launch before the target forward pass), and
3. On eager steps (prefill chunks, CED steps, varlen steps), **39 `glp_steer` kernel launches** across `layers 1..39` inside `DeepseekV4Model.forward`, even when 100% of active requests target the unsteered `deepseek-v4.1-flash` model (`rows[t] == 0`).

This directly contributes to the ungraphed per-step decode overhead observed between `candidate 10` (where GLP dual-serving was not enabled in `rc-tp3.json`) and `candidate 20` on PR #709.

## Change

1. **Track Active Steered Requests on `DeepseekV41ModelState` (`vllm/models/deepseek_v4_1/nvidia/model_state.py`)**:
   - Maintain `_steered_req_ids: set[str]` across `add_request` and `remove_request` and `_steering_rows_dirty_len: int` tracking the high-water mark of rows written by `mark_rows`.
   - In `prepare_inputs`:
     - When at least one request in `input_batch` is steered (`has_steered == True`), set `self.steering.step_active = True`, run `mark_rows(...)`, and update `_steering_rows_dirty_len`.
     - When no request in `input_batch` is steered (`has_steered == False`), set `self.steering.step_active = False`, skip `copy_to_uva()` and `mark_rows(...)`, and if `_steering_rows_dirty_len > 0` from a prior steered step, zero `self.steering.rows[:dirty_len]` once and reset `_steering_rows_dirty_len = 0`.
   - In `prepare_dummy_inputs`, zero `self.steering.rows`, set `self.steering.step_active = True`, and reset `_steering_rows_dirty_len = 0` so CUDA-graph capture continues to record `glp_steer` nodes for dual-serving replay.
2. **Skip Eager `glp_steer` Launches on Unsteered Steps (`vllm/models/deepseek_v4_1/glp.py`)**:
   - In `GLPSteering.steer`, after validating tensor shapes, return immediately when `not self.step_active` and the current CUDA stream is not capturing a graph.

## Verification

- Verified on `dgx4` inside `spark-ds41f:19f2c20ed4d6-next-21`:
  - `pytest --noconftest -q tests/models/test_deepseek_v4_1_glp.py tests/kernels/test_deepseek_v41_tilelang_glp.py tests/test_weightless.py`: **88 passed, 1 skipped**.
