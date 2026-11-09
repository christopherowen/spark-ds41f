"""import_check.py: where the changed modules import from in a GPU container, and the source facts
that tell the arms apart (read from each module's file; the TileLang kernels are JITImpl objects,
which inspect.getsource cannot read: q.sh's import step stopped there)."""
import importlib, inspect
mods = {}
for m in ["vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu.spec_decode.adaptive_verification",
          "vllm.v1.worker.gpu.spec_decode.tilelang_copy"]:
    mods[m] = importlib.import_module(m); print(m, mods[m].__file__)
a = mods["vllm.v1.worker.gpu.spec_decode.adaptive_verification"]
r = mods["vllm.v1.worker.gpu.model_runner"]
src = {m: open(mod.__file__).read() for m, mod in mods.items()}
print("_DEAD_ROWS_TAU", a._DEAD_ROWS_TAU, "dead_rows_enabled", a.dead_rows_enabled())
print("SINGLE_REQ_TILE_SPILL_SCALE", getattr(a, "SINGLE_REQ_TILE_SPILL_SCALE", None),
      "_ZERO_CONFIDENCE_LOG_TAU", getattr(a, "_ZERO_CONFIDENCE_LOG_TAU", None))
print("declare_step_kernels: copy drafter declares dead rows:",
      "dead_rows_enabled() or self.copy_drafter" in inspect.getsource(r.GPUModelRunner.declare_step_kernels))
print("dead_row_args has the zero-confidence floor:", "has_copy_tail" in inspect.getsource(a.AdaptiveVerificationManager.dead_row_args))
print("reallocate_drafts sets one request's capacity to the budget:",
      "scheduled_drafts[0] = draft_budget" in inspect.getsource(a.AdaptiveVerificationManager.reallocate_drafts))
print("copy_apply survival:", "dspark_survival *= dspark_confidences" in src["vllm.v1.worker.gpu.spec_decode.tilelang_copy"])
