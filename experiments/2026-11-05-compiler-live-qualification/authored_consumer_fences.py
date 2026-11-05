"""Run portable consumer-fence reproducers on the current CUDA target; no vLLM."""
import importlib.util
from pathlib import Path
import tilelang

source = Path(tilelang.__file__).resolve().parents[1]
path = source / "testing/python/issue/test_tilelang_issue_3529.py"
spec = importlib.util.spec_from_file_location("consumer_fence_checks", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ["test_ws_generic_stage_reads_fence_before_release", "test_ws_mma_sync_stage_reads_fence_before_release", "test_ws_generic_stage_reads_fast_math_correctness"]:
    getattr(module, name)()
    print(name, "passed", flush=True)
