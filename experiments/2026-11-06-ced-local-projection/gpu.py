"""GPU qualification driver for the exact mounted source; not a serving build."""
import resource
import sys
import vllm  # Resolve the image package and its native extensions before test paths.
sys.path.insert(0, "/src")
import pytest

status = pytest.main([
    "--noconftest", "-p", "no:cacheprovider", "-q",
    "/src/tests/kernels/test_deepseek_v41_tilelang_ced.py",
    "/src/tests/kernels/test_deepseek_v41_tilelang_mhc.py",
    "/src/tests/models/test_deepseek_v4_1_compressor.py",
])
try:
    import torch
    print("gpu_peak_allocated_bytes", torch.cuda.max_memory_allocated())
    print("gpu_peak_reserved_bytes", torch.cuda.max_memory_reserved())
except Exception as error:
    print("gpu_peak_counter_error", repr(error))
print("process_maxrss_kib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
sys.exit(status)
