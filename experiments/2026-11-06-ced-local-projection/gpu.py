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
print("process_maxrss_kib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
sys.exit(status)
