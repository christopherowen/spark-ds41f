"""CPU-only orchestration tests; TileLang GPU execution is not simulated here."""
import resource
import sys

import tilelang
import tilelang.language as T
import vllm.tilelang_utils as tl

# On a CPU-only host this utility intentionally exposes None. Import the real
# TileLang definitions for declaration-only use; no CUDA device is exposed.
tl.tilelang, tl.T = tilelang, T
import pytest

status = pytest.main([
    "--noconftest", "-p", "no:cacheprovider", "-q",
    "/src/tests/models/test_deepseek_v4_1_ced_sp.py",
    "/src/tests/models/test_deepseek_v4_1_ced_model.py",
    "/src/tests/models/test_deepseek_v4_1_sp_prefill.py",
    "/src/tests/models/test_deepseek_v4_1_compressor_prepare.py",
])
print("process_maxrss_kib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
sys.exit(status)
