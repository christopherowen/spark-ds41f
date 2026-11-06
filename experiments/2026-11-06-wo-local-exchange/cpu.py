"""CPU-only orchestration and bit-exactness unit tests; no GPU device is exposed."""
import resource
import sys

import tilelang
import tilelang.language as T
import vllm.tilelang_utils as tl
from vllm.platforms import current_platform

current_platform.device_type = "cpu"
tl.tilelang, tl.T = tilelang, T
import pytest

status = pytest.main([
    "--noconftest", "-p", "no:cacheprovider", "-q",
    "/src/tests/models/test_deepseek_v4_1_attention.py",
    "/src/tests/models/test_deepseek_v4_1_sp_prefill.py",
    "/src/tests/models/test_deepseek_v4_1_l2_schedule.py",
])
print("process_maxrss_kib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
sys.exit(status)
