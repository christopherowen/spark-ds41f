"""Import installed compiler declarations for CPU model simulations.

vLLM deliberately hides these APIs when no GPU driver is active. The simulations
still import modules containing decorated kernel definitions. Bind the real
installed compiler APIs for those definitions; no computational kernel or test
assertion is replaced, and the containers expose no GPU devices.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

import vllm.tilelang_utils as tilelang_utils
import vllm.triton_utils as triton_utils
import vllm.triton_utils.importing as triton_importing

tilelang_utils._ensure_tilelang_imported()
triton_importing.HAS_TRITON = triton_utils.HAS_TRITON = True
triton_utils.triton = triton
triton_utils.tl = tl
triton_utils.tldevice = tldevice
triton_utils.gluon = gluon
triton_utils.gl = gl
triton_utils.aggregate = aggregate


def pytest_sessionfinish(session, exitstatus):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU model simulations initialized CUDA")
    print("CPU declaration harness: CUDA context remained uninitialized")
