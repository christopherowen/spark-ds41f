#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only verification for 0731-sp-engram-prefill.patch."""

from __future__ import annotations

import os
import sys

import pytest
import torch


def _init_cpu_stubs() -> None:
    assert not torch.cuda.is_available(), "CPU verification must run with GPUs hidden"
    import triton
    import triton.language as tl

    import vllm.tilelang_utils as u
    import vllm.triton_utils as tu

    tu.triton = triton
    tu.tl = tl
    u._ensure_tilelang_imported()


def main() -> int:
    _init_cpu_stubs()
    args = [
        "-p",
        "tblib_stub",
        "-p",
        "no:cacheprovider",
        "-q",
        "tests/models/test_deepseek_v4_1_sp_prefill.py",
        "tests/models/test_deepseek_v41_sp_project_reduce_scatter.py",
        "tests/models/test_deepseek_v4_1_engram.py",
    ]
    return int(pytest.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
