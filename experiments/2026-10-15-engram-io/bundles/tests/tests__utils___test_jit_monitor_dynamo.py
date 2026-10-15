# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The JIT monitor reports torch.compile compilations during inference: a function
compiled during serving also loads its kernels there (CPU only)."""

import pytest
import torch

from vllm.utils import jit_monitor


def _chain(error: BaseException):
    while error is not None:
        yield error
        error = error.__cause__ or error.__context__


def test_torch_compile_during_inference_is_an_error(monkeypatch):
    monkeypatch.setattr(jit_monitor, "_active", False)
    monkeypatch.setattr(jit_monitor, "_dynamo_hook_installed", False)
    jit_monitor._setup_dynamo_compile_hook()
    monkeypatch.setattr(jit_monitor, "_active", True)
    monkeypatch.setattr(jit_monitor, "_mode", "error")
    torch._dynamo.reset()
    compiled = torch.compile(lambda x: x * 2 + 1, backend="eager")
    with pytest.raises(Exception) as caught:
        compiled(torch.ones(3))
    assert any("torch.compile compilation during inference" in str(e)
               for e in _chain(caught.value))


def test_torch_compile_before_activation_is_allowed(monkeypatch):
    monkeypatch.setattr(jit_monitor, "_active", False)
    monkeypatch.setattr(jit_monitor, "_dynamo_hook_installed", False)
    jit_monitor._setup_dynamo_compile_hook()
    monkeypatch.setattr(jit_monitor, "_mode", "error")
    torch._dynamo.reset()
    compiled = torch.compile(lambda x: x - 1, backend="eager")
    assert torch.equal(compiled(torch.ones(2)), torch.zeros(2))
