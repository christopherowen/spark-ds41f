# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hidden states the runner hands the drafter: the target's override buffer cut to
the step's rows, or the step's hidden states when the target has none (no method, or
the method returns None, as DS4.1's inherited one does without an MTP buffer)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.model_runner import _mtp_target_states

pytestmark = pytest.mark.cpu_test


def test_override_buffer_is_cut_to_the_step():
    buffer = torch.arange(40.0).view(10, 4)
    model = SimpleNamespace(get_mtp_target_hidden_states=lambda: buffer)
    hidden = torch.zeros(3, 2)
    assert torch.equal(_mtp_target_states(model, hidden), buffer[:3])


@pytest.mark.parametrize(
    "model",
    [SimpleNamespace(), SimpleNamespace(get_mtp_target_hidden_states=lambda: None)],
    ids=["no method", "returns None"],
)
def test_without_an_override_the_drafter_takes_the_hidden_states(model):
    hidden = torch.ones(3, 2)
    assert _mtp_target_states(model, hidden) is hidden
