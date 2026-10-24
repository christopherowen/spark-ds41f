# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model graph manager's persistent outputs: a model's declared buffers are
adopted when their rows fit the output, and a captured forward that wrote its
output into them needs no copy; anything else is copied as before."""

import torch

from vllm.v1.worker.gpu.cudagraph_utils import _output_buffer, _store_output


def test_declared_buffer_is_adopted_when_it_fits():
    declared = torch.zeros(8, 4, dtype=torch.bfloat16)
    output = torch.ones(6, 4, dtype=torch.bfloat16)
    assert _output_buffer(declared, output) is declared
    for other in (
        torch.zeros(5, 4, dtype=torch.bfloat16),  # too few rows
        torch.zeros(8, 3, dtype=torch.bfloat16),  # another width
        torch.zeros(8, 4, dtype=torch.float32),  # another dtype
    ):
        buffer = _output_buffer(other, output)
        assert buffer is not other and buffer.shape == output.shape
    assert _output_buffer(None, output).shape == output.shape


def test_output_written_in_place_is_not_copied():
    buffer = torch.zeros(8, 4)
    in_place = buffer[:6]
    in_place.fill_(3.0)
    _store_output(buffer, in_place, 6)
    assert (buffer[:6] == 3.0).all() and (buffer[6:] == 0).all()
    fresh = torch.full((6, 4), 5.0)
    _store_output(buffer, fresh, 6)
    assert (buffer[:6] == 5.0).all() and (buffer[6:] == 0).all()
    # A view at the buffer's start with other strides is still copied.
    column_major = torch.full((4, 6), 7.0).t()
    _store_output(buffer, column_major, 6)
    assert (buffer[:6] == 7.0).all()
