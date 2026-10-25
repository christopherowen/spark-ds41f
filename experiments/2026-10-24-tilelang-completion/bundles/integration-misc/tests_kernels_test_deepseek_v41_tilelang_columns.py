# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill SP's attention front glue against torch, bit for bit: the head weights
copied beside the normalized halves (torch's cat) and the gathered rows split into
contiguous parts (torch's split and contiguous copies), NaN and infinities copied
as bits, every buffer compared whole from identical garbage."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from vllm.models.deepseek_v4_1.tilelang import columns  # noqa: E402

DEVICE = torch.device("cuda")
Q, KV, HEADS = 1280, 512, 64  # DeepSeek-V4.1 Flash: q_lora_rank, KV, index heads


def _rows(rows, width, seed, dtype=torch.bfloat16):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(rows, width, generator=gen, device=DEVICE).to(dtype)
    if rows:
        x[0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
    return x


def _bits(x):
    return x.view(torch.int16) if x.element_size() == 2 else x.view(torch.int32)


@pytest.mark.parametrize("rows", [1, 7, 2048, 8193])
def test_copy_columns_beside_the_normalized_halves(rows):
    weights = _rows(rows, HEADS, rows)
    front = _rows(rows, Q + KV + HEADS, rows + 1)
    expected = front.clone()
    expected[:, Q + KV :] = weights
    columns.copy_columns(weights, front[:, Q + KV :])
    assert torch.equal(_bits(front), _bits(expected))


def test_copy_columns_between_strided_views():
    wide = _rows(33, 4096, 3)
    out = _rows(40, 2048, 4)
    expected = out.clone()
    expected[5:38, 512:1536] = wide[:, 1024:2048]
    columns.copy_columns(wide[:, 1024:2048], out[5:38, 512:1536])
    assert torch.equal(_bits(out), _bits(expected))


@pytest.mark.parametrize("widths", [(Q, KV, HEADS), (Q, KV), (1024, 512, 64)])
@pytest.mark.parametrize("rows", [1, 6, 4097, 16384])
def test_split_columns_like_torch(widths, rows):
    gathered = _rows(rows, sum(widths), rows + len(widths))
    expected = [part.contiguous() for part in gathered.split(widths, dim=-1)]
    parts = columns.split_columns(gathered, widths)
    assert len(parts) == len(widths)
    for part, reference in zip(parts, expected):
        assert part.is_contiguous() and part.shape == reference.shape
        assert torch.equal(_bits(part), _bits(reference))


@pytest.mark.parametrize("parts", [3, 2])
@pytest.mark.parametrize("rows", [1, 5, 96, 8192])
def test_join_columns_like_torch_cat(parts, rows):
    """The DSpark drafter's eager context join: the target's auxiliary hidden
    states side by side, as torch's cat."""
    width = 5120  # DeepSeek-V4.1 Flash's hidden size, one a target layer
    states = [_rows(rows, width, rows + 11 * i) for i in range(parts)]
    out = _rows(rows, parts * width, rows + 99)
    joined = columns.join_columns(states, out)
    assert joined.data_ptr() == out.data_ptr()
    assert torch.equal(_bits(out), _bits(torch.cat(states, dim=-1)))


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((8192, 5120), torch.bfloat16),  # CED's full hidden rows
        ((37, 4), torch.float32),  # pre_mix / sink columns
        ((3, 8), torch.bfloat16),
    ],
)
def test_zero_fills_every_byte(shape, dtype):
    """CED's full-row output starts as zeros (torch's new_zeros), any dtype."""
    out = torch.full(shape, float("nan"), dtype=dtype, device=DEVICE)
    assert columns.zero_(out) is out
    assert not out.view(torch.uint8).any()
    with pytest.raises(ValueError, match="16-byte"):
        columns.zero_(torch.empty(3, 5, dtype=torch.bfloat16, device=DEVICE))


def test_float32_parts():
    gathered = _rows(9, 64 + 32 + 16, 5, torch.float32)
    expected = [p.contiguous() for p in gathered.split((64, 32, 16), dim=-1)]
    for part, reference in zip(columns.split_columns(gathered, (64, 32, 16)), expected):
        assert torch.equal(_bits(part), _bits(reference))


def test_refuses_what_it_cannot_copy_in_vectors():
    x = _rows(4, 64, 6)
    with pytest.raises(ValueError, match="16-byte"):
        columns.copy_columns(x[:, 1:9], _rows(4, 8, 7))
    with pytest.raises(ValueError, match="16-byte"):
        columns.split_columns(_rows(4, 12, 8), (4, 8))
    with pytest.raises(ValueError, match="contiguous"):
        columns.split_columns(_rows(4, 128, 9)[:, :64], (32, 32))
    with pytest.raises(ValueError, match="contiguous parts"):
        columns.join_columns([x, _rows(5, 64, 10)], _rows(4, 128, 11))
