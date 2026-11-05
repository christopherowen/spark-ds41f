# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact rank-order arithmetic with tails, striding, mixed types and aliasing."""

import pytest
import torch

pytest.importorskip("tilelang")
from vllm.models.deepseek_v4_1.tilelang.collectives import rank_order_sum

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(parts, dtype):
    total = parts[0].float().clone()
    for part in parts[1:]:
        total.add_(part.float())
    return total.to(dtype)


def same_bits(got, expected):
    torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(
        torch.signbit(got[expected == 0]), torch.signbit(expected[expected == 0])
    )


@pytest.mark.parametrize("world", [1, 2, 3, 4, 8, 9])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["tail", "transpose", "gapped", "offset"])
def test_exact_strided_sum(world, dtype, layout):
    torch.manual_seed(123)
    parts = []
    for rank in range(world):
        if layout == "transpose":
            part = torch.randn((19, 7), dtype=dtype, device="cuda").t()
        elif layout == "gapped":
            part = torch.randn((7, 43), dtype=dtype, device="cuda")[:, 2:40:2]
        elif layout == "offset":
            part = torch.randn(134, dtype=dtype, device="cuda")[1:].view(7, 19)
        else:
            part = torch.randn((7, 19), dtype=dtype, device="cuda")
        part[0, :6] = torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 0.125],
            dtype=dtype,
            device="cuda",
        )
        parts.append(part)
    before = [p.clone() for p in parts]
    expected = reference(parts, dtype)
    storage = torch.full((7, 25), -123, dtype=dtype, device="cuda")
    out = storage[:, 3:22]
    assert rank_order_sum(parts, out) is out
    same_bits(out, expected)
    assert (storage[:, :3] == -123).all() and (storage[:, 22:] == -123).all()
    for p, original in zip(parts, before):
        same_bits(p, original)


@pytest.mark.parametrize("shape", [(), (0, 3), (2, 1, 3, 1), (2,) * 9])
def test_rank_normalization_and_empty_inputs(shape):
    parts = [torch.randn(shape, device="cuda") for _ in range(3)]
    same_bits(rank_order_sum(parts), reference(parts, torch.float32))


def test_mixed_source_and_output_dtypes():
    parts = [
        torch.randn((3, 17), dtype=dtype, device="cuda")
        for dtype in (torch.float16, torch.bfloat16, torch.float64)
    ]
    out = torch.empty((3, 17), dtype=torch.float64, device="cuda")
    same_bits(rank_order_sum(parts, out), reference(parts, out.dtype))


def test_overlapping_output_and_graph_replay():
    parts = [torch.randn((7, 7), device="cuda") for _ in range(3)]
    expected = reference(parts, torch.float32)
    rank_order_sum(parts, parts[0].t())
    same_bits(parts[0].t(), expected)
    parts = [torch.randn((7, 19), device="cuda") for _ in range(4)]
    out = torch.empty((19, 7), device="cuda").t()
    rank_order_sum(parts, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rank_order_sum(parts, out)
    for seed in (21, 22, 23):
        torch.manual_seed(seed)
        for part in parts:
            part.normal_()
        graph.replay()
        torch.cuda.synchronize()
        same_bits(out, reference(parts, out.dtype))


def test_explicit_cuda_contract_errors():
    x = torch.empty((3, 7), device="cuda")
    with pytest.raises(ValueError, match="same shape"):
        rank_order_sum([x, x[:1]])
    with pytest.raises(ValueError, match="same shape"):
        rank_order_sum([x, x.cpu()])
    with pytest.raises(ValueError, match="must not overlap"):
        rank_order_sum([x], x[:1].expand_as(x))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_declared_keys_cover_the_actual_strided_and_alias_launches(monkeypatch, dtype):
    from types import SimpleNamespace

    import vllm.distributed as distributed
    from vllm.models.deepseek_v4_1.tilelang import collectives as ops

    recorded = []

    def factory(*args, **kwargs):
        recorded.append((args, kwargs))
        return lambda *buffers: None

    monkeypatch.setattr(ops, "rank_order_strided_kernel", factory)
    monkeypatch.setattr(
        distributed,
        "get_tp_group",
        lambda: SimpleNamespace(
            world_size=3,
            device_communicator=SimpleNamespace(rank_order_reduce_scatter=True),
        ),
    )
    declared = [
        (args, kwargs)
        for fn, args, kwargs in ops.rank_order_sum_calls()
        if fn is factory
    ]
    parts = [torch.empty((3, 7), dtype=dtype, device="cuda") for _ in range(3)]
    ops.rank_order_sum(parts, parts[0])
    assert len(recorded) == 2  # reduction into independent scratch, then copy-out
    assert all(key in declared for key in recorded)
