# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent copy/cast reference, immutable raw logits, masks and replay."""

import pytest
import torch

pytest.importorskip("tilelang")
from vllm.v1.worker.gpu.sample import tilelang_sampling as kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["dense", "padded", "transpose", "columns"])
@pytest.mark.parametrize("shape", [(0, 17), (3, 0), (1, 1), (3, 1031), (7, 129280)])
def test_owned_fp32_copy(dtype, layout, shape):
    rows, cols = shape
    if layout == "dense":
        src = torch.randn(shape, device="cuda", dtype=dtype)
    elif layout == "padded":
        src = torch.randn((rows, cols + 7), device="cuda", dtype=dtype)[:, 3 : 3 + cols]
    elif layout == "transpose":
        src = torch.randn((cols, rows), device="cuda", dtype=dtype).t()
    else:
        src = torch.randn((rows, cols * 2), device="cuda", dtype=dtype)[:, ::2]
    if src.numel() >= 8:
        src[0, :8] = torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0, -2.0, 0.125],
            dtype=dtype,
            device="cuda",
        )
    before = src.clone()
    expected = torch.empty(shape, dtype=torch.float32, device="cuda").copy_(src)
    got = kernels.copy_logits(src)
    assert got.shape == src.shape and got.dtype == torch.float32
    assert got.is_contiguous()
    torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)
    # Signed zero and all non-NaN bits are preserved through the cast.
    finite = ~expected.isnan()
    assert torch.equal(
        got.view(torch.int32)[finite], expected.view(torch.int32)[finite]
    )
    if got.numel():
        got.fill_(7)
        assert got.data_ptr() != src.data_ptr()
    torch.testing.assert_close(src, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_copy_graph_replay_reads_new_values_and_keeps_raw_logits(dtype):
    backing = torch.randn((8, 1040), device="cuda", dtype=dtype)
    src = backing[1:7, 3:1034]
    kernels.copy_logits(src)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = kernels.copy_logits(src)
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        backing.normal_()
        before = src.clone()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(got, src.float(), rtol=0, atol=0)
        torch.testing.assert_close(src, before, rtol=0, atol=0)


def test_sampler_declarations_include_the_copy_specializations():
    calls = kernels.sampling_calls([torch.float16, torch.bfloat16, torch.float32])
    # The declaration goes through the same factory/key as the runtime launch.
    expected = [
        kernels._call(kernels.copy_logits_kernel, kernels._DTYPES[d], vectorized)
        for d in (torch.float16, torch.bfloat16, torch.float32)
        for vectorized in (False, True)
    ]
    assert all(c in calls for c in expected)


@pytest.mark.parametrize("cols", [1020, 1024, 1028])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_vector_copy_handles_partial_blocks_and_padded_rows(dtype, cols):
    pitch = (cols + 31) // 16 * 16
    src = torch.randn((5, pitch), dtype=dtype, device="cuda")[:, :cols]
    before = src.clone()
    got = kernels.copy_logits(src)
    torch.testing.assert_close(got, src.float(), rtol=0, atol=0)
    assert got.is_contiguous() and got.data_ptr() != src.data_ptr()
    assert torch.equal(src, before)
