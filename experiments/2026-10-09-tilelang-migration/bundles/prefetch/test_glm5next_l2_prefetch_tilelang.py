# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TileLang L2 prefetch kernel runs on a segment table, eagerly and inside a
CUDA graph, without touching the data it prefetches."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


def _plan(*tensors):
    from vllm.models.glm5next.nvidia import l2_prefetch as l2pf

    segments = [l2pf.tensor_segment(f"t{i}", t) for i, t in enumerate(tensors)]
    return l2pf.L2PrefetchPlan([s for s in segments if s is not None], tensors[0].device)


def test_tilelang_prefetch_leaves_data_unchanged():
    from vllm.models.glm5next.nvidia.l2_prefetch_tilelang import compile_launcher

    launch = compile_launcher(grid=2, block=128, chunk=4096)
    # Sizes that end mid-chunk and below the 16-byte granule.
    a = torch.randn(3 * 1024 * 1024 + 7, device="cuda")
    b = torch.randint(0, 255, (4096 * 5 + 12,), dtype=torch.uint8, device="cuda")
    c = torch.ones(3, dtype=torch.uint8, device="cuda")
    before = [t.clone() for t in (a, b, c)]
    plan = _plan(a, b, c)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    launch(plan.segs, side.cuda_stream)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    for t, ref in zip((a, b, c), before):
        assert torch.equal(t, ref)


def test_tilelang_prefetch_captures_in_a_graph():
    from vllm.models.glm5next.nvidia.l2_prefetch_tilelang import compile_launcher

    launch = compile_launcher(grid=2, block=128, chunk=4096)
    w = torch.randn(8 * 1024 * 1024, device="cuda")
    plan = _plan(w)
    side = torch.cuda.Stream()
    launch(plan.segs, side.cuda_stream)  # compile and warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        launch(plan.segs, side.cuda_stream)
        main.wait_stream(side)
    graph.replay()
    torch.cuda.synchronize()
