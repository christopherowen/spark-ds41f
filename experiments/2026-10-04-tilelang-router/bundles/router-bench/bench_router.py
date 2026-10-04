"""Router kernel time: TileKernels' TileLang gate against vLLM's routers, per call."""
import torch

import vllm._custom_ops as ops
from tile_kernels.moe import moe_topk_gate_forward
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk


def timed(fn, iterations=2000):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            fn()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations // 20):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000 / iterations


results = 0
for experts, topk in ((384, 6), (128, 3)):
    bias = torch.randn(experts, device="cuda") * 0.1
    for rows in (1, 6, 16, 48, 96, 512, 8192):
        logits = torch.randn(rows, experts, device="cuda") * 3
        ids = torch.empty(rows, topk, dtype=torch.int64, device="cuda")
        weights = torch.empty(rows, topk, dtype=torch.float32, device="cuda")
        tile = timed(lambda: moe_topk_gate_forward(logits, topk, False, 0, 1.5, 0, bias=bias,
                                                   out=(ids, weights)))
        if experts == 384:
            name, other = "triton dsv4_topk", timed(lambda: dsv4_topk(logits, bias, torch.int64, 1.5))
        else:
            ids32 = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
            expert = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
            name, other = "cuda topk_hash_softplus_sqrt", timed(
                lambda: ops.topk_hash_softplus_sqrt(weights, ids32, expert, logits, True, 1.5, bias, None, None,
                                            is_padding=None, bias_vl=None, image_sentinel_lo=0))
        print(f"{experts:4d} experts top {topk}  rows {rows:5d}: tilekernels {tile:7.2f} us   {name} {other:7.2f} us")
        results += 1
print(f"{results} passed")
