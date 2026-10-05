"""TileKernels lagged mHC per sublayer under CUDA graphs: pre and post_pre at decode and prefill rows,
with fn warm in L2 and cold (40 layers' weights cycled), and each kernel alone."""
import re
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4_1.tilelang import mhc as M
from vllm.v1.worker.workspace import init_workspace_manager

init_workspace_manager(torch.device("cuda"))
H = 5120
config = SimpleNamespace(hidden_size=H, hc_mult=4, rms_norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20)
mhc = M.TileKernelsMHC(config)
source = M._project(H, "post_pre").get_kernel_source()
print("projection MMA:", sorted(set(re.findall(r"mma_sync<[^>]*>", source))))


def timed(calls, rounds=20):
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000 / (rounds * len(calls))


count = 0
LAYERS = 40
fns = [torch.randn(M.MIXES, 4 * H, device="cuda") * 0.02 for _ in range(LAYERS)]
scale = torch.ones(3, device="cuda")
base = torch.zeros(M.MIXES, device="cuda")
weight = torch.ones(H, dtype=torch.bfloat16, device="cuda")
for rows in (1, 6, 16, 48, 64, 512, 8192):
    residual = torch.randn(rows, 4, H, device="cuda").bfloat16()
    x = torch.randn(rows, H, device="cuda").bfloat16()
    pre = torch.full((rows, 4), 0.25, device="cuda")
    post = torch.rand(rows, 4, device="cuda")
    comb = torch.softmax(torch.randn(rows, 4, 4, device="cuda"), -1)
    rounds = 20 if rows <= 512 else 3
    warm_pre = timed([lambda: mhc.pre(residual, fns[0], scale, base, weight, pre)] * 10, rounds)
    warm_pp = timed([lambda: mhc.post_pre(x, residual, post, comb, fns[0], scale, base, weight, pre)] * 10, rounds)
    cold_pp = timed([lambda f=f: mhc.post_pre(x, residual, post, comb, f, scale, base, weight, pre) for f in fns], max(1, rounds // 4))
    # Kernels alone.
    splits = H // M.BLOCK_H
    P = torch.empty(splits, rows, 1, M.MIXES, device="cuda")
    S = torch.empty(splits, rows, 1, device="cuda")
    C = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
    Q = torch.empty(splits, rows, device="cuda")
    out = torch.empty_like(residual)
    flat, flat_out = residual.view(rows, 4 * H), out.view(rows, 4 * H)
    project = M._project(H, "post_pre")
    k_part = timed([lambda: project(x, flat, post, comb, pre, fns[0], flat_out, P, S, C, Q)] * 10, rounds)
    y = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
    po, co, pr = torch.empty(rows, 4, device="cuda"), torch.empty(rows, 4, 4, device="cuda"), torch.empty(rows, 4, device="cuda")
    fin = M._finalize(H, 4 * H, 1e-20, 1e-6, 20)
    k_lag = timed([lambda: fin(P, S, C, Q, scale, base, weight, po, co, pr, y)] * 10, rounds)
    k_post = timed([lambda: mhc.post(x, residual, post, comb)] * 10, rounds)
    print(f"rows {rows:5d}: pre {warm_pre:7.1f} us  post_pre {warm_pp:7.1f} us (fn cold {cold_pp:7.1f})"
          f"  | project {k_part:6.1f}  finalize {k_lag:6.1f}  (standalone post {k_post:6.1f})")
    count += 1
print(f"{count} passed")
