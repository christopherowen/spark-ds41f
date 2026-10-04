"""Gate plus routing per MoE layer at decode: before, step 1 (TileKernels) and fused."""
import torch
from vllm.models.deepseek_v4_1.tilelang import linear, router as R
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
from tile_kernels.moe import moe_topk_gate_forward

linear._scratch = lambda specs, scratch: [torch.empty(s, dtype=d, device="cuda") for s, d in specs]

class Gate(torch.nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.weight = torch.nn.Parameter((torch.randn(experts, 5120, device="cuda") * 0.02).bfloat16(), requires_grad=False)
        self.out_dtype = torch.float32
        self.quant_method = linear.TileLangLinearMethod(); self.quant_method.process_weights_after_loading(self)
    def forward(self, x):
        return self.quant_method.apply(self, x), None

def timed(fn, iterations=2000):
    for _ in range(50): fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20): fn()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations // 20): graph.replay()
    stop.record(); torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000 / iterations

count = 0
for experts, topk in ((384, 6), (128, 3)):
    gate = Gate(experts); bias = torch.randn(experts, device="cuda") * 0.1
    kw = dict(top_k=topk, global_num_experts=experts, e_score_correction_bias=bias, renormalize=True, routed_scaling_factor=1.5, scoring_func="sqrtsoftplus")
    fused = R.TileLangGateRouter(gate=gate, **kw); step1 = R.TileKernelsRouter(**kw)
    for rows in (1, 6, 16, 48, 64):
        x = torch.randn(rows, 5120, device="cuda").bfloat16()
        gate_only = timed(lambda: gate(x))
        before = timed(lambda: dsv4_topk(gate(x)[0], bias, torch.int64, 1.5)) if experts == 384 else float("nan")
        one = timed(lambda: step1._compute_routing(None, gate(x)[0], None))
        two = timed(lambda: fused._compute_routing(None, x, None))
        print(f"{experts:4d} experts rows {rows:3d}: gate alone {gate_only:6.2f} us | gate+Triton {before:6.2f} | gate+TileKernels {one:6.2f} | fused {two:6.2f} us")
        count += 1
print(f"{count} passed")
