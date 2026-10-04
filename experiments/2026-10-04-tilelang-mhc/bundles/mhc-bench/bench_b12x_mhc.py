"""B12X's lagged mHC post_pre per sublayer under CUDA graphs, prepared as vLLM's B12xMHC prepares it."""
import torch

from b12x.norm import mhc as bm
from b12x.norm.mhc import _impl
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall

H = 5120


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


device = torch.device("cuda")
count = 0
with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
    fns = [torch.randn(24, 4 * H, device="cuda") * 0.02 for _ in range(40)]
    scale, base = torch.ones(3, device="cuda"), torch.zeros(24, device="cuda")
    weight = torch.ones(H, dtype=torch.bfloat16, device="cuda")
    for rows in (1, 6, 16, 48, 64, 512, 8192):
        residual = torch.randn(rows, 4, H, device="cuda").bfloat16()
        x = torch.randn(rows, H, device="cuda").bfloat16()
        pre = torch.full((rows, 4), 0.25, device="cuda")
        post = torch.rand(rows, 4, device="cuda")
        comb = torch.softmax(torch.randn(rows, 4, 4, device="cuda"), -1)
        invocation = {"lagged_mix": True, "has_norm_weight": True, "rms_eps": 1e-20, "hc_eps": 1e-6,
                      "sinkhorn_iters": 20, "norm_eps": 1e-20, "operation": "post_pre", "has_fn_bf16": False,
                      "expanded_residual": False}
        plan = bm.plan(bm.Caps(device=device, max_tokens=rows, hidden_size=H), invocation=FrozenMapping(invocation))
        outputs = dict(out=torch.empty(rows, 4, H, dtype=torch.bfloat16, device="cuda"),
                       y=torch.empty(rows, H, dtype=torch.bfloat16, device="cuda"),
                       post=torch.empty(rows, 4, device="cuda"), comb=torch.empty(rows, 4, 4, device="cuda"),
                       pre_out=torch.empty(rows, 4, device="cuda"))
        options = dict(rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20, norm_weight=weight, norm_eps=1e-20,
                       pre_mix=pre)

        def prepare_call(state):
            spec, = state.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            bound = state.bind(scratch=scratch, tokens=rows, **outputs)
            return PreparedCall(run=lambda: _impl._b12x_mhc_post_pre_impl(
                x, residual, post, comb, fns[0], scale, base, **options, binding=bound, _state=state),
                owners=(scratch, bound))

        session.prepare((plan.request(name=f"mhc-post-pre-{rows}", prepare_call=prepare_call),))
        spec, = plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = bm.bind(plan, scratch=scratch, tokens=rows, **outputs)
        run = lambda f: bm.run_post_pre(x, residual, post, comb, f, scale, base, **options, binding=binding)
        rounds = 20 if rows <= 512 else 3
        warm = timed([lambda: run(fns[0])] * 10, rounds)
        cold = timed([lambda f=f: run(f) for f in fns], max(1, rounds // 4))
        print(f"b12x rows {rows:5d}: post_pre {warm:7.1f} us (fn cold {cold:7.1f})")
        count += 1
print(f"{count} passed")
