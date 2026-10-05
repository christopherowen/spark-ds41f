"""mHC sublayer entry (post_pre) per call with fn cold in L2, as in serving.

Serving runs 80 sublayers, each with its own 2 MB fn, so every call reads fn
from DRAM. Calls cycle SETS distinct fn and residual sets (80 MB) inside one
CUDA graph; "warm" reuses one set. Times are per call, in microseconds.
"""
import importlib.util
import sys
from types import SimpleNamespace

import torch

from vllm.v1.worker.workspace import init_workspace_manager

init_workspace_manager(torch.device("cuda"))
H, SETS = 5120, 40
config = SimpleNamespace(hidden_size=H, hc_mult=4, rms_norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20)
NAMES = sys.argv[1:] or ["mhc_base", "mhc_v4"]
ROWS = (1, 2, 6, 12, 16, 32, 64, 96)


def load(name):
    spec = importlib.util.spec_from_file_location(name, f"/b/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(calls, rounds=10):
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    graph.replay()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000 / (rounds * len(calls))


modules = {name: load(name) for name in NAMES}
mhcs = {name: m.TileKernelsMHC(config) for name, m in modules.items()}
gen = torch.Generator(device="cuda").manual_seed(0)
fns = [torch.randn(24, 4 * H, device="cuda", generator=gen) * 0.02 for _ in range(SETS)]
scale, base = torch.ones(3, device="cuda"), torch.zeros(24, device="cuda")
weight = (torch.randn(H, device="cuda", generator=gen) * 0.1 + 1).bfloat16()
passed = failed = 0
print(f"{'rows':>5} {'variant':>18} {'cold call':>10} {'warm call':>10} {'cold project':>13} {'warm project':>13}")
for rows in ROWS:
    sets = [dict(
        residual=torch.randn(rows, 4, H, device="cuda", generator=gen).bfloat16(),
        x=torch.randn(rows, H, device="cuda", generator=gen).bfloat16(),
        pre=torch.softmax(torch.randn(rows, 4, device="cuda", generator=gen), -1),
        post=torch.rand(rows, 4, device="cuda", generator=gen) * 2,
        comb=torch.softmax(torch.randn(rows, 4, 4, device="cuda", generator=gen), -1),
        fn=fns[i]) for i in range(SETS)]
    projections, calls = {}, {}
    for name, mhc in mhcs.items():
        module = modules[name]

        def call(s, mhc=mhc):
            return mhc.post_pre(s["x"], s["residual"], s["post"], s["comb"], s["fn"], scale, base, weight, s["pre"])

        cold = timed([lambda s=s: call(s) for s in sets])
        warm = timed([lambda: call(sets[0])] * SETS)
        splits = H // module.BLOCK_H
        buffers = [(
            s["x"], s["residual"].view(rows, 4 * H), s["post"], s["comb"], s["pre"], s["fn"],
            torch.empty(rows, 4 * H, dtype=torch.bfloat16, device="cuda"),
            torch.empty(splits, rows, 1, 24, device="cuda"), torch.empty(splits, rows, 1, device="cuda"),
            torch.empty(rows, H, dtype=torch.bfloat16, device="cuda"), torch.empty(splits, rows, device="cuda"),
        ) for s in sets]
        stages = ("async", "sync") if hasattr(module, "STAGE_FN") else (None,)
        for stage in stages:
            kwargs = {} if stage is None else {"stage_fn": stage}
            project = module.project_streams(H, "post_pre", block_M=16, **kwargs)
            alone = timed([lambda b=b: project(*b) for b in buffers])
            alone_warm = timed([lambda: project(*buffers[0])] * SETS)
            label = name if stage is None else f"{name}:{stage}"
            print(f"{rows:5d} {label:>18} {cold:10.1f} {warm:10.1f} {alone:13.1f} {alone_warm:13.1f}")
            for t in buffers[0][6:]:
                t.fill_(float("nan"))
            project(*buffers[0])
            projections[label] = [t.clone() for t in buffers[0][6:]]
        calls[name] = call(sets[0])
    ref_label = next(iter(projections))
    ref = projections[ref_label]
    for label, out in projections.items():
        if label == ref_label:
            continue
        r_out, p, s_, c, q_ = (torch.equal(a, b) for a, b in zip(out, ref))
        s_close = torch.allclose(out[2], ref[2], rtol=2e-6, atol=0)
        q_close = torch.allclose(out[4], ref[4], rtol=2e-6, atol=0)
        ok = r_out and p and c and s_close and q_close
        passed += ok
        failed += not ok
        print(f"      {label}: R_out {r_out}, P {p}, C {c} bit-equal; S equal {s_} close {s_close}; Q equal {q_} close {q_close}")
    for name, out in calls.items():
        if name == NAMES[0]:
            continue
        diffs = [(a.float() - b.float()).abs().max().item() for a, b in zip(out, calls[NAMES[0]])]
        print(f"      {name} call vs {NAMES[0]}: max |diff| residual {diffs[0]:.3g}, post {diffs[1]:.3g}, comb {diffs[2]:.3g}, y {diffs[3]:.3g}, pre {diffs[4]:.3g}")
print(f"{passed} passed" + (f", {failed} failed" if failed else ""))
