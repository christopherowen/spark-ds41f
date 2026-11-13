"""PR #1005 rework, paired screen: #1005's two changes and one alternative, switched
inside one process on the same serving layers (ffnlayer.py), graphs interleaved
window by window, so each variant's increment over the base is a paired difference.

The container runs candidate-47 without #1005 (overlay ovl/no1005). Candidate-47's
own sources of the three files #1005 changes are mounted at /alt (read-only); the
switched code is those files' own functions, compiled from their source (ast) in the
live module's namespace:
  kernel  moe_swiglu_cast from /alt/moe.py, compiled with the plan's own call
          arguments and bound as the layer plan's decode ``swiglu`` kernel
  hook    MoERunner._forward_impl, MoERunner._apply_quant_method from
          /alt/moe_runner.py; SharedExperts.maybe_forward_async and
          record_input_ready from /alt/shared_experts.py
Variants (the switches are set while a variant's graphs are captured):
  base   candidate-47 without #1005
  kern   + 0765's moe_swiglu_cast
  hook   + 0765's runner/shared-expert change (input-ready event recorded before the gate)
  c47    + both (candidate-47 as built)
  post   + 0765's runner change with the input-ready event recorded where the deferred
         launch happens, after the gate GEMM (record_input_ready returns False: the
         post-gate hook as the 1107 runner code wrote it, which #1005 makes reachable)
Every variant's graph outputs are hashed (bit-exactness against base).

usage: pairbench.py LABEL [--windows N] [--ms ...] [--shapes tp4,tp3] [--variants ...]
"""

import argparse
import ast
import hashlib
import json
import random
import statistics
import sys
import time

sys.path.insert(0, "/h")
import torch

import ffnlayer as F

p = argparse.ArgumentParser()
p.add_argument("label")
p.add_argument("--windows", type=int, default=12)
p.add_argument("--replays", type=int, default=8)
p.add_argument("--layers", type=int, default=6)
p.add_argument("--shapes", default="tp4,tp3")
p.add_argument("--ms", default="1,2,3,4,6,8,12,16,24,32,48")
p.add_argument("--variants", default="base,kern,hook,c47,post")
args = p.parse_args()
SHAPES = {"tp4": 576, "tp3": 768}
MS = [int(m) for m in args.ms.split(",")]
NL, R, WIN = args.layers, args.replays, args.windows
VARIANTS = args.variants.split(",")
ALT = "/alt"


def extract(path, cls, names, module):
    src = open(path).read()
    tree = ast.parse(src)
    nodes = []
    for n in tree.body:
        if cls is None and isinstance(n, ast.FunctionDef) and n.name in names:
            nodes.append(n)
        if cls is not None and isinstance(n, ast.ClassDef) and n.name == cls:
            nodes += [f for f in n.body if isinstance(f, ast.FunctionDef) and f.name in names]
    out = {}
    for f in nodes:
        ns = dict(module.__dict__)
        exec(compile(ast.Module(body=[f], type_ignores=[]), path, "exec"), ns)
        out[f.name] = ns[f.name]
    assert set(out) == set(names), (path, names, list(out))
    return out


from vllm.model_executor.layers.fused_moe.runner import moe_runner as MR  # noqa: E402
from vllm.model_executor.layers.fused_moe.runner import shared_experts as SE  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import moe as MOE  # noqa: E402

ALT_RUNNER = extract(f"{ALT}/moe_runner.py", "MoERunner", ["_forward_impl", "_apply_quant_method"], MR)
ALT_SHARED = extract(f"{ALT}/shared_experts.py", "SharedExperts", ["record_input_ready", "maybe_forward_async"], SE)
ALT_MOE = extract(f"{ALT}/moe.py", None, ["moe_swiglu_cast"], MOE)
LIVE = {
    "_forward_impl": MR.MoERunner._forward_impl,
    "_apply_quant_method": MR.MoERunner._apply_quant_method,
    "maybe_forward_async": SE.SharedExperts.maybe_forward_async,
}
assert not hasattr(SE.SharedExperts, "record_input_ready"), "live tree already has #1005's hook"


def set_hook(mode):
    if mode == "base":
        MR.MoERunner._forward_impl = LIVE["_forward_impl"]
        MR.MoERunner._apply_quant_method = LIVE["_apply_quant_method"]
        SE.SharedExperts.maybe_forward_async = LIVE["maybe_forward_async"]
        if "record_input_ready" in SE.SharedExperts.__dict__:
            delattr(SE.SharedExperts, "record_input_ready")
        return
    MR.MoERunner._forward_impl = ALT_RUNNER["_forward_impl"]
    MR.MoERunner._apply_quant_method = ALT_RUNNER["_apply_quant_method"]
    SE.SharedExperts.maybe_forward_async = ALT_SHARED["maybe_forward_async"]
    if mode == "pre":
        SE.SharedExperts.record_input_ready = ALT_SHARED["record_input_ready"]
    elif mode == "post":
        SE.SharedExperts.record_input_ready = lambda self, x: False
    else:
        raise ValueError(mode)


SPEC = {  # variant: (hook mode, alt kernel)
    "base": ("base", False),
    "kern": ("base", True),
    "hook": ("pre", False),
    "c47": ("pre", True),
    "post": ("post", False),
    "postk": ("post", True),
}


def plans(layers):
    return [layer.ffn.experts._quant_method.modular_experts._plan for layer in layers]


def digest(ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def trace(vc, layer, M):
    events = []
    saved = {}
    targets = [
        (SE.SharedExperts, "maybe_forward_async"),
        (SE.SharedExperts, "maybe_sync_shared_experts_stream"),
        (SE.SharedExperts, "record_input_ready"),
    ]
    from vllm.models.deepseek_v4_1.tilelang import router as RT

    targets.append((RT.TileLangGateRouter, "_compute_routing"))
    for cls, name in targets:
        if not hasattr(cls, name):
            continue
        orig = getattr(cls, name)
        saved[(cls, name)] = (orig, name in cls.__dict__)

        def f(self, *a, _o=orig, _n=name, **k):
            if _n == "_compute_routing":
                events.append("router(" + ("post_gate_hook" if getattr(self, "_post_gate_hook", None) else "no hook") + ")")
            elif _n == "maybe_forward_async":
                events.append(f"shared.maybe_forward_async(recorded={k.get('input_ready_recorded', False)})")
            else:
                events.append(f"shared.{_n}")
            r = _o(self, *a, **k)
            if _n == "record_input_ready":
                events[-1] += f"->{r}"
            return r

        setattr(cls, name, f)
    try:
        inp = F.FFNInputs(layer, M)
        with F.forward_context(vc, inp):
            F.ffn_sublayer(layer, inp)
        torch.cuda.synchronize()
    finally:
        for (cls, name), (orig, own) in saved.items():
            if own:
                setattr(cls, name, orig)
            else:
                delattr(cls, name)
    return events


def run_shape(shape, I):
    from vllm.models.deepseek_v4_1 import l2_prefetch

    t0 = time.time()
    vc = F.make_config(I)
    F.init_dist(vc)
    layers = F.build_layers(vc, NL, first_index=10)
    live_k = plans(layers)[0].swiglu
    fn, a, kw = plans(layers)[0].kernels()["swiglu"]
    assert fn is MOE.moe_swiglu_cast
    alt_k = ALT_MOE["moe_swiglu_cast"](*a, **kw)
    src_live, src_alt = live_k.get_kernel_source(), alt_k.get_kernel_source()
    marker = "if ((0 <= expert) && (0 <= orders[0]))"
    out = {"intermediate": I, "swiglu_args": repr(a), "kernel_source_differs": src_live != src_alt,
           "alt_kernel_has_early_exit": marker in src_alt, "live_kernel_has_early_exit": marker in src_live,
           "schedule": {}, "digests": {}, "timing": {}}
    print(f"[{shape}] built in {time.time() - t0:.1f}s; swiglu args {a}; alt kernel differs {out['kernel_source_differs']}", flush=True)

    def set_variant(v):
        hook, kern = SPEC[v]
        set_hook(hook)
        for pl in plans(layers):
            pl.swiglu = alt_k if kern else live_k

    for v in VARIANTS:
        set_variant(v)
        out["schedule"][v] = {str(M): trace(vc, layers[0], M) for M in (1, 48)}
        print(f"[{shape}] {v}: M=1 {out['schedule'][v]['1']}", flush=True)
    set_variant("base")

    flush_buf = torch.full((96 * 1024 * 1024 // 2,), 1.0, dtype=torch.bfloat16, device=F.DEV)
    inputs = {M: [F.FFNInputs(layer, M, seed=31 * i) for i, layer in enumerate(layers)] for M in MS}

    def body(M, mode, evs=None):
        res = []
        for i, (layer, inp) in enumerate(zip(layers, inputs[M])):
            if mode == "cold":
                flush_buf.add_(1.0)
                evs[i][0].record()
            l2_prefetch.issue(layer.attn._l2pf_ffn, M)
            r = F.ffn_sublayer(layer, inp)
            l2_prefetch.join()
            if mode == "cold":
                evs[i][1].record()
            res.append(r[0])
        return res

    for v in VARIANTS:  # eager warmup of every variant and M, largest M first
        set_variant(v)
        for M in sorted(MS, reverse=True):
            with F.forward_context(vc, inputs[M][0]):
                for _ in range(2):
                    eager = body(M, "chain")
                torch.cuda.synchronize()
                out["digests"].setdefault(v, {})[f"eager/{M}"] = digest(eager)
    graphs = {}
    for v in VARIANTS:
        set_variant(v)
        for M in sorted(MS, reverse=True):
            with F.forward_context(vc, inputs[M][0]):
                for mode in ("chain", "cold"):
                    evs = None
                    if mode == "cold":
                        evs = [(torch.cuda.Event(enable_timing=True, external=True),
                                torch.cuda.Event(enable_timing=True, external=True)) for _ in layers]
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        res = body(M, mode, evs)
                    g.replay()
                    torch.cuda.synchronize()
                    graphs[(v, mode, M)] = (g, evs, res)
                    out["digests"][v][f"{mode}/{M}"] = digest(res)
    set_variant("base")
    print(f"[{shape}] captured {len(graphs)} graphs", flush=True)

    t_end = time.time() + 1.5
    while time.time() < t_end:
        for (g, _, _) in graphs.values():
            g.replay()
        torch.cuda.synchronize()

    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def time_graph(g, evs, mode):
        g.replay()  # untimed lead-in
        if mode == "chain":
            start.record()
            for _ in range(R):
                g.replay()
            stop.record()
            torch.cuda.synchronize()
            return start.elapsed_time(stop) * 1000.0 / (R * NL)
        tot = 0.0
        for _ in range(R):
            g.replay()
            torch.cuda.synchronize()
            tot += sum(a.elapsed_time(b) for a, b in evs) * 1000.0
        return tot / (R * NL)

    rng = random.Random(1005)
    samples = {}
    cells = [(mode, M) for M in MS for mode in ("chain", "cold")]
    for w in range(WIN):
        for (mode, M) in (cells if w % 2 == 0 else cells[::-1]):
            order = VARIANTS[:]
            rng.shuffle(order)
            for v in order:
                g, evs, _ = graphs[(v, mode, M)]
                samples.setdefault(f"{v}/{mode}/{M}", []).append(time_graph(g, evs, mode))
    for (v, mode, M), (g, evs, res) in graphs.items():
        g.replay()
        torch.cuda.synchronize()
        out["digests"][v][f"{mode}/{M}/after"] = digest(res)
    for key, xs in samples.items():
        out["timing"][key] = {"median_us": statistics.median(xs), "windows_us": [round(x, 3) for x in xs]}
    for mode in ("chain", "cold"):
        for M in MS:
            base = samples[f"base/{mode}/{M}"]
            line = f"[{shape}] {mode:5s} M={M:2d} base {statistics.median(base):8.2f}"
            for v in VARIANTS[1:]:
                d = [x - b for x, b in zip(samples[f"{v}/{mode}/{M}"], base)]
                line += f" | {v} {statistics.median(d):+6.2f}"
            print(line, flush=True)
    del graphs, layers
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return out


result = {"label": args.label, "layers": NL, "replays": R, "windows": WIN, "ms": MS, "variants": VARIANTS, "shapes": {}}
for shape in args.shapes.split(","):
    result["shapes"][shape] = run_shape(shape, SHAPES[shape])
    with open(f"/out/{args.label}.json", "w") as f:
        json.dump(result, f, indent=1)
print("done", flush=True)
