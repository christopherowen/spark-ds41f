"""PR #1005 rework harness: the serving FFN sublayer (ffnlayer.py) timed by CUDA-graph
replay, on whatever vLLM tree the container mounts (one arm per container).

Per shape (TP4: I = 576, TP3: I = 768) it builds NL serving decoder layers and, for
each decode M, captures two graphs:
  chain  NL layers back to back, each: issue the layer's FFN L2 prefetch (as the
         attention's WO projection does), the FFN sublayer, join the prefetch stream.
         Timed by events around R replays; reported per layer.
  cold   NL layers, each: write a 96 MiB buffer (evicts L2), then the same
         prefetch + FFN sublayer + join between two in-graph timing events.
         Reported per layer (mean of the NL in-graph intervals).
Windows interleave every (mode, M) in turn; the GPU runs >= 1 s of graph replays
before the first timed window. Medians over windows are reported.

Before capture, one eager pass per shape records the shared-expert schedule the runner
takes (method calls in order). Every graph's outputs (all NL layers, after replay) are
hashed for bit-exactness across arms.

usage: ffnbench.py LABEL [--windows N] [--shapes tp4,tp3] [--ms 1,2,3,6,12,24,48]
writes /out/LABEL.json
"""

import argparse
import hashlib
import json
import statistics
import sys
import time

sys.path.insert(0, "/h")
import torch

import ffnlayer as F

p = argparse.ArgumentParser()
p.add_argument("label")
p.add_argument("--windows", type=int, default=21)
p.add_argument("--replays", type=int, default=10)
p.add_argument("--layers", type=int, default=6)
p.add_argument("--shapes", default="tp4,tp3")
p.add_argument("--ms", default="1,2,3,6,12,24,48")
args = p.parse_args()

SHAPES = {"tp4": 576, "tp3": 768}
MS = [int(m) for m in args.ms.split(",")]
NL, R, WIN = args.layers, args.replays, args.windows
result = {"label": args.label, "layers": NL, "replays": R, "windows": WIN, "ms": MS, "shapes": {}}


def trace_schedule(vc, layer, M):
    from vllm.model_executor.layers.fused_moe.runner import shared_experts as se_mod
    from vllm.models.deepseek_v4_1.tilelang import router as router_mod

    events = []
    saved = {}

    def wrap(cls, name, tag):
        if not hasattr(cls, name):
            return
        orig = getattr(cls, name)
        saved[(cls, name)] = orig

        def f(self, *a, **k):
            if tag == "route":
                hook = getattr(self, "_post_gate_hook", None)
                events.append(f"router._compute_routing({'post_gate_hook' if hook else 'no hook'})")
            elif name == "maybe_forward_async":
                events.append(f"shared.maybe_forward_async(input_ready_recorded={k.get('input_ready_recorded', False)})")
            else:
                events.append(f"shared.{name}")
            return orig(self, *a, **k)

        setattr(cls, name, f)

    wrap(se_mod.SharedExperts, "maybe_forward_async", "shared")
    wrap(se_mod.SharedExperts, "maybe_sync_shared_experts_stream", "shared")
    wrap(se_mod.SharedExperts, "record_input_ready", "shared")
    wrap(router_mod.TileLangGateRouter, "_compute_routing", "route")
    try:
        inp = F.FFNInputs(layer, M)
        with F.forward_context(vc, inp):
            F.ffn_sublayer(layer, inp)
        torch.cuda.synchronize()
    finally:
        for (cls, name), orig in saved.items():
            setattr(cls, name, orig)
    return events


def digest(ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def run_shape(shape, I):
    from vllm.models.deepseek_v4_1 import l2_prefetch

    t0 = time.time()
    vc = F.make_config(I)
    F.init_dist(vc)
    layers = F.build_layers(vc, NL, first_index=10)
    print(f"[{shape}] built {NL} layers in {time.time() - t0:.1f}s", flush=True)
    out = {"intermediate": I, "schedule": {}, "graphs": {}}
    for M in (1, 48):
        out["schedule"][str(M)] = trace_schedule(vc, layers[0], M)
    print(f"[{shape}] schedule M=1: {out['schedule']['1']}", flush=True)
    flush_buf = torch.empty(96 * 1024 * 1024 // 2, dtype=torch.bfloat16, device=F.DEV)
    flush_buf.fill_(1.0)
    graphs = {}
    # Every graph's inputs stay alive with it (a graph does not own the tensors it reads).
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

    # Eager warmup of every M, largest first, before any capture, so the workspace
    # reaches its peak before a graph binds it (as the runner warms before capturing).
    eager_digest = {}
    for M in sorted(MS, reverse=True):
        with F.forward_context(vc, inputs[M][0]):
            for _ in range(2):
                eager = body(M, "chain")
            torch.cuda.synchronize()
            eager_digest[M] = digest(eager)
    for M in sorted(MS, reverse=True):
        outs = {}
        with F.forward_context(vc, inputs[M][0]):
            for mode in ("chain", "cold"):
                evs = None
                if mode == "cold":
                    evs = [
                        (torch.cuda.Event(enable_timing=True, external=True),
                         torch.cuda.Event(enable_timing=True, external=True))
                        for _ in layers
                    ]
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    res = body(M, mode, evs)
                g.replay()
                torch.cuda.synchronize()
                graphs[(mode, M)] = (g, evs, res)
                outs[mode] = digest(res)
        out["graphs"][str(M)] = {"eager_digest": eager_digest[M], "graph_digest": outs}
        print(f"[{shape}] M={M} captured; digests eager {eager_digest[M]} graph {outs}", flush=True)
    graphs = {k: graphs[k] for k in sorted(graphs, key=lambda k: (k[1], k[0]))}

    # warm the GPU: >= 1 s of replays
    t_end = time.time() + 1.5
    while time.time() < t_end:
        for (mode, M), (g, _, _) in graphs.items():
            g.replay()
        torch.cuda.synchronize()

    samples = {f"{mode}/{M}": [] for (mode, M) in graphs}
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for w in range(WIN):
        order = list(graphs.items()) if w % 2 == 0 else list(reversed(list(graphs.items())))
        for (mode, M), (g, evs, _) in order:
            g.replay()  # untimed lead-in
            if mode == "chain":
                start.record()
                for _ in range(R):
                    g.replay()
                stop.record()
                torch.cuda.synchronize()
                us = start.elapsed_time(stop) * 1000.0 / (R * NL)
            else:
                tot = 0.0
                for _ in range(R):
                    g.replay()
                    torch.cuda.synchronize()
                    tot += sum(a.elapsed_time(b) for a, b in evs) * 1000.0
                us = tot / (R * NL)
            samples[f"{mode}/{M}"].append(us)
    stats = {}
    for k, xs in samples.items():
        stats[k] = {
            "median_us": statistics.median(xs),
            "mean_us": statistics.fmean(xs),
            "stdev_us": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min_us": min(xs),
            "windows_us": [round(x, 3) for x in xs],
        }
        print(f"[{shape}] {k:10s} median {stats[k]['median_us']:9.2f} us  sd {stats[k]['stdev_us']:6.2f}", flush=True)
    out["timing"] = stats
    for (mode, M), (g, evs, res) in graphs.items():
        g.replay()
        torch.cuda.synchronize()
        out["graphs"][str(M)].setdefault("graph_digest_after_timing", {})[mode] = digest(res)
    del graphs
    del layers
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return out


for shape in args.shapes.split(","):
    result["shapes"][shape] = run_shape(shape, SHAPES[shape])
    with open(f"/out/{args.label}.json", "w") as f:
        json.dump(result, f, indent=1)
print("done", flush=True)
