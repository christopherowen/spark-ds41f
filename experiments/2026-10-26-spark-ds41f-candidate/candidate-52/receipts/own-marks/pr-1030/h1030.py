#!/usr/bin/env python3
"""Own-mark harness for PR #1030 (ours, not the PR's), run on both arms of one session.

What the PR's bench does not give on both arms: an arm-independent reference, and CUDA-graph replay
timing of the decode-path kernel. Here, for the PR bench's six moe_swiglu_cast cases (TP4 inter=576 and
TP3 inter=768; decode block_M=16 rows=16, mid 32/96, prefill 64/2048; its _swiglu_case inputs):

  kernels   image = the arm's moe_swiglu_cast (candidate-50's on base, candidate-52's on c52)
            old   = the PR bench's embedded 1107 moe_swiglu_cast_1107_baseline (same in both arms)
  refs      ds    = candidate-52's test-file reference _tile_kernels_swiglu, the PR's DeepSeek Expert.forward
                    reference ((silu(gate)*up*w).bfloat16() then tile_kernels per_token_cast, UE8M0), mounted
                    from candidate-52's tree in BOTH arms at /ref, so both arms meet the same reference
            torch = a pure-torch cross-check of the same arithmetic: F.silu(gate)*up, *w, .bfloat16(), then
                    per-32 UE8M0 = 2^ceil(log2(amax/448)), e4m3 cast of x/s (with and without amax floor 1e-4)
  diffs     FP8 bytes and scale bytes over live rows (order >= 0), as the PR bench counts them
  timing    eager per-launch CUDA events (the PR bench's method) and CUDA-graph replay: a graph of 20 launches,
            >= 200 ms warm-up per graph, then 60 windows interleaved image/old, per-launch time per window

and the PR bench's end-to-end routed MoE (MoeExpertsV41Plan.run, E=32, top-6, hidden 384, inter 576) at
rows 12 and 300: accuracy of image and old plans against candidate-52's _arithmetic (DeepSeek reference),
and graph-replay timing of plan.run.
usage: h1030.py LABEL  (writes /out/h1030-LABEL.json)
"""

import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/opt/spark-ds41f/candidate/vllm")
LABEL = sys.argv[1]
WINDOWS = 60
REPS = 20
WARM_MS = 200.0


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bra = load("bench_reference_alignment", "/bench/bench_reference_alignment.py")
ref = load("ref_c52_test_moe", "/ref/test_deepseek_v41_tilelang_moe.py")
import vllm.models.deepseek_v4_1.tilelang.moe as moe_mod  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.linear import cast_mxfp8  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.moe import moe_swiglu_cast, workspace_bytes  # noqa: E402

LIMIT = bra.LIMIT


def sha(*ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def summ(xs):
    xs = list(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {"median": round(statistics.median(xs), 4), "mean": round(statistics.fmean(xs), 4),
            "min": round(min(xs), 4), "ci95": round(1.96 * sd / math.sqrt(len(xs)), 4), "n": len(xs)}


def torch_ref(h, w, inter, floor):
    hf = h.float()
    gate = hf[:, :inter].clamp(max=LIMIT)
    up = hf[:, inter:].clamp(min=-LIMIT, max=LIMIT)
    x = F.silu(gate) * up
    x = w[:, None].float() * x
    xb = x.bfloat16().float().view(h.shape[0], inter // 32, 32)
    amax = xb.abs().amax(-1)
    if floor:
        amax = amax.clamp(min=1e-4)
    t = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=h.device)
    bits = t.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    s = torch.exp2(e.float())
    q = (xb / s[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(h.shape[0], inter)
    sf = (e + 127).clamp(0, 255).to(torch.uint8)
    return q, sf


def ndiff(q, sf, rq, rsf, live, cols):
    return {"fp8": int((q[live].view(torch.uint8) != rq[live].view(torch.uint8)).sum().item()),
            "scale": int((sf[live, :cols] != rsf[live, :cols]).sum().item()),
            "fp8_values": int((q[live].float() != rq[live].float()).sum().item())}


def ev(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3


def graph_of(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REPS):
            fn()
    torch.cuda.synchronize()
    return g


def warm(g):
    t0 = time.perf_counter()
    n = 0
    while (time.perf_counter() - t0) * 1e3 < WARM_MS:
        g.replay()
        n += 1
        if n % 50 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return round((time.perf_counter() - t0) * 1e3, 1)


def timed_pair(fns):
    """fns: {arm: fn}. Graph replay, interleaved windows; plus eager per-launch events."""
    graphs = {a: graph_of(f) for a, f in fns.items()}
    warm_ms = {a: warm(g) for a, g in graphs.items()}
    per = {a: [] for a in fns}
    order = list(fns)
    for wi in range(WINDOWS):
        for a in (order if wi % 2 == 0 else order[::-1]):
            per[a].append(ev(graphs[a].replay) / REPS)
    eager = {}
    for a, f in fns.items():
        for _ in range(40):
            f()
        torch.cuda.synchronize()
        eager[a] = summ(ev(f) for _ in range(200))
    return {a: {"graph_us_per_launch": summ(per[a]), "eager_us_per_launch": eager[a],
                "warmup_ms": warm_ms[a]} for a in fns}


def main():
    moe_src = open(moe_mod.__file__).read()
    out = {"label": LABEL, "device": torch.cuda.get_device_name(0),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "moe_py": moe_mod.__file__, "moe_py_sha256": hashlib.sha256(moe_src.encode()).hexdigest()[:12],
           "image_kernel_rounds_to_bf16": "x_bf16[j] = T.cast(val, in_dtype)" in moe_src,
           "image_kernel_has_certificate": "cert_e" in moe_src,
           "windows": WINDOWS, "reps_per_graph": REPS, "warmup_ms_min": WARM_MS,
           "swiglu_cast": {}, "e2e_moe": {}}
    num_experts, topk = 384, 6
    for shape_name, inter in (("tp4_inter576", 576), ("tp3_inter768", 768)):
        cols = inter // 32
        for tier, bm, rows in (("decode_blockM16_rows16", 16, 16), ("mid_blockM32_rows96", 32, 96),
                               ("prefill_blockM64_rows2048", 64, 2048)):
            layout, h = bra._swiglu_case(rows, inter, num_experts, topk, block_M=bm)
            order, w, te, psum = layout["order"], layout["layout_weights"], layout["tile_expert"], layout["psum"]
            # The PR bench's live mask (order >= 0) also admits layout rows past psum[-1], whose order and
            # weights are never written (uninitialized); neither kernel nor the reference writes those rows.
            # Diffs here are over rows < psum[-1] with order >= 0; the PR's live count is kept as well.
            n = int(psum[-1].item())
            live_pr = order >= 0
            live = live_pr & (torch.arange(order.shape[0], device=order.device) < n)
            n_live = int(live.sum().item())
            rq, rsf = bra._swiglu_outputs(h.shape[0], inter)
            ref._tile_kernels_swiglu(inter, num_experts, bm)(h, rq, rsf, psum, w, 1.0, None)
            tq1, tsf1 = torch_ref(h, w, inter, floor=True)
            tq0, tsf0 = torch_ref(h, w, inter, floor=False)
            k = {"image": moe_swiglu_cast(inter, bm, LIMIT),
                 "old1107": bra.moe_swiglu_cast_1107_baseline(inter, bm, LIMIT)}
            outs = {}
            for a, kern in k.items():
                q, sf = bra._swiglu_outputs(h.shape[0], inter)
                kern(h, q, sf, order, w, te)
                outs[a] = (q, sf)
            torch.cuda.synchronize()
            rec = {"live_rows": n_live, "live_elements": n_live * inter, "live_scales": n_live * cols,
                   "live_rows_pr_mask": int(live_pr.sum().item()), "psum_last": n,
                   "pr_mask_rows_past_psum_all_zero": {
                       a_: bool((q_[live_pr & ~live].view(torch.uint8) == 0).all().item())
                       for a_, (q_, _) in (("ref", (rq, rsf)),)},
                   "input_sha": sha(h[:n], order[:n], w[:n]), "ref_ds_sha": sha(rq[live], rsf[live, :cols]),
                   "ref_ds_vs_torch_floor": ndiff(rq, rsf, tq1, tsf1, live, cols),
                   "ref_ds_vs_torch_nofloor": ndiff(rq, rsf, tq0, tsf0, live, cols)}
            for a, (q, sf) in outs.items():
                rec[a] = {"out_sha": sha(q[live], sf[live, :cols]),
                          "diffs_vs_ds_ref": ndiff(q, sf, rq, rsf, live, cols),
                          "diffs_vs_torch_floor": ndiff(q, sf, tq1, tsf1, live, cols),
                          "diffs_vs_torch_nofloor": ndiff(q, sf, tq0, tsf0, live, cols)}
            rec["image_equals_old1107"] = rec["image"]["out_sha"] == rec["old1107"]["out_sha"]
            for a, (q, sf) in outs.items():
                rec[a]["diffs_vs_ds_ref_pr_mask"] = ndiff(q, sf, rq, rsf, live_pr, cols)

            def mk(kern, q, sf):
                return lambda: kern(h, q, sf, order, w, te)

            t = timed_pair({a: mk(k[a], *outs[a]) for a in k})
            for a in k:
                rec[a].update(t[a])
            out["swiglu_cast"].setdefault(shape_name, {})[tier] = rec
            print(f"[{LABEL}] {shape_name} {tier}: image diffs {rec['image']['diffs_vs_ds_ref']} "
                  f"old {rec['old1107']['diffs_vs_ds_ref']} | graph us image "
                  f"{rec['image']['graph_us_per_launch']['median']} old {rec['old1107']['graph_us_per_launch']['median']}",
                  flush=True)

    # End-to-end routed MoE, as the PR bench builds it.
    hidden, inter, ne, tk = 384, 576, 32, 6
    w13_packed, w13_sf, w13 = bra._experts(ne, 2 * inter, hidden, 1)
    w2_packed, w2_sf, w2 = bra._experts(ne, hidden, inter, 2)
    plans = {"image": bra._plan(ne, tk, hidden, inter, swiglu_limit=LIMIT),
             "old1107": bra._plan(ne, tk, hidden, inter, swiglu_limit=LIMIT)}
    swapped = []
    for attr, bmv in (("swiglu", 16), ("mid_swiglu", 32), ("prefill_swiglu", 64)):
        if hasattr(plans["old1107"], attr):
            setattr(plans["old1107"], attr, bra.moe_swiglu_cast_1107_baseline(inter, bmv, LIMIT))
            swapped.append(attr)
    out["e2e_moe"]["old1107_swapped_attrs"] = swapped
    pw = {a: p.prepare_weights(w13_packed, w13_sf, w2_packed, w2_sf) for a, p in plans.items()}
    for rows in (12, 300):
        ids, rw = bra._routes(rows, ne, tk, 4, torch.int32)
        ws = {a: torch.empty(workspace_bytes(p.workspace_specs(rows, torch.int32)), dtype=torch.uint8,
                             device="cuda") for a, p in plans.items()}
        rec = {"accuracy": {}}
        for name, scale in (("small_scale", 0.1), ("normal", 0.5), ("heavy_clamp", 25.0)):
            x = (torch.randn((rows, hidden), generator=bra._gen(3), device="cuda") * scale).bfloat16()
            r_ds = ref._arithmetic(x, ids, rw, w13, w2, inter)
            tokens = cast_mxfp8(x)
            row = {}
            for a, p in plans.items():
                y = torch.empty_like(x)
                p.run(tokens, ids, rw, pw[a], y, ws[a])
                torch.cuda.synchronize()
                row[a] = {"vs_ds_ref": bra._bf16_ulp_stats(y, r_ds), "out_sha": sha(y)}
            rec["accuracy"][name] = row
        x = (torch.randn((rows, hidden), generator=bra._gen(3), device="cuda") * 0.5).bfloat16()
        tokens = cast_mxfp8(x)
        ys = {a: torch.empty_like(x) for a in plans}

        def mkrun(a):
            return lambda: plans[a].run(tokens, ids, rw, pw[a], ys[a], ws[a])

        try:
            t = timed_pair({a: mkrun(a) for a in plans})
            rec["timing"] = t
        except Exception as exc:  # record, then fall back to eager
            torch.cuda.synchronize()
            rec["graph_capture_error"] = repr(exc)[:400]
            rec["timing"] = {}
            for a in plans:
                f = mkrun(a)
                for _ in range(40):
                    f()
                torch.cuda.synchronize()
                rec["timing"][a] = {"eager_us_per_launch": summ(ev(f) for _ in range(200))}
        out["e2e_moe"][f"rows{rows}"] = rec
        print(f"[{LABEL}] e2e rows{rows}: " + json.dumps({a: rec["timing"][a].get("graph_us_per_launch", rec["timing"][a].get("eager_us_per_launch"))["median"] for a in plans}), flush=True)
    out["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"/out/h1030-{LABEL}.json", "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
