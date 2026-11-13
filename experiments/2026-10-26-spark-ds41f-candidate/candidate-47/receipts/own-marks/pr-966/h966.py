#!/usr/bin/env python3
"""Own-mark harness for PR #966 (ours, not the PR's): the routed-MoE plan and the
full FFN layer of the PR's bench_moe_l2_evict.py, timed eagerly and by CUDA-graph
replay, with output hashes for cross-image bit-exactness.

Arms in one process, interleaved window by window:
  default       the image's own grouped_fp8_fp4_gemm default (1107: EVICT_NORMAL;
                candidate-47: evict_first when swizzle_panel <= 0)
  evict_normal  explicit kwarg (candidate image only; 1107 has no such kwarg)
  evict_first   explicit kwarg (candidate image only)
The workload (weights, routing, shared-expert GEMMs, l2_prefetch FFN plan) is the
PR bench's, built from the PR's own build_plan.
usage: h966.py LABEL   (writes /out/h966-LABEL.json)
"""

import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time

import torch

spec = importlib.util.spec_from_file_location("b966", "/bench/bench_moe_l2_evict.py")
b966 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b966)

from vllm.models.deepseek_v4_1 import l2_prefetch  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.gemm import (  # noqa: E402
    mxfp8_gemm_decode,
    pack_weight_tiles,
    scale_words,
)
from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows, mxfp8_rows  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.moe import workspace_bytes  # noqa: E402
from vllm.models.glm5next.nvidia import l2_prefetch as _l2pf  # noqa: E402

LABEL = sys.argv[1]
DEV = torch.device("cuda")
E, H, TOPK, NL = 384, 5120, 6, 16
WINDOWS = 15


def summ(xs):
    xs = list(xs)
    n = len(xs)
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    return {
        "median": round(statistics.median(xs), 3),
        "mean": round(mean, 3),
        "ci95": round(1.96 * sd / math.sqrt(n), 3),
        "n": n,
    }


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def make_x(M, g):
    padded = mxfp8_rows(M)
    x_bf16 = torch.randn((M, H), generator=g, dtype=torch.bfloat16, device=DEV)
    x_val = torch.randint(0, 120, (padded, H), generator=g, dtype=torch.uint8, device=DEV).view(
        torch.float8_e4m3fn
    )
    x_sf = torch.randint(120, 128, (padded, 4 * scale_words(H)), generator=g, dtype=torch.uint8, device=DEV)
    return MXFP8Rows(x_bf16, x_val, x_sf)


def timed(fn, pre=None):
    """One event pair around fn(); pre() runs before the start event."""
    if pre is not None:
        pre()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    return s, e


def main():
    out = {"label": LABEL, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "device": torch.cuda.get_device_name(0), "windows": WINDOWS, "shapes": {}}
    l2_prefetch.warmup()
    for shape_name, I in [("TP3", 768), ("TP4", 576)]:
        res = {"arms": [], "bit_exact": {}, "moe": {}, "ffn": {}}
        arms = {"default": b966.build_plan(I, eviction_policy=None, max_rows=4096)}
        try:
            arms["evict_normal"] = b966.build_plan(I, eviction_policy="evict_normal", max_rows=4096)
            arms["evict_first"] = b966.build_plan(I, eviction_policy="evict_first", max_rows=4096)
        except Exception as exc:  # 1107 has no eviction_policy kwarg
            res["explicit_policy_arms_error"] = repr(exc)[:300]
        res["arms"] = list(arms)
        g = gen(42 + I)
        w13 = torch.randint(0, 256, (E, 2 * I, H // 2), generator=g, dtype=torch.uint8, device=DEV)
        w13_scale = torch.randint(120, 128, (E, 2 * I, H // 32), generator=g, dtype=torch.uint8, device=DEV)
        w2 = torch.randint(0, 256, (E, H, I // 2), generator=g, dtype=torch.uint8, device=DEV)
        w2_scale = torch.randint(120, 128, (E, H, I // 32), generator=g, dtype=torch.uint8, device=DEV)
        weights = arms["default"].prepare_weights(w13, w13_scale, w2, w2_scale)
        del w13, w13_scale, w2, w2_scale

        # Bit-exactness: random routing (the PR bench's check) at every decode M.
        for M in [1, 2, 4, 6, 8, 16, 32, 64]:
            gm = gen(1000 + M)
            x = make_x(M, gm)
            topk_ids = torch.randint(0, E, (M, TOPK), generator=gm, dtype=torch.int32, device=DEV)
            topk_w = torch.softmax(torch.randn((M, TOPK), generator=gm, dtype=torch.float32, device=DEV), dim=-1)
            shared = torch.randn((M, H), generator=gm, dtype=torch.bfloat16, device=DEV)
            wsb = workspace_bytes(arms["default"].workspace_specs(M, torch.int32))
            hashes, outs = {}, {}
            for name, plan in arms.items():
                ws = torch.zeros(wsb, dtype=torch.uint8, device=DEV)
                o = torch.empty((M, H), dtype=torch.bfloat16, device=DEV)
                plan.run(x, topk_ids, topk_w, weights, o, workspace=ws, shared=shared)
                torch.cuda.synchronize()
                outs[name] = o
                hashes[name] = sha(o)
            res["bit_exact"][str(M)] = {
                "sha256_16": hashes,
                "nan": bool(torch.isnan(outs["default"]).any().item()),
                "equal_in_image": all(torch.equal(outs["default"], o) for o in outs.values()),
            }

        # Shared-expert GEMMs and the FFN prefetch plan, as in the PR bench.
        gs = gen(7 + I)
        sh_w13 = torch.randint(0, 120, (2 * I, H), generator=gs, dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn)
        sh_w13_packed = pack_weight_tiles(sh_w13)
        sh_s13 = torch.randint(120, 128, ((2 * I) // 32, scale_words(H)), generator=gs, dtype=torch.int32, device=DEV).view(torch.uint32)
        sh_w2 = torch.randint(0, 120, (H, I), generator=gs, dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn)
        sh_w2_arg = pack_weight_tiles(sh_w2) if (I % 128 == 0) else sh_w2
        sh_s2 = torch.randint(120, 128, (H // 32, scale_words(I)), generator=gs, dtype=torch.int32, device=DEV).view(torch.uint32)
        sh_gu = mxfp8_gemm_decode(2 * I, H, block_M=16, block_N=128, block_K=128, padded_rows=True, tile_contiguous=True)
        sh_dn = mxfp8_gemm_decode(H, I, block_M=16, block_N=128, block_K=128 if I % 128 == 0 else 64,
                                  padded_rows=True, tile_contiguous=(I % 128 == 0))

        for section, Ms in (("moe", [1, 2, 4, 6, 8, 16, 32, 64]), ("ffn", [1, 2, 4, 6, 8, 16, 32])):
            for M in Ms:
                gm = gen(2000 + M)
                x = make_x(M, gm)
                tl = [((torch.arange(M * TOPK, device=DEV, dtype=torch.int32).view(M, TOPK) + l * 23) % E).contiguous()
                      for l in range(NL)]
                topk_w = torch.softmax(torch.randn((M, TOPK), generator=gm, dtype=torch.float32, device=DEV), dim=-1)
                shared_out = torch.randn((M, H), generator=gm, dtype=torch.bfloat16, device=DEV)
                padded = mxfp8_rows(M)
                sh_q = torch.randint(0, 120, (padded, I), generator=gm, dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn)
                sh_sf = torch.randint(120, 128, (padded, scale_words(I)), generator=gm, dtype=torch.int32, device=DEV).view(torch.uint32)
                sh_gu_out = torch.empty((M, 2 * I), dtype=torch.bfloat16, device=DEV)
                residual = torch.randn((M, 4, H), generator=gm, dtype=torch.bfloat16, device=DEV)
                wsb = workspace_bytes(arms["default"].workspace_specs(M, torch.int32))
                ws = torch.empty(wsb, dtype=torch.uint8, device=DEV)
                o = torch.empty((M, H), dtype=torch.bfloat16, device=DEV)
                segs = [_l2pf.tensor_segment("sh_w13", sh_w13_packed, min_bytes=0),
                        _l2pf.tensor_segment("sh_w2", sh_w2_arg, min_bytes=0),
                        _l2pf.tensor_segment("residual", residual, min_bytes=0)]
                pf = l2_prefetch.make_plan(segs, l2_prefetch.BUDGET_FFN, DEV)

                def prefetch():
                    l2_prefetch.issue(pf, M)
                    l2_prefetch.join()

                def body(plan, l):
                    if section == "ffn":
                        sh_gu(x.values, sh_w13_packed, x.scales.view(torch.uint32), sh_s13, sh_gu_out)
                    plan.run(x, tl[l], topk_w, weights, o, workspace=ws, shared=shared_out)
                    if section == "ffn":
                        sh_dn(sh_q, sh_w2_arg, sh_sf, sh_s2, shared_out)

                # warm up and capture: one graph per routing table (replayed with the
                # eager FFN prefetch before it, as the PR bench times its eager loop),
                # and one graph of 16 back-to-back layers (no prefetch).
                graphs, chains = {}, {}
                for name, plan in arms.items():
                    for i in range(2 * NL):
                        if section == "ffn":
                            prefetch()
                        body(plan, i % NL)
                    torch.cuda.synchronize()
                    gl = []
                    for l in range(NL):
                        gg = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(gg):
                            body(plan, l)
                        gl.append(gg)
                    graphs[name] = gl
                    gc = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gc):
                        for l in range(NL):
                            body(plan, l)
                    chains[name] = gc
                    for gg in gl:
                        gg.replay()
                    gc.replay()
                torch.cuda.synchronize()
                pre = prefetch if section == "ffn" else None
                samples = {name: {"eager": [], "graph": [], "chain16_graph": []} for name in arms}
                for _ in range(WINDOWS):
                    for name, plan in arms.items():
                        ev = [timed(lambda l=l: body(plan, l), pre) for l in range(NL)]
                        torch.cuda.synchronize()
                        samples[name]["eager"].append(statistics.fmean(s.elapsed_time(e) * 1e3 for s, e in ev))
                        ev = [timed(graphs[name][l].replay, pre) for l in range(NL)]
                        torch.cuda.synchronize()
                        samples[name]["graph"].append(statistics.fmean(s.elapsed_time(e) * 1e3 for s, e in ev))
                        s, e = timed(chains[name].replay)
                        torch.cuda.synchronize()
                        samples[name]["chain16_graph"].append(s.elapsed_time(e) * 1e3 / NL)
                row = {name: {mode: summ(v) for mode, v in d.items()} for name, d in samples.items()}
                if "evict_normal" in arms:
                    row["paired_first_minus_normal_us"] = {
                        mode: summ([a - b for a, b in zip(samples["evict_first"][mode], samples["evict_normal"][mode])])
                        for mode in ("eager", "graph", "chain16_graph")
                    }
                res[section][str(M)] = row
                print(shape_name, section, M, json.dumps({n: {m: v["median"] for m, v in d.items()} for n, d in row.items()
                                                          if not n.startswith("paired")}), flush=True)
                del graphs, chains
                torch.cuda.empty_cache()
        out["shapes"][shape_name] = res
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"/out/h966-{LABEL}.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", f"/out/h966-{LABEL}.json")


if __name__ == "__main__":
    main()
