# usage: python3 /h/d963.py SECTION OUT.json      (SECTION: gemm | vocab | screen | sconf | sab)
#
# Rework diagnosis for #963 and #1001 (candidate-51 round 1), in one process on one arm:
# each variant of the changed kernels is compiled from its own copy of the module
# (/h/var/*.py, built by mkvar.py from the base and #963 files), and the variants of a
# cell are timed in the order v1..vn, vn..v1 (in-process ABBA), by CUDA-graph replay
# (20 calls a graph, L2-hot) and CUPTI kernel time with a sync after each call.
#   gemm:   packed_bf16_gemm, variants base / pdl (pdl.chain() only) / rb (row bounds only)
#           / 963, the PR's 4 shapes and weights, M 1..512
#   vocab:  packed_vocab_projection and packed_vocab_greedy_partials+finish, variants
#           base / 963 / exc (#963's exceptions, base decode) / dec (base exceptions,
#           #963's decode), TP4 32320x5120, the PR's stressed weights and natural weights
#   screen: project_head_greedy_pairs (certified screen) with the base's and #963's
#           vllm_vocab_decode_vec16 (the only #963 change on that path), TP4, FP32 head,
#           both weight sets, plus a per-kernel breakdown
#   sab:    project_head_greedy_pairs with SCREEN_TILE_CONFIGS[128] s4a2 / s2 in four layers
#           built in the order s4a2, s2, s2, s4a2 (a layer's slot moves graph-replay time by
#           up to 5% for identical kernels, so each config takes an early and a late slot;
#           the two layers of one config are an in-process A/A), #1001's bench shapes and
#           weights, FP32 and BF16 heads, on the arm's own vocab module
#   sconf:  project_head_greedy_pairs with SCREEN_TILE_CONFIGS[128] s4a2 / s2 and
#           SCREEN_TILE_CONFIGS[32] s4a2 / s3 / s2 (#1001's diagnosis on the screen), on the
#           arm's own vocab module, #1001's bench shapes and weights, FP32 and BF16 heads
import importlib.util
import json
import sys
import traceback

sys.path.insert(0, "/h")
import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from omlib import env_info, gpu_warmup, graph_hot_us, h, kernel_isolated_us  # noqa: E402

import vllm.models.deepseek_v4_1.tilelang.vocab as vocab_arm  # noqa: E402

PKG = "vllm.models.deepseek_v4_1.tilelang"
section, out_path = sys.argv[1], sys.argv[2]
res = {"env": env_info(), "section": section, "cells": {}, "errors": []}
res["warmup_ms"] = gpu_warmup(250.0)


def save():
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)


def load(name):
    spec = importlib.util.spec_from_file_location(f"{PKG}._rw_{name}", f"/h/var/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def abba(fns, calls=20, replays=30):
    names = list(fns)
    out = {n: {"graph_us": []} for n in names}
    for n in names + names[::-1]:
        out[n]["graph_us"].append(graph_hot_us(fns[n], calls=calls, replays=replays)["median_us"])
    for n in names:
        out[n]["kernel_isolated_us"] = kernel_isolated_us(fns[n], calls=10)
    return out


def breakdown(fn, calls=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
            torch.cuda.synchronize()
    d = {}
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None) or getattr(ev, "self_cuda_time_total", 0.0)
        if t and ev.count:
            d[ev.key[:50]] = round(t / calls, 2)
    return d


def stressed(N, K):
    torch.manual_seed(42)
    W = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.02
    W.view(-1)[::300] = 1e-6
    return W


def natural(N, K):
    gen = torch.Generator(device="cuda").manual_seed(20261108 + N)
    W = (torch.randn(N, K, generator=gen, device="cuda") * 0.02).bfloat16()
    W[0, 0] = 200.0
    W[100, 50] = -150.0
    return W


def make_layer(mod, p, K, out_dtype):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(p.values, requires_grad=False)
    layer.weight_exception_rows = torch.nn.Parameter(p.exception_rows, requires_grad=False)
    layer.weight_exception_entries = torch.nn.Parameter(p.exception_entries, requires_grad=False)
    layer.weight_row_offsets = torch.nn.Parameter(p.row_offsets, requires_grad=False)
    layer.packed_bf16_in_features = K
    layer.packed_bf16_exponent_base = p.exponent_base
    layer.packed_bf16_exceptions = p.exceptions
    mod.prepare_packed_head(layer, out_dtype=out_dtype)
    return layer


def run_gemm():
    mods = {n: load(f"gemm_{n}") for n in ["base", "pdl", "rb", "963"]}
    shapes = [("compressor_down_512x10240", 512, 10240), ("compressor_up_5120x512", 5120, 512),
              ("indexer_wq_b_4096x1280", 4096, 1280), ("indexer_wk_128x5120", 128, 5120)]
    packed = {}
    for name, N, K in shapes:
        torch.manual_seed(42)
        W = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.02
        W.view(-1)[::290] = 1e-6
        packed[name] = vocab_arm.pack_bf16_weight(W)
    import tilelang
    tilelang.prepare([(mod.packed_bf16_gemm, (N, K, packed[name].exponent_base),
                       dict(block_M=bm, block_N=64, num_stages=2, threads=128))
                      for mod in mods.values() for name, N, K in shapes for bm in (16, 32, 64)])
    for name, N, K in shapes:
        p = packed[name]
        for M in [1, 6, 16, 32, 64, 128, 256, 512]:
            try:
                bm = 16 if M <= 16 else (32 if M <= 32 else 64)
                A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
                fns, outs = {}, {}
                for v, mod in mods.items():
                    k = mod.packed_bf16_gemm(N, K, p.exponent_base, block_M=bm, block_N=64, num_stages=2, threads=128)
                    C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
                    fns[v] = lambda k=k, C=C: k(A, p.values, p.exception_entries, p.row_offsets, C)  # noqa: E731
                    fns[v]()
                    outs[v] = C
                torch.cuda.synchronize()
                hashes = {v: h(c) for v, c in outs.items()}
                d = abba(fns)
                d["hashes"] = hashes
                d["bit_equal"] = len(set(hashes.values())) == 1
                res["cells"][f"gemm|{name}|M={M}"] = d
                print(f"[gemm] {name} M={M}: " + " ".join(f"{v}={d[v]['graph_us']}/{d[v]['kernel_isolated_us']}" for v in mods) + f" eq={d['bit_equal']}", flush=True)
            except Exception:
                res["errors"].append({"where": f"gemm {name} {M}", "tb": traceback.format_exc()})
                print(traceback.format_exc(), flush=True)
        save()


def run_vocab():
    mods = {n: load(f"vocab_{n}") for n in ["base", "963", "exc", "dec"]}
    N, K = 32320, 5120
    cfg128 = dict(block_N=32, threads=256, step_groups=1, num_stages=2)
    for wname, wfn in [("stressed", stressed), ("natural", natural)]:
        W = wfn(N, K)
        p = vocab_arm.pack_bf16_weight(W)
        del W
        res["cells"][f"vocab|{wname}|weights"] = {"exponent_base": int(p.exponent_base), "exceptions": int(p.exceptions)}
        import tilelang
        calls = []
        for mod in mods.values():
            for bm in (16, 32, 64, 128):
                cfg = vocab_arm.tile_config(bm, K) if bm != 128 else dict(cfg128)
                nt = (N + cfg["block_N"] - 1) // cfg["block_N"]
                calls.append((mod.packed_vocab_projection, (N, K, p.exponent_base), dict(block_M=bm, **cfg)))
                calls.append((mod.packed_vocab_greedy_partials, (N, K, p.exponent_base), dict(block_M=bm, **cfg)))
                calls.append((mod.finish_vocab_greedy_pairs, (nt,), {}))
        tilelang.prepare(calls)
        for M in [1, 16, 32, 64, 96, 107, 128]:
            try:
                bm = vocab_arm.row_tile(M)
                cfg = vocab_arm.tile_config(bm, K) if bm != 128 else dict(cfg128)
                torch.manual_seed(1000 + M)
                A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
                pf, gf, po, go = {}, {}, {}, {}
                nt = (N + cfg["block_N"] - 1) // cfg["block_N"]
                for v, mod in mods.items():
                    kp = mod.packed_vocab_projection(N, K, p.exponent_base, block_M=bm, **cfg)
                    C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
                    pf[v] = lambda kp=kp, C=C: kp(A, p.values, p.exception_rows, p.exception_entries, p.row_offsets, C)  # noqa: E731
                    pf[v]()
                    po[v] = C
                    kg = mod.packed_vocab_greedy_partials(N, K, p.exponent_base, block_M=bm, **cfg)
                    kf = mod.finish_vocab_greedy_pairs(nt)
                    pv = torch.empty((M, nt), device="cuda", dtype=torch.float32)
                    pi = torch.empty((M, nt), device="cuda", dtype=torch.int32)
                    pairs = torch.empty((M, 4), device="cuda", dtype=torch.float32)
                    gf[v] = lambda kg=kg, kf=kf, pv=pv, pi=pi, pairs=pairs: (  # noqa: E731
                        kg(A, p.values, p.exception_rows, p.exception_entries, p.row_offsets, pv, pi, N),
                        kf(pv, pi, pairs, 0))
                    gf[v]()
                    go[v] = pairs
                torch.cuda.synchronize()
                for kind, fns, outs in [("proj", pf, po), ("greedy", gf, go)]:
                    hashes = {v: h(o if kind == "proj" else o[:, :2]) for v, o in outs.items()}
                    d = abba(fns)
                    d["hashes"] = hashes
                    d["bit_equal"] = len(set(hashes.values())) == 1
                    d["block_M"] = bm
                    res["cells"][f"vocab|{wname}|{kind}|M={M}"] = d
                    print(f"[vocab] {wname} {kind} M={M}: " + " ".join(f"{v}={d[v]['graph_us']}/{d[v]['kernel_isolated_us']}" for v in mods) + f" eq={d['bit_equal']}", flush=True)
            except Exception:
                res["errors"].append({"where": f"vocab {wname} {M}", "tb": traceback.format_exc()})
                print(traceback.format_exc(), flush=True)
            save()


def run_screen():
    mods = {n: load(f"vocab_{n}") for n in ["base", "963"]}
    N, K = 32320, 5120
    for wname, wfn in [("stressed", stressed), ("natural", natural)]:
        W = wfn(N, K)
        p = vocab_arm.pack_bf16_weight(W)
        del W
        layers = {v: make_layer(mod, p, K, torch.float32) for v, mod in mods.items()}
        for M in [1, 16, 64, 96, 97, 104, 107, 112, 120, 128]:
            try:
                torch.manual_seed(2000 + M)
                x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
                fns, hashes = {}, {}
                for v, mod in mods.items():
                    ref = mod.project_head_greedy_pairs(layers[v], x, vocab=N, vocab_start=0, pair_width=4, scratch_pairs=False).clone()
                    hashes[v] = h(ref[:, :2])
                    fns[v] = lambda mod=mod, L=layers[v]: mod.project_head_greedy_pairs(L, x, vocab=N, vocab_start=0, pair_width=4, scratch_pairs=True)  # noqa: E731
                d = abba(fns)
                d["hashes"] = hashes
                d["bit_equal"] = len(set(hashes.values())) == 1
                d["breakdown"] = {v: breakdown(fns[v]) for v in fns}
                res["cells"][f"screen|{wname}|M={M}"] = d
                print(f"[screen] {wname} M={M}: " + " ".join(f"{v}={d[v]['graph_us']}/{d[v]['kernel_isolated_us']}" for v in mods) + f" eq={d['bit_equal']}", flush=True)
                print(f"    {d['breakdown']}", flush=True)
            except Exception:
                res["errors"].append({"where": f"screen {wname} {M}", "tb": traceback.format_exc()})
                print(traceback.format_exc(), flush=True)
            save()
        del layers
        torch.cuda.empty_cache()


def run_sconf():
    K = 5120
    S128 = {"s4a2": dict(block_N=32, threads=256, step_groups=1, num_stages=4, act_stages=2),
            "s2": dict(block_N=32, threads=256, step_groups=1, num_stages=2)}
    S32 = {"s4a2": dict(block_N=32, threads=128, step_groups=2, num_stages=4, act_stages=2),
           "s3": dict(block_N=32, threads=128, step_groups=2, num_stages=3),
           "s2": dict(block_N=32, threads=128, step_groups=2, num_stages=2)}
    res["image_screen_tile_configs"] = {str(k): dict(v) for k, v in vocab_arm.SCREEN_TILE_CONFIGS.items()}
    orig = {k: dict(v) for k, v in vocab_arm.SCREEN_TILE_CONFIGS.items()}
    for sname, n, org_n in [("tp4", 32320, 32320), ("tp3", 43136, 43072)]:
        W = natural(n, K)
        p = vocab_arm.pack_bf16_weight(W, in_place=True)
        for dname, dt in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
            for tile, variants, ms in [(128, S128, [72, 96, 97, 104, 107, 112, 120, 128]), (32, S32, [17, 24, 32])]:
                layers = {}
                for vn, cfg in variants.items():
                    vocab_arm.SCREEN_TILE_CONFIGS[tile] = dict(cfg)
                    layers[vn] = make_layer(vocab_arm, p, K, dt)
                vocab_arm.SCREEN_TILE_CONFIGS.clear()
                vocab_arm.SCREEN_TILE_CONFIGS.update({k: dict(v) for k, v in orig.items()})
                gen = torch.Generator(device="cuda").manual_seed(777 + n)
                x_all = torch.randn(128, K, generator=gen, device="cuda", dtype=torch.bfloat16)
                for M in ms:
                    try:
                        x = x_all[:M]
                        fns, hashes = {}, {}
                        for vn, L in layers.items():
                            ref = vocab_arm.project_head_greedy_pairs(L, x, vocab=org_n, vocab_start=0, pair_width=4, scratch_pairs=False).clone()
                            hashes[vn] = h(ref[:, :2])
                            fns[vn] = lambda L=L, x=x: vocab_arm.project_head_greedy_pairs(L, x, vocab=org_n, vocab_start=0, pair_width=4, scratch_pairs=True)  # noqa: E731
                        d = abba(fns)
                        d["hashes"] = hashes
                        d["bit_equal"] = len(set(hashes.values())) == 1
                        d["breakdown"] = {vn: breakdown(fns[vn]) for vn in fns}
                        res["cells"][f"sconf|{sname}|{dname}|tile{tile}|M={M}"] = d
                        print(f"[sconf] {sname} {dname} tile{tile} M={M}: " + " ".join(f"{v}={d[v]['graph_us']}/{d[v]['kernel_isolated_us']}" for v in fns) + f" eq={d['bit_equal']}", flush=True)
                    except Exception:
                        res["errors"].append({"where": f"sconf {sname} {dname} {tile} {M}", "tb": traceback.format_exc()})
                        print(traceback.format_exc(), flush=True)
                save()
                del layers
                torch.cuda.empty_cache()


def run_sab():
    K = 5120
    S128 = {"s4a2": dict(block_N=32, threads=256, step_groups=1, num_stages=4, act_stages=2),
            "s2": dict(block_N=32, threads=256, step_groups=1, num_stages=2)}
    order = [("s4a2", "a"), ("s2", "a"), ("s2", "b"), ("s4a2", "b")]
    res["image_screen_tile_configs"] = {str(k): dict(v) for k, v in vocab_arm.SCREEN_TILE_CONFIGS.items()}
    orig = {k: dict(v) for k, v in vocab_arm.SCREEN_TILE_CONFIGS.items()}
    for sname, n, org_n in [("tp4", 32320, 32320), ("tp3", 43136, 43072)]:
        W = natural(n, K)
        p = vocab_arm.pack_bf16_weight(W, in_place=True)
        for dname, dt in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
            layers = {}
            for cfg_name, slot in order:
                vocab_arm.SCREEN_TILE_CONFIGS[128] = dict(S128[cfg_name])
                layers[f"{cfg_name}_{slot}"] = make_layer(vocab_arm, p, K, dt)
            vocab_arm.SCREEN_TILE_CONFIGS.clear()
            vocab_arm.SCREEN_TILE_CONFIGS.update({k: dict(v) for k, v in orig.items()})
            gen = torch.Generator(device="cuda").manual_seed(777 + n)
            x_all = torch.randn(128, K, generator=gen, device="cuda", dtype=torch.bfloat16)
            for M in [1, 16, 64, 72, 96, 97, 104, 107, 112, 120, 128]:
                try:
                    x = x_all[:M]
                    fns, hashes = {}, {}
                    for ln, L in layers.items():
                        ref = vocab_arm.project_head_greedy_pairs(L, x, vocab=org_n, vocab_start=0, pair_width=4, scratch_pairs=False).clone()
                        hashes[ln] = h(ref[:, :2])
                        fns[ln] = lambda L=L, x=x: vocab_arm.project_head_greedy_pairs(L, x, vocab=org_n, vocab_start=0, pair_width=4, scratch_pairs=True)  # noqa: E731
                    d = abba(fns)
                    d["hashes"] = hashes
                    d["bit_equal"] = len(set(hashes.values())) == 1
                    res["cells"][f"sab|{sname}|{dname}|M={M}"] = d
                    print(f"[sab] {sname} {dname} M={M}: " + " ".join(f"{v}={d[v]['graph_us']}/{d[v]['kernel_isolated_us']}" for v in fns) + f" eq={d['bit_equal']}", flush=True)
                except Exception:
                    res["errors"].append({"where": f"sab {sname} {dname} {M}", "tb": traceback.format_exc()})
                    print(traceback.format_exc(), flush=True)
            save()
            del layers
            torch.cuda.empty_cache()


{"gemm": run_gemm, "vocab": run_vocab, "screen": run_screen, "sconf": run_sconf, "sab": run_sab}[section]()
save()
print(f"wrote {out_path}; errors: {len(res['errors'])}", flush=True)
