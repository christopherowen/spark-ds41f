# usage: python3 /h/h963r1.py OUT.json
#
# Round 1 (candidate-51 rework of #963): h963.py with section D run for both screen-config
# sets (SCREEN_SETS) on each arm and M 17, 24 added; everything else unchanged.
#
# #963's own mark on the image's own kernels (no patch step): the timing half of the PR's
# screen_pack12_kernels.py (its shapes, weights, seeds, M lists and tile choices), run
# unchanged on both arms, plus CUDA-graph replay and CUPTI kernel time beside the PR's
# eager loop, output hashes for the cross-arm 0-ULP check, the certified two-pass greedy
# screen that serving runs (#997, in the base), and the split-K packed GEMM that serving's
# compressor runs (the only serving caller of the packed BF16 header #963 changes).
# Sections:
#   A packed_bf16_gemm (the PR's 4 shapes, M 1..512 plus 256)
#   B packed_vocab_projection (tp4 32320, tp3 43096; image tile_config; at the 128-row
#     tile also the other TILE_CONFIGS[128] variant, to separate #963 from #1001)
#   C packed_vocab_greedy_partials + finish_vocab_greedy_pairs (M <= 128, as the PR)
#   D project_head_greedy_pairs through prepare_packed_head (certified screen), FP32 head
#   E TileLangSplitLinearMethod split-K decode (compressor (1024, 5120) x2 FP32 and
#     (512, 5120) BF16), weights with the PR's gemm exception injection
import json
import sys
import traceback

sys.path.insert(0, "/h")
import torch  # noqa: E402

from omlib import eager_us, env_info, gpu_warmup, graph_hot_us, h, kernel_isolated_us, kernel_us  # noqa: E402

import vllm.models.deepseek_v4_1.tilelang.gemm as gemm_mod  # noqa: E402
import vllm.models.deepseek_v4_1.tilelang.vocab as vocab_mod  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.vocab import (  # noqa: E402
    finish_vocab_greedy_pairs,
    pack_bf16_weight,
    prepare_packed_head,
    project_head_greedy_pairs,
    row_tile,
    tile_config,
)

out_path = sys.argv[1]
CFG_S4A2 = dict(block_N=32, threads=256, step_groups=1, num_stages=4, act_stages=2)
CFG_S2 = dict(block_N=32, threads=256, step_groups=1, num_stages=2)
# Round 1: the screen-config sets each arm's certified screen runs (base: the image's
# SCREEN_TILE_CONFIGS; r1: #1001-r1's matched-stage screen tiles).
SCREEN_SETS = {
    "s4a2": {128: dict(block_N=32, threads=256, step_groups=1, num_stages=4, act_stages=2)},
    "r1": {128: dict(block_N=32, threads=256, step_groups=1, num_stages=2)},
}
res = {"env": env_info(), "tile_configs_128": dict(vocab_mod.TILE_CONFIGS[128]),
       "A_gemm": {}, "B_proj": {}, "C_greedy": {}, "D_screen": {}, "E_splitk": {}, "errors": []}
res["warmup_ms"] = gpu_warmup(250.0)


def err(where):
    tb = traceback.format_exc()
    print(f"[error] {where}: {tb}", flush=True)
    res["errors"].append({"where": where, "traceback": tb})


def save():
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)


def cell(fn, big=False):
    d = {}
    d["eager_us"] = eager_us(fn, warmup=10 if big else 25, rep=20 if big else 150)
    d["graph"] = graph_hot_us(fn, calls=3 if big else 20, replays=10 if big else 30)
    d["kernel"] = kernel_us(fn, calls=10 if big else 30)
    d["kernel_isolated_us"] = kernel_isolated_us(fn, calls=10 if big else 20)
    return d


# A. packed_bf16_gemm, the PR's shapes and weights
gemm_shapes = [
    ("compressor_down_512x10240", 512, 10240),
    ("compressor_up_5120x512", 5120, 512),
    ("indexer_wq_b_4096x1280", 4096, 1280),
    ("indexer_wk_128x5120", 128, 5120),
]
for name, N, K in gemm_shapes:
    try:
        torch.manual_seed(42)
        W = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.02
        W.view(-1)[::290] = 1e-6
        p = pack_bf16_weight(W)
        res["A_gemm"][name] = {"exponent_base": int(p.exponent_base), "exceptions": int(p.exceptions)}
        for M in [1, 6, 16, 32, 64, 128, 256, 512]:
            bm = 16 if M <= 16 else (32 if M <= 32 else 64)
            k = gemm_mod.packed_bf16_gemm(N, K, p.exponent_base, block_M=bm, block_N=64, num_stages=2, threads=128)
            A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
            C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
            fn = lambda: k(A, p.values, p.exception_entries, p.row_offsets, C)  # noqa: E731
            fn()
            torch.cuda.synchronize()
            d = cell(fn)
            d["out_hash"] = h(C)
            d["block_M"] = bm
            res["A_gemm"][name][str(M)] = d
            print(f"[A] {name} M={M}: eager {d['eager_us']} graph {d['graph']['median_us']} kernel {d['kernel']['per_call_us']} {d['out_hash']}", flush=True)
        save()
    except Exception:
        err(f"A {name}")

# B/C/D. vocabulary head, the PR's weights (every 300th element 1e-6)
vocab_shapes = [("tp4_32320x5120", 32320, 5120), ("tp3_43096x5120", 43096, 5120)]
for name, N, K in vocab_shapes:
    try:
        torch.manual_seed(42)
        W = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.02
        W.view(-1)[::300] = 1e-6
        p = pack_bf16_weight(W)
        del W
        res["B_proj"][name] = {"exponent_base": int(p.exponent_base), "exceptions": int(p.exceptions)}
        res["C_greedy"][name] = {}
        own = {bm: tile_config(bm, K) for bm in [16, 32, 64, 128]}
        variants = {bm: {"own": own[bm]} for bm in own}
        alt = CFG_S2 if own[128].get("num_stages") == 4 else CFG_S4A2
        variants[128]["alt"] = dict(alt)
        res["B_proj"][name]["variants"] = {str(bm): v for bm, v in variants.items()}
        proj = {}
        greedy = {}
        for bm, vs in variants.items():
            for vname, cfg in vs.items():
                proj[(bm, vname)] = vocab_mod.packed_vocab_projection(N, K, p.exponent_base, block_M=bm, **cfg)
                greedy[(bm, vname)] = vocab_mod.packed_vocab_greedy_partials(N, K, p.exponent_base, block_M=bm, **cfg)
        for M in [1, 6, 12, 16, 24, 32, 48, 64, 96, 107, 128, 512, 2048]:
            bm = row_tile(M)
            torch.manual_seed(1000 + M)
            A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
            C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
            row = {}
            for vname in variants[bm]:
                kp = proj[(bm, vname)]
                fn = lambda kp=kp: kp(A, p.values, p.exception_rows, p.exception_entries, p.row_offsets, C)  # noqa: E731
                fn()
                torch.cuda.synchronize()
                d = cell(fn, big=M > 128)
                d["out_hash"] = h(C)
                row[vname] = d
                print(f"[B] {name} M={M} {vname}: eager {d['eager_us']} graph {d['graph']['median_us']} kernel {d['kernel']['per_call_us']} {d['out_hash']}", flush=True)
                if M <= 128:
                    cfg = variants[bm][vname]
                    nt = (N + cfg["block_N"] - 1) // cfg["block_N"]
                    pv = torch.empty((M, nt), device="cuda", dtype=torch.float32)
                    pi = torch.empty((M, nt), device="cuda", dtype=torch.int32)
                    pairs = torch.empty((M, 4), device="cuda", dtype=torch.float32)
                    kg = greedy[(bm, vname)]
                    kf = finish_vocab_greedy_pairs(nt)
                    fg = lambda kg=kg, kf=kf, pv=pv, pi=pi, pairs=pairs: (  # noqa: E731
                        kg(A, p.values, p.exception_rows, p.exception_entries, p.row_offsets, pv, pi, N),
                        kf(pv, pi, pairs, 0),
                    )
                    fg()
                    torch.cuda.synchronize()
                    dg = cell(fg)
                    dg["pairs_hash"] = h(pairs[:, :2])
                    res["C_greedy"][name].setdefault(str(M), {"block_M": bm})[vname] = dg
                    print(f"[C] {name} M={M} {vname}: eager {dg['eager_us']} graph {dg['graph']['median_us']} kernel {dg['kernel']['per_call_us']} {dg['pairs_hash']}", flush=True)
            row["block_M"] = bm
            res["B_proj"][name][str(M)] = row
            save()
        # D. the certified two-pass greedy screen that serving's greedy verification runs,
        # with each screen-config set on both arms (round 1: #1001-r1 changes
        # SCREEN_TILE_CONFIGS, so the same-set cells separate #963-r1 from it)
        try:
            res["D_screen"][name] = {"image_screen_tile_configs": {str(k): dict(v) for k, v in vocab_mod.SCREEN_TILE_CONFIGS.items()}}
            orig_screen = {k: dict(v) for k, v in vocab_mod.SCREEN_TILE_CONFIGS.items()}
            layers = {}
            for sname, sset in SCREEN_SETS.items():
                for bm, cfg in sset.items():
                    vocab_mod.SCREEN_TILE_CONFIGS[bm] = dict(cfg)
                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(p.values, requires_grad=False)
                layer.weight_exception_rows = torch.nn.Parameter(p.exception_rows, requires_grad=False)
                layer.weight_exception_entries = torch.nn.Parameter(p.exception_entries, requires_grad=False)
                layer.weight_row_offsets = torch.nn.Parameter(p.row_offsets, requires_grad=False)
                layer.packed_bf16_in_features = K
                layer.packed_bf16_exponent_base = p.exponent_base
                layer.packed_bf16_exceptions = p.exceptions
                prepare_packed_head(layer, out_dtype=torch.float32)
                layers[sname] = layer
                vocab_mod.SCREEN_TILE_CONFIGS.clear()
                vocab_mod.SCREEN_TILE_CONFIGS.update({k: dict(v) for k, v in orig_screen.items()})
            own = [sn for sn, sset in SCREEN_SETS.items() if all(orig_screen[bm] == cfg for bm, cfg in sset.items())]
            res["D_screen"][name]["own_set"] = own
            for M in [1, 6, 16, 17, 24, 32, 64, 96, 97, 104, 107, 112, 120, 128]:
                torch.manual_seed(2000 + M)
                x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
                row = {"block_M": row_tile(M)}
                for sname, layer in layers.items():
                    ref = project_head_greedy_pairs(layer, x, vocab=N, vocab_start=0, pair_width=4, scratch_pairs=False).clone()
                    fn = lambda x=x, layer=layer: project_head_greedy_pairs(layer, x, vocab=N, vocab_start=0, pair_width=4, scratch_pairs=True)  # noqa: E731
                    d = cell(fn)
                    d["pairs_hash"] = h(ref[:, :2])
                    row[sname] = d
                    print(f"[D] {name} M={M} {sname}: eager {d['eager_us']} graph {d['graph']['median_us']} kernel {d['kernel']['per_call_us']} {d['pairs_hash']}", flush=True)
                res["D_screen"][name][str(M)] = row
            del layers, layer
        except Exception:
            err(f"D {name}")
        del p, proj, greedy
        torch.cuda.empty_cache()
        save()
    except Exception:
        err(f"B/C {name}")

# E. split-K packed GEMM of the compressor (serving's caller of the changed header)
try:
    from vllm.models.deepseek_v4_1.tilelang import linear
    from vllm.models.deepseek_v4_1.tilelang.linear import TileLangSplitLinearMethod

    for name, n, k, parts, out_dtype in [
        ("compressor_wkv_wgate_1024x5120_x2_fp32", 1024, 5120, 2, torch.float32),
        ("compressor_wkv_512x5120_bf16", 512, 5120, 1, torch.bfloat16),
    ]:
        try:
            torch.manual_seed(42)
            W = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02
            W.view(-1)[::290] = 1e-6
            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(W.contiguous(), requires_grad=False)
            layer.out_dtype = out_dtype
            method = TileLangSplitLinearMethod(parts)
            method.process_weights_after_loading(layer)
            method.declare_layer_kernels(layer)
            res["E_splitk"][name] = {"exponent_base": int(layer.packed_bf16_exponent_base),
                                     "exceptions": int(layer.packed_bf16_exceptions),
                                     "split_rows": int(linear._split_rows(layer))}
            scratch = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
            for M in [1, 6, 16, 24, 32, 48, 64, 96, 107, 128]:
                torch.manual_seed(3000 + M)
                x = torch.randn((M, k), device="cuda", dtype=torch.bfloat16)
                outs = tuple(torch.empty((M, n // parts), device="cuda", dtype=out_dtype) for _ in range(parts))
                fn = lambda x=x, outs=outs: linear._split_bf16(layer, x, outs, scratch)  # noqa: E731
                fn()
                torch.cuda.synchronize()
                d = cell(fn)
                d["out_hash"] = h(torch.cat([o.view(torch.uint8).flatten() for o in outs]))
                res["E_splitk"][name][str(M)] = d
                print(f"[E] {name} M={M}: eager {d['eager_us']} graph {d['graph']['median_us']} kernel {d['kernel']['per_call_us']} {d['out_hash']}", flush=True)
            del layer, scratch
            torch.cuda.empty_cache()
            save()
        except Exception:
            err(f"E {name}")
except Exception:
    err("E import")

save()
print(f"wrote {out_path}; errors: {len(res['errors'])}", flush=True)
