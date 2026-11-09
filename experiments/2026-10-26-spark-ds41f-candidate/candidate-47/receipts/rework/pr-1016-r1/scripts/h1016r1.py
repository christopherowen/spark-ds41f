#!/usr/bin/env python3
"""Rework harness for PR #1016 (round 1), ours.

Times, by CUDA-graph replay, the chained Q-B + indexer Q-B decode pair in three
L2 states (warm: 20 back-to-back pairs; cold: 48 MiB flush before one pair; NEXT:
flush, then the NEXT-window L2 prefetch of the pair's weights and scales, joined,
before one pair), with tile-contiguous weights, at M in (1, 2, 3, 6, 12, 24, 48).

Pairs:
  mark_tp4 / mark_tp3  the own-mark pair (h1016's): plain Q-B decode with the
                       arm's (N, 1280) tile + the fused RoPE/MXFP4 indexer kernel
                       built from the arm's (4096, 1280) tile, as the PR bench does.
  prod_tp4 / prod_tp3  the pair as the model launches it: plain Q-B decode with the
                       arm's tile + the indexer kernel exactly as linear.py builds
                       ``decode_rope_mxfp4`` (kernel defaults, block_K=pick_block_K,
                       padded_rows, tile_contiguous); DECODE_FP8_CONFIGS never
                       reaches it.
  qb_tp8               the TP8 Q-B plain decode (4096 x 1280), the model's other
                       consumer of the (4096, 1280) tiles.
Arms are config tables evaluated in this process on the same weights; "image" is
the image's fp8_decode_config. Identical kernels are timed once per window and
shared by the arms that use them. Windows interleave the distinct kernels, order
alternating each window. The GPU runs >= 100 ms of the block's work before each
(pair, M) block's windows. Bits: every arm's outputs are torch.equal to the 1107
arm's and hashed for cross-image comparison.

usage: h1016r1.py LABEL MODE   MODE in {screen, mark}; writes /out/h1016r1-LABEL.json
"""

import hashlib
import json
import math
import statistics
import sys
import time

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4_1 import l2_prefetch as ds_l2pf
from vllm.models.deepseek_v4_1.tilelang import gemm as ds_gemm
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    DECODE_ROWS,
    fp8_decode_config,
    mxfp8_gemm_decode,
    mxfp8_gemm_rope_mxfp4_decode,
    pack_scale_words,
    pack_weight_tiles,
    pick_block_K,
)
from vllm.models.deepseek_v4_1.tilelang.linear import cast_mxfp8

LABEL, MODE = sys.argv[1], sys.argv[2]
DEV = torch.device("cuda")
WINDOWS = 40
REPS = 20
WARM_MS = 100.0
BATCH_SIZES = (1, 2, 3, 6, 12, 24, 48)
K = 1280
HEADS, HEAD_DIM, ROPE_DIM = 32, 128, 64


def block_m_for(m):
    return 16 if m <= 16 else (32 if m <= 32 else 64)


def d(**kw):
    return kw


# 1107's DECODE_FP8_CONFIGS for the three tile-contiguous Q-B shapes (gemm.py at
# 1107, blob c2443adcb611), threads=64 at (4096, 1280) block_M=64 included.
T1107 = {
    12288: {16: d(block_N=128, block_K=128, num_stages=3), 32: d(block_N=64, block_K=128, num_stages=3),
            64: d(block_N=64, block_K=128, num_stages=2)},
    8192: {16: d(block_N=128, block_K=128, num_stages=3), 32: d(block_N=64, block_K=128, num_stages=3),
           64: d(block_N=64, block_K=128, num_stages=3)},
    4096: {16: d(block_N=64, block_K=128, num_stages=3), 32: d(block_N=128, block_K=128, num_stages=3),
           64: d(block_N=64, block_K=128, num_stages=2, threads=64)},
}
# PR #1016 as submitted (candidate-47's 0768).
TPR = {
    12288: {16: d(block_N=128, block_K=128, num_stages=3), 32: d(block_N=128, block_K=128, num_stages=3),
            64: d(block_N=128, block_K=128, num_stages=2)},
    8192: {16: d(block_N=128, block_K=128, num_stages=3), 32: d(block_N=128, block_K=128, num_stages=3),
           64: d(block_N=128, block_K=128, num_stages=2)},
    4096: {16: d(block_N=128, block_K=128, num_stages=3), 32: d(block_N=128, block_K=128, num_stages=3),
           64: d(block_N=64, block_K=128, num_stages=2)},
}


def mix(base, over):
    out = {n: dict(v) for n, v in base.items()}
    for (n, bm), cfg in over.items():
        out[n][bm] = cfg
    return out


TABLES = {"1107": T1107, "pr1016": TPR,
          # This rework (r1): 1107 + #1016's 64-row Q-B tiles (12288, 8192) only.
          "r1": mix(T1107, {(12288, 64): TPR[12288][64], (8192, 64): TPR[8192][64]}),
          # 1107 + threads=64 dropped at (4096, 1280) block_M=64 only (#1016's unstated change).
          "thr48": mix(T1107, {(4096, 64): TPR[4096][64]})}
if MODE == "screen":
    TABLES.update({
        # 1107 + #1016's block_M=64 rows only.
        "m48": mix(T1107, {(12288, 64): TPR[12288][64], (8192, 64): TPR[8192][64], (4096, 64): TPR[4096][64]}),
        # 1107 + the Q-B retile at block_M=64 (12288, 8192) only.
        "qb48": mix(T1107, {(12288, 64): TPR[12288][64], (8192, 64): TPR[8192][64]}),
        # 1107 + threads=64 dropped at (4096, 1280) block_M=64 only.
        "thr48": mix(T1107, {(4096, 64): TPR[4096][64]}),
        # 1107 + #1016's block_M=32 rows only.
        "m24": mix(T1107, {(12288, 32): TPR[12288][32], (8192, 32): TPR[8192][32]}),
    })


def cfg_of(arm, n, bm):
    if arm == "image":
        return fp8_decode_config(n, K, bm)
    c = dict(block_M=bm, block_N=64, block_K=pick_block_K(K), num_stages=3, threads=128)
    c.update(TABLES[arm][n][bm])
    return c


ARMS = list(TABLES) + ["image"]
PROD_ROPE = dict(head_dim=HEAD_DIM, rope_dim=ROPE_DIM, block_K=pick_block_K(K), padded_rows=True,
                 tile_contiguous=True)


def summ(xs):
    xs = list(xs)
    n = len(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    return {"median": round(statistics.median(xs), 3), "mean": round(statistics.fmean(xs), 3),
            "ci95": round(1.96 * sd / math.sqrt(n), 3), "n": n}


def sha(*ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def ev_time(fn, pre=None, div=1):
    if pre is not None:
        pre()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / div


def spin(fn, ms=WARM_MS):
    """Run ``fn`` (a warm graph replay) until at least ``ms`` of GPU time elapsed."""
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    total = 0.0
    while total < ms:
        for _ in range(20):
            fn()
        e.record()
        torch.cuda.synchronize()
        total = s.elapsed_time(e)
    return round(total, 1)


KCACHE = {}


def qb_kernel(n, cfg):
    key = ("qb", n, tuple(sorted(cfg.items())))
    if key not in KCACHE:
        KCACHE[key] = mxfp8_gemm_decode(n, K, **cfg, padded_rows=True, tile_contiguous=True)
    return KCACHE[key]


def rope_kernel(cfg):
    key = ("rope", tuple(sorted(cfg.items())))
    if key not in KCACHE:
        KCACHE[key] = mxfp8_gemm_rope_mxfp4_decode(4096, K, **cfg)
    return KCACHE[key]


def main():
    out = {"label": LABEL, "mode": MODE, "device": torch.cuda.get_device_name(0), "windows": WINDOWS,
           "reps_warm": REPS, "warm_ms_before_block": WARM_MS,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "gemm_file": ds_gemm.__file__,
           "image_tile_contiguous_fp8": sorted(list(k) for k in ds_gemm.TILE_CONTIGUOUS_FP8),
           "image_cfg": {f"{n}x{K}/bM{bm}": fp8_decode_config(n, K, bm)
                         for n in (16384, 12288, 8192, 6144, 4096, 2048) for bm in (16, 32, 64)},
           "tables": {a: {str(n): v for n, v in t.items()} for a, t in TABLES.items()},
           "prod_rope_kwargs": PROD_ROPE, "pairs": {}, "diag": {}, "layer_chain": {}}
    g = torch.Generator(device=DEV).manual_seed(20261109)
    ds_l2pf.warmup()
    flush_buf = torch.empty((48 * 1024 * 1024,), dtype=torch.uint8, device=DEV)
    x_bf16 = torch.randn((DECODE_ROWS, K), generator=g, device=DEV, dtype=torch.bfloat16)
    mx = cast_mxfp8(x_bf16)
    a_fp8 = mx.values
    s_a_w = mx.scales.contiguous().view(torch.uint32)

    def weights(n):
        w = (torch.randn((n, K), generator=g, device=DEV, dtype=torch.bfloat16) * 0.1).to(torch.float8_e4m3fn)
        s = pack_scale_words(torch.randint(120, 126, (n // 32, K // 32), generator=g, device=DEV,
                                           dtype=torch.uint8))
        return pack_weight_tiles(w), s

    w_idx, s_idx = weights(4096)
    cos_sin = torch.randn((4096, ROPE_DIM), generator=g, device=DEV, dtype=torch.float32)
    pos_all = torch.arange(DECODE_ROWS, device=DEV, dtype=torch.int64)
    qb_w = {n: weights(n) for n in (8192, 12288)}
    qb_w[4096] = (w_idx, s_idx)  # TP8 Q-B: same shape as the indexer, its own call

    def prep_cold():
        flush_buf.add_(1)

    def plan_for(tensors):
        segs = [s for s in (ds_l2pf._l2pf.tensor_segment(f"t{i}", t, min_bytes=0)
                            for i, t in enumerate(tensors)) if s is not None]
        return ds_l2pf.make_plan(segs, ds_l2pf.BUDGET_NEXT, DEV)

    def bench_block(variants, plan, m):
        """variants: {vid: (run, outs)}; returns samples {vid: {state: [...]}} and spin ms."""
        graphs = {}
        for vid, (run, _) in variants.items():
            for _ in range(5):
                run()
            torch.cuda.synchronize()
            gw, g1 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(gw):
                for _ in range(REPS):
                    run()
            with torch.cuda.graph(g1):
                run()
            gw.replay()
            g1.replay()
            graphs[vid] = (gw, g1)
        torch.cuda.synchronize()

        def prep_next():
            flush_buf.add_(1)
            ds_l2pf.issue(plan, m)
            ds_l2pf.join()

        vids = list(variants)
        spun = spin(graphs[vids[0]][0].replay)
        samples = {v: {"graph_warm": [], "graph_cold": [], "graph_next": []} for v in vids}
        for w in range(WINDOWS):
            order = vids if w % 2 == 0 else vids[::-1]
            for v in order:
                gw, g1 = graphs[v]
                samples[v]["graph_warm"].append(ev_time(gw.replay, div=REPS))
                samples[v]["graph_cold"].append(ev_time(g1.replay, prep_cold))
                samples[v]["graph_next"].append(ev_time(g1.replay, prep_next))
        del graphs
        return samples, spun

    pair_defs = [
        ("mark_tp4", 8192, "mark"), ("mark_tp3", 12288, "mark"),
        ("prod_tp4", 8192, "prod"), ("prod_tp3", 12288, "prod"),
        ("qb_tp8", 4096, "qb"),
    ]
    for pname, n_qb, kind in pair_defs:
        w_q, s_q = qb_w[n_qb]
        tensors = [w_q, s_q] if kind == "qb" else [w_q, s_q, w_idx, s_idx]
        plan = plan_for(tensors)
        rows = {}
        for m in BATCH_SIZES:
            bm = block_m_for(m)
            pos = pos_all[:m]
            arm_vid, variants, vcfg = {}, {}, {}
            for arm in ARMS:
                cq = cfg_of(arm, n_qb, bm)
                if kind == "mark":
                    ci = dict(head_dim=HEAD_DIM, rope_dim=ROPE_DIM, **cfg_of(arm, 4096, bm), padded_rows=True,
                              tile_contiguous=True)
                elif kind == "prod":
                    ci = PROD_ROPE
                else:
                    ci = None
                vkey = json.dumps([cq, ci], sort_keys=True)
                arm_vid[arm] = vkey
                if vkey in variants:
                    continue
                kq = qb_kernel(n_qb, cq)
                ki = rope_kernel(ci) if ci is not None else None
                y = torch.empty((m, n_qb), device=DEV, dtype=torch.bfloat16)
                p = torch.empty((m, HEADS * (HEAD_DIM // 2)), device=DEV, dtype=torch.uint8)
                sc = torch.empty((m, HEADS * (HEAD_DIM // 32)), device=DEV, dtype=torch.uint8)
                y.fill_(0)
                p.fill_(0)
                sc.fill_(0)

                def run(kq=kq, ki=ki, y=y, p=p, sc=sc, w_q=w_q, s_q=s_q):
                    kq(a_fp8, w_q, s_a_w, s_q, y)
                    if ki is not None:
                        ki(a_fp8, w_idx, s_a_w, s_idx, pos, cos_sin, p, sc)

                run()
                torch.cuda.synchronize()
                outs = (y, p, sc) if ki is not None else (y,)
                variants[vkey] = (run, outs)
                vcfg[vkey] = {"qb": cq, "indexer": ci}
            ref = variants[arm_vid["1107"]][1]
            bits = {arm: sha(*variants[v][1]) for arm, v in arm_vid.items()}
            equal = {arm: all(torch.equal(a, c) for a, c in zip(variants[v][1], ref)) for arm, v in arm_vid.items()}
            samples, spun = bench_block(variants, plan, m)
            vnames = {v: f"v{i}" for i, v in enumerate(variants)}
            row = {"variants": {vnames[v]: {"configs": vcfg[v],
                                            **{st: summ(xs) for st, xs in samples[v].items()}}
                                for v in variants},
                   "arm_variant": {arm: vnames[v] for arm, v in arm_vid.items()},
                   "bits_sha256_16": bits, "bit_equal_to_1107_arm": equal, "spin_ms": spun,
                   "paired_minus_1107_us": {}}
            v0 = arm_vid["1107"]
            for arm, v in arm_vid.items():
                if arm == "1107":
                    continue
                row["paired_minus_1107_us"][arm] = (
                    "same-kernels" if v == v0 else
                    {st: summ([a - b for a, b in zip(samples[v][st], samples[v0][st])]) for st in samples[v]})
            row["raw_us"] = {vnames[v]: {st: [round(x, 3) for x in xs] for st, xs in samples[v].items()}
                             for v in variants}
            rows[str(m)] = row
            med = {arm: {st: row["variants"][row["arm_variant"][arm]][st]["median"] for st in samples[v0]}
                   for arm in ("1107", "image")}
            print(pname, m, json.dumps(med), "distinct", len(variants), "equal", all(equal.values()),
                  flush=True)
        out["pairs"][pname] = rows

    if MODE == "screen":
        # Indexer epilogue: rows per serial epilogue pass = 2 * threads / block_N.
        diag_cfgs = {
            "bN64_t128_s3 (1107 bench, bM16)": d(block_M=16, block_N=64, block_K=128, num_stages=3, threads=128),
            "bN128_t128_s3 (#1016 bench, bM16)": d(block_M=16, block_N=128, block_K=128, num_stages=3, threads=128),
            "bN128_t256_s2 (model's kernel)": d(block_M=16, block_N=128, block_K=128, num_stages=2, threads=256),
            "bN128_t256_s3": d(block_M=16, block_N=128, block_K=128, num_stages=3, threads=256),
            "bN64_t256_s3": d(block_M=16, block_N=64, block_K=128, num_stages=3, threads=256),
        }
        plan_i = plan_for([w_idx, s_idx])
        for m in (1, 6, 12, 16):
            pos = pos_all[:m]
            variants = {}
            for name, c in diag_cfgs.items():
                ki = rope_kernel(dict(head_dim=HEAD_DIM, rope_dim=ROPE_DIM, **c, padded_rows=True,
                                      tile_contiguous=True))
                p = torch.zeros((m, HEADS * (HEAD_DIM // 2)), device=DEV, dtype=torch.uint8)
                sc = torch.zeros((m, HEADS * (HEAD_DIM // 32)), device=DEV, dtype=torch.uint8)

                def run(ki=ki, p=p, sc=sc):
                    ki(a_fp8, w_idx, s_a_w, s_idx, pos, cos_sin, p, sc)

                run()
                variants[name] = (run, (p, sc))
            torch.cuda.synchronize()
            ref = next(iter(variants.values()))[1]
            eq = all(all(torch.equal(a, b) for a, b in zip(o, ref)) for _, o in variants.values())
            samples, _ = bench_block(variants, plan_i, m)
            out["diag"].setdefault("indexer_rope_bM16", {})[str(m)] = {
                "equal": eq, **{name: {st: summ(xs) for st, xs in samples[name].items()} for name in variants}}
            print("diag rope", m, json.dumps({nm: {st: summ(xs)["median"] for st, xs in samples[nm].items()}
                                              for nm in variants}), "equal", eq, flush=True)
        # TP8 Q-B plain kernel at block_M=16: block_N 64 vs 128.
        plan_q = plan_for([w_idx, s_idx])
        for m in (1, 6, 12, 16):
            variants = {}
            for bn in (64, 128):
                kq = qb_kernel(4096, d(block_M=16, block_N=bn, block_K=128, num_stages=3, threads=128))
                y = torch.zeros((m, 4096), device=DEV, dtype=torch.bfloat16)

                def run(kq=kq, y=y):
                    kq(a_fp8, w_idx, s_a_w, s_idx, y)

                run()
                variants[f"bN{bn}"] = (run, (y,))
            torch.cuda.synchronize()
            eq = torch.equal(variants["bN64"][1][0], variants["bN128"][1][0])
            samples, _ = bench_block(variants, plan_q, m)
            out["diag"].setdefault("qb4096_plain_bM16", {})[str(m)] = {
                "equal": eq, **{nm: {st: summ(xs) for st, xs in samples[nm].items()} for nm in variants}}
            print("diag qb4096", m, json.dumps({nm: {st: summ(xs)["median"] for st, xs in samples[nm].items()}
                                                for nm in variants}), "equal", eq, flush=True)

    if MODE == "mark":
        from tests.kernels.deepseek_v41_pdl_chains import ROWS as CHAIN_ROWS, LayerChain
        with set_current_vllm_config(VllmConfig()):
            for clabel, m in CHAIN_ROWS.items():
                chain = LayerChain(m, seed=100 + m)
                chain.poison()
                chain.run()
                torch.cuda.synchronize()
                dig = chain.digests()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        chain.run()
                torch.cuda.current_stream().wait_stream(s)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=s):
                    chain.run()
                graph.replay()
                torch.cuda.synchronize()
                spun = spin(graph.replay)
                warm, cold = [], []
                for _ in range(30):
                    warm.append(ev_time(lambda: [graph.replay() for _ in range(10)], div=10))
                    cold.append(ev_time(graph.replay, prep_cold))
                out["layer_chain"][f"{clabel}_M{m}"] = {
                    "digests_sha256": hashlib.sha256(json.dumps(dig, sort_keys=True).encode()).hexdigest()[:16],
                    "graph_warm_us": summ(warm), "graph_cold_us": summ(cold), "spin_ms": spun,
                    "raw_warm_us": [round(x, 2) for x in warm], "raw_cold_us": [round(x, 2) for x in cold]}
                print("layer_chain", clabel, m, out["layer_chain"][f"{clabel}_M{m}"]["graph_warm_us"]["median"],
                      out["layer_chain"][f"{clabel}_M{m}"]["graph_cold_us"]["median"], flush=True)
                del graph, chain
                torch.cuda.empty_cache()

    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"/out/h1016r1-{LABEL}.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", f"/out/h1016r1-{LABEL}.json")


if __name__ == "__main__":
    main()
