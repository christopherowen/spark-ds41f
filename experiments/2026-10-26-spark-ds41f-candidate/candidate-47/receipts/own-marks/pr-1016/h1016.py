#!/usr/bin/env python3
"""Own-mark harness for PR #1016 (ours, not the PR's): the PR bench's chained
Q-B + indexer Q-B decode pair, timed eagerly and by CUDA-graph replay (warm, cold-L2,
and cold-L2 + NEXT-window prefetch), plus the 20-kernel PDL LayerChain graph with
the image's tile-contiguous set, with output hashes and chain digests for
cross-image bit-exactness.

Arms in one process, interleaved window by window:
  old1107  the PR bench's OLD_1107_CONFIGS, tile-contiguous weights
  image    the image's fp8_decode_config, tile-contiguous weights
On the 1107 image the two arms should coincide (A/A); the harness records whether
the image's configs equal OLD_1107_CONFIGS.
usage: h1016.py LABEL   (writes /out/h1016-LABEL.json)
"""

import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time

import torch

spec = importlib.util.spec_from_file_location("b1016", "/bench/bench_qb_tile_contiguous.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.models.deepseek_v4_1 import l2_prefetch as ds_l2pf  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import gemm as ds_gemm  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.gemm import (  # noqa: E402
    DECODE_ROWS,
    fp8_decode_config,
    mxfp8_gemm_decode,
    mxfp8_gemm_rope_mxfp4_decode,
    pack_scale_words,
    pack_weight_tiles,
)
from vllm.models.deepseek_v4_1.tilelang.linear import cast_mxfp8  # noqa: E402
from tests.kernels.deepseek_v41_pdl_chains import ROWS as CHAIN_ROWS, LayerChain  # noqa: E402

LABEL = sys.argv[1]
DEV = torch.device("cuda")
WINDOWS = 30
REPS = 20


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


def main():
    out = {"label": LABEL, "device": torch.cuda.get_device_name(0), "windows": WINDOWS,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "image_tile_contiguous_fp8": sorted(list(k) for k in ds_gemm.TILE_CONTIGUOUS_FP8),
           "image_cfg_vs_old1107": {}, "chained": {}, "layer_chain": {}}
    g = torch.Generator(device=DEV).manual_seed(20261109)
    ds_l2pf.warmup()
    flush_buf = torch.empty((48 * 1024 * 1024,), dtype=torch.uint8, device=DEV)
    x_bf16 = torch.randn((DECODE_ROWS, 1280), generator=g, device=DEV, dtype=torch.bfloat16)
    mx = cast_mxfp8(x_bf16)
    a_fp8 = mx.values
    s_a_w = mx.scales.contiguous().view(torch.uint32)
    for (n, k), per_bm in b.OLD_1107_CONFIGS.items():
        for bm, old in per_bm.items():
            img = fp8_decode_config(n, k, bm)
            out["image_cfg_vs_old1107"][f"{n}x{k}/bM{bm}"] = {"image": img, "old1107_equal": img == old}

    heads, head_dim, rope_dim = 32, 128, 64
    w_idx = (torch.randn((4096, 1280), generator=g, device=DEV, dtype=torch.bfloat16) * 0.1).to(torch.float8_e4m3fn)
    w_idx_tc = pack_weight_tiles(w_idx)
    s_idx_bw = pack_scale_words(torch.randint(120, 126, (4096 // 32, 40), generator=g, device=DEV, dtype=torch.uint8))
    cos_sin = torch.randn((4096, rope_dim), generator=g, device=DEV, dtype=torch.float32)
    pos_all = torch.arange(DECODE_ROWS, device=DEV, dtype=torch.int64)

    def prep_cold():
        flush_buf.add_(1)

    for label, n_qb in (("tp4_8192_plus_4096", 8192), ("tp3_12288_plus_4096", 12288)):
        w_q = (torch.randn((n_qb, 1280), generator=g, device=DEV, dtype=torch.bfloat16) * 0.1).to(torch.float8_e4m3fn)
        w_q_tc = pack_weight_tiles(w_q)
        s_q_bw = pack_scale_words(torch.randint(120, 126, (n_qb // 32, 40), generator=g, device=DEV, dtype=torch.uint8))
        plan_tc = b._make_prefetch_plan([w_q_tc, s_q_bw, w_idx_tc, s_idx_bw], DEV)

        def prep_next():
            flush_buf.add_(1)
            ds_l2pf.issue(plan_tc, m)
            ds_l2pf.join()

        rows = {}
        for m in b.BATCH_SIZES:
            bm = b._block_m_for(m)
            cfgs = {
                "old1107": (b.OLD_1107_CONFIGS[(n_qb, 1280)][bm], b.OLD_1107_CONFIGS[(4096, 1280)][bm]),
                "image": (fp8_decode_config(n_qb, 1280, bm), fp8_decode_config(4096, 1280, bm)),
            }
            pos = pos_all[:m]
            runs, outs = {}, {}
            for arm, (cq, ci) in cfgs.items():
                kq = mxfp8_gemm_decode(n_qb, 1280, **cq, padded_rows=True, tile_contiguous=True)
                ki = mxfp8_gemm_rope_mxfp4_decode(4096, 1280, head_dim=head_dim, rope_dim=rope_dim, **ci,
                                                  padded_rows=True, tile_contiguous=True)
                y = torch.empty((m, n_qb), device=DEV, dtype=torch.bfloat16)
                p = torch.empty((m, heads * (head_dim // 2)), device=DEV, dtype=torch.uint8)
                sc = torch.empty((m, heads * (head_dim // 32)), device=DEV, dtype=torch.uint8)

                def run(kq=kq, ki=ki, y=y, p=p, sc=sc):
                    kq(a_fp8, w_q_tc, s_a_w, s_q_bw, y)
                    ki(a_fp8, w_idx_tc, s_a_w, s_idx_bw, pos, cos_sin, p, sc)

                run()
                torch.cuda.synchronize()
                outs[arm] = (y, p, sc)
                runs[arm] = run
            bits = {arm: sha(*o) for arm, o in outs.items()}
            equal = all(torch.equal(a, c) for a, c in zip(outs["old1107"], outs["image"]))
            graphs = {}
            for arm, run in runs.items():
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
                graphs[arm] = (gw, g1)
            torch.cuda.synchronize()
            modes = ("eager_warm", "graph_warm", "eager_cold", "graph_cold", "eager_next", "graph_next")
            samples = {arm: {mo: [] for mo in modes} for arm in runs}
            for _ in range(WINDOWS):
                for arm, run in runs.items():
                    gw, g1 = graphs[arm]
                    sm = samples[arm]

                    def eager_reps(run=run):
                        for _ in range(REPS):
                            run()

                    sm["eager_warm"].append(ev_time(eager_reps, div=REPS))
                    sm["graph_warm"].append(ev_time(gw.replay, div=REPS))
                    sm["eager_cold"].append(ev_time(run, prep_cold))
                    sm["graph_cold"].append(ev_time(g1.replay, prep_cold))
                    sm["eager_next"].append(ev_time(run, prep_next))
                    sm["graph_next"].append(ev_time(g1.replay, prep_next))
            row = {arm: {mo: summ(v) for mo, v in d.items()} for arm, d in samples.items()}
            row["paired_image_minus_old1107_us"] = {
                mo: summ([a - c for a, c in zip(samples["image"][mo], samples["old1107"][mo])]) for mo in modes}
            row["bits_sha256_16"] = bits
            row["bit_equal_old1107_vs_image"] = equal
            row["configs"] = {arm: {"qb": c[0], "indexer_qb": c[1]} for arm, c in cfgs.items()}
            rows[str(m)] = row
            print(label, m, json.dumps({arm: {mo: row[arm][mo]["median"] for mo in modes} for arm in runs}), "equal", equal, flush=True)
            del graphs
        out["chained"][label] = rows

    with set_current_vllm_config(VllmConfig()):
        for clabel, m in CHAIN_ROWS.items():
            chain = LayerChain(m, seed=100 + m)
            chain.poison()
            chain.run()
            torch.cuda.synchronize()
            dig = chain.digests()
            graph = b._capture_chain(chain)
            graph.replay()
            torch.cuda.synchronize()
            warm, cold = [], []
            for _ in range(WINDOWS):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(10):
                    graph.replay()
                e.record()
                torch.cuda.synchronize()
                warm.append(s.elapsed_time(e) * 1e3 / 10)
                cold.append(ev_time(graph.replay, prep_cold))
            out["layer_chain"][f"{clabel}_M{m}"] = {
                "digests_sha256": hashlib.sha256(json.dumps(dig, sort_keys=True).encode()).hexdigest()[:16],
                "digests": dig, "graph_warm_us": summ(warm), "graph_cold_us": summ(cold)}
            print("layer_chain", clabel, m, out["layer_chain"][f"{clabel}_M{m}"]["graph_warm_us"]["median"],
                  out["layer_chain"][f"{clabel}_M{m}"]["graph_cold_us"]["median"], flush=True)
            del graph, chain
            torch.cuda.empty_cache()
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"/out/h1016-{LABEL}.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", f"/out/h1016-{LABEL}.json")


if __name__ == "__main__":
    main()
