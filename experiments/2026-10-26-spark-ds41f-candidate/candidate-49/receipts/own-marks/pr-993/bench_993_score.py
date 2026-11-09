#!/usr/bin/env python3
"""own-marks harness (candidate-49, ours, not the PR's): PR #993 ships no bench
script, only its P5 receipts. This re-measures its claim on the integrated image:
indexer_score_v41 of the 1107 release (baseline, /tmp_host/indexer_baseline.py)
against the image's own indexer (candidate), warm and cold-L2, rows 1 and 6,
plus a bitwise comparison of the Scores both kernels write on identical inputs.
"""
import hashlib
import importlib.util
import json
import math
import sys
import time

import torch

from vllm.models.deepseek_v4_1.tilelang import indexer as cand_mod


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base_mod = load_module(
    "/tmp_host/indexer_baseline.py", "vllm.models.deepseek_v4_1.tilelang.indexer_baseline"
)
DEV = "cuda"
FLUSH = torch.empty(96 * 1024 * 1024, dtype=torch.uint8, device=DEV)  # > 24 MB L2


def inputs(rows, width, page_size, L, seed, candidates=None):
    g = torch.Generator(device=DEV).manual_seed(seed)
    nb = cand_mod.page_bytes(page_size)
    used = math.ceil(L / page_size)
    total = used + 4
    pool = torch.randint(0, 256, (total, nb), generator=g, device=DEV, dtype=torch.uint8)
    pool[:, page_size * 64 :] = torch.randint(
        118, 131, (total, page_size * 4), generator=g, device=DEV, dtype=torch.uint8
    )
    q = torch.randint(-128, 128, (rows, 32, 64), generator=g, device=DEV, dtype=torch.int8)
    qs = torch.randint(118, 131, (rows, 32 * 4), generator=g, device=DEV, dtype=torch.uint8)
    qs = qs.view(torch.uint32).reshape(rows, 32)
    w = torch.randn((rows, 32), generator=g, device=DEV).to(torch.bfloat16)
    if candidates is None:
        table_w = math.ceil(width / page_size)
    else:
        table_w = math.ceil(L / page_size)
    table = torch.zeros((rows, table_w), dtype=torch.int32, device=DEV)
    perm = torch.randperm(total, generator=g, device=DEV)[:used].to(torch.int32)
    table[:, :used] = perm
    lengths = torch.tensor(
        [max(1, L - (rows - 1 - r)) for r in range(rows)], dtype=torch.int32, device=DEV
    )
    active = torch.tensor([L], dtype=torch.int32, device=DEV)
    args = (
        q.view(torch.int8),
        qs.view(torch.uint32).reshape(rows, -1),
        w,
        pool.view(torch.int8)[:, :nb],
        pool.view(torch.uint32)[:, : nb // 4],
        table,
        lengths,
        active,
    )
    if candidates is not None:
        blocks = width // 8
        cl = []
        cands = torch.full((rows, blocks), -1, dtype=torch.int32, device=DEV)
        for r in range(rows):
            nblk = min(blocks, math.ceil(L / 8))
            ids = (
                torch.randperm(math.ceil(L / 8), generator=g, device=DEV)[:nblk]
                .sort()
                .values.to(torch.int32)
            )
            if candidates == "masked":
                ids[1::17] = -1
            cands[r, :nblk] = ids
            cl.append(min(width, nblk * 8) - (3 if candidates == "masked" else 0))
        args = args + (cands, torch.tensor(cl, dtype=torch.int32, device=DEV))
    keep = (pool, q, qs)
    return args, keep


def plan(mod, width, page_size, cand):
    return mod.IndexerV41Plan(width, page_size, candidates=cand)


def run_score(p, args, scores):
    p.score(*args, scores)


def _st(x):
    x = sorted(x)
    n = len(x)
    return {
        "median_us": round(x[n // 2], 3),
        "mean_us": round(sum(x) / n, 3),
        "p10_us": round(x[n // 10], 3),
        "p90_us": round(x[(9 * n) // 10], 3),
    }


def time_pair(fb, fc, iters, cold):
    for _ in range(5):
        fb()
        fc()
    torch.cuda.synchronize()
    tb, tc = [], []
    for i in range(iters):
        order = ((fb, tb), (fc, tc)) if i % 2 == 0 else ((fc, tc), (fb, tb))
        for fn, acc in order:
            if cold:
                FLUSH.fill_(i & 0xFF)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            acc.append(s.elapsed_time(e) * 1000.0)
    return _st(tb), _st(tc)


def bits(t):
    return t.view(torch.int16)


def main():
    out_path = sys.argv[1]
    res = {
        "device": torch.cuda.get_device_name(0),
        "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "exactness": [],
        "sanity": {},
        "timing": [],
    }
    # sanity: the two modules really build different kernels
    for cand in (False, True):
        width = 16384 if cand else 32768
        bs = base_mod.indexer_score_v41(width, 256, candidates=cand).get_kernel_source()
        cs = cand_mod.indexer_score_v41(width, 256, candidates=cand).get_kernel_source()
        tag = "candidates" if cand else "dense"
        res["sanity"][tag] = {
            "base_has_vec_loader": "__tl_load_k_shared_" in bs,
            "cand_has_vec_loader": "__tl_load_k_shared_" in cs,
            "sources_differ": bs != cs,
            "base_src_sha256": hashlib.sha256(bs.encode()).hexdigest()[:16],
            "cand_src_sha256": hashlib.sha256(cs.encode()).hexdigest()[:16],
        }
    print("sanity", json.dumps(res["sanity"]), flush=True)

    # bit-exactness: identical inputs, both outputs pre-filled with one sentinel,
    # compared bitwise over the whole [rows, width] buffer
    cases = []
    for ps, width in ((128, 524288), (256, 1048576)):
        for rows, L in ((1, 4096), (6, 9000), (6, 65523), (16, 131072), (6, 262144), (1, 262139)):
            cases.append(("dense", ps, width, rows, L, None))
    cases.append(("dense", 256, 1048576, 6, 1048576, None))
    for width in (4096, 16384):
        for rows, L, kind in ((6, 65536, "full"), (6, 65536, "masked"), (16, 262144, "masked"), (1, 9000, "full")):
            cases.append(("candidates", 256, width, rows, L, kind))
    for i, (mode, ps, width, rows, L, kind) in enumerate(cases):
        args, keep = inputs(rows, width, ps, L, 1000 + i, candidates=kind)
        pb = plan(base_mod, width, ps, mode == "candidates")
        pc = plan(cand_mod, width, ps, mode == "candidates")
        sb = torch.full((rows, width), 0.123, dtype=torch.bfloat16, device=DEV)
        sc = sb.clone()
        run_score(pb, args, sb)
        run_score(pc, args, sc)
        torch.cuda.synchronize()
        eq = bool(torch.equal(bits(sb), bits(sc)))
        nd = int((bits(sb) != bits(sc)).sum().item())
        written = int((bits(sc) != bits(torch.full_like(sc, 0.123))).sum().item())
        res["exactness"].append(
            {
                "mode": mode,
                "page_size": ps,
                "width": width,
                "rows": rows,
                "L": L,
                "candidates": kind,
                "bitwise_equal": eq,
                "diff_elems": nd,
                "written_elems": written,
                "scores_sha256": hashlib.sha256(bits(sc).cpu().numpy().tobytes()).hexdigest()[:16],
            }
        )
        print("exact", json.dumps(res["exactness"][-1]), flush=True)
        del args, keep, sb, sc
    res["exactness_all_equal"] = all(e["bitwise_equal"] for e in res["exactness"])

    # timing grid: base (1107) and cand (image) interleaved per iteration
    grid = []
    for rows in (6, 1):
        for L in (4096, 16384, 32768, 65536, 131072, 262144):
            grid.append(("dense", 128, 524288, rows, L, None))
        for L in (65536, 262144, 524288, 1048576):
            grid.append(("dense", 256, 1048576, rows, L, None))
    for width in (4096, 16384):
        grid.append(("candidates", 256, width, 6, 262144, "full"))
    for mode, ps, width, rows, L, kind in grid:
        args, keep = inputs(rows, width, ps, L, 7, candidates=kind)
        pb = plan(base_mod, width, ps, mode == "candidates")
        pc = plan(cand_mod, width, ps, mode == "candidates")
        sb = torch.empty((rows, width), dtype=torch.bfloat16, device=DEV)
        sc = torch.empty((rows, width), dtype=torch.bfloat16, device=DEV)

        def fb():
            run_score(pb, args, sb)

        def fc():
            run_score(pc, args, sc)

        iters = 60 if L <= 262144 else 30
        wb, wc = time_pair(fb, fc, iters, cold=False)
        cb, cc = time_pair(fb, fc, iters, cold=True)
        row = {
            "mode": mode,
            "page_size": ps,
            "width": width,
            "rows": rows,
            "L": L,
            "iters": iters,
            "warm": {"base": wb, "cand": wc, "speedup_median": round(wb["median_us"] / wc["median_us"], 3)},
            "cold": {"base": cb, "cand": cc, "speedup_median": round(cb["median_us"] / cc["median_us"], 3)},
        }
        res["timing"].append(row)
        print("time", json.dumps(row), flush=True)
        del args, keep, sb, sc
        torch.cuda.empty_cache()
    res["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print("SAVED", out_path, "exactness_all_equal", res["exactness_all_equal"], flush=True)


if __name__ == "__main__":
    main()
