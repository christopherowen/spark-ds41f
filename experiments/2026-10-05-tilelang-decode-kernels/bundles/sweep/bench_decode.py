"""Decode kernels of the TileLang family at TP4 serving shapes: projections and sparse MLA.

usage: bench_decode.py [gemm] [mla]

GEMM: every block-32 FP8 decode projection of the TP4 decode step, swept over
tile width, pipeline stages and K shards. Each configuration is timed at 6,
12, 24 and 48 rows inside one CUDA graph, with the weights cold in L2 (calls
cycle distinct weight copies, as serving reads each layer's own weights) and
warm (one copy). A split configuration must give the same bits from its
decode route (split-K partials and the ordered reduce) and from the prefill
kernel folding the same shards, and match a dequantized FP32 reference.

MLA: the sparse MLA plan at TP4 (16 heads, a 128-entry window and 512
indexed entries) from the image's module and from the candidate module with
each of its options. Options that keep the arithmetic must reproduce the
image module's bits.

Times are microseconds per call. Prints a JSON summary on the last line.
"""
import concurrent.futures
import importlib.util
import itertools
import json
import sys

import torch

DEV = torch.device("cuda")
ROWS = (6, 12, 24, 48)
SMS = torch.cuda.get_device_properties(DEV).multi_processor_count
GEN = torch.Generator(device="cuda").manual_seed(0)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timed(calls, rounds=20):
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


# ---------------------------------------------------------------- projections

GEMM_SHAPES = {  # (N, K) per rank at TP4, calls per decode step
    "q_b": ((8192, 1280), 39),
    "indexer_q_b": ((4096, 1280), 8),
    "q_a_kv": ((1792, 5120), 39),
    "draft_main": ((6400, 5120), 2),
}


def fp8(shape):
    return (torch.randn(shape, device=DEV, generator=GEN) * 0.5).to(torch.float8_e4m3fn)


def sf_words(rows, k, g):
    exps = torch.randint(124, 131, (rows, k // 32), device=DEV, generator=GEN, dtype=torch.uint8)
    return g.pack_scale_words(exps)


def dequant(x, words):
    exps = words.contiguous().view(torch.uint8)[:, : x.shape[1] // 32].float() - 127
    return x.float() * torch.exp2(exps).repeat_interleave(32, dim=1)


SMEM_LIMIT = 96 * 1024  # GB10 allows 99 KiB per block; keep room for scales and barriers


def gemm_configs(g, k):
    k_blocks = k // 128  # the prefill kernel's K block fixes the shard boundaries
    for block_N, stages, shards in itertools.product((64, 128), (2, 3, 4), range(1, 11)):
        smem = stages * (g.DECODE_ROWS + block_N) * 128
        if k_blocks % shards == 0 and smem <= SMEM_LIMIT - 4096:
            yield dict(block_N=block_N, block_K=128, num_stages=stages, threads=128, shards=shards)


def compile_route(g, n, k, cfg):
    tile = dict(block_M=g.DECODE_ROWS, block_N=cfg["block_N"], block_K=cfg["block_K"],
                num_stages=cfg["num_stages"], threads=cfg["threads"])
    s = cfg["shards"]
    if s > 1:
        return g.mxfp8_gemm_partials(n, k, s, **tile), g.splitk_reduce(n, s)
    return g.mxfp8_gemm(n, k, **tile, padded_rows=True), None


def bench_gemm(g_base, g):
    results, failures = {}, []
    for name, ((n, k), calls_per_step) in GEMM_SHAPES.items():
        copies = max(2, -(-160 * 2**20 // (n * k)))  # >= 160 MB of distinct weights
        weights = [fp8((n, k)) for _ in range(copies)]
        wsf = [sf_words(n, k, g) for _ in range(copies)]
        a = fp8((g.DECODE_ROWS, k))
        asf = sf_words(g.DECODE_ROWS, k, g)
        configs = list(gemm_configs(g, k))
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            routes = list(pool.map(lambda c: compile_route(g, n, k, c), configs))
        base_tile = dict(block_M=g.DECODE_ROWS, **{key: v for key, v in g_base.default_config(g.DECODE_ROWS, k).items() if key != "block_M"})
        base = g_base.mxfp8_gemm(n, k, **base_tile, padded_rows=True)
        rows_out = {}
        for rows in ROWS:
            out = torch.empty(rows, n, dtype=torch.bfloat16, device=DEV)

            def run_base(i):
                return lambda: base(a, weights[i], asf, wsf[i], out)

            entry = {"base": {"cold": timed([run_base(i % copies) for i in range(2 * copies)]),
                              "warm": timed([run_base(0)] * 16)}}
            for cfg, (kernel, reduce) in zip(configs, routes):
                s = cfg["shards"]
                part = torch.empty(s, rows, n, dtype=torch.float32, device=DEV) if s > 1 else None

                def run(i, kernel=kernel, reduce=reduce, part=part):
                    if reduce is None:
                        return lambda: kernel(a, weights[i], asf, wsf[i], out)
                    return lambda: (kernel(a, weights[i], asf, wsf[i], part), reduce(part, out))

                key = "n{block_N}-st{num_stages}-s{shards}".format(**cfg)
                try:
                    entry[key] = {"cold": timed([run(i % copies) for i in range(2 * copies)]),
                                  "warm": timed([run(0)] * 16)}
                except Exception as error:  # a configuration that cannot launch is reported, not fatal
                    print(f"  {name} {key} rows={rows}: {type(error).__name__}: {str(error)[:120]}")
                    torch.cuda.synchronize()
            rows_out[rows] = entry
        # Correctness of every split, at 12 rows: decode route == prefill kernel, and close to FP32.
        rows = 12
        ref = (dequant(a[:rows], asf[:rows]) @ dequant(weights[0], wsf[0]).T)
        big = 80  # prefill rows: the first 12 rows repeat the decode rows
        a_big = torch.cat([a[:rows], fp8((big - rows, k))])
        asf_big = torch.cat([asf[:rows], sf_words(big - rows, k, g)])
        for s in sorted({c["shards"] for c in configs}):
            cfg = next(c for c in configs if c["shards"] == s)
            kernel, reduce = routes[configs.index(cfg)]
            out = torch.empty(rows, n, dtype=torch.bfloat16, device=DEV)
            if reduce is None:
                kernel(a, weights[0], asf, wsf[0], out)
            else:
                part = torch.empty(s, rows, n, dtype=torch.float32, device=DEV)
                kernel(a, weights[0], asf, wsf[0], part)
                reduce(part, out)
            prefill = g.mxfp8_gemm(n, k, **g.default_config(g.DECODE_ROWS + 1, k), shards=s)
            out_big = torch.empty(big, n, dtype=torch.bfloat16, device=DEV)
            prefill(a_big, weights[0], asf_big, wsf[0], out_big)
            same = torch.equal(out.view(torch.int16), out_big[:rows].view(torch.int16))
            err = ((out.float() - ref).norm() / ref.norm()).item()
            ok = same and err < 1e-2
            print(f"  {name} shards={s}: decode == prefill {same}, rel err {err:.2e} {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"{name} shards={s}")
        results[name] = {"shape": [n, k], "calls_per_step": calls_per_step, "rows": rows_out}
        for rows in (6, 48):
            entry = rows_out[rows]
            ranked = sorted((k_ for k_ in entry if "warm" in entry[k_]), key=lambda k_: entry[k_]["warm"])
            print(f"{name} {rows} rows warm: base {entry['base']['warm']:.1f} us; best " + ", ".join(
                f"{k_} {entry[k_]['warm']:.1f}" for k_ in ranked[:5]))
        best = {}
        for rows, entry in rows_out.items():
            key = min((k_ for k_ in entry if k_ != "base" and "cold" in entry[k_]), key=lambda k_: entry[k_]["cold"])
            best[rows] = (key, entry[key]["cold"], entry["base"]["cold"])
        print(f"{name} {n}x{k}: " + "; ".join(f"{r} rows base {b:.1f} -> {key} {v:.1f} us" for r, (key, v, b) in best.items()))
    return results, failures


# ---------------------------------------------------------------- sparse MLA

MLA = dict(heads=16, swa_width=128, idx_width=512, idx_page=64, swa_page=256, max_rows=96)
MLA_VARIANTS = {  # name: (module, plan options, keeps the image module's bits)
    "image": ("base", {}, True),
    "scalar-1buf": ("new", dict(stage_buffers=1, dequant_vec=1), True),
    "scalar-2buf": ("new", dict(stage_buffers=2, dequant_vec=1), True),
    "vec8-1buf": ("new", dict(stage_buffers=1, dequant_vec=8), True),
    "vec8-2buf": ("new", dict(stage_buffers=2, dequant_vec=8), True),
    "vec8-2buf-seg2": ("new", dict(stage_buffers=2, dequant_vec=8, segment_chunks=2), False),
}


def mla_inputs(rows, sets=8):
    m = MLA
    swa_pages, idx_pages = 64, 512
    out = []
    for _ in range(sets):
        swa = torch.empty(swa_pages, m["swa_page"] * 528, dtype=torch.uint8, device=DEV)
        swa_view = swa.view(swa_pages * m["swa_page"], 528)
        swa_view[:, :512] = fp8((swa_view.shape[0], 512)).view(torch.uint8)
        swa_view[:, 512:] = torch.randint(122, 128, (swa_view.shape[0], 16), device=DEV, generator=GEN, dtype=torch.uint8)
        idx = torch.empty(idx_pages, m["idx_page"] * 288, dtype=torch.uint8, device=DEV)
        idx_view = idx.view(idx_pages * m["idx_page"], 288)
        idx_view[:, :256] = torch.randint(0, 256, (idx_view.shape[0], 256), device=DEV, generator=GEN, dtype=torch.uint8)
        idx_view[:, 256:] = (torch.rand(idx_view.shape[0], 32, device=DEV, generator=GEN) * 0.1 + 0.01).to(torch.float8_e4m3fn).view(torch.uint8)
        slots = swa_pages * m["swa_page"]
        out.append(dict(
            q=(torch.randn(rows, m["heads"], 512, device=DEV, generator=GEN) * 0.1).bfloat16(),
            swa_cache=swa,
            swa_indices=torch.randint(0, slots, (rows, m["swa_width"]), device=DEV, generator=GEN, dtype=torch.int32),
            swa_lengths=torch.full((rows,), m["swa_width"], dtype=torch.int32, device=DEV),
            sinks=torch.randn(m["heads"], device=DEV, generator=GEN),
            idx_cache=idx,
            idx_indices=torch.randint(0, idx_pages * m["idx_page"], (rows, m["idx_width"]), device=DEV, generator=GEN, dtype=torch.int32),
            idx_lengths=torch.full((rows,), m["idx_width"], dtype=torch.int32, device=DEV),
            idx_page_table=torch.randperm(idx_pages, device=DEV, generator=GEN).to(torch.int32).repeat(rows, 1),
        ))
    return out


def bench_mla(mla_base, mla_new):
    m = MLA
    modules = {"base": mla_base, "new": mla_new}
    plans = {}
    for name, (module, opts, _) in MLA_VARIANTS.items():
        plans[name] = modules[module].SparseMlaV41Plan(
            m["heads"], m["swa_width"], m["idx_width"], m["idx_page"],
            swa_page_size=m["swa_page"], max_rows=m["max_rows"], **opts)
    results, failures = {}, []
    for rows in ROWS:
        sets = mla_inputs(rows)
        entry = {}
        outs = {}
        for name, plan in plans.items():
            kernel = plan.kernel_for(rows)
            scratch = [torch.empty(s, dtype=d, device=DEV) for s, d in plan.scratch_specs(rows)]
            out = torch.empty(rows, m["heads"], 512, dtype=torch.bfloat16, device=DEV)

            def run(inp, plan=plan, scratch=scratch, out=out):
                return lambda: plan.run(
                    inp["q"], inp["swa_cache"], inp["swa_indices"], inp["swa_lengths"], inp["sinks"], out,
                    idx_cache=inp["idx_cache"], idx_indices=inp["idx_indices"],
                    idx_lengths=inp["idx_lengths"], idx_page_table=inp["idx_page_table"], scratch=scratch)

            run(sets[0])()
            torch.cuda.synchronize()
            outs[name] = out.clone()
            entry[name] = {"kernel": kernel, "us": timed([run(s) for s in sets * 2])}
        ref = outs["image"]
        for name, (_, _, same_bits) in MLA_VARIANTS.items():
            if name == "image":
                continue
            equal = torch.equal(outs[name].view(torch.int16), ref.view(torch.int16))
            close = ((outs[name].float() - ref.float()).norm() / ref.float().norm()).item()
            entry[name]["bit_equal"] = equal
            entry[name]["rel_diff"] = close
            if same_bits and not equal:
                failures.append(f"mla {name} rows={rows}")
        results[rows] = entry
        print(f"mla rows {rows}: " + "; ".join(
            f"{n} {e['us']:.1f} us ({e['kernel']}{'' if e.get('bit_equal', True) else ', bits differ %.1e' % e['rel_diff']})"
            for n, e in entry.items()))
    return results, failures


def main():
    parts = set(sys.argv[1:]) or {"gemm", "mla"}
    summary, failures = {}, []
    if "gemm" in parts:
        g_base, g = load("gemm_base", "/b/gemm_base.py"), load("gemm_new", "/b/gemm.py")
        summary["gemm"], f = bench_gemm(g_base, g)
        failures += f
    if "mla" in parts:
        summary["mla"], f = bench_mla(load("mla_base", "/b/sparse_mla_base.py"), load("mla_new", "/b/sparse_mla.py"))
        failures += f
    summary["failures"] = failures
    print(json.dumps(summary))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
