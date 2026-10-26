"""How far the routed experts' chain is from an ideal read of the same bytes, at c1.

usage: stream_bound.py --tp 4|3 --out DIR [--rows 6] [--live N] [--layers L] [--sets S]
                       [--replays N] [--routes ROUTES.json] [--start T]

Per layer, with the reference routes (as c1_layer.py takes them), four CUDA graphs of
``--layers`` layers are timed:

- ``chain``: the routed experts as the plan runs them (dispatch, FC1, the fused SwiGLU
  and cast, FC2, combine), on MXFP8 rows; no shared expert, no attention;
- ``read_one``: one launch a layer that reads exactly the layer's live experts' FC1
  and FC2 bytes (E2M1 values, scale codes, row bases), from a buffer laid out one row an
  expert;
- ``read_two``: two launches a layer, the FC1 bytes and then the FC2 bytes;
- ``read_all``: one launch for every layer's bytes.

The read kernel is a stream: 48 CTAs split the experts' rows into equal contiguous runs
of 32 KB chunks; each thread issues eight 16-byte loads a chunk and folds them into a
register, which it stores at the end, so no load is dropped. It begins with
``pdl.chain()``, as the chain's kernels do.

So, per layer: ``chain - read_one`` bounds what any restructuring of the routed experts
saves at these routes (one continuous FC1-FC2 stream, persistent GEMMs, fused SwiGLU);
``read_two - read_one`` is the price of one launch boundary inside an ideal stream; and
``read_one - read_all / layers`` the price of the boundary between layers. Weight sets
rotate (``--sets``), so no layer reads bytes the previous one left in L2.

Writes ``result.json`` to ``--out``: microseconds a layer for each graph (median of
``--replays``), the bytes, the distinct experts and the bounds.
"""

import argparse
import json
import statistics
from pathlib import Path

import tilelang
import tilelang.language as T
import torch

DEVICE = "cuda"
HIDDEN, EXPERTS, TOPK, LAYERS = 5120, 384, 6, 40
INTER = {4: 576, 3: 768}
CTAS, THREADS, UNROLL, VEC = 48, 256, 8, 4
CHUNK_WORDS = THREADS * UNROLL * VEC  # 32 KB


def gemm_bytes(n, k):
    """One expert's bytes in a grouped GEMM: E2M1 values, a 2-byte scale code per row
    and 128-wide K block, a 1-byte base exponent per row."""
    return n * k // 2 + n * 2 * -(-k // 128) + n


def padded_words(nbytes):
    words = -(-nbytes // 4)
    return -(-words // CHUNK_WORDS) * CHUNK_WORDS


@tilelang.jit
def expert_stream(words: int, ctas: int = CTAS, threads: int = THREADS):
    """Read rows ``ids`` of ``W`` (``words`` 32-bit words each, a multiple of a chunk) as
    one stream over ``ctas`` CTAs; each thread stores the XOR of what it read."""
    from vllm.models.deepseek_v4_1.tilelang import pdl

    assert words % CHUNK_WORDS == 0
    per_row = words // CHUNK_WORDS
    rows_all, n = T.dynamic("rows_all, n")

    @T.prim_func
    def expert_stream_read(
        W: T.Tensor((rows_all, words), T.int32),
        ids: T.Tensor((n,), T.int32),
        out: T.Tensor((ctas * threads,), T.int32),
    ):
        with T.Kernel(ctas, threads=threads) as c:
            pdl.chain()
            tx = T.get_thread_binding()
            acc = T.alloc_local((VEC,), T.int32)
            vals = T.alloc_local((UNROLL, VEC), T.int32)
            for v in T.vectorized(VEC):
                acc[v] = 0
            total = n * per_row
            per = T.ceildiv(total, ctas)
            for i in T.serial(per):
                g = c * per + i
                if g < total:
                    row = ids[g // per_row]
                    base = (g % per_row) * CHUNK_WORDS
                    for u in T.unroll(UNROLL):
                        for v in T.vectorized(VEC):
                            vals[u, v] = W[row, base + (u * threads + tx) * VEC + v]
                    for u in T.unroll(UNROLL):
                        for v in T.vectorized(VEC):
                            acc[v] = acc[v] ^ vals[u, v]
            out[c * threads + tx] = acc[0] ^ acc[1] ^ acc[2] ^ acc[3]

    return expert_stream_read


def graph(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def timed(g, replays):
    out = []
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    for _ in range(replays):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        stop.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(stop) * 1000)
    return statistics.median(out)


def main():
    import c1_layer
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.models.deepseek_v4_1.tilelang import prepare
    from vllm.models.deepseek_v4_1.tilelang.linear import cast_mxfp8
    from vllm.models.deepseek_v4_1.tilelang.moe import MoeExpertsV41Plan, workspace_bytes

    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, choices=(4, 3), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--live", type=int, default=3)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--sets", type=int, default=4)
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--routes")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    inter = INTER[args.tp]
    routes = c1_layer.reference_routes(args.routes or c1_layer.default_routes(), args.start, args.rows, args.live)
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    fc1, fc2 = gemm_bytes(2 * inter, HIDDEN), gemm_bytes(HIDDEN, inter)
    w_one, w_fc1, w_fc2 = padded_words(fc1 + fc2), padded_words(fc1), padded_words(fc2)

    with set_current_vllm_config(VllmConfig()):
        plans, experts = [], []
        for _ in range(args.sets):
            plan = MoeExpertsV41Plan(EXPERTS, TOPK, HIDDEN, inter, swiglu_limit=10.0)
            plan.declare_kernels()

            def codes(*s):
                return torch.empty(s, dtype=torch.uint8, device=DEVICE).random_(generator=gen)

            def exponents(*s):
                return torch.randint(118, 122, s, generator=gen, device=DEVICE, dtype=torch.uint8)

            experts.append(plan.prepare_weights(codes(EXPERTS, 2 * inter, HIDDEN // 2),
                                                exponents(EXPERTS, 2 * inter, HIDDEN // 32),
                                                codes(EXPERTS, HIDDEN, inter // 2),
                                                exponents(EXPERTS, HIDDEN, inter // 32)))
            plans.append(plan)
        read = {words: expert_stream(words) for words in {w_one, w_fc1, w_fc2}}
        prepare.freeze("stream_bound.py: every kernel was prepared before the first launch")

        # The ideal reads' buffers: a row an expert, one set of rows per weight set.
        def rows(words):
            return [torch.empty((EXPERTS, words), dtype=torch.int32, device=DEVICE).random_(generator=gen)
                    for _ in range(args.sets)]

        buf_one, buf_fc1, buf_fc2 = rows(w_one), rows(w_fc1), rows(w_fc2)
        sink = torch.empty((CTAS * THREADS,), dtype=torch.int32, device=DEVICE)
        refs = [i % LAYERS for i in range(args.layers)]
        live = [sorted({e for row in routes[r] for e in row if e >= 0}) for r in refs]
        ids = [torch.tensor(e, dtype=torch.int32, device=DEVICE) for e in live]
        # read_all: every layer's rows in one buffer, one launch.
        flat = torch.cat([buf_one[i % args.sets][ids[i].long()] for i in range(args.layers)])
        all_ids = torch.arange(flat.shape[0], dtype=torch.int32, device=DEVICE)

        layers = []
        for i, r in enumerate(refs):
            topk = torch.tensor(routes[r], dtype=torch.int32, device=DEVICE).contiguous()
            weights = (torch.rand((args.rows, TOPK), generator=gen, device=DEVICE) + 0.1).contiguous()
            weights[topk < 0] = 0.0
            x = cast_mxfp8((torch.randn((args.rows, HIDDEN), generator=gen, device=DEVICE) * 0.5).bfloat16())
            y = torch.empty((args.rows, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
            plan = plans[i % args.sets]
            workspace = torch.empty(workspace_bytes(plan.workspace_specs(args.rows, torch.int32)),
                                    dtype=torch.uint8, device=DEVICE)
            layers.append((plan, x, topk, weights, experts[i % args.sets], y, workspace))

        def chain():
            for plan, x, topk, weights, ex, y, workspace in layers:
                plan.run(x, topk, weights, ex, y, workspace)

        def read_one():
            for i in range(args.layers):
                read[w_one](buf_one[i % args.sets], ids[i], sink)

        def read_two():
            for i in range(args.layers):
                read[w_fc1](buf_fc1[i % args.sets], ids[i], sink)
                read[w_fc2](buf_fc2[i % args.sets], ids[i], sink)

        def read_all():
            read[w_one](flat, all_ids, sink)

        times = {name: timed(graph(fn), args.replays) / args.layers
                 for name, fn in (("chain", chain), ("read_one", read_one), ("read_two", read_two),
                                  ("read_all", read_all))}

    distinct = [len(e) for e in live]
    mean_e = statistics.mean(distinct)
    nbytes = mean_e * (fc1 + fc2)
    result = {
        "tp": args.tp, "rows": args.rows, "live": args.live, "layers": args.layers, "reference_layers": refs,
        "distinct_experts": distinct, "distinct_mean": round(mean_e, 2),
        "bytes_per_layer": round(nbytes), "read_bytes_per_layer": round(mean_e * 4 * w_one),
        "us_per_layer": {k: round(v, 2) for k, v in times.items()},
        "gbs": {k: round(nbytes / v / 1e3, 1) for k, v in times.items()},
        "bound_us_per_layer": {
            "chain_minus_read_one": round(times["chain"] - times["read_one"], 2),
            "read_two_minus_read_one": round(times["read_two"] - times["read_one"], 2),
            "read_one_minus_read_all": round(times["read_one"] - times["read_all"], 2),
        },
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
