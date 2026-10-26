"""A single-stream (c1) decode layer's kernels at one TP4 or TP3 rank's shapes, on one GPU.

usage: c1_layer.py off|on --tp 4|3 --out DIR [--iket DIR] [--rows 6] [--live N]
                   [--layers L] [--sets S] [--replays N] [--graph] [--evict]
                   [--routes ROUTES.json] [--start T] [--head] [--mla] [--ar]
                   [--ctas N] [--iters N] [--skip REGEX]

The layer is vLLM's PDL layer chain (tests/kernels/deepseek_v41_pdl_chains.py) at the
rank's shapes, with every routed expert set taken from DeepSeek's reference routes:

- the attention entry: mHC ``post_pre`` and its finalize, with its output's MXFP8 cast;
- the fused Q-A/KV GEMM (replicated, 1792 x 5120), the q latent's RMSNorm, Q-B (cast,
  GEMM; 16 heads a rank at TP4, 24 at TP3, whose 64 heads are padded to 72);
- with ``--mla``, sparse MLA decode at the rank's heads (a 128-entry window and 512
  indexed entries, the ``decode`` route), on its own inputs;
- WO-A's split-K partials and reduce-cast (2 groups at TP4, 3 at TP3), then WO-B;
- the FFN entry; the shared expert on its own stream (gate/up, SwiGLU, cast, down;
  576 columns at TP4, 768 at TP3) beside the routed experts (dispatch, FC1, SwiGLU and
  cast, FC2), and the combine, which adds the shared output before one rounding.

Routes: layer ``i`` of the chain takes the reference's layer ``i % 40`` routes for the
greedy tokens ``--start`` to ``--start + rows - 1`` (experiments/2026-10-18-json-count,
``routes_last_position``); rows from ``--live`` on are dead (expert -1, as dead and
padding rows are). Each distinct expert of a layer is one tile of the grouped GEMMs.

Weights come in ``--sets`` independent sets (dense projections, mHC ``fn`` and the 384
experts), layer ``i`` using set ``i % sets``, so a layer does not find the previous
layer's weights in L2 (24 MB on GB10; a set's experts are 1.7 GB at TP4).

``--head`` adds the target's packed 12-bit vocabulary head at the rank's shard (32,320
rows at TP4, 43,136 at TP3) on the last layer's output, once a pass. ``--ar`` puts a
copy where each of the layer's two all-reduces runs: a kernel off the PDL chain, so the
next entry launches only after it ends, as after the one-shot all-reduce. Rows past the
reference's eight tokens stand for other requests (``reference_routes``).

A pass is ``--layers`` layers back to back (and the head). It runs once to warm up,
then ``--replays`` times, eagerly or (``--graph``) as one CUDA graph; ``--evict``
reads 128 MiB before each timed pass, so the first layer's weights are cold too.
``on`` enables the IKET mode first (``iket.enable``; ``--ctas``, ``--iters``,
``--skip`` are its sampling and skip list).

Writes ``result.json`` to ``--out``: the shapes, the routes' distinct experts a layer,
microseconds a pass (median and p10/p90), and a SHA-256 of every buffer the passes write
(the same seed, routes and shapes give the same bits on every build that changes no
arithmetic).
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

DEVICE = "cuda"
HIDDEN = 5120
EXPERTS, TOPK, LAYERS = 384, 6, 40
QA_KV = (1792, HIDDEN)
Q_LATENT = 1280
O_LORA = 1024
LIMIT = 10.0
RANK = {
    4: dict(heads=16, groups=2, inter=576, vocab=32320),
    3: dict(heads=24, groups=3, inter=768, vocab=43136),
}


def default_routes():
    """The json-count experiment's reference routes, beside this experiment's folder."""
    return Path(__file__).resolve().parents[2] / "2026-10-18-json-count" / "runs" / "ref-woa1.json"


def reference_routes(path, start, rows, live):
    """[layer][row] -> six experts (or -1s for a dead row), for every reference layer.

    Rows past the reference's tokens stand for other requests: row r takes token
    ``start + r % tokens``'s routes with the expert ids relabelled by request
    ``r // tokens``'s fixed permutation (the identity for the first), so each
    request has the reference's routing structure over its own experts."""
    data = json.loads(Path(path).read_text())
    steps = data["steps"] if "steps" in data else data
    tokens = len(steps) - start
    if tokens < 1 or (rows > tokens and start != 0):
        raise SystemExit(f"{path}: {len(steps)} reference tokens, start {start}, {rows} rows")
    perms = [list(range(EXPERTS))] + [torch.randperm(EXPERTS, generator=torch.Generator().manual_seed(q)).tolist()
                                      for q in range(1, -(-rows // tokens))]
    out = []
    for layer in range(LAYERS):
        out.append([[perms[r // tokens][e] for e in steps[start + r % tokens]["routes_last_position"][str(layer)]]
                    if r < live else [-1] * TOPK for r in range(rows)])
    return out


class WeightSet:
    """One layer's weights at the rank's shapes: two mHC entries, the attention
    projections, the shared expert and the 384 routed experts."""

    def __init__(self, tp, rows, gen):
        from tests.kernels.deepseek_v41_pdl_chains import _block32, _Entry, _Projection
        from vllm.models.deepseek_v4_1.tilelang.activation import swiglu_clamp_kernel
        from vllm.models.deepseek_v4_1.tilelang.moe import MoeExpertsV41Plan
        from vllm.models.deepseek_v4_1.tilelang.norm import TileLangRMSNorm
        from vllm.models.deepseek_v4_1.tilelang.wo import TileLangWOProjection

        shape = RANK[tp]
        q_b, wo_a, wo_b = shape["heads"] * 512, shape["groups"] * O_LORA, shape["groups"] * O_LORA
        inter = shape["inter"]
        self.attention_entry = _Entry(rows, gen)
        self.ffn_entry = _Entry(rows, gen)
        self.qa_kv = _block32(*QA_KV, gen, mxfp8_input=True)
        self.q_norm = TileLangRMSNorm(Q_LATENT)
        self.q_norm.weight.data = (torch.randn(Q_LATENT, generator=gen, device=DEVICE) * 0.1 + 1.0).bfloat16()
        self.q_norm.declare_kernels()
        self.q_b = _block32(q_b, Q_LATENT, gen)
        self.wo = TileLangWOProjection(_Projection(wo_a, 4096, gen), _Projection(HIDDEN, wo_b, gen),
                                       groups=shape["groups"])
        self.wo.declare_kernels()
        self.gate_up = _block32(2 * inter, HIDDEN, gen, mxfp8_input=True)
        self.down = _block32(HIDDEN, inter, gen)
        self.swiglu = swiglu_clamp_kernel(inter, LIMIT)
        self.plan = MoeExpertsV41Plan(EXPERTS, TOPK, HIDDEN, inter, swiglu_limit=LIMIT, out_dtype=torch.float32)
        self.plan.declare_kernels()

        def codes(*s):  # packed E2M1 pairs
            return torch.empty(s, dtype=torch.uint8, device=DEVICE).random_(generator=gen)

        def exponents(*s):  # UE8M0 scales 2^-9 to 2^-6
            return torch.randint(118, 122, s, generator=gen, device=DEVICE, dtype=torch.uint8)

        w13, w13_sf = codes(EXPERTS, 2 * inter, HIDDEN // 2), exponents(EXPERTS, 2 * inter, HIDDEN // 32)
        w2, w2_sf = codes(EXPERTS, HIDDEN, inter // 2), exponents(EXPERTS, HIDDEN, inter // 32)
        self.experts = self.plan.prepare_weights(w13, w13_sf, w2, w2_sf)
        del w13, w13_sf, w2, w2_sf


class Layer:
    """One layer's buffers over a weight set, and its launches (``run``)."""

    def __init__(self, tp, rows, weights, ids, gen, aux, ar=False):
        from vllm.models.deepseek_v4_1.tilelang import linear
        from vllm.models.deepseek_v4_1.tilelang.moe import workspace_bytes

        self.tp, self.rows, self.w, self.aux, self.ar = tp, rows, weights, aux, ar
        shape = RANK[tp]
        inter = shape["inter"]
        self.ids = torch.tensor(ids, dtype=torch.int32, device=DEVICE).contiguous()
        self.route_weights = (torch.rand((rows, TOPK), generator=gen, device=DEVICE) + 0.1).contiguous()
        self.route_weights[self.ids < 0] = 0.0
        self.workspace = torch.empty(workspace_bytes(weights.plan.workspace_specs(rows, self.ids.dtype)),
                                     dtype=torch.uint8, device=DEVICE)

        def empty(cols, dtype=torch.bfloat16):
            return torch.empty((rows, cols), dtype=dtype, device=DEVICE)

        self.qa, self.qn, self.qb = empty(QA_KV[0]), empty(Q_LATENT), empty(shape["heads"] * 512)
        self.wo_out = empty(HIDDEN)
        # With ``ar``, a copy stands in for each all-reduce: a kernel off the PDL chain
        # between the sublayer's output and the next entry, as the one-shot is.
        self.wo_ar, self.out_ar = empty(HIDDEN), empty(HIDDEN)
        self.h, self.act, self.shared_out, self.out = empty(2 * inter), empty(inter), empty(HIDDEN), empty(HIDDEN)
        w = weights
        self.casts = {
            name: [torch.empty(s, dtype=d, device=DEVICE)
                   for s, d in linear._activation_specs(rows, k, layer.tilelang_decode_rows)]
            for name, k, layer in (("q_b", Q_LATENT, w.q_b), ("wo_b", shape["groups"] * O_LORA, w.wo.b),
                                   ("down", inter, w.down))
        }
        self.partials = [torch.empty(s, dtype=d, device=DEVICE) for s, d in w.wo.a.partial_specs(rows)]

    def _project(self, name, layer, x, out):
        from vllm.models.deepseek_v4_1.tilelang import linear
        from vllm.models.deepseek_v4_1.tilelang.pdl import per_token_cast

        xq, sf = self.casts[name]
        per_token_cast(x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True, out=(xq[: self.rows], sf[: self.rows]))
        linear._block32_gemm(layer, xq, sf, out, self.rows)

    def run(self, x, streams, post, comb, pre, mla=None):
        """The layer on ``x`` (the previous layer's output) and the residual streams;
        returns this layer's output and its streams and mixes for the next layer."""
        from vllm.models.deepseek_v4_1.tilelang import linear

        rows, w = self.rows, self.w
        a, f = w.attention_entry, w.ffn_entry
        a(x, streams, post, comb, pre)
        linear._block32_gemm(w.qa_kv, a.mx.values, a.mx.scales, self.qa, rows)
        w.q_norm(self.qa[:, :Q_LATENT], out=self.qn)
        self._project("q_b", w.q_b, self.qn, self.qb)
        if mla is not None:
            mla()
        xq, sf = self.casts["wo_b"]
        w.wo.a(self.qb, xq[:rows], sf[:rows], self.partials[0] if self.partials else None)
        linear._block32_gemm(w.wo.b, xq, sf, self.wo_out, rows)
        attention_out = self.wo_out
        if self.ar:
            self.wo_ar.copy_(self.wo_out)
            attention_out = self.wo_ar
        f(attention_out, a.streams, a.post, a.comb, a.pre)
        main = torch.cuda.current_stream()
        self.aux.wait_stream(main)
        with torch.cuda.stream(self.aux):
            linear._block32_gemm(w.gate_up, f.mx.values, f.mx.scales, self.h, rows)
            w.swiglu(self.h, self.act)
            self._project("down", w.down, self.act, self.shared_out)

        def shared():
            main.wait_stream(self.aux)
            return self.shared_out

        w.plan.run(f.mx, self.ids, self.route_weights, w.experts, self.out, self.workspace, shared=shared)
        if self.ar:
            self.out_ar.copy_(self.out)
            return self.out_ar, f.streams, f.post, f.comb, f.pre
        return self.out, f.streams, f.post, f.comb, f.pre

    def outputs(self):
        a, f, rows = self.w.attention_entry, self.w.ffn_entry, self.rows
        out = {"q-a/kv": self.qa, "q latent": self.qn, "q-b": self.qb, "wo-b": self.wo_out,
               "shared down": self.shared_out, "layer output": self.out}
        for name, entry in (("attention", a), ("ffn", f)):
            out[f"{name} y"] = entry.y
            out[f"{name} cast"] = entry.mx.values[:rows]
        return out


def mla_call(tp, rows):
    """Sparse MLA decode at the rank's heads, with the IKET experiment's hot-harness
    inputs (a full window and index list). That harness launches both decode routes;
    here only the one production picks for ``rows`` (``kernel_for``) runs."""
    import iket_hot
    from vllm.models.deepseek_v4_1.tilelang.sparse_mla import SparseMlaV41Plan

    iket_hot.HEADS = RANK[tp]["heads"]
    run = SparseMlaV41Plan.run
    chosen = {}

    def production_route(self, q, *args, kernel=None, scratch=None, **kwargs):
        route = chosen.setdefault("route", self.kernel_for(q.shape[0]))
        if route == "extend":
            # Past the decode routes (c16's rows): the hot harness's first call runs
            # it, without scratch, and its second does nothing.
            if "out" in chosen and chosen["out"] is not args[6]:
                return args[6]
            chosen["out"] = args[6]
            return run(self, q, *args, kernel="extend", **kwargs)
        if kernel != route:
            return args[6]  # out, left unwritten
        chosen["out"] = args[6]
        return run(self, q, *args, kernel=kernel, scratch=scratch, **kwargs)

    SparseMlaV41Plan.run = production_route
    call, _ = iket_hot.mla_case(rows)
    call()  # resolves the route and the output it writes
    return call, (chosen["out"],), chosen["route"]


def head_call(tp, rows, gen, x):
    """The packed 12-bit head at the rank's vocabulary shard, on ``x``."""
    from vllm.models.deepseek_v4_1.tilelang import prepare, vocab

    n, k, base = RANK[tp]["vocab"], HIDDEN, 120
    args = (n, k, base)
    kwargs = dict(block_M=vocab.row_tile(rows), out_dtype=vocab._OUT[torch.float32],
                  **vocab.tile_config(vocab.row_tile(rows), k))
    prepare.declare(lambda: [prepare.call(vocab.packed_vocab_projection, *args, **kwargs)])
    kernel = vocab.packed_vocab_projection(*args, **kwargs)
    packed = torch.empty((n, k + k // 2), dtype=torch.uint8, device=DEVICE).random_(generator=gen)
    # Codes 1-8 only (exponents base .. base + 7); no exceptions.
    packed[:, k:] = (torch.randint(1, 9, (n, k // 2), generator=gen, device=DEVICE, dtype=torch.uint8)
                     | (torch.randint(1, 9, (n, k // 2), generator=gen, device=DEVICE, dtype=torch.uint8) << 4))
    rows_off = torch.zeros((n + 1,), dtype=torch.int32, device=DEVICE)
    none = torch.zeros((1,), dtype=torch.int32, device=DEVICE)
    logits = torch.empty((rows, n), dtype=torch.float32, device=DEVICE)
    return (lambda: kernel(x, packed, none, none, rows_off, logits)), logits


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("off", "on"))
    ap.add_argument("--tp", type=int, choices=(4, 3), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iket")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--live", type=int, default=3)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--sets", type=int, default=4)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--evict", action="store_true")
    ap.add_argument("--routes", help="reference routes (default: the json-count experiment's)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--first-layer", type=int, default=0, help="reference layer of chain layer 0")
    ap.add_argument("--head", action="store_true")
    ap.add_argument("--mla", action="store_true")
    ap.add_argument("--ar", action="store_true", help="a copy off the PDL chain for each all-reduce")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctas", type=int, default=0, help="VLLM_DS41_IKET_CTAS")
    ap.add_argument("--iters", type=int, default=0, help="VLLM_DS41_IKET_ITERS")
    ap.add_argument("--skip", default="", help="VLLM_DS41_IKET_SKIP")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    if args.mode == "on":
        from vllm.models.deepseek_v4_1.tilelang import iket

        iket.enable(args.iket, args.ctas, args.iters, args.skip)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.models.deepseek_v4_1.tilelang import mhc, prepare

    routes = reference_routes(args.routes or default_routes(), args.start, args.rows, args.live)
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    started = time.monotonic()
    with set_current_vllm_config(VllmConfig()):
        sets = [WeightSet(args.tp, args.rows, gen) for _ in range(args.sets)]
        aux = torch.cuda.Stream()
        chain_layers = [(args.first_layer + i) % LAYERS for i in range(args.layers)]
        layers = [Layer(args.tp, args.rows, sets[i % args.sets], routes[ref], gen, aux, args.ar)
                  for i, ref in enumerate(chain_layers)]
        x = (torch.randn((args.rows, HIDDEN), generator=gen, device=DEVICE)).bfloat16()
        streams = torch.randn((args.rows, mhc.MULT * HIDDEN), generator=gen, device=DEVICE).bfloat16()
        post = torch.rand((args.rows, mhc.MULT), generator=gen, device=DEVICE) * 2
        comb = torch.softmax(torch.randn((args.rows, mhc.MULT, mhc.MULT), generator=gen, device=DEVICE), dim=-1)
        pre = torch.softmax(torch.randn((args.rows, mhc.MULT), generator=gen, device=DEVICE), dim=-1) + 1e-6
        mla, mla_outs, mla_route = mla_call(args.tp, args.rows) if args.mla else (None, (), None)
        head, logits = head_call(args.tp, args.rows, gen, layers[-1].out) if args.head else (None, None)
        prepare.freeze("c1_layer.py: every kernel was prepared before the first launch")
        print(f"built {args.sets} weight sets and {args.layers} layers in {time.monotonic() - started:.1f} s",
              flush=True)

        def one_pass():
            state = (x, streams, post, comb, pre)
            for layer in layers:
                state = layer.run(*state, mla=mla)
            if head is not None:
                head()

        evict = torch.ones(128 << 20, dtype=torch.uint8, device=DEVICE).view(torch.int32)
        one_pass()
        torch.cuda.synchronize()
        eager_bits = digest([t for layer in layers for t in layer.outputs().values()]
                            + list(mla_outs) + ([logits] if logits is not None else []))
        if args.graph:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                one_pass()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                one_pass()
            run = g.replay
        else:
            run = one_pass
        times = []
        for _ in range(args.replays):
            if args.evict:
                evict.max()
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            stop.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(stop) * 1000)
        bits = digest([t for layer in layers for t in layer.outputs().values()]
                      + list(mla_outs) + ([logits] if logits is not None else []))
        finite = all(bool(torch.isfinite(layer.out).all()) for layer in layers)

    distinct = [len({e for row in routes[ref] for e in row if e >= 0}) for ref in chain_layers]
    result = {
        "mode": args.mode, "tp": args.tp, "rank_shapes": RANK[args.tp], "rows": args.rows, "live": args.live,
        "start": args.start, "layers": args.layers, "reference_layers": chain_layers, "sets": args.sets,
        "graph": args.graph, "evict": args.evict, "head": args.head, "mla": mla_route, "ar": args.ar,
        "distinct_experts": distinct, "distinct_mean": round(statistics.mean(distinct), 2),
        "replays": args.replays,
        "us_per_pass": round(statistics.median(times), 2),
        "us_per_layer": round(statistics.median(times) / args.layers, 2),
        "p10_p90": [round(q, 2) for q in statistics.quantiles(times, n=10)[::8]] if len(times) > 2 else times,
        "finite": finite, "replay_equals_eager": bits == eager_bits, "digest": bits,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
