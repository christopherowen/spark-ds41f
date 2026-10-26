#!/usr/bin/env python3
"""A decode step's critical path from rank 0's profiled traces, and what PDL could recover.

usage: critical_path.py [TRACES] [--out critical_path.json]

TRACES is request-inventory-5's ports directory on dgx1
(results/private/determinism/request-inventory-5/ports): decode-trace/dgx1 holds one
torch-profiler trace a request type, in profile_decode.py's order.

A decode step is one target forward (the 40 layers, the head, sampling, the drafter's
inputs) plus the drafter forward that follows it, from one target vocab_embedding_kernel
to the next. The step's kernels come from three kinds of launch: the target's and the
drafter's CUDA graphs, small graphs between them, and eager kernels (step preparation,
sampling glue, metadata).

Inside a CUDA graph, the stream a kernel reports is the graph executor's internal stream,
not the capture stream. The executor spreads the target's single main chain over about 80
internal streams (it changes stream at every fork and join), so consecutive kernels of one
capture stream often sit on different internal streams. The capture streams are recovered
from the model's code instead:
- main: everything not below;
- shared: the shared expert (input cast, gate/up, SwiGLU, cast, down), forked after the FFN
  entry and joined by moe_combine (vLLM's aux stream; stream 28 when eager);
- prefetch: l2_prefetch_kernel, forked by every window and joined once, before the final
  mHC post.

The critical path is walked backwards from the step's last kernel. A kernel's binding
predecessor is the latest-ending of its possible dependencies: the previous kernel of its
capture stream, the shared expert's last kernel for moe_combine, the last prefetch for the
kernel after the join, and for the first kernel of a launch, the previous launch's last
kernel. Every gap on the path is the time between the predecessor's end and the kernel's
start, attributed to one mechanism:
- graph chain: consecutive kernels of one capture stream inside a graph (PDL's case:
  same-stream boundaries become programmatic edges under stream capture);
  "switch" when the executor put them on different internal streams;
- graph join: moe_combine waiting for the shared expert;
- prefetch join: the final mHC post waiting for the last L2 prefetch;
- launch boundary: the first kernel of a graph launch or an eager run after the previous
  launch's kernel; "host" when the host issued the launch after the GPU went idle;
- eager chain: consecutive eager kernels; "host" likewise.

The traces were taken with Python stacks on, which slows the host: eager and launch-boundary
gaps are upper bounds. Kernel durations and graph-internal gaps are not affected.
"""

import argparse
import bisect
import re
import gzip
import json
import os
import statistics
from collections import Counter, defaultdict

DECODE = (
    "greedy",
    "sampled",
    "top-p",
    "top-k",
    "top-p-top-k",
    "structured",
    "logprobs",
    "prompt-logprobs",
    "penalties",
    "stop",
    "thinking",
    "n2",
    "mixed-c8",
)
# (first step kept, end), request-inventory-5's as in the fusion survey: the first step of
# a window carries the request's prefill; sampled's step 24 holds a 26 ms host stall at the
# request's end; mixed-c8's first two steps and those from 19 run before or after all
# eight requests share the batch. Other traces name theirs with --window. Steps longer than
# STALL times the window's median are host stalls and are left out in any window.
WINDOWS = {"greedy": (1, None), "sampled": (1, 24), "mixed-c8": (2, 19)}
STALL = 1.5
PREFETCH = "l2_prefetch_kernel"
MARK = "vocab_embedding_kernel"
SHARED_SHAPE = (
    "per_token_cast",
    "gemm_mxfp8",
    "activation_swiglu_clamp",
    "per_token_cast",
    "gemm_mxfp8",
)
SMALL_US = 5.0
# a ready kernel that starts this much after its dependency waited for SMs, not for a launch
RESOURCE_US = 2.0
# Kernels the prototype chains (vLLM pdl branch; base names, without TileLang's
# "_kernel" suffix): ours begin with pdl.chain(), TileKernels' casts and norms take
# its PDL switch, and the decode MXFP8 GEMM tiles chain.
CHAINED = {
    "activation_swiglu_clamp",
    "cache_writer",
    "compress",
    "compress_gather_partial_state",
    "gemm_bf16_splitk",
    "gemm_fp8w_bf16",
    "gemm_fp8w_bf16_cast",
    "gemm_fp8w_bf16_partials",
    "gemm_mxfp8",
    "gemm_mxfp8_partials",
    "gemm_splitk_reduce",
    "gemm_splitk_reduce_cast",
    "gemm_splitk_reduce_two",
    "indexer_mxfp4_quantize",
    "indexer_score",
    "indexer_topk_merge",
    "indexer_topk_segment",
    "metadata_chunk",
    "metadata_decode",
    "metadata_pages",
    "metadata_requests",
    "metadata_tokens",
    "mhc_collapse",
    "mhc_collapse_column",
    "mhc_finalize",
    "mhc_post",
    "mhc_project_streams",
    "moe_combine",
    "moe_dispatch",
    "moe_grouped_fp8_fp4_gemm",
    "moe_swiglu_cast_kernel",
    "rope",
    "router_partials",
    "routing_masks",
    "sparse_mla",
    "per_token_cast_kernel",
    "norm_forward_and_per_token_cast_kernel",
}
# Weight GEMMs whose first weight tiles could load before the wait (a later step):
# the time one DRAM round trip and first tile take, an assumption, not a measurement.
WEIGHT_GEMMS = (
    "gemm_mxfp8",
    "gemm_mxfp8_partials",
    "gemm_fp8w_bf16_partials",
    "moe_grouped_fp8_fp4_gemm",
    "gemm_bf16_splitk",
    "gemm_bf16_partials",
    "vocab_packed_projection",
)
FIRST_TILE_US = 1.0


def load(path):
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    host = {}
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            host[e["args"].get("correlation")] = (
                e["name"],
                e["ts"],
                e["ts"] + e["dur"],
            )
    ks = []
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in (
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        ):
            continue
        a = e.get("args", {})
        api = host.get(a.get("correlation"), (None, None, None))
        graph = api[0] == "cudaGraphLaunch"
        ks.append(
            dict(
                ts=e["ts"],
                dur=e["dur"],
                end=e["ts"] + e["dur"],
                name=e["name"],
                grid=tuple(a.get("grid", ())),
                # capture order inside a graph (the node id's low word)
                node=a.get("graph node id", 0) & 0xFFFFFFFF,
                block=tuple(a.get("block", ())),
                stream=a.get("stream"),
                launch=("G", a.get("correlation")) if graph else ("E", a.get("stream")),
                api=api[0],
                api_ts=api[1],
                api_end=api[2],
                graph=a.get("graph id", 0),
                memcpy=e.get("cat") != "kernel",
            )
        )
    ks.sort(key=lambda k: k["ts"])
    return ks


def steps(ks, skip, last):
    starts = [k["ts"] for k in ks if k["name"] == MARK]
    out, i = [], 0

    def moe(a, b):
        return sum(1 for k in ks[a:b] if k["name"].startswith("moe_grouped"))

    idx = [bisect.bisect_left([k["ts"] for k in ks], t) for t in starts]
    while i + 2 < len(starts):
        if moe(idx[i], idx[i + 1]) == 80 and moe(idx[i + 1], idx[i + 2]) == 6:
            out.append((idx[i], idx[i + 2]))
            i += 2
        else:
            i += 1
    out = out[skip:last]
    spans = [ks[b]["ts"] - ks[a]["ts"] for a, b in out]
    median = statistics.median(spans)
    return [w for w, span in zip(out, spans) if span <= STALL * median]


def label(ks):
    """Capture stream of every kernel: main, shared or prefetch."""
    for k in ks:
        k["cap"] = "prefetch" if k["name"] == PREFETCH else "main"
    # Eager kernels run on the main stream (the embedding's) or, in an eager forward, the
    # shared expert's aux stream; eager copies are the main stream's.
    main = next(k["stream"] for k in ks if k["name"] == MARK)
    seqs = defaultdict(list)
    for k in ks:
        if k["cap"] != "prefetch":
            seqs[(k["launch"], k["stream"])].append(k)
    for seq in seqs.values():
        for i, k in enumerate(seq):
            if k["launch"][0] == "E" and k["stream"] != main and not k["memcpy"]:
                k["cap"] = "shared"
            if k["launch"][0] != "G" or not k["name"].startswith(SHARED_SHAPE[2]):
                continue
            window = seq[i - 2 : i + 3]
            assert len(window) == 5 and all(
                w["name"].startswith(s) for w, s in zip(window, SHARED_SHAPE)
            ), [w["name"] for w in window]
            for w in window:
                w["cap"] = "shared"


def critical_path(ks):
    """Edges (pred, kernel, kind) of the critical path through ks, last kernel first."""
    body = [k for k in ks if k["cap"] != "prefetch"]
    prefetch = [k for k in ks if k["cap"] == "prefetch"]
    main = next(k["stream"] for k in body if k["name"] == MARK)
    prev_cap, last, seg, first = {}, {}, {}, {}
    current = None
    for k in body:
        # one launch a segment: a graph replay, or a run of eager kernels between replays
        key = k["launch"] if k["launch"][0] == "G" else "E"
        if key != current:
            current = key
            first[id(k)] = True
        seg[id(k)] = key
        chain = (key, k["cap"]) if key != "E" else ("E", k["stream"], k["cap"])
        prev_cap[id(k)] = last.get(chain)
        last[chain] = k
    for k in prefetch:
        seg[id(k)] = k["launch"] if k["launch"][0] == "G" else "E"
        prev_cap[id(k)] = last.get((seg[id(k)], "prefetch"))
        last[(seg[id(k)], "prefetch")] = k
    shared = [k for k in body if k["cap"] == "shared"]
    ends = sorted(body, key=lambda x: x["end"])
    end_keys = [x["end"] for x in ends]

    def latest(pool, k, same_launch):
        """The latest-ending kernel of pool finished by k's start (same launch, or before it)."""
        best = None
        for x in pool:
            if (
                x["end"] <= k["ts"] + 0.05
                and (seg[id(x)] == seg[id(k)]) == same_launch
                and x is not k
            ):
                if best is None or x["end"] > best["end"]:
                    best = x
        return best

    edges = []
    k = ends[-1]
    lo = body[0]["ts"]
    while k is not None and k["ts"] >= lo:
        cands = []
        p = prev_cap[id(k)]
        if p is not None:
            cands.append((p, "chain"))
        if k["name"].startswith("moe_combine"):
            s = latest(shared, k, True)
            if s is not None:
                cands.append((s, "join"))
        if k["cap"] == "prefetch":
            # every window forks the side stream off the main stream when it is issued
            m = latest([x for x in body if x["cap"] == "main"], k, True)
            if m is not None:
                cands.append((m, "fork"))
        if k["cap"] == "shared" and (
            p is None
            or seg[id(p)] != seg[id(k)]
            or k["name"].startswith(SHARED_SHAPE[0])
            and p["name"].startswith(SHARED_SHAPE[4])
        ):
            # the branch's first kernel: forked from the main kernel that produced its input
            m = latest([x for x in body if x["cap"] == "main"], k, True)
            if m is not None:
                cands.append((m, "fork"))
        if k["name"].startswith(("mhc_post", "mhc_collapse")):
            pf = latest(prefetch, k, True)
            if pf is not None:
                cands.append((pf, "prefetch-join"))
        if k["memcpy"] and k["launch"][0] == "E" and k["stream"] != main:
            # a copy on a side stream waits, through an event, for the work before it
            before = [
                x
                for x in body
                if x["end"] <= k["ts"] + 0.05 and x["stream"] != k["stream"]
            ]
            if before:
                cands.append((max(before, key=lambda x: x["end"]), "launch"))
        if first.get(id(k)):
            i = bisect.bisect_right(end_keys, k["ts"] + 0.05)
            before = [
                x
                for x in ends[max(0, i - 64) : i]
                if seg[id(x)] != seg[id(k)] or seg[id(k)] == "E"
            ]
            if before:
                cands.append((max(before, key=lambda x: x["end"]), "launch"))
        if not cands:
            edges.append((None, k, "start"))
            break
        pred, kind = max(cands, key=lambda c: c[0]["end"])
        if k["ts"] - pred["end"] > RESOURCE_US:
            # ready, but every SM was held: it started when a kernel running meanwhile drained
            held = [
                x
                for x in body
                if x["ts"] < k["ts"]
                and pred["end"] < x["end"] <= k["end"]
                and seg[id(x)] == seg[id(k)]
                and x is not k
            ]
            if held:
                pred, kind = max(held, key=lambda x: x["end"]), "resource"
        edges.append((pred, k, kind))
        k = pred
    return edges


def classify(pred, k, kind):
    if kind == "start":
        return "start"
    if pred["memcpy"] or k["memcpy"]:
        base = "memcpy"
    elif kind == "chain" and k["launch"][0] == "G":
        base = "graph chain" + (" (switch)" if pred["stream"] != k["stream"] else "")
    elif kind == "chain":
        base = "eager chain"
    elif kind == "join":
        return "graph join (shared expert)"
    elif kind == "fork":
        return "graph fork (%s)" % (
            "shared expert" if k["cap"] == "shared" else "prefetch"
        )
    elif kind == "prefetch-join":
        return "prefetch join"
    elif kind == "resource":
        return "resource wait (%s behind %s)" % (k["cap"], family(pred["name"]))
    else:
        base = "launch boundary"
    if (
        base in ("eager chain", "launch boundary", "memcpy")
        and k["api_end"] is not None
    ):
        # the host issued this launch after the predecessor finished: the GPU waited on it
        issued = k["api_ts"] if k["api"] == "cudaGraphLaunch" else k["api_end"]
        if issued > pred["end"]:
            return base + " (host)"
    return base


def base(name):
    return re.sub(r"_kernel(_\d+)?$", "", name)


def family(name):
    n = name.replace("_kernel_kernel", "_kernel")
    for prefix, fam in (
        ("moe_grouped", "moe_grouped_gemm"),
        ("gemm_", "gemm"),
        ("oneshot_", "sparknet"),
        ("ncclDev", "nccl"),
        ("Memcpy", "memcpy"),
        ("Memset", "memcpy"),
        ("void at::", "torch"),
        ("mhc_", "mhc"),
        ("sparse_mla", "sparse_mla"),
        ("indexer", "indexer"),
        ("vocab_packed", "lm_head"),
    ):
        if n.startswith(prefix):
            return fam
    return "small"


def analyse(ks, windows):
    label(ks)
    per = defaultdict(float)
    count = Counter()
    gaps = defaultdict(list)
    pairs = defaultdict(lambda: [0, 0.0, []])
    span_total = kernel_total = 0.0
    idle_total = 0.0
    small_on_path = 0.0
    small_count = 0
    into_weight_gemm = defaultdict(float)
    on_path = defaultdict(float)
    chained = defaultdict(float)
    chained_count = Counter()
    unchained = defaultdict(float)
    prologue = 0.0
    n = 0
    for a, b in windows:
        step = ks[a:b]
        span = ks[b]["ts"] - ks[a]["ts"]
        edges = critical_path(step)
        n += 1
        span_total += span
        end = None
        for x in sorted(
            (x for x in step if x["cap"] != "prefetch"), key=lambda x: x["ts"]
        ):
            if end is not None and x["ts"] > end:
                idle_total += x["ts"] - end
            end = x["end"] if end is None else max(end, x["end"])
        for pred, k, kind in edges:
            own = k["end"] - (k["ts"] if pred is None else max(k["ts"], pred["end"]))
            kernel_total += own
            on_path[(k["cap"], family(k["name"]))] += own
            if k["dur"] < SMALL_US and not k["memcpy"]:
                small_on_path += own
                small_count += 1
            if pred is None:
                continue
            g = max(0.0, k["ts"] - pred["end"])
            c = classify(pred, k, kind)
            per[c] += g
            count[c] += 1
            gaps[c].append(g)
            key = (
                c,
                pred["name"].replace("_kernel_kernel", "_kernel")[:40],
                k["name"].replace("_kernel_kernel", "_kernel")[:40],
            )
            pairs[key][0] += 1
            pairs[key][1] += g
            pairs[key][2].append(g)
            if c.startswith("graph chain") and family(k["name"]) in (
                "gemm",
                "moe_grouped_gemm",
                "lm_head",
            ):
                into_weight_gemm[family(pred["name"])] += g
            if c.startswith(("graph chain", "eager chain")):
                where = "graph" if c.startswith("graph") else "eager"
                if base(k["name"]) in CHAINED:
                    chained[where] += g
                    chained_count[where] += 1
                    if where == "graph" and base(k["name"]).startswith(WEIGHT_GEMMS):
                        pred_own = pred["end"] - pred["ts"]
                        prologue += min(FIRST_TILE_US, pred_own)
                else:
                    unchained[(where, base(k["name"]))] += g
    step_us = span_total / n
    out = {
        "steps": n,
        "step_ms": round(step_us / 1000, 3),
        "critical_path_kernel_ms": round(kernel_total / n / 1000, 3),
        "gpu_idle_us": round(idle_total / n, 1),
        "path_kernel_us_by_stream_family": {
            "%s/%s" % key: round(v / n, 1)
            for key, v in sorted(on_path.items(), key=lambda x: -x[1])
        },
        "small_kernels_on_path": {
            "count": round(small_count / n, 1),
            "us": round(small_on_path / n, 1),
            "percent": round(100 * small_on_path / span_total, 2),
        },
        "gaps": {},
    }
    for c in sorted(per, key=lambda c: -per[c]):
        out["gaps"][c] = {
            "us": round(per[c] / n, 1),
            "percent": round(100 * per[c] / span_total, 2),
            "count": round(count[c] / n, 1),
            "median_us": round(statistics.median(gaps[c]), 2),
            "p90_us": round(sorted(gaps[c])[int(0.9 * (len(gaps[c]) - 1))], 2),
        }
    top = sorted(pairs.items(), key=lambda x: -x[1][1])[:40]
    out["top_pairs"] = [
        {
            "kind": c,
            "pred": p,
            "kernel": k,
            "count": round(v[0] / n, 1),
            "us": round(v[1] / n, 1),
            "median_us": round(statistics.median(v[2]), 2),
        }
        for (c, p, k), v in top
    ]
    out["chain_gaps_into_weight_gemms_by_pred_family_us"] = {
        k: round(v / n, 1)
        for k, v in sorted(into_weight_gemm.items(), key=lambda x: -x[1])
    }

    def pct(us):
        return round(100 * us / span_total, 2)

    out["pdl"] = {
        "chained_gap_us": {w: round(v / n, 1) for w, v in chained.items()},
        "chained_gap_percent": {w: pct(v) for w, v in chained.items()},
        "chained_boundaries": {w: round(v / n, 1) for w, v in chained_count.items()},
        "unchained_gap_us": {
            f"{w}/{b}": round(v / n, 1)
            for (w, b), v in sorted(unchained.items(), key=lambda x: -x[1])[:15]
        },
        "weight_prologue_model_us": round(prologue / n, 1),
        "weight_prologue_model_percent": pct(prologue),
    }
    return out


def layer_sequence(ks, windows, index):
    """The critical path of target layer ``index`` in the middle step: from the kernel
    after the previous layer's MoE all-reduce through this layer's."""
    label(ks)
    a, b = windows[len(windows) // 2]
    edges = list(reversed(critical_path(ks[a:b])))
    reduces = [
        i for i, (_, k, _) in enumerate(edges) if base(k["name"]) == "oneshot_allreduce"
    ]
    lo, hi = reduces[2 * index - 1] + 1, reduces[2 * index + 1] + 1
    kernels, path = [], 0.0
    for pred, k, kind in edges[lo:hi]:
        own = k["end"] - max(k["ts"], pred["end"])
        path += max(0.0, k["ts"] - pred["end"]) + own
        ctas = 1
        for g in k["grid"] or (1,):
            ctas *= g
        kernels.append(
            {
                "name": base(k["name"]),
                "ctas": ctas,
                "threads": (k["block"] or (128,))[0],
                "us": round(k["dur"], 1),
                "pdl": base(k["name"]) in CHAINED,
            }
        )
    return {
        "source": f"greedy step {len(windows) // 2}, target layer {index}",
        "path_us": round(path, 1),
        "kernels": kernels,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "traces",
        nargs="?",
        default="results/private/determinism/request-inventory-5/ports",
    )
    ap.add_argument(
        "--out", default=os.path.join(os.path.dirname(__file__), "critical_path.json")
    )
    ap.add_argument(
        "--layer-sequence", help="write one c1 greedy layer's critical path here"
    )
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument(
        "--scenarios",
        default=",".join(DECODE),
        help="the windows' request types in trace order (profile_decode.py's)",
    )
    ap.add_argument(
        "--window",
        action="append",
        default=[],
        help="NAME=FIRST:END, the steps of a window to keep (END may be empty)",
    )
    args = ap.parse_args()
    for item in args.window:
        name, _, span = item.partition("=")
        first, _, end = span.partition(":")
        WINDOWS[name] = (int(first), int(end) if end else None)
    files = sorted(os.listdir(os.path.join(args.traces, "decode-trace/dgx1")))
    names = args.scenarios.split(",")
    trace = dict(
        zip(names, (os.path.join(args.traces, "decode-trace/dgx1", f) for f in files))
    )
    out = {"source": args.traces, "decode": {}}
    for name, (skip, last) in ((n, w) for n, w in WINDOWS.items() if n in trace):
        ks = load(trace[name])
        out["decode"][name] = analyse(ks, steps(ks, skip, last))
        d = out["decode"][name]
        print(
            f"{name}: {d['steps']} steps of {d['step_ms']} ms; path kernels {d['critical_path_kernel_ms']} ms"
        )
        for c, v in d["gaps"].items():
            print(
                f"  {c:34} {v['us']:8.1f} us {v['percent']:5.2f}%  n={v['count']:6.1f}  median {v['median_us']}"
            )
        print(
            "  pdl:",
            json.dumps({k: v for k, v in d["pdl"].items() if k != "unchained_gap_us"}),
        )
        if name == "greedy" and args.layer_sequence:
            with open(args.layer_sequence, "w") as handle:
                json.dump(
                    layer_sequence(ks, steps(ks, skip, last), args.layer),
                    handle,
                    indent=1,
                )
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)


if __name__ == "__main__":
    main()
