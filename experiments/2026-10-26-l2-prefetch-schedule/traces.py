"""Decode traces as kernel lists, for this experiment's scripts.

``load``, ``steps`` and ``label`` follow experiments/2026-10-25-pdl/critical_path.py
(PR #589): a kernel's capture order inside a CUDA graph is the low word of its graph
node id, a graph launch is keyed by its correlation id, and a decode step runs from one
target ``vocab_embedding_kernel`` to the one two launches later (80 routed GEMMs in the
target forward, 6 in the drafter's). ``label`` marks each kernel's capture stream:
main, the shared expert's aux stream, or the L2 prefetch side stream.
"""

import bisect
import gzip
import json
import re
import statistics
from collections import defaultdict

PREFETCH = "l2_prefetch_kernel"
MARK = "vocab_embedding_kernel"
SHARED_SHAPE = (
    "per_token_cast",
    "gemm_mxfp8",
    "activation_swiglu_clamp",
    "per_token_cast",
    "gemm_mxfp8",
)
# Steps longer than STALL times the window's median are host stalls.
STALL = 1.5
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
# (first step kept, end): request-inventory-5's, as in the PDL and fusion-survey
# analyses. A window's first step carries the request's prefill; mixed-c8's first two
# steps and those from 19 run before or after all eight requests share the batch.
WINDOWS = {"greedy": (1, None), "sampled": (1, 24), "mixed-c8": (2, 19)}


def base(name: str) -> str:
    return re.sub(r"_kernel(_\d+)?$", "", name)


def load(path: str) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    host = {}
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            host[e["args"].get("correlation")] = e["name"]
    ks = []
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        a = e.get("args", {})
        graph = host.get(a.get("correlation")) == "cudaGraphLaunch"
        ks.append(
            dict(
                ts=e["ts"],
                dur=e["dur"],
                end=e["ts"] + e["dur"],
                name=e["name"],
                grid=tuple(a.get("grid", ())),
                block=tuple(a.get("block", ())),
                node=a.get("graph node id", 0) & 0xFFFFFFFF,
                stream=a.get("stream"),
                launch=("G", a.get("correlation")) if graph else ("E", a.get("stream")),
                memcpy=e.get("cat") != "kernel",
            )
        )
    ks.sort(key=lambda k: k["ts"])
    return ks


def steps(ks: list[dict], skip: int, last: int | None) -> list[tuple[int, int]]:
    starts = [k["ts"] for k in ks if k["name"] == MARK]
    times = [k["ts"] for k in ks]
    idx = [bisect.bisect_left(times, t) for t in starts]

    def moe(a, b):
        return sum(1 for k in ks[a:b] if k["name"].startswith("moe_grouped"))

    out, i = [], 0
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


def label(ks: list[dict]) -> None:
    """Each kernel's capture stream (``cap``): main, shared or prefetch."""
    for k in ks:
        k["cap"] = "prefetch" if k["name"] == PREFETCH else "main"
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
            if len(window) == 5 and all(
                w["name"].startswith(s) for w, s in zip(window, SHARED_SHAPE)
            ):
                for w in window:
                    w["cap"] = "shared"
