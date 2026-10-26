# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The L2 prefetch windows' schedules on DeepSeek-V4.1's forty decode layers at one
rank's shapes (``tests/kernels/deepseek_v41_l2_schedule_chain``): step time, bits,
and each window consumer's time against its L2-speed floor.

usage: benchmark_deepseek_v41_l2_schedule.py [--rows 6,48,96] [--tp 4]
           [--schedules head,chained,typed,off] [--rounds 8] [--replays 10] [--trace 5]

Per row count, one CUDA graph per schedule holds the forty layers (a target forward
without its head). The graphs replay in alternation: ``--rounds`` rounds, the
schedules in an order that rotates by one each round, ``--replays`` timed replays
each. Before timing, each graph's replay must give the eager run's bits in every
buffer the layers write, and every schedule the same bits.

Then ``--trace`` replays of each graph run under the torch profiler. Each kernel is
placed by capture order (the graph node id's low word) and named by its role in its
layer; each prefetch kernel is its window's fill. A consumer's floor is its GEMM's
time with its weight already in L2 (the median of four back-to-back calls after a
first). Its excess is its time above the floor; its L2-speed share is the floor over
its time, summed over the step's calls.

Prints JSON lines: "plans" (each window's plan), "bits", "time" (per schedule, the
median step in microseconds, per round, and against the first schedule), "floors",
"windows" (per schedule, window and layer kind: fills a step, fill p50, and how many
start after their first consumer's start) and "consumers" (per schedule, consumer and
layer kind: calls, p50, excess a step, L2-speed share).
"""

import argparse
import json
import re
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from tests.kernels.deepseek_v41_l2_schedule_chain import ScheduleChain
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4_1.tilelang import linear


def graph(fn) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def replay_us(g: torch.cuda.CUDAGraph) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    g.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000


def base(name: str) -> str:
    return re.sub(r"_kernel(_\d+)?$", "", name.split("(")[0].split("<")[0].strip())


def kernels_of(prof) -> list[list[dict]]:
    """Each traced replay's kernels in capture order."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.json"
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    replays = defaultdict(list)
    for e in events:
        if e.get("cat") == "kernel":
            a = e.get("args", {})
            replays[a.get("correlation")].append(
                dict(
                    name=base(e["name"]),
                    ts=e["ts"],
                    dur=e["dur"],
                    end=e["ts"] + e["dur"],
                    node=a.get("graph node id", 0) & 0xFFFFFFFF,
                    grid=tuple(a.get("grid", ())),
                )
            )
    out = []
    for ks in replays.values():
        if len(ks) < 100:
            continue
        # capture order; time order if this profiler gives no node ids
        by_node = len({k["node"] for k in ks}) == len(ks)
        ks.sort(key=lambda k: (k["node"] if by_node else k["ts"]))
        out.append(ks)
    return out


def roles(ks: list[dict], chain: ScheduleChain) -> None:
    """Name each kernel's role (``role``, ``layer``) by walking capture order with the
    layers' known shape; prefetch kernels get the role of the anchor before them."""
    i, anchor = 0, None

    def take(name, role, layer, after_spin=False):
        nonlocal i, anchor
        while i < len(ks):
            k = ks[i]
            i += 1
            if k["name"] == "l2_prefetch":
                k["role"], k["layer"] = f"fill@{anchor[1]}", anchor[0]
                continue
            if k["name"].startswith(name):
                k["role"], k["layer"] = role, layer
                anchor = (layer, role)
                return k
            k.setdefault("role", "other")
            k.setdefault("layer", layer)
        raise ValueError(f"layer {layer}: no {name} for {role}")

    def maybe(name, role, layer):
        """Take the next non-prefetch kernel if it is ``name`` (a reduce some row
        counts launch and others do not)."""
        j = i
        while j < len(ks) and ks[j]["name"] == "l2_prefetch":
            j += 1
        if j < len(ks) and ks[j]["name"].startswith(name):
            take(name, role, layer)

    for L in chain.layers:
        n = L.index
        if L.engram:
            take("gemm_mxfp8", "engram", n)
        take("mhc_project_streams", "attention mHC", n)
        take("gemm_mxfp8", "fused Q-A/KV", n)
        take("gemm_mxfp8", "Q-B", n)
        if L.kind == "own":
            take("gemm_bf16", "compressor", n)
        if L.kind in ("own", "idx"):
            take("gemm_mxfp8", "index Q-B", n)
            take("gemm_bf16", "head weights", n)
            maybe("gemm_splitk_reduce", "head weights reduce", n)
        take("gemm_fp8w_bf16", "WO-A", n)
        maybe("gemm_splitk_reduce_cast", "WO-A reduce", n)
        take("gemm_mxfp8", "WO-B", n)
        take("mhc_project_streams", "FFN mHC", n)
        take("gemm_mxfp8", "shared gate/up", n)
        take("per_token_cast", "shared cast", n)
        take("gemm_mxfp8", "shared down", n)
        take("gemm_bf16", "router", n)
        take("moe_combine", "combine", n)
    while i < len(ks):
        k = ks[i]
        i += 1
        if k["name"] == "l2_prefetch":
            k["role"], k["layer"] = f"fill@{anchor[1]}", anchor[0]


# The window a fill serves, by the anchor it is issued after.
WINDOW = {
    "fused Q-A/KV": "WO",
    "Q-B": "WO",  # IDX in a typed schedule's owner layer
    "compressor": "WO",  # a typed schedule's owner layer
    "WO-A": "FFN",
    "WO-A reduce": "FFN",
    "WO-B": "FFN",
    "combine": "NEXT",  # ENGRAM in a typed schedule before an Engram layer
    "engram": "NEXT",
}
CONSUMERS = {
    "WO": ("WO-A", "WO-B"),
    "FFN": ("FFN mHC", "router", "shared gate/up", "shared down"),
    "NEXT": ("attention mHC", "fused Q-A/KV", "Q-B"),
    "IDX": ("compressor", "index Q-B", "head weights"),
    "ENGRAM": ("engram",),
}


def windows(ks: list[dict], chain: ScheduleChain, schedule: str) -> list[dict]:
    """Every fill of a replay: window, layer (the layer its consumers belong to),
    layer kind, fill start and end, and its consumers' kernels."""
    by = {(k.get("layer"), k.get("role")): k for k in ks}
    out = []
    for k in ks:
        if not k.get("role", "").startswith("fill@"):
            continue
        anchor, n = k["role"][5:], k["layer"]
        L = chain.layers[n]
        window, target = WINDOW[anchor], n
        if (
            window == "WO"
            and anchor == "Q-B"
            and schedule == "typed"
            and L.kind == "own"
        ):
            window = "IDX"
        if anchor == "combine":
            target = n + 1
            if schedule == "typed" and chain.layers[n + 1].engram:
                window = "ENGRAM"
        if anchor == "engram":
            target = n
        T = chain.layers[target]
        consumers = [by[(target, r)] for r in CONSUMERS[window] if (target, r) in by]
        out.append(
            dict(
                window=window,
                layer=target,
                kind=T.kind,
                engram=T.engram,
                start=k["ts"],
                end=k["end"],
                dur=k["dur"],
                consumers=consumers,
            )
        )
    return out


def floors(chain: ScheduleChain) -> dict[str, float]:
    """Each consumer GEMM's time with its weight in L2: four back-to-back calls after
    a first, under the profiler; the median of the four."""
    swa = next(L for L in chain.layers if L.kind == "swa" and not L.engram)
    own = next(L for L in chain.layers if L.kind == "own")
    eng = next(L for L in chain.layers if L.engram)
    rows = chain.rows
    a, f = swa.attention_entry, swa.ffn_entry
    calls = {
        "attention mHC": (lambda: a(*chain.inputs), "mhc_project_streams"),
        "FFN mHC": (
            lambda: f(swa.wo_out, a.streams, a.post, a.comb, a.pre),
            "mhc_project_streams",
        ),
        "fused Q-A/KV": (
            lambda: linear._block32_gemm(
                swa.qa_kv, a.mx.values, a.mx.scales, swa.qa, rows
            ),
            "gemm_mxfp8",
        ),
        "Q-B": (
            lambda: linear._block32_gemm(
                swa.q_b, swa.casts["q_b"].xq, swa.casts["q_b"].sf, swa.qb, rows
            ),
            "gemm_mxfp8",
        ),
        "WO-A": (
            lambda: swa.wo.a(
                swa.qb,
                swa.wo_xq[:rows],
                swa.wo_sf[:rows],
                swa.partials[0] if swa.partials else None,
            ),
            "gemm_fp8w_bf16_partials",
        ),
        "WO-B": (
            lambda: linear._block32_gemm(
                swa.wo.b, swa.wo_xq, swa.wo_sf, swa.wo_out, rows
            ),
            "gemm_mxfp8",
        ),
        "router": (lambda: swa.gate(f.y, swa.logits, swa.gate_scratch), "gemm_bf16"),
        "shared gate/up": (
            lambda: linear._block32_gemm(
                swa.gate_up, f.mx.values, f.mx.scales, swa.h, rows
            ),
            "gemm_mxfp8",
        ),
        "shared down": (
            lambda: linear._block32_gemm(
                swa.down,
                swa.casts["down"].xq,
                swa.casts["down"].sf,
                swa.shared_out,
                rows,
            ),
            "gemm_mxfp8",
        ),
        "compressor": (
            lambda: own.compressor(own.attention_entry.y, own.latent, own.comp_scratch),
            "gemm_bf16",
        ),
        "index Q-B": (
            lambda: linear._block32_gemm(
                own.index_q_b,
                own.casts["index_q_b"].xq,
                own.casts["index_q_b"].sf,
                own.iq,
                rows,
            ),
            "gemm_mxfp8",
        ),
        "head weights": (
            lambda: own.weights_proj(own.attention_entry.y, own.iw, own.wproj_scratch),
            "gemm_bf16",
        ),
        "engram": (
            lambda: linear._block32_gemm(
                eng.wkv, eng.casts["wkv"].xq, eng.casts["wkv"].sf, eng.kv, rows
            ),
            "gemm_mxfp8",
        ),
    }
    out = {}
    for role, (fn, name) in calls.items():
        fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            prof.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
        durs = sorted(
            (e["ts"], e["dur"])
            for e in events
            if e.get("cat") == "kernel" and base(e["name"]).startswith(name)
        )
        out[role] = round(statistics.median(d for _, d in durs[1:]), 2)
    return out


def summarize(traced: dict, chain: ScheduleChain, floor: dict) -> tuple[dict, dict]:
    win = defaultdict(lambda: defaultdict(list))
    con = defaultdict(lambda: defaultdict(list))
    for schedule, replays in traced.items():
        for ks in replays:
            for w in windows(ks, chain, schedule):
                kind = (
                    "eng"
                    if w["engram"] and w["window"] in ("NEXT", "ENGRAM")
                    else w["kind"]
                )
                t = win[(schedule, w["window"], kind)]
                t["dur"].append(w["dur"])
                first = min((c["ts"] for c in w["consumers"]), default=w["end"])
                t["late"].append(w["end"] - first)
                for c in w["consumers"]:
                    u = con[(schedule, c["role"], kind)]
                    u["dur"].append(c["dur"])
                    u["lead"].append(w["end"] - c["ts"])
    n = {s: len(r) for s, r in traced.items()}
    windows_out = {
        f"{s} {window} [{kind}]": {
            "per_step": round(len(t["dur"]) / n[s], 1),
            "fill_p50_us": round(statistics.median(t["dur"]), 1),
            "late_fraction": round(sum(x > 0 for x in t["late"]) / len(t["late"]), 2),
            "late_p50_us": round(statistics.median(t["late"]), 1),
        }
        for (s, window, kind), t in sorted(win.items())
    }
    consumers_out = {}
    for (s, role, kind), u in sorted(con.items()):
        fl = floor.get(role)
        total = sum(u["dur"])
        consumers_out[f"{s} {role} [{kind}]"] = {
            "per_step": round(len(u["dur"]) / n[s], 1),
            "p50_us": round(statistics.median(u["dur"]), 1),
            "floor_us": fl,
            "starts_before_fill_end": round(
                sum(x > 0 for x in u["lead"]) / len(u["lead"]), 2
            ),
            "excess_us_per_step": round(
                sum(max(0.0, d - fl) for d in u["dur"]) / n[s], 1
            )
            if fl
            else None,
            "l2_share": round(fl * len(u["dur"]) / total, 3) if fl else None,
        }
    return windows_out, consumers_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="6,48,96")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--schedules", default="head,chained,typed,off")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--trace", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    schedules = args.schedules.split(",")
    with set_current_vllm_config(VllmConfig()):
        for rows in map(int, args.rows.split(",")):
            chain = ScheduleChain(rows, tp=args.tp, seed=args.seed)
            head = f"tp{args.tp} rows {rows}"
            print(
                json.dumps(
                    {
                        "plans": head,
                        **chain.plan_summary(),
                        "cycles_per_us": round(chain.cycles_per_us, 1),
                    }
                ),
                flush=True,
            )
            chain.run(schedules[0])
            torch.cuda.synchronize()
            eager = chain.digests()
            graphs, bits = {}, {}
            for s in schedules:
                graphs[s] = graph(lambda s=s, run=chain.run: run(s))
                graphs[s].replay()
                torch.cuda.synchronize()
                got = chain.digests()
                bits[s] = sorted(k for k in eager if got[k] != eager[k])
            print(
                json.dumps(
                    {
                        "bits": head,
                        "differs_from_eager": bits,
                        "equal": not any(bits.values()),
                    }
                ),
                flush=True,
            )
            for g in graphs.values():
                for _ in range(3):
                    replay_us(g)
            times = {s: [] for s in schedules}
            for r in range(args.rounds):
                order = (
                    schedules[r % len(schedules) :] + schedules[: r % len(schedules)]
                )
                for s in order:
                    times[s].append(
                        statistics.median(
                            replay_us(graphs[s]) for _ in range(args.replays)
                        )
                    )
            ref = statistics.median(times[schedules[0]])
            print(
                json.dumps(
                    {
                        "time": head,
                        "rounds": args.rounds,
                        "replays": args.replays,
                        **{
                            s: {
                                "step_us": round(statistics.median(v), 1),
                                "rounds_us": [round(x, 1) for x in v],
                                "vs_first_percent": round(
                                    100 * (statistics.median(v) - ref) / ref, 2
                                ),
                                "rounds_vs_first_percent": [
                                    round(100 * (x - y) / y, 2)
                                    for x, y in zip(v, times[schedules[0]])
                                ],
                            }
                            for s, v in times.items()
                        },
                    }
                ),
                flush=True,
            )
            floor = floors(chain)
            print(json.dumps({"floors": head, **floor}), flush=True)
            traced = {}
            for s in schedules:
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(args.trace):
                        graphs[s].replay()
                        torch.cuda.synchronize()
                replays = kernels_of(prof)
                for ks in replays:
                    roles(ks, chain)
                traced[s] = replays
            win, con = summarize(traced, chain, floor)
            print(json.dumps({"windows": head, **win}), flush=True)
            print(json.dumps({"consumers": head, **con}), flush=True)
            del graphs, chain
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
