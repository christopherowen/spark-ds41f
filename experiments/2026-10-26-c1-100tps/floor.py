#!/usr/bin/env python3
"""Bandwidth floor of a TP4 decode step, from rank 0's profiled traces.

usage: floor.py [TRACES] [--out floor.json]

TRACES is request-inventory-5's ports directory on dgx1
(results/private/determinism/request-inventory-5/ports), the traces the fusion
survey read. Steps are chosen as there: one target forward plus the drafter
forward after it, from one target vocab_embedding_kernel to the next, leaving
out each window's first step (the prompt) and the same edge steps.

Three things per window:

- Bytes. Every weight byte a rank reads from DRAM in one step, from the
  checkpoint's shapes at TP4 (below). The routed experts' bytes depend on how
  many distinct experts each layer's rows reach, which no trace records. They
  are inferred from FC1's kernel time at an assumed share of the attainable
  rate (ETA_FC1): an FC1 slower than the attainable rate means fewer experts.
- DRAM busy and idle time. The union of every kernel that streams weights
  (GEMMs, expert GEMMs, LM heads, the mHC projection's fn read, the L2
  prefetch's side stream) is the time DRAM has a weight stream in flight. The
  rest of the step is DRAM-idle time, split by what runs then: collectives,
  attention and indexer, other small kernels, or nothing (launch gaps).
- The floor: bytes / attainable rate. The step can be no shorter at the same
  rows. busy - floor is streaming below the attainable rate; idle is what
  overlap (fusion, PDL, collectives) could hand back at most.

The attainable rate, first estimate: the best weight stream measured in these
traces, the packed LM head (248 MB a call, one read of each byte), with FC1's
rate assumed (by_eta). With --expert-rate (the expert-rate bundle's output.txt) the
"measured" block replaces both assumptions: FC1 and FC2 at six rows are fitted to
a + b * experts on cold weights, the rate is FC1's marginal one (its bytes an expert
over b), and each window's experts a layer come from inverting that fit at its
traced FC1 time. GB10's LPDDR5x is specified at 273 GB/s.
"""
import argparse
import gzip
import json
import os
import statistics
from collections import defaultdict

DECODE = ("greedy", "sampled", "top-p", "top-k", "top-p-top-k", "structured", "logprobs",
          "prompt-logprobs", "penalties", "stop", "thinking", "n2", "mixed-c8")
MARK = "vocab_embedding_kernel"
PREFETCH = "l2_prefetch_kernel"
SPEC_GBS = 273.0
ETA_FC1 = (1.0, 0.9, 0.8)  # FC1's rate as a share of the attainable rate


def fp8(n, k):
    """E4M3 weight with one UE8M0 scale per 32 x 32 block."""
    return n * k + (n // 32) * (k // 32)


def bf16(n, k):
    return 2 * n * k


def fp32(n, k):
    return 4 * n * k


# One routed expert at TP4 (576 of 2,304 columns): E2M1 values, scales as
# 4-bit delta codes per 128-wide K block plus a base exponent a row
# (coded_scales in tilelang/moe.py: half the bytes of one UE8M0 a 32).
FC1_BYTES = 1152 * 5120 // 2 + 1152 * 5120 // 64 + 1152
FC2_BYTES = 5120 * 576 // 2 + 5120 * 576 // 64 + 5120

ATTENTION = fp8(1792, 5120) + fp8(8192, 1280) + 2 * fp8(1024, 4096) + fp8(5120, 2048)
SHARED = fp8(1152, 5120) + fp8(5120, 576)
MHC_FN = 2 * fp32(24, 4 * 5120)  # mix projection: 4 + 4 + 16 coefficients from 4 streams
HEAD = 32320 * 5120 * 3 // 2  # exact 12-bit packed BF16, vocabulary shard

DENSE = {  # bytes a step, per rank
    "attention projections (40 layers)": 40 * ATTENTION,
    "router (40 layers, BF16)": 40 * bf16(384, 5120),
    "shared expert (40 layers)": 40 * SHARED,
    "mHC fn (80 boundaries, FP32)": 40 * MHC_FN,
    "indexer Q-B and head weights (8 layers)": 8 * (fp8(4096, 1280) + bf16(32, 5120)),
    "compressors (layers 2, 8, 14 ratio 2; 20 ratio 1; BF16)": 3 * bf16(1024, 5120) + bf16(512, 5120),
    "Engram wkv (layers 1, 14)": 2 * fp8(6400, 6144),
    "target LM head": HEAD,
    "DSpark main projection (column TP)": fp8(1280, 15360),
    "drafter attention, shared expert, router, fn (3 layers)": 3 * (ATTENTION + SHARED + bf16(128, 5120) + MHC_FN),
    "drafter LM head": HEAD,
    "Markov head (BF16, L2-resident across its 5 calls)": bf16(32320, 256),
}

STREAM = ("gemm_mxfp8_kernel", "gemm_mxfp8_partials", "gemm_fp8w_bf16_partials", "gemm_bf16_partials",
          "gemm_bf16_splitk_kernel", "gemm_bf16_kernel", "moe_grouped", "vocab_packed",
          "mhc_project_streams", PREFETCH)
COLLECTIVE = ("oneshot_allreduce", "oneshot_allgather", "ncclDev")
ATTN = ("sparse_mla", "indexer", "metadata", "cache_writer", "rope", "compress")


def load(path):
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    ks = [(e["ts"], e["dur"], e["name"], e.get("tid"), tuple(e.get("args", {}).get("grid", ())))
          for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    ks.sort()
    return ks


def steps(ks):
    """(target kernels, drafter kernels, start, end) for each full decode step."""
    starts = [k[0] for k in ks if k[2] == MARK]
    segs = [[k for k in ks if starts[i] <= k[0] < starts[i + 1]] for i in range(len(starts) - 1)]
    moe = lambda s: sum(1 for k in s if k[2].startswith("moe_grouped"))
    out, i = [], 0
    while i + 2 < len(starts):
        if moe(segs[i]) == 80 and moe(segs[i + 1]) == 6:
            out.append((segs[i], segs[i + 1], starts[i], starts[i + 2]))
            i += 2
        else:
            i += 1
    return out


def union(intervals):
    total, end = 0.0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def idle_split(ks, start, end):
    """DRAM-idle time in [start, end), by the highest-priority kernel running then."""
    edges = []
    for k in ks:
        a, b = max(k[0], start), min(k[0] + k[1], end)
        if b > a:
            edges += [(a, 1, k), (b, -1, k)]
    edges.sort(key=lambda e: (e[0], e[1]))
    active, last, out = [], start, defaultdict(float)
    for t, kind, k in edges:
        if t > last:
            names = [x[2] for x in active]
            if not any(n.startswith(STREAM) for n in names):
                if any(n.startswith(COLLECTIVE) for n in names):
                    out["collectives"] += t - last
                elif any(n.startswith(ATTN) for n in names):
                    out["attention, indexer, cache"] += t - last
                elif names:
                    out["other small kernels"] += t - last
                else:
                    out["no kernel (launch gaps, host)"] += t - last
            last = t
        if kind == 1:
            active.append(k)
        else:
            active.remove(k)
    if end > last:
        out["no kernel (launch gaps, host)"] += end - last
    return out


def linfit(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def expert_fits(path):
    """{rows: {"fc1": (a, b), "fc2": (a, b)}} in us, from the expert-rate bundle's output."""
    cases = [json.loads(line) for line in open(path) if line.startswith('{"case": "experts"')]
    return {r: {k: linfit([c["experts"] for c in cases if c["rows"] == r],
                          [c[k + "_us"] for c in cases if c["rows"] == r]) for k in ("fc1", "fc2")}
            for r in sorted({c["rows"] for c in cases})}


def window(ks, skip, last, fits=None):
    sp = steps(ks)[skip:last]
    n = len(sp)
    acc = defaultdict(float)
    fc1, fc2, dfc1, dfc2, rows = [], [], [], [], defaultdict(list)
    for t, d, a, b in sp:
        both = t + d
        span = b - a
        acc["step"] += span
        acc["busy"] += union((max(k[0], a), min(k[0] + k[1], b)) for k in both if k[2].startswith(STREAM))
        for key, v in idle_split(both, a, b).items():
            acc["idle: " + key] += v
        t1 = [k for k in t if k[2].startswith("moe_grouped") and k[4][0] == 9]
        t2 = [k for k in t if k[2].startswith("moe_grouped") and k[4][0] == 40]
        fc1.append(sum(k[1] for k in t1))
        fc2.append(sum(k[1] for k in t2))
        dfc1.append(sum(k[1] for k in d if k[2].startswith("moe_grouped") and k[4][0] == 9))
        dfc2.append(sum(k[1] for k in d if k[2].startswith("moe_grouped") and k[4][0] == 40))
        rows[t1[0][4][1] // 6].append(span)
        for k in both:
            if k[2].startswith("vocab_packed"):
                acc["head_us"] += k[1]
                acc["head_calls"] += 1
            if k[2].startswith(COLLECTIVE):
                acc["collective_kernels"] += k[1]
    per = {k: v / n for k, v in acc.items()}
    head_gbs = HEAD * per["head_calls"] / per["head_us"] / 1e3
    out = {"steps": n, "step_ms": per["step"] / 1e3, "dram_busy_ms": per["busy"] / 1e3,
           "dram_idle_ms": {k[6:]: round(v / 1e3, 3) for k, v in per.items() if k.startswith("idle: ")},
           "collective_kernel_ms": round(per["collective_kernels"] / 1e3, 3),
           "attainable_gbs (LM head)": round(head_gbs, 1),
           "fc1_ms": statistics.mean(fc1) / 1e3, "fc2_ms": statistics.mean(fc2) / 1e3,
           "fc2_rate_vs_fc1": (FC2_BYTES / FC1_BYTES) * statistics.mean(fc1) / statistics.mean(fc2),
           "drafter_fc1_ms": statistics.mean(dfc1) / 1e3, "drafter_fc2_ms": statistics.mean(dfc2) / 1e3,
           "step_ms_by_routed_rows": {r: (len(v), round(statistics.mean(v) / 1e3, 2)) for r, v in sorted(rows.items())},
           "dense_mb": round(sum(DENSE.values()) / 1e6, 1), "by_eta": {}}
    for eta in ETA_FC1:
        rate = eta * head_gbs * 1e3  # bytes per us
        experts = out["fc1_ms"] * 1e3 * rate / FC1_BYTES / 40  # distinct experts a target layer
        d_experts = out["drafter_fc1_ms"] * 1e3 * rate / FC1_BYTES / 3
        routed = 40 * experts * (FC1_BYTES + FC2_BYTES)
        d_routed = 3 * d_experts * (FC1_BYTES + FC2_BYTES)
        total = sum(DENSE.values()) + routed + d_routed
        floor = total / (head_gbs * 1e3) / 1e3
        out["by_eta"][f"{eta:.2f}"] = {
            "experts_per_target_layer": round(experts, 1), "experts_per_drafter_layer": round(d_experts, 1),
            "routed_mb": round((routed + d_routed) / 1e6, 1), "total_mb": round(total / 1e6, 1),
            "floor_ms": round(floor, 2), "floor_ms_at_spec": round(total / (SPEC_GBS * 1e3) / 1e3, 2),
            "busy_minus_floor_ms": round(out["dram_busy_ms"] - floor, 2),
            "floor_share_of_step": round(floor / out["step_ms"], 3)}
    if fits:
        # the target's buckets: the six-row fit unless most steps routed three rows
        routed_rows = sum(r * len(v) for r, v in rows.items()) / n
        (a1, b1), (a2, b2) = (fits[3] if routed_rows <= 4.5 and 3 in fits else fits[6]).values()
        (da1, db1), (da2, db2) = fits[6].values()  # the drafter: 5 rows, 15 routes
        rate = FC1_BYTES / fits[6]["fc1"][1] / 1e3  # GB/s: FC1's marginal stream at six rows
        experts = (out["fc1_ms"] * 1e3 / 40 - a1) / b1
        d_experts = (out["drafter_fc1_ms"] * 1e3 / 3 - da1) / db1
        routed = (40 * experts + 3 * d_experts) * (FC1_BYTES + FC2_BYTES)
        total = sum(DENSE.values()) + routed
        floor = total / (rate * 1e3) / 1e3
        out["measured"] = {
            "rate_gbs (FC1 marginal, six rows)": round(rate, 1), "fit_rows": 3 if routed_rows <= 4.5 else 6,
            "experts_per_target_layer": round(experts, 1), "experts_per_drafter_layer": round(d_experts, 1),
            "routed_mb": round(routed / 1e6, 1), "total_mb": round(total / 1e6, 1),
            "floor_ms": round(floor, 2), "floor_ms_at_spec": round(total / (SPEC_GBS * 1e3) / 1e3, 2),
            "busy_minus_floor_ms": round(out["dram_busy_ms"] - floor, 2),
            "floor_share_of_step": round(floor / out["step_ms"], 3),
            "moe_launch_fixed_ms": round((40 * (a1 + a2) + 3 * (da1 + da2)) / 1e3, 3),
            "head_excess_ms": round((per["head_us"] - HEAD * per["head_calls"] / (rate * 1e3)) / 1e3, 3)}
    for k in ("step_ms", "dram_busy_ms", "fc1_ms", "fc2_ms", "fc2_rate_vs_fc1", "drafter_fc1_ms", "drafter_fc2_ms"):
        out[k] = round(out[k], 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="?", default="results/private/determinism/request-inventory-5/ports")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "floor.json"))
    ap.add_argument("--expert-rate", help="the expert-rate bundle's output.txt")
    args = ap.parse_args()
    fits = expert_fits(args.expert_rate) if args.expert_rate else None
    folder = os.path.join(args.traces, "decode-trace/dgx1")
    trace = dict(zip(DECODE, (os.path.join(folder, f) for f in sorted(os.listdir(folder)))))
    out = {"source": args.traces, "expert_rate": args.expert_rate, "dense_bytes": {k: v for k, v in DENSE.items()},
           "fc1_bytes_per_expert": FC1_BYTES, "fc2_bytes_per_expert": FC2_BYTES, "windows": {}}
    for name, (skip, last) in (("greedy", (1, None)), ("sampled", (1, 24)), ("mixed-c8", (2, 19))):
        out["windows"][name] = window(load(trace[name]), skip, last, fits)
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)
        handle.write("\n")
    for name, w in out["windows"].items():
        e = w["by_eta"]["1.00"]
        print(f"{name}: step {w['step_ms']} ms, DRAM busy {w['dram_busy_ms']} ms, "
              f"floor {e['floor_ms']} ms at {w['attainable_gbs (LM head)']} GB/s ({e['total_mb']} MB)")
        m = w.get("measured")
        if m:
            print(f"  measured: {m['experts_per_target_layer']} experts a layer, floor {m['floor_ms']} ms at "
                  f"{m['rate_gbs (FC1 marginal, six rows)']} GB/s ({m['total_mb']} MB), busy - floor "
                  f"{m['busy_minus_floor_ms']} ms, MoE launch fixed {m['moe_launch_fixed_ms']} ms")


if __name__ == "__main__":
    main()
