#!/usr/bin/env python3
"""Each profiled request type's kernels by family, and every kernel that is not TileLang with
the vLLM line that launched it.

usage: inventory_kernels.py RUN_DIR LABEL [--node dgx1]

RUN_DIR is results/private/determinism/<run>; the traces are LABEL/<workload>-trace/<node>/,
one a profiler window, in the order the workload printed its scenarios (LABEL/<workload>.json,
one JSON line a scenario; rejected scenarios have no window). Families by kernel name: ATen
(and CUB), NCCL, cuBLAS and CUTLASS, vLLM's CUDA kernels, Triton (vLLM names its Triton
kernels with a leading underscore), and the rest, which are TileLang's (ours and TileKernels').
An eagerly launched kernel's call site is the innermost vllm/ Python frame around the op that
launched it (the arm records stacks); a kernel replayed from a CUDA graph has no stack and is
placed by the TileLang kernels launched just before and after it on its stream.
"""
import bisect
import collections
import gzip
import json
import re
import sys
from pathlib import Path

run, label = Path(sys.argv[1]), sys.argv[2]
node = sys.argv[sys.argv.index("--node") + 1] if "--node" in sys.argv else "dgx1"
# --traces DIR --names a,b,...: traces read straight from a profiler directory, named in
# order (a run still in progress).
traces_dir = Path(sys.argv[sys.argv.index("--traces") + 1]) if "--traces" in sys.argv else None
trace_names = sys.argv[sys.argv.index("--names") + 1].split(",") if "--names" in sys.argv else []


def family(name: str) -> str:
    if name.startswith(("void at::", "at::")) or "cub::" in name or "at_cuda_detail" in name \
            or name.startswith("void (anonymous namespace)::elementwise_kernel_with_index"):
        return "aten"
    if "nccl" in name.lower():
        return "nccl"
    if re.search(r"cublas|cutlass|gemv|gemmk|xmma|sm\d+_", name, re.I):
        return "cublas/cutlass"
    if name.startswith("void vllm::") or name.startswith("vllm::"):
        return "vllm-cuda"
    if name.startswith("_") and not name.startswith("__"):
        return "triton"
    if name.startswith(("void ", "memcpy", "memset")):
        return "other-cuda"
    return "tilelang"


def short(name: str) -> str:
    m = re.search(r"at::native::(?:\(anonymous namespace\)::)?(\w+)", name)
    inner = re.search(r"(FillFunctor<\w+>|direct_copy_kernel_cuda|\w+Functor\w*<[^,>]+|\w+_kernel_cuda)", name)
    if m:
        return m.group(1) + (f"[{inner.group(1)}]" if inner else "")
    return name[:80]


def analyze(trace: Path) -> dict:
    events = json.load(gzip.open(trace))["traceEvents"]
    ops, launches, frames = {}, {}, collections.defaultdict(list)
    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("cat") == "cpu_op" and e.get("args", {}).get("External id") is not None:
            ops[e["args"]["External id"]] = e
        elif e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = e
        elif e.get("cat") == "python_function" and "vllm/" in e["name"]:
            frames[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"]))
    for row in frames.values():
        row.sort()
    starts = {tid: [f[0] for f in row] for tid, row in frames.items()}

    def site(op: dict) -> str:
        row, ts = frames.get(op["tid"], []), op["ts"]
        i = bisect.bisect_right(starts.get(op["tid"], []), ts)
        for j in range(i - 1, max(i - 5000, -1), -1):
            begin, end, name = row[j]
            if begin <= ts <= end:
                return re.sub(r"^.*?(vllm/)", r"\1", name)
        return "?"

    kernels = sorted((e for e in events if e.get("cat") == "kernel"),
                     key=lambda e: (e["args"].get("stream"), e["ts"]))
    by_stream = collections.defaultdict(list)
    for e in kernels:
        by_stream[e["args"].get("stream")].append(e)
    counts, foreign = collections.Counter(), collections.Counter()
    for row in by_stream.values():
        for i, e in enumerate(row):
            kind = family(e["name"])
            counts[kind] += 1
            if kind == "tilelang":
                continue
            graphed = e["args"].get("graph id", 0) not in (0, None)
            if graphed:
                before = next((row[j]["name"] for j in range(i - 1, max(i - 8, -1), -1)
                               if family(row[j]["name"]) == "tilelang"), "-")
                after = next((row[j]["name"] for j in range(i + 1, min(i + 8, len(row)))
                              if family(row[j]["name"]) == "tilelang"), "-")
                where = f"graph: {before[:40]} -> {after[:40]}"
            else:
                op = ops.get(e["args"].get("External id"))
                # A Triton or TileLang launch has no torch op: its CUDA launch call
                # places it in the Python stack.
                caller = op or launches.get(e["args"].get("correlation"))
                where = f"{op['name'] if op else 'launch'} @ {site(caller) if caller else '?'}"
            foreign[(kind, short(e["name"]), where)] += 1
    return {"counts": dict(counts), "foreign": foreign}


union = collections.defaultdict(set)
for name, trace in zip(trace_names, sorted(traces_dir.glob("*.json.gz")) if traces_dir else []):
    result = analyze(trace)
    print(f"## {name}  ({trace.name})")
    print("   kernels by family: " + ", ".join(f"{k} {v}" for k, v in sorted(result["counts"].items())))
    for (kind, kernel, where), n in result["foreign"].most_common():
        print(f"   {n:5d}  {kind:15} {kernel[:60]:60} {where}")
        union[(kind, kernel, where)].add(name)
for workload in () if traces_dir else ("decode", "prefill"):
    lines = [json.loads(line) for line in (run / label / f"{workload}.json").read_text().splitlines()
             if line.startswith("{")] if (run / label / f"{workload}.json").exists() else []
    names = [line.get("scenario", workload) for line in lines if line.get("status", 200) == 200]
    traces = sorted((run / label / f"{workload}-trace" / node).glob("*.json.gz"))
    for line in lines:
        if line.get("status", 200) != 200:
            print(f"## {line['scenario']}: rejected ({line.get('status')}): {line.get('message', '')[:120]}")
    for name, trace in zip(names, traces):
        result = analyze(trace)
        print(f"## {name}  ({trace.name})")
        print("   kernels by family: " + ", ".join(f"{k} {v}" for k, v in sorted(result["counts"].items())))
        for (kind, kernel, where), n in result["foreign"].most_common():
            print(f"   {n:5d}  {kind:15} {kernel[:60]:60} {where}")
            union[(kind, kernel, where)].add(name)
    if len(names) != len(traces):
        print(f"!! {workload}: {len(names)} scenarios but {len(traces)} traces")

print("\n## Every kernel that is not TileLang, with the scenarios that launch it")
for (kind, kernel, where), scenarios in sorted(union.items()):
    print(f"   {kind:15} {kernel[:60]:60} {where}\n      {', '.join(sorted(scenarios))}")
