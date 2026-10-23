#!/usr/bin/env python3
"""Attribute one node's start-up memory from sampler.py, the allocator trace and the
engine log (`docker logs -t`).

usage: analyze.py SAMPLES.jsonl ENGINE.log [--torch torch-rankN.tsv] [--settle 150]
         [--series OUT.csv]

Prints each component (GiB) at the engine's start-up phases, at the start-up low of
MemAvailable and --settle seconds after it, and the low's difference from that level.
The driver row is what /proc/meminfo leaves unaccounted: the GPU's allocations, which
live in system memory on GB10 but belong to no process's RSS.
"""
import argparse
import bisect
import datetime as dt
import json
import re

GiB_KB = 2**20
PHASES = (
    ("weights read", r"Loading weights took"),
    ("model loaded", r"Model loading took"),
    ("b12x weights prep", r"b12x weights preparation uses"),
    ("KV reserved", r"memory for KV Cache as specified"),
    ("b12x state prep", r"b12x state preparation uses"),
    ("b12x bind prep", r"b12x bind preparation uses"),
    ("JIT warmup", r"JIT kernel warmup starting"),
    ("target capture", r"Capturing CUDA graphs \(FULL\)"),
    ("DSpark capture", r"Capturing model for DSpark speculator"),
    ("capture done", r"Graph capturing finished"),
    ("init engine done", r"init engine \(profile"),
    ("API up", r"Application startup complete"),
)
STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z ")


def cet(epoch: float) -> str:
    offset = 2 if epoch < dt.datetime(2026, 10, 25, 1, tzinfo=dt.timezone.utc).timestamp() else 1
    zone = dt.timezone(dt.timedelta(hours=offset))
    return dt.datetime.fromtimestamp(epoch, zone).strftime("%H:%M:%S.%f")[:-4]


def engine_phases(path: str) -> list[tuple[str, float]]:
    found, last = [], None
    for line in open(path, errors="replace"):
        match = STAMP.match(line)
        if match:
            base = dt.datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
            last = base.timestamp() + float(match.group(2) or 0)
        if last is None:
            continue
        for name, pattern in PHASES:
            if re.search(pattern, line):
                label = name
                if any(n == name for n, _ in found):
                    label = f"{name} (again)"
                if not any(n == label for n, _ in found):
                    found.append((label, last))
    return sorted(found, key=lambda item: item[1])


def components(mi: dict, cg: dict) -> dict:
    driver = (mi["MemTotal"] - mi["MemFree"] - mi["Buffers"] - mi["Cached"] - mi["SwapCached"]
              - mi["AnonPages"] - mi["KReclaimable"] - mi["SUnreclaim"] - mi["KernelStack"]
              - mi["PageTables"] - mi["SecPageTables"] - mi["Percpu"])
    values = {
        "MemAvailable": mi["MemAvailable"], "MemFree": mi["MemFree"],
        "file cache": mi["Active(file)"] + mi["Inactive(file)"], "anon": mi["AnonPages"],
        "shmem": mi["Shmem"], "slab unreclaimable": mi["SUnreclaim"],
        "slab reclaimable": mi["KReclaimable"],
        "page tables": mi["PageTables"] + mi["SecPageTables"] + mi["KernelStack"],
        "driver (GPU)": driver,
    }
    values = {key: value / GiB_KB for key, value in values.items()}
    for key in ("anon", "file", "kernel", "shmem"):
        if key in cg:
            values[f"container {key}"] = cg[key] / 2**30
    return values


def at(times: list[float], records: list, t: float):
    index = bisect.bisect_right(times, t) - 1
    return records[max(index, 0)] if records else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples")
    parser.add_argument("engine")
    parser.add_argument("--torch")
    parser.add_argument("--settle", type=float, default=150)
    parser.add_argument("--series")
    args = parser.parse_args()

    mem, procs, gpu, names = [], [], [], {}
    for line in open(args.samples):
        record = json.loads(line)
        kind = record["k"]
        if kind == "m" and "MemTotal" in record["mi"]:
            mem.append((record["t"], components(record["mi"], record["cg"])))
        elif kind == "p":
            procs.append((record["t"], {pid: rss for pid, rss in record["rss"] if rss}))
        elif kind == "g" and "apps" in record:
            gpu.append((record["t"], record["apps"]))
        elif kind == "new":
            cmd = record["cmd"]
            if record["comm"] == "vllm" or "vllm serve" in cmd:
                role = "API server"
            elif record["comm"].startswith("VLLM::EngineCor"):
                role = "EngineCore"
            elif record["comm"].startswith("VLLM::Worker"):
                role = "worker"
            elif "multiprocessing" in cmd and "spawn" in cmd:
                role = "spawned (compiler?)"
            else:
                role = "other"
            names[record["pid"]] = role
    trace = []
    if args.torch:
        for line in open(args.torch):
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "t" or len(parts) < 4 or not parts[2]:
                continue
            trace.append((float(parts[0]), int(parts[2]) / 2**30, int(parts[3]) / 2**30))

    phases = engine_phases(args.engine)
    if not phases:
        raise SystemExit("no start-up phases in the engine log")
    begin = phases[0][1]
    end = next((t for name, t in phases if name == "API up"), phases[-1][1]) + 60
    window = [(t, c) for t, c in mem if begin <= t <= end]
    low_t, low = min(window, key=lambda row: row[1]["MemAvailable"])
    later = [(t, c) for t, c in mem if t >= low_t + args.settle]
    settle_t, settle = later[0] if later else mem[-1]

    mem_t = [t for t, _ in mem]
    proc_t = [t for t, _ in procs]
    gpu_t = [t for t, _ in gpu]
    trace_t = [t for t, *_ in trace]

    def row(t: float) -> dict:
        values = dict(at(mem_t, mem, t)[1])
        snapshot = at(proc_t, procs, t)
        if snapshot:
            roles = {}
            for pid, rss in snapshot[1].items():
                role = names.get(pid, "other")
                roles[role] = roles.get(role, 0) + rss / GiB_KB
            for role in ("API server", "EngineCore", "worker", "spawned (compiler?)", "other"):
                values[f"RSS {role}"] = roles.get(role, 0.0)
        apps = at(gpu_t, gpu, t)
        if apps:
            values["GPU worker (nvidia-smi)"] = sum(m or 0 for _, comm, m in apps[1] if "Worker" in comm) / 1024
            values["GPU other processes"] = sum(m or 0 for _, comm, m in apps[1] if "Worker" not in comm) / 1024
        point = at(trace_t, trace, t)
        if point:
            values["torch allocated"], values["torch reserved"] = point[1], point[2]
            if "GPU worker (nvidia-smi)" in values:
                values["GPU non-torch"] = values["GPU worker (nvidia-smi)"] - point[2]
        return values

    columns = [(name, t) for name, t in phases] + [("LOW", low_t), (f"low +{args.settle:.0f} s", settle_t)]
    rows = {name: row(t) for name, t in columns}
    keys = list(rows["LOW"].keys())
    header = ["component"] + [name for name, _ in columns] + ["low - settled"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    print("| time CET | " + " | ".join(cet(t) for _, t in columns) + " | |")
    for key in keys:
        cells = [f"{rows[name].get(key, float('nan')):.3f}" for name, _ in columns]
        delta = rows["LOW"].get(key, 0) - rows[f"low +{args.settle:.0f} s"].get(key, 0)
        print(f"| {key} | " + " | ".join(cells) + f" | {delta:+.3f} |")

    if args.series:
        with open(args.series, "w") as handle:
            handle.write("t_rel_low," + ",".join(keys) + "\n")
            for t, _ in mem:
                if low_t - 120 <= t <= low_t + args.settle + 30:
                    values = row(t)
                    handle.write(f"{t - low_t:.2f}," + ",".join(f"{values.get(k, float('nan')):.4f}" for k in keys) + "\n")


if __name__ == "__main__":
    main()
