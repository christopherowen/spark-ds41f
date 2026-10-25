#!/usr/bin/env python3
"""One node's memory by component and serving process across one arm, from sampler.py.

usage: process_memory.py SAMPLES.jsonl START STOP   (epoch seconds bounding the arm)

At the arm's GPU-memory plateau (nvidia-smi's per-process total within 64 MiB of its
peak), +5 and +20 minutes after it, its lowest MemAvailable and its end, prints
MemAvailable, the driver's share (what /proc/meminfo leaves unaccounted: GPU allocations
on GB10), nvidia-smi's GPU total, anonymous, file, shmem and slab memory, and the four
largest container processes' RSS. Then, from the last smaps records of the arm, each
host-side process's (API server, engine core, headless launcher) anonymous PSS and the
workers' rollups. Used on experiments/2026-10-23-next-release/memory's soak samples for
the README's rank-0 table.
"""
import bisect
import json
import sys

KIB_PER_GIB = 2**20


def main() -> None:
    path, start, stop = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    meminfo, rss, gpu, names, rollups = [], [], [], {}, {}
    for line in open(path):
        record = json.loads(line)
        kind = record["k"]
        if kind == "new":
            names[record["pid"]] = record["comm"]
        if not start <= record["t"] <= stop:
            continue
        if kind == "m":
            meminfo.append(record)
        elif kind == "p":
            rss.append(record)
        elif kind == "g":
            gpu.append((record["t"], sum(a[2] for a in record.get("apps") or []) / 1024))
        elif kind in ("s", "w"):
            for pid, roll in (record.get("roll") or {}).items():
                if roll:
                    rollups[int(pid)] = roll

    def near(series, t):
        times = [s["t"] if isinstance(s, dict) else s[0] for s in series]
        return series[min(bisect.bisect_left(times, t), len(series) - 1)]

    def parts(record):
        mi = record["mi"]
        driver = (mi["MemTotal"] - mi["MemFree"] - mi["Buffers"] - mi["Cached"] - mi["SwapCached"]
                  - mi["AnonPages"] - mi["KReclaimable"] - mi["SUnreclaim"] - mi["KernelStack"]
                  - mi["PageTables"] - mi["SecPageTables"] - mi["Percpu"])
        return [mi["MemAvailable"], driver, mi["AnonPages"], mi["Active(file)"] + mi["Inactive(file)"],
                mi["Shmem"], mi["KReclaimable"] + mi["SUnreclaim"]]

    peak = max(v for _, v in gpu)
    plateau = next(t for t, v in gpu if v >= peak - 0.0625)
    low = min(meminfo, key=lambda r: r["mi"]["MemAvailable"])
    print(f"{'mark':14s} {'avail':>6s} {'driver':>6s} {'nvsmi':>6s} {'anon':>5s} {'file':>5s} "
          f"{'shmem':>5s} {'slab':>5s}  largest processes (RSS GiB)")
    for label, t in (("GPU plateau", plateau), ("+5 min", plateau + 300), ("+20 min", plateau + 1200),
                     ("lowest", low["t"]), ("end", meminfo[-1]["t"])):
        values = [v / KIB_PER_GIB for v in parts(near(meminfo, t))]
        procs = sorted(near(rss, t)["rss"], key=lambda r: -r[1])[:4]
        shown = " ".join(f"{names.get(pid, '?')[:14]}:{kb / KIB_PER_GIB:.2f}" for pid, kb in procs)
        print(f"{label:14s} {values[0]:6.2f} {values[1]:6.2f} {near(gpu, t)[1]:6.2f} {values[2]:5.2f} "
              f"{values[3]:5.2f} {values[4]:5.2f} {values[5]:5.2f}  {shown}")
    alive = {pid for pid, _ in rss[-1]["rss"]}
    for pid, roll in sorted(rollups.items()):
        if pid not in alive:
            continue  # start-up helpers (humming's nvrtc and ops probes) live a second or two
        print(f"  {names.get(pid, '?')[:16]:16s} pid {pid}: Rss {roll['Rss'] / KIB_PER_GIB:.2f}, anon PSS "
              f"{roll['Pss_Anon'] / KIB_PER_GIB:.2f}, file PSS {roll['Pss_File'] / KIB_PER_GIB:.2f}, shmem PSS "
              f"{roll['Pss_Shmem'] / KIB_PER_GIB:.2f} GiB")


if __name__ == "__main__":
    main()
