#!/usr/bin/env python3
"""servebench's closed loop with every node's GPU clocks sampled alongside.

usage: load_with_clocks.py URL --label NAME --output PATH [--nodes dgx1,dgx2,dgx3,dgx4]
                           [servebench run options after --]

Each node runs ``nvidia-smi`` every 500 ms (SM and memory clocks, power, temperature,
active clock-event reasons) for the run's length, under ``timeout`` so nothing outlives
it. The output JSON holds per node the percentiles of each over the measurement window
(warm-up excluded) and the servebench result's path; the last stdout line is a summary.
"""
import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from servebench import stats  # noqa: E402
from servebench.__main__ import main as servebench  # noqa: E402

FIELDS = ("clocks.sm", "clocks.mem", "power.draw", "temperature.gpu", "clocks_event_reasons.active")


def sampler(node: str, seconds: int) -> subprocess.Popen:
    command = ["timeout", str(seconds), "nvidia-smi", "--query-gpu=timestamp," + ",".join(FIELDS),
               "--format=csv,noheader,nounits", "-lms", "500"]
    if node != socket.gethostname():
        command = ["ssh", "-n", "-o", "BatchMode=yes", node, " ".join(command)]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)


def parse(text: str, window: tuple[float, float], started: float) -> dict:
    """Percentiles of each field over samples inside the window (seconds after start)."""
    columns = {field: [] for field in FIELDS}
    samples = 0
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 1 + len(FIELDS):
            continue
        try:
            moment = time.mktime(time.strptime(parts[0].split(".")[0], "%Y/%m/%d %H:%M:%S"))
        except ValueError:
            continue
        if not window[0] <= moment - started <= window[1]:
            continue
        samples += 1
        for field, value in zip(FIELDS, parts[1:]):
            columns[field].append(value)
    out = {"samples": samples}
    for field in FIELDS[:-1]:
        values = []
        for value in columns[field]:
            try:
                values.append(float(value))
            except ValueError:
                pass
        if values:
            out[field] = stats.percentiles(values, (1, 10, 50, 90, 99)) | {
                "min": min(values), "max": max(values), "mean": sum(values) / len(values)}
    reasons = columns[FIELDS[-1]]
    out["event_reasons"] = {r: reasons.count(r) for r in sorted(set(reasons))}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nodes", default="dgx1,dgx2,dgx3,dgx4")
    parser.add_argument("--warmup", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("rest", nargs=argparse.REMAINDER, help="servebench run options after --")
    args = parser.parse_args()
    nodes = args.nodes.split(",")
    run_output = str(Path(args.output).with_suffix("")) + "-servebench.json"
    rest = [a for a in args.rest if a != "--"]
    seconds = int(args.warmup + args.duration + 180)
    started = time.time()
    samplers = {node: sampler(node, seconds) for node in nodes}
    code = servebench(["run", args.url, "--warmup", str(args.warmup), "--duration", str(args.duration),
                       "--label", args.label, "--output", run_output, *rest])
    for process in samplers.values():
        process.terminate()
    window = (args.warmup, args.warmup + args.duration)
    clocks = {node: parse(process.communicate(timeout=30)[0], window, started) for node, process in samplers.items()}
    result = {"label": args.label, "servebench": run_output, "window_s": window, "nodes": clocks}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1) + "\n")
    summary = {"summary": "load with clocks", "label": args.label, "output": args.output,
               "sm_mhz_p10_p50_p90": {n: [c.get("clocks.sm", {}).get(f"p{q}") for q in (10, 50, 90)] for n, c in clocks.items()},
               "event_reasons": {n: c["event_reasons"] for n, c in clocks.items()}}
    print(json.dumps(summary), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
