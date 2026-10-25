#!/usr/bin/env python3
"""Each node's memory once the arm serves (a lab extras script).

usage: mem_snapshot.py URL --label LABEL [--container NAME]

For every node of the production node map: MemAvailable from /proc/meminfo, and the
serving container's own memory lines (weight loading, KV cache, CUDA graphs). One JSON line
per node. Arms with the same KV settings compare by MemAvailable: the difference is what
their weights and workspaces leave free. The log lines are the named container's: the arm's
own (spark-ds41f since the rename), or they are the stopped production container's.
"""
import argparse
import json
import re
import subprocess

NODES = ("dgx1", "dgx2", "dgx3", "dgx4")
CONTAINER = "dsv41-karmic-kraken"
LINES = re.compile(r"Model loading took|KV cache|CUDA graph|memory profiling|weights take|GiB")


def run(node: str, command: str) -> str:
    result = subprocess.run(["ssh", "-o", "ConnectTimeout=10", node, command], capture_output=True, text=True,
                            timeout=120, check=False)
    return result.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--label", required=True)
    ap.add_argument("--container", default=CONTAINER, help="the serving container whose memory lines to read")
    args = ap.parse_args()
    for node in NODES:
        meminfo = run(node, "grep -E '^(MemAvailable|MemTotal):' /proc/meminfo")
        kib = {k: int(v) for k, v in re.findall(r"^(\w+):\s+(\d+) kB", meminfo, re.MULTILINE)}
        logs = run(node, f"docker logs {args.container} 2>&1 | grep -E 'Model loading took|KV cache|CUDA graph' | tail -6")
        print(json.dumps({
            "label": args.label, "node": node,
            "mem_available_gib": round(kib.get("MemAvailable", 0) / 2**20, 3),
            "mem_total_gib": round(kib.get("MemTotal", 0) / 2**20, 3),
            "log": [line[-220:] for line in logs.splitlines() if LINES.search(line)],
        }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
