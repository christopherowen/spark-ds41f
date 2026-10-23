#!/usr/bin/env python3
"""After an instrumented boot (and its soak): keep each node's timestamped engine log and
allocator trace before a later boot replaces the container. From
experiments/2026-10-21-startup-memory/collect.py, for the four-node ring.

usage (a lab extra, run on the head node from the repository root):
  collect.py HEAD_URL --out DIR --label LABEL [--wait SECONDS] [--trace DIR]
Prints one JSON line: request counters before and after the wait (foreign traffic shows
as a moving counter) and the files kept.
"""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

NODES = ("dgx1", "dgx2", "dgx3", "dgx4")
CONTAINER = "dsv41-karmic-kraken"


def counters(url: str) -> dict:
    try:
        text = urllib.request.urlopen(f"{url}/metrics", timeout=10).read().decode()
    except OSError as error:
        return {"error": str(error)}
    wanted = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:request_success_total")
    totals = dict.fromkeys(wanted, 0.0)
    for line in text.splitlines():
        for name in wanted:
            if line.startswith(name + "{") or line.startswith(name + " "):
                totals[name] += float(line.rsplit(" ", 1)[1])
    return totals


def on_node(node: str, command: str) -> subprocess.CompletedProcess:
    argv = ["bash", "-c", command] if node == "dgx1" else ["ssh", "-o", "BatchMode=yes", node, command]
    return subprocess.run(argv, capture_output=True, timeout=300)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--wait", type=float, default=150)
    parser.add_argument("--trace", default="cache/startup-memory/attrib",
                        help="the allocator trace directory, relative to the node checkout")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    before = counters(args.url)
    time.sleep(args.wait)
    after = counters(args.url)
    kept = {}
    for node in NODES:
        log = on_node(node, f"docker logs -t {CONTAINER} 2>&1")
        target = out / f"engine-{node}-{args.label}.log"
        target.write_bytes(log.stdout)
        kept[target.name] = len(log.stdout)
        trace = on_node(node, f"cat ~/projects/spark-ds41f/{args.trace}/torch-rank*.tsv 2>/dev/null")
        target = out / f"torch-{node}-{args.label}.tsv"
        target.write_bytes(trace.stdout)
        kept[target.name] = len(trace.stdout)
    print(json.dumps({"label": args.label, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "requests_before": before, "requests_after": after, "kept": kept}))


if __name__ == "__main__":
    main()
