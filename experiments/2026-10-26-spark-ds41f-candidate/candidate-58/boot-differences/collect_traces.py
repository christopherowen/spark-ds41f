#!/usr/bin/env python3
"""Copy every node's debug traces (step and row traces) into the run, then clear them.

usage: collect_traces.py URL --out DIR --cluster-config PROFILE [--trace-dir /cache/spark-ds41f/boot-traces]

A lab extra (the lab passes the head URL first; it is not used), run while the arm
still serves. For each node of the profile's node map it copies the files the
overlay's traces wrote under the trace directory (a container path below /cache,
which the profile mounts from {checkout}/cache) to DIR/<node>/, checks each copy's
size, and removes the node's copies through the serving container (the files are
the container's), so the next boot starts with an empty directory. Prints one JSON
line with what it collected.
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import spark_ds41f as spark  # noqa: E402


def ssh(target: str, command: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", target, command],
                          text=True, capture_output=True, timeout=timeout)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="the head URL the lab passes (unused)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument("--trace-dir", default="/cache/spark-ds41f/boot-traces")
    args = parser.parse_args(argv)
    if not args.trace_dir.startswith("/cache/"):
        raise SystemExit("--trace-dir must be below /cache, the mount of {checkout}/cache")
    cluster, nodes, _ = spark.configuration(argparse.Namespace(cluster_config=args.cluster_config))
    container = cluster["container"]["name"]
    host_dir = spark.host_path(cluster, "{checkout}/cache") + args.trace_dir[len("/cache"):]
    out = ROOT / args.out
    summary = {}
    for node in nodes["nodes"]:
        name = node["name"]
        target, transfer = spark.ssh_target(nodes, node), spark.transfer_target(nodes, node)
        listing = ssh(target, f"find {shlex.quote(host_dir)} -maxdepth 1 -type f -printf '%s %p\\n' 2>/dev/null")
        copied = []
        for line in listing.stdout.splitlines():
            size, path = line.split(" ", 1)
            local = out / name / Path(path).name
            local.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{transfer}:{path}", str(local)],
                           check=True, timeout=1200)
            if local.stat().st_size < int(size):
                raise SystemExit(f"{name}: copy of {path} is {local.stat().st_size} bytes, not {size}")
            copied.append(local.name)
        if copied:
            names = " ".join(shlex.quote(f"{args.trace_dir}/{c}") for c in copied)
            removed = ssh(target, f"docker exec {shlex.quote(container)} rm -f {names}")
            if removed.returncode:
                print(json.dumps({"node": name, "warning": "could not remove the copied traces",
                                  "stderr": removed.stderr[-400:]}), flush=True)
        summary[name] = copied
    print(json.dumps({"out": args.out, "trace_dir": args.trace_dir, "nodes": summary}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
