#!/usr/bin/env python3
"""Collect every node's sparknet traces and serving log after a lab step.

usage: collect_node_traces.py URL --out DIR [--cluster-config PATH] [--log-lines N]

A lab script step (the lab passes the head URL first; it is not used). For each
node of the cluster's node map it saves the serving container's log to
``DIR/<node>.log`` and copies the trace files a traced sparknet runtime wrote
(``SPARKNET_ROCE_TRACE=1``; the container shares the host's IPC namespace, so
they are in the host's ``/dev/shm``) to ``DIR/<node>/``. A trace file leaves
``/dev/shm`` only once its copy has the same size, so traces do not accumulate
in memory across boots. Prints one JSON line with what it collected.
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import spark_ds41f as spark

TRACE_GLOB = "/dev/shm/sparknet-trace-r*-p*.bin"


def ssh(target: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", target, command],
                          text=True, capture_output=True, timeout=300)


def collect(cluster_config: str, out: Path, log_lines: int) -> dict:
    cluster, nodes, _ = spark.configuration(argparse.Namespace(cluster_config=cluster_config))
    container = cluster["container"]["name"]
    summary = {}
    for node in nodes["nodes"]:
        name = node["name"]
        target, transfer = spark.ssh_target(nodes, node), spark.transfer_target(nodes, node)
        log = ssh(target, f"docker logs --tail {log_lines} {shlex.quote(container)} 2>&1")
        (out / f"{name}.log").write_text(log.stdout)
        listing = ssh(target, f"stat -c '%s %n' {TRACE_GLOB} 2>/dev/null")
        copied = []
        for line in listing.stdout.splitlines():
            size, path = line.split(" ", 1)
            local = out / name / Path(path).name
            local.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", f"{transfer}:{path}", str(local)],
                           check=True, timeout=600)
            if local.stat().st_size != int(size):
                raise SystemExit(f"{name}: copy of {path} is {local.stat().st_size} bytes, not {size}")
            ssh(target, f"rm -f {shlex.quote(path)}")
            copied.append(local.name)
        summary[name] = {"log_lines": log.stdout.count("\n"), "traces": copied}
    return summary


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="the head URL the lab passes (unused)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cluster-config", default="config/cluster.json")
    parser.add_argument("--log-lines", type=int, default=20000)
    args = parser.parse_args(argv)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"out": args.out, "nodes": collect(args.cluster_config, out, args.log_lines)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
