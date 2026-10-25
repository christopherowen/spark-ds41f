#!/usr/bin/env python3
"""Each node's memory once the arm serves (a lab extras script).

usage: mem_snapshot.py URL --label LABEL

For every node of the production node map: MemAvailable from /proc/meminfo, and the
running serving container's own startup lines (weight loading, the KV cache size, CUDA
graphs). One JSON line per node, naming the container and its image, so a reader can check
that the lines belong to the arm's build. Arms with the same KV settings compare by
MemAvailable: the difference is what their weights and workspaces leave free.

Only a running container is read: a stopped one (production's, before an arm replaced it)
keeps its own log. The KV line is vLLM's figure; scripts/kv_capacity.py says what the pool
holds.
"""
import argparse
import json
import re
import subprocess

NODES = ("dgx1", "dgx2", "dgx3", "dgx4")
# bin/spark's serving names: the container since the rename, and every dsv41-* before it.
SERVING = ("^spark-ds41f$", "^dsv41-")
STARTUP = "Model loading took|GPU KV cache size|CUDA graph"
LINES = re.compile(r"Model loading took|KV cache size|CUDA graph|memory profiling|weights take|GiB")


def run(node: str, command: str) -> str:
    result = subprocess.run(["ssh", "-o", "ConnectTimeout=10", node, command], capture_output=True, text=True,
                            timeout=120, check=False)
    return result.stdout


def serving_container(node: str) -> dict | None:
    """The node's running serving container: its name, image and creation time."""
    filters = " ".join(f"--filter name='{pattern}'" for pattern in SERVING)
    rows = run(node, f"docker ps {filters} --format '{{{{.Names}}}}\t{{{{.Image}}}}\t{{{{.CreatedAt}}}}'")
    found = [line.split("\t") for line in rows.splitlines() if line.count("\t") == 2]
    if len(found) != 1:
        return None
    name, image, created = found[0]
    return {"name": name, "image": image, "created": created}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    for node in NODES:
        meminfo = run(node, "grep -E '^(MemAvailable|MemTotal):' /proc/meminfo")
        kib = {k: int(v) for k, v in re.findall(r"^(\w+):\s+(\d+) kB", meminfo, re.MULTILINE)}
        container = serving_container(node)
        # The startup lines come first; "GPU KV cache usage" lines later do not match.
        logs = run(node, f"docker logs {container['name']} 2>&1 | grep -E '{STARTUP}' | head -12") if container else ""
        print(json.dumps({
            "label": args.label, "node": node,
            "mem_available_gib": round(kib.get("MemAvailable", 0) / 2**20, 3),
            "mem_total_gib": round(kib.get("MemTotal", 0) / 2**20, 3),
            "container": container,
            "log": [line[-220:] for line in logs.splitlines() if LINES.search(line)],
        }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
