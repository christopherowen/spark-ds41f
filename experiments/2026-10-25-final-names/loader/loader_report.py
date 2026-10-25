#!/usr/bin/env python3
"""How each node's serving container loaded its weights (a lab extras script).

usage: loader_report.py URL --label LABEL [--nodes dgx1,dgx2,dgx3]

For every node named: the running serving container (bin/spark's names: spark-ds41f since
the rename, dsv41-* before it), its weight-loading lines (vLLM's "Loading weights took" and
"Model loading took", the loader's own summary, and the per-rank weight digest when the arm
sets SPARK_DS41F_WEIGHT_DIGEST=1) with their seconds after the container started, the
seconds to "Application startup complete" (the head's API), the bytes its cgroup has read
from disk since it started, and the node's memory now (MemAvailable, MemFree and the page
cache, Cached). One JSON line per node, then a summary line. Run it first among an arm's
extras: Engram rows read while serving add to the cgroup's bytes.
"""
import argparse
import datetime as dt
import json
import re
import subprocess

# bin/spark's serving names: the container since the rename, and every dsv41-* before it.
SERVING = ("^spark-ds41f$", "^dsv41-")
LOADING = re.compile(r"Loading weights took|Model loading took|weights loaded|weight digest|Application startup complete")
DIGEST = re.compile(r"weight digest (?:rank (\d+) )?([0-9a-f]{64})")


def run(node: str, command: str) -> str:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, command],
                            capture_output=True, text=True, timeout=180, check=False)
    return result.stdout


def serving_container(node: str) -> dict | None:
    """The node's running serving container: a stopped one is production's, kept by the lab."""
    filters = " ".join(f"--filter name='{pattern}'" for pattern in SERVING)
    rows = run(node, f"docker ps {filters} --format '{{{{.Names}}}}\t{{{{.ID}}}}\t{{{{.Image}}}}'")
    found = [line.split("\t") for line in rows.splitlines() if line.count("\t") == 2]
    if len(found) != 1:
        return None
    name, short_id, image = found[0]
    full, _, started = run(node, f"docker inspect -f '{{{{.Id}}}} {{{{.State.StartedAt}}}}' {name}").strip().partition(" ")
    return {"name": name, "id": full or short_id, "image": image, "started": started}


def instant(text: str) -> dt.datetime | None:
    """A docker timestamp (RFC 3339, nanoseconds) to the microsecond."""
    found = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)", text)
    if not found:
        return None
    zone = "+00:00" if found.group(3) == "Z" else found.group(3)
    return dt.datetime.fromisoformat(f"{found.group(1)}.{(found.group(2) or '0')[:6]:0<6}{zone}")


def timed_lines(logs: str, started: str) -> list[dict]:
    """``docker logs -t`` lines as seconds after the container started, and the text."""
    origin, lines = instant(started), []
    for line in logs.splitlines():
        stamp, _, text = line.partition(" ")
        at = instant(stamp)
        seconds = None if at is None or origin is None else round((at - origin).total_seconds(), 1)
        lines.append({"seconds": seconds, "text": text[-240:]})
    return lines


def read_bytes(io_stat: str) -> int | None:
    """Bytes read from every device, from a cgroup v2 io.stat."""
    values = [int(value) for value in re.findall(r"\brbytes=(\d+)", io_stat)]
    return sum(values) if values else None


def meminfo_gib(text: str) -> dict:
    kib = {key: int(value) for key, value in re.findall(r"^(\w+):\s+(\d+) kB", text, re.MULTILINE)}
    return {f"{key.lower()}_gib": round(kib[key] / 2**20, 3) for key in ("MemAvailable", "MemFree", "Cached")
            if key in kib}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--label", required=True)
    parser.add_argument("--nodes", default="dgx1,dgx2,dgx3,dgx4")
    args = parser.parse_args()
    digests, read = {}, {}
    for node in args.nodes.split(","):
        container = serving_container(node)
        record = {"label": args.label, "node": node, "container": container,
                  **meminfo_gib(run(node, "grep -E '^(MemAvailable|MemFree|Cached):' /proc/meminfo"))}
        if container is not None:
            logs = run(node, f"docker logs -t {container['name']} 2>&1 | grep -E '{LOADING.pattern}' | tail -16")
            record["log"] = timed_lines(logs, container["started"])
            record["digests"] = [found.group(2) for found in (DIGEST.search(line["text"]) for line in record["log"])
                                 if found]
            ready = [line["seconds"] for line in record["log"] if "Application startup complete" in line["text"]]
            record["api_ready_seconds"] = ready[0] if ready else None
            stat = run(node, f"cat /sys/fs/cgroup/system.slice/docker-{container['id']}.scope/io.stat 2>/dev/null")
            record["cgroup_read_gib"] = (None if read_bytes(stat) is None
                                         else round(read_bytes(stat) / 2**30, 3))
            digests[node], read[node] = record["digests"], record["cgroup_read_gib"]
        print(json.dumps(record), flush=True)
    print(json.dumps({"summary": "weight loading", "label": args.label, "digests": digests,
                      "cgroup_read_gib": read}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
