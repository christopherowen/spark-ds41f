#!/usr/bin/env python3
"""The host limit reads of the booted arm: every rank's COPY-LIMITS log lines.

usage (a lab extra, run on the head node from the repository root, after an arm's
workload and before the next boot replaces the container):
  limit_reads_logs.py HEAD_URL LABEL

The arms mount overlay/limit-reads' copy_drafts.py, which counts the adaptive
budget's reads of the host limits that found the latest limit copy in flight, and
what those reads changed once the copy landed. The nodes are the window's: its hold
records the production profile, whose node map the arms share. One JSON line a node:
its last COPY-LIMITS line parsed, the crc of each logged window, and any assert, CUDA
error or traceback line.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import spark_ds41f as spark  # noqa: E402
from spark_ds41f import lab  # noqa: E402

TROUBLE = re.compile(r"cudaErrorAssert|device-side assert|nan or inf|Traceback|EngineDeadError")
FIELD = re.compile(r"(\w+)=([0-9a-f]+)")


def main(label: str) -> None:
    # The window's production profile (its hold record), as lab.py resolves it: the node
    # map of the cluster the arm runs on, not the site default's.
    lab.PRODUCTION_CONFIG = lab.resolve_production(None, lab.read_hold())
    nodes, container = lab.nodes_config(), lab.container_name()
    for node in nodes["nodes"]:
        result = spark.run_ssh(nodes, node, "docker", "logs", container)
        text = (result.stdout or "") + (result.stderr or "")
        reads = [line for line in text.splitlines() if "COPY-LIMITS" in line]
        windows = [dict(FIELD.findall(line.split("COPY-LIMITS", 1)[1])) for line in reads]
        last = {k: v if k == "crc" else int(v) for k, v in windows[-1].items()} if windows else {}
        trouble = [line for line in text.splitlines() if TROUBLE.search(line)]
        print(json.dumps({"label": label, "node": node["name"], "windows": len(windows),
                          "last": last, "crcs": [(w.get("reads"), w.get("crc")) for w in windows],
                          "trouble": trouble[:50]}), flush=True)


if __name__ == "__main__":
    main(sys.argv[2])
