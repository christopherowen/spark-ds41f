#!/usr/bin/env python3
"""The nonfinite census of the booted arm: every rank's NONFINITE log lines.

usage (a lab extra, run on the head node from the repository root, last of an arm's
extras, before the next boot replaces the container):
  census_logs.py HEAD_URL LABEL

The census image logs a row whose MoE gate logits were not finite instead of
failing route_partials' assert (vLLM cb46d6c82). One JSON line a node: the count of
NONFINITE lines, the first 400 of them, and any assert, CUDA error or traceback line.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import spark_ds41f as spark  # noqa: E402
from spark_ds41f import lab  # noqa: E402

TROUBLE = re.compile(r"cudaErrorAssert|device-side assert|nan or inf|Traceback|EngineDeadError")


def main(label: str) -> None:
    nodes, container = lab.nodes_config(), lab.container_name()
    for node in nodes["nodes"]:
        result = spark.run_ssh(nodes, node, "docker", "logs", container)
        text = (result.stdout or "") + (result.stderr or "")
        census = [line for line in text.splitlines() if "NONFINITE" in line]
        trouble = [line for line in text.splitlines() if TROUBLE.search(line)]
        print(json.dumps({"label": label, "node": node["name"], "nonfinite": len(census),
                          "lines": census[:400], "trouble": trouble[:50]}), flush=True)


if __name__ == "__main__":
    main(sys.argv[2])
