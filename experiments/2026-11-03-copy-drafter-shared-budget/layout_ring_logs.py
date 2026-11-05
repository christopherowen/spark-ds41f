#!/usr/bin/env python3
"""The layout ring of the booted arm: every rank's per-step records, compared.

usage (a lab extra, run on the head node from the repository root, after an arm's
workload; the files outlive the engine, so it reads a crashed boot too):
  layout_ring_logs.py HEAD_URL LABEL

The arm mounts overlay/layout-ring's model_runner.py, which writes one 40-byte record
a prepared step to /cache/spark-ds41f/layout-ring/rank<r>.bin on its node: the step,
the token count and its padded count, the request count, CRCs of the compacted token
counts, the scheduler's drafts and the budget's eligible drafts per request, the draft
budget, and the copy drafter's cumulative budget reads and stale reads. The nodes are
the window's: its hold records the production profile, whose node map the arms share.
One JSON line a node (its rank's record count and last record), then one line that
compares the ranks at equal steps: the steps whose layouts differ, by field, the first
of them with its neighbours, and each rank's last records.
"""
import base64
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import spark_ds41f as spark  # noqa: E402
from spark_ds41f import lab  # noqa: E402

RING_DIR = "cache/spark-ds41f/layout-ring"
RECORD = struct.Struct("<IiiiIIIiII")
FIELDS = ("step", "tokens", "padded", "reqs", "scheduled_crc", "drafts_crc", "eligible_crc",
          "draft_budget", "reads", "stale")
LAYOUT = ("tokens", "padded", "reqs", "scheduled_crc", "drafts_crc", "eligible_crc", "draft_budget")
CONTEXT = 4


def decode(blob: bytes) -> list[dict]:
    """The records of one rank's file; a partial trailing record is dropped."""
    count = len(blob) // RECORD.size
    return [dict(zip(FIELDS, RECORD.unpack_from(blob, i * RECORD.size))) for i in range(count)]


def compare(by_rank: dict[int, list[dict]]) -> dict:
    """The ranks' records at equal steps: which steps' layouts differ, and how."""
    ranks = sorted(by_rank)
    common = min((len(by_rank[r]) for r in ranks), default=0)
    divergent, kinds, stale_mismatch = [], dict.fromkeys(LAYOUT, 0), 0
    for i in range(common):
        rows = [by_rank[r][i] for r in ranks]
        differs = [f for f in LAYOUT if len({row[f] for row in rows}) > 1]
        for f in differs:
            kinds[f] += 1
        if differs:
            divergent.append(i)
        if len({row["stale"] - row["reads"] for row in rows}) > 1:
            stale_mismatch += 1

    def window(center: int) -> dict:
        lo, hi = max(0, center - CONTEXT), min(common, center + CONTEXT + 1)
        return {str(r): by_rank[r][lo:hi] for r in ranks}

    return {
        "ranks": ranks,
        "records": {str(r): len(by_rank[r]) for r in ranks},
        "common": common,
        "divergent": len(divergent),
        "kinds": kinds,
        "stale_mismatch": stale_mismatch,
        "first": divergent[0] if divergent else None,
        "first_context": window(divergent[0]) if divergent else {},
        "divergent_steps": divergent[:64],
        "tail": {str(r): by_rank[r][-CONTEXT * 2:] for r in ranks},
    }


def fetch(nodes: dict, node: dict, repo: str) -> dict[int, list[dict]]:
    script = (f'for f in {repo}/{RING_DIR}/rank*.bin; do [ -f "$f" ] || continue; '
              'echo "FILE $f"; base64 -w0 "$f"; echo; done')
    result = spark.run_ssh(nodes, node, "bash", "-lc", script)
    files, name = {}, None
    for line in (result.stdout or "").splitlines():
        if line.startswith("FILE "):
            name = line[5:].strip()
        elif name:
            rank = int(Path(name).stem.removeprefix("rank"))
            files[rank] = decode(base64.b64decode(line))
            name = None
    return files


def main(label: str) -> None:
    # The window's production profile (its hold record), as lab.py resolves it: the node
    # map of the cluster the arm runs on, not the site default's.
    lab.PRODUCTION_CONFIG = lab.resolve_production(None, lab.read_hold())
    cluster, nodes, _ = lab.production_configuration()
    repo = spark.repository_path(cluster)
    by_rank: dict[int, list[dict]] = {}
    for node in nodes["nodes"]:
        files = fetch(nodes, node, repo)
        for rank, records in files.items():
            by_rank[rank] = records
            print(json.dumps({"label": label, "node": node["name"], "rank": rank,
                              "records": len(records), "last": records[-1] if records else None}),
                  flush=True)
        if not files:
            print(json.dumps({"label": label, "node": node["name"], "rank": None, "records": 0,
                              "error": (result_error(nodes, node, repo))}), flush=True)
    print(json.dumps({"label": label, "compare": compare(by_rank)}), flush=True)


def result_error(nodes: dict, node: dict, repo: str) -> str:
    result = spark.run_ssh(nodes, node, "ls", "-la", f"{repo}/{RING_DIR}")
    return (result.stdout or "") + (result.stderr or "")


if __name__ == "__main__":
    main(sys.argv[2])
