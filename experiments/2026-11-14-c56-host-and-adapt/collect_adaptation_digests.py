#!/usr/bin/env python3
"""Whether every rank's online-adaptation state is the same: the digests the ranks logged.

usage: collect_adaptation_digests.py BASE_URL --nodes HOST [HOST ...] [--container NAME]

With VLLM_DSPARK_ONLINE_DIGEST_STEPS=N each rank's worker logs, at load and every N
updates, a bit digest of its adapted Markov W1 and confidence head
("DSpark online adaptation digest: rank=R update=U conf=... w1=... drift=..."). This
reads those lines from the serving container's log on every node ("localhost" is read
without ssh), one JSON line per node, then the summary: for every update count logged
by all ranks, whether their digests are equal. BASE_URL is unused (lab.py passes it).
An arm without adaptation logs nothing and reports "digests": 0.
"""
import argparse
import json
import re
import subprocess

LINE = re.compile(
    r"DSpark online adaptation digest: rank=(\d+) update=(\d+) (.*?) drift=(\S+)")


def node_log(node: str, container: str) -> str:
    command = ["docker", "logs", container]
    if node != "localhost":
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node,
                   "docker", "logs", container]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300)
    return done.stdout + done.stderr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--container", default="spark-ds41f")
    args = parser.parse_args()
    by_update: dict[int, dict[int, str]] = {}
    drift: dict[int, dict[int, float]] = {}
    total = 0
    for node in args.nodes:
        lines = []
        for match in LINE.finditer(node_log(node, args.container)):
            rank, update, state, value = match.groups()
            rank, update = int(rank), int(update)
            by_update.setdefault(update, {})[rank] = state
            drift.setdefault(update, {})[rank] = float(value)
            lines.append({"rank": rank, "update": update, "state": state,
                          "drift": float(value)})
        total += len(lines)
        print(json.dumps({"node": node, "digests": len(lines), "lines": lines}), flush=True)
    ranks = {rank for states in by_update.values() for rank in states}
    complete = {u: s for u, s in sorted(by_update.items()) if set(s) == ranks}
    differing = [u for u, s in complete.items() if len(set(s.values())) > 1]
    last = max(complete) if complete else None
    print(json.dumps({
        "summary": "online adaptation digests",
        "digests": total,
        "ranks": sorted(ranks),
        "updates_compared": len(complete),
        "last_update": last,
        "identical": (not differing) if complete else None,
        "differing_updates": differing,
        "last_state": complete[last] if last is not None else None,
        "last_drift": drift[last] if last is not None else None,
    }), flush=True)


if __name__ == "__main__":
    main()
