#!/usr/bin/env python3
"""The lab's table for a loader run: per boot, how the weights loaded and what the
startup left free; per loader, the means over its boots; and the gates (README, "Lab spec").

usage: tables_arms.py RUN LABEL [LABEL ...]   (labels <loader>-<boot>, e.g. direct-1)
"""
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOK = re.compile(r"Loading weights took ([\d.]+) seconds")
MODEL = re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds")


def lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def summary(rows: list[dict]) -> dict:
    found = [row for row in rows if "summary" in row]
    return found[-1] if found else {}


def boot(run: str, label: str) -> dict:
    out = ROOT / "results/private/determinism" / run
    memory_path = out / f"startup-memory-{label}.json"
    memory = json.loads(memory_path.read_text()) if memory_path.is_file() else {}
    nodes = {}
    for row in lines(out / f"loader-{label}.jsonl"):
        if "node" not in row:
            continue
        texts = [line["text"] for line in row.get("log", [])]
        took = [float(m.group(1)) for m in map(TOOK.search, texts) if m]
        model = [m for m in map(MODEL.search, texts) if m]
        nodes[row["node"]] = {
            "loading_seconds": max(took) if took else None,
            "model_gib": float(model[0].group(1)) if model else None,
            "api_ready_seconds": row.get("api_ready_seconds"),
            "read_gib": row.get("cgroup_read_gib"),
            "cached_gib_after": row.get("cached_gib"),
            "digests": tuple(row.get("digests", [])),
            **{key: memory.get(row["node"], {}).get(key) for key in ("min_available_gib", "min_free_gib", "max_cached_gib")},
        }
    texts = {kind: summary(lines(out / f"{kind}-{label}.jsonl")) for kind in ("determinism", "greedy")}
    speed = {kind: summary(lines(out / f"{kind}-{label}.jsonl")).get("tps_mean") for kind in ("c1-distinct", "c8-distinct")}
    return {"nodes": nodes, "texts": texts, "speed": speed}


def fmt(value, digits=2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> int:
    run, labels = sys.argv[1], sys.argv[2:]
    boots = {label: boot(run, label) for label in labels}
    print(f"loader run {run}: {', '.join(labels)}")
    print(f"{'boot':<16}{'node':<6}{'load s':>8}{'API s':>8}{'read GiB':>10}{'min avail':>11}"
          f"{'min free':>10}{'max cache':>11}{'cache after':>12}")
    for label, data in boots.items():
        for node, row in sorted(data["nodes"].items()):
            print(f"{label:<16}{node:<6}{fmt(row['loading_seconds'], 1):>8}{fmt(row['api_ready_seconds'], 1):>8}"
                  f"{fmt(row['read_gib']):>10}{fmt(row['min_available_gib'], 3):>11}{fmt(row['min_free_gib'], 3):>10}"
                  f"{fmt(row['max_cached_gib']):>11}{fmt(row['cached_gib_after']):>12}")
        print(f"{'':<16}c1 {fmt(data['speed']['c1-distinct'])} tok/s, c8 {fmt(data['speed']['c8-distinct'])} tok/s")
    families = {}
    for label in labels:
        families.setdefault(label.rsplit("-", 1)[0], []).append(label)
    print("\nper loader (mean over its boots; dgx1 is the startup guard's tightest node)")
    for family, members in families.items():
        def values(key, node=None):
            return [row[key] for label in members for name, row in boots[label]["nodes"].items()
                    if (node is None or name == node) and row[key] is not None]
        load, ready = values("loading_seconds"), values("api_ready_seconds")
        dgx1 = values("min_available_gib", "dgx1")
        print(f"  {family}: loading {fmt(statistics.mean(load) if load else None, 1)} s (slowest rank "
              f"{fmt(max(load) if load else None, 1)}), API {fmt(statistics.mean(ready) if ready else None, 1)} s, "
              f"dgx1 lowest MemAvailable {fmt(min(dgx1) if dgx1 else None, 3)} GiB "
              f"(mean {fmt(statistics.mean(dgx1) if dgx1 else None, 3)}), "
              f"page cache after boot up to {fmt(max(values('cached_gib_after') or [0]))} GiB")
    print("\ngates")
    digests = {}
    for label, data in boots.items():
        for node, row in data["nodes"].items():
            digests.setdefault(node, set()).add(row["digests"])
    weights = all(len(found) == 1 and () not in found for found in digests.values()) and bool(digests)
    print(f"  weights bit-identical on every rank across every boot: {'pass' if weights else 'FAIL'}")
    ok = weights
    for kind, key in (("determinism", "reference_sha256"), ("greedy", "reference_text_sha256")):
        found = {data["texts"][kind].get(key) for data in boots.values()}
        same = len(found) == 1 and None not in found
        ok &= same
        print(f"  temperature-0 {kind} digests equal: {'pass' if same else 'FAIL'} {sorted(map(str, found))}")
    floor = [row["min_available_gib"] for data in boots.values() for row in data["nodes"].values()
             if row["min_available_gib"] is not None]
    above = bool(floor) and min(floor) >= 5.0
    ok &= above
    print(f"  every node stayed above the 5 GiB startup guard: {'pass' if above else 'FAIL'} "
          f"(lowest {fmt(min(floor) if floor else None, 3)} GiB)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
