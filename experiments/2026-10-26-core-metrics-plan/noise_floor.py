#!/usr/bin/env python3
"""Each release metric's noise floor, from the same-session repeats the release tables already hold.

usage: noise_floor.py [RELEASE_TABLE ...] [--json OUT]

With no files, reads every manifests/suites/*/release-table.txt. A release comparison runs the
candidate, the previous release and the candidate again in one session, so each table's last
block ("######## X -> X-again") compares one build with itself: an A/A pair. For every metric
it collects those A/A changes across releases and reports:

- n: how many A/A pairs measured it;
- the median and largest |change| (the noise floor a verdict has to clear);
- the median 95% interval the table printed for it (servebench's batch-means or Welch
  interval, or none for percentile rows judged against the 3% tolerance);
- false calls: A/A pairs whose verdict was not "same", that is how often the metric's own
  interval called a build better or worse than itself.

A metric whose A/A changes exceed its printed interval needs the repeat floor (the verdict
block's "repeat" column) or more samples; one whose changes sit far inside it could be judged
on fewer windows.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROW = re.compile(r"^(?P<metric>\S.*?)\s+(?P<base>-?\d[\d.]*)\s+(?P<cand>-?\d[\d.]*)\s+(?P<change>[+-]\d[\d.]*)%"
                 r"(?:\s+(?P<ci>\d[\d.]*)%)?\s+(?P<verdict>(?:better|worse|same)(?: \(no interval\))?)\s*$")
GEOMEAN = re.compile(r"^(?P<metric>decode geometric mean \(\d+\))\s+(?P<change>[+-]\d[\d.]*)%\s+(?P<ci>\d[\d.]*)%\s+"
                     r"(?P<verdict>better|worse|same)\s*$")
BLOCK = re.compile(r"^######## (?P<base>.+?) -> (?P<cand>.+?)\s*$")


def suite_kind(path: Path) -> str:
    name = path.parent.name
    for kind in ("single-stream", "sampled"):
        if name.endswith(kind):
            return kind
    return "release"


def aa_rows(path: Path) -> list[dict]:
    """The rows of the table's A/A block (a build against its own repeat), if it has one."""
    rows, step, inside = [], None, False
    for line in path.read_text().splitlines():
        block = BLOCK.match(line)
        if block:
            base, cand = block["base"], block["cand"]
            inside = cand == f"{base}-again" or cand.startswith(base) and cand.endswith("-again")
            step = None
            continue
        if not inside:
            continue
        if line.startswith("== "):
            step = line[3:].strip()
            continue
        match = ROW.match(line) or GEOMEAN.match(line)
        if match and step:
            rows.append({"step": step, "metric": match["metric"].strip(), "change_pct": float(match["change"]),
                         "ci95_pct": float(match["ci"]) if match["ci"] else None, "verdict": match["verdict"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tables", nargs="*", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    tables = args.tables or sorted((ROOT / "manifests/suites").glob("*/release-table.txt"))
    pooled: dict[tuple[str, str, str], list[dict]] = {}
    sources = []
    for path in tables:
        rows = aa_rows(path)
        if not rows:
            continue
        sources.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
        shape = "tp3" if "tp3" in path.parent.name else "tp4"
        for row in rows:
            pooled.setdefault((suite_kind(path), row["step"], row["metric"]), []).append(row | {"table": path.parent.name,
                                                                                               "shape": shape})
    out = []
    for (kind, step, metric), rows in sorted(pooled.items()):
        changes = [abs(r["change_pct"]) for r in rows]
        cis = [r["ci95_pct"] for r in rows if r["ci95_pct"] is not None]
        false_calls = sum(1 for r in rows if not r["verdict"].startswith("same"))
        out.append({"suite": kind, "step": step, "metric": metric, "n": len(rows),
                    "median_abs_change_pct": statistics.median(changes), "max_abs_change_pct": max(changes),
                    "median_ci95_pct": statistics.median(cis) if cis else None, "false_calls": false_calls,
                    "pairs": [{k: r[k] for k in ("table", "change_pct", "ci95_pct", "verdict")} for r in rows]})
    print(f"A/A blocks read: {len(sources)}")
    for source in sources:
        print(f"  {source}")
    width = max(len(f"{r['step']}: {r['metric']}") for r in out)
    print(f"\n{'suite':14s} {'metric':{width}s} {'n':>3} {'med|A/A|':>9} {'max|A/A|':>9} {'med ±95%':>9} {'false':>6}")
    for r in out:
        if r["step"] == "grid" and r["metric"].startswith("decode ") and "geometric" not in r["metric"]:
            continue  # the twenty shapes are summarized below
        ci = f"{r['median_ci95_pct']:.2f}%" if r["median_ci95_pct"] is not None else "none"
        print(f"{r['suite']:14s} {r['step'] + ': ' + r['metric']:{width}s} {r['n']:3d} {r['median_abs_change_pct']:8.2f}% "
              f"{r['max_abs_change_pct']:8.2f}% {ci:>9} {r['false_calls']:3d}/{r['n']}")
    shapes = [r for r in out if r["step"] == "grid" and r["metric"].startswith("decode ") and "geometric" not in r["metric"]]
    if shapes:
        calls = sum(r["false_calls"] for r in shapes)
        pairs = sum(r["n"] for r in shapes)
        print(f"\ngrid shapes: {len(shapes)} shapes, {pairs} A/A shape pairs; median |A/A| "
              f"{statistics.median(c for r in shapes for c in [abs(p['change_pct']) for p in r['pairs']]):.2f}%, "
              f"median ±95% {statistics.median(p['ci95_pct'] for r in shapes for p in r['pairs'] if p['ci95_pct']):.2f}%, "
              f"called better or worse {calls}/{pairs}")
    if args.json:
        args.json.write_text(json.dumps({"sources": sources, "metrics": out}, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
