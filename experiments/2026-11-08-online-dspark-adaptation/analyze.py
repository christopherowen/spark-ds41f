#!/usr/bin/env python3
"""The run's tables from its receipts: acceptance per turn, held-out, c1 rate, gates, digests.

usage: analyze.py RECEIPTS_DIR RUN [--labels ON ON2 OFF OFF2] [--json OUT]

RECEIPTS_DIR is what `bin/spark candidate receipts` wrote (its determinism/RUN/ holds
each arm's workload, gate and digest outputs). Prints Markdown tables; with --json,
also writes the per-arm numbers. Means over both boots of an arm, with each boot.
"""
import argparse
import json
import statistics
from pathlib import Path


def lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if raw.startswith("{"):
            out.append(json.loads(raw))
    return out


def summary(rows: list[dict], name: str) -> dict | None:
    return next((r for r in rows if r.get("summary") == name), None)


def pct(a: float | None, b: float | None) -> str:
    if a is None or b is None or not b:
        return "n/a"
    return f"{100 * (a - b) / b:+.2f}%"


def fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("receipts")
    parser.add_argument("run")
    parser.add_argument("--labels", nargs=4,
                        default=["adapt-on", "adapt-on-again", "adapt-off", "adapt-off-again"])
    parser.add_argument("--json")
    args = parser.parse_args()
    root = Path(args.receipts) / "determinism" / args.run
    on, on2, off, off2 = args.labels
    arms = {"on": [on, on2], "off": [off, off2]}
    data = {}
    for label in args.labels:
        workload = lines(root / f"workload-{label}.jsonl")
        data[label] = {
            "rows": [r for r in workload if "set" in r],
            "coding": summary(workload, "coding conversations"),
            "heldout": summary(workload, "held-out prompts"),
            "total": summary(workload, "online adaptation workload"),
            "determinism": summary(lines(root / f"determinism-{label}.jsonl"),
                                   "temperature-0 determinism"),
            "greedy": summary(lines(root / f"greedy-{label}.jsonl"),
                              "temperature-0 determinism"),
            "digests": summary(lines(root / f"digests-{label}.jsonl"),
                               "online adaptation digests"),
        }

    def arm_mean(arm: str, get) -> float | None:
        return mean([get(data[label]) for label in arms[arm] if data[label]["coding"]])

    turns = sorted({r["turn"] for d in data.values() for r in d["rows"] if r["set"] == "coding"})
    print("## Acceptance length per turn (coding, pooled over the four conversations)\n")
    print("| Turn | " + " | ".join(args.labels) + " | on mean | off mean | on vs off |")
    print("| --- | " + " | ".join("---:" for _ in args.labels) + " | ---: | ---: | ---: |")
    for t in turns:
        def per_turn(d, t=t):
            return (d["coding"] or {}).get("acceptance_length_per_turn", {}).get(str(t))
        cells = [fmt(per_turn(data[label])) for label in args.labels]
        a, b = arm_mean("on", per_turn), arm_mean("off", per_turn)
        print(f"| {t + 1} | " + " | ".join(cells) + f" | {fmt(a)} | {fmt(b)} | {pct(a, b)} |")

    def row(name, get, digits=3):
        cells = [fmt(get(data[label]), digits) for label in args.labels]
        a, b = arm_mean("on", get), arm_mean("off", get)
        print(f"| {name} | " + " | ".join(cells) + f" | {fmt(a, digits)} | {fmt(b, digits)} | {pct(a, b)} |")

    print("\n## Totals\n")
    print("| | " + " | ".join(args.labels) + " | on mean | off mean | on vs off |")
    print("| --- | " + " | ".join("---:" for _ in args.labels) + " | ---: | ---: | ---: |")
    row("coding acceptance length", lambda d: (d["coding"] or {}).get("acceptance_length"))
    row("held-out acceptance length", lambda d: (d["heldout"] or {}).get("acceptance_length"))
    row("coding c1 decode tok/s", lambda d: (d["coding"] or {}).get("tps_mean"), 2)
    row("held-out c1 decode tok/s", lambda d: (d["heldout"] or {}).get("tps_mean"), 2)

    print("\n## Text, gates and rank state\n")
    print("| Arm | workload text | gate (log-probs) ref / long | gate (text) ref / long | identical | ranks' digests |")
    print("| --- | --- | --- | --- | --- | --- |")
    for label in args.labels:
        d = data[label]
        text = (d["total"] or {}).get("text_sha256", "n/a")[:12]
        det, gre, dig = d["determinism"] or {}, d["greedy"] or {}, d["digests"] or {}
        identical = f"{det.get('identical')} / {gre.get('identical')}"
        if dig.get("digests"):
            digest = (f"{'equal' if dig.get('identical') else 'DIFFER'} at "
                      f"{dig.get('updates_compared')} update counts, last {dig.get('last_update')}")
        else:
            digest = "none logged"
        print(f"| {label} | `{text}` | `{str(det.get('reference_sha256'))[:8]}` / "
              f"`{str(det.get('long_sha256'))[:8]}` | `{str(gre.get('reference_sha256'))[:8]}` / "
              f"`{str(gre.get('long_sha256'))[:8]}` | {identical} | {digest} |")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {label: {k: v for k, v in d.items() if k != "rows"} | {"rows": d["rows"]}
             for label, d in data.items()}, indent=1) + "\n")


if __name__ == "__main__":
    main()
