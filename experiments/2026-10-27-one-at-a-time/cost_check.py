#!/usr/bin/env python3
"""cost-check.json's table: each arm's boots against candidate 9's, the boot the sample.

usage: cost_check.py [RUN]   (default one-at-a-time-cost-check, from the repository root)

Labels are <arm>-<boot>. Per boot: the decode grid's geometric mean against the first
candidate 9 boot (step time, lower is better) and each prefill size's mean rate (higher
is better). Per arm: the change of its boots' mean against candidate 9's, with a Welch
interval over boots. An arm without a fix that cost it shows the fix's cost as a gain;
null-overlay shows what being an overlay arm alone does.
"""

import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from servebench import engine, stats  # noqa: E402


def main() -> int:
    run = sys.argv[1] if len(sys.argv) > 1 else "one-at-a-time-cost-check"
    base = ROOT / "results/private/suites" / run
    boots = sorted(path.parent for path in base.glob("*/suite.json"))
    if not boots:
        raise SystemExit(f"no boots in {base}")
    grids = {boot.name: json.loads((boot / "grid.json").read_text()) for boot in boots
             if (boot / "grid.json").exists()}
    reference = grids.get("candidate-9-a") or next(iter(grids.values()))
    metrics: dict[str, dict[str, float]] = {}
    for boot in boots:
        row = {}
        if boot.name in grids:
            row["grid"] = 100 * (engine.boot_ratios([grids[boot.name]], reference)[0] - 1)
        if (boot / "prefill.json").exists():
            for size in json.loads((boot / "prefill.json").read_text())["prefill"]:
                row[f"{size['size'] // 1000}K"] = size["tps"]["mean"]
        metrics[boot.name] = row
    names = sorted({key for row in metrics.values() for key in row}, key=lambda k: (k != "grid", len(k), k))
    print("boot".ljust(20) + "".join(f"{n:>12}" for n in names))
    for label, row in metrics.items():
        print(label.ljust(20) + "".join(f"{row[n]:>+11.2f}%" if n == "grid" else f"{row[n]:>12.1f}"
                                        for n in names if n in row))
    sets = collections.defaultdict(list)
    for label in metrics:
        sets[label.rsplit("-", 1)[0]].append(label)
    print("\narm against candidate 9 (boots: mean change ±95% Welch interval, verdict)")
    for arm, labels in sets.items():
        if arm == "candidate-9":
            continue
        cells = []
        for n in names:
            a = [metrics[l][n] for l in sets["candidate-9"] if n in metrics[l]]
            b = [metrics[l][n] for l in labels if n in metrics[l]]
            if n == "grid":  # already % of the reference; as step-time ratios around 100
                a, b = [100 + v for v in a], [100 + v for v in b]
            test = stats.welch(a, b)
            if test is None:
                cells.append(f"{n}: {len(b)} boots")
                continue
            verdict = stats.verdict(test["change_pct"], test["ci95_pct"], n != "grid", 0.0)
            cells.append(f"{n} {test['change_pct']:+.2f}% ±{test['ci95_pct']:.2f}% {verdict}")
        print(f"  {arm} ({len(labels)} boots): " + "; ".join(cells))
    c9 = sets["candidate-9"]
    for n in names:
        values = [metrics[l][n] for l in c9 if n in metrics[l]]
        if len(values) > 1:
            sd = statistics.stdev(values) if n == "grid" else 100 * statistics.stdev(values) / statistics.fmean(values)
            print(f"candidate 9 boot-to-boot sd, {n}: {sd:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
