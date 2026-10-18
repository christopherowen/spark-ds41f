#!/usr/bin/env python3
"""How far repeated DSpark cost profiles of one build disagree.

usage: compare_profiles.py COST_DIR... [--label NAME]

Each COST_DIR holds the pinned curves (dspark-costs-*.json) of one fresh profile of the
same build. For the verification curve and the sixteen-request curve, prints each
profile's slope between neighbouring decode shapes (1-6, 6-16, 16-48, 48-96 rows; ms a
row) and the spread across profiles, (max - min) / mean, per slope; then the largest
spread. The draft budget prices extra rows from these slopes, so their spread is the
noise it acts on.
"""
import argparse
import glob
import json
import os
import statistics

PAIRS = ((1, 6), (6, 16), (16, 48), (48, 96))


def load(directory: str) -> dict:
    paths = glob.glob(os.path.join(os.path.expanduser(directory), "dspark-costs-*.json"))
    if len(paths) != 1:
        raise SystemExit(f"{directory}: expected one pinned profile, found {len(paths)}")
    with open(paths[0]) as handle:
        return json.load(handle)


def slopes(curve: list) -> dict:
    points = {int(x): float(y) for x, y in curve}
    return {f"{a}-{b}": (points[b] - points[a]) / (b - a) for a, b in PAIRS if a in points and b in points}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    profiles = [load(d) for d in args.dirs]
    worst = 0.0
    for name, pick in (("verify", lambda p: p["verify_curve"]),
                       ("verify, 16 requests", lambda p: p["verify_curves_by_num_reqs"].get("16", []))):
        rows = [slopes(pick(p)) for p in profiles]
        keys = [k for k in rows[0] if all(k in r for r in rows)]
        print(f"{args.label} {name}: replays {profiles[0].get('profile_replays')}")
        for key in keys:
            values = [r[key] for r in rows]
            spread = (max(values) - min(values)) / statistics.fmean(values)
            worst = max(worst, spread)
            print(f"  {key:>6} rows: " + "  ".join(f"{v:7.4f}" for v in values) + f"   spread {spread:6.2%}")
    print(f"{args.label} largest slope spread: {worst:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
