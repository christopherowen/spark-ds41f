#!/usr/bin/env python3
"""a968.py OUTDIR: summary of PR #968's own-mark runs (PR screen + h968 harness + tests), ABBA on sp8."""
import json
import os
import re
import sys

O = sys.argv[1]
ARMS = ["base-boot1", "c52-boot1", "c52-boot2", "base-boot2"]


def jsonl(p):
    p = os.path.join(O, p)
    if not os.path.exists(p):
        return None
    recs = {}
    for line in open(p):
        line = line.strip()
        if line.startswith("{") and '"case"' in line:
            r = json.loads(line)
            recs[r["case"]] = r
    return recs


S = {"screen": {}, "h968": {}, "pytest": {}}
lines = []
scr = {a: jsonl(f"screen-{a}.txt") for a in ARMS}
cases = []
for a in ARMS:
    if scr[a]:
        cases = list(scr[a])
        break
for case in cases:
    row = {a: {k: v for k, v in scr[a][case].items() if k != "case"} for a in ARMS if scr[a] and case in scr[a]}
    S["screen"][case] = row
    lines.append(f"{case}:")
    for a, v in row.items():
        lines.append(f"   {a}: {json.dumps(v, sort_keys=True)}")
for a in ARMS:
    p = os.path.join(O, f"h968-{a}.json")
    if os.path.exists(p):
        d = json.load(open(p))
        S["h968"][a] = {k: d[k] for k in ("scheduler_py_sha256", "scheduler_has_max_decode_new_tokens", "emitted_and_metrics_digest",
                                           "emitted_only_digest", "total_rows_adaptive_off", "total_rows_adaptive_on")}
        S["h968"][a]["per_config_rows"] = {f"{'async' if c['async'] else 'sync'}-mt{c['max_tokens']}-{'on' if c['adaptive'] else 'off'}": c["rows"]
                                           for c in d["configs"]}
        S["h968"][a]["_emitted"] = {f"{'async' if c['async'] else 'sync'}-mt{c['max_tokens']}-{'on' if c['adaptive'] else 'off'}": [c["emitted"], c["sink_metrics"]]
                                    for c in d["configs"]}
if S["h968"]:
    keys = next(iter(S["h968"].values()))["_emitted"].keys()
    mism = []
    ref = S["h968"].get("base-boot1") or next(iter(S["h968"].values()))
    for k in keys:
        for a, v in S["h968"].items():
            if v["_emitted"][k] != ref["_emitted"][k]:
                mism.append([a, k])
    S["h968_cross_arm_emitted_and_metrics_identical"] = not mism
    S["h968_mismatches"] = mism
    # within-arm: adaptive off vs on emitted identical
    S["h968_within_arm_off_on_identical"] = {
        a: all(v["_emitted"][k.replace("-on", "-off")] == v["_emitted"][k] for k in keys if k.endswith("-on")) for a, v in S["h968"].items()}
    for v in S["h968"].values():
        del v["_emitted"]
    lines.append("h968 digests: " + json.dumps({a: [v["emitted_and_metrics_digest"], v["total_rows_adaptive_off"], v["total_rows_adaptive_on"]]
                                               for a, v in S["h968"].items()}))
    lines.append(f"h968 cross-arm identical: {S['h968_cross_arm_emitted_and_metrics_identical']} mismatches {mism[:5]}; "
                 f"within-arm off==on: {S['h968_within_arm_off_on_identical']}")
for k in (1, 2):
    p = os.path.join(O, f"pytest-c52-boot{k}.txt")
    if os.path.exists(p):
        t = open(p).read()
        m = re.findall(r"^=*\s*(\d+ passed.*?) in [\d.]+s", t, re.M) or re.findall(r"(\d+ (?:passed|failed).*?) in [\d.]+s", t)
        S["pytest"][f"c52-boot{k}"] = m[-1] if m else t.strip().splitlines()[-1]
lines.append("pytest: " + json.dumps(S["pytest"]))
print("\n".join(lines))
json.dump(S, open(os.path.join(O, "summary.json"), "w"), indent=1)
open(os.path.join(O, "summary.txt"), "w").write("\n".join(lines) + "\n")
