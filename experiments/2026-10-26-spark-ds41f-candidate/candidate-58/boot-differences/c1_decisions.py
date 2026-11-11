"""Per arm: do the boots' per-step acceptance sequences agree, request by request?"""
import json, os, re, sys
from itertools import combinations

d = sys.argv[1]
arms = {}
for name in sorted(os.listdir(d)):
    m = re.fullmatch(r"(c1-.*)-boot(\d+)", name)
    if m and os.path.exists(f"{d}/{name}/c1.json"):
        arms.setdefault(m.group(1), {})[int(m.group(2))] = f"{d}/{name}/c1.json"

def seqs(p):
    out = []
    for r in json.load(open(p))["requests"]:
        counts = [c for _, c in r["events"]]
        out.append((r["prompt_id"], tuple(b - a for a, b in zip([0] + counts, counts))))
    return out

for arm, boots in arms.items():
    s = {b: seqs(p) for b, p in sorted(boots.items())}
    parts = []
    for a, b in combinations(sorted(s), 2):
        n = min(len(s[a]), len(s[b]))
        first = next((i for i in range(n) if s[a][i] != s[b][i]), None)
        if first is None:
            parts.append(f"b{a}=b{b} ({n} req)")
        else:
            pa, sa = s[a][first]; pb, sb = s[b][first]
            step = next((k for k in range(min(len(sa), len(sb))) if sa[k] != sb[k]), min(len(sa), len(sb)))
            diff = sum(1 for i in range(n) if s[a][i] != s[b][i])
            parts.append(f"b{a}!=b{b} first req {first} ({pa}{'' if pa == pb else '/' + pb}) step {step}, {diff}/{n} req differ")
    print(f"{arm}: " + "; ".join(parts))
