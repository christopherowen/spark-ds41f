#!/usr/bin/env python3
"""Compare two long_probe outputs phase by phase (e.g. unfixed vs fixed boot). usage: long_compare.py A.json B.json"""
import json, sys
a, b = (json.load(open(p))["phases"] for p in sys.argv[1:3])
for ph in ("cold", "solo", "conc1", "conc2", "solo2"):
    out = []
    for x, y in zip(a[ph], b[ph]):
        m = min(len(x["tokens"]), len(y["tokens"])); f = next((k for k in range(m) if x["tokens"][k] != y["tokens"][k]), None)
        u = m if f is None else f; l = next((k for k in range(u) if x["lp"][k] != y["lp"][k]), None)
        out.append((f, l))
    print(f"{ph:6s} A vs B: tokens identical {sum(f is None for f, _ in out)}/{len(out)}; (fork, first lp diff) per prompt {out}")
