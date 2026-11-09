"""#977 confirmation analysis: confirm-<arm>-boot<k>.jsonl (the PR bench at 4K/16K/32K, 201 replays)."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__)); from common import cell, jl
d = sys.argv[1]
def load(p): return {(r["scenario"], r["reqs"], r["ctx"]): r for r in jl(p)}
b1, b2 = load(f"{d}/confirm-base-boot1.jsonl"), load(f"{d}/confirm-base-boot2.jsonl")
c1, c2 = load(f"{d}/confirm-c53-boot1.jsonl"), load(f"{d}/confirm-c53-boot2.jsonl")
cells = {f"{k[0]}/c{k[1]}/{k[2]}": cell([b1[k]["med_us"], b2[k]["med_us"]], [c1[k]["med_us"], c2[k]["med_us"]]) for k in sorted(b1)}
summ = {v: sum(c["verdict"] == v for c in cells.values()) for v in ("better", "same", "worse")}
json.dump({"cells": cells, "summary": summ}, open(f"{d}/confirm-analysis.json", "w"), indent=1)
print(summ)
