#!/usr/bin/env python3
"""For each rank's trace: per shared step, the stages (excluding swaidx/idxidx slot ids) that differ from the solo reference, first few in execution order.
usage: stage_first.py P_ROWS TRACE [TRACE...] [--n 26,28,29] [--show 6]"""
import json, re, sys
p = int(sys.argv[1]); files = [a for a in sys.argv[2:] if not a.startswith("--") and a.endswith(".jsonl")]
want = None
if "--n" in sys.argv: want = {int(x) for x in sys.argv[sys.argv.index("--n") + 1].split(",")}
show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 6
RANK = {"pre": 0, "q": 1, "pos": 2, "swalen": 3, "swaidx": 4, "idxlen": 5, "idxidx": 6, "mla": 7, "attn": 8, "ffnin": 9, "ffn": 10}
SKIP = {"swaidx"}
def order(st):
    m = re.search(r"layers\.(\d+)", st[0]); return (int(m.group(1)) if m else -1, RANK.get(st[1], 99), st[0])
for f in sorted(files, key=lambda f: int(re.search(r"rank(\d+)", f).group(1))):
    rank = int(re.search(r"rank(\d+)", f).group(1))
    steps = [json.loads(l) for l in open(f) if l.strip()]; steps = [s for s in steps if s.get("stages")]
    ref = next((s for s in steps if s["num_reqs"] == 1 and s["n"] == p), None)
    if ref is None: print("rank", rank, "no solo"); continue
    ids = ref["input_ids"][:p]; R = {(st[0], st[1]): st for st in ref["stages"]}; seen = set()
    for s in steps:
        if s is ref or s["input_ids"][:p] != ids or s["qsl"][1] != p: continue
        if want and s["n"] not in want: continue
        key = (s["n"], s["cg_mode"]);
        if key in seen: continue
        seen.add(key); bad = []
        for st in sorted(s["stages"], key=order):
            r = R.get((st[0], st[1]))
            if r is None or st[1] in SKIP: continue
            pat = "".join("." if x == y else "X" for x, y in zip(st[3][:p], r[3][:p]))
            if "X" in pat: bad.append(f"L{order(st)[0]}.{st[1]}:{pat}")
        print(f"rank {rank} n={s['n']:3d} {s['cg_mode'].split('.')[-1]:5s} differing stages {len(bad):3d}: " + "  ".join(bad[:show]))
