"""Grid cells of step957.py boots: per shape, each boot's median step (ms), and the
interval verdict against the base arm (round 3's rule: faster when the slower candidate
boot beats the faster base boot, slower when the faster candidate boot is slower than the
slower base boot, else the same).
usage: cells.py JSONDIR BASE_PREFIX ARM_PREFIX [ARM_PREFIX...]  (prefix: e.g. step-base)"""
import json, statistics, sys
from pathlib import Path

d = Path(sys.argv[1]); base = sys.argv[2]; arms = sys.argv[3:]


def boots(prefix):
    out = []
    for b in (1, 2):
        p = d / f"{prefix}-boot{b}.json"
        out.append(json.loads(p.read_text()))
    return out


def medians(res, key="step_ms"):
    return {r["shape"]: statistics.median(r[key]) for r in res["rows"]}


B = [medians(r) for r in boots(base)]
BF = [medians(r, "forward_ms") for r in boots(base)]
result = {"base": base, "arms": {}}
for arm in arms:
    A = [medians(r) for r in boots(arm)]
    AF = [medians(r, "forward_ms") for r in boots(arm)]
    cells = {}
    for shape in B[0]:
        b = [B[0][shape], B[1][shape]]; a = [A[0][shape], A[1][shape]]
        if max(a) < min(b):
            v = "faster"
        elif min(a) > max(b):
            v = "slower"
        else:
            v = "same"
        delta = statistics.mean(a) - statistics.mean(b)
        bf = [BF[0][shape], BF[1][shape]]; af = [AF[0][shape], AF[1][shape]]
        cells[shape] = {"base_ms": [round(x, 3) for x in b], "arm_ms": [round(x, 3) for x in a],
                        "delta_ms": round(delta, 3), "delta_pct": round(100 * delta / statistics.mean(b), 2),
                        "verdict": v,
                        "forward_base_ms": [round(x, 3) for x in bf], "forward_arm_ms": [round(x, 3) for x in af],
                        "forward_delta_ms": round(statistics.mean(af) - statistics.mean(bf), 3)}
    result["arms"][arm] = cells
    print(f"== {arm} vs {base}")
    for s, c in cells.items():
        print(f"{s:>6}  {c['base_ms'][0]:.3f}/{c['base_ms'][1]:.3f} -> {c['arm_ms'][0]:.3f}/{c['arm_ms'][1]:.3f}"
              f"  {c['delta_ms']:+.3f} ms ({c['delta_pct']:+.2f}%)  {c['verdict']}   fwd {c['forward_delta_ms']:+.3f}")
print(json.dumps(result))
