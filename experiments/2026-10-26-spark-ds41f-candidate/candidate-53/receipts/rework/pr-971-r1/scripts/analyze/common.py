"""Shared cell verdict for candidate-53 own marks.
A cell has two boots per arm (base b1/b2, candidate c1/c2, run ABBA in one session).
delta = mean(c) - mean(b); spread = the larger boot-to-boot change within an arm.
Lower is better unless higher=True. 'better'/'worse' only when every candidate boot is on the
same side of every base boot AND |delta| exceeds the spread; otherwise 'same'."""
import json, statistics, math

def cell(b, c, higher=False, samples_b=None, samples_c=None):
    mb, mc = statistics.mean(b), statistics.mean(c)
    d = mc - mb
    spread = max(abs(b[0] - b[1]) if len(b) > 1 else 0.0, abs(c[0] - c[1]) if len(c) > 1 else 0.0)
    if higher:
        better = min(c) > max(b) and abs(d) > spread
        worse = max(c) < min(b) and abs(d) > spread
    else:
        better = max(c) < min(b) and abs(d) > spread
        worse = min(c) > max(b) and abs(d) > spread
    out = {"base": [round(x, 3) for x in b], "candidate": [round(x, 3) for x in c],
           "delta": round(d, 3), "delta_pct": round(100.0 * d / mb, 2) if mb else None,
           "boot_spread": round(spread, 3), "verdict": "better" if better else "worse" if worse else "same"}
    if samples_b and samples_c and len(samples_b) > 2 and len(samples_c) > 2:
        vb, vc = statistics.variance(samples_b), statistics.variance(samples_c)
        se = math.sqrt(vb / len(samples_b) + vc / len(samples_c))
        dd = statistics.mean(samples_c) - statistics.mean(samples_b)
        out["welch95"] = [round(dd - 1.96 * se, 3), round(dd + 1.96 * se, 3)]
    return out

def jl(path):
    return [json.loads(l) for l in open(path) if l.strip().startswith("{")]
