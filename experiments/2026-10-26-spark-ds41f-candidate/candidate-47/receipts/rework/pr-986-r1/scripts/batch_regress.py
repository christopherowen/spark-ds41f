"""Per-batch serving step time against verified rows a step (closed loops, all four boots).
Pure Python: least squares of step_ms on rows (and on rows + requests) over the 30 s batches."""
import json, random, sys
D = sys.argv[1]
arms = {
    "c47": ("candidate-47-release-tp4/candidate-47", "candidate-47-release-tp4-single-stream/candidate-47"),
    "c47-again": ("candidate-47-release-tp4/candidate-47-again", "candidate-47-release-tp4-single-stream/candidate-47-again"),
    "no986": ("candidate-47-no986-release-tp4/candidate-47-no986", "candidate-47-no986-release-tp4-single-stream/candidate-47-no986"),
    "no986-again": ("candidate-47-no986-release-tp4/candidate-47-no986-again", "candidate-47-no986-release-tp4-single-stream/candidate-47-no986-again"),
}

def ols(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else float("nan")
    return my - b * mx, b

def ols2(x1, x2, ys):
    # y = a + b1 x1 + b2 x2 via normal equations on centred data
    n = len(ys); m1, m2, my = sum(x1) / n, sum(x2) / n, sum(ys) / n
    s11 = sum((a - m1) ** 2 for a in x1); s22 = sum((a - m2) ** 2 for a in x2)
    s12 = sum((a - m1) * (b - m2) for a, b in zip(x1, x2))
    s1y = sum((a - m1) * (y - my) for a, y in zip(x1, ys)); s2y = sum((a - m2) * (y - my) for a, y in zip(x2, ys))
    det = s11 * s22 - s12 * s12
    b1 = (s1y * s22 - s2y * s12) / det; b2 = (s2y * s11 - s1y * s12) / det
    return my - b1 * m1 - b2 * m2, b1, b2

def corr(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sxx * syy) ** 0.5

out = {}
for step in ("c16", "c8", "c1"):
    pts = []
    for arm, (rel, ss) in arms.items():
        d = json.load(open(f"{D}/{ss if step == 'c1' else rel}/{step}.json"))
        sb = d["summary"]["server_batches"]
        for i in range(len(sb["steps_per_s"])):
            tok = sb["output_tokens_per_step"][i]; acc = sb["accepted_per_draft"][i]; ver = sb["draft_tokens_per_draft"][i]
            req = tok / (1 + acc); rows = req * (1 + ver)
            pts.append({"arm": arm, "batch": i, "step_ms": 1000.0 / sb["steps_per_s"][i], "rows": rows,
                        "requests": req, "tokens": tok, "yield": tok / rows})
    rows = [p["rows"] for p in pts]; req = [p["requests"] for p in pts]; t = [p["step_ms"] for p in pts]
    a, b = ols(rows, t)
    rng = random.Random(0); bs = []
    for _ in range(2000):
        idx = [rng.randrange(len(t)) for _ in t]
        bs.append(ols([rows[i] for i in idx], [t[i] for i in idx])[1])
    bs.sort(); lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs))]
    a2, b1, b2 = ols2(rows, req, t)
    n = len(t); tm = sum(t) / n; rm = sum(rows) / n; qm = sum(req) / n
    rsd = (sum((x - rm) ** 2 for x in rows) / n) ** 0.5; qsd = (sum((x - qm) ** 2 for x in req) / n) ** 0.5
    out[step] = {"n": n, "step_ms_mean": tm, "rows_mean": rm, "rows_sd": rsd, "requests_mean": qm, "requests_sd": qsd,
                 "slope_ms_per_row": b, "slope_ci95_bootstrap": [lo, hi], "intercept_ms": a,
                 "corr_rows_step_ms": corr(rows, t), "with_requests": {"intercept": a2, "ms_per_row": b1, "ms_per_request": b2},
                 "elasticity_at_mean": b * rm / tm, "points": pts}
    print(f"{step}: n={n} step {tm:.2f} ms rows {rm:.2f}±{rsd:.2f} reqs {qm:.2f}±{qsd:.2f} slope {b:.3f} ms/row "
          f"[{lo:.3f},{hi:.3f}] r={out[step]['corr_rows_step_ms']:.2f} elasticity {out[step]['elasticity_at_mean']:.2f} | "
          f"with reqs: {b1:.3f} ms/row {b2:.3f} ms/req")
json.dump(out, open(sys.argv[2], "w"), indent=1)
