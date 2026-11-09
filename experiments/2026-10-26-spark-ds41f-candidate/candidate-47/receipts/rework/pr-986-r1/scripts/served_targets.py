"""Served targets for the replay: the 1107-budget arm's (candidate-47-no986, two boots) rows-a-step
counters per closed loop, the decoding requests a step, the prompt-token rate as a prefill share,
and the per-batch regression of serving step time on tokens and wasted rows (all four boots)."""
import json, sys
D, REG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, __import__("os").environ.get("REPO", "."))  # the repository root, for servebench
from servebench import report
arms = {"c16": ["candidate-47-no986-release-tp4/candidate-47-no986", "candidate-47-no986-release-tp4/candidate-47-no986-again"],
        "c8": ["candidate-47-no986-release-tp4/candidate-47-no986", "candidate-47-no986-release-tp4/candidate-47-no986-again"],
        "c1": ["candidate-47-no986-release-tp4-single-stream/candidate-47-no986", "candidate-47-no986-release-tp4-single-stream/candidate-47-no986-again"]}
nmax = {"c16": 16, "c8": 8, "c1": 1}
reg = json.load(open(REG))

def ols2(x1, x2, ys):
    n = len(ys); m1, m2, my = sum(x1) / n, sum(x2) / n, sum(ys) / n
    s11 = sum((a - m1) ** 2 for a in x1); s22 = sum((a - m2) ** 2 for a in x2)
    s12 = sum((a - m1) * (b - m2) for a, b in zip(x1, x2))
    s1y = sum((a - m1) * (y - my) for a, y in zip(x1, ys)); s2y = sum((a - m2) * (y - my) for a, y in zip(x2, ys))
    det = s11 * s22 - s12 * s12
    b1 = (s1y * s22 - s2y * s12) / det; b2 = (s2y * s11 - s1y * s12) / det
    res = [y - (my + b1 * (a - m1) + b2 * (b - m2)) for a, b, y in zip(x1, x2, ys)]
    s2 = sum(r * r for r in res) / (n - 3)
    return my - b1 * m1 - b2 * m2, b1, (s2 * s22 / det) ** 0.5, b2, (s2 * s11 / det) ** 0.5

out = {"source": "candidate-47-no986 (1107 adaptive_verification.py), two boots", "concurrency": {}, "token_wasted_regression": {}}
for step, dirs in arms.items():
    cards = [report.rows_a_step(json.load(open(f"{D}/{d}/{step}.json"))["summary"]) for d in dirs]
    sums = [json.load(open(f"{D}/{d}/{step}.json"))["summary"]["server"] for d in dirs]
    cov = [sum(c["positions"][k]["coverage"] for c in cards) / 2 for k in range(5)]
    surv = [sum(c["positions"][k]["survival"] for c in cards) / 2 for k in range(5)]
    req = sum(c["decoding_requests_per_step"] for c in cards) / 2
    prompt_per_step = sum(s["prompt_tps"]["mean"] / s["steps_per_s"]["mean"] for s in sums) / 2
    lat = [json.load(open(f"{D}/{d}/{step}.json"))["summary"]["latency"] for d in dirs]
    conc = {"c16": 16, "c8": 8, "c1": 1}[step]
    # decoding requests a step: the closed loop's in-flight requests less the share of each
    # request's life before its first token (TTFT / e2e); c1 has its one request
    req_true = 1.0 if conc == 1 else sum(conc * (1 - l["ttft_s"]["mean"] / l["e2e_s"]["mean"]) for l in lat) / 2
    tok = sum(c["tokens_per_step"] for c in cards) / 2
    out["concurrency"][step] = {"target": {"coverage": cov, "survival": surv, "tokens_per_step": tok},
                                "requests": req_true, "requests_drafting_estimate": req,
                                "prefill_rate": prompt_per_step / 21.0, "nmax": nmax[step], "context": 150,
                                "rows": sum(c["verified_rows_per_step"] for c in cards) / 2,
                                "tokens": sum(c["tokens_per_step"] for c in cards) / 2}
    P = reg[step]["points"]
    tok = [p["tokens"] for p in P]; wasted = [p["rows"] - p["tokens"] for p in P]; t = [p["step_ms"] for p in P]
    a, b1, se1, b2, se2 = ols2(tok, wasted, t)
    out["token_wasted_regression"][step] = {"intercept": a, "ms_per_token": b1, "ms_per_token_se": se1,
                                            "ms_per_wasted_row": b2, "ms_per_wasted_row_se": se2, "n_batches": len(P)}
# #986's context band at the contexts production's agent traffic runs (64-100K prompts at about
# two streams, TODO.md): the c1 fit and its serving models, at long context (the tp4-ring grid
# shows no context cost from 4K to 1M, so the step-time models carry none).
for ctx in (32768, 65536, 131072):
    out["concurrency"][f"c1-ctx{ctx // 1024}k"] = dict(out["concurrency"]["c1"], context=ctx, base="c1")
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
