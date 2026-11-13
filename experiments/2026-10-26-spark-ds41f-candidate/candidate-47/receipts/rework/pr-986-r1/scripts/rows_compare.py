"""Rows a step (servebench report.rows_a_step) for candidate-47 (with #986) and
candidate-47-no986 (1107's adaptive_verification.py), two boots each, closed loops c16, c8,
Poisson, and the single-stream c1; with the mean change and each arm's boot spread."""
import json, sys
sys.path.insert(0, __import__("os").environ.get("REPO", "."))  # the repository root, for servebench
from servebench import report
D, OUT = sys.argv[1], sys.argv[2]
arms = {
    "with986": {"release": ["candidate-47-release-tp4/candidate-47", "candidate-47-release-tp4/candidate-47-again"],
                "single": ["candidate-47-release-tp4-single-stream/candidate-47", "candidate-47-release-tp4-single-stream/candidate-47-again"]},
    "no986": {"release": ["candidate-47-no986-release-tp4/candidate-47-no986", "candidate-47-no986-release-tp4/candidate-47-no986-again"],
              "single": ["candidate-47-no986-release-tp4-single-stream/candidate-47-no986", "candidate-47-no986-release-tp4-single-stream/candidate-47-no986-again"]},
}
out = {"source": "results/private/suites on dgx1 (read-only copies)", "steps": {}}
for step, kind in (("c16", "release"), ("c8", "release"), ("poisson", "release"), ("c1", "single")):
    rec = {}
    for arm, dirs in arms.items():
        boots = []
        for d in dirs[kind]:
            j = json.load(open(f"{D}/{d}/{step}.json"))
            card = report.rows_a_step(j["summary"])
            s = j["summary"]["server"]
            boots.append({"dir": d, "tokens": card["tokens_per_step"], "requests": card["decoding_requests_per_step"],
                          "rows": card["verified_rows_per_step"], "yield": card["row_yield"], "wasted": card["wasted_rows_per_step"],
                          "steps_per_s": s["steps_per_s"]["mean"], "output_tps": j["summary"]["output_tps"]["mean"],
                          "positions": [{k: p[k] for k in ("position", "survival", "coverage", "conditional_acceptance", "verified_yield")}
                                        for p in card["positions"][:6]]})
        mean = {k: sum(b[k] for b in boots) / 2 for k in ("tokens", "requests", "rows", "yield", "wasted", "steps_per_s", "output_tps")}
        mean["coverage"] = [sum(b["positions"][k]["coverage"] for b in boots) / 2 for k in range(6)]
        mean["survival"] = [sum(b["positions"][k]["survival"] for b in boots) / 2 for k in range(6)]
        rec[arm] = {"boots": boots, "mean": mean,
                    "boot_spread": {k: abs(boots[0][k] - boots[1][k]) for k in ("tokens", "rows", "wasted", "yield")}}
    a, b = rec["with986"]["mean"], rec["no986"]["mean"]
    rec["with986_vs_no986_pct"] = {k: 100 * (a[k] / b[k] - 1) for k in ("tokens", "requests", "rows", "yield", "wasted", "steps_per_s", "output_tps")}
    rec["with986_minus_no986"] = {k: a[k] - b[k] for k in ("tokens", "rows", "wasted")}
    rec["with986_minus_no986"]["coverage"] = [x - y for x, y in zip(a["coverage"], b["coverage"])]
    rec["with986_minus_no986"]["survival"] = [x - y for x, y in zip(a["survival"], b["survival"])]
    out["steps"][step] = rec
    d = rec["with986_vs_no986_pct"]
    print(f"{step:8s} rows {b['rows']:.2f}->{a['rows']:.2f} ({d['rows']:+.2f}%) tokens {b['tokens']:.3f}->{a['tokens']:.3f} ({d['tokens']:+.2f}%) "
          f"yield {d['yield']:+.2f}% wasted {b['wasted']:.2f}->{a['wasted']:.2f} ({d['wasted']:+.2f}%) steps/s {d['steps_per_s']:+.2f}% "
          f"| boot spread rows {rec['with986']['boot_spread']['rows']:.2f}/{rec['no986']['boot_spread']['rows']:.2f}")
    print("         coverage d", [round(x, 4) for x in rec["with986_minus_no986"]["coverage"][:5]], " survival d", [round(x, 4) for x in rec["with986_minus_no986"]["survival"][:5]])
json.dump(out, open(OUT, "w"), indent=1)
