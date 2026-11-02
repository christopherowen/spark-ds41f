#!/usr/bin/env python3
"""Tokens a step and tok/s ceilings for DSpark's acceptance levers, at c1 and c8 on TP4 and TP3.

usage: ceilings.py [--survival survival.json] [--out ceilings.json]

Step prices are ../2026-10-26-suffix-drafter/cost.py's Shape: #598's c1 model (the byte floor at
266.6 GB/s, live rows' new experts, the grid's 0.366 / 0.435 ms a row past the suite's), extended
there to TP3 and to eight streams and calibrated on the release suites, with prose routes N(r)
from the reference route probe (runs/routes-summary.json there). Every request of a c8 step is
the same request here (the suite's mix is what the calibration absorbed).

survival.json holds, per traffic, the measured survival S_k = P(at least k drafts accepted) with
every draft verified, and the production policy's measured tokens a step and verified drafts a
step at c1 and c8 (see measurements below). Each lever is priced against that traffic's own
production point:

- verify all n drafts: 1 + sum S_k tokens from 1 + n live rows;
- best static k: verify the first k drafts, k the best for that shape (no confidences needed);
- oracle verification: the same tokens as verify-all from live rows = tokens (each accepted
  draft's row and nothing else), the bound on any verification policy;
- more drafts: positions 6-8 at position 5's step acceptance (an upper bound: the checkpoint's
  drafter is trained on a block of 5);
- tree at position 1: a second chain from the second candidate, verified on every step, with
  P(second right | first wrong) = q and the branch continuing at the main chain's step
  acceptance (n more rows: the second candidate and its n - 1 continuations);
- drafter at the external figure: S_k = p^k with p set so that verify-all gives the external
  accept length (SGLang's 4.27 on GSM8K-200), i.e. what fixing a drafter defect could be worth.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COST = HERE.parent / "2026-10-26-suffix-drafter"
sys.path.insert(0, str(COST))
import cost  # noqa: E402

SHAPES = [("tp4", 1), ("tp3", 1), ("tp4", 8), ("tp3", 8)]


def tokens(survival: list[float], k: int | None = None) -> float:
    return 1.0 + sum(survival[: len(survival) if k is None else k])


def step_acceptance(survival: list[float]) -> list[float]:
    prev, out = 1.0, []
    for s in survival:
        out.append(s / prev if prev else 0.0)
        prev = s
    return out


def geometric(accept_length: float, n: int) -> list[float]:
    """S_k = p^k with 1 + sum S_k = accept_length."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        if 1 + sum(p**k for k in range(1, n + 1)) < accept_length:
            lo = p
        else:
            hi = p
    return [p**k for k in range(1, n + 1)]


class Pricer:
    def __init__(self, routes):
        self.shapes = {key: cost.Shape(key[0], key[1], routes) for key in SHAPES}
        self.routes = routes

    def tok_s(self, key, tokens_a_request: float, live: float, dead: float = 0.0) -> float:
        shape = self.shapes[key]
        reqs = key[1]
        ms = shape.step_ms([(live, self.routes, dead)] * reqs)
        return 1e3 * reqs * tokens_a_request / ms


def levers(pricer: Pricer, traffic: dict, external: float | None) -> dict:
    s = traffic["survival"]
    n = len(s)
    out = {"survival": s, "step_acceptance": [round(x, 4) for x in step_acceptance(s)],
           "verify_all_tokens": round(tokens(s), 4)}
    rows = {}
    for key in SHAPES:
        name = f"{key[0]}-c{key[1]}"
        prod = traffic.get("production", {}).get(f"c{key[1]}")
        if prod is None:
            continue
        # Production's verified rows split into live and dead as #598 split the suite's.
        verified = prod["verified_drafts"]
        live = 1 + verified * cost.LIVE_SHARE
        dead = verified - verified * cost.LIVE_SHARE
        base = pricer.tok_s(key, prod["tokens"], live, dead)
        table = {"production": (prod["tokens"], base)}
        table["verify all"] = (tokens(s), pricer.tok_s(key, tokens(s), 1 + n))
        best = max(range(n + 1), key=lambda k: pricer.tok_s(key, tokens(s, k), 1 + k))
        table[f"best static k ({best})"] = (tokens(s, best), pricer.tok_s(key, tokens(s, best), 1 + best))
        table["oracle verification"] = (tokens(s), pricer.tok_s(key, tokens(s), tokens(s)))
        p5 = step_acceptance(s)[-1]
        more = s + [s[-1] * p5 ** j for j in (1, 2, 3)]
        table["8 drafts, verify all (upper bound)"] = (tokens(more), pricer.tok_s(key, tokens(more), 1 + 8))
        table["8 drafts, oracle verification (upper bound)"] = (tokens(more), pricer.tok_s(key, tokens(more), tokens(more)))
        for q in (0.2, 0.35, 0.5):
            cont = step_acceptance(s)[1:]
            branch, run = 0.0, 1.0
            for a in cont:  # accepted drafts after the second candidate, chained
                run *= a
                branch += run
            gain = (1 - s[0]) * q * (1 + branch)
            table[f"tree at position 1, q={q}"] = (tokens(s) + gain, pricer.tok_s(key, tokens(s) + gain, 1 + 2 * n))
        if external:
            ext = geometric(external, n)
            table[f"drafter at {external} (fix to the external figure), verify all"] = (
                tokens(ext), pricer.tok_s(key, tokens(ext), 1 + n))
            table[f"drafter at {external}, oracle verification"] = (
                tokens(ext), pricer.tok_s(key, tokens(ext), tokens(ext)))
        rows[name] = {lever: {"tokens_a_step": round(t, 3), "tok_s": round(v, 1),
                              "vs_production": f"{100 * (v / base - 1):+.1f}%"}
                      for lever, (t, v) in table.items()}
    out["shapes"] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survival", default=str(HERE / "survival.json"))
    ap.add_argument("--out", default=str(HERE / "ceilings.json"))
    args = ap.parse_args()
    data = json.load(open(args.survival))
    probe = cost.load_probe_routes(COST / "runs" / "routes-summary.json")
    routes = probe.get("prose") or cost.Routes(cost.json_routes_table(), slope=2.4)
    pricer = Pricer(routes)
    result = {"routes": routes.name, "traffic": {}}
    for name, traffic in data["traffic"].items():
        result["traffic"][name] = levers(pricer, traffic, traffic.get("external_accept_length"))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    for name, t in result["traffic"].items():
        print(f"== {name}: verify-all {t['verify_all_tokens']} tokens a step, step acceptance {t['step_acceptance']}")
        for shape, table in t["shapes"].items():
            print(f"  {shape}")
            for lever, v in table.items():
                print(f"    {lever:58s} {v['tokens_a_step']:6.3f} {v['tok_s']:7.1f} {v['vs_production']}")


if __name__ == "__main__":
    main()
