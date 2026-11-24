#!/usr/bin/env python3
"""Compare bi_probe runs against a reference run.

usage: bi_compare.py REF.json RUN.json [RUN.json ...] [--detail]

Per run: prompts with identical text; and when both carry log-probabilities, the first
token position whose id differs (fork), the first position whose log-probability differs
at all, the largest |delta logprob| before the fork, and the reference run's top-1 minus
top-2 margin at the fork.
"""
import json
import sys


def load(p):
    return json.load(open(p))


def margin(top):
    if not top:
        return None
    v = sorted(top.values(), reverse=True)
    return v[0] - v[1] if len(v) > 1 else None


def cmp(ref, run, detail=False):
    n = len(ref["outputs"])
    same_text = sum(a == b for a, b in zip(ref["outputs"], run["outputs"]))
    out = {"n": n, "text_identical": same_text}
    rr, qq = ref.get("records", []), run.get("records", [])
    if not rr or not qq or "tokens" not in rr[0] or "tokens" not in qq[0]:
        forks = []
        for i, (a, b) in enumerate(zip(ref["outputs"], run["outputs"])):
            if a != b:
                k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
                forks.append((i, k))
        out["text_forks(prompt,char)"] = forks
        return out
    tok_same = lp_same = 0
    rows = []
    for i, (a, b) in enumerate(zip(rr, qq)):
        ta, tb = a["tokens"], b["tokens"]
        la, lb = a["token_logprobs"], b["token_logprobs"]
        m = min(len(ta), len(tb))
        fork = next((j for j in range(m) if ta[j] != tb[j]), None)
        if fork is None and len(ta) != len(tb):
            fork = m
        upto = fork if fork is not None else m
        first_lp = next((j for j in range(upto) if la[j] != lb[j]), None)
        maxd = max((abs(la[j] - lb[j]) for j in range(upto)), default=0.0)
        ndiff = sum(la[j] != lb[j] for j in range(upto))
        if fork is None:
            tok_same += 1
        if fork is None and first_lp is None:
            lp_same += 1
        mg = margin(a["top_logprobs"][fork]) if fork is not None and fork < len(a.get("top_logprobs") or []) else None
        rows.append({"prompt": i, "fork_tok": fork, "first_lp_diff_tok": first_lp, "lp_diff_positions": ndiff,
                     "compared_positions": upto, "max_abs_dlp_before_fork": maxd, "ref_margin_at_fork": mg})
    out["tokens_identical"] = tok_same
    out["logprobs_bit_identical"] = lp_same
    out["prompts_with_any_lp_diff"] = sum(1 for r in rows if r["first_lp_diff_tok"] is not None)
    firsts = sorted(r["first_lp_diff_tok"] for r in rows if r["first_lp_diff_tok"] is not None)
    out["first_lp_diff_tok(min/median/max)"] = (firsts[0], firsts[len(firsts) // 2], firsts[-1]) if firsts else None
    tot = sum(r["compared_positions"] for r in rows)
    out["lp_diff_positions/compared"] = f"{sum(r['lp_diff_positions'] for r in rows)}/{tot}"
    out["max_abs_dlp"] = max(r["max_abs_dlp_before_fork"] for r in rows)
    out["forks(prompt,tok,ref_margin)"] = [(r["prompt"], r["fork_tok"], None if r["ref_margin_at_fork"] is None else round(r["ref_margin_at_fork"], 5)) for r in rows if r["fork_tok"] is not None]
    if detail:
        out["rows"] = rows
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    detail = "--detail" in sys.argv
    ref = load(args[0])
    for p in args[1:]:
        r = cmp(ref, load(p), detail)
        rows = r.pop("rows", None)
        print(p.split("/")[-1], json.dumps(r))
        if rows:
            for row in rows:
                if row["first_lp_diff_tok"] is not None or row["fork_tok"] is not None:
                    print("   ", json.dumps(row))
