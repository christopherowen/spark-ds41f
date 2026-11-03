#!/usr/bin/env python3
"""Analyze round-5 sustainment receipts.

For the scoring cases: teacher-forced logprob of every fixture token (the
prompt tail), the fraction of probability mass on the actual token at each
burst position, and the model's preferred alternative when it disagrees.
For all cases: the greedy continuation text after the fixture tail.
"""
import json
import math

R = json.load(open("/Users/christopherowen/projects/spark-ds41f/experiments/2026-11-03-loop-onset/runs/round5/receipt.json"))
by = {q["name"]: q for q in R["requests"]}
FIXTURE = json.load(open("/Users/christopherowen/projects/spark-ds41f/experiments/2026-11-03-short-repetition-bursts/fixture.json"))
LOOP = FIXTURE["token_ids"]
L = len(LOOP)

from tokenizers import Tokenizer
tok = Tokenizer.from_file("/Users/christopherowen/projects/spark-ds41f/.work/loop-analysis/tokenizer.json")

def summarize(name):
    req = by[name]
    ch = req["response"]["choices"][0]
    text = ch.get("text") or tok.decode(ch.get("token_ids") or [], skip_special_tokens=False)
    out = {"name": name, "continuation": text}
    gen_lps = (ch.get("logprobs") or {}).get("token_logprobs") or []
    out["generated_token_logprobs"] = [round(x, 3) if isinstance(x, float) else None for x in gen_lps]
    plp = ch.get("prompt_logprobs")
    if plp:
        n = len(plp)
        prompt_ids = ch.get("prompt_token_ids")
        if prompt_ids is None:
            prompt_ids = req["body"]["prompt"]
        tail = plp[n - L:]  # per-token entries; entry i scores prompt_ids[i]
        rows = []
        for i, entry in enumerate(tail):
            if not isinstance(entry, dict):
                continue
            tid = prompt_ids[n - L + i]
            hit = entry.get(str(tid))
            if hit is None:  # actual token outside the requested top-k
                continue
            rows.append((i, tid, hit["logprob"]))
        # aggregate over ' The' (455) occurrences and their following '.' (16)
        thes = [lp for i, tid, lp in rows if tid == 455 and lp is not None]
        dots = [lp for i, tid, lp in rows if tid == 16 and lp is not None]
        import statistics
        def stats(v):
            return {"n": len(v), "mean_lp": round(sum(v) / len(v), 3), "mean_p": round(math.exp(sum(v) / len(v)), 4),
                    "min_lp": round(min(v), 3), "frac_below_p10": round(sum(1 for x in v if x < math.log(0.1)) / len(v), 3)} if v else None
        out["teacher_forced"] = {
            "' The' (455)": stats(thes),
            "'.' (16)": stats(dots),
            "all_fixture_tokens": stats([lp for _, _, lp in rows if lp is not None]),
        }
        # where the model disagrees most: top alternative at the lowest-' The' positions
        worst = sorted([(lp, i) for i, tid, lp in rows if tid == 455])[:5]
        out["worst_the_positions"] = [
            {"pos_in_fixture": i, "lp": round(lp, 3),
             "top3": [(v["decoded_token"], round(v["logprob"], 3)) for v in sorted(tail[i].values(), key=lambda x: -x["logprob"])[:3]]}
            for lp, i in worst
        ]
    return out

for name in ("score-short-100", "score-short-75", "continue-long-100", "continue-long-75"):
    print(json.dumps(summarize(name), indent=1, ensure_ascii=False))
    print()

# unresolvable cases: burst check + what the model concluded
for name in ("unresolvable-100", "unresolvable-75"):
    ch = by[name]["response"]["choices"][0]
    ids = ch["token_ids"]
    msg = ch["message"]
    reasoning = msg.get("reasoning") or ""
    print(f"== {name}: {len(ids)} tokens, reasoning {len(reasoning)} chars ==")
    print("reasoning tail:", reasoning[-600:] if reasoning else "(none)")
    print("answer:", (msg.get("content") or "")[:300])
    print()
