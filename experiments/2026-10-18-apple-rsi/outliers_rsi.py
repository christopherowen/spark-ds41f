#!/usr/bin/env python3
"""The positions where a served model disagrees most with the task's reference log-probabilities.

usage: outliers_rsi.py URL --label LABEL [--port P] [--ids a,b] [--worst N] [--hidden PATH]

For each held-out fidelity sequence named (default: the two with the largest single-position
k3 on r6n), sends its token ids as the prompt with `prompt_logprobs=5` and `max_tokens=1`, as
the task's checker does but with the served top five, and prints one JSON line per sequence:
its mean k3, then its N worst positions with the token, the reference's and our
log-probability of it, k3, our top five (ids and log-probabilities, decoded through
/detokenize) and the 24 tokens before it. The token ids and reference values stay in the
private task copy; the output goes to results/private.
"""
import argparse
import json
import math
import os
import sys
import urllib.request
from urllib.parse import urlparse

HIDDEN = "~/spark-lab/apple-rsi-src/support_deepseek_v41_flash_4xb200_vllm/tests/hidden/fidelity.json"
SERVED = "deepseek-v4.1-flash"


def post(base, path, body, timeout=900):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def k3(lr):
    try:
        return math.exp(lr) - 1 - lr
    except OverflowError:
        return math.inf


def text(base, ids):
    try:
        return post(base, "/detokenize", {"model": SERVED, "tokens": ids})["prompt"]
    except Exception as exc:  # noqa: BLE001 - decoding is a convenience
        return f"<detokenize failed: {exc}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int)
    ap.add_argument("--ids", default="code-20480-0,math-8192-0")
    ap.add_argument("--worst", type=int, default=5)
    ap.add_argument("--hidden", default=HIDDEN)
    args = ap.parse_args()
    url = urlparse(args.url)
    base = f"{url.scheme}://{url.hostname}:{args.port or url.port}"
    with open(os.path.expanduser(args.hidden)) as handle:
        sequences = {s["id"]: s for s in json.load(handle)["sequences"]}
    for sid in args.ids.split(","):
        seq = sequences[sid]
        ids, ref = seq["token_ids"], seq["reference_logprobs"]
        reply = post(base, "/v1/completions", {"model": SERVED, "prompt": ids, "max_tokens": 1,
                                               "temperature": 0.0, "seed": 0, "prompt_logprobs": 5})
        entries = reply["choices"][0]["prompt_logprobs"]
        rows = []
        for pos in range(1, len(ids)):
            entry = entries[pos]
            own = entry[str(ids[pos])]["logprob"]
            top = sorted(((int(t), e["logprob"]) for t, e in entry.items()), key=lambda x: -x[1])[:5]
            lr = ref[pos - 1] - own
            rows.append({"pos": pos, "token": ids[pos], "ref_lp": round(ref[pos - 1], 4), "our_lp": round(own, 4),
                         "k3": k3(lr), "top": top})
        mean = sum(r["k3"] for r in rows) / len(rows)
        worst = sorted(rows, key=lambda r: -r["k3"])[: args.worst]
        for r in worst:
            r["token_text"] = text(base, [r["token"]])
            r["top"] = [[t, round(lp, 4), text(base, [t])] for t, lp in r["top"]]
            r["before"] = text(base, ids[max(0, r["pos"] - 24): r["pos"]])
        share = sum(r["k3"] for r in worst) / (mean * len(rows)) if mean else 0.0
        print(json.dumps({"label": args.label, "id": sid, "positions": len(rows), "k3_mean": mean,
                          "worst_share_of_total": round(share, 4), "worst": worst}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
