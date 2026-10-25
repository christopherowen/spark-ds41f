#!/usr/bin/env python3
"""The route probe's routes reduced to what the cost model needs.

usage: routes_summary.py ROUTES.json OUT.json

For each case:
- ``distinct``: N(r), the distinct experts r consecutive reply rows reach in a target layer,
  averaged over layers and over windows (starting every r/2 rows), for r = 1..96;
- ``distinct_by_layer``: the same at 6, 17, 33, 49 and 65 rows, per layer;
- ``greedy_agreement``: the share of reply positions where the reference's greedy token is
  the forced one (how close the ideal reply is to the model's own);
- for copied replies, ``source_overlap``: at each reply row whose exact (m = 8) source is
  in the prompt, the share of its six experts that the source row also routes to, per layer
  averaged (a copy routing like its source would score 1).
"""
import json
import statistics
import sys

WIDTHS = (6, 17, 33, 49, 65)


def distinct(layer_routes, r):
    n = len(layer_routes)
    starts = range(0, n - r + 1, max(1, r // 2))
    return [len({e for t in range(s, s + r) for e in layer_routes[t]}) for s in starts]


def sources(ids, start, m=8):
    """For each reply position, the position in the prompt that the exact m-token match
    copies (the token after the latest earlier occurrence of the m tokens before it)."""
    latest, out = {}, {}
    for e in range(m, len(ids)):
        key = tuple(ids[e - m : e])
        if e >= start and key in latest and latest[key] < start:
            out[e] = latest[key]
        latest[key] = e
    return out


def main() -> None:
    data = json.load(open(sys.argv[1]))
    out = {"snapshot": data["snapshot"], "tokens": data["tokens"], "seconds": data["seconds"],
           "bytes_read": data["bytes_read"], "cases": {}}
    for case in data["cases"]:
        start, ids = case["reply_start"], case["token_ids"]
        reply = [layer[start:] for layer in case["routes"]]
        n = len(reply[0])
        table = {r: round(statistics.mean(v for layer in reply for v in distinct(layer, r)), 2)
                 for r in range(1, min(96, n) + 1)}
        by_layer = {r: [round(statistics.mean(distinct(layer, r)), 1) for layer in reply] for r in WIDTHS if r <= n}
        greedy = case["greedy_after"]
        agree = sum(p == t for p, t in zip(greedy, ids[start:])) / len(greedy)
        summary = {"reply_tokens": n, "prompt_tokens": start, "distinct": table,
                   "distinct_by_layer": by_layer, "greedy_agreement": round(agree, 4),
                   "seconds": case["seconds"]}
        src = sources(ids, start)
        if src:
            overlaps = [statistics.mean(len(set(layer[e]) & set(layer[s])) / 6 for e, s in src.items())
                        for layer in case["routes"]]
            summary["source_overlap"] = {"rows": len(src), "mean": round(statistics.mean(overlaps), 3),
                                         "by_layer": [round(x, 3) for x in overlaps]}
        out["cases"][case["case"]] = summary
    with open(sys.argv[2], "w") as handle:
        json.dump(out, handle, indent=1)
        handle.write("\n")
    for name, s in out["cases"].items():
        print(name, s["reply_tokens"], "greedy", s["greedy_agreement"], {r: s["distinct"].get(r) for r in (1, 2, 6, 17, 33, 49, 65)},
              "source overlap", s.get("source_overlap", {}).get("mean"))


if __name__ == "__main__":
    main()
