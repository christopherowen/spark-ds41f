"""Where the target forward's end waits for the L2 prefetch stream (nsys sqlite, graph nodes).

usage: python3 ajoin.py ARM.sqlite [ARM.sqlite ...]

Per profiled step: the target forward ends at the model-level join of the prefetch side
stream, just before the LM head (``vocab_packed_projection_kernel``, once a step). For each
head launch: the end of the last L2 prefetch kernel before it, the end of the last other
kernel before it, and the wait = max(0, prefetch end - other end); with the prefetch stream's
busy time in the step and the forward's span (first kernel after the previous head to the
head's start)."""
import json, sqlite3, sys, statistics

def steps(path):
    db = sqlite3.connect(path)
    S = dict(db.execute("SELECT id, value FROM StringIds"))
    ks = [(s, e, st, S[n]) for s, e, st, n in db.execute(
        "SELECT start, end, streamId, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    heads = [i for i, k in enumerate(ks) if k[3] == "vocab_packed_projection_kernel"]
    out, prev = [], 0
    for h in heads:
        seg = ks[prev:h]
        pf = [k for k in seg if k[3] == "l2_prefetch_kernel"]
        other = [k for k in seg if k[3] != "l2_prefetch_kernel"]
        if not pf or not other:
            prev = h + 1; continue
        pf_end, other_end = max(k[1] for k in pf), max(k[1] for k in other)
        out.append({"forward_span_us": round((ks[h][0] - seg[0][0]) / 1e3, 1),
                    "prefetch_busy_us": round(sum(k[1] - k[0] for k in pf) / 1e3, 1),
                    "prefetch_launches": len(pf),
                    "join_wait_us": round(max(0, pf_end - other_end) / 1e3, 1),
                    "other_kernel_sum_us": round(sum(k[1] - k[0] for k in other) / 1e3, 1)})
        prev = h + 1
    return out

res = {}
for p in sys.argv[1:]:
    st = steps(p)
    res[p] = {"steps": st, "median": {k: statistics.median(s[k] for s in st) for k in st[0]}}
print(json.dumps(res, indent=1))
