"""Kernel timeline of the profiled engine-grid steps (nsys --cuda-graph-trace=node sqlite).

usage: python3 ansys.py BASE.sqlite CAND.sqlite [OUT.json]

The profiled window holds step957.py's ``--profile-shapes 1x2,1x6`` grid with one warm-up
and four replays, settle 0: ten dummy steps in the order 1x2, 1x6 (warm-up), then
replays alternating the order: 1x2 1x6 | 1x6 1x2 | 1x2 1x6 | 1x6 1x2. A step is split from
the next at the largest idle gaps of the GPU timeline. Per step: span, the busy time of
each stream, and every kernel's time by name; then each kernel name's mean time per step
in the two arms and the difference, largest first.
"""

import collections
import json
import sqlite3
import statistics
import sys

ORDER = ["1x2", "1x6", "1x2", "1x6", "1x6", "1x2", "1x2", "1x6", "1x6", "1x2"]


def kernels(path):
    db = sqlite3.connect(path)
    cols = [r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    name_col = "shortName" if "shortName" in cols else "demangledName"
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    rows = db.execute(f"SELECT start, end, streamId, {name_col} FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    return [(s, e, st, strings.get(n, str(n))) for s, e, st, n in rows], cols


def split_steps(ks, n):
    gaps = []
    end = ks[0][1]
    for i in range(1, len(ks)):
        gaps.append((ks[i][0] - end, i))
        end = max(end, ks[i][1])
    cuts = sorted(i for _, i in sorted(gaps, reverse=True)[: n - 1])
    out, prev = [], 0
    for c in cuts + [len(ks)]:
        out.append(ks[prev:c])
        prev = c
    return out


def summarize(step):
    start = min(k[0] for k in step)
    end = max(k[1] for k in step)
    by_stream = collections.Counter()
    by_name = collections.Counter()
    count = collections.Counter()
    for s, e, st, name in step:
        by_stream[st] += (e - s) / 1e3
        by_name[name] += (e - s) / 1e3
        count[name] += 1
    return {"span_us": (end - start) / 1e3, "kernels": len(step), "by_stream_us": dict(by_stream),
            "by_name_us": dict(by_name), "count": dict(count)}


def arm(path):
    ks, cols = kernels(path)
    steps = split_steps(ks, len(ORDER))
    res = collections.defaultdict(list)
    for shape, step in zip(ORDER, steps):
        res[shape].append(summarize(step))
    return res, cols, len(ks)


def main():
    base, cols, nb = arm(sys.argv[1])
    cand, _, nc = arm(sys.argv[2])
    out = {"columns": cols, "kernels_in_window": {"base": nb, "cand": nc}, "shapes": {}}
    for shape in ("1x2", "1x6"):
        b, c = base[shape][1:], cand[shape][1:]  # drop the warm-up step
        names = set()
        for x in b + c:
            names |= set(x["by_name_us"])
        rows = []
        for name in names:
            mb = statistics.mean(x["by_name_us"].get(name, 0.0) for x in b)
            mc = statistics.mean(x["by_name_us"].get(name, 0.0) for x in c)
            nb_ = statistics.mean(x["count"].get(name, 0) for x in b)
            nc_ = statistics.mean(x["count"].get(name, 0) for x in c)
            rows.append({"kernel": name[:120], "base_us": round(mb, 1), "cand_us": round(mc, 1),
                         "delta_us": round(mc - mb, 1), "base_launches": nb_, "cand_launches": nc_})
        rows.sort(key=lambda r: -abs(r["delta_us"]))
        streams = sorted({s for x in b + c for s in x["by_stream_us"]})
        out["shapes"][shape] = {
            "span_us": {"base": [round(x["span_us"], 1) for x in b], "cand": [round(x["span_us"], 1) for x in c]},
            "kernels_per_step": {"base": [x["kernels"] for x in b], "cand": [x["kernels"] for x in c]},
            "stream_busy_us": {str(s): {"base": round(statistics.mean(x["by_stream_us"].get(s, 0) for x in b), 1),
                                        "cand": round(statistics.mean(x["by_stream_us"].get(s, 0) for x in c), 1)}
                               for s in streams},
            "sum_kernel_us": {"base": round(statistics.mean(sum(x["by_name_us"].values()) for x in b), 1),
                              "cand": round(statistics.mean(sum(x["by_name_us"].values()) for x in c), 1)},
            "by_kernel": rows[:60],
        }
    text = json.dumps(out, indent=1)
    if len(sys.argv) > 3:
        open(sys.argv[3], "w").write(text + "\n")
    print(text[:20000])


if __name__ == "__main__":
    main()
