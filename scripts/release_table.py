#!/usr/bin/env python3
"""The release comparison: every metric of a candidate against the previous release.

usage: scripts/release_table.py RUN BASE CANDIDATE [ARM...]   (from the deployment checkout)

Reads a "release" profile run (scripts/lab.py): results/private/bench/RUN-<arm>/bench.json
and results/private/determinism/RUN/<workload>-<arm>.jsonl. Each arm after the first is
compared with the first, one verdict per metric:

- faster / slower: the difference exceeds the two arms' 95% intervals combined
  (root sum of squares of the relative half-widths); a paired metric, when its own
  interval excludes zero;
- same: within them;
- a metric without an interval (TTFT medians, mixed traffic) is judged against the 3%
  budget and marked "(no interval)".

Prefill intervals are 1.96 sd / sqrt(n) from the repeats (a normal approximation: with
two or three repeats a t interval spans tens of percent and could never decide).

Sections, most trustworthy first:
1. determinism and references;
2. quality;
3. decode on text-independent workloads: the sixteen-prompt acceptance sweep, paired
   prompt by prompt, and distinct prompts at 1, 4, 8 and 16 concurrent streams. The single
   stream's verdict row pairs its prompts too (the same prompt in the same turn on both
   arms): the median ratio with its exact interval and a boot floor in quadrature (one
   boot against another: 0.35% from 31 single-stream A/A pairs; the release has no repeat);
   the old row with the interval over prompts stays beside it;
4. prefill by size;
5. latency: short-prompt TTFT and mixed traffic;
6. the bench's decode points. Each is one prompt repeated on N streams, so a release
   that changes the text (a precision change) changes the workload there; they show
   where a release moves, not by how much.
"""
import json
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from servebench import stats  # noqa: E402

BUDGET = 3.0  # percent, for metrics without an interval
DISTINCT = (1, 4, 8, 16)
C1_BOOT_FLOOR = 0.35  # percent: one boot against another (scripts/servebench_suite_table.py AA_FLOOR_PCT)


def bench(run, arm):
    path = f"results/private/bench/{run}-{arm}/bench.json"
    return json.load(open(path))["suites"] if os.path.exists(path) else {}


def lines(run, kind, arm):
    path = f"results/private/determinism/{run}/{kind}-{arm}.jsonl"
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path) if x.strip().startswith("{")]


def summary(run, kind, arm):
    found = [x for x in lines(run, kind, arm) if "summary" in x]
    return found[0] if found else None


def verdict(delta, interval, higher_better=True):
    """faster / slower / same for a relative change and its combined interval (percent)."""
    if delta is None:
        return ""
    better = delta if higher_better else -delta
    if interval is None:
        word = "faster" if better > BUDGET else "slower" if better < -BUDGET else "same"
        return f"{word} (no interval)"
    return "faster" if better > interval else "slower" if better < -interval else "same"


def row(label, base, value, base_ci=None, ci=None, higher_better=True, unit="", fmt="{:.1f}"):
    if base is None or value is None:
        return f"  {label:28s} {'-':>12}"
    delta = 100 * (value - base) / base if base else None
    interval = math.hypot(base_ci, ci) if base_ci is not None and ci is not None else None
    spread = f" ±{interval:.1f}%" if interval is not None else ""
    return (f"  {label:28s} {fmt.format(base):>10}{unit} -> {fmt.format(value):>10}{unit}  "
            f"{delta:+6.1f}%{spread:8s}  {verdict(delta, interval, higher_better)}")


def paired(run, base, arm, field):
    """Mean change of a per-prompt field over the prompts both arms ran, with its 95% interval."""
    def by_prompt(a):
        return {x["prompt"]: x for x in lines(run, "sweep", a) if "prompt" in x and "summary" not in x}
    pa, pb = by_prompt(base), by_prompt(arm)
    logs = [math.log(pb[p][field] / pa[p][field]) for p in sorted(set(pa) & set(pb))
            if pa[p].get(field) and pb[p].get(field)]
    if len(logs) < 2:
        return None
    mean, sd = statistics.mean(logs), statistics.stdev(logs)
    half = 1.96 * sd / math.sqrt(len(logs))
    return 100 * (math.exp(mean) - 1), 100 * half, len(logs)


def paired_c1(run, base, arm):
    """The single stream's prompts in turn, paired by turn: the median tok/s ratio as a
    change, its exact interval's side toward no change, and the margin with the boot floor."""
    def by_sample(a):
        return {x["sample"]: x["tps"] for x in lines(run, "c1-distinct", a) if "sample" in x and x.get("tps")}
    pa, pb = by_sample(base), by_sample(arm)
    interval = stats.median_interval([math.log(pb[k] / pa[k]) for k in sorted(set(pa) & set(pb))])
    if interval is None or interval["low"] is None:
        return None
    change = 100 * math.expm1(interval["median"])
    side = change - 100 * math.expm1(interval["low"]) if change >= 0 else 100 * math.expm1(interval["high"]) - change
    return change, side, math.hypot(side, C1_BOOT_FLOOR), interval["n"]


def main():
    run, arms = sys.argv[1], sys.argv[2:]
    if len(arms) < 2:
        raise SystemExit(__doc__)
    base = arms[0]
    suites = {a: bench(run, a) for a in arms}
    for arm in arms[1:]:
        print(f"== {arm} against {base} ({run})")

        print("determinism (requests differing from each prompt alone; references)")
        for a in (base, arm):
            s = summary(run, "determinism", a)
            if s:
                diffs = "; ".join(f"{k} {v['token_diffs']}/{v['logprob_diffs']}" for k, v in s["scenarios"].items())
                refs = f"{s.get('reference_sha256', '')[:12]} / {s.get('long_sha256', '')[:12]}"
                print(f"  {a:12s} {'identical' if s['identical'] else 'DIFFERS'}: {diffs}  refs {refs}")

        print("quality")
        for a in (base, arm):
            q = suites[a].get("quality", {})
            if q:
                print(f"  {a:12s} {q.get('passed', '?')}/{len(q.get('runs', [])) or '?'}"
                      f"{'' if q.get('ok', True) else '  FAILED'}")

        print("decode, text-independent")
        for field, label in (("tps", "acceptance sweep tok/s"), ("accepted_per_draft", "acceptance sweep accepted")):
            p = paired(run, base, arm, field)
            if p:
                delta, half, n = p
                word = "faster" if delta > half else "slower" if delta < -half else "same"
                if field != "tps":
                    word = "higher" if delta > half else "lower" if delta < -half else "same"
                print(f"  {label:28s} paired over {n} prompts: {delta:+6.1f}% ±{half:.1f}%  {word}")
        c1 = paired_c1(run, base, arm)
        if c1:
            change, side, margin, n = c1
            word = "faster" if change > margin else "slower" if change < -margin else "same"
            print(f"  {'1 stream tok/s, paired':28s} over {n} prompts: {change:+6.2f}% ±{side:.2f}%, "
                  f"margin {margin:.2f}% with the boot floor  {word}")
        for streams in DISTINCT:
            a_s, b_s = summary(run, f"c{streams}-distinct", base), summary(run, f"c{streams}-distinct", arm)
            if a_s and b_s:
                label = ("1 stream, 12 prompts in turn" if streams == 1
                         else f"{streams} different prompts at once")
                print(row(f"{label} tok/s", a_s["tps_mean"], b_s["tps_mean"], a_s["ci95_pct"], b_s["ci95_pct"]))

        print("prefill tok/s (cold, source text)")
        pa = suites[base].get("prefill", {}).get("points", {})
        pb = suites[arm].get("prefill", {}).get("points", {})
        def normal_pct(stats):
            n, sd, mean = stats.get("n", 0), stats.get("sd"), stats.get("mean")
            return 100 * 1.96 * sd / math.sqrt(n) / mean if n and sd is not None and mean else None

        for size in sorted(set(pa) & set(pb), key=int):
            x, y = pa[size]["prefill_tps"], pb[size]["prefill_tps"]
            print(row(f"{int(size):,} tokens", x["mean"], y["mean"], normal_pct(x), normal_pct(y), fmt="{:.0f}"))

        print("latency")
        t = []
        for a in (base, arm):
            vals = [x["median_ms"] for x in lines(run, "ttft", a) if x.get("median_ms") is not None]
            t.append(statistics.median(vals) if vals else None)
        print(row("short-prompt TTFT ms", t[0], t[1], higher_better=False))
        ma, mb = summary(run, "mixed", base), summary(run, "mixed", arm)
        if ma and mb:
            for key, label, higher in (("short_ttft_ms_median", "mixed short TTFT ms", False),
                                       ("long_ttft_ms_mean", "mixed long TTFT ms", False),
                                       ("stream_gap_ms_p95", "mixed stream gap p95 ms", False),
                                       ("stream_chunks_per_s", "mixed chunks/s", True)):
                print(row(label, ma.get(key), mb.get(key), higher_better=higher))

        print("bench decode points (one prompt on N streams: they follow the text)")
        da = suites[base].get("decode", {}).get("points", {})
        db = suites[arm].get("decode", {}).get("points", {})
        for point in da:
            if point in db:
                x, y = da[point], db[point]
                text = (f"  acceptance {x['accepted_per_draft']['mean']:.2f} -> {y['accepted_per_draft']['mean']:.2f}"
                        if "accepted_per_draft" in x and "accepted_per_draft" in y else "")
                print(row(f"{point} tok/s", x["tps"]["mean"], y["tps"]["mean"],
                          x["tps"].get("ci95_pct"), y["tps"].get("ci95_pct")) + text)


if __name__ == "__main__":
    main()
