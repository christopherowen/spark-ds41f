#!/usr/bin/env python3
"""Decompose time to first token from a ttft_trace.py report.

usage: analyze_trace.py REPORT.json [--json OUT.json]

Isolated requests (one at a time, idle engine): client TTFT against the server's own
first-token latency, queue time and prefill time for that request alone, and the
single-stream decode step from the request's chunk gaps.

Closed-loop phases: every token-bearing chunk of every stream is a step end as the
client sees it; chunks less than GAP_S apart belong to the same engine step. For each
request:
- boundary: send time to the first step end after it (the rest of the step in flight,
  plus the constant delivery delay);
- lookahead: whole steps after that boundary that do not carry the request;
- first step: the step that produced its first token (its prefill, beside the other
  streams' decode rows);
compared with the median step that carries no first token. The server counters give the
phase means of queue time (until the scheduler takes the request) and prefill time
(from that decision to the first token, which includes any step already queued ahead).
"""

import argparse
import json
import statistics

GAP_S = 0.005


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


def summary(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"n": len(xs), "mean": statistics.mean(xs), "p50": pct(xs, 0.5),
            "p10": pct(xs, 0.1), "p90": pct(xs, 0.9)}


def first_token(r):
    for t, completion, _ in r["chunks"]:
        if completion > 0:
            return t
    return None


def mean_delta(before, after, name):
    count = after.get(name + "_count", 0) - before.get(name + "_count", 0)
    total = after.get(name + "_sum", 0) - before.get(name + "_sum", 0)
    return (total / count if count else None), count


def isolated(requests):
    rows = []
    for r in requests:
        b, a = r["metrics_before"], r["metrics_after"]
        server, n = mean_delta(b, a, "vllm:time_to_first_token_seconds")
        queue, _ = mean_delta(b, a, "vllm:request_queue_time_seconds")
        prefill, _ = mean_delta(b, a, "vllm:request_prefill_time_seconds")
        tf = first_token(r)
        times = [t for t, c, _ in r["chunks"] if c > 0]
        gaps = [y - x for x, y in zip(times, times[1:])]
        tokens = [c for _, c, _ in r["chunks"] if c > 0]
        rows.append({
            "key": r["key"][:8], "prompt_tokens": r["prompt_tokens"], "requests_counted": n,
            "client_ttft": tf - r["start"], "server_ttft": server, "queue": queue,
            "prefill": prefill, "delivery": (tf - r["start"]) - server if server else None,
            "frontend": server - queue - prefill if server is not None else None,
            "step_p50": pct(gaps, 0.5),
            "tokens_per_step": (tokens[-1] - tokens[0]) / max(len(tokens) - 1, 1),
        })
    return rows


def closed(phase, requests, info):
    chunks = sorted(t for r in requests for t, c, _ in r["chunks"] if c > 0)
    ends = []
    for t in chunks:
        if not ends or t - ends[-1][-1] > GAP_S:
            ends.append([t])
        else:
            ends[-1].append(t)
    step_end = [group[0] for group in ends]
    firsts = set()
    rows = []
    for r in requests:
        tf = first_token(r)
        if tf is None:
            continue
        f = max(i for i, t in enumerate(step_end) if t <= tf + 1e-9)
        firsts.add(f)
        j = next(i for i, t in enumerate(step_end) if t >= r["start"])
        rows.append({
            "stream": r["stream"], "index": r["index"], "key": r["key"][:8],
            "prompt_tokens": r["prompt_tokens"],
            "phase_start": r["start"] - info["start"] < 0.05,
            "client_ttft": tf - r["start"],
            "boundary": step_end[j] - r["start"] if j < f else tf - r["start"],
            "lookahead_steps": max(f - j - 1, 0),
            "lookahead": step_end[f - 1] - step_end[j] if f - j >= 2 else 0.0,
            "first_step": step_end[f] - step_end[f - 1] if f >= 1 and j < f else None,
        })
    durations = [b - a for a, b in zip(step_end, step_end[1:])]
    plain = [d for i, d in enumerate(durations, start=1) if i not in firsts]
    b, a = info["metrics_before"], info["metrics_after"]
    server = {}
    for name in ("vllm:time_to_first_token_seconds", "vllm:request_queue_time_seconds",
                 "vllm:request_prefill_time_seconds"):
        server[name.split(":")[1]], server[name.split(":")[1] + "_count"] = mean_delta(b, a, name)
    steady = [x for x in rows if not x["phase_start"]]
    return {
        "phase": phase, "requests": len(rows),
        "foreign_requests": (a.get("vllm:request_success_total", 0)
                             - b.get("vllm:request_success_total", 0)) - len(requests),
        "server_means": server,
        "plain_step": summary(plain),
        "steps_seen": len(step_end),
        "server_steps": a.get("vllm:iteration_tokens_total_count", 0)
        - b.get("vllm:iteration_tokens_total_count", 0),
        "all": {k: summary([x[k] for x in rows]) for k in
                ("client_ttft", "boundary", "lookahead", "first_step")},
        "after_start": {k: summary([x[k] for x in steady]) for k in
                        ("client_ttft", "boundary", "lookahead", "first_step")},
        "lookahead_steps": {str(n): sum(1 for x in steady if x["lookahead_steps"] == n)
                            for n in sorted({x["lookahead_steps"] for x in steady})},
        "rows": rows,
    }


def analyze(report):
    out = {"isolated": isolated([r for r in report["requests"] if r["phase"] == "isolated"])}
    for phase, info in report["phases"].items():
        out[phase] = closed(phase, [r for r in report["requests"] if r["phase"] == phase], info)
    return out


def summary_line(out):
    """Milliseconds per phase: TTFT p50 and mean after the opening burst, its parts' means,
    the plain step's p50 and how many requests waited a step that did not carry them."""
    ms = lambda x: None if x is None else round(1000 * x, 1)  # noqa: E731
    line = {"isolated_ttft_mean": ms(statistics.mean(r["client_ttft"] for r in out["isolated"]))
            if out["isolated"] else None}
    for phase, p in out.items():
        if phase == "isolated":
            continue
        after = p["after_start"]
        line[phase] = {
            "ttft_p50": ms(after["client_ttft"]["p50"]), "ttft_mean": ms(after["client_ttft"]["mean"]),
            "boundary_mean": ms(after["boundary"]["mean"]), "lookahead_mean": ms(after["lookahead"]["mean"]),
            "first_step_mean": ms(after["first_step"]["mean"]) if after["first_step"] else None,
            "plain_step_p50": ms(p["plain_step"]["p50"]), "lookahead_steps": p["lookahead_steps"],
            "foreign_requests": p["foreign_requests"],
        }
    return line


def render(report, out):
    ms = lambda x: "-" if x is None else f"{1000 * x:7.1f}"  # noqa: E731
    print("isolated: tokens  client  server  queue  prefill  frontend  delivery  step_p50  tok/step")
    for r in out["isolated"]:
        print(f"  {r['key']} {r['prompt_tokens']:>5} {ms(r['client_ttft'])} {ms(r['server_ttft'])}"
              f" {ms(r['queue'])} {ms(r['prefill'])} {ms(r['frontend'])} {ms(r['delivery'])}"
              f" {ms(r['step_p50'])} {r['tokens_per_step']:6.2f}")
    for phase in report["phases"]:
        p = out[phase]
        print(f"\n{phase}: {p['requests']} requests, foreign {p['foreign_requests']:.0f},"
              f" steps seen {p['steps_seen']} / server {p['server_steps']:.0f}")
        s = p["plain_step"]
        print(f"  plain step p50 {ms(s['p50'])} mean {ms(s['mean'])}")
        for label in ("all", "after_start"):
            print(f"  {label}:")
            for k, v in p[label].items():
                if v:
                    print(f"    {k:12s} mean {ms(v['mean'])} p50 {ms(v['p50'])} p90 {ms(v['p90'])} (n={v['n']})")
        print(f"  lookahead steps (after start): {p['lookahead_steps']}")
        sm = p["server_means"]
        print("  server means: ttft {} queue {} prefill {}".format(
            ms(sm["time_to_first_token_seconds"]), ms(sm["request_queue_time_seconds"]),
            ms(sm["request_prefill_time_seconds"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--json")
    a = ap.parse_args()
    report = json.load(open(a.report))
    out = analyze(report)
    render(report, out)
    print(json.dumps({"summary": summary_line(out)}))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
