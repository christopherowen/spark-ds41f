"""Turning a raw run into a summary, and comparing two summaries."""

from . import prom, stats
from .client import tokens_at


def summarize(raw: dict, *, slo_ttft_s: float | None = None, slo_tpot_ms: float | None = None) -> dict:
    records = raw["records"]
    failed = [r for r in records if r["error"]]
    summary: dict = {
        "requests": {"sent": len(records), "failed": len(failed), "rejected": raw.get("rejected", 0),
                     "approximate_token_times": sum(r["token_times"] == "approximate" for r in records)},
    }
    if failed:
        summary["requests"]["first_error"] = failed[0]["error"]
    summary["requests"]["foreign"] = _foreign(raw)
    if raw["window"] is None:  # burst
        events = [e for r in records for e in r["events"]]
        if events:
            first = min(r["first"] for r in records if r["first"] is not None)
            last = max(e[0] for e in events)
            total = sum(r["completion_tokens"] or 0 for r in records if not r["error"])
            summary["output_tps"] = {"n": 1, "mean": total / (last - first) if last > first else None}
        summary["server"] = prom.window(raw["scrapes"][0], raw["scrapes"][1],
                                        raw["scrape_times"][1] - raw["scrape_times"][0])
        summary["latency"] = _latency(records, slo_ttft_s, slo_tpot_ms, None)
        return summary

    w0, w1 = raw["window"]
    bounds = raw["boundaries"]
    per_batch_tps = []
    for a, b in zip(bounds, bounds[1:]):
        produced = sum(tokens_at(r, b) - tokens_at(r, a) for r in records)
        per_batch_tps.append(produced / (b - a))
    summary["output_tps"] = stats.describe(per_batch_tps) | {"batches": [round(v, 2) for v in per_batch_tps]}

    server_batches = []
    times, scrapes = raw["scrape_times"], raw["scrapes"]
    for k in range(len(bounds) - 1):
        if times[k] is not None and times[k + 1] is not None:
            server_batches.append(prom.window(scrapes[k], scrapes[k + 1], times[k + 1] - times[k]))
    keys = sorted({key for batch in server_batches for key in batch})
    summary["server"] = {key: stats.describe([batch.get(key) for batch in server_batches]) for key in keys}
    summary["server_batches"] = {key: [batch.get(key) for batch in server_batches] for key in keys}

    # In flight: time-weighted mean of open requests over the window.
    open_time = sum(max(0.0, min(r["end"], w1) - max(r["sent"], w0)) for r in records)
    summary["inflight_mean"] = open_time / (w1 - w0)
    in_window = [r for r in records if w0 <= r["sent"] < w1]
    summary["requests"]["started_in_window"] = len(in_window)
    completed = [r for r in records if not r["error"] and w0 <= r["end"] < w1]
    summary["requests_per_s"] = len(completed) / (w1 - w0)
    summary["latency"] = _latency(in_window, slo_ttft_s, slo_tpot_ms, w1 - w0)
    return summary


FOREIGN_SLACK_S = 0.5


def _foreign(raw: dict) -> int | None:
    """Requests the server finished during the run that were not the run's own: someone
    else's load on the server (a production client reaching a lab arm, 2026-10-18), so the
    run's speeds are not the build's. The first and last scrapes bound the count; the run's
    own requests are those that ended within them, or within FOREIGN_SLACK_S after the
    last (the server counts a request finished just before its last chunk reaches the
    client). The run's own requests still in flight at the last scrape that finish within
    the slack are counted as its own, so the count can fall short by up to the run's
    concurrency, never above. None when the server exports no counter."""
    pairs = [(t, s) for t, s in zip(raw["scrape_times"], raw["scrapes"])
             if t is not None and s and "requests_finished" in s]
    if len(pairs) < 2:
        return None
    (t0, first), (t1, last) = pairs[0], pairs[-1]
    finished = last["requests_finished"] - first["requests_finished"]
    own = sum(1 for r in raw["records"]
              if r["end"] is not None and t0 <= r["end"] <= t1 + FOREIGN_SLACK_S)
    return max(0, round(finished - own))


def _latency(records: list[dict], slo_ttft_s, slo_tpot_ms, seconds) -> dict:
    ok = [r for r in records if not r["error"] and r["first"] is not None]
    ttft = [r["first"] - r["sent"] for r in ok]
    tpot_ms = [1000 * (r["end"] - r["first"]) / (r["completion_tokens"] - 1)
               for r in ok if (r["completion_tokens"] or 0) > 1 and r["end"] > r["first"]]
    out = {"ttft_s": stats.percentiles(ttft), "tpot_ms": stats.percentiles(tpot_ms),
           "per_user_tps": stats.percentiles([1000 / t for t in tpot_ms if t > 0]),
           "e2e_s": stats.percentiles([r["end"] - r["sent"] for r in ok])}
    if slo_ttft_s is not None or slo_tpot_ms is not None:
        def meets(r: dict) -> bool:
            if slo_ttft_s is not None and r["first"] - r["sent"] > slo_ttft_s:
                return False
            tokens = r["completion_tokens"] or 0
            if slo_tpot_ms is not None and tokens > 1:
                return 1000 * (r["end"] - r["first"]) / (tokens - 1) <= slo_tpot_ms
            return True
        good = sum(meets(r) for r in ok)
        out["slo"] = {"ttft_s": slo_ttft_s, "tpot_ms": slo_tpot_ms,
                      "attainment_pct": 100 * good / len(records) if records else None,
                      "goodput_rps": good / seconds if seconds else None}
    return out


# (label, path into the summary, higher is better)
COMPARED = (
    ("output tok/s", ("output_tps",), True),
    ("engine steps/s", ("server", "steps_per_s"), True),
    ("output tokens/step", ("server", "output_tokens_per_step"), True),
    ("accepted/draft", ("server", "accepted_per_draft"), True),
    ("TTFT p50 s", ("latency", "ttft_s", "p50"), False),
    ("TTFT p99 s", ("latency", "ttft_s", "p99"), False),
    ("TPOT p50 ms", ("latency", "tpot_ms", "p50"), False),
    ("TPOT p99 ms", ("latency", "tpot_ms", "p99"), False),
    ("per-user tok/s p50", ("latency", "per_user_tps", "p50"), True),
    ("goodput req/s", ("latency", "slo", "goodput_rps"), True),
)
WORKLOAD_KEYS = ("mode", "concurrency", "rate", "warmup_s", "duration_s", "batches",
                 "max_tokens", "fixed_length", "seed", "context_tokens")


def _get(tree: dict, path: tuple):
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    return tree


def compare(base: dict, candidate: dict, *, tolerance_pct: float = 3.0) -> dict:
    """Per metric: both values, the change, its interval where both sides have batch
    samples, and a verdict; plus any workload differences that make it unfair."""
    warnings = []
    for key in WORKLOAD_KEYS:
        if base["workload"].get(key) != candidate["workload"].get(key):
            warnings.append(f"workload {key}: {base['workload'].get(key)} vs {candidate['workload'].get(key)}")
    if base["workload"].get("prompts_sha256") != candidate["workload"].get("prompts_sha256"):
        warnings.append("different prompt sets")
    rows = []
    for label, path, higher in COMPARED:
        a, b = _get(base["summary"], path), _get(candidate["summary"], path)
        if isinstance(a, dict) or isinstance(b, dict):
            # A batch-mean metric: compare its per-window samples.
            samples_a = _batch_samples(base, path)
            samples_b = _batch_samples(candidate, path)
            a_mean = (a or {}).get("mean")
            b_mean = (b or {}).get("mean")
            if a_mean is None or b_mean is None:
                continue
            test = stats.welch(samples_a, samples_b) if samples_a and samples_b else None
            change = 100 * (b_mean / a_mean - 1) if a_mean else 0.0
            ci = test["ci95_pct"] if test else None
            rows.append({"metric": label, "base": a_mean, "candidate": b_mean, "change_pct": change,
                         "ci95_pct": ci, "verdict": stats.verdict(change, ci, higher, tolerance_pct)})
        elif a is not None and b is not None and a:
            change = 100 * (b / a - 1)
            rows.append({"metric": label, "base": a, "candidate": b, "change_pct": change,
                         "ci95_pct": None, "verdict": stats.verdict(change, None, higher, tolerance_pct)})
    return {"warnings": warnings, "rows": rows}


def _batch_samples(run: dict, path: tuple) -> list[float] | None:
    if path == ("output_tps",):
        return run["summary"]["output_tps"].get("batches")
    if path[0] == "server":
        return run["summary"].get("server_batches", {}).get(path[1])
    return None


def format_comparison(result: dict, base_label: str, candidate_label: str) -> str:
    width = max([22] + [len(row["metric"]) for row in result["rows"]])
    lines = [f"{'metric':{width}s} {base_label:>12s} {candidate_label:>12s} {'change':>8s} {'±95%':>7s}  verdict"]
    for row in result["rows"]:
        ci = "" if row["ci95_pct"] is None else f"{row['ci95_pct']:.2f}%"
        values = " ".join(f"{v:12.3f}" if v is not None else f"{'':12s}" for v in (row["base"], row["candidate"]))
        lines.append(f"{row['metric']:{width}s} {values} {row['change_pct']:+7.2f}% {ci:>7s}  {row['verdict']}")
    if result.get("note"):
        lines.append(result["note"])
    for warning in result["warnings"]:
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines)
