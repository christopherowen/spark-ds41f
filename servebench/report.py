"""Turning a raw run into a summary, and comparing two summaries.

Every run keeps a compact table of its requests. Two runs of one workload send the same
prompt as their i-th request (the seed fixes the order), so their requests pair by index.
A closed loop's window holds whichever prompts fell in it, and greedy text gives each
prompt its own acceptance: windows differ by text more than builds do (the window
interval at one stream is ±3.7% where the same build repeats within 0.13%). The paired
rows compare each request's decode rate (output tokens after the first, over the time from
the first to the last) and its time to first token with the same request on the other
build: the median of the ratios with an exact interval over requests."""

import math

from . import prom, stats
from .client import tokens_at

TABLE_COLUMNS = ("index", "prompt", "slot", "sent", "first", "last", "end", "prompt_tokens",
                 "completion_tokens", "first_tokens", "chunks", "error")


def request_table(raw: dict) -> dict:
    """One row per request (``TABLE_COLUMNS``), times in seconds from the run's start; the
    window, when the run has one, on the same clock. ``last`` is the last output chunk,
    ``first_tokens`` the output tokens of the first and ``chunks`` the output chunks (the
    engine steps that gave the request tokens, with per-chunk usage)."""
    origin = raw["start"]

    def at(moment):
        return None if moment is None else round(moment - origin, 4)

    rows = []
    for r in sorted(raw["records"], key=lambda r: r.get("index", 0)):
        events = r["events"]
        rows.append([r.get("index"), r.get("prompt_id"), r.get("slot"), at(r["sent"]), at(r["first"]),
                     at(events[-1][0]) if events else None, at(r["end"]), r["prompt_tokens"],
                     r["completion_tokens"], events[0][1] if events else None, len(events), r["error"]])
    window = [at(moment) for moment in raw["window"]] if raw.get("window") else None
    return {"columns": list(TABLE_COLUMNS), "window": window, "rows": rows}


def _requests(run: dict) -> tuple[dict, list | None] | None:
    table = run.get("request_table")
    if not table:
        return None
    return {row[0]: dict(zip(table["columns"], row)) for row in table["rows"]}, table.get("window")


def _decode_rate(r: dict) -> float | None:
    tokens = (r["completion_tokens"] or 0) - (r["first_tokens"] or 0)
    if r["error"] or r["first"] is None or r["last"] is None or tokens <= 0 or r["last"] <= r["first"]:
        return None
    return tokens / (r["last"] - r["first"])


def _ttft(r: dict) -> float | None:
    return None if r["error"] or r["first"] is None else r["first"] - r["sent"]


def paired(base: dict, candidate: dict, confidence: float = 0.95) -> dict | None:
    """Decode rate and time to first token, request by request, over the requests both runs
    sent inside their windows (the same index and prompt): the median ratio as a change in
    %, with its exact interval (``low_pct``, ``high_pct``). None unless both runs kept their
    request tables. ``same_chunks`` counts pairs streamed in as many chunks on both builds
    (the same text and drafting give the same steps)."""
    sides = _requests(base), _requests(candidate)
    if not all(sides):
        return None
    (a, window_a), (b, window_b) = sides

    def inside(r: dict, window: list | None) -> bool:
        return window is None or window[0] <= r["sent"] < window[1]

    pairs, mismatched = [], 0
    for index in sorted(set(a) & set(b)):
        x, y = a[index], b[index]
        if x["prompt"] != y["prompt"]:
            mismatched += 1
        elif inside(x, window_a) and inside(y, window_b):
            pairs.append((x, y))
    out = {"pairs": len(pairs), "prompt_mismatches": mismatched,
           "same_chunks": sum(x["chunks"] == y["chunks"] for x, y in pairs)}
    for name, value in (("decode_tps", _decode_rate), ("ttft_s", _ttft)):
        values = [(value(x), value(y)) for x, y in pairs]
        values = [(u, v) for u, v in values if u and v]
        interval = stats.median_interval([math.log(v / u) for u, v in values], confidence)
        if interval is None:
            continue
        change = 100 * math.expm1(interval["median"])
        entry = {"n": interval["n"], "base": stats.median_interval([u for u, _ in values])["median"],
                 "candidate": stats.median_interval([v for _, v in values])["median"], "change_pct": change,
                 "low_pct": None, "high_pct": None, "ci95_pct": None, "coverage": interval["coverage"]}
        if interval["low"] is not None:
            low, high = 100 * math.expm1(interval["low"]), 100 * math.expm1(interval["high"])
            entry |= {"low_pct": low, "high_pct": high, "ci95_pct": max(change - low, high - change)}
        out[name] = entry
    return out


def window_paired(base: dict, candidate: dict, confidence: float = 0.95) -> dict | None:
    """Output tok/s window by window (window k of one run against window k of the other),
    for runs that kept no request table: the median ratio with its exact interval. With a
    real change the faster build's windows drift onto later prompts, so the interval holds
    that text as well (about ±2% at one stream for a 2% change)."""
    a = base["summary"].get("output_tps", {}).get("batches")
    b = candidate["summary"].get("output_tps", {}).get("batches")
    if not a or not b or len(a) != len(b):
        return None
    interval = stats.median_interval([math.log(v / u) for u, v in zip(a, b) if u and v], confidence)
    if interval is None or interval["low"] is None:
        return None
    change = 100 * math.expm1(interval["median"])
    low, high = 100 * math.expm1(interval["low"]), 100 * math.expm1(interval["high"])
    return {"n": interval["n"], "change_pct": change, "low_pct": low, "high_pct": high,
            "ci95_pct": max(change - low, high - change), "coverage": interval["coverage"]}


def _bounded_verdict(entry: dict, higher_is_better: bool) -> str:
    """better / worse when the (asymmetric) interval excludes no change; else same."""
    if entry["low_pct"] is None:
        return "same (no interval)"
    if entry["low_pct"] > 0 or entry["high_pct"] < 0:
        return "better" if (entry["low_pct"] > 0) == higher_is_better else "worse"
    return "same"


def rows_a_step(summary: dict) -> dict:
    """A run's step accounting from its counters (window means): every decoding request
    drafts every step, so it adds 1 + accepted tokens and 1 + verified rows (the target row
    and its verified drafts). Row yield is tokens per verified row; wasted rows are verified
    rows that gave no token (rejected rows and the rows after them). Per draft position:
    survival (accepted there a draft), coverage (verified there a draft), conditional
    acceptance given survival through the previous position, and verified-row yield."""
    server = summary.get("server", {})

    def mean(key):
        value = server.get(key)
        return value.get("mean") if isinstance(value, dict) else None

    tokens, accepted, verified = (mean(k) for k in ("output_tokens_per_step", "accepted_per_draft",
                                                    "draft_tokens_per_draft"))
    out = {"tokens_per_step": tokens, "accepted_per_draft": accepted, "verified_per_draft": verified}
    if accepted is not None:
        out["acceptance_length"] = 1 + accepted
    if tokens and accepted is not None and verified is not None:
        requests = tokens / (1 + accepted)
        rows = requests * (1 + verified)
        out |= {"decoding_requests_per_step": requests, "verified_rows_per_step": rows,
                "row_yield": tokens / rows, "wasted_rows_per_step": rows - tokens}
    positions = sorted({int(key.rsplit("_", 1)[1]) for key in server if key.startswith("survival_at_")})
    pos_rows = []
    for k in positions:
        survival = mean(f"survival_at_{k}")
        coverage = mean(f"coverage_at_{k}")
        prev_survival = 1.0 if k == 0 else mean(f"survival_at_{k - 1}")
        cond_accept = (survival / prev_survival
                       if survival is not None and prev_survival is not None and prev_survival > 0
                       else None)
        verified_yield = (survival / coverage
                          if survival is not None and coverage is not None and coverage > 0
                          else None)
        pos_rows.append({"position": k, "survival": survival, "coverage": coverage,
                         "conditional_acceptance": cond_accept, "verified_yield": verified_yield})
    out["positions"] = pos_rows
    return out


def _log_term(u: float, v: float) -> dict:
    log_ratio = math.log(v / u)
    return {
        "base": u,
        "candidate": v,
        "ratio": v / u,
        "log_change": log_ratio,
        "change_pct": 100.0 * math.expm1(log_ratio),
    }


def spec_decomposition(base_summary: dict, candidate_summary: dict) -> dict | None:
    """Exact log-throughput decomposition of server decode output between two runs:
    ``log(tps_b / tps_a) = log((1 + tau_b) / (1 + tau_a)) + log(steps_s_b / steps_s_a)
    + log(reqs_step_b / reqs_step_a)``, plus verified-row vs row-yield decomposition of
    tokens per step and per-position deltas when speculative counters are present."""
    a, b = rows_a_step(base_summary), rows_a_step(candidate_summary)
    r_a = (base_summary.get("server", {}).get("steps_per_s") or {}).get("mean")
    r_b = (candidate_summary.get("server", {}).get("steps_per_s") or {}).get("mean")
    if (not r_a or not r_b or not a.get("tokens_per_step") or not b.get("tokens_per_step")
            or a.get("accepted_per_draft") is None or b.get("accepted_per_draft") is None):
        return None
    l_a, l_b = 1.0 + a["accepted_per_draft"], 1.0 + b["accepted_per_draft"]
    if l_a <= 0 or l_b <= 0:
        return None
    q_a, q_b = a["tokens_per_step"] / l_a, b["tokens_per_step"] / l_b
    tps_a, tps_b = r_a * a["tokens_per_step"], r_b * b["tokens_per_step"]
    out: dict = {
        "server_tps": _log_term(tps_a, tps_b),
        "acceptance_length": _log_term(l_a, l_b),
        "steps_per_s": _log_term(r_a, r_b),
        "requests_per_step": _log_term(q_a, q_b),
    }
    if (a.get("verified_rows_per_step") and b.get("verified_rows_per_step")
            and a.get("row_yield") and b.get("row_yield")):
        out["verified_rows_per_step"] = _log_term(a["verified_rows_per_step"], b["verified_rows_per_step"])
        out["row_yield"] = _log_term(a["row_yield"], b["row_yield"])
    pos_a = {p["position"]: p for p in a["positions"]}
    pos_b = {p["position"]: p for p in b["positions"]}
    positions = []
    for k in sorted(set(pos_a) & set(pos_b)):
        pa, pb = pos_a[k], pos_b[k]
        positions.append({
            "position": k,
            "survival_delta": (pb["survival"] - pa["survival"]
                               if pa["survival"] is not None and pb["survival"] is not None else None),
            "coverage_delta": (pb["coverage"] - pa["coverage"]
                               if pa["coverage"] is not None and pb["coverage"] is not None else None),
            "conditional_acceptance_delta": (
                pb["conditional_acceptance"] - pa["conditional_acceptance"]
                if pa.get("conditional_acceptance") is not None and pb.get("conditional_acceptance") is not None
                else None
            ),
            "verified_yield_delta": (
                pb["verified_yield"] - pa["verified_yield"]
                if pa.get("verified_yield") is not None and pb.get("verified_yield") is not None
                else None
            ),
        })
    out["positions"] = positions
    return out


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


def _foreign(raw: dict) -> int | None:
    """Requests the server finished during the run that were not the run's own: someone
    else's load on the server (a production client reaching a lab arm, 2026-10-18), so the
    run's speeds are not the build's. The run's fences (one scrape before its first request,
    one after its last drained) contain every own request, so the count is the server's
    finished requests between them less the run's own successful ones, exactly. None when
    the run has no fences or the server exports no counter."""
    fences = raw.get("fences") or {}
    scrapes = fences.get("scrapes") or []
    if len(scrapes) != 2 or not all(s and "requests_finished" in s for s in scrapes):
        return None
    finished = scrapes[1]["requests_finished"] - scrapes[0]["requests_finished"]
    own = sum(1 for r in raw["records"] if r["end"] is not None and not r["error"])
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
    result = {"warnings": warnings, "rows": rows}
    pairs = paired(base, candidate)
    if pairs is not None:
        # Each paired row goes after the window metric it sharpens.
        for name, label, after, higher in (("decode_tps", "decode tok/s, paired", "output tok/s", True),
                                           ("ttft_s", "TTFT, paired", "TTFT p99 s", False)):
            if name in pairs:
                entry = pairs[name]
                row = {"metric": label, "base": entry["base"], "candidate": entry["candidate"],
                       "change_pct": entry["change_pct"], "ci95_pct": entry["ci95_pct"],
                       "low_pct": entry["low_pct"], "high_pct": entry["high_pct"], "pairs": entry["n"],
                       "higher_is_better": higher, "verdict": _bounded_verdict(entry, higher)}
                at = next((i + 1 for i, r in enumerate(rows) if r["metric"] == after), len(rows))
                rows.insert(at, row)
        result["paired"] = pairs
        result["note"] = (f"paired: {pairs['pairs']} requests in both windows, each the same prompt on both builds; "
                          f"{pairs['same_chunks']} streamed in as many chunks on both"
                          + (f"; {pairs['prompt_mismatches']} indices held different prompts (different prompt sets?)"
                             if pairs["prompt_mismatches"] else ""))
    elif base["workload"].get("mode") == "closed":
        windows = window_paired(base, candidate)
        if windows is not None:
            row = {"metric": "output tok/s, window-paired", "base": None, "candidate": None,
                   "change_pct": windows["change_pct"], "ci95_pct": windows["ci95_pct"],
                   "low_pct": windows["low_pct"], "high_pct": windows["high_pct"], "pairs": windows["n"],
                   "higher_is_better": True, "verdict": _bounded_verdict(windows, True)}
            at = next((i + 1 for i, r in enumerate(rows) if r["metric"] == "output tok/s"), len(rows))
            rows.insert(at, row)
            result["note"] = (f"window-paired: {windows['n']} windows, window k against window k (a run without "
                              "request tables); the faster build's windows drift onto later prompts")
    decomp = spec_decomposition(base["summary"], candidate["summary"])
    if decomp is not None:
        result["spec_decomposition"] = decomp
    return result


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
    decomp = result.get("spec_decomposition")
    if decomp is not None:
        line = (
            f"speculative log-tps: total {decomp['server_tps']['change_pct']:+.2f}% "
            f"(1+tau {decomp['acceptance_length']['change_pct']:+.2f}%, "
            f"steps/s {decomp['steps_per_s']['change_pct']:+.2f}%, "
            f"reqs/step {decomp['requests_per_step']['change_pct']:+.2f}%"
        )
        if "row_yield" in decomp and "verified_rows_per_step" in decomp:
            line += (
                f"; rows/step {decomp['verified_rows_per_step']['change_pct']:+.2f}%, "
                f"row yield {decomp['row_yield']['change_pct']:+.2f}%"
            )
        line += ")"
        lines.append(line)
    for warning in result["warnings"]:
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines)
