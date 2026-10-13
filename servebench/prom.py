"""Server counters from a Prometheus text endpoint, summed over label sets.

An adapter maps the quantities the report uses to a server's metric names. With the vLLM
adapter a window's throughput splits into engine steps per second times output tokens
per step, and speculative decoding's acceptance is reported on its own: the step rate
is the engine's speed, and tokens per step moves with the text.
"""

import re
import urllib.request

VLLM = {
    "steps": "vllm:iteration_tokens_total_count",
    "step_tokens": "vllm:iteration_tokens_total_sum",
    "generation_tokens": "vllm:generation_tokens_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "preemptions": "vllm:num_preemptions_total",
    "requests_running": "vllm:num_requests_running",
    "requests_waiting": "vllm:num_requests_waiting",
}


def parse(text: str) -> dict[str, float]:
    """Every series summed by metric name (labels dropped); NaN and comments skipped."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_part, _, rest = line.partition(" ")
        if "{" in line:
            name_part = line[:line.index("{")]
            rest = line[line.rindex("}") + 1:]
        value = rest.strip().split(" ")[0] if rest.strip() else ""
        try:
            number = float(value)
        except ValueError:
            continue
        if number != number:  # NaN
            continue
        totals[name_part] = totals.get(name_part, 0.0) + number
    return totals


def by_label(text: str, name: str, label: str) -> dict[str, float]:
    """One metric's series summed by the value of one label (e.g. draft position)."""
    pattern = re.compile(r'(?:^|,)\s*' + re.escape(label) + r'="([^"]*)"')
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith(name + "{"):
            continue
        match = pattern.search(line[len(name) + 1:line.rindex("}")])
        try:
            number = float(line[line.rindex("}") + 1:].strip().split(" ")[0])
        except ValueError:
            continue
        if match is None or number != number:
            continue
        totals[match.group(1)] = totals.get(match.group(1), 0.0) + number
    return totals


def fetch(base_url: str, timeout: float = 10.0) -> str | None:
    """``base_url``/metrics as text, or None if unreachable."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - a missing endpoint only drops the server columns
        return None


def scrape(base_url: str, names: dict[str, str], timeout: float = 10.0) -> dict[str, float] | None:
    """The adapter's quantities from ``base_url``/metrics, or None if unreachable."""
    text = fetch(base_url, timeout)
    if text is None:
        return None
    series = parse(text)
    return {key: series[name] for key, name in names.items() if name in series}


def require_idle(base_url: str, purpose: str) -> None:
    """Raise unless the server reports no running or waiting requests."""
    counters = scrape(base_url, VLLM) or {}
    busy = counters.get("requests_running", 0) + counters.get("requests_waiting", 0)
    if busy:
        raise RuntimeError(f"server not idle ({busy:.0f} requests): {purpose} needs it to itself")


def window(start: dict | None, end: dict | None, seconds: float) -> dict:
    """Counter deltas over a window turned into rates and ratios."""
    if not start or not end or seconds <= 0:
        return {}
    delta = {key: end[key] - start[key] for key in end if key in start}
    out: dict[str, float] = {}
    if delta.get("steps"):
        out["steps_per_s"] = delta["steps"] / seconds
        if "generation_tokens" in delta:
            out["output_tokens_per_step"] = delta["generation_tokens"] / delta["steps"]
        if "step_tokens" in delta:
            out["scheduled_tokens_per_step"] = delta["step_tokens"] / delta["steps"]
    if "generation_tokens" in delta:
        out["generation_tps"] = delta["generation_tokens"] / seconds
    if "prompt_tokens" in delta:
        out["prompt_tps"] = delta["prompt_tokens"] / seconds
    if delta.get("drafts"):
        out["accepted_per_draft"] = delta.get("accepted_tokens", 0) / delta["drafts"]
        out["draft_tokens_per_draft"] = delta.get("draft_tokens", 0) / delta["drafts"]
    if "preemptions" in delta:
        out["preemptions"] = delta["preemptions"]
    return out
