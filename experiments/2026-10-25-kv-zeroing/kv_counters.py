#!/usr/bin/env python3
"""The server's counters since its boot, and each node's NaN reports, as a lab extra.

usage: kv_counters.py URL [--nodes dgx1,dgx2,dgx3,dgx4] [--container spark-ds41f]

- Counters from URL/metrics: requests whose logits held a NaN
  (``vllm:corrupted_requests``, counted only under VLLM_COMPUTE_NANS_IN_LOGITS, which
  every arm of this experiment sets; "counting" is false when the series is absent),
  preemptions, prefix-cache queries and hits, drafted and accepted tokens.
- With --nodes, each node's serving container log (running or stopped, over ssh):
  TileLang's device asserts ("Device assert failed: ..."; the MoE routers refuse NaN
  or infinite gate logits this way, the drafter's included), CUDA's device-side assert
  and illegal-address reports. A NaN that reaches no logits shows here.

Run last in a job, after the workloads. Prints one JSON line with a "summary"; exits 1
if the endpoint does not answer (the engine died), 0 otherwise. Stdlib only.
"""
import argparse
import json
import subprocess
import sys
import urllib.request

SERIES = {
    "nan_requests": ("vllm:corrupted_requests_total", "vllm:corrupted_requests"),
    "preemptions": ("vllm:num_preemptions_total",),
    "prefix_cache_queries": ("vllm:prefix_cache_queries_total",),
    "prefix_cache_hits": ("vllm:prefix_cache_hits_total",),
    "draft_tokens": ("vllm:spec_decode_num_draft_tokens_total",),
    "accepted_tokens": ("vllm:spec_decode_num_accepted_tokens_total",),
    "requests_finished": ("vllm:request_success_total",),
}
REPORTS = {
    "device_asserts": "Device assert failed",
    "nonfinite_gate_logits": "contain nan or inf",
    "cuda_device_asserts": "device-side assert triggered",
    "illegal_addresses": "illegal memory access",
}


def metrics(base: str) -> dict[str, float] | None:
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=60) as response:
            lines = response.read().decode().splitlines()
    except OSError:
        return None
    totals: dict[str, float] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def node_reports(node: str, container: str) -> dict[str, int | str]:
    """Each report's line count in the node's serving container log."""
    pattern = "|".join(REPORTS.values())
    # A missing container is an error; a log without reports is grep's status 1.
    command = (f"docker inspect {container} > /dev/null && "
               f"{{ docker logs {container} 2>&1 | grep -E '{pattern}' || test $? -eq 1; }}")
    result = subprocess.run(["ssh", "-o", "ConnectTimeout=10", node, command],
                            capture_output=True, text=True, timeout=300, check=False)
    if result.returncode != 0:
        return {"error": (result.stderr or result.stdout).strip()[:300]}
    lines = result.stdout.splitlines()
    counts: dict[str, int | str] = {key: sum(text in line for line in lines)
                                    for key, text in REPORTS.items()}
    if lines:
        counts["first"] = lines[0][:300]
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--nodes", default="")
    parser.add_argument("--container", default="spark-ds41f")
    args = parser.parse_args()
    totals = metrics(args.url.rstrip("/"))
    summary: dict = {"summary": "server counters", "answered": totals is not None}
    if totals is not None:
        summary["counting"] = any(name in totals for name in SERIES["nan_requests"])
        for key, names in SERIES.items():
            summary[key] = sum(totals.get(name, 0.0) for name in names)
    if args.nodes:
        summary["nodes"] = {node: node_reports(node, args.container)
                            for node in args.nodes.split(",") if node}
        summary["nan_reports"] = sum(
            int(counts.get(key, 0)) for counts in summary["nodes"].values()
            for key in REPORTS)
    print(json.dumps(summary))
    return 0 if totals is not None else 1


if __name__ == "__main__":
    sys.exit(main())
