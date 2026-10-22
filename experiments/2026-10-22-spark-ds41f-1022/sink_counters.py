#!/usr/bin/env python3
"""The server's sink counters, as a lab extra.

usage: sink_counters.py URL

Reads vllm:sink_* from /metrics (labels kept) and prints one JSON line with a "summary": the
flagged rows by kind, the draft-zero rejections, the drafter cuts and the fallthroughs since
the server started. Run after the gate's ordinary workloads and before the collapse probes,
it shows whether ordinary text raised any flag. Stdlib only.
"""
import json
import sys
import urllib.request


def main() -> int:
    base = sys.argv[1].rstrip("/")
    with urllib.request.urlopen(base + "/metrics", timeout=60) as response:
        lines = response.read().decode().splitlines()
    counters = {}
    for line in lines:
        if line.startswith("vllm:sink_") and not line.startswith("#"):
            name, value = line.rsplit(" ", 1)
            counters[name] = float(value)

    def total(prefix: str) -> float:
        return sum(v for k, v in counters.items() if k.startswith(prefix) and "_created" not in k)

    flagged = {k: v for k, v in counters.items() if k.startswith("vllm:sink_flagged_rows_total")}
    summary = {
        "flagged_rows": total("vllm:sink_flagged_rows_total"),
        "draft_zero_rejections": total("vllm:sink_draft_zero_rejections_total"),
        "draft_cuts": total("vllm:sink_draft_cuts_total"),
        "fallthroughs": total("vllm:sink_fallthroughs_total"),
        "flagged_by_row": flagged,
        "series": len(counters),
    }
    print(json.dumps({"summary": summary}))
    return 0 if counters else 1


if __name__ == "__main__":
    sys.exit(main())
