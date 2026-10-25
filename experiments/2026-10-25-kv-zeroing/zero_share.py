#!/usr/bin/env python3
"""The KV block zeroing kernel's share of each profiled step, by scenario.

usage: zero_share.py RUN_DIR LABEL [--node dgx1]

RUN_DIR/LABEL holds the request inventory's traces (decode-trace/<node>/ and
prefill-trace/<node>/, one profiler window a scenario, in the order decode.json and
prefill.json list them; experiments/2026-10-25-request-inventory). Steps are cut as the
kernel audit cuts them (experiments/2026-10-25-kernel-audit, kernel_shares.py): from one
``execute_context_*`` annotation on the eager stream to the next, dummy steps dropped,
the step with the scenario's most context tokens its prefill. For each scenario and kind
of step it prints the steps, the ``zero_kv_blocks`` launches, their time, the steps'
busy time (the union of kernel, copy and fill intervals on every stream, the L2 prefetch
side stream excluded) and the zeroing's share of it. Stdlib only.
"""
import collections
import gzip
import json
import re
import sys
from pathlib import Path

PREFETCH = {"l2_prefetch_kernel"}


def scenario_traces(run: Path, label: str, node: str, workload: str) -> dict[str, Path]:
    lines = [json.loads(line) for line in (run / label / f"{workload}.json").read_text().splitlines()
             if line.startswith("{")]
    names = [line["scenario"] for line in lines if line.get("status", 200) == 200]
    traces = sorted((run / label / f"{workload}-trace" / node).glob("*.json.gz"))
    if len(names) != len(traces):
        raise SystemExit(f"{workload}: {len(names)} scenarios but {len(traces)} traces")
    return dict(zip(names, traces))


def load(trace: Path):
    events = json.load(gzip.open(trace))["traceEvents"]
    kernels = sorted((e["ts"], e["dur"], e["name"]) for e in events if e.get("ph") == "X"
                     and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"))
    annotations = [e for e in events if e.get("cat") == "gpu_user_annotation"
                   and e["name"].startswith("execute_context")]
    stream = collections.Counter(e["tid"] for e in annotations).most_common(1)[0][0]
    steps = sorted((e["ts"], e["name"]) for e in annotations if e["tid"] == stream)
    end = max(k[0] + k[1] for k in kernels)
    windows = []
    for i, (start, name) in enumerate(steps):
        m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", name)
        context, generation = int(m.group(1)), int(m.group(3))
        if context == 0 and generation == 0:
            continue
        stop = steps[i + 1][0] if i + 1 < len(steps) else end
        windows.append((start, stop, int(m.group(2)) if context else 0))
    most = max(tokens for _, _, tokens in windows)
    windows = [(a, b, "prefill" if tokens and tokens == most else "decode" if not tokens
                else "other") for a, b, tokens in windows]
    return kernels, windows


def busy(kernels) -> float:
    total, start, stop = 0.0, None, None
    for begin, end in sorted((k[0], k[0] + k[1]) for k in kernels if k[2] not in PREFETCH):
        if stop is None or begin > stop:
            if stop is not None:
                total += stop - start
            start, stop = begin, end
        else:
            stop = max(stop, end)
    return total + (stop - start if stop is not None else 0.0)


def main() -> int:
    run, label = Path(sys.argv[1]), sys.argv[2]
    node = sys.argv[sys.argv.index("--node") + 1] if "--node" in sys.argv else "dgx1"
    for workload in ("decode", "prefill"):
        for name, trace in scenario_traces(run, label, node, workload).items():
            kernels, windows = load(trace)
            acc = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])
            for a, b, kind in windows:
                inside = [k for k in kernels if a <= k[0] < b]
                zero = [k for k in inside if "zero_kv_blocks" in k[2]]
                row = acc[kind]
                row[0] += 1
                row[1] += len(zero)
                row[2] += sum(k[1] for k in zero)
                row[3] += busy(inside)
            for kind, (steps, launches, zero_us, busy_us) in sorted(acc.items()):
                print(f"{workload:7s} {name:16s} {kind:8s} steps {steps:3d}  launches {launches:3d}  "
                      f"zeroing {zero_us:8.1f} us  busy {busy_us / 1e3:8.1f} ms  "
                      f"share {100 * zero_us / busy_us if busy_us else 0:.3f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
