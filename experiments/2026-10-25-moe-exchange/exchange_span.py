#!/usr/bin/env python3
"""Each sequence-parallel layer's MoE output stage in the profile arms' prefill traces:
from its first combine launch to the next sublayer entry, which reads the exchanged
parts.

usage: exchange_span.py RUN LABEL [LABEL...] [--node NAME]

RUN is the profile run (results/private/determinism/RUN, exchange-profile.json's
moe-exchange-profile); each LABEL's prefill-trace/NODE holds one torch-profiler trace a
scenario, the long prompt first. The step is the long prompt's prefill: from the first
vocab_embedding_kernel to the next. A layer's stage starts at its first moe_combine and
ends where the next mhc_project_streams, mhc_post or mhc_collapse starts; it is an SP
exchange stage when NCCL runs inside it. Per arm: the step, the stages' span (the time
the layer's output takes to reach its consumer), the combine launches and their time,
the device copies, and NCCL's time; then each arm's step and stages against the first
arm's. Before the change (26382f5cf) a stage is one combine, four copies, then the
exchange alone; after it, the combines interleave with the exchange and no copy is left.
The fusion survey's trace reading (ceilings.py), without its ceilings.
"""
import gzip
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFETCH = "l2_prefetch_kernel"
MARK = "vocab_embedding_kernel"
ENTRIES = ("mhc_project_streams", "mhc_post", "mhc_collapse")


def load(path: Path) -> list[tuple]:
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    ks = [(e["ts"], e["dur"], e["name"]) for e in events
          if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
          and e["name"] != PREFETCH]
    return sorted(ks)


def stages(path: Path) -> tuple[float, list[dict]]:
    """The prefill step's span (us) and its MoE output stages."""
    ks = load(path)
    marks = [k[0] for k in ks if k[2] == MARK]
    if len(marks) < 2:
        raise SystemExit(f"{path}: no complete step")
    rows = [k for k in ks if marks[0] <= k[0] < marks[1]]
    out, current = [], None
    for ts, dur, name in rows:
        if current is not None and name.startswith(ENTRIES):
            current["span"] = ts - current["start"]
            out.append(current)
            current = None
        if current is None and name.startswith("moe_combine"):
            current = {"start": ts, "combines": 0, "combine_us": 0.0, "copies": 0,
                       "copy_us": 0.0, "nccl_us": 0.0}
        if current is None:
            continue
        if name.startswith("moe_combine"):
            current["combines"] += 1
            current["combine_us"] += dur
        elif name.startswith("Memcpy DtoD"):
            current["copies"] += 1
            current["copy_us"] += dur
        elif name.startswith("ncclDev"):
            current["nccl_us"] += dur
    return marks[1] - marks[0], [s for s in out if s["nccl_us"] > 0]


def trace(run: str, label: str, node: str) -> Path:
    traces = sorted((ROOT / "results/private/determinism" / run / label / "prefill-trace" / node)
                    .glob("*.json.gz"))
    if not traces:
        raise SystemExit(f"no prefill trace for {label} on {node}")
    return traces[0]


def main(argv: list[str]) -> int:
    node = "dgx1"
    if "--node" in argv:
        index = argv.index("--node")
        node = argv[index + 1]
        argv = argv[:index] + argv[index + 2:]
    run, labels = argv[0], argv[1:]
    results = {}
    for label in labels:
        step, sp = stages(trace(run, label, node))
        total = lambda key: sum(s[key] for s in sp)  # noqa: E731
        results[label] = {
            "step_ms": round(step / 1000, 2),
            "sp_stages": len(sp),
            "stage_ms_total": round(total("span") / 1000, 2),
            "stage_ms_median": round(statistics.median(s["span"] for s in sp) / 1000, 3),
            "combines_a_stage": statistics.median(s["combines"] for s in sp),
            "combine_ms_total": round(total("combine_us") / 1000, 2),
            "copies": total("copies"),
            "copy_ms_total": round(total("copy_us") / 1000, 2),
            "nccl_ms_total": round(total("nccl_us") / 1000, 2),
        }
    print(json.dumps({"node": node, "arms": results}, indent=1))
    base = results[labels[0]]
    for label in labels[1:]:
        r = results[label]
        step = r["step_ms"] - base["step_ms"]
        stage = r["stage_ms_total"] - base["stage_ms_total"]
        print(f"{label} against {labels[0]} on {node}: step {step:+.2f} ms "
              f"({100 * step / base['step_ms']:+.2f}%), SP MoE stages {stage:+.2f} ms, "
              f"copies {base['copies']} -> {r['copies']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
