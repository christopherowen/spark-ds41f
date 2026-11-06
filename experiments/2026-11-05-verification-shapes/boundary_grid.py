"""Record the actual model path at the repaired native verification shapes."""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from servebench import engine
from spark_ds41f import bench, lab


def validate_grid(raw, shapes, replays, workers):
    """Require every requested model step to reach the full graph and drafter."""
    if (
        raw.get("workers") != workers
        or raw.get("replays") != replays
        or raw.get("shapes") != shapes
        or len(raw.get("grid", [])) != len(shapes)
    ):
        raise RuntimeError("engine grid identity or shape count differs")
    for requested, timed in zip(shapes, raw["grid"]):
        expected = {k: v for k, v in requested.items() if k != "kind"}
        if timed.get("shape") != expected:
            raise RuntimeError("engine returned a different requested shape")
        mode = [True, requested["num_tokens"], requested["num_reqs"]]
        if timed.get("modes") != [mode] * replays:
            raise RuntimeError("engine did not record every full-graph model step")
        for key in ("step_ms", "forward_ms", "drafter_ms"):
            values = timed.get(key, [])
            if len(values) != replays or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in values
            ):
                raise RuntimeError(f"incomplete or invalid engine samples: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--cluster-config", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("preserve the existing run")
    hold = lab.read_hold()
    if not hold or hold.get("holder") != "spark-lab: verification capture policies":
        raise RuntimeError("this task does not own the window")
    shapes = [
        {"kind": "decode", "num_reqs": r, "num_tokens": r * 6, "context_len": c}
        for c in (4096, 65536)
        for r in (3, 5, 6, 7)
    ]
    record = {
        "label": args.label,
        "started": lab.now(),
        "shapes": shapes,
        "window": hold["window"],
        "scope": "dummy full-model steps; no scheduling or acceptance",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _, nodes, _ = lab.spark.configuration(args)
    halted = []
    watch = bench.MemoryWatch(nodes, 4, halted.append)
    fans = []
    # Preserve errors and partial metadata; never manufacture an empty successful grid.
    try:
        watch.wait_ready()
        record["cooling"] = bench.cool_nodes(nodes, nodes["nodes"], 55, 600)
        fans = bench.start_fan_curve(nodes, nodes["nodes"])
        record["fan_curve"] = fans
        problems = bench.fan_curve_problems(fans)
        if problems:
            raise RuntimeError(str(problems))
        raw = engine.run(args.url, shapes, replays=args.replays, warmup=2, settle=1)
        record["raw"] = raw
        validate_grid(raw, shapes, args.replays, len(nodes["nodes"]))
        record["engine"] = engine.summarize(raw)
        record["execution_evidence"] = {
            "full_graph_and_drafter_every_step": True,
            "dimensions": "live requests/tokens; graph padding is not reported",
        }
        watch.check()
        current = lab.read_hold()
        if not current or (current.get("holder"), current.get("window")) != (
            hold["holder"],
            hold["window"],
        ):
            raise RuntimeError("exact window ownership lost during the grid")
        if halted:
            raise RuntimeError(str(halted))
        if not raw["prefix_cache_reset"]:
            raise RuntimeError("dummy-step prefix cache reset failed")
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["thermal"] = watch.thermal_summary()
        record["memory"] = watch.close()
        failures = bench.stop_fan_curve(nodes, nodes["nodes"], fans)
        record["fan_stop_failures"] = failures
        if failures and "error" not in record:
            record["error"] = "fan policy restoration failed"
        record["ended"] = lab.now()
        args.out.write_text(json.dumps(record, indent=2) + "\n")
    if record.get("error"):
        raise RuntimeError(record["error"])


if __name__ == "__main__":
    main()
