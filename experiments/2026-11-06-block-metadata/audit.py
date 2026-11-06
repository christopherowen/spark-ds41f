#!/usr/bin/env python3
"""Summarize pinned source identities and already-recorded GPU measurements."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def embedded_json(path, marker):
    text = path.read_text()
    start = text.index("{", text.index(marker))
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


def main(source, recorded_root, output):
    promoted = "b6de07aad5cb21f2cf8381036c1a9eaf471b07fc"
    candidate = "5f926df6d3126266a9c88c82c06544060008c33c"
    functions = {
        "vllm/v1/worker/gpu/tilelang_inputs.py": [
            "gather_block_tables_kernel", "gather_block_tables", "gather_chunks"],
        "vllm/v1/worker/gpu/block_table.py": [
            "append_block_ids", "apply_staged_writes", "gather_block_tables"],
        "vllm/v1/worker/gpu/buffer_utils.py": ["stage_write"],
    }
    identities = []
    for path, names in functions.items():
        for name in names:
            hashes = {}
            for ref in [promoted, candidate]:
                tree = ast.parse(git(source, "show", ref + ":" + path))
                function = next(n for n in ast.walk(tree)
                                if isinstance(n, ast.FunctionDef) and n.name == name)
                hashes[ref] = hashlib.sha256(ast.dump(function).encode()).hexdigest()
            identities.append({"path": path, "function": name, "ast_sha256": hashes,
                               "identical": len(set(hashes.values())) == 1})
    assert git(source, "rev-parse", "HEAD") == candidate
    assert all(x["identical"] for x in identities)
    root = Path(__file__).resolve().parent
    trace_dir = recorded_root / "results/private/determinism/request-inventory-5/ports"
    traces = []
    for path in sorted(trace_dir.glob("*-kernels-dgx*.json")):
        data = json.loads(path.read_text())
        for name, grid, block, count, total in data["kernels"]:
            if "gather_block_tables" in name:
                traces.append({"receipt": str(path.relative_to(recorded_root)),
                               "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "name": name, "grid": grid, "block": block,
                               "count": count, "total_ms": total,
                               "mean_us_per_call": 1000 * total / count,
                               "window_span_ms": data["span_ms"],
                               "fraction_of_window_percent": 100 * total / data["span_ms"],
                               "interval": None})
    assert len(traces) == 8, "Expected all four recorded ranks, decode and prefill"
    runner = embedded_json(root / "runs/recorded-integration-runner-26.log", "=== runner benchmark")
    attention = embedded_json(root / "runs/recorded-audit-attention-6.log", "=== live-page benchmark")
    records = {
        "kind": "source audit and historical receipts, not a new candidate benchmark",
        "deployment_base": "eb8a49f6d92231acaaf3f9fd63a4b7d3a4183854",
        "promoted_source": promoted,
        "audited_source": candidate,
        "audited_tree": git(source, "rev-parse", "HEAD^{tree}"),
        "functions": identities,
        "historical_runner_gather": runner["gather_block_tables"],
        "historical_attention_metadata": attention,
        "historical_serving_gather": traces,
        "limitations": [
            "Historical runs retain their original workload, source and uncertainty.",
            "No new serving run, boot/repeat boot, throughput or memory comparison.",
            "No full current 32K-to-1M c1/c8/c16 hardware matrix.",
            "Trace summaries have no per-call samples for a confidence interval.",
        ],
    }
    output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recorded-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.recorded_root, args.output)
