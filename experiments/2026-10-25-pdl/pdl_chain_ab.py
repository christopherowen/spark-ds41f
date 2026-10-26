#!/usr/bin/env python3
"""The layer-chain benchmark on the base build and on the PDL build, in alternation
on one GPU: time per layer and whether every buffer's bits agree.

usage: pdl_chain_ab.py [ROUNDS]

Each build is the image's vLLM tree with package files in place: /tmp/pdl-tree/base/vllm
holds symbolic links to the image's files with the base build's files (mounted under
/b/base: the files it changes over the image's tree) copied over theirs, and
/tmp/pdl-tree/pdl/vllm is the base build with the PDL branch's files (/b/pdl: the files
it changes over the base) copied over those. Each round runs
benchmarks/kernels/benchmark_deepseek_v41_pdl_chain.py once per build, the base build's
first in odd rounds and the PDL build's first in even rounds, each in a fresh process
on the bundle's TileLang cache. A probe first confirms that each tree imports its own
mHC module, chained only in the PDL build.

Prints each run's lines, then one JSON line per shape (medians over rounds, the
difference) and a {"bits": ...} line per shape. Exit status 0 only when every run
succeeded, every output is finite, every replay gave its eager bits, and every
buffer's digest is equal across builds and runs.
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

IMAGE_TREE = Path("/opt/spark-ds41f/candidate/vllm")
TREES = Path("/tmp/pdl-tree")
# Each build's files, in the order they are copied over the image's tree.
LAYERS = {"base": [Path("/b/base")], "pdl": [Path("/b/base"), Path("/b/pdl")]}
MHC = "vllm/models/deepseek_v4_1/tilelang/mhc.py"
BENCH = [
    "benchmarks/kernels/benchmark_deepseek_v41_pdl_chain.py",
    "--layers",
    "4",
    "--replays",
    "50",
]
PROBE = (
    "import inspect, json, os, vllm.models.deepseek_v4_1.tilelang.mhc as m; "
    "print(json.dumps({'chained': 'pdl.chain()' in inspect.getsource(m), "
    "'file': os.path.realpath(m.__file__)}))"
)


def build_tree(name: str) -> Path:
    tree = TREES / name / "vllm"
    tree.parent.mkdir(parents=True)
    subprocess.run(["cp", "-as", str(IMAGE_TREE), str(tree)], check=True)
    for layer in LAYERS[name]:
        placed = []
        for source in sorted(p for p in layer.rglob("*") if p.is_file()):
            target = tree / source.relative_to(layer)
            if target.is_symlink() or target.exists():
                target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            placed.append(str(source.relative_to(layer)))
        print(f"{name} build, {layer}: {len(placed)} files: {', '.join(placed)}")
    return tree


def environment(tree: Path) -> dict:
    env = dict(os.environ)
    rest = [
        p for p in env.get("PYTHONPATH", "").split(":") if p and p != str(IMAGE_TREE)
    ]
    env["PYTHONPATH"] = ":".join([str(tree), *rest])
    return env


def run(tree: Path, label: str) -> list[dict]:
    print(f"=== {label}", flush=True)
    process = subprocess.run(
        [sys.executable, *BENCH],
        cwd=tree,
        env=environment(tree),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=5400,
    )
    print(process.stdout[-6000:], flush=True)
    if process.returncode:
        raise SystemExit(f"{label}: exit {process.returncode}")
    return [
        json.loads(line)
        for line in process.stdout.splitlines()
        if line.startswith('{"shape"')
    ]


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    shutil.rmtree(TREES, ignore_errors=True)
    builds = {name: build_tree(name) for name in LAYERS}
    for name, tree in builds.items():
        probe = subprocess.run(
            [sys.executable, "-c", PROBE],
            cwd="/tmp",
            env=environment(tree),
            capture_output=True,
            text=True,
        )
        print(f"{name} build imports: {probe.stdout.strip()} {probe.stderr[-500:]}")
        found = json.loads(probe.stdout or "{}")
        want = {"chained": name == "pdl", "file": str(tree / MHC)}
        if found != want:
            raise SystemExit(
                f"the {name} tree does not import the {name} build: {want}"
            )
    results = {name: [] for name in builds}
    for r in range(rounds):
        order = ("base", "pdl") if r % 2 == 0 else ("pdl", "base")
        for name in order:
            results[name].append(run(builds[name], f"round {r + 1}: {name}"))
    ok = True
    reference = results["base"][0]
    for i, want in enumerate(reference):
        shape = want["shape"]
        rows = {name: [runs[i] for runs in results[name]] for name in builds}
        differ = sorted(
            {
                key
                for name in builds
                for row in rows[name]
                for key, digest in row["digests"].items()
                if digest != want["digests"][key]
            }
        )
        good = all(
            row["finite"] and row["replay_equals_eager"]
            for name in builds
            for row in rows[name]
        )
        ok = ok and good and not differ
        summary = {"shape": shape, "rows": want["rows"]}
        for kind in ("warm_us", "cold_us"):
            base = statistics.median(row[kind] for row in rows["base"])
            pdl = statistics.median(row[kind] for row in rows["pdl"])
            summary[kind] = {
                "base": round(base, 2),
                "pdl": round(pdl, 2),
                "saved": round(base - pdl, 2),
                "saved_percent": round(100 * (base - pdl) / base, 2),
            }
        summary["finite_and_replays_equal"] = good
        print(json.dumps(summary), flush=True)
        print(
            json.dumps(
                {
                    "bits": f"{shape}: base vs PDL build, all rounds",
                    "differs_at": differ,
                }
            ),
            flush=True,
        )
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
