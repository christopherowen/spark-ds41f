#!/usr/bin/env python3
"""The layer-chain benchmark on the image's vLLM build and on the PDL build, in
alternation on one GPU: time per layer and whether every buffer's bits agree.

usage: pdl_chain_ab.py [ROUNDS]

The PDL build is the image's vLLM tree with the PDL branch's package files in place:
/tmp/pdl-tree/vllm holds symbolic links to the image's files, and the branch's files
(mounted under /b/pdl) are copied over theirs. Each round runs
benchmarks/kernels/benchmark_deepseek_v41_pdl_chain.py once per build, the image's
first in odd rounds and the PDL build's first in even rounds, each in a fresh process
on the bundle's TileLang cache. A probe first confirms which build each tree imports.

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
PDL_TREE = Path("/tmp/pdl-tree/vllm")
BRANCH = Path("/b/pdl")
BENCH = [
    "benchmarks/kernels/benchmark_deepseek_v41_pdl_chain.py",
    "--layers",
    "4",
    "--replays",
    "50",
]
PROBE = (
    "import inspect, vllm.models.deepseek_v4_1.tilelang.mhc as m; "
    "print('pdl.chain()' in inspect.getsource(m))"
)


def build_tree() -> list[str]:
    shutil.rmtree(PDL_TREE.parent, ignore_errors=True)
    PDL_TREE.parent.mkdir(parents=True)
    subprocess.run(["cp", "-as", str(IMAGE_TREE), str(PDL_TREE)], check=True)
    placed = []
    for source in sorted(p for p in BRANCH.rglob("*") if p.is_file()):
        target = PDL_TREE / source.relative_to(BRANCH)
        if target.is_symlink() or target.exists():
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        placed.append(str(source.relative_to(BRANCH)))
    return placed


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
    print("PDL build files:", ", ".join(build_tree()), flush=True)
    builds = {"image": IMAGE_TREE, "pdl": PDL_TREE}
    for name, tree in builds.items():
        probe = subprocess.run(
            [sys.executable, "-c", PROBE],
            cwd="/tmp",
            env=environment(tree),
            capture_output=True,
            text=True,
        )
        chained = probe.stdout.strip()
        print(
            f"{name} build imports chained mHC: {chained} {probe.stderr[-500:]}",
            flush=True,
        )
        if chained != str(name == "pdl"):
            raise SystemExit(f"the {name} tree does not import the {name} build")
    results = {name: [] for name in builds}
    for r in range(rounds):
        order = ("image", "pdl") if r % 2 == 0 else ("pdl", "image")
        for name in order:
            results[name].append(run(builds[name], f"round {r + 1}: {name}"))
    ok = True
    reference = results["image"][0]
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
            image = statistics.median(row[kind] for row in rows["image"])
            pdl = statistics.median(row[kind] for row in rows["pdl"])
            summary[kind] = {
                "image": round(image, 2),
                "pdl": round(pdl, 2),
                "saved": round(image - pdl, 2),
                "saved_percent": round(100 * (image - pdl) / image, 2),
            }
        summary["finite_and_replays_equal"] = good
        print(json.dumps(summary), flush=True)
        print(
            json.dumps(
                {
                    "bits": f"{shape}: image vs PDL build, all rounds",
                    "differs_at": differ,
                }
            ),
            flush=True,
        )
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
