#!/usr/bin/env python3
"""The c1 layer harness on the base build and a prototype build, in alternation on one
GPU: microseconds a layer and whether every buffer's bits agree.

usage: ab.py ROUNDS CASE [CASE ...]     (a CASE: "tp=4 live=3 ar=on", c1_layer.py's
                                         options; "on" passes a flag)

Each build is the image's vLLM tree with package files in place, as the PDL experiment's
pdl_chain_ab.py builds them: /tmp/c1-ab/base/vllm links to the image's files with the
base build's files (/b/base: those it changes over the image's tree) copied over
theirs; /tmp/c1-ab/<name>/vllm is the base build with a prototype's files
(/b/<name>: those it changes over the base) copied over those. Each round runs
c1_layer.py once per case and build, each in a fresh process on the bundle's TileLang
cache; the build order rotates by one a round.

Prints each run's result line, then per case the medians over rounds, base minus each
prototype in each round, and whether the digests agree. Exit status 0 only when every run
succeeded, every output is finite, every graph replay gave its eager bits, and every
case's digest is equal across builds and rounds.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

IMAGE_TREE = Path("/opt/spark-ds41f/candidate/vllm")
TREES = Path("/tmp/c1-ab")
BUNDLE = Path("/b")
BASE = BUNDLE / "base"
LAYER = BUNDLE / "c1_layer.py"
ROUTES = BUNDLE / "ref-woa1.json"
COMMON = ["--graph", "--replays", "30", "--mla", "--head", "--routes", str(ROUTES)]


def builds():
    others = sorted(d for d in BUNDLE.iterdir() if d.is_dir() and d != BASE and (d / "vllm").is_dir())
    return {"base": [BASE], **{d.name: [BASE, d] for d in others}}


def build_tree(name, layers):
    tree = TREES / name / "vllm"
    tree.parent.mkdir(parents=True)
    subprocess.run(["cp", "-as", str(IMAGE_TREE), str(tree)], check=True)
    for layer in layers:
        placed = []
        for source in sorted(p for p in layer.rglob("*") if p.is_file()):
            target = tree / source.relative_to(layer)
            if target.is_symlink() or target.exists():
                target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            placed.append(str(source.relative_to(layer)))
        print(f"{name} build, {layer}: {len(placed)} files", flush=True)
    return tree


def run(tree, label, case, out):
    env = dict(os.environ)
    rest = [p for p in env.get("PYTHONPATH", "").split(":") if p and p != str(IMAGE_TREE)]
    env["PYTHONPATH"] = ":".join([str(tree), *rest])
    options = [f"--{key}" if value == "on" else f"--{key}={value}"
               for key, value in (item.split("=") for item in case.split())]
    argv = [sys.executable, str(LAYER), "off", *options, *COMMON, "--out", out]
    print(f"=== {label}", flush=True)
    process = subprocess.run(argv, cwd=tree, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, timeout=5400)
    lines = process.stdout.splitlines()
    print("\n".join(lines[-4:]), flush=True)
    if process.returncode:
        print(process.stdout[-3000:], flush=True)
        raise SystemExit(f"{label}: exit {process.returncode}")
    return json.loads(next(line for line in reversed(lines) if line.startswith("{")))


def main():
    rounds = int(sys.argv[1])
    cases = sys.argv[2:]
    shutil.rmtree(TREES, ignore_errors=True)
    layers = builds()
    trees = {name: build_tree(name, parts) for name, parts in layers.items()}
    names = list(trees)
    results = {(case, name): [] for case in cases for name in names}
    for r in range(rounds):
        order = names[r % len(names):] + names[: r % len(names)]
        for case in cases:
            for name in order:
                out = f"/tmp/c1-ab-out/{r}-{name}-{case.replace(' ', '_').replace('=', '')}"
                results[(case, name)].append(run(trees[name], f"round {r + 1}, {case}: {name}", case, out))
    ok = True
    for case in cases:
        base = results[(case, "base")]
        digests = {res["digest"] for name in names for res in results[(case, name)]}
        good = all(res["finite"] and res["replay_equals_eager"] for name in names for res in results[(case, name)])
        ok = ok and good and len(digests) == 1
        summary = {"case": case, "distinct_mean": base[0]["distinct_mean"],
                   "base_us_per_layer": round(statistics.median(x["us_per_layer"] for x in base), 2)}
        for name in names[1:]:
            other = results[(case, name)]
            saved = [b["us_per_layer"] - o["us_per_layer"] for b, o in zip(base, other)]
            summary[name] = {
                "us_per_layer": round(statistics.median(x["us_per_layer"] for x in other), 2),
                "saved_us_per_layer": round(statistics.median(saved), 2),
                "saved_percent": round(100 * statistics.median(saved) / summary["base_us_per_layer"], 2),
                "rounds_saved": [round(s, 2) for s in saved],
            }
        summary["bits_equal"] = len(digests) == 1
        summary["finite_and_replays_equal"] = good
        print(json.dumps(summary), flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
