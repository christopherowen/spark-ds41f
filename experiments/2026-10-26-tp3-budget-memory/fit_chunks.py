#!/usr/bin/env python3
"""A prefill chunk's fixed cost, from one servebench prefill suite run at several budgets.

usage (on the head node, from the repository root):
  fit_chunks.py RUN LABEL=BUDGET [LABEL=BUDGET ...]

e.g. fit_chunks.py prefill-budget-tp3 b4096=4096 b4096-again=4096 b6144=6144 b8192=8192

Reads results/private/suites/RUN/LABEL/prefill{,-1m}.json. For each prompt size N it fits
TTFT = K_N + ceil(N / budget) * b by least squares over the arms: b is what every chunk
pays whatever its size (the chunk count is the only thing a budget changes for a
prompt), K_N everything else (the per-token work, tokenization, the first chunk's Engram
wait, the last chunk's decoder rows). Prints b and K_N per size and the worst residual.
"""
import argparse
import json
import math
import statistics
from pathlib import Path


def ttfts(path: Path) -> dict[int, tuple[int, float]]:
    found: dict[int, list[tuple[int, float]]] = {}
    if not path.exists():
        return {}
    for sample in json.loads(path.read_text())["samples"]:
        if not sample.get("error"):
            found.setdefault(sample["size"], []).append((sample["prompt_tokens"], sample["ttft_s"]))
    return {size: (runs[0][0], statistics.fmean(t for _, t in runs)) for size, runs in found.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("arms", nargs="+", help="LABEL=BUDGET")
    args = parser.parse_args()
    arms = [(label, int(budget)) for label, budget in (arm.split("=") for arm in args.arms)]
    data = {}
    for label, budget in arms:
        base = Path("results/private/suites") / args.run / label
        data[label] = (budget, {**ttfts(base / "prefill.json"), **ttfts(base / "prefill-1m.json")})
    sizes = sorted(set.intersection(*(set(d) for _, d in data.values())))
    for size in sizes:
        points = [(math.ceil(d[size][0] / budget), d[size][1]) for budget, d in data.values()]
        chunks = [c for c, _ in points]
        if len(set(chunks)) < 2:
            print(f"size {size}: every budget gives {chunks[0]} chunks; no fit")
            continue
        mx, my = statistics.fmean(chunks), statistics.fmean(t for _, t in points)
        b = sum((c - mx) * (t - my) for c, t in points) / sum((c - mx) ** 2 for c in chunks)
        k = my - b * mx
        worst = max(abs(t - k - b * c) for c, t in points)
        print(f"size {size}: b = {b * 1000:.1f} ms a chunk, K = {k:.2f} s, worst residual "
              f"{worst * 1000:.0f} ms over {len(points)} arms")


if __name__ == "__main__":
    main()
