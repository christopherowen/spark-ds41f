#!/usr/bin/env python3
"""The core metrics of each recorded release, from the suite runs it already keeps.

usage: core_metrics.py [SUITE_DIR ...] [--json OUT]

With no directories, reads every manifests/suites/<date>-<baseline>-<shape>[-single-stream|-sampled]
and groups a release's three suites. Prints one card per release and shape and, with --json,
writes them. Standard library only; nothing is measured, every figure is derived from the
records:

- decode: steps a second as step time, tokens a step, and from the speculative counters the
  decoding requests a step, verified rows a step, row yield (tokens a verified row) and
  wasted rows (verified rows that produced no token);
- the grid-to-serving gap: the serving step against the engine grid's step at the same
  requests and rows (bilinear in requests and rows a request). The grid times identical
  dummy rows, so the gap is what it cannot see: the distinct experts real text reaches,
  real sampling, the host between steps and Engram reads;
- the grid's own split: target forward, drafter, and the rest of the step;
- first token: short-prompt TTFT in serving steps, Poisson tails and goodput;
- prefill: tok/s and time to first token by size, the marginal cost of each further
  thousand tokens, the context tax (the rate at the smallest size over the largest), and a
  least-squares fit TTFT = a + b N + c N^2;
- memory and power, from conditions.json: each node's lowest MemAvailable and its headroom
  over the steady memory guard (3 GiB), swap growth, and a lower bound on tokens a joule
  from the highest GPU power each node reported (sampled every 15 s, only the maximum kept).
"""
import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITES = ROOT / "manifests/suites"
STEADY_GUARD_GIB = 3.0  # config/cluster.json host.memguard_gib
NAME = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<baseline>.+)-(?P<shape>tp\d)(?:-(?P<suite>single-stream|sampled))?$")


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def mean(tree: dict | None, *path):
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    return tree.get("mean") if isinstance(tree, dict) else tree


def interval(values: list[float]) -> tuple[float | None, float | None]:
    """Mean and the 95% t half-width as % of it, over per-window values."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return (values[0] if values else None), None
    m = statistics.fmean(values)
    t = {9: 2.262, 8: 2.306, 7: 2.365, 6: 2.447, 5: 2.571, 4: 2.776, 3: 3.182, 2: 4.303, 1: 12.706}.get(len(values) - 1, 2.0)
    return m, 100 * t * statistics.stdev(values) / math.sqrt(len(values)) / m


def decode(run: dict) -> dict:
    """Throughput, step time and the row accounting of one closed-loop or Poisson run."""
    s = run["summary"]
    batches = s.get("server_batches", {})
    step_ms, step_ci = interval([1000 / v for v in batches.get("steps_per_s", []) if v])
    tokens = mean(s, "server", "output_tokens_per_step")
    accepted = mean(s, "server", "accepted_per_draft")
    drafted = mean(s, "server", "draft_tokens_per_draft")
    out = {"tps": mean(s, "output_tps"), "tps_ci_pct": (s.get("output_tps") or {}).get("ci95_pct"),
           "step_ms": step_ms, "step_ci_pct": step_ci, "tokens_per_step": tokens,
           "accepted_per_draft": accepted, "verified_per_draft": drafted,
           "prefix_cache_hit_rate": mean(s, "server", "prefix_cache_hit_rate"),
           "preemptions": mean(s, "server", "preemptions"),
           "concurrency": run["workload"].get("concurrency") if run["workload"].get("mode") == "closed" else None,
           "inflight_mean": s.get("inflight_mean")}
    if tokens and accepted is not None and drafted is not None:
        # Every decoding request drafts every step, so each contributes 1 + accepted tokens
        # and 1 + verified rows (the target row and its verified drafts).
        requests = tokens / (1 + accepted)
        rows = requests * (1 + drafted)
        out |= {"decoding_requests_per_step": requests, "verified_rows_per_step": rows,
                "row_yield": tokens / rows, "wasted_rows_per_step": rows - tokens,
                "accepted_per_verified": accepted / drafted if drafted else None}
    latency = s.get("latency", {})
    out |= {"ttft_p50_ms": 1000 * latency["ttft_s"]["p50"] if latency.get("ttft_s", {}).get("n") else None,
            "ttft_p99_ms": 1000 * latency["ttft_s"]["p99"] if latency.get("ttft_s", {}).get("n") else None,
            "tpot_p50_ms": latency.get("tpot_ms", {}).get("p50"),
            "tpot_p99_ms": latency.get("tpot_ms", {}).get("p99")}
    if out["ttft_p50_ms"] and step_ms:
        out["ttft_p50_steps"] = out["ttft_p50_ms"] / step_ms
    slo = latency.get("slo")
    if slo:
        out |= {"goodput_rps": slo.get("goodput_rps"), "slo_attainment_pct": slo.get("attainment_pct")}
    return out


def grid_table(grid: dict) -> dict[tuple[int, int, int], dict]:
    """(requests, rows a request, context) -> step, forward and drafter ms."""
    table = {}
    for entry in grid["engine"]:
        shape = entry["shape"]
        if shape.get("kind") != "decode":
            continue
        requests = shape["num_reqs"]
        table[(requests, shape["num_tokens"] // requests, shape["context_len"])] = {
            "step_ms": entry["step_ms"]["mean"], "forward_ms": (entry.get("forward_ms") or {}).get("mean"),
            "drafter_ms": (entry.get("drafter_ms") or {}).get("mean")}
    return table


def grid_step(table: dict, requests: float, rows: float, context: int = 4096) -> float | None:
    """The grid's step at fractional requests and rows a request: bilinear between the
    nearest measured shapes at ``context`` (linear beyond the last ones)."""
    have = {(r, k): v["step_ms"] for (r, k, c), v in table.items() if c == context}
    reqs = sorted({r for r, _ in have if all((r, k) in have for k in (2, 3, 6))})
    if not reqs:
        return None

    def bracket(points: list[int], x: float) -> tuple[int, int]:
        if x <= points[0]:
            return points[0], points[1] if len(points) > 1 else points[0]
        for low, high in zip(points, points[1:]):
            if x <= high:
                return low, high
        return points[-2], points[-1]

    def along(fixed: dict[int, float], x: float) -> float:
        keys = sorted(fixed)
        low, high = bracket(keys, x)
        if low == high:
            return fixed[low]
        return fixed[low] + (fixed[high] - fixed[low]) * (x - low) / (high - low)

    by_request = {r: along({k: have[(r, k)] for k in (2, 3, 6)}, rows) for r in reqs}
    return along(by_request, requests) if len(reqs) > 1 else by_request[reqs[0]]


def grid_card(grid: dict) -> dict:
    table = grid_table(grid)
    out = {}
    for key, label in (((1, 3, 4096), "1x3@4K"), ((1, 6, 4096), "1x6@4K"), ((16, 3, 4096), "16x48@4K"),
                       ((8, 3, 4096), "8x24@4K")):
        if key in table:
            v = table[key]
            rest = (v["step_ms"] - v["forward_ms"] - v["drafter_ms"]) if v["forward_ms"] and v["drafter_ms"] else None
            out[label] = {"step_ms": v["step_ms"], "forward_ms": v["forward_ms"], "drafter_ms": v["drafter_ms"],
                          "rest_ms": rest}
    ones = {k: v["step_ms"] for (r, k, c), v in table.items() if r == 1 and c == 4096}
    if 2 in ones and 6 in ones:
        out["ms_per_identical_row_1req"] = (ones[6] - ones[2]) / 4
    longest = max((c for (r, k, c) in table if r == 1 and k == 3), default=None)
    if longest and (1, 3, 4096) in table and longest > 4096:
        out["context_slope_pct"] = 100 * (table[(1, 3, longest)]["step_ms"] / table[(1, 3, 4096)]["step_ms"] - 1)
        out["context_slope_at"] = longest
    return out


def solve3(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination for a 3x3 system."""
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for i in range(3):
        pivot = max(range(i, 3), key=lambda r: abs(m[r][i]))
        m[i], m[pivot] = m[pivot], m[i]
        for r in range(i + 1, 3):
            f = m[r][i] / m[i][i]
            m[r] = [x - f * y for x, y in zip(m[r], m[i])]
    x = [0.0] * 3
    for i in (2, 1, 0):
        x[i] = (m[i][3] - sum(m[i][j] * x[j] for j in range(i + 1, 3))) / m[i][i]
    return x


def prefill_card(run: dict, budget: int | None) -> dict:
    samples = [s for s in run["samples"] if not s.get("error") and s.get("ttft_s")]
    sizes = []
    for row in run["prefill"]:
        ttfts = [s["ttft_s"] for s in samples if s["size"] == row["size"]]
        sizes.append({"size": row["size"], "prompt_tokens": row["prompt_tokens"], "tps": row["tps"].get("mean"),
                      "tps_ci_pct": row["tps"].get("ci95_pct"), "ttft_s": statistics.fmean(ttfts) if ttfts else None,
                      "chunks": math.ceil(row["prompt_tokens"] / budget) if budget and row["prompt_tokens"] else None})
    for low, high in zip(sizes, sizes[1:]):
        if low["ttft_s"] and high["ttft_s"]:
            high["marginal_ms_per_1k"] = 1000 * (high["ttft_s"] - low["ttft_s"]) / ((high["prompt_tokens"] - low["prompt_tokens"]) / 1000)
    out = {"sizes": sizes}
    if sizes and sizes[0]["tps"] and sizes[-1]["tps"]:
        out["context_tax_pct"] = 100 * (sizes[0]["tps"] / sizes[-1]["tps"] - 1)
    points = [(s["prompt_tokens"] / 1e6, s["ttft_s"]) for s in samples if s.get("prompt_tokens")]
    if len({s["size"] for s in samples if s.get("prompt_tokens")}) >= 3:
        # Normal equations for TTFT = a + b N + c N^2 (N in millions of tokens).
        sums = [sum(n ** k for n, _ in points) for k in range(5)]
        rhs = [sum(t * n ** k for n, t in points) for k in range(3)]
        a, b, c = solve3([[sums[i + j] for j in range(3)] for i in range(3)], rhs)
        residual = max(abs(t - a - b * n - c * n * n) for n, t in points)
        out["fit"] = {"a_s": a, "b_ms_per_1k": b, "c_ms_per_1k_per_1m": 2 * c,
                      "worst_residual_ms": 1000 * residual,
                      "note": "TTFT = a + b N + c N^2; b is the first token's cost of 1K more tokens at no context, "
                              "c_ms_per_1k_per_1m how much that cost rises per 1M of context"}
    return out


def conditions_card(conditions: dict | None, tps: dict[str, float | None]) -> dict:
    if not conditions:
        return {}
    memory = conditions.get("memory", {})
    thermal = conditions.get("thermal", {})
    nodes = sorted(set(memory) | set(thermal))
    out = {"nodes": {}}
    for node in nodes:
        m, t = memory.get(node, {}), thermal.get(node, {})
        low = m.get("min_available_gib")
        out["nodes"][node] = {"min_available_gib": low,
                              "headroom_over_guard_gib": low - STEADY_GUARD_GIB if low is not None else None,
                              "swap_growth_gib": m.get("swap_growth_gib"), "max_power_w": t.get("max_power_w"),
                              "max_gpu_c": t.get("max_gpu_c"), "thermal_slowdown_ms": t.get("thermal_slowdown_ms"),
                              "power_cap_ms": t.get("power_cap_ms")}
    lows = [v["min_available_gib"] for v in out["nodes"].values() if v["min_available_gib"] is not None]
    if lows:
        out["lowest_node"] = min(out["nodes"], key=lambda n: out["nodes"][n]["min_available_gib"] or 1e9)
        out["lowest_available_gib"] = min(lows)
    power = [v["max_power_w"] for v in out["nodes"].values() if v["max_power_w"]]
    if power:
        out["max_power_sum_w"] = sum(power)
        out["tokens_per_joule_at_least"] = {step: value / sum(power) for step, value in tps.items() if value}
    out["halted"] = conditions.get("halted")
    return out


def budget_for(shape: str) -> int | None:
    recipe = load(ROOT / "config/cluster.json") or {}
    name = {"tp4": "tp4-ring", "tp3": "tp3-triangle"}.get(shape)
    return ((recipe.get("shapes", {}).get(name) or {}).get("sizing") or {}).get("max_num_batched_tokens")


def card(base: Path | None, single: Path | None, sampled: Path | None, shape: str) -> dict:
    out: dict = {"suites": {}}
    tps: dict[str, float | None] = {}
    if base:
        suite = load(base / "suite.json") or {}
        out["label"] = suite.get("label")
        out["started_utc"] = suite.get("started_utc")
        out["suites"]["release"] = str(base.relative_to(ROOT))
        grid = load(base / "grid.json")
        table = grid_table(grid) if grid else {}
        if grid:
            out["grid"] = grid_card(grid)
        for step in ("c16", "c8", "c4", "poisson"):
            run = load(base / f"{step}.json")
            if not run:
                continue
            out[step] = decode(run)
            tps[step] = out[step]["tps"]
            d = out[step]
            if table and d.get("decoding_requests_per_step") and step != "poisson":
                predicted = grid_step(table, d["decoding_requests_per_step"],
                                      d["verified_rows_per_step"] / d["decoding_requests_per_step"])
                if predicted and d["step_ms"]:
                    d["grid_step_ms"] = predicted
                    d["grid_gap_ms"] = d["step_ms"] - predicted
        prefill = load(base / "prefill.json")
        if prefill:
            out["prefill"] = prefill_card(prefill, budget_for(shape))
        out["conditions"] = conditions_card(load(base / "conditions.json"), tps)
    if single is None and base and (base / "c1.json").exists():
        single = base  # a promotion suite that ran its single stream itself (1022)
    if single:
        out["suites"]["single-stream"] = str(single.relative_to(ROOT))
        run = load(single / "c1.json")
        if run:
            out["c1"] = decode(run)
            grid = load(base / "grid.json") if base else None
            d = out["c1"]
            if grid and d.get("decoding_requests_per_step"):
                predicted = grid_step(grid_table(grid), 1, d["verified_rows_per_step"] / d["decoding_requests_per_step"])
                if predicted:
                    d["grid_step_ms"] = predicted
                    d["grid_gap_ms"] = d["step_ms"] - predicted
            if single is not base:  # else the power maximum covers the whole release suite
                out["c1_conditions"] = conditions_card(load(single / "conditions.json"), {"c1": d["tps"]})
    if sampled:
        out["suites"]["sampled"] = str(sampled.relative_to(ROOT))
        out["sampled"] = {step: decode(run) for step in ("c16", "c8", "c4", "c1")
                          if (run := load(sampled / f"{step}.json"))}
    return out


def collect(dirs: list[Path]) -> dict[tuple[str, str, str], dict[str, Path]]:
    groups: dict[tuple[str, str, str], dict[str, Path]] = {}
    for path in dirs:
        match = NAME.match(path.name)
        if not match:
            print(f"skipped {path}: not <date>-<baseline>-<shape>[-suite]", file=sys.stderr)
            continue
        key = (match["date"], match["baseline"], match["shape"])
        groups.setdefault(key, {})[match["suite"] or "release"] = path
    return groups


def fmt(value, spec: str = ".1f", width: int = 8) -> str:
    if not isinstance(value, (int, float)):
        return f"{'-':>{width}}"
    sign, spec = ("+", spec[1:]) if spec.startswith("+") else ("", spec)
    return f"{value:{sign}{width}{spec}}"


def print_cards(cards: dict) -> None:
    for shape in ("tp4", "tp3"):
        rows = [(k, v) for k, v in cards.items() if v["shape"] == shape]
        if not rows:
            continue
        print(f"\n=== {shape.upper()}: one stream (single-stream suite, greedy, distinct prompts)")
        print(f"{'release':28s} {'tok/s':>8} {'step ms':>8} {'tok/stp':>8} {'rows':>8} {'yield':>8} "
              f"{'wasted':>8} {'grid ms':>8} {'gap ms':>8} {'TTFT ms':>8} {'in steps':>8} {'tok/J>=':>8}")
        for name, c in rows:
            d = c.get("c1")
            if not d:
                continue
            tpj = (c.get("c1_conditions") or {}).get("tokens_per_joule_at_least", {}).get("c1")
            print(f"{name:28s} {fmt(d['tps'])} {fmt(d['step_ms'], '.2f')} {fmt(d.get('tokens_per_step'), '.3f')} "
                  f"{fmt(d.get('verified_rows_per_step'), '.2f')} {fmt(d.get('row_yield'), '.3f')} "
                  f"{fmt(d.get('wasted_rows_per_step'), '.2f')} {fmt(d.get('grid_step_ms'), '.2f')} "
                  f"{fmt(d.get('grid_gap_ms'), '.2f')} {fmt(d.get('ttft_p50_ms'), '.0f')} "
                  f"{fmt(d.get('ttft_p50_steps'), '.2f')} {fmt(tpj, '.3f')}")
        top = "c16" if shape == "tp4" else "c8"
        print(f"\n=== {shape.upper()}: {top} closed loop (release suite)")
        print(f"{'release':28s} {'tok/s':>8} {'step ms':>8} {'tok/stp':>8} {'req/stp':>8} {'rows':>8} {'yield':>8} "
              f"{'grid ms':>8} {'gap ms':>8} {'TTFT50':>8} {'TTFT99':>8} {'tok/J>=':>8}")
        for name, c in rows:
            d = c.get(top)
            if not d:
                continue
            tpj = (c.get("conditions") or {}).get("tokens_per_joule_at_least", {}).get(top)
            print(f"{name:28s} {fmt(d['tps'])} {fmt(d['step_ms'], '.2f')} {fmt(d.get('tokens_per_step'), '.2f')} "
                  f"{fmt(d.get('decoding_requests_per_step'), '.2f')} {fmt(d.get('verified_rows_per_step'), '.1f')} "
                  f"{fmt(d.get('row_yield'), '.3f')} {fmt(d.get('grid_step_ms'), '.2f')} {fmt(d.get('grid_gap_ms'), '.2f')} "
                  f"{fmt(d.get('ttft_p50_ms'), '.0f')} {fmt(d.get('ttft_p99_ms'), '.0f')} {fmt(tpj, '.3f')}")
        print(f"\n=== {shape.upper()}: Poisson (release suite)")
        print(f"{'release':28s} {'tok/s':>8} {'TTFT50':>8} {'TTFT99':>8} {'TPOT50':>8} {'TPOT99':>8} {'goodput':>8} {'attain%':>8}")
        for name, c in rows:
            d = c.get("poisson")
            if d:
                print(f"{name:28s} {fmt(d['tps'])} {fmt(d.get('ttft_p50_ms'), '.0f')} {fmt(d.get('ttft_p99_ms'), '.0f')} "
                      f"{fmt(d.get('tpot_p50_ms'), '.1f')} {fmt(d.get('tpot_p99_ms'), '.1f')} "
                      f"{fmt(d.get('goodput_rps'), '.3f')} {fmt(d.get('slo_attainment_pct'), '.1f')}")
        print(f"\n=== {shape.upper()}: grid split at 1 request x 3 rows, 4K (target forward, drafter, rest)")
        print(f"{'release':28s} {'step':>8} {'forward':>8} {'drafter':>8} {'rest':>8} {'ms/row':>8} {'ctx %':>8}")
        for name, c in rows:
            g = c.get("grid", {})
            v = g.get("1x3@4K")
            if v:
                print(f"{name:28s} {fmt(v['step_ms'], '.2f')} {fmt(v['forward_ms'], '.2f')} {fmt(v['drafter_ms'], '.2f')} "
                      f"{fmt(v['rest_ms'], '.2f')} {fmt(g.get('ms_per_identical_row_1req'), '.3f')} "
                      f"{fmt(g.get('context_slope_pct'), '+.1f')}")
        print(f"\n=== {shape.upper()}: prefill (release suite)")
        print(f"{'release':28s} {'size':>9} {'tok/s':>8} {'TTFT s':>8} {'chunks':>7} {'ms/1K':>8}   fit a s, b ms/1K, +ms/1K per 1M, worst ms; context tax")
        for name, c in rows:
            p = c.get("prefill")
            if not p:
                continue
            for i, s in enumerate(p["sizes"]):
                tail = ""
                if i == 0 and p.get("fit"):
                    f = p["fit"]
                    tail = (f"   {f['a_s']:.2f}, {f['b_ms_per_1k']:.1f}, {f['c_ms_per_1k_per_1m']:+.1f}, "
                            f"{f['worst_residual_ms']:.0f}; {p.get('context_tax_pct', 0):+.1f}%")
                print(f"{name if i == 0 else '':28s} {s['size']:9,d} {fmt(s['tps'], '.0f')} {fmt(s['ttft_s'], '.2f')} "
                      f"{fmt(s['chunks'], 'd', 7)} {fmt(s.get('marginal_ms_per_1k'), '.1f')}{tail}")
        print(f"\n=== {shape.upper()}: memory and power over the release suite (lowest node; steady guard {STEADY_GUARD_GIB:.0f} GiB)")
        print(f"{'release':28s} {'node':>6} {'low GiB':>8} {'headroom':>8} {'swap':>6} {'sum W':>7}  per-node lows")
        for name, c in rows:
            k = c.get("conditions") or {}
            if not k.get("nodes"):
                print(f"{name:28s} {'-':>6}")
                continue
            lows = " ".join(f"{n}:{v['min_available_gib']:.2f}" for n, v in k["nodes"].items() if v["min_available_gib"] is not None)
            low = k.get("lowest_available_gib")
            swap = max((v["swap_growth_gib"] or 0) for v in k["nodes"].values())
            print(f"{name:28s} {k.get('lowest_node', '-'):>6} {fmt(low, '.2f')} {fmt(low - STEADY_GUARD_GIB if low else None, '.2f')} "
                  f"{fmt(swap, '.2f', 6)} {fmt(k.get('max_power_sum_w'), '.0f', 7)}  {lows}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="*", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    dirs = [d.resolve() for d in args.dirs] or sorted(p for p in SUITES.iterdir() if p.is_dir())
    cards = {}
    for (date, baseline, shape), suites in sorted(collect(dirs).items()):
        name = f"{baseline}-{shape}"
        cards[name] = {"date": date, "baseline": baseline, "shape": shape,
                       **card(suites.get("release"), suites.get("single-stream"), suites.get("sampled"), shape)}
    print_cards(cards)
    if args.json:
        args.json.write_text(json.dumps(cards, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
