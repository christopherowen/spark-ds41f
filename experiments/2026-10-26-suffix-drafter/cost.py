#!/usr/bin/env python3
"""What a verification row costs, and what each copy-drafting variant nets, at TP4 and TP3.

usage: cost.py [--routes runs/routes.json] [--proxy runs/agent-proxy.json]
               [--cases runs/copy-cases.json] [--out results.json]

A step, per rank (the c1 model of experiments/2026-10-26-c1-100tps, PR #598, extended to wide
blocks, to TP3 and to eight streams):

    step = overhead + (dense + drafter_routed + 40 * U * expert) / rate
           + (rows + padding - suite rows) * row

- U is the distinct routed experts a target layer reads. One request's r consecutive rows reach
  N(r); eight independent requests reach E * (1 - prod(1 - N_i / E)) of the E = 384 experts.
- dense and expert are bytes a rank, from floor.py's shapes (#598) with the TP3 shards 4/3 as
  wide (Q-A/KV, router, mHC fn, compressors and the Markov head replicated).
- rate is FC1's marginal stream rate, 266.6 GB/s (#598's expert-rate bundle on dgx4).
- row is the engine grid's price of a row that reaches no new expert (identical dummy
  tokens): 0.366 ms at TP4 (r6v static table, #598), 0.435 ms at TP3 (1022 grid, 1x2 -> 1x6
  rows). Every row past the release suite's (4.07 at c1, 8 x 2.97 at c8) pays it, routed or
  not; padding is the rows a step's CUDA graph adds to reach its capture size.
- overhead is solved per shape and concurrency from the release suites (1022 c1 on both
  shapes, 1025 c8 at TP4 and 1022 c8 at TP3), at #598's split of today's rows into live and
  dead (2.89 live of 4.07 at c1).
- The wide-verify bundle (runs/wide-verify-dgx4.txt) checks the expert price at wide blocks:
  FC1 and FC2 keep their per-expert time from 6 to 65 rows, with the rows spread evenly or
  piled on three experts.

N(r) has three sources: the reference routes of copied and ordinary text (routes_probe.py, when
its run is present), json-count's eight-token reference routes extended at 2.4 experts a row
(#598's extrapolation), and independent uniform routing (an upper bound).
"""
import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from recipe import capture_sizes  # noqa: E402

JSON_ROUTES = HERE.parent / "2026-10-18-json-count" / "runs" / "ref-woa1.json"
E, TOP, LAYERS, DRAFTER_LAYERS = 384, 6, 40, 3
RATE = 266.6e6  # bytes a ms
DRAFTER_EXPERTS = 9.2  # distinct experts a drafter layer, the traced c1 window (#598)
WIDTHS = (6, 17, 33, 49, 65)

# Release suites: steps a second, tokens a step, verified drafts a request-step, requests
SUITES = {
    ("tp4", 1): (32.733, 2.2708, 3.0733, 1),   # manifests/suites/2026-10-22-spark-ds41f-1022-tp4/c1.json
    ("tp3", 1): (26.30, 2.263, 3.064, 1),      # manifests/suites/2026-10-22-spark-ds41f-1022-tp3/c1.json
    ("tp4", 8): (11.667, 16.714, 1.970, 8),    # manifests/suites/2026-10-25-spark-ds41f-1025-tp4/c8.json
    ("tp3", 8): (9.156, 16.695, 1.965, 8),     # manifests/suites/2026-10-22-spark-ds41f-1022-tp3/c8.json
}
LIVE_SHARE = (2.89 - 1) / 3.0733  # verified drafts above the dead-row cut at c1 (#598)
DEAD_ROW = {"tp4": 0.366, "tp3": (24.26 - 22.52) / 4}
MAX_SEQS = {"tp4": 16, "tp3": 8}
# DSpark on copied text (the copy probe, 2026-10-23, candidate 1 at TP4): tokens a step
DSPARK_MEASURED = {"copy": 4.96, "edit": 5.10, "tool": 5.41, "prose": 2.24}


def fp8(n, k):
    return n * k + (n // 32) * (k // 32)


def bf16(n, k):
    return 2 * n * k


def shape_bytes(tp: int) -> tuple[float, float]:
    """(dense bytes, one routed expert's bytes) a rank reads, floor.py's shapes at ``tp``."""
    f = 4 / tp
    inter = 2304 // tp
    fc1 = 2 * inter * 5120 // 2 + 2 * inter * 5120 // 64 + 2 * inter
    fc2 = 5120 * inter // 2 + 5120 * inter // 64 + 5120
    qakv = fp8(1792, 5120)
    attention = qakv + f * (fp8(8192, 1280) + 2 * fp8(1024, 4096) + fp8(5120, 2048))
    shared = f * (fp8(1152, 5120) + fp8(5120, 576))
    fn = 2 * 4 * 24 * 4 * 5120
    head = f * 32320 * 5120 * 3 // 2
    dense = (40 * attention + 40 * bf16(384, 5120) + 40 * shared + 40 * fn
             + f * 8 * (fp8(4096, 1280) + bf16(32, 5120)) + 3 * bf16(1024, 5120) + bf16(512, 5120)
             + f * 2 * fp8(6400, 6144) + head + f * fp8(1280, 15360)
             + 3 * (attention + shared + bf16(128, 5120) + fn) + head + bf16(32320, 256))
    return dense, fc1 + fc2


def json_routes_table():
    """N(r) for r = 1..8 from json-count's reference routes (#598's distinct())."""
    steps = json.load(open(JSON_ROUTES))["steps"]
    routes = [[s["routes_last_position"][str(layer)] for s in steps] for layer in range(LAYERS)]
    return {r: statistics.mean(len({e for t in range(s, s + r) for e in routes[layer][t]})
                               for layer in range(LAYERS) for s in range(len(steps) - r + 1))
            for r in range(1, len(steps) + 1)}


def uniform(r: float) -> float:
    return E * (1 - (1 - TOP / E) ** r)


class Routes:
    """N(r), distinct experts of r consecutive rows of one request, linear between table rows,
    extended past the table at ``slope`` experts a row and never above uniform routing."""

    def __init__(self, table: dict[int, float], slope: float | None = None, name: str = ""):
        self.table, self.name = dict(table), name
        top = max(table)
        self.slope = slope if slope is not None else table[top] - table[top - 1]

    def __call__(self, r: float) -> float:
        if r <= 0:
            return 0.0
        top = max(self.table)
        if r >= top:
            value = self.table[top] + (r - top) * self.slope
        else:
            lo = max(1, int(r))
            value = self.table[lo] + (self.table[min(lo + 1, top)] - self.table[lo]) * (r - lo)
        return min(value, uniform(r))


class Shape:
    """One topology's step prices at one concurrency."""

    def __init__(self, name: str, concurrency: int, prose: Routes):
        self.name, self.c = name, concurrency
        tp = 4 if name == "tp4" else 3
        self.dense, self.expert = shape_bytes(tp)
        self.dead_row = DEAD_ROW[name]
        self.ladder = capture_sizes(MAX_SEQS[name] * 6)
        self.prose = prose
        steps_s, tokens, drafts, reqs = SUITES[(name, concurrency)]
        self.suite_step = 1e3 / steps_s
        self.suite_tokens = tokens / reqs  # a request-step
        self.prose_rows = 1 + drafts
        self.prose_live = 1 + drafts * LIVE_SHARE
        u = self.union([prose(self.prose_live)] * reqs)
        self.base_rows = reqs * self.prose_rows  # the rows whose per-row costs the overhead holds
        self.overhead = self.suite_step - self.stream_ms(u, reqs)

    def stream_ms(self, u: float, reqs: int) -> float:
        drafter = DRAFTER_LAYERS * self.union([DRAFTER_EXPERTS] * reqs, experts=128) * self.expert
        return (self.dense + drafter + LAYERS * u * self.expert) / RATE

    @staticmethod
    def union(ns, experts=E) -> float:
        return experts * (1 - math.prod(1 - min(n, experts) / experts for n in ns))

    def padding(self, rows: int) -> int:
        """Rows the step's graph adds; past the ladder the design extends it in the same steps."""
        top = self.ladder[-1]
        if rows <= top:
            return next(s for s in self.ladder if s >= rows) - rows
        return -rows % 8

    def step_ms(self, requests) -> float:
        """``requests``: (routed rows, their routes, other rows) per request. Every row past
        the suite's pays the grid's per-row price (routed or not, padding included: the
        wide-verify bundle's per-row FC1/FC2 term is part of it), and routed rows bring their
        new experts' bytes."""
        u = self.union([routes(live) for live, routes, _ in requests])
        rows = sum(live + other for live, _, other in requests)
        extra = rows + self.padding(round(rows)) - self.base_rows
        return self.overhead + self.stream_ms(u, len(requests)) + extra * self.dead_row


def request_rows(kind: str, alloc: int, live: int, shape: Shape, copy: Routes):
    """(routed rows, routes, other rows) of one request-step from the walk's histogram key."""
    if kind == "dspark_prose":
        extra = max(0, alloc - (1 + 5))
        return shape.prose_live, shape.prose, shape.prose_rows - shape.prose_live + extra
    return live, copy, alloc - live


def walk_cost(hist, shape: Shape, copy: Routes, reasoning: float = 0.0, samples: int = 20000,
              dspark_rates: dict | None = None, seed: int = 0) -> dict:
    """tok/s of one walk's histogram: c1 sums every step; c8 draws eight request-steps a step.
    ``reasoning`` adds that many hidden-reasoning tokens per output token, decoded as prose.
    ``dspark_rates`` re-prices DSpark steps at a measured tokens-a-step (copy-probe categories)."""
    entries = []
    for kind, alloc, live, tokens, count in hist:
        if kind == "dspark_prose" and shape.c > 1:
            tokens = tokens * shape.suite_tokens / 2.25
        entries.append((kind, alloc, live, tokens, count))
    if dspark_rates is not None:
        dspark_tokens = sum(t * c for k, _, _, t, c in entries if k != "suffix")
        rate = dspark_rates
        entries = [e for e in entries if e[0] == "suffix"]
        if dspark_tokens:
            entries.append(("dspark_copy" if rate > 3 else "dspark_prose", 6, 6, rate, dspark_tokens / rate))
    out_tokens = sum(t * c for _, _, _, t, c in entries)
    if reasoning:
        n = out_tokens * reasoning / shape.suite_tokens
        entries.append(("dspark_prose", 6, 0, shape.suite_tokens, n))
    if shape.c == 1:
        time = tokens = 0.0
        for kind, alloc, live, t, count in entries:
            time += count * shape.step_ms([request_rows(kind, alloc, live, shape, copy)])
            tokens += count * t
        return {"tok_s": round(1e3 * tokens / time, 2), "tokens_per_step": round(tokens / sum(e[4] for e in entries), 3)}
    rng = random.Random(seed)
    weights = [e[4] for e in entries]
    total_t = total_ms = 0.0
    for _ in range(samples):
        draw = rng.choices(entries, weights=weights, k=shape.c)
        total_ms += shape.step_ms([request_rows(k, a, l, shape, copy) for k, a, l, _, _ in draw])
        total_t += sum(e[3] for e in draw)
    return {"tok_s": round(1e3 * total_t / total_ms, 2), "tokens_per_step": round(total_t / samples, 3)}


def walk_stats(hist) -> dict:
    """Tokens a step, the share of tokens from suffix steps, and per suffix step: verified
    drafts, accepted drafts and wasted rows (verified past the first rejection, plus rows
    allocated and left unused)."""
    steps = sum(c for *_, c in hist)
    tokens = sum(t * c for _, _, _, t, c in hist)
    sfx = [(a, l, t, c) for k, a, l, t, c in hist if k == "suffix"]
    n = sum(c for *_, c in sfx)
    drafts = sum((l - 1) * c for _, l, _, c in sfx)
    accepted = sum((t - 1) * c for _, _, t, c in sfx)
    unused = sum(max(0, a - 6) * c for k, a, l, t, c in hist if k != "suffix")
    unused += sum((a - l) * c for a, l, _, c in sfx)
    return {"tokens_per_step": round(tokens / steps, 3), "suffix_token_share": round(sum(t * c for _, _, t, c in sfx) / tokens, 3),
            "suffix_steps_share": round(n / steps, 3),
            "drafts_per_suffix_step": round(drafts / n, 2) if n else 0,
            "accepted_per_suffix_step": round(accepted / n, 2) if n else 0,
            "wasted_rows_per_token": round((drafts - accepted + unused) / tokens, 3)}


def width_table(shape: Shape, routes: dict[str, Routes]) -> list[dict]:
    """One request's step at c1 verifying R live rows of copied text, by route source."""
    rows = []
    for width in WIDTHS:
        row = {"rows": width, "padded_to": width + shape.padding(width)}
        for name, n in routes.items():
            ms = shape.step_ms([(width, n, 0)])
            row[name] = {"experts_a_layer": round(n(width), 1), "step_ms": round(ms, 2),
                         "ms_a_token_if_all_accepted": round(ms / width, 2)}
        rows.append(row)
    return rows


def load_probe_routes(path: Path) -> dict[str, Routes]:
    """N(r) per case from the route probe: windows of r consecutive reply rows, every layer."""
    data = json.load(open(path))
    out = {}
    for case in data["cases"]:
        start = case["reply_start"]
        layers = [r[start:] for r in case["routes"]]
        n = len(layers[0])
        table = {}
        for r in range(1, min(96, n) + 1):
            vals = []
            for layer in layers:
                for s in range(0, n - r + 1, max(1, r // 2)):
                    vals.append(len({e for t in range(s, s + r) for e in layer[t]}))
            table[r] = statistics.mean(vals)
        out[case["case"]] = Routes(table, name=case["case"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default=str(HERE / "runs" / "routes-dgx4.json"))
    ap.add_argument("--proxy", default=str(HERE / "runs" / "agent-proxy.json"))
    ap.add_argument("--cases", default=str(HERE / "runs" / "copy-cases.json"))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--samples", type=int, default=4000)
    args = ap.parse_args()

    json_table = json_routes_table()
    json8 = Routes(json_table, slope=2.4, name="json-count, +2.4 a row")
    sources = {"json-count +2.4/row": json8, "uniform": Routes({1: 6.0, 2: uniform(2)}, slope=TOP, name="uniform")}
    probe = load_probe_routes(Path(args.routes)) if Path(args.routes).exists() else {}
    for name, n in probe.items():
        sources[f"probe {name}"] = n
    prose_routes = json8  # the suite's calibration (#598)

    results = {"inputs": {"json_routes": {r: round(v, 2) for r, v in json_table.items()},
                          "probe_routes": {k: {r: round(v(r), 2) for r in (1, 2, 4, 6, 8, 17, 33, 49, 65)}
                                           for k, v in probe.items()},
                          "uniform": {r: round(uniform(r), 1) for r in (6, 17, 33, 49, 65)}},
               "shapes": {}}
    shapes = {(name, c): Shape(name, c, prose_routes) for name in ("tp4", "tp3") for c in (1, 8)}
    for (name, c), shape in shapes.items():
        dense, expert = shape_bytes(4 if name == "tp4" else 3)
        info = {"dense_mb": round(dense / 1e6, 1), "expert_mb": round(expert / 1e6, 3),
                "overhead_ms": round(shape.overhead, 2), "suite_step_ms": round(shape.suite_step, 2),
                "dead_row_ms": round(shape.dead_row, 3), "ladder_top": shape.ladder[-1]}
        if c == 1:
            info["widths"] = width_table(shape, sources)
        results["shapes"][f"{name}-c{c}"] = info

    copy_sources = {"shared": json8, "independent": sources["uniform"]}
    if "copy" in probe:
        copy_sources["probe"] = probe["copy"]

    def evaluate(hists, **kw):
        table = {}
        for key, hist in hists.items():
            row = {"walk": walk_stats(hist)}
            for source, routes in copy_sources.items():
                row[source] = {f"{name}-c{c}": walk_cost(hist, shape, routes, samples=args.samples, **kw)
                               for (name, c), shape in shapes.items()}
            table[key] = row
        return table

    if Path(args.cases).exists():
        data = json.load(open(args.cases))
        results["copy_cases"] = {}
        for cat, v in data["categories"].items():
            results["copy_cases"][cat] = {
                "walk": evaluate(v["histograms"]),
                "measured_dspark": evaluate(v["histograms"], dspark_rates=DSPARK_MEASURED[cat]),
            }
    if Path(args.proxy).exists():
        data = json.load(open(args.proxy))
        results["agent_proxy"] = {
            "corpus": {k: v for k, v in data.items() if k != "histograms"},
            "visible": evaluate(data["histograms"]),
            "reasoning_1to1": evaluate(data["histograms"], reasoning=1.0),
        }
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)
        handle.write("\n")
    for key, info in results["shapes"].items():
        print(key, {k: v for k, v in info.items() if k != "widths"})
        for row in info.get("widths", []):
            print("  ", row["rows"], row["padded_to"], {k: (v["experts_a_layer"], v["step_ms"])
                                                        for k, v in row.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
