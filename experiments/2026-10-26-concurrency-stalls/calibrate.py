"""Calibrate stallsim's step costs on measured records, then check it against the rest.

Fitted (each to its own records):
- prefill: m0 and m1 from the 1025 release suites' cold TTFT (manifests/suites/
  2026-10-25-spark-ds41f-1025-{tp4,tp3}/prefill.json), with each chunk's fixed cost F from
  #599's budget fits (TP4 139-168 ms, TP3 192-203 ms: the midpoints);
- the plain decode step at three loads, so that simulated closed loops run the measured engine
  steps a second (c1 single-stream; c8 and c16 on the ring, c4 and c8 on the triangle);
- the fixed frontend time, to one stream's TTFT;
- alpha, what decode rows add to a long-prefill step, from the 131K prefill rate beside
  decoding streams (experiments/2026-10-23-next-release at 16,384 on the ring;
  experiments/2026-10-19-prefill-budget-tp3 at 8,192 on the triangle).

Checked, not fitted: TTFT p50/p99 and TPOT at c8/c16 (c4/c8), the Poisson loads' TTFT tail
(16 and 8 slots), and the median step during a long prefill.

Writes calibration.json (the shapes) and runs/calibration.txt (the report).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stallsim import Engine, Lineage, Policy, Shape, percentile, summarize  # noqa: E402
import workloads  # noqa: E402

ROOT = HERE.parents[1]
SUITES = ROOT / "manifests/suites"

# Measured anchors (spark-ds41f-1025 unless noted).
TARGETS = {
    "tp4-ring": {
        "closed": {1: {"step": 1 / 33.0, "ttft_p50": 0.1279, "tok_s": 75.3},
                   8: {"step": 1 / 11.667, "ttft_p50": 0.2969, "ttft_p99": 0.3687, "tpot_p50": 0.0404, "tok_s": 194.9},
                   16: {"step": 1 / 8.133, "ttft_p50": 0.4067, "ttft_p99": 0.4882, "tpot_p50": 0.0631, "tok_s": 251.4}},
        "poisson": {"rate": 0.7, "ttft_p50": 0.2801, "ttft_p90": 0.6476, "ttft_p99": 6.2884, "tpot_p50": 0.0369,
                    "tok_s": 180.4},
        "stall": {"budget": 16384, "streams": 15, "size": 131900, "alone_tps": 6730.0, "under_tps": 6308.0,
                  "step_p50": 2.36, "step_p99": 2.61, "source": "experiments/2026-10-23-next-release (16,384, candidate 1)"},
        "F": 0.1535,
    },
    "tp3-triangle": {
        "closed": {1: {"step": 1 / 26.73, "ttft_p50": 0.1558, "tok_s": 60.5},
                   4: {"step": 1 / 13.65, "ttft_p50": 0.2662, "ttft_p99": 0.3186, "tpot_p50": 0.0338, "tok_s": 118.2},
                   8: {"step": 1 / 9.30, "ttft_p50": 0.366, "ttft_p99": 0.4396, "tpot_p50": 0.0509, "tok_s": 154.7}},
        "poisson": {"rate": 0.42, "ttft_p50": 0.3152, "ttft_p90": 7.9557, "ttft_p99": 10.2333, "tpot_p50": 0.0379,
                    "tok_s": 118.8},
        "stall": {"budget": 8192, "streams": 7, "size": 131900, "alone_tps": 5543.0 * 5628 / 5543, "under_tps": 5278.0 * 5628 / 5543,
                  "step_p50": 1.444 * 5543 / 5628, "step_p99": None,
                  "source": "experiments/2026-10-19-prefill-budget-tp3 (r6x + skip at 8,192), scaled by 1025's 131K prefill gain (+1.5%)"},
        "F": 0.1975,
    },
}


def base_shapes() -> dict:
    return {
        "tp4-ring": Shape(name="tp4-ring", budget=16384, max_seqs=16, pool_blocks=49483, retention_interval=8192,
                          fill_blocks=3810, dec_knots=[[1, 0.029], [8, 0.080], [16, 0.115]], F=0.1535,
                          m0=125e-6, m1=134e-12, short_e=0.016, short_s=0.00126, alpha=1.0,
                          accept_knots=[[1, 2.284], [8, 2.089], [16, 1.933]]),
        "tp3-triangle": Shape(name="tp3-triangle", budget=8192, max_seqs=8, pool_blocks=23121, retention_interval=512,
                              fill_blocks=1905, dec_knots=[[1, 0.036], [4, 0.068], [8, 0.103]], F=0.1975,
                              m0=140e-6, m1=172e-12, short_e=0.020, short_s=0.00158, alpha=0.7,
                              accept_knots=[[1, 2.269], [4, 2.167], [8, 2.081]]),
    }


def cold_samples(shape: str) -> list:
    name = "tp4" if shape == "tp4-ring" else "tp3"
    data = json.loads((SUITES / f"2026-10-25-spark-ds41f-1025-{name}/prefill.json").read_text())
    return [(s["prompt_tokens"], s["ttft_s"]) for s in data["samples"] if s["ttft_s"]]


def cold_ttft(shape: Shape, tokens: int) -> float:
    """One cold prompt on an idle engine (servebench prefill, max_tokens 1)."""
    engine = Engine(shape, Policy(), seed=1)
    request = engine.submit("doc", 0.0, tokens, 1, Lineage())
    engine.run(1e6)
    return request.first - request.sent


def fit_prefill(shape: Shape, samples: list) -> tuple[float, float, float]:
    """Least squares for m0 and m1 given F: TTFT is linear in both."""
    rows = []
    for tokens, ttft in samples:
        base = dataclasses.replace(shape, m0=0.0, m1=0.0)
        t0 = cold_ttft(base, tokens)
        t1 = cold_ttft(dataclasses.replace(base, m0=1e-6), tokens) - t0
        t2 = cold_ttft(dataclasses.replace(base, m1=1e-12), tokens) - t0
        rows.append((t1, t2, ttft - t0))
    a11 = sum(r[0] * r[0] for r in rows)
    a12 = sum(r[0] * r[1] for r in rows)
    a22 = sum(r[1] * r[1] for r in rows)
    b1 = sum(r[0] * r[2] for r in rows)
    b2 = sum(r[1] * r[2] for r in rows)
    det = a11 * a22 - a12 * a12
    x1 = (b1 * a22 - b2 * a12) / det
    x2 = (a11 * b2 - a12 * b1) / det
    fitted = dataclasses.replace(shape, m0=x1 * 1e-6, m1=x2 * 1e-12)
    worst = max(abs(cold_ttft(fitted, n) - t) / t for n, t in samples)
    return fitted.m0, fitted.m1, worst


def closed_run(shape: Shape, users: int, seconds: float = 360.0, warm: float = 60.0, seed: int = 7,
               policy: Policy | None = None) -> dict:
    engine = Engine(shape, policy or Policy(), seed=seed)
    workloads.closed_loop(engine, random.Random(seed), users, 0.0, warm + seconds, stagger=0.05)
    engine.run(warm + seconds + 120.0)
    return summarize(engine, warm, warm + seconds)


def poisson_run(shape: Shape, rate: float, seconds: float = 300.0, warm: float = 60.0, seed: int = 7,
                policy: Policy | None = None) -> dict:
    engine = Engine(shape, policy or Policy(), seed=seed)
    workloads.chat_poisson(engine, random.Random(seed), rate, 0.0, warm + seconds)
    engine.run(warm + seconds + 300.0)
    return summarize(engine, warm, warm + seconds)


def stall_run(shape: Shape, streams: int, size: int, seed: int = 3, policy: Policy | None = None,
              probes: tuple = ()) -> dict:
    engine = Engine(shape, policy or Policy(), seed=seed)
    sent = workloads.stall_probe(engine, streams, size, probes=probes)
    engine.run(4000.0)
    doc = sent["doc"]
    ttft = doc.first - doc.sent
    during = []
    for r in engine.requests:
        if r.cls != "stream" or r.first is None:
            continue
        t = r.first
        for gap in r.gaps:
            end, t = t + gap, t + gap
            if end > doc.sent and end - gap < doc.first:
                during.append(gap)
    return {"ttft": ttft, "tps": size / ttft, "step_p50": percentile(during, 0.5),
            "step_p99": percentile(during, 0.99), "worst": max(during) if during else None,
            "probes": [p.first - p.sent if p.first else None for p in sent["probes"]]}


def calibrate(name: str, shape: Shape, log) -> Shape:
    target = TARGETS[name]
    shape = dataclasses.replace(shape, F=target["F"])
    samples = cold_samples(name)
    users = sorted(target["closed"])
    for rounds in range(4):
        m0, m1, worst = fit_prefill(shape, samples)
        shape = dataclasses.replace(shape, m0=m0, m1=m1)
        for _ in range(4):
            knots = []
            for (n, t), users_n in zip(shape.dec_knots, users):
                run = closed_run(shape, users_n, seconds=240.0)
                measured = target["closed"][users_n]["step"]
                simulated = 1.0 / (run["steps"] / run["window_s"])
                knots.append([n, t * measured / simulated])
            shape = dataclasses.replace(shape, dec_knots=knots)
        c1 = closed_run(shape, 1, seconds=240.0)
        fe0 = shape.fe0 + target["closed"][1]["ttft_p50"] - c1["all"]["ttft_p50"]
        shape = dataclasses.replace(shape, fe0=max(fe0, 0.002))
        stall = target["stall"]
        if shape.budget == stall["budget"]:
            lo, hi = 0.0, 2.5
            for _ in range(14):
                mid = (lo + hi) / 2
                run = stall_run(dataclasses.replace(shape, alpha=mid), stall["streams"], stall["size"])
                if run["tps"] > stall["under_tps"]:
                    lo = mid
                else:
                    hi = mid
            shape = dataclasses.replace(shape, alpha=round((lo + hi) / 2, 3))
        log(f"  round {rounds + 1}: m0 {shape.m0 * 1e6:.1f} us, m1 {shape.m1 * 1e12:.1f} ps/token^2 "
            f"(worst cold TTFT error {worst * 100:.1f}%), knots "
            + ", ".join(f"{n}:{t * 1000:.1f} ms" for n, t in shape.dec_knots)
            + f", fe0 {shape.fe0 * 1000:.1f} ms, alpha {shape.alpha:.3f}")
    return shape


def check(name: str, shape: Shape, log) -> dict:
    target = TARGETS[name]
    rows = []

    def row(label, measured, simulated, scale=1.0, unit=""):
        if measured is None or simulated is None:
            rows.append((label, measured, simulated, None))
            log(f"  {label:<44} measured {'-' if measured is None else f'{measured * scale:9.3f}'}{unit}  "
                f"model {'-' if simulated is None else f'{simulated * scale:9.3f}'}{unit}")
            return
        error = (simulated - measured) / measured * 100
        rows.append((label, measured, simulated, error))
        log(f"  {label:<44} measured {measured * scale:9.3f}{unit}  model {simulated * scale:9.3f}{unit}  {error:+6.1f}%")

    log(f"{name}: cold prefill (fitted)")
    for tokens, ttft in sorted(dict(cold_samples(name)).items()):
        row(f"cold TTFT, {tokens:,} tokens", ttft, cold_ttft(shape, tokens), unit=" s")
    for users, t in sorted(target["closed"].items()):
        run = closed_run(shape, users)
        log(f"{name}: closed loop c{users} (steps/s and c1 TTFT fitted; the rest checked)")
        row("step", t["step"], run["window_s"] / run["steps"], 1000, " ms")
        row("output tok/s", t["tok_s"], run["output_tok_s"])
        row("TTFT p50", t["ttft_p50"], run["all"]["ttft_p50"], 1000, " ms")
        if "ttft_p99" in t:
            row("TTFT p99", t["ttft_p99"], run["all"]["ttft_p99"], 1000, " ms")
        if "tpot_p50" in t:
            row("TPOT p50", t["tpot_p50"], run["all"]["tpot_p50"], 1000, " ms")
    p = target["poisson"]
    runs = [poisson_run(shape, p["rate"], seed=seed) for seed in (1, 2, 3, 4, 5)]
    log(f"{name}: Poisson {p['rate']} req/s, 256 tokens (checked; five seeds' median)")

    def med(key, sub="all"):
        values = [r[sub][key] if sub else r[key] for r in runs]
        return percentile([v for v in values if v is not None], 0.5)

    row("output tok/s", p["tok_s"], percentile([r["output_tok_s"] for r in runs], 0.5))
    row("TTFT p50", p["ttft_p50"], med("ttft_p50"), 1000, " ms")
    row("TTFT p90", p["ttft_p90"], med("ttft_p90"), 1000, " ms")
    row("TTFT p99", p["ttft_p99"], med("ttft_p99"), 1, " s")
    row("TPOT p50", p["tpot_p50"], med("tpot_p50"), 1000, " ms")
    log(f"  (TTFT p99 over the five seeds: " + ", ".join(f"{r['all']['ttft_p99']:.2f}" for r in runs) + " s)")
    s = target["stall"]
    run = stall_run(shape, s["streams"], s["size"])
    log(f"{name}: stall step, {s['streams']} streams, cold {s['size']:,} ({s['source']}; rate fitted, steps checked)")
    row("131K prefill under decode, tok/s", s["under_tps"], run["tps"])
    row("decode step during the prefill, p50", s["step_p50"], run["step_p50"], 1, " s")
    row("decode step during the prefill, p99", s["step_p99"], run["step_p99"], 1, " s")
    return {"rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(HERE / "calibration.json"))
    parser.add_argument("--report", default=str(HERE / "runs/calibration.txt"))
    parser.add_argument("--check-only", action="store_true", help="check calibration.json, do not refit")
    args = parser.parse_args(argv)
    lines = []

    def log(message: str) -> None:
        print(message, flush=True)
        lines.append(message)

    shapes = load_shapes(Path(args.out)) if args.check_only else {}
    if not args.check_only:
        for name, shape in base_shapes().items():
            log(f"{name}: fitting")
            shapes[name] = calibrate(name, shape, log)
        Path(args.out).write_text(json.dumps({n: dataclasses.asdict(s) for n, s in shapes.items()}, indent=1) + "\n")
    checks = {name: check(name, shape, log) for name, shape in shapes.items()}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n")
    return 0


def load_shapes(path: Path = HERE / "calibration.json") -> dict:
    data = json.loads(Path(path).read_text())
    return {name: Shape(**fields) for name, fields in data.items()}


if __name__ == "__main__":
    sys.exit(main())
