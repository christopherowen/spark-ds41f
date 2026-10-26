#!/usr/bin/env python3
"""Expected tok/s of copy drafting as built, on #601's recorded traffic.

usage: estimate.py [--bench runs/copy-bench-dgx2.txt] [--out estimate.json]
                   [--samples 3000]

The step model, the walks and the workloads are experiments/2026-10-26-suffix-drafter's
(cost.py, variants.py, runs/agent-proxy.json, runs/copy-cases.json): DSpark alone
against the go variant (exact, minimum match 16, at most 16 drafts, the cap growing
from 8, the allocation one step late), priced by the measured routes (the edit tool
call's for the agent proxy, each copy-probe category's own) and the two bounds. Three
things of the build are added:

- **The copy kernels run every step**, copy or not: ``copy_match`` and
  ``copy_select`` at the batch's requests and context, the dgx2 timings of
  benchmark_copy_drafts.py (UVA history, as the runner holds it). The agent proxy's
  streams average 147K tokens (31.0M output and 84.1M context tokens over 785
  streams), so a step's context is taken at 64K and at 256K, the two measured
  contexts around half of it.
- **The decode-row envelope**: a step's decode rows stay within every running
  request at DSpark's 6 rows plus one 17-row block (max_decode_rows). At eight
  streams on TP4 (16 slots) the eight free slots' rows leave room for five wide
  requests a step; on TP3 (8 slots, all busy) for one. A wide allocation beyond the
  room is laid out at 6 rows: a copy step verifies 5 drafts (its tokens at most 6),
  a DSpark step after a copy (the lag's dead rows) drops its extra rows.
- **The capture ladder ends at the envelope** (the recipe's rows: max_num_seqs x 6
  + 11), so a wide step pads to the extended ladder.

Block verification of a step without a copy runs at DSpark's block (no cost), greedy
verification costs the same at both widths, and a wide step's verification rows are
inside the grid's per-row price (cost.py's 0.366 / 0.435 ms), so they add nothing
here. The routed experts of a wide step are cost.py's model; wide_verify_rows.py's
dgx2 timing of them is compared with it in the output (``routed_check``).
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = HERE.parent / "2026-10-26-suffix-drafter"
sys.path.insert(0, str(RECORD))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import cost  # noqa: E402
from recipe import capture_sizes  # noqa: E402

GO = "exact_m16_k16_grow"
ALONE = "dspark_m0_k0_full"
BLOCK_ROWS, WIDE_ROWS = 6, 17
CONTEXTS = (65536, 262144)
# FC1 and FC2's time a row a layer past the experts' (the suffix-drafter record's
# wide-verify bundle: 0.5 + 1.7 us), part of the grid's row price.
ROW_GEMM_US = 2.2


def copy_kernel_us(bench: list[dict], requests: int, context: int) -> float:
    for row in bench:
        if (row.get("case") == "copy" and row["requests"] == requests
                and row["context"] == context and row["history"] == "uva"):
            return row["us"]
    raise SystemExit(f"no copy-kernel timing for {requests} requests at {context}")


def extended(shape: cost.Shape) -> cost.Shape:
    """The shape with the capture ladder ending at the envelope."""
    seqs = cost.MAX_SEQS[shape.name]
    shape.ladder = capture_sizes(seqs * BLOCK_ROWS + WIDE_ROWS - BLOCK_ROWS)
    return shape


def room(shape: cost.Shape) -> int:
    """Rows a step's wide allocations may take past 6 each (the scheduler's
    copy_rows_left): one block's 11 and the free slots' 6 each."""
    seqs = cost.MAX_SEQS[shape.name]
    return (WIDE_ROWS - BLOCK_ROWS) + (seqs - shape.c) * BLOCK_ROWS


def entries_of(hist, shape, reasoning=0.0):
    """cost.walk_cost's entries: DSpark's prose steps at the suite's rate at c > 1,
    and hidden reasoning 1:1 decoded as prose."""
    entries = []
    for kind, alloc, live, tokens, count in hist:
        if kind == "dspark_prose" and shape.c > 1:
            tokens = tokens * shape.suite_tokens / 2.25
        entries.append((kind, alloc, live, tokens, count))
    if reasoning:
        out_tokens = sum(t * c for _, _, _, t, c in entries)
        entries.append(("dspark_prose", 6, 0, shape.suite_tokens,
                        out_tokens * reasoning / shape.suite_tokens))
    return entries


def capped(kind, alloc, live, tokens):
    """A wide allocation the step has no room for: laid out at 6 rows."""
    if kind == "suffix":
        return kind, BLOCK_ROWS, min(live, BLOCK_ROWS), min(tokens, BLOCK_ROWS)
    return kind, BLOCK_ROWS, live, tokens


def tok_s(hist, shape, routes, kernel_ms, reasoning=0.0, samples=3000, seed=0):
    entries = entries_of(hist, shape, reasoning)
    if shape.c == 1:
        time = tokens = 0.0
        for kind, alloc, live, t, count in entries:
            rows = cost.request_rows(kind, alloc, live, shape, routes)
            time += count * (shape.step_ms([rows]) + kernel_ms)
            tokens += count * t
        return 1e3 * tokens / time
    rng = random.Random(seed)
    weights = [e[4] for e in entries]
    total_t = total_ms = 0.0
    capacity = room(shape)
    for _ in range(samples):
        left = capacity
        requests, tokens = [], 0.0
        for kind, alloc, live, t, _ in rng.choices(entries, weights=weights, k=shape.c):
            if alloc > BLOCK_ROWS:
                if alloc - BLOCK_ROWS <= left:
                    left -= alloc - BLOCK_ROWS
                else:
                    kind, alloc, live, t = capped(kind, alloc, live, t)
            requests.append(cost.request_rows(kind, alloc, live, shape, routes))
            tokens += t
        total_ms += shape.step_ms(requests) + kernel_ms
        total_t += tokens
    return 1e3 * total_t / total_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    # The copy kernels' timings come from the first file; the routed layer's
    # timings are averaged over every file (they vary a few percent run to run).
    ap.add_argument("--bench", nargs="+", default=[
        str(HERE / "runs" / name) for name in (
            "copy-bench-dgx2.txt", "run4-copy-bench-dgx2.txt", "run3-copy-bench-dgx2.txt")])
    ap.add_argument("--out", default=str(HERE / "estimate.json"))
    # Eight-stream steps drawn per arm: at 200,000 the gains move by under 0.1 point
    # between seeds (3,000 draws, #601's, move them by about 0.4).
    ap.add_argument("--samples", type=int, default=200000)
    args = ap.parse_args()
    runs = [[json.loads(line) for line in open(path) if line.startswith("{")]
            for path in args.bench]
    bench = [row for row in runs[0] if row.get("case") != "step"]
    for streams in (1, 8, 16):
        for arm in ("dspark", "copy"):
            rows = [row for run in runs for row in run
                    if row.get("case") == "step" and row["streams"] == streams
                    and row["arm"] == arm]
            bench.append(dict(rows[0], step_ms=statistics.fmean(r["step_ms"] for r in rows),
                              runs=len(rows)))

    json8 = cost.Routes(cost.json_routes_table(), slope=2.4, name="json-count +2.4/row")
    uniform = cost.Routes({1: 6.0, 2: cost.uniform(2)}, slope=cost.TOP, name="uniform")
    probe = cost.load_probe_routes(RECORD / "runs" / "routes-summary.json")
    proxy = json.load(open(RECORD / "runs" / "agent-proxy.json"))["histograms"]
    cases = json.load(open(RECORD / "runs" / "copy-cases.json"))["categories"]
    routes_by = {"measured": probe["tool"], "ordinary": json8, "independent": uniform}
    model_rate = cost.RATE

    # The routed experts a wide step adds: dgx2's timing of the whole routed layer
    # (dispatch to combine), less the wide-verify bundle's per-row FC1 and FC2 time
    # (0.5 + 1.7 us a row a layer, which the grid's row price holds), per added
    # expert, against cost.py's stream rate.
    steps = {(row["streams"], row["arm"]): row for row in bench if row.get("case") == "step"}
    expert = cost.shape_bytes(4)[1]
    check, per_expert = {}, []
    for streams in (1, 8, 16):
        a, b = steps[(streams, "dspark")], steps[(streams, "copy")]
        added = b["experts_a_layer"] - a["experts_a_layer"]
        ms = b["step_ms"] - a["step_ms"]
        rows_us = (b["rows"] - a["rows"]) * ROW_GEMM_US * cost.LAYERS / 1e3
        us = (ms - rows_us) * 1e3 / (cost.LAYERS * added)
        per_expert.append(us)
        check[f"c{streams}"] = {
            "rows": [a["rows"], b["rows"]],
            "experts_a_layer": [a["experts_a_layer"], b["experts_a_layer"]],
            "measured_ms_added": round(ms, 3),
            "measured_us_an_expert": round(us, 2),
            "model_ms_added": round(cost.LAYERS * added * expert / model_rate, 3),
            "model_us_an_expert": round(expert / model_rate * 1e3, 2),
        }
    ratio = statistics.fmean(per_expert) / (expert / model_rate * 1e3)
    out = {"inputs": {
        "copy_kernel_us": {f"{n}x{ctx}": copy_kernel_us(bench, n, ctx)
                           for n in (1, 8) for ctx in CONTEXTS},
        "routed_check": check,
        "expert_rate_gbs": {"model": round(model_rate / 1e6, 1),
                            "measured": round(model_rate / ratio / 1e6, 1)},
    }}

    for rate_name, rate in (("model_rate", model_rate), ("measured_rate", model_rate / ratio)):
        cost.RATE = rate  # Shape solves its overhead at this rate
        shapes = {(name, c): extended(cost.Shape(name, c, json8))
                  for name in ("tp4", "tp3") for c in (1, 8)}
        out["inputs"]["room_rows"] = {f"{s}-c{c}": room(sh) for (s, c), sh in shapes.items()}
        out["inputs"]["ladder_top"] = {s: shapes[(s, 1)].ladder[-1] for s in ("tp4", "tp3")}

        def table(hist_go, hist_alone, case_routes, reasoning=0.0, contexts=CONTEXTS):
            rows = {}
            for (name, c), shape in shapes.items():
                for source, routes in case_routes.items():
                    alone = tok_s(hist_alone, shape, routes, 0.0, reasoning, args.samples)
                    for context in contexts:
                        kernel_ms = copy_kernel_us(bench, c, context) / 1e3
                        hybrid = tok_s(hist_go, shape, routes, kernel_ms, reasoning,
                                       args.samples)
                        rows[f"{name}-c{c} {source} {context // 1024}K"] = {
                            "dspark_tok_s": round(alone, 2),
                            "hybrid_tok_s": round(hybrid, 2),
                            "gain_pct": round(100 * (hybrid / alone - 1), 2),
                        }
            return rows

        result = {
            "agent_proxy": table(proxy[GO], proxy[ALONE], routes_by),
            "agent_proxy_reasoning_1to1": table(proxy[GO], proxy[ALONE], routes_by,
                                                reasoning=1.0),
            "copy_probe": {
                category: table(data["histograms"][GO], data["histograms"][ALONE],
                                {"measured": probe[category]}, contexts=(CONTEXTS[0],))
                for category, data in cases.items()},
        }
        out[rate_name] = result
        for key in ("agent_proxy", "agent_proxy_reasoning_1to1"):
            print(f"{rate_name} {key}")
            for name, row in result[key].items():
                print(f"  {name:32s} {row['dspark_tok_s']:8.2f} {row['hybrid_tok_s']:8.2f} "
                      f"{row['gain_pct']:+6.2f}%")
    cost.RATE = model_rate
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)
        handle.write("\n")


if __name__ == "__main__":
    main()
