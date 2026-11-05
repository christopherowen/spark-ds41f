#!/usr/bin/env python3
"""Finite offline alternative screen and pinned host-selector checks, not a speed test."""

import argparse
import ast
import itertools
import json
import logging
import random
from pathlib import Path
from types import SimpleNamespace

import replay as screen

np = screen.historical.np
ALPHAS = (0.01, 0.02, 0.035, 0.05, 0.07)
MANAGER_SHA = "54c2e186cdc17aea950d3f8d6549b1f463cc780c836dac75f6c33eaf9d45c7d1"


def aggregate_oracle(options):
    """Maximize sum(tokens)/sum(cost) over independent frozen-step choices.

    Dinkelbach's finite-choice iteration; a per-step ratio argmax is not an
    aggregate-rate upper bound. Each option is (tokens, cost), with cost > 0.
    """
    if not options or any(not row for row in options):
        raise ValueError("oracle requires choices for every step")
    if any(not np.isfinite(t) or t < 0 or not np.isfinite(c) or c <= 0
           for row in options for t, c in row):
        raise ValueError("invalid oracle tokens or costs")
    rate = 0.0
    for iteration in range(100):
        choices = [max(row, key=lambda pair: pair[0] - rate * pair[1]) for row in options]
        tokens, cost = map(sum, zip(*choices))
        updated = tokens / cost
        if abs(tokens - rate * cost) <= 1e-10 * max(1.0, tokens):
            return {"tokens": tokens, "model_ms": cost, "rate": updated,
                    "iterations": iteration + 1}
        rate = updated
    raise RuntimeError("finite oracle did not converge")


def source_checks(path):
    """Execute the exact pinned CPU method, without importing the GPU runtime.

    This checks host objective/capacity semantics only, not native arithmetic,
    GPU allocation, generated outputs, TP transport, or serving invariance.
    """
    identity = screen.receipt(path)
    if identity["sha256"] != MANAGER_SHA:
        raise ValueError("source does not match the inspected C38/C40 manager")
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "AdaptiveVerificationManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "get_num_tokens")
    idle_limit = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "EXPLORE_AFTER_IDLE_STEPS"
                              for t in n.targets))
    namespace = {"np": np, "logger": logging.getLogger(__name__),
                 "EXPLORE_AFTER_IDLE_STEPS": idle_limit}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

    def invoke(conf, capacities, non_draft, sampling, limit, idle=0, order=None):
        b, depth = len(capacities), len(conf[0])
        ids = [str(i) for i in range(b)]
        state = SimpleNamespace(req_id_to_index=dict(zip(ids, range(b))),
                                num_computed_tokens_np=np.zeros(b),
                                prefill_len=SimpleNamespace(np=np.where(sampling, 0, 10000)))
        manager = SimpleNamespace(req_states=state, num_draft_steps=min(5, depth),
                                  num_speculative_steps=depth, _stale_idx=0,
                                  _stale_confidences=[SimpleNamespace(np=np.array(conf))],
                                  _max_total_logits=b + limit, num_bonus_tokens=1,
                                  row_cost=0.035, _steps_without_drafts=idle,
                                  _slot_draft_limits=np.zeros(b, dtype=int))
        order = list(range(b)) if order is None else order
        total = namespace["get_num_tokens"](
            manager, {str(i): non_draft[i] + capacities[i] for i in order},
            {str(i): [0] * capacities[i] for i in order})
        return total - sum(non_draft), manager._steps_without_drafts

    rng = random.Random(20261106)
    cases = 0
    for _ in range(160):
        b, depth = rng.randrange(1, 4), rng.randrange(1, 5)
        conf = [[rng.choice((0.0, 0.1, 0.5, 0.9, 1.0)) for _ in range(depth)] for _ in range(b)]
        caps = [rng.randrange(depth + 1) for _ in range(b)]
        limit = rng.randrange(sum(caps) + 1)
        non_draft = [rng.choice((1, 7)) for _ in range(b)]
        sampling = [True] * b
        # Include one still-prefilling request with no eligible drafts.
        if cases % 3 == 0:
            caps[0], sampling[0] = 0, False
        selected, _ = invoke(conf, caps, non_draft, sampling, limit)
        curves = [np.concatenate(([0.0], np.cumsum(np.cumprod(c)))) for c in conf]
        alternatives = [(sum(k), (sum(sampling) + sum(curves[i][n] for i, n in enumerate(k)))
                         / (1 + .035 * (sum(non_draft) + sum(k))))
                        for k in itertools.product(*(range(c + 1) for c in caps)) if sum(k) <= limit]
        best = max(value for _, value in alternatives)
        achieved = max(value for budget, value in alternatives if budget == selected)
        if abs(best - achieved) > 1e-12:
            raise AssertionError("pinned host objective is below exhaustive prefix optimum")
        reverse, _ = invoke(conf, caps, non_draft, sampling, limit, order=list(reversed(range(b))))
        if reverse != selected:
            raise AssertionError("host budget changed with request ordering")
        cases += 1
    # Longer copy proposals, ties, logits cap, empty confidences and exploration.
    assert invoke([[1.] * 17] * 2, [17, 17], [1, 1], [True, True], 19)[0] == 19
    assert invoke([[1.] * 5] * 2, [5, 5], [1, 1], [True, True], 10)[0] == 10
    assert invoke([[0.] * 5] * 2, [5, 5], [1, 1], [True, True], 10)[0] == 0
    assert invoke([[0.] * 5] * 2, [5, 5], [1, 1], [True, True], 10, idle_limit - 1)[0] == 2
    assert invoke([[1.] * 5], [5], [1], [True], 0)[0] == 0
    return {"source": identity, "exhaustive_prefix_cases": cases, "boundary_cases": 5,
            "scope": "pinned CPU selector only; no GPU/native output or performance qualification"}


def compare(steps, cells):
    if not steps or not cells:
        return {"status": "missing trace or grid"}
    totals = {str(a): [0, 0.0, 0] for a in ALPHAS}
    totals["measured_cost"] = [0, 0.0, 0]
    budget_options, allocation_options = [], []
    b = len(steps[0])
    budgets = sorted(n - b for n in cells if b <= n <= 6 * b)
    if not budgets:
        return {"status": "no eligible measured rows"}
    for rows in steps:
        conf = [c for c, _ in rows]
        expected = screen.historical.expected_curve(conf, 5)
        matches = {}
        for k in budgets:
            counts = screen.historical.allocate(conf, k, 5)
            matches[k] = b + sum(min(n, m) for n, (_, m) in zip(counts, rows))
        for name in totals:
            k = max(budgets, key=lambda n: expected[n] / (
                cells[b + n] if name == "measured_cost" else 1 + float(name) * (b + n)))
            totals[name][0] += matches[k]
            totals[name][1] += cells[b + k]
            totals[name][2] += k
        budget_options.append([(matches[k], cells[b + k]) for k in budgets])
        allocation_options.append([(b + min(k, sum(m for _, m in rows)), cells[b + k]) for k in budgets])
    control = totals["0.035"][0] / totals["0.035"][1]
    policies = {name: {"tokens": t, "model_ms": c, "draft_rows": k,
                       "modeled_delta_pct": 100 * (t / c / control - 1)}
                for name, (t, c, k) in totals.items()}
    oracles = {"budget_only": aggregate_oracle(budget_options),
               "budget_and_allocation": aggregate_oracle(allocation_options)}
    for oracle in oracles.values():
        oracle["headroom_pct"] = 100 * (oracle["rate"] / control - 1)
    return {"steps": len(steps), "measured_rows": [b + k for k in budgets],
            "policies": policies, "frozen_trace_oracles": oracles}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--grid", action="append", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validated = screen.analyze(args.trace, args.grid)
    raw = [[r for r in step if not r["r"].startswith("_warmup")]
           for step in screen.historical.load_trace(args.trace)]
    raw = [step for step in raw if step]
    streams, anchors = screen.historical.committed_streams(raw)
    steps = screen.historical.decode_steps(raw, streams, anchors, 5, 1, 16)
    output = {"schema": "verification-budget-audit/1", "performance_verdict": "unsupported",
              "audit_script": screen.receipt(__file__), "replay_script": screen.receipt(screen.__file__),
              "trace": validated["trace"], "acceptance_replay": validated["acceptance_replay"],
              "limitations": validated["limitations"] + [
                  "Oracles know future token matches, fix every observed batch, and choose only measured cells.",
                  "Oracle headroom bounds this artificial frozen-step model only, not serving throughput.",
                  "Parameter sweep is descriptive on the same trace, not held-out policy selection.",
                  "One measured row count at long context cannot distinguish denominator designs."],
              "source_checks": source_checks(args.source), "grids": []}
    for path in args.grid:
        grid = json.loads(Path(path).read_text())
        cases = {}
        for context in (4096, 65536, 262144):
            for b in (1, 4, 8, 16):
                cases[f"c{b}@{context}"] = compare([s for s in steps if len(s) == b],
                                                  screen.grid_cells(grid, b, context))
        output["grids"].append({"input": screen.receipt(path), "cases": cases})
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
