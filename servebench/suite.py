"""Suites: a fixed, named set of measurements, so every build is measured the same way.

A suite is data (``servebench/suites/<name>.json`` or any JSON file): steps, each a
servebench command (``model``, ``engine``, ``run``, ``prefill``) with its arguments. A run
writes each step's result into one directory beside ``suite.json``, which records the
suite itself and its SHA-256. Comparing two suite runs compares step by step, and refuses
runs of different suites: a change to a suite is a new suite.

``model`` steps can score another run's text: ``corpus`` (a path, or the first arm's
model result when the lab passes ``--corpus``) makes every build score the same tokens.

``metadata`` is recorded in ``suite.json`` as given and never read here: a caller that knows
the deployment (its image, topology, configuration) stores it beside the measurements.

``engine`` steps need the server's dev-mode collective RPC (``needs_dev_rpc``), which a
production server should not expose: a suite with one runs against a development server.
"""

import datetime
import hashlib
import json
import math
import time
from pathlib import Path

from . import report

SCHEMA = "servebench-suite/1"
BUILTIN = Path(__file__).with_name("suites")
COMMANDS = ("model", "engine", "run", "prefill", "stall")


def load(name: str) -> dict:
    """A built-in suite by name, or a JSON file."""
    path = BUILTIN / f"{name}.json" if (BUILTIN / f"{name}.json").is_file() else Path(name)
    suite = json.loads(path.read_text())
    problems = check(suite)
    if problems:
        raise ValueError(f"suite {name}: " + "; ".join(problems))
    return suite


def check(suite: dict) -> list[str]:
    problems = []
    if not suite.get("name") or not isinstance(suite.get("steps"), list) or not suite["steps"]:
        return ["a suite needs a name and a list of steps"]
    ids = [step.get("id") for step in suite["steps"]]
    if len(set(ids)) != len(ids) or not all(ids):
        problems.append("every step needs a unique id")
    for step in suite["steps"]:
        if step.get("command") not in COMMANDS:
            problems.append(f"step {step.get('id')}: command must be one of {', '.join(COMMANDS)}")
    return problems


def needs_dev_rpc(suite: dict) -> bool:
    """Whether a step times the engine grid, which needs the dev-mode collective RPC."""
    return any(step["command"] == "engine" for step in suite["steps"])


def digest(suite: dict) -> str:
    return hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest()


def arguments(step: dict, url: str, output: str, label: str | None, corpus: str | None) -> list[str]:
    """The servebench command line for one step: its arguments as options (``key`` becomes
    ``--key``, true becomes a flag, lists are comma-joined, objects JSON)."""
    argv = [step["command"], url, "--output", output]
    if label:
        argv += ["--label", f"{label}-{step['id']}"]
    for key, value in step.get("args", {}).items():
        option = "--" + key.replace("_", "-")
        if value is True:
            argv.append(option)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            argv += [option, ",".join(str(v) for v in value)]
        elif isinstance(value, dict):
            argv += [option, json.dumps(value, sort_keys=True)]
        else:
            argv += [option, str(value)]
    if step["command"] == "model" and corpus and "corpus" not in step.get("args", {}):
        argv += ["--corpus", corpus]
    return argv


def output_name(step: dict) -> str:
    return f"{step['id']}.json.gz" if step["command"] == "model" else f"{step['id']}.json"


def run(main, url: str, suite: dict, out_dir: str, *, label: str | None = None, corpus: str | None = None,
        metadata: dict | None = None, log=print) -> dict:
    """Run every step through servebench's ``main`` and write ``suite.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = {"schema": SCHEMA, "suite": suite, "suite_sha256": digest(suite), "label": label, "corpus": corpus,
             "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "steps": []}
    if metadata is not None:
        index["metadata"] = metadata
    for step in suite["steps"]:
        output = str(out / output_name(step))
        began = time.monotonic()
        log(f"servebench suite {suite['name']}: step {step['id']} ({step['command']})")
        try:
            code = main(arguments(step, url, output, label, corpus))
        except Exception as error:  # noqa: BLE001 - recorded; the remaining steps still run
            code = f"{type(error).__name__}: {error}"[:300]
        index["steps"].append({"id": step["id"], "command": step["command"], "output": output_name(step),
                               "exit": code, "seconds": round(time.monotonic() - began, 1)})
        (out / "suite.json").write_text(json.dumps(index, indent=1) + "\n")
    index["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    (out / "suite.json").write_text(json.dumps(index, indent=1) + "\n")
    return index


def compare(read_json, base_dir: str, candidate_dir: str, compare_one) -> dict:
    """Step by step with ``compare_one(base_result, candidate_result)``; refuses different suites."""
    base, candidate = (json.loads((Path(d) / "suite.json").read_text()) for d in (base_dir, candidate_dir))
    if base["suite_sha256"] != candidate["suite_sha256"]:
        raise SystemExit(f"different suites: {base['suite']['name']} {base['suite_sha256'][:12]} vs "
                         f"{candidate['suite']['name']} {candidate['suite_sha256'][:12]}; a changed suite is a new suite")
    steps = {}
    for b, c in zip(base["steps"], candidate["steps"]):
        if b["exit"] != 0 or c["exit"] != 0:
            steps[b["id"]] = {"skipped": f"exit {b['exit']} / {c['exit']}"}
            continue
        # A run kept as a reference may leave out a large output (the model step's text).
        missing = [str(Path(d) / r["output"]) for d, r in ((base_dir, b), (candidate_dir, c))
                   if not (Path(d) / r["output"]).is_file()]
        if missing:
            steps[b["id"]] = {"skipped": "not kept: " + ", ".join(missing)}
            continue
        steps[b["id"]] = compare_one(read_json(str(Path(base_dir) / b["output"])),
                                     read_json(str(Path(candidate_dir) / c["output"])))
    return {"suite": base["suite"]["name"], "suite_sha256": base["suite_sha256"],
            "labels": [base.get("label"), candidate.get("label")], "steps": steps}


def step_accounting(read_json, run_dir: str) -> dict:
    """Rows a step (report.rows_a_step) of every completed ``run`` step of a suite run."""
    index = json.loads((Path(run_dir) / "suite.json").read_text())
    out = {}
    for step in index["steps"]:
        path = Path(run_dir) / step["output"]
        if step["command"] == "run" and step["exit"] == 0 and path.is_file():
            out[step["id"]] = report.rows_a_step(read_json(str(path))["summary"])
    return out


def format_accounting(by_label: dict) -> str:
    """One line per run step and arm: tokens and verified rows a step, row yield, wasted rows,
    and per draft position survival / coverage."""
    lines = ["rows a step (window means; survival = accepted at k a draft, coverage = verified at k a draft)",
             f"{'step':8s} {'arm':24s} {'tokens':>7s} {'requests':>8s} {'rows':>6s} {'yield':>6s} {'wasted':>7s}"
             "  by position (survival/coverage)"]
    steps = []
    for cards in by_label.values():
        steps += [step for step in cards if step not in steps]
    for step in steps:
        for label, cards in by_label.items():
            card = cards.get(step)
            if card is None:
                continue
            if "verified_rows_per_step" not in card:
                lines.append(f"{step:8s} {label[:24]:24s} {'-':>7s}  (no speculative counters)")
                continue
            positions = " ".join(f"{p['position']}:{p['survival']:.3f}/{p['coverage']:.3f}" for p in card["positions"]
                                 if p["survival"] is not None and p["coverage"] is not None)
            lines.append(f"{step:8s} {label[:24]:24s} {card['tokens_per_step']:7.3f} "
                         f"{card['decoding_requests_per_step']:8.2f} {card['verified_rows_per_step']:6.2f} "
                         f"{card['row_yield']:6.3f} {card['wasted_rows_per_step']:7.2f}  {positions or '-'}")
    return "\n".join(lines)


HEADLINE = ("decode geometric mean", "decode tok/s, paired", "output tok/s", "prefill ")
PAIRED = "decode tok/s, paired"


def with_floor(reference_to_candidate: dict, repeat: dict, *, floors: dict | None = None,
               reference_to_repeat: dict | None = None) -> list[dict]:
    """Headline rows of a reference -> candidate suite comparison, judged against the run's
    own repeat (candidate -> candidate again on another boot). A boot carries a few tenths
    of a percent of its own that no within-run interval sees (2026-10-14: the same build
    moved -0.44% to +0.52% on the grid between boots), so a change counts only beyond its
    interval plus the repeat's own change.

    A paired row's interval is a few hundredths of a percent, so the boot is nearly all of
    its uncertainty, and one repeat is a single draw of it: judged against that alone, a
    build would differ from itself about half the time. ``floors`` gives such rows a
    recorded A/A bound by step id: the 95% bound of one boot against another (in %) over
    past repeats. The run's own repeat widens it when larger. With ``reference_to_repeat``
    the change is the mean over both candidate boots, whose boot term is smaller by
    sqrt(3/4). The margin is the interval and the boot term in quadrature."""
    floors = floors or {}
    second = (reference_to_repeat or {}).get("steps", {})
    rows = []
    for step, outcome in reference_to_candidate["steps"].items():
        if "skipped" in outcome or step not in repeat["steps"] or "skipped" in repeat["steps"][step]:
            continue
        schema, result = outcome
        if "rows" not in result:
            continue
        again = {row["metric"]: row for row in repeat["steps"][step][1].get("rows", [])}
        other = ({row["metric"]: row for row in second[step][1].get("rows", [])}
                 if step in second and "skipped" not in second[step] else {})
        window = next((r for r in result["rows"] if r["metric"] == "output tok/s"), None)
        # The paired row is the step's verdict row; the window row follows it.
        for row in sorted(result["rows"], key=lambda r: r["metric"] != PAIRED):
            if not row["metric"].startswith(HEADLINE) or row["metric"] not in again or row["ci95_pct"] is None:
                continue
            if row["metric"] == PAIRED and step not in floors:
                continue  # its interval alone would miscall; only a recorded floor judges it
            repeat_pct = again[row["metric"]]["change_pct"]
            # Grid rows are step times; every other row says which way is better.
            higher_is_better = row.get("higher_is_better", not row["metric"].startswith("decode geometric"))
            out = {"step": step, "metric": row["metric"], "change_pct": row["change_pct"],
                   "ci95_pct": row["ci95_pct"], "repeat_pct": repeat_pct, "boots": 1, "floor_pct": None}
            if row["metric"] == PAIRED and step in floors:
                boots = [row] + ([other[PAIRED]] if PAIRED in other and other[PAIRED]["low_pct"] is not None else [])
                out["boots"] = len(boots)
                out["change_pct"] = sum(b["change_pct"] for b in boots) / len(boots)
                # The interval's side toward no change decides; each side keeps its own width.
                below = max(b["change_pct"] - b["low_pct"] for b in boots)
                above = max(b["high_pct"] - b["change_pct"] for b in boots)
                out["ci95_pct"] = below if out["change_pct"] >= 0 else above
                aa = max(floors[step], abs(repeat_pct))
                out["floor_pct"] = aa * math.sqrt((1 + 1 / out["boots"]) / 2)
                out["margin_pct"] = math.hypot(out["ci95_pct"], out["floor_pct"])
                if window is not None:
                    out["window"] = {"change_pct": window["change_pct"], "ci95_pct": window["ci95_pct"]}
            else:
                out["margin_pct"] = row["ci95_pct"] + abs(repeat_pct)
            if abs(out["change_pct"]) <= out["margin_pct"]:
                out["verdict"] = "same"
            else:
                out["verdict"] = "better" if (out["change_pct"] > 0) == higher_is_better else "worse"
            rows.append(out)
    return rows


def format_floor(rows: list[dict], reference: str, candidate: str, repeat: str) -> str:
    lines = [f"verdicts: {reference} -> {candidate}, against the run's own repeat ({candidate} -> {repeat})",
             f"{'metric':36s} {'change':>8s} {'±95%':>7s} {'repeat':>8s} {'floor':>7s} {'margin':>7s}  verdict"]
    for row in rows:
        floor = f"{row['floor_pct']:6.2f}%" if row["floor_pct"] is not None else f"{'-':>7s}"
        line = (f"{row['step'] + ': ' + row['metric']:36s} {row['change_pct']:+7.2f}% {row['ci95_pct']:6.2f}% "
                f"{row['repeat_pct']:+7.2f}% {floor} {row['margin_pct']:6.2f}%  {row['verdict']}")
        if row["floor_pct"] is not None:
            line += f" ({'both candidate boots; ' if row['boots'] == 2 else ''}"
            line += (f"window tok/s {row['window']['change_pct']:+.2f}% ±{row['window']['ci95_pct']:.2f}%)"
                     if row.get("window") else "recorded A/A floor)")
        lines.append(line)
    if any(row["floor_pct"] is not None for row in rows):
        lines.append("paired rows: margin = ±95% (over requests) and floor (the recorded A/A bound, or the "
                     "repeat when larger, for the boots averaged) in quadrature; other rows: ±95% + |repeat|")
    return "\n".join(lines)
