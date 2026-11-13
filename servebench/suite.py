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

from . import model_evidence as model_evidence_api, report

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
        metadata: dict | None = None, log=print, model_evidence: dict | None = None) -> dict:
    """Run every step through servebench's ``main`` and write ``suite.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prepared_model = None
    if model_evidence and (suite["name"] == model_evidence["contract"]["suite"]
                           or any(step["command"] == "model" for step in suite["steps"])):
        prior = [out / "suite.json", out / "model.json.gz", out / "model.json.gz.evidence.json",
                 out / "model.preflight.evidence.json"]
        if any(path.exists() for path in prior):
            raise FileExistsError("prior opted-in model evidence exists; preserve it and use a distinct run")
        try:
            prepared_model = model_evidence_api.prepare(model_evidence, suite, digest(suite), label, metadata)
        except Exception as error:
            with (out / "model.preflight.evidence.json").open("x") as stream:
                json.dump({"schema": model_evidence_api.SCHEMA, "status": "preflight-failed",
                    "contract": {"path": model_evidence["path"], "sha256": model_evidence["sha256"]},
                    "model_requests_started": False, "parser_worker": getattr(error, "receipt", None),
                    "errors": [f"{type(error).__name__}: {error}"]}, stream, indent=2)
                stream.write("\n")
            raise
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
        # Native process outcome must survive any later parser/resource failure.
        (out / "suite.json").write_text(json.dumps(index, indent=1) + "\n")
        if prepared_model is not None and step["command"] == "model":
            try:
                evidence = model_evidence_api.evaluate(output, code, prepared_model)
            except Exception as error:
                evidence = {"schema": model_evidence_api.SCHEMA, "status": "failed",
                    "contract": prepared_model["binding"], "process_exit": code,
                    "reference_provenance": prepared_model.get("reference_provenance"),
                    "references": prepared_model.get("references", []),
                    "required_evidence_passed": False,
                    "errors": [f"{type(error).__name__}: {error}"]}
            evidence_path = Path(output + ".evidence.json")
            with evidence_path.open("x") as stream:
                json.dump(evidence, stream, indent=2, allow_nan=False)
                stream.write("\n")
            index["model_evidence"] = {"output": evidence_path.name,
                "required_evidence_passed": evidence["required_evidence_passed"],
                "contract_sha256": prepared_model["binding"]["sha256"]}
            if not evidence["required_evidence_passed"]:
                index["stopped_before_timing"] = True
                log("required model evidence failed; remaining suite steps were not run")
                break
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
    width = max([24, *map(len, by_label)])
    lines = ["rows a step (window means; survival = accepted at k a draft, coverage = verified at k a draft)",
             f"{'step':8s} {'arm':{width}s} {'tokens':>7s} {'requests':>8s} {'rows':>6s} {'yield':>6s} {'wasted':>7s}"
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
                lines.append(f"{step:8s} {label:{width}s} {'-':>7s}  (no speculative counters)")
                continue
            positions = " ".join(f"{p['position']}:{p['survival']:.3f}/{p['coverage']:.3f}" for p in card["positions"]
                                 if p["survival"] is not None and p["coverage"] is not None)
            cond_yield = " ".join(
                f"{p['position']}:{p['conditional_acceptance']:.3f}/{p['verified_yield']:.3f}"
                for p in card["positions"]
                if p.get("conditional_acceptance") is not None and p.get("verified_yield") is not None
            )
            pos_str = positions or "-"
            if cond_yield:
                pos_str += f" [cond/yield {cond_yield}]"
            lines.append(f"{step:8s} {label:{width}s} {card['tokens_per_step']:7.3f} "
                         f"{card['decoding_requests_per_step']:8.2f} {card['verified_rows_per_step']:6.2f} "
                         f"{card['row_yield']:6.3f} {card['wasted_rows_per_step']:7.2f}  {pos_str}")
    return "\n".join(lines)


HEADLINE = ("decode geometric mean", "decode tok/s, paired", "output tok/s", "prefill ")
PAIRED = "decode tok/s, paired"
GEOMEAN = "decode geometric mean"


def floor_metric(metric: str) -> bool:
    """Rows a recorded A/A floor judges: a run step's paired decode row and an engine grid
    step's geometric mean. Every other headline row keeps the run's own repeat."""
    return metric == PAIRED or metric.startswith(GEOMEAN)


def _step_rows(comparison: dict | None, step: str) -> dict:
    """A comparison's rows of one step by metric, or nothing when the step is absent or skipped."""
    steps = (comparison or {}).get("steps", {})
    if step not in steps or "skipped" in steps[step]:
        return {}
    return {row["metric"]: row for row in steps[step][1].get("rows", [])}


def _usable(row: dict | None) -> bool:
    if row is None or row.get("ci95_pct") is None:
        return False
    return row["metric"] != PAIRED or row.get("low_pct") is not None


def _sides(row: dict) -> tuple[float, float]:
    """The interval's width below and above the change (a paired row's sides differ)."""
    low, high = row.get("low_pct"), row.get("high_pct")
    below = row["change_pct"] - low if low is not None else row["ci95_pct"]
    above = high - row["change_pct"] if high is not None else row["ci95_pct"]
    return below, above


def boot_term(candidate_bound: float, reference_bound: float | None, candidate_boots: int,
              reference_boots: int) -> float:
    """The boot noise left in a change averaged over ``candidate_boots`` boots of the
    candidate and ``reference_boots`` of the reference, from bounds on one boot against
    another (each the larger of a recorded A/A floor and an observed repeat): a pair's
    difference carries two boots, so each side contributes bound^2 / (2 x its boots). With
    one bound for both sides this is bound x sqrt((1/b + 1/r) / 2): bound for one boot of
    each, x0.866 for two candidate boots, x0.707 for two of each."""
    reference_bound = candidate_bound if reference_bound is None else reference_bound
    return math.sqrt(candidate_bound ** 2 / (2 * candidate_boots) + reference_bound ** 2 / (2 * reference_boots))


def with_floor(reference_to_candidate: dict, repeat: dict, *, floors: dict | None = None,
               reference_to_repeat: dict | None = None, reference_repeat: dict | None = None,
               reference_again_to_candidate: dict | None = None,
               reference_again_to_repeat: dict | None = None) -> list[dict]:
    """Headline rows of a reference -> candidate suite comparison, judged against the run's
    own repeat (candidate -> candidate again on another boot). A boot carries a few tenths
    of a percent of its own that no within-run interval sees (2026-10-14: the same build
    moved -0.44% to +0.52% on the grid between boots), so a change counts only beyond its
    interval plus the repeat's own change.

    A paired row's interval is a few hundredths of a percent, so the boot is nearly all of
    its uncertainty, and one repeat is a single draw of it: judged against that alone, a
    build would differ from itself about half the time. ``floors`` gives such rows (and an
    engine grid step's geometric mean) a recorded A/A bound by step id: the 95% bound of
    one boot against another (in %) over past repeats. The run's own repeat widens it when
    larger. With ``reference_to_repeat`` the change is the mean over both candidate boots,
    whose boot term is smaller by sqrt(3/4). The margin is the interval and the boot term
    in quadrature.

    With the reference's own repeat (``reference_repeat``: reference -> reference again,
    and the reference's second boot against both candidate boots), every headline row is
    the mean over all four pairings, and the boot term counts both boots on each side
    (``boot_term``): x0.707 of one pair's. Floored rows bound each side by the larger of the
    floor and that side's repeat; other rows by the repeats themselves, added to the
    interval as before. Without it, rows are judged as they always were."""
    floors = floors or {}
    rows = []
    for step, outcome in reference_to_candidate["steps"].items():
        if "skipped" in outcome or step not in repeat["steps"] or "skipped" in repeat["steps"][step]:
            continue
        schema, result = outcome
        if "rows" not in result:
            continue
        again = _step_rows(repeat, step)
        second = _step_rows(reference_to_repeat, step)
        own = _step_rows(reference_repeat, step)
        third = _step_rows(reference_again_to_candidate, step)
        fourth = _step_rows(reference_again_to_repeat, step)
        window = next((r for r in result["rows"] if r["metric"] == "output tok/s"), None)
        # The paired row is the step's verdict row; the window row follows it.
        for row in sorted(result["rows"], key=lambda r: r["metric"] != PAIRED):
            metric = row["metric"]
            if not metric.startswith(HEADLINE) or metric not in again or row["ci95_pct"] is None:
                continue
            floored = floor_metric(metric) and step in floors
            if metric == PAIRED and not floored:
                continue  # its interval alone would miscall; only a recorded floor judges it
            repeat_pct = again[metric]["change_pct"]
            reference_repeat_pct = own[metric]["change_pct"] if metric in own else None
            # Pairings (reference boot, candidate boot): the first always; the candidate's
            # second boot for floored rows or whenever the reference's second boot is used.
            pairings = [(1, 1, row)]
            if reference_repeat_pct is not None and _usable(third.get(metric)):
                pairings.append((2, 1, third[metric]))
            if (floored or len(pairings) > 1) and _usable(second.get(metric)):
                pairings.append((1, 2, second[metric]))
                if len(pairings) == 3 and _usable(fourth.get(metric)):
                    pairings.append((2, 2, fourth[metric]))
            if not _usable(row):
                continue
            reference_boots = len({r for r, _, _ in pairings})
            candidate_boots = len({c for _, c, _ in pairings})
            used = [p for _, _, p in pairings]
            # Grid rows are step times; every other row says which way is better.
            higher_is_better = row.get("higher_is_better", not metric.startswith(GEOMEAN))
            change = sum(p["change_pct"] for p in used) / len(used)
            out = {"step": step, "metric": metric, "change_pct": change, "ci95_pct": row["ci95_pct"],
                   "repeat_pct": repeat_pct, "boots": candidate_boots, "reference_boots": reference_boots,
                   "reference_repeat_pct": reference_repeat_pct if reference_boots == 2 else None,
                   "floor_pct": None}
            if floored:
                # The interval's side toward no change decides; each side keeps its own width.
                below = max(_sides(p)[0] for p in used)
                above = max(_sides(p)[1] for p in used)
                out["ci95_pct"] = below if change >= 0 else above
                out["recorded_floor_pct"] = floors[step]
                candidate_bound = max(floors[step], abs(repeat_pct))
                reference_bound = (max(floors[step], abs(reference_repeat_pct)) if reference_boots == 2 else None)
                out["floor_pct"] = boot_term(candidate_bound, reference_bound, candidate_boots, reference_boots)
                out["margin_pct"] = math.hypot(out["ci95_pct"], out["floor_pct"])
                if window is not None and metric == PAIRED:
                    out["window"] = {"change_pct": window["change_pct"], "ci95_pct": window["ci95_pct"]}
            else:
                out["ci95_pct"] = max(p["ci95_pct"] for p in used)
                reference_bound = abs(reference_repeat_pct) if reference_boots == 2 else None
                out["boot_pct"] = boot_term(abs(repeat_pct), reference_bound, candidate_boots, reference_boots)
                out["margin_pct"] = out["ci95_pct"] + out["boot_pct"]
            if abs(out["change_pct"]) <= out["margin_pct"]:
                out["verdict"] = "same"
            else:
                out["verdict"] = "better" if (out["change_pct"] > 0) == higher_is_better else "worse"
            rows.append(out)
    return rows


def _boots_note(row: dict) -> str:
    if row.get("reference_boots", 1) == 2:
        return "both boots of each"
    return "both candidate boots" if row["boots"] == 2 else ""


def format_floor(rows: list[dict], reference: str, candidate: str, repeat: str,
                 reference_repeat: str | None = None, floors_note: str | None = None) -> str:
    averaged = reference_repeat is not None and any(row.get("reference_boots", 1) == 2 for row in rows)
    head = f"verdicts: {reference} -> {candidate}, against the run's own repeat ({candidate} -> {repeat})"
    if averaged:
        head += (f" and the reference's ({reference_repeat}); each change is the mean over both boots of each "
                 f"({candidate}, {repeat}; {reference}, {reference_repeat})")
    lines = [head, f"{'metric':36s} {'change':>8s} {'±95%':>7s} {'repeat':>8s} {'floor':>7s} {'margin':>7s}  verdict"]
    for row in rows:
        floor = f"{row['floor_pct']:6.2f}%" if row["floor_pct"] is not None else f"{'-':>7s}"
        line = (f"{row['step'] + ': ' + row['metric']:36s} {row['change_pct']:+7.2f}% {row['ci95_pct']:6.2f}% "
                f"{row['repeat_pct']:+7.2f}% {floor} {row['margin_pct']:6.2f}%  {row['verdict']}")
        notes = [_boots_note(row)] if _boots_note(row) else []
        if row.get("reference_repeat_pct") is not None:
            notes[-1] += f", reference repeat {row['reference_repeat_pct']:+.2f}%"
        if row["floor_pct"] is not None:
            notes.append(f"window tok/s {row['window']['change_pct']:+.2f}% ±{row['window']['ci95_pct']:.2f}%"
                         if row.get("window") else "recorded A/A floor")
        if notes:
            line += f" ({'; '.join(notes)})"
        lines.append(line)
    if floors_note:
        lines.append(floors_note)
    if any(row["floor_pct"] is not None for row in rows):
        lines.append("paired rows: margin = ±95% (over requests) and floor in quadrature, where floor is the boot "
                     "term of the boots averaged, each boot bounded by the recorded A/A bound or its repeat when "
                     "larger (the grid's geometric mean likewise when it has a recorded floor); other rows: "
                     "±95% + |repeat|" + (", over the boots averaged" if averaged else ""))
    return "\n".join(lines)
