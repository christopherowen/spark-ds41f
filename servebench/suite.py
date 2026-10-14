"""Suites: a fixed, named set of measurements, so every build is measured the same way.

A suite is data (``servebench/suites/<name>.json`` or any JSON file): steps, each a
servebench command (``model``, ``engine``, ``run``, ``prefill``) with its arguments. A run
writes each step's result into one directory beside ``suite.json``, which records the
suite itself and its SHA-256. Comparing two suite runs compares step by step, and refuses
runs of different suites: a change to a suite is a new suite.

``model`` steps can score another run's text: ``corpus`` (a path, or the first arm's
model result when the lab passes ``--corpus``) makes every build score the same tokens.
"""

import datetime
import hashlib
import json
import time
from pathlib import Path

SCHEMA = "servebench-suite/1"
BUILTIN = Path(__file__).with_name("suites")
COMMANDS = ("model", "engine", "run", "prefill")


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
        log=print) -> dict:
    """Run every step through servebench's ``main`` and write ``suite.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = {"schema": SCHEMA, "suite": suite, "suite_sha256": digest(suite), "label": label, "corpus": corpus,
             "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "steps": []}
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
        steps[b["id"]] = compare_one(read_json(str(Path(base_dir) / b["output"])),
                                     read_json(str(Path(candidate_dir) / c["output"])))
    return {"suite": base["suite"]["name"], "suite_sha256": base["suite_sha256"],
            "labels": [base.get("label"), candidate.get("label")], "steps": steps}
