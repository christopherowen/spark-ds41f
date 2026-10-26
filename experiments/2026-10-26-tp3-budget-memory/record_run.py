#!/usr/bin/env python3
"""One kernel-lab run's record for runs/: the verdict, each pytest summary line, the
failed or erroring test ids, the last RECORD line a step printed (a check's JSON), and
each step's exit status with the last JSON object it printed whole (a benchmark's).

usage: record_run.py OUTPUT_DIR BUNDLE VLLM_REF RECORD.json

OUTPUT_DIR is a copy of `scripts/lab.py kernel-local --out`'s directory (output.txt,
verdict.json); BUNDLE names bundles/<BUNDLE>; VLLM_REF is the branch and commit it was
generated from. Writes RECORD.json and prints its summary.
"""
import json
import re
import sys
from pathlib import Path


def last_json(lines: list[str]):
    """The last JSON object printed whole: an indented dump opens and closes with a
    brace alone at column 0."""
    for end in reversed([i for i, line in enumerate(lines) if line == "}"]):
        starts = [i for i in range(end) if lines[i] == "{"]
        if starts:
            try:
                return json.loads("\n".join(lines[starts[-1]:end + 1]))
            except json.JSONDecodeError:
                continue
    return None


def steps(text: str) -> list[dict]:
    """Each bundle step (run.py's "=== label" ... "=== label: exit N")."""
    found, lines = [], None
    for line in text.splitlines():
        end = re.fullmatch(r"=== (.*): exit (\S+)", line)
        if end:
            found.append({"step": end[1], "exit": end[2], "result": last_json(lines or [])})
            lines = None
        elif line.startswith("=== "):
            lines = []
        elif lines is not None:
            lines.append(line)
    return found


def main() -> None:
    out, bundle, ref, record = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    text = (out / "output.txt").read_text()
    verdict = json.loads((out / "verdict.json").read_text())
    checks = [json.loads(line[7:]) for line in text.splitlines() if line.startswith("RECORD ")]
    data = {
        "bundle": f"bundles/{bundle}",
        "vllm": ref,
        "node": verdict.get("node"),
        "image_id": verdict.get("image"),
        # kernel-local's "exit" verdict reads the output's last line as a pytest summary,
        # so a bundle ending in a check reads unpassed: run.py's status is "exit".
        "passed": verdict.get("passed"),
        "exit": verdict.get("exit"),
        "seconds": verdict.get("seconds"),
        "tests": [line for line in text.splitlines() if re.search(r"\d+ (passed|failed)", line)],
        "failed_or_errors": sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M))),
        "check": checks[-1] if checks else None,
        "steps": steps(text),
    }
    record.write_text(json.dumps(data, indent=1) + "\n")
    print(bundle, data["passed"], data["tests"], f"{len(data['failed_or_errors'])} failed or erroring")


if __name__ == "__main__":
    main()
