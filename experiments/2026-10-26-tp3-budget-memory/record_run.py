#!/usr/bin/env python3
"""One kernel-lab run's record for runs/: the verdict, each pytest summary line, the
failed or erroring test ids, and the last RECORD line a step printed (a check's JSON).

usage: record_run.py OUTPUT_DIR BUNDLE VLLM_REF RECORD.json

OUTPUT_DIR is a copy of `scripts/lab.py kernel-local --out`'s directory (output.txt,
verdict.json); BUNDLE names bundles/<BUNDLE>; VLLM_REF is the branch and commit it was
generated from. Writes RECORD.json and prints its summary.
"""
import json
import re
import sys
from pathlib import Path


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
        "passed": verdict.get("passed"),
        "seconds": verdict.get("seconds"),
        "tests": [line for line in text.splitlines() if re.search(r"\d+ (passed|failed)", line)],
        "failed_or_errors": sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", text, re.M))),
        "check": checks[-1] if checks else None,
    }
    record.write_text(json.dumps(data, indent=1) + "\n")
    print(bundle, data["passed"], data["tests"], f"{len(data['failed_or_errors'])} failed or erroring")


if __name__ == "__main__":
    main()
