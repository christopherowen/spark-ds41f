"""Assemble the CPU experiment from an exact prepared vLLM commit.

Usage: python3 prepare_bundle.py /path/to/vllm /new/output/directory
The destination must not exist. No source checkout is modified.
"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    source = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve()
    here = Path(__file__).resolve().parent
    identity = json.loads((here / "base.json").read_text())
    revision = identity["vllm_head"]
    tree = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", revision + "^{tree}"], text=True
    ).strip()
    if tree != identity["vllm_tree"]:
        raise ValueError("prepared source tree does not match the recorded identity")
    contents = {}
    for name, record in identity["files"].items():
        data = subprocess.check_output(
            ["git", "-C", str(source), "show", revision + ":" + name]
        )
        if hashlib.sha256(data).hexdigest() != record["tested_sha256"]:
            raise ValueError("source hash mismatch: " + name)
        if record["overlay"]:
            contents[name] = data
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in contents.items():
        path = destination / "overlay" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    shutil.copyfile(here / "base.json", destination / "identity.json")
    for name in ("run_cpu.py", "cpu_declarations.py", "test_delayed_ownership.py"):
        shutil.copyfile(here / name, destination / name)
    print(json.dumps({"destination": str(destination), "vllm_head": revision,
                      "vllm_tree": tree, "overlaid_files": len(contents)}))


if __name__ == "__main__":
    main()
