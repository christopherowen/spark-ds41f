#!/usr/bin/env python3
"""On dgx4: a copy of the checkout's release-test bundle that runs only the named gate
files, for a quick rerun of fixed tests (the full gate stays the record).

usage: make_targeted.py CHECKOUT OUT_BUNDLE LABEL FILE...
"""
import json
import shutil
import sys
from pathlib import Path

checkout, out, label, *files = sys.argv[1:]
source = Path(checkout) / "experiments/2026-10-25-retire-conditionals/bundles/release-tests"
target = Path(out)
if target.exists():
    shutil.rmtree(target)
shutil.copytree(source, target)
manifest = json.loads((target / "manifest.json").read_text())
missing = [f for f in files if f not in manifest["files"]]
assert not missing, missing
manifest["files"] = {f: manifest["files"][f] for f in files}
manifest["deselect"] = {k: v for k, v in manifest["deselect"].items() if k.split("::")[0] in files}
(target / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
candidate = json.loads((target / "candidate.json").read_text())
candidate["argv"][-1] = f"/c/release-tests/{label}"
candidate["description"] = f"Targeted rerun ({label}): " + ", ".join(files)
(target / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
print(f"{target}: {len(files)} files, label {label}")
