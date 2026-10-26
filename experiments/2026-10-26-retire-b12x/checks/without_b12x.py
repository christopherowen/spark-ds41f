#!/usr/bin/env python3
"""Run a command with B12X uninstalled from this container (a kernel-lab wrapper: the lab
removes the container afterwards, and the image is untouched).

usage: without_b12x.py SCRIPT [ARGS...]    (SCRIPT runs under this interpreter)

The bundle's PYTHONPATH must not name the image's B12X tree (/opt/spark-ds41f/candidate/
b12x). This uninstalls the b12x distribution from site-packages with pip, then checks, in a
fresh interpreter, that importlib finds no b12x module and importlib.metadata no b12x
distribution, and that no sys.path entry holds a b12x package. It prints one JSON line with
what it removed and what it checked, then runs SCRIPT in place of itself. Exits 2 when B12X
is still importable.
"""

import json
import os
import subprocess
import sys

CHECK = r"""
import importlib.metadata, importlib.util, json, os, sys
try:
    importlib.metadata.version("b12x")
    distribution = True
except importlib.metadata.PackageNotFoundError:
    distribution = False
trees = [p for p in sys.path if p and os.path.isdir(os.path.join(p, "b12x"))]
print(json.dumps({"find_spec": importlib.util.find_spec("b12x") is not None,
                  "distribution": distribution, "path_trees": trees}))
"""


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if "candidate/b12x" in os.environ.get("PYTHONPATH", ""):
        raise SystemExit("PYTHONPATH names the image's B12X tree")
    removed = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "b12x"],
                             text=True, capture_output=True)
    state = json.loads(subprocess.run([sys.executable, "-c", CHECK], text=True,
                                      capture_output=True, check=True).stdout)
    absent = not state["find_spec"] and not state["distribution"] and not state["path_trees"]
    print(json.dumps({"without_b12x": absent, "pip_exit": removed.returncode,
                      "pip": removed.stdout.strip().splitlines()[-1:], **state}), flush=True)
    if not absent:
        return 2
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
