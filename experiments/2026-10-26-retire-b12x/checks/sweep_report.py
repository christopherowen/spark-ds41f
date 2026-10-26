#!/usr/bin/env python3
"""Compare an import-sweep bundle's two runs (B12X installed, then uninstalled).

usage: sweep_report.py OUTPUT.txt

Reads the two {"sweep": ...} lines import_sweep.py printed and prints one JSON line:
modules imported, failures in each run, the modules that fail only without B12X (with
their errors), the modules whose failure is the overlay's deletion stub (in both runs),
and, from the run with B12X, each vLLM module that imported B12X with the B12X modules it
imported.
"""

import json
import sys
from collections import defaultdict

DELETED = "No module named"


def main() -> int:
    runs = {}
    for line in open(sys.argv[1], encoding="utf-8"):
        if line.startswith('{"sweep"'):
            record = json.loads(line)
            runs["installed" if record["sweep"].endswith("-installed") else "uninstalled"] = record
    installed, uninstalled = runs["installed"], runs["uninstalled"]
    only_without = {name: error for name, error in uninstalled["failures"].items()
                    if name not in installed["failures"]}
    only_with = sorted(set(installed["failures"]) - set(uninstalled["failures"]))
    # A deletion stub raises ModuleNotFoundError naming its own full module path.
    stubs = sorted(name for name, error in installed["failures"].items()
                   if error == f"ModuleNotFoundError: {DELETED} {name!r}"
                   and name in uninstalled["failures"])
    importers = defaultdict(list)
    for module, importer in installed["b12x_imports"].items():
        importers[importer.split(":")[0]].append(module)
    print(json.dumps({
        "modules": installed["modules"],
        "failures": {"installed": len(installed["failures"]),
                     "uninstalled": len(uninstalled["failures"])},
        "fail_only_without_b12x": only_without,
        "fail_only_with_b12x": only_with,
        "deleted_module_stubs": stubs,
        "b12x_imported_by": {name: sorted(modules) for name, modules in sorted(importers.items())},
        "b12x_imports_without_b12x": uninstalled["b12x_imports"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
