#!/usr/bin/env python3
"""Import every module of the vllm package in one interpreter and report the failures and
the B12X imports (a kernel-lab script).

usage: import_sweep.py LABEL

Walks the vllm package with pkgutil (a package that fails to import is reported, and its
submodules are not visited) and imports every module in name order, keeping each
failure's exception. A meta-path hook (it finds nothing itself) records every b12x module
imported and the vllm module whose frame imported it. Run once with B12X installed and once
without it (without_b12x.py), the difference between the two runs' failures is what needs
B12X at import.

Prints progress, then one JSON line: {"sweep": LABEL, "modules": n, "failures": {module:
error}, "b12x_imports": {b12x module: importing vllm module}}.
"""

import importlib
import json
import os
import pkgutil
import sys
import traceback

IMPORTS: dict[str, str] = {}


class B12xImports:
    """Records the importer of each b12x module; finds nothing, so imports proceed."""

    def find_spec(self, name, path=None, target=None):
        if (name == "b12x" or name.startswith("b12x.")) and name not in IMPORTS:
            importer = "?"
            for frame, _ in traceback.walk_stack(sys._getframe(1)):
                module = frame.f_globals.get("__name__", "")
                if module == "vllm" or module.startswith("vllm."):
                    importer = f"{module}:{frame.f_lineno}"
                    break
            IMPORTS[name] = importer
        return None


def main() -> int:
    label = sys.argv[1]
    sys.meta_path.insert(0, B12xImports())
    failures: dict[str, str] = {}

    def walk_error(name: str) -> None:
        error = sys.exc_info()[1]
        failures[name] = f"{type(error).__name__}: {str(error)[:400]}"

    import vllm

    names = sorted(info.name for info in pkgutil.walk_packages(vllm.__path__, "vllm.", walk_error))
    for count, name in enumerate(names, 1):
        if name in failures:
            continue
        try:
            importlib.import_module(name)
        except BaseException as error:  # a module may also exit or be interrupted
            failures[name] = f"{type(error).__name__}: {str(error)[:400]}"
        if count % 250 == 0:
            print(f"{label}: {count}/{len(names)} imported, {len(failures)} failed", flush=True)
    print(json.dumps({"sweep": label, "modules": len(names), "failures": failures,
                      "b12x_imports": dict(sorted(IMPORTS.items()))}), flush=True)
    sys.stdout.flush()
    os._exit(0)  # modules may leave threads or atexit work behind


if __name__ == "__main__":
    sys.exit(main())
