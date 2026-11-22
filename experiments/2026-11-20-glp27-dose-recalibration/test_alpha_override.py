#!/usr/bin/env python3
"""Standalone test for the WEIGHTLESS_STEER_ALPHA override logic.

The vLLM-side tests for this change live in the patch beside this file
(tests/models/test_deepseek_v4_1_glp.py) and run in the candidate tree's
suite, which needs the prepared vLLM tree and a GPU host. This test runs
anywhere: it extracts ``_alpha_override`` — the exact function the patch adds
to vllm/models/deepseek_v4_1/glp.py — from a patched copy of that file and
exercises it directly. It needs the patch applied to a tree (or the patched
file) at the path given by --glp, defaulting to a sibling ``glp.py``.

Usage: test_alpha_override.py [--glp PATH]
Exit code 0 when every case behaves as the reader promises.
"""
import argparse
import ast
import math
import sys


def load_override(path):
    tree = ast.parse(open(path).read(), path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_alpha_override":
            namespace = {"math": math}
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
            return namespace["_alpha_override"]
    raise SystemExit(f"{path}: no _alpha_override function found (patch not applied?)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glp", default="glp.py")
    args = parser.parse_args()
    override = load_override(args.glp)

    cases = [
        # (raw env value, expected result or "refused")
        (None, None),          # unset: serve the file's glp.alpha_default
        ("2", 2.0),
        ("2.0", 2.0),
        ("0.75", 0.75),
        ("3.5", 3.5),
        ("high", "refused"),   # not a number
        ("", "refused"),       # empty is not a number either (envs maps "" to None upstream, but refuse if it arrives)
        ("0", "refused"),      # zero steering is "off", which is the unset state, not a dose
        ("-1", "refused"),
        ("inf", "refused"),
        ("-inf", "refused"),
        ("nan", "refused"),
    ]
    failures = 0
    for raw, expected in cases:
        try:
            got = override(raw)
        except ValueError:
            got = "refused"
        ok = got == expected
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  WEIGHTLESS_STEER_ALPHA={raw!r} -> {got!r} (expected {expected!r})")
    print(f"{len(cases) - failures}/{len(cases)} cases pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
