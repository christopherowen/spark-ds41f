#!/usr/bin/env python3
"""Report names in runtime-evaluated annotations that the module never binds.

Without `from __future__ import annotations` (Python 3.12), parameter and return
annotations of every def, and annotations at module and class level, are
evaluated when the def or class statement runs. A name there that the module
does not import or define raises NameError at that moment.
"""
import ast
import builtins
import sys


def bound_names(tree):
    names = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def runtime_annotations(tree):
    lazy = any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(a.name == "annotations" for a in n.names) for n in tree.body)
    if lazy:
        return
    def visit(body, in_function):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                for a in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
                    if a is not None and a.annotation is not None:
                        yield node.name, a.annotation
                if node.returns is not None:
                    yield node.name, node.returns
                yield from visit(node.body, True)
            elif isinstance(node, ast.ClassDef):
                yield from visit(node.body, False)
            elif isinstance(node, ast.AnnAssign) and not in_function:
                yield "<class/module>", node.annotation
            else:
                for field in ("body", "orelse", "finalbody", "handlers"):
                    sub = getattr(node, field, None)
                    if isinstance(sub, list):
                        yield from visit(sub, in_function)
    yield from visit(tree.body, False)


problems = 0
for path in sys.argv[1:]:
    tree = ast.parse(open(path).read(), path)
    names = bound_names(tree)
    for owner, ann in runtime_annotations(tree):
        if isinstance(ann, ast.Constant):
            continue  # string annotation: not evaluated
        for n in ast.walk(ann):
            if isinstance(n, ast.Name) and n.id not in names:
                problems += 1
                print(f"{path}:{n.lineno}: {owner}: '{n.id}' is evaluated at runtime but never bound")
print(f"{len(sys.argv) - 1} files, {problems} unbound runtime annotation names")
