#!/usr/bin/env python3
"""Report loaded names that a module never binds anywhere (a coarse F821)."""
import ast
import builtins
import sys

problems = 0
for path in sys.argv[1:]:
    tree = ast.parse(open(path).read(), path)
    bound = set(dir(builtins)) | {"__file__", "__name__", "__spec__", "__path__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.MatchAs) and node.name:
            bound.add(node.name)
        elif hasattr(ast, "TypeVar") and isinstance(node, ast.TypeVar):
            bound.add(node.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound:
            problems += 1
            print(f"{path}:{node.lineno}: '{node.id}' is loaded but never bound")
print(f"{len(sys.argv) - 1} files, {problems} unbound loads")
