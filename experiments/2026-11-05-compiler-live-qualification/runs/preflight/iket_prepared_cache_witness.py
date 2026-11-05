#!/usr/bin/env python3
"""Exercise extracted source methods only; never import TileLang, TVM or Torch.

The prepared object and parse key are test doubles. The JIT call routing and
freeze refusal methods are extracted without rewriting from the recorded source.
This is a control-flow witness, not another native/compiler/GPU qualification.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace


C42_HEAD = "454202b2ba8d0cf10b8caf434a956755e039483f"
PRE_BYPASS_HEAD = "925388ba06cdb4d210c820e03480e5b48d1d4723"
C44_HEAD = "d297f60c852dab7dd4fd5a8cfd4ed0d02b071f2f"
JIT = "tilelang/jit/__init__.py"
CACHE = "tilelang/cache/kernel_cache.py"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract(source: bytes, name: str, owner: str | None = None):
    tree = ast.parse(source)
    nodes = tree.body
    if owner is not None:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    method = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    assert not method.decorator_list, "Witness only accepts undecorated source methods"
    method_hash = digest(ast.dump(method, include_attributes=False).encode())
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    return module, method_hash, [method.lineno, method.end_lineno]


def git_bytes(source_root: Path, revision: str, name: str) -> bytes:
    # Disable lazy fetch: the witness only consumes already-local source objects.
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
    return subprocess.check_output(["git", "-C", str(source_root), "show", f"{revision}:{name}"], env=env)


def observe(jit: bytes, cache: bytes, cache_enabled: bool) -> dict:
    class KernelNotPreparedError(RuntimeError):
        pass

    scope = {
        "KernelNotPreparedError": KernelNotPreparedError,
        "_frozen_reason": "test",
        "get_prim_func_name": lambda func, default: "iket_test_scale_add",
        "is_cache_enabled": lambda: cache_enabled,
    }
    refuse, refuse_hash, refuse_lines = extract(cache, "_refuse_if_frozen")
    call, call_hash, call_lines = extract(jit, "__call__", "JITImpl")
    exec(compile(refuse, "<recorded _refuse_if_frozen>", "exec"), scope)
    exec(compile(call, "<recorded JITImpl.__call__>", "exec"), scope)
    prepared = object()
    calls = []
    func = SimpleNamespace(__name__="scale_add", parse_args=lambda *a, **k: ("prepared-1024", {}))

    def compile_attempt(*args, **kwargs):
        calls.append([list(args), kwargs])
        scope["_refuse_if_frozen"](func)
        raise AssertionError("Frozen source guard should refuse compilation")

    impl = SimpleNamespace(
        mode="lazy",
        func=func,
        _kernel_cache={"prepared-1024": prepared},
        is_lazy_mode=lambda: True,
        _can_use_call_form_cache=lambda has_tune_params: False,
        compile=compile_attempt,
    )
    try:
        returned = scope["__call__"](impl, 1024)
        result = {"returned_prepared_object": returned is prepared, "exception": None}
    except KernelNotPreparedError as error:
        result = {"returned_prepared_object": False, "exception": type(error).__name__, "message": str(error)}
    result.update({
        "cache_enabled": cache_enabled,
        "compile_attempts": len(calls),
        "prepared_object_remains_in_wrapper_cache": impl._kernel_cache["prepared-1024"] is prepared,
        "JIT_call_AST_sha256": call_hash,
        "JIT_call_lines": call_lines,
        "freeze_refusal_AST_sha256": refuse_hash,
        "freeze_refusal_lines": refuse_lines,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--c42-source", type=Path)
    parser.add_argument("--c44-source", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    c42 = args.c42_source or root / ".work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-69650cdb/src/tilelang"
    c44 = args.c44_source or root / "experiments/2026-11-05-compiler-cache-qualification/.work/source"
    old_jit = git_bytes(c42, PRE_BYPASS_HEAD, JIT)
    new_jit = git_bytes(c42, C42_HEAD, JIT)
    cache = git_bytes(c42, C42_HEAD, CACHE)
    c44_jit = git_bytes(c44, C44_HEAD, JIT)
    assert new_jit == c44_jit, "The metadata proposal must leave the failing JIT source unchanged"
    old = observe(old_jit, cache, False)
    new_disabled = observe(new_jit, cache, False)
    new_enabled = observe(new_jit, cache, True)
    proposed = observe(c44_jit, cache, False)
    assert old["returned_prepared_object"] and old["compile_attempts"] == 0
    assert new_disabled["exception"] == "KernelNotPreparedError" and new_disabled["compile_attempts"] == 1
    assert new_enabled["returned_prepared_object"] and new_enabled["compile_attempts"] == 0
    assert proposed == new_disabled
    print(json.dumps({
        "status": "source_control_flow_witness_passed",
        "native_or_GPU_test": False,
        "TileLang_TVM_Torch_imported": False,
        "assumptions": [
            "tilelang.prepare has populated the same call key in impl._kernel_cache, as the actual source does",
            "IKET default no-persistent-cache mode has disabled cache and prepare.freeze(test) has frozen compilation",
            "The object and parse key are doubles; no kernel is compiled or executed",
        ],
        "source": {
            "pre_bypass_head": PRE_BYPASS_HEAD,
            "pre_bypass_JIT_sha256": digest(old_jit),
            "C42_head": C42_HEAD,
            "C42_JIT_sha256": digest(new_jit),
            "C42_cache_sha256": digest(cache),
            "C44_head": C44_HEAD,
            "C44_JIT_sha256": digest(c44_jit),
            "C42_C44_JIT_bytes_identical": True,
        },
        "cases": {
            "pre_bypass_prepared_frozen_cache_disabled": old,
            "C42_prepared_frozen_cache_disabled": new_disabled,
            "C42_prepared_frozen_cache_enabled": new_enabled,
            "C44_prepared_frozen_cache_disabled": proposed,
        },
        "conclusion": "The cache-bypass patch ignores a valid explicitly prepared wrapper object while cache is disabled; frozen compilation then refuses. VERSION-only C44 cannot correct this branch.",
    }, indent=2))


if __name__ == "__main__":
    main()
