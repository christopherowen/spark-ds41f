#!/usr/bin/env python3
"""Source-only controls; these do not execute or prove CUDA ordering."""

import ast
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest


SOURCE = Path(sys.argv.pop(1)).resolve()
BASE = "5f926df6d3126266a9c88c82c06544060008c33c"
COPY = "vllm/v1/worker/gpu/spec_decode/copy_drafts.py"
RUNNER = "vllm/v1/worker/gpu/model_runner.py"


def old(path):
    return subprocess.check_output(["git", "-C", str(SOURCE), "show", f"{BASE}:{path}"])


def tree(path, previous=False):
    return ast.parse(old(path) if previous else (SOURCE / path).read_bytes())


def method(path, cls, name, previous=False):
    return next(
        fn
        for node in tree(path, previous).body
        if isinstance(node, ast.ClassDef) and node.name == cls
        for fn in node.body
        if isinstance(fn, ast.FunctionDef) and fn.name == name
    )


def semantic(node):
    return ast.dump(node, include_attributes=False)


def executable(fn):
    ns = {}
    code = ast.Module(
        body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0), fn],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), "actual-source-method", "exec"), ns)
    return ns[fn.name]


class SourceControls(unittest.TestCase):
    def test_kernels_policy_default_envelope_and_topk_are_byte_identical(self):
        paths = (
            "vllm/v1/worker/gpu/spec_decode/tilelang_copy.py",
            "vllm/v1/worker/gpu/async_utils.py",
            "vllm/v1/utils.py",
            "vllm/config/speculative.py",
            "vllm/config/vllm.py",
            "vllm/models/deepseek_v4_1/tilelang/indexer.py",
            "vllm/v1/engine/core.py",
            "vllm/v1/executor/multiproc_executor.py",
            "vllm/v1/executor/uniproc_executor.py",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256(old(path)).digest(),
                                 hashlib.sha256((SOURCE / path).read_bytes()).digest())
        for name in ("_prepare", "attach_async_output", "register_warmups",
                     "add_request", "apply_staged_writes"):
            with self.subTest(method=name):
                self.assertEqual(semantic(method(COPY, "CopyDrafter", name, True)),
                                 semantic(method(COPY, "CopyDrafter", name)))

    def test_only_readiness_fields_removed_from_constructor(self):
        original = method(COPY, "CopyDrafter", "__init__", True)
        names = {"_prepared", "_prepared_waiter"}
        def removed(node):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, ast.AnnAssign) else [])
            return any(isinstance(target, ast.Attribute) and target.attr in names
                       for target in targets)
        original.body = [node for node in original.body if not removed(node)]
        self.assertEqual(semantic(original), semantic(method(COPY, "CopyDrafter", "__init__")))
        current = (SOURCE / COPY).read_text()
        for removed in ("_prepared", "_prepared_waiter", "_join", "wait_stream", "wait_event"):
            self.assertNotIn(removed, current)

    def test_runner_removes_only_prepare_stream_keywords(self):
        original = tree(RUNNER, True)
        removed = []
        for node in ast.walk(original):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "prepare_alloc"):
                removed.extend(kw.arg for kw in node.keywords
                               if kw.arg in ("copy_stream", "main_stream"))
                node.keywords = [kw for kw in node.keywords
                                 if kw.arg not in ("copy_stream", "main_stream")]
        self.assertEqual(removed, ["copy_stream", "main_stream"])
        self.assertEqual(semantic(original), semantic(tree(RUNNER)))
        calls = [node for node in ast.walk(tree(RUNNER)) if isinstance(node, ast.Call)]
        def line(receiver, name, after=0):
            return next(node.lineno for node in calls
                        if isinstance(node.func, ast.Attribute)
                        and node.func.attr == name
                        and ast.unparse(node.func.value) == receiver
                        and node.lineno > after)
        prepare = line("self.copy_drafter", "prepare_alloc")
        attach = line("self.copy_drafter", "attach_async_output", after=prepare)
        dspark = line("self.speculator", "propose", after=attach)
        self.assertLess(prepare, attach)
        self.assertLess(attach, dspark)
        self.assertLess(dspark, line("self.copy_drafter", "propose", after=dspark))

    def test_actual_prepare_rotates_ring_and_preserves_verified_fallback(self):
        prepare = executable(method(COPY, "CopyDrafter", "prepare_alloc"))
        class Tensor:
            shape = (2,)
            def __getitem__(self, item):
                return ("no_drafts", item.start, item.stop)
        buffers = [SimpleNamespace(gpu=object()) for _ in range(3)]
        calls = []
        obj = SimpleNamespace(_ring_idx=0, _ring_size=3, _alloc_buffers=buffers,
                              _no_drafts=Tensor(), _prepare=lambda *args: calls.append(args))
        idx, sampled, verified, cut, match = Tensor(), object(), object(), object(), object()
        for step in range(12):
            given = verified if step % 2 else None
            prepare(obj, idx, sampled, given, cut, match_idx_mapping=match)
            self.assertIs(obj.alloc, buffers[(step + 1) % 3].gpu)
            self.assertEqual(calls[-1], (idx, match, sampled,
                             verified if given is not None else ("no_drafts", None, 2), cut))
        self.assertEqual(len(calls), 12)

    def test_actual_output_waits_for_alloc_then_returns_independent_host_list(self):
        # Execute unchanged production get_output with host doubles. This checks
        # the lifetime gate/read order, not CUDA event completion itself.
        get_output = executable(method("vllm/v1/worker/gpu/async_utils.py", "AsyncOutput", "get_output"))
        events = []
        class Event:
            def __init__(self, label):
                self.label = label
            def synchronize(self):
                events.append(self.label)
        class Array:
            def __init__(self, values, label):
                self.values, self.label = values, label
            def tolist(self):
                events.append(self.label)
                return copy.deepcopy(self.values)
        alloc = [16, 5]
        obj = SimpleNamespace(
            copy_event=Event("sample_copy_complete"),
            sampled_token_ids=Array([[1], [1]], "read_tokens"),
            num_sampled_tokens_np=Array([1, 1], "read_counts"),
            model_runner_output=SimpleNamespace(),
            boundary_checkpoint_tokens_np=None, num_verified_draft_tokens_np=None,
            copy_draft_alloc_np=Array(alloc, "read_alloc"),
            copy_draft_alloc_event=Event("alloc_copy_complete"),
            sampling_mask_tensors=None, num_nans=None, logprobs_tensors=None,
            sink_events=None, routed_experts_cpu=None, _has_fault=None,
            prompt_logprobs_dict={},
        )
        output = get_output(obj)
        self.assertLess(events.index("alloc_copy_complete"), events.index("read_alloc"))
        self.assertEqual(output.copy_draft_alloc, [16, 5])
        alloc[:] = [5, 16]
        self.assertEqual(output.copy_draft_alloc, [16, 5])


if __name__ == "__main__":
    unittest.main(verbosity=2)
