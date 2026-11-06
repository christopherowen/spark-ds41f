#!/usr/bin/env python3
"""CPU-only source fixture witness; not a native kernel or release-gate replay."""
import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import traceback
from types import ModuleType

NAME = '_topk_log_softmax_kernel'
PRODUCER = 'tests/v1/worker/test_gpu_tilelang_logprob_softmax.py'
CONSUMER = 'tests/v1/worker/test_gpu_verified_logprob_token_ids.py'
RUNTIME = 'vllm/v1/worker/gpu/sample/logprob.py'
UNCHANGED = [CONSUMER, RUNTIME, 'vllm/v1/engine/async_llm.py',
             'vllm/v1/engine/core_client.py',
             'tests/v1/engine/test_request_submission_cleanup.py',
             'tests/v1/engine/test_utility_call_cleanup.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def fixture_setter(root):
    tree = ast.parse((root / PRODUCER).read_text())
    fn = function(tree, 'test_compute_token_logprobs_launches_tilelang')
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr in ('setattr', 'setitem')
             and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant)
             and n.args[1].value == NAME]
    assert len(calls) == 1
    return fn, calls[0]


def assertion(root):
    tree = ast.parse((root / CONSUMER).read_text())
    fn = function(tree, 'test_tilelang_worker_modules_hold_no_live_triton_reference_kernels')
    return next(n for n in ast.walk(fn) if isinstance(n, ast.Assert)
                and ast.unparse(n.test) == 'name not in mod.__dict__')


@contextlib.contextmanager
def lazy_module(root):
    packages = ('vllm', 'vllm.v1', 'vllm.v1.worker', 'vllm.v1.worker.gpu')
    refs_name = packages[-1] + '._triton_refs'
    absent = object()
    before = {name: sys.modules.get(name, absent) for name in (*packages, refs_name)}
    created = {name: ModuleType(name) for name in (*packages, refs_name)}
    reference = object()
    setattr(created[refs_name], NAME, reference)
    for name in packages:
        created[name].__path__ = []
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            setattr(created[parent], child, created[name])
    setattr(created[packages[-1]], '_triton_refs', created[refs_name])
    sys.modules.update(created)
    mod = ModuleType('vllm.v1.worker.gpu.sample.logprob')
    tree = ast.parse((root / RUNTIME).read_text())
    nodes = [n for n in tree.body if
             isinstance(n, ast.FunctionDef) and n.name == '__getattr__' or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_TRITON_REFS' for t in n.targets)]
    assert len(nodes) == 2
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(root / RUNTIME), 'exec'), mod.__dict__)
    try:
        yield mod, reference
    finally:
        for name, value in before.items():
            if value is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def run_case(source, setter, consumer, pytest, preexisting=False, postcondition=None):
    # Only the original fixture setter and exact lazy consumer assertion execute.
    # Package imports resolve to module/reference doubles; no Torch/Triton import.
    with lazy_module(source) as (mod, reference):
        if preexisting:
            mod.__dict__[NAME] = reference
        with pytest.MonkeyPatch.context() as patch:
            ns = {'logprob': mod, 'monkeypatch': patch, 'patch': patch}
            expression = ast.Expr(value=setter)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[expression], type_ignores=[])),
                         '<actual fixture setter>', 'exec'), ns)
            assert mod.__dict__[NAME] is None  # poison still prevents a reference launch
        remains = NAME in mod.__dict__
        try:
            exec(compile(ast.fix_missing_locations(ast.Module(body=[consumer], type_ignores=[])),
                         '<actual lazy-reference consumer>', 'exec'),
                 {'mod': mod, 'mod_name': mod.__name__, 'name': NAME})
            outcome = 'pass'
        except AssertionError as error:
            outcome = 'AssertionError: ' + str(error)
        post_outcome = None
        if postcondition is not None:
            try:
                exec(compile(ast.fix_missing_locations(ast.Module(body=[postcondition], type_ignores=[])),
                             '<actual producer cleanup regression>', 'exec'), {'logprob': mod})
                post_outcome = 'pass'
            except AssertionError:
                post_outcome = 'AssertionError'
        if remains:
            assert mod.__dict__[NAME] is reference
        else:
            assert getattr(mod, NAME) is reference and NAME not in mod.__dict__
        try:
            getattr(mod, '_nonexistent_triton_ref_attr')
        except AttributeError:
            pass
        else:
            raise AssertionError('unknown lazy attribute was accepted')
        return {'preexisting_pollution': preexisting, 'real_global_after_cleanup': remains,
                'consumer_outcome': outcome, 'producer_postcondition': post_outcome}


def metadata_first(args):
    """Persist actual dependency identity before any source/witness assertion."""
    metadata = {'kind': 'actual_image_pytest_metadata', 'python': sys.version,
                'executable': sys.executable, 'platform': sys.platform,
                'witness_sha256': digest(Path(__file__)),
                'expected_image_id': os.environ.get('EXPECTED_IMAGE_ID'),
                'gpu_device_paths': sorted(str(p) for p in Path('/dev').glob('nvidia*')),
                'GPU_or_native_gate_replay': False}
    try:
        import pytest
        import _pytest.monkeypatch
        metadata.update({'status': 'captured',
                         'pytest_version': importlib.metadata.version('pytest'),
                         'pytest_module': {'path': pytest.__file__,
                                           'sha256': digest(Path(pytest.__file__))},
                         'pytest_monkeypatch_source': {
                             'path': _pytest.monkeypatch.__file__,
                             'sha256': digest(Path(_pytest.monkeypatch.__file__))}})
    except Exception as error:
        metadata.update({'status': 'dependency_capture_failed',
                         'error': type(error).__name__ + ': ' + str(error)})
    if args.metadata_output:
        with args.metadata_output.open('x') as stream:
            json.dump(metadata, stream, indent=2)
            stream.write('\n')
    print(json.dumps(metadata), flush=True)
    if metadata['status'] != 'captured':
        raise RuntimeError(metadata['error'])
    if metadata['gpu_device_paths']:
        raise RuntimeError('CPU witness refuses visible NVIDIA device paths')
    return pytest, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-source', type=Path)
    parser.add_argument('--after-source', type=Path)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--metadata-output', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise SystemExit('remote Linux CPU area only; no workstation test fallback')
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    pytest, metadata = metadata_first(args)
    if args.metadata_only:
        return
    if args.before_source is None or args.after_source is None:
        raise RuntimeError('--before-source and --after-source are required for fixture proof')
    before, after = args.before_source.resolve(), args.after_source.resolve()
    for path in UNCHANGED:
        assert (before / path).read_bytes() == (after / path).read_bytes(), path
    old_fn, old = fixture_setter(before)
    new_fn, new = fixture_setter(after)
    assert old.func.attr == 'setattr' and new.func.attr == 'setitem'
    # Numerical/routing assertions and calls remain literal source subtrees.
    for kind in (ast.Assert,):
        old_nodes = [ast.dump(n, include_attributes=False) for n in ast.walk(old_fn) if isinstance(n, kind)]
        new_nodes = [ast.dump(n, include_attributes=False) for n in ast.walk(new_fn) if isinstance(n, kind)]
        assert all(n in new_nodes for n in old_nodes)
    old_checks = [ast.dump(n, include_attributes=False) for n in ast.walk(old_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_assert_same']
    new_checks = [ast.dump(n, include_attributes=False) for n in ast.walk(new_fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_assert_same']
    assert old_checks == new_checks
    consumer = assertion(before)
    postcondition = next(n for n in new_fn.body if isinstance(n, ast.Assert)
                         and ast.unparse(n.test) == repr(NAME) + ' not in logprob.__dict__')
    red = run_case(before, old, consumer, pytest)
    green = run_case(after, new, consumer, pytest, postcondition=postcondition)
    not_hidden = run_case(after, new, consumer, pytest, preexisting=True, postcondition=postcondition)
    cases = {'original_cleanup_red': red, 'corrected_cleanup_green': green,
             'preexisting_pollution_not_hidden': not_hidden}
    # Preserve modern-pytest/non-reproducing outcomes before verdict assertions.
    print(json.dumps({'kind': 'fixture_case_outcomes', 'cases': cases}), flush=True)
    assert red['real_global_after_cleanup'] and red['consumer_outcome'].startswith('AssertionError:')
    assert not green['real_global_after_cleanup'] and green['consumer_outcome'] == 'pass'
    assert green['producer_postcondition'] == 'pass'
    assert not_hidden['real_global_after_cleanup'] and not_hidden['consumer_outcome'].startswith('AssertionError:')
    assert not_hidden['producer_postcondition'] == 'AssertionError'
    result = {'status': 'source_fixture_cleanup_witness_passed',
              'pytest_version': metadata['pytest_version'],
              'pytest_monkeypatch_source': metadata['pytest_monkeypatch_source'],
              'source': {'producer_before_sha256': digest(before / PRODUCER),
                         'producer_after_sha256': digest(after / PRODUCER),
                         'runtime_sha256': digest(before / RUNTIME), 'consumer_sha256': digest(before / CONSUMER)},
              'cases': cases,
              'GPU_or_native_gate_replay': False,
              'limits': 'Exact setter/getattr/consumer source with real pytest cleanup; fake module/reference objects, no numerical or GPU execution.'}
    if args.output:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Metadata has already been flushed if dependency capture was reached.
        print(json.dumps({'status': 'fixture_witness_failed',
                          'error': type(error).__name__ + ': ' + str(error),
                          'GPU_or_native_gate_replay': False}), flush=True)
        traceback.print_exc()
        raise SystemExit(1)
