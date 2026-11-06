#!/usr/bin/env python3
"""Draft C45 installed-compiler gate; root scheduling/controller review is pending.

The host accepts a frozen, hash-bound passed pre-feature release/top-k authority,
never a tag or a caller-guessed image digest. It is not a window controller.
"""
import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import runpy
import shutil
import subprocess
import sys
import traceback
import unittest
import uuid
import xml.etree.ElementTree as ET


def repository_root(directory):
    return directory.parents[1] if len(directory.parents) > 1 else Path("/")


HERE = Path(__file__).resolve().parent
ROOT = repository_root(HERE)
HEAD = "a470d5daf8529fcf2b661f1119c8f4bb1aecc5e2"
TREE = "80fcfc4521b06e0c9776bbf1b5419849557e1090"
PATCHSET = "2018cea5190861fd1ebaf51a5ed95289debdcba4b2f931fd333af95794a90fff"
VERSION = "0.1.15+snapshot.35f15411cf3af0fc06a1cb0097215b1372a40464"
NAMESPACE = "0.1.15_snapshot_35f15411cf3af0fc06a1cb0097215b1372a40464/linux-aarch64"
VLLM_HEAD = "5f926df6d3126266a9c88c82c06544060008c33c"
VLLM_TREE = "ca1496c6481dc5bc4b4234b743e4ca023a6c7597"
TOPK_SHA = "906039870d8dd00d933d1ee1c083510373292e951f971568b4d3cf9c59cbdbfc"
JIT_SHA = "4a0a7bdf18e1d3b1d6502b813e75a7cc5825388360784f896de76979acfc5da0"
CONTROL = "testing/python/cache/test_tilelang_prepared_disabled_cache.py"
CONTROL_SHA = "cd8263bbd5c31c9879926b9b7ca0b3115c5989326f0135bea67e77f1cd10ee23"
STANDALONE = ROOT / "experiments/2026-11-05-compiler-live-qualification/authored_standalone.py"
HISTORICAL_PLAN = ROOT / "experiments/2026-11-05-compiler-live-qualification/gpu-gate-plan.json"
HISTORICAL_PLAN_SHA = "f946f1fb04212452c5692e5f44d464598e57805fe988fb8a0d496c3c271eb5f9"
RELEASE_FILES = 136  # Exact current draft C40 vLLM source; re-review if the base changes.
PREREQUISITE_LOGS = ("build-prepare.log", "build-check.log", "build-image.log", "build-image-apply.log",
                     "gate-manifest.log", "gate-bundle.log", "release-tests.log", "topk-initcheck.log")


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    require(path.stat().st_size < 1048576, "oversized compact gate input: " + str(path))
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def original_assets(plan):
    keys = ("stage_files_from_canonical_source_readonly_only", "static_source_assertion_assets",
            "dynamic_test_helper_assets", "isolated_example_import_assets")
    return [row for key in keys for row in plan[key]]


def verify_plan(plan):
    """Check the preserved asset/selectors contract without a compiler import."""
    require(sha(HISTORICAL_PLAN) == HISTORICAL_PLAN_SHA
            and plan.get("historical_plan", {}).get("sha256") == HISTORICAL_PLAN_SHA,
            "preserved C42 feature contract changed")
    original = load(HISTORICAL_PLAN)
    for key in ("stage_files_from_canonical_source_readonly_only", "static_source_assertion_assets",
                "dynamic_test_helper_assets", "isolated_example_import_assets", "retained_feature_pytest_selectors",
                "exact_scale_cases", "fix_specific", "prepare_freeze", "static_source_assertion_cases"):
        require(plan.get(key) == original[key], "original feature coverage/asset contract changed: " + key)
    assets = original_assets(plan)
    require(len(assets) == 23 and len({r["source_relative"] for r in assets}) == 23,
            "original23 asset coverage differs")
    control = plan.get("prepared_control_asset", {})
    require(control.get("source_relative") == CONTROL and control.get("sha256") == CONTROL_SHA,
            "new22 actual-source control asset differs")
    return assets


def source_binding(source, plan):
    """Small metadata helper for the root's future guarded pre-feature builder.

    It reads a prepared source tree; it does not prepare, compile or import it.
    Its output is bound into the immutable pre-feature authority.
    """
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(source), *arguments], text=True).strip()
    require(git("rev-parse", "HEAD") == HEAD and git("rev-parse", "HEAD^{tree}") == TREE,
            "canonical C45 compiler source identity differs")
    require(not git("status", "--porcelain"), "canonical C45 source must be clean")
    require((source / "VERSION").read_text().strip() == VERSION, "canonical VERSION differs")
    assets = verify_plan(plan)
    for row in assets:
        path = source / row["source_relative"]
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], str(path))
    require(sha(source / "tilelang/jit/__init__.py") == JIT_SHA, "corrected JIT source differs")
    require(sha(source / CONTROL) == CONTROL_SHA, "new22-control source differs")
    return {"schema": "c45-compiler-source-binding/1", "head": HEAD, "tree": TREE,
            "version": VERSION, "patchset_sha256": PATCHSET,
            "driver_sha256": sha(HERE / "compiler_gate.py"),
            "plan_sha256": sha(HERE / "gpu-gate-plan.json"),
            "installed_source_sha256": {
                "tilelang/jit/__init__.py": JIT_SHA,
                "tilelang/cache/kernel_cache.py": sha(source / "tilelang/cache/kernel_cache.py")},
            "prepared_control_sha256": CONTROL_SHA,
            "prepared_control_bytes": (source / CONTROL).stat().st_size,
            "standalone_sha256": sha(STANDALONE), "original_assets": assets}


def authority(path, expected_sha, binding):
    """A passed prerequisite stage is distinct from completed C45 qualification."""
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha or "") and sha(path) == expected_sha,
            "pre-feature authority bytes are not frozen")
    require(path.resolve().is_relative_to((ROOT / "results/private/candidate-rounds/candidate-45-image").resolve()),
            "pre-feature authority must be retained with this new C45 image")
    value = load(path)
    require(value.get("schema") == "c45-compiler-feature-authority/1"
            and value.get("stage") == "release-and-topk-passed-features-pending"
            and value.get("passed") is True and not value.get("error") and value.get("candidate") == 45,
            "no passed C45 pre-feature prerequisite stage")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", value.get("image", ""))
            and re.fullmatch(r"[0-9a-f]{40}", value.get("commit", "")), "image/commit authority missing")
    expected_source = {"vllm_head": VLLM_HEAD, "vllm_tree": VLLM_TREE,
                       "tilelang_head": HEAD, "tilelang_tree": TREE,
                       "tilelang_version": VERSION, "tilelang_patchset_sha256": PATCHSET}
    require(all(value.get("source", {}).get(k) == v for k, v in expected_source.items()),
            "pre-feature source authority differs from this draft base/compiler")
    require(value.get("source_binding") == binding, "canonical source/plan binding changed")
    gate, audit = value.get("gate", {}), value.get("native_gate_audit", {})
    require(gate.get("passed") is True and not gate.get("failed") and not gate.get("errors")
            and audit.get("passed") is True and audit.get("files") == audit.get("expected_files") == RELEASE_FILES
            and audit.get("passed_cases", 0) > 0 and audit.get("failed_cases") == 0
            and audit.get("error_cases") == 0
            and re.fullmatch(r"[0-9a-f]{64}", audit.get("results_sha256", "")),
            "full original per-file native release gate must have passed first")
    wanted = list(PREREQUISITE_LOGS)
    steps = [row for row in value.get("steps", []) if row.get("log") in wanted]
    require([row["log"] for row in steps] == wanted
            and all(row.get("exit") == 0 and row.get("ended") for row in steps),
            "completed first-build/release/top-k process outcomes are missing or repeated")
    require(value.get("topk_initcheck", {}).get("passed") is True
            and value["topk_initcheck"].get("source_sha256") == TOPK_SHA,
            "strict actual-image top-k prerequisite differs")
    cache = value.get("release_cache", {})
    require(cache.get("namespace_absent_before_import") is True and cache.get("namespace") == NAMESPACE
            and cache.get("version") == VERSION and Path(cache.get("host_root", "")).is_absolute(),
            "source-bound builder release-cache first-use proof missing")
    installed = value.get("installed_compiler_identity", {})
    provider = installed.get("provider", {})
    require(installed.get("passed") is True
            and provider.get("module_version") == provider.get("distribution_version") == VERSION
            and provider.get("installed_source_sha256") == binding["installed_source_sha256"]
            and provider.get("loaded_native_libraries") and provider.get("installed_library_inventory"),
            "pre-feature installed/loaded compiler identity is missing")
    module = Path(provider.get("module", ""))
    loaded, inventory = provider["loaded_native_libraries"], provider["installed_library_inventory"]
    require(module.is_absolute() and re.fullmatch(r"[0-9a-f]{64}", provider.get("module_sha256", ""))
            and all(Path(path).is_absolute() and Path(path).is_relative_to(module.parent)
                    and re.fullmatch(r"[0-9a-f]{64}", digest) and inventory.get(path) == digest
                    for path, digest in loaded.items())
            and provider.get("tilelang_loader") in loaded and provider.get("tvm_loader") in loaded,
            "actual selected compiler loader paths/hash inventory are not bound")
    require(installed.get("cache", {}).get("base_key") == {"version": VERSION, "platform": "aarch64"}
            and installed["cache"].get("namespace") == NAMESPACE
            and installed["cache"].get("root") == cache.get("container_root"),
            "builder cache and installed identity do not describe the same mapped namespace")
    return value


def installed_identity(binding):
    require(Path("/.dockerenv").is_file(), "native identity is container-only")
    import tilelang
    import torch
    from tilelang.cache.kernel_cache import KernelCache
    import tvm.base
    module = Path(tilelang.__file__).resolve()
    dist = importlib.metadata.distribution("tilelang")
    require(module == Path(dist.locate_file("tilelang/__init__.py")).resolve(), "compiler distribution/module differs")
    require(not any(module.is_relative_to(Path(p)) for p in ("/tests", "/test-support", "/packet", "/out")),
            "compiler source overlay is forbidden")
    require(tilelang.__version__ == dist.version == VERSION, "installed module/distribution/version pin differs")
    require(torch.cuda.get_device_capability() == (12, 1), "feature workload requires recorded SM121")
    actual_sources = {}
    for relative, expected in binding["installed_source_sha256"].items():
        path = module.parent.parent / relative
        require(path.resolve().is_relative_to(module.parent) and sha(path) == expected,
                "actual installed compiler source differs: " + relative)
        actual_sources[relative] = sha(path)
    libraries = {str(dist.locate_file(p).resolve()): sha(dist.locate_file(p))
                 for p in dist.files if str(p).endswith(".so")}
    require(libraries, "installed compiler native libraries absent")
    loaded = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5]).resolve()
        if path.name.startswith(("libtilelang", "libtvm_compiler", "libtvm_runtime")) and ".so" in path.name:
            require(path.is_relative_to(module.parent), "loaded compiler DSO escaped installed package: " + str(path))
            digest = sha(path)
            require(libraries.get(str(path)) == digest, "loaded DSO differs from installed distribution inventory")
            loaded[str(path)] = digest
    selected, tvm_selected = Path(tilelang._LIB_PATH).resolve(), Path(tvm.base._LIB._name).resolve()
    require(str(selected) in loaded and str(tvm_selected) in loaded,
            "actual selected TileLang and TVM loaders must be mapped and hash-bound")
    require(not tilelang.env.should_use_kernel_cache_lib_stamp(), "retain the reviewed source-version cache policy")
    base = KernelCache._get_base_key()
    namespace = KernelCache._get_cache_namespace()
    cache_root = Path(os.environ["TILELANG_CACHE_DIR"])
    require(cache_root.is_absolute() and base == {"version": VERSION, "platform": "aarch64"}
            and namespace == NAMESPACE and Path(KernelCache._get_namespace_root()) == cache_root / NAMESPACE,
            "actual cache base key/namespace/mapped root differs")
    return {"provider": {"module": str(module), "module_sha256": sha(module),
                          "module_version": tilelang.__version__, "distribution_version": dist.version,
                          "installed_source_sha256": actual_sources, "installed_library_inventory": libraries,
                          "loaded_native_libraries": loaded, "tilelang_loader": str(selected),
                          "tvm_loader": str(tvm_selected)},
            "cache": {"base_key": base, "namespace": namespace, "root": str(cache_root),
                      "namespace_root": str(cache_root / namespace)}}


def junit(path, minimum, allow_sm120_skips=False):
    cases = ET.parse(path).getroot().findall(".//testcase")
    failures = [x.attrib for x in cases if x.find("failure") is not None or x.find("error") is not None]
    skipped = [{"case": x.attrib, "reason": ET.tostring(x.find("skipped"), encoding="unicode")}
               for x in cases if x.find("skipped") is not None]
    passed = len(cases) - len(failures) - len(skipped)
    require(not failures and passed >= minimum, str((path, passed, failures)))
    require(not skipped or (allow_sm120_skips and len(skipped) <= 2 and all(
        x["case"].get("name", "").startswith("test_nvf4_mma_block_scale_fragment_a_unrolls_k_atoms_and_packs_scales[")
        and "Requires CUDA compute eq 12.0, but have 12.1" in x["reason"] for x in skipped)), str(skipped))
    return {"passed": passed, "skipped": skipped, "failures": failures,
            "case_names": [x.get("name") for x in cases],
            "passed_case_names": [x.get("name") for x in cases if x.find("failure") is None
                                  and x.find("error") is None and x.find("skipped") is None],
            "junit_sha256": sha(path)}


def test_support():
    import numpy
    import torch
    support = Path("/test-support")
    require(not (support / "tilelang").exists() and not (support / "src").exists(), "support cannot overlay compiler")
    require("/tests" not in sys.path, "canonical static-source root cannot be on sys.path")
    sys.path.insert(0, str(support))
    for name, relative in (("examples", "examples"), ("examples.dequantize_gemm", "examples/dequantize_gemm")):
        require(name not in sys.modules, "unrelated examples package already loaded")
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(support / relative)]
        sys.modules[name] = importlib.util.module_from_spec(spec)
    import examples
    require(all(Path(p).resolve().is_relative_to(support) for p in examples.__path__), "example path differs")
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def absent_caches(output):
    paths = {"tilelang": output / "tilelang-cache", "cuda": output / "cuda-cache",
             "torch_extensions": output / "torch-extensions"}
    record = {key: {"path": str(path), "absent": not os.path.lexists(path)} for key, path in paths.items()}
    record["namespace"] = {"path": str(paths["tilelang"] / NAMESPACE),
                           "absent": not os.path.lexists(paths["tilelang"] / NAMESPACE)}
    save(output / "cache-before-import.json", record)
    require(all(row["absent"] for row in record.values()), "fresh process cache already exists; preserve and refuse")
    require(os.environ.get("TILELANG_CACHE_DIR") == str(paths["tilelang"])
            and os.environ.get("CUDA_CACHE_PATH") == str(paths["cuda"]), "actual cache environment differs")


def inside(output, tests, plan, binding, expected):
    receipt = {"groups": [], "passed": False}
    try:
        absent_caches(output)  # Must precede the first mapped compiler import.
        identity = installed_identity(binding)
        require(identity["provider"] == expected["installed_compiler_identity"]["provider"],
                "feature provider differs from the passed new-image builder identity")
        test_support()
        save(output / "installed-identity.json", identity)
        receipt["installed_identity"] = identity

        def child(name, flags):
            argv = [sys.executable, __file__, *flags]
            with (output / (name + ".log")).open("x") as stream:
                result = subprocess.run(argv, cwd=output, stdout=stream, stderr=subprocess.STDOUT)
            row = {"name": name, "argv": argv, "exit": result.returncode}
            receipt["groups"].append(row)
            require(result.returncode == 0, str(row))
            row["identity"] = load(output / (name + "-identity.json"))
            require(row["identity"]["before"] == row["identity"]["after"] == identity,
                    "feature child changed compiler identity")
            return row

        def pytest_group(name, selectors, minimum, expression=None, sm120_skips=False):
            xml = output / (name + ".xml")
            flags = ["--pytest-child", "--", "--import-mode=importlib", "--confcutdir=" + str(tests),
                     "-q", *[str(tests / s) for s in selectors], "--junitxml=" + str(xml)]
            if expression:
                flags += ["-k", expression]
            row = child(name, flags)
            row.update(junit(xml, minimum, sm120_skips))
            return row

        retained = pytest_group("retained-features", plan["retained_feature_pytest_selectors"], 203, sm120_skips=True)
        for selector in plan["exact_scale_cases"]:
            name = selector.split("::")[-1]
            require(sum(n.startswith(name + "[") for n in retained["passed_case_names"]) == 2, name)
        minimums = {879: 3, 880: 5, 881: 1, 883: 19}
        for fix in plan["fix_specific"]:
            if fix["kind"] == "pytest":
                pytest_group("fix-" + str(fix["pr"]), fix["selectors"], minimums[fix["pr"]], fix.get("pytest_k"))
            else:
                spec = importlib.util.spec_from_file_location("compiler_consumer_tests", tests / fix["source"])
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for name in fix["functions"]:
                    getattr(module, name)()
                receipt["groups"].append({"name": "fix-882", "functions": fix["functions"], "passed": True})
        child("prepare", ["--prepare-child"])
        prepare = load(output / "standalone.json")
        require(prepare and all(x["status"] == "passed" for x in prepare), "native prepare/freeze check failed")
        child("prepared-controls", ["--controls-child", "--control", str(tests / CONTROL)])
        controls = load(output / "prepared-controls.json")
        require(controls == {"tests": 22, "failures": 0, "errors": 0, "skipped": 0, "passed": True},
                "all22 actual-installed-source method controls are required")
        receipt["final_installed_identity"] = installed_identity(binding)
        require(receipt["final_installed_identity"] == identity, "final installed identity differs")
        receipt["passed"] = True
    except BaseException:
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        save(output / "receipt.json", receipt)


def child_main(args, binding):
    before = installed_identity(binding)
    require(before == load(Path("/out/installed-identity.json")), "child provider/cache identity differs")
    name = "prepare" if args.prepare_child else "prepared-controls"
    if args.pytest_child:
        arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        xml = next(x.split("=", 1)[1] for x in arguments if x.startswith("--junitxml="))
        name = Path(xml).stem
    test_support()
    try:
        if args.pytest_child:
            import pytest
            return pytest.main(arguments)
        if args.prepare_child:
            sys.argv = ["/packet/authored_standalone.py", "prepare"]
            runpy.run_path(sys.argv[0], run_name="__main__")
            return 0
        require(args.control.resolve().is_relative_to(Path("/tests")) and sha(args.control) == CONTROL_SHA,
                "prepared control source differs")
        os.environ["TILELANG_PREPARED_REWORK_SOURCE"] = str(Path(before["provider"]["module"]).parent.parent)
        namespace = runpy.run_path(str(args.control), run_name="c45_prepared_controls")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(namespace["PreparedDisabledCacheTests"])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        summary = {"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                   "skipped": len(result.skipped), "passed": result.wasSuccessful()}
        save(Path("/out/prepared-controls.json"), summary)
        return 0 if summary == {"tests": 22, "failures": 0, "errors": 0, "skipped": 0, "passed": True} else 1
    finally:
        after = installed_identity(binding)
        save(Path("/out") / (name + "-identity.json"), {"before": before, "after": after})
        require(before == after, "child compiler identity changed")


def cleanup_owned(boot, image, token, name):
    cidfile = boot / "container.cid"
    target = cidfile.read_text().strip() if cidfile.exists() else name
    require(target == name or re.fullmatch(r"[0-9a-f]{64}", target), "invalid owned container target")
    process = subprocess.run(["docker", "inspect", target], text=True, capture_output=True)
    inspected = {"exit": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
    save(boot / "container.json", inspected)
    if process.returncode:
        require(process.returncode == 1 and "no such" in (process.stdout + process.stderr).lower(),
                "Docker/owned cleanup state is unreachable")
        return inspected
    state = json.loads(process.stdout)[0]
    require(state["Image"] == image and state["Config"].get("Labels", {}).get("spark.qualification-token") == token,
            "owned feature container cannot be validated; leave foreign state untouched")
    removed = subprocess.run(["docker", "rm", "-f", state["Id"]], text=True, capture_output=True)
    save(boot / "cleanup.json", {"container_id": state["Id"], "exit": removed.returncode,
                               "stdout": removed.stdout, "stderr": removed.stderr})
    require(removed.returncode == 0, "owned container cleanup failed")
    return inspected


def host(output, source, receipt_path, receipt_sha):
    plan = load(HERE / "gpu-gate-plan.json")
    require(plan.get("dispatch_ready") is True, "draft feature controls await remote checks/root integration")
    require(not output.exists(), "preserve every prior gate output; never rerun a completed point")
    binding = source_binding(source, plan)
    expected = authority(receipt_path, receipt_sha, binding)
    image = expected["image"]
    require(subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() == expected["commit"],
            "feature client and new-image execution commit differ")
    require(not subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip(),
            "feature client must be clean and published by the parent")
    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
    labels = inspected["Config"].get("Labels", {})
    require(inspected["Id"] == image and all(labels.get(key) == value for key, value in {
        "local.spark-ds41f.tilelang.tree": TREE, "local.spark-ds41f.tilelang.patch-head": HEAD,
        "local.spark-ds41f.vllm.tree": VLLM_TREE, "local.spark-ds41f.vllm.patch-head": VLLM_HEAD}.items()),
        "actual immutable new image/source labels differ")
    output.mkdir(parents=True)
    save(output / "source-binding.json", binding)
    save(output / "pre-feature-authority.json", expected)
    result = {"image": image, "commit": expected["commit"], "tilelang_head": HEAD, "tilelang_tree": TREE,
              "version": VERSION, "namespace": NAMESPACE, "source_binding_sha256": canonical_sha(binding),
              "authority_source": str(receipt_path), "authority_sha256": receipt_sha,
              "plan_sha256": sha(HERE / "gpu-gate-plan.json"), "boots": [], "passed": False}
    try:
        staged, support = output / "canonical-tests", output / "test-support"
        control_asset = dict(plan["prepared_control_asset"], bytes=binding["prepared_control_bytes"])
        inputs = (plan["stage_files_from_canonical_source_readonly_only"] + plan["static_source_assertion_assets"]
                  + plan["dynamic_test_helper_assets"] + [control_asset])
        for row in inputs:
            path = source / row["source_relative"]
            require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], str(path))
            target = staged / row["source_relative"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for row in plan["isolated_example_import_assets"]:
            path = source / row["source_relative"]
            require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]
                    and row["source_relative"].startswith("examples/"), str(path))
            target = support / row["source_relative"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for number in (1, 2):
            boot = output / ("boot" + str(number))
            boot.mkdir()
            name, token = "compiler-c45-features-" + str(number), uuid.uuid4().hex
            save(boot / "ownership.json", {"name": name, "token": token, "image": image})
            cache_paths = [boot / name for name in ("tilelang-cache", "cuda-cache", "torch-extensions")]
            require(all(not os.path.lexists(path) for path in cache_paths), "new feature cache is not absent")
            save(boot / "host-cache-absence.json", {"paths": [str(p) for p in cache_paths], "absent": True,
                                                    "namespace": NAMESPACE})
            argv = ["docker", "run", "--name", name, "--label", "spark.qualification-token=" + token,
                    "--cidfile", str((boot / "container.cid").resolve()), "--gpus", "all", "--network", "none",
                    "--ipc=host", "--cpus=2", "--memory=32g", "--memory-swap=32g", "--ulimit=core=0",
                    "--ulimit=memlock=-1:-1", "-e", "TILELANG_CACHE_DIR=/out/tilelang-cache",
                    "-e", "CUDA_CACHE_PATH=/out/cuda-cache", "-e", "TORCH_EXTENSIONS_DIR=/out/torch-extensions",
                    "-e", "MAX_JOBS=2", "-e", "DO_NOT_TRACK=1", "-e", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                    "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", str(staged.resolve()) + ":/tests:ro",
                    "-v", str(boot.resolve()) + ":/out", "-v", str(support.resolve()) + ":/test-support:ro",
                    "-v", str((output / "source-binding.json").resolve()) + ":/packet/source-binding.json:ro",
                    "-v", str((output / "pre-feature-authority.json").resolve()) + ":/packet/pre-feature-authority.json:ro",
                    "-v", str(STANDALONE) + ":/packet/authored_standalone.py:ro", "-w", "/out", "--entrypoint", "python3"]
            for filename in ("compiler_gate.py", "gpu-gate-plan.json"):
                argv += ["-v", str(HERE / filename) + ":/packet/" + filename + ":ro"]
            argv += [image, "/packet/compiler_gate.py", "--inside", "--output", "/out", "--tests", "/tests"]
            row = {"name": name, "argv": argv, "exit": None}
            result["boots"].append(row)
            try:
                with (output / ("boot" + str(number) + ".log")).open("x") as stream:
                    process = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
                row["exit"] = process.returncode
            finally:
                state = cleanup_owned(boot, image, token, name)
            require(row["exit"] == 0 and state["exit"] == 0, str(row))
            container = json.loads(state["stdout"])[0]
            require(container["Image"] == image and container["State"].get("OOMKilled") is False,
                    "feature image/OOM outcome failed")
            row["container_id"], row["receipt"] = container["Id"], load(boot / "receipt.json")
            require(row["receipt"].get("passed") is True, "combined compiler feature gate failed")
        require(result["boots"][0]["container_id"] != result["boots"][1]["container_id"], "feature processes reused container")
        require(result["boots"][0]["receipt"]["installed_identity"] == result["boots"][1]["receipt"]["installed_identity"],
                "fresh feature-process installed identity differs")
        result["passed"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        save(output / "receipt.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for flag in ("inside", "pytest-child", "prepare-child", "controls-child", "identity-only", "apply"):
        parser.add_argument("--" + flag, action="store_true")
    for flag in ("output", "source", "tests", "control", "binding", "image-receipt"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--image-receipt-sha256")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.identity_only or args.inside or args.pytest_child or args.prepare_child or args.controls_child:
        require(load(HERE / "gpu-gate-plan.json").get("dispatch_ready") is True,
                "draft native modes await remote controls and live parent guard integration")
    if args.identity_only:
        require(args.binding and args.output, "identity snapshot needs canonical source binding and fresh output")
        snapshot = {"passed": False}
        try:
            binding = load(args.binding)
            require(sha(Path(__file__)) == binding["driver_sha256"], "identity helper source differs")
            snapshot.update(installed_identity(binding))
            snapshot["passed"] = True
        except BaseException:
            snapshot["error"] = traceback.format_exc()
            raise
        finally:
            save(args.output, snapshot)
        return 0
    if args.inside or args.pytest_child or args.prepare_child or args.controls_child:
        binding = load(Path("/packet/source-binding.json"))
        require(sha(Path(__file__)) == binding["driver_sha256"]
                and sha(Path("/packet/gpu-gate-plan.json")) == binding["plan_sha256"]
                and sha(Path("/packet/authored_standalone.py")) == binding["standalone_sha256"], "staged packet bytes changed")
        if args.inside:
            require(args.output == Path("/out") and args.tests == Path("/tests"), "inside output/test roots differ")
            inside(args.output, args.tests, load(HERE / "gpu-gate-plan.json"), binding,
                   load(Path("/packet/pre-feature-authority.json")))
            return 0
        return child_main(args, binding)
    require(args.apply and args.output and args.source and args.image_receipt,
            "draft host dispatch needs --apply/source/new output/frozen authority; live parent guard integration remains mandatory")
    host(args.output, args.source, args.image_receipt, args.image_receipt_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
