#!/usr/bin/env python3
"""Qualify the installed C42 compiler using immutable canonical test inputs."""
import argparse
import hashlib
import importlib.metadata
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import sysconfig
import traceback
import uuid
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
TREE = "bae5fe12b58cdf364efeccbabc04cd08205c43b9"
HEAD = "454202b2ba8d0cf10b8caf434a956755e039483f"


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def junit(path, minimum, allow_sm120_skips=False):
    cases = ET.parse(path).getroot().findall(".//testcase")
    failures = [x.attrib for x in cases if x.find("failure") is not None or x.find("error") is not None]
    skipped = [{"case": x.attrib, "reason": ET.tostring(x.find("skipped"), encoding="unicode")}
               for x in cases if x.find("skipped") is not None]
    passed = len(cases) - len(failures) - len(skipped)
    assert not failures and passed >= minimum, (path, passed, failures)
    assert not skipped or (allow_sm120_skips and len(skipped) <= 2 and all(
        x["case"].get("name", "").startswith("test_nvf4_mma_block_scale_fragment_a_unrolls_k_atoms_and_packs_scales[")
        and "Requires CUDA compute eq 12.0, but have 12.1" in x["reason"] for x in skipped)), skipped
    return {"passed": passed, "skipped": skipped, "failures": failures,
            "case_names": [x.get("name") for x in cases],
            "passed_case_names": [x.get("name") for x in cases if x.find("failure") is None
                                  and x.find("error") is None and x.find("skipped") is None],
            "junit_sha256": sha(path)}


def installed_identity():
    import tilelang
    import torch
    module = Path(tilelang.__file__).resolve()
    assert module.is_relative_to(Path(sysconfig.get_paths()["purelib"]).resolve()), module
    assert torch.cuda.get_device_capability() == (12, 1)
    dist = importlib.metadata.distribution("tilelang")
    libraries = {str(dist.locate_file(p).resolve()): sha(dist.locate_file(p))
                 for p in dist.files if str(p).endswith(".so")}
    assert libraries, "installed compiler native libraries are absent"
    loaded = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5]).resolve()
        if path.name.startswith(("libtilelang", "libtvm_compiler", "libtvm_runtime")) and ".so" in path.name:
            assert path.is_relative_to(module.parent), path
            loaded[str(path)] = sha(path)
    selected = Path(tilelang._LIB_PATH).resolve()
    assert str(selected) in loaded and loaded[str(selected)] == sha(selected), selected
    import tvm.base
    tvm_selected = Path(tvm.base._LIB._name).resolve()
    assert str(tvm_selected) in loaded, tvm_selected
    identity = {"module": str(module), "module_sha256": sha(module), "version": dist.version,
                "installed_library_inventory": libraries, "loaded_native_libraries": loaded,
                "tilelang_loader": str(selected), "tvm_loader": str(tvm_selected)}
    return identity


def isolate_examples(support):
    """Bind only the two canonical namespace parents, avoiding vLLM's examples."""
    for name, relative in (("examples", "examples"), ("examples.dequantize_gemm", "examples/dequantize_gemm")):
        assert name not in sys.modules, "an unrelated examples package was already imported"
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(support / relative)]
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module


def test_support():
    import numpy
    import torch
    support = Path("/test-support")
    assert not (support / "tilelang").exists() and not (support / "src").exists()
    assert "/tests" not in sys.path
    sys.path.insert(0, str(support))
    isolate_examples(support)
    import examples
    assert all(Path(p).resolve().is_relative_to(support) for p in examples.__path__)
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def inside(output, tests, plan):
    identity = installed_identity()
    test_support()
    save(output / "installed-identity.json", identity)
    receipt = {"installed_identity": identity, "groups": [], "passed": False}
    try:
        def pytest_group(name, selectors, minimum, expression=None, sm120_skips=False):
            xml = output / (name + ".xml")
            argv = [sys.executable, __file__, "--pytest-child", "--", "--import-mode=importlib", "--confcutdir=" + str(tests),
                    "-q", *[str(tests / s) for s in selectors], "--junitxml=" + str(xml)]
            if expression:
                argv += ["-k", expression]
            with (output / (name + ".log")).open("x") as stream:
                result = subprocess.run(argv, cwd=output, stdout=stream, stderr=subprocess.STDOUT)
            row = {"name": name, "argv": argv, "exit": result.returncode}
            receipt["groups"].append(row)
            assert result.returncode == 0, row
            row["identity"] = load(output / (name + "-identity.json"))
            assert row["identity"]["before"] == row["identity"]["after"] == identity
            row.update(junit(xml, minimum, sm120_skips))
            return row
        retained = pytest_group("retained-features", plan["retained_feature_pytest_selectors"], 203, sm120_skips=True)
        for selector in plan["exact_scale_cases"]:
            name = selector.split("::")[-1]
            assert sum(n.startswith(name + "[") for n in retained["passed_case_names"]) == 2, name
        minimums = {879: 3, 880: 5, 881: 1, 883: 19}
        for fix in plan["fix_specific"]:
            if fix["kind"] == "pytest":
                pytest_group("fix-" + str(fix["pr"]), fix["selectors"], minimums[fix["pr"]], fix.get("pytest_k"))
            else:
                spec = importlib.util.spec_from_file_location("compiler_consumer_tests", tests / fix["source"])
                test_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(test_module)
                for name in fix["functions"]:
                    getattr(test_module, name)()
                receipt["groups"].append({"name": "fix-882", "functions": fix["functions"], "passed": True})
        argv = [sys.executable, __file__, "--prepare-child"]
        with (output / "prepare.log").open("x") as stream:
            result = subprocess.run(argv, cwd=output, stdout=stream, stderr=subprocess.STDOUT)
        receipt["groups"].append({"name": "prepare-freeze", "argv": argv, "exit": result.returncode})
        assert result.returncode == 0
        receipt["prepare_identity"] = load(output / "prepare-identity.json")
        assert receipt["prepare_identity"]["before"] == receipt["prepare_identity"]["after"] == identity
        prepare = load(output / "standalone.json")
        assert prepare and all(x["status"] == "passed" for x in prepare), prepare
        receipt["final_installed_identity"] = installed_identity()
        assert receipt["final_installed_identity"] == identity
        receipt["passed"] = True
    except BaseException:
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        save(output / "receipt.json", receipt)


def cleanup_owned_container(boot, image, token, name, suffix=""):
    try:
        cidfile = boot / "container.cid"
        cid = cidfile.read_text().strip() if cidfile.exists() else ""
        target = cid if len(cid) == 64 and all(c in "0123456789abcdef" for c in cid) else name
        inspected = subprocess.run(["docker", "inspect", target], text=True, capture_output=True)
        state = {"exit": inspected.returncode, "stdout": inspected.stdout, "stderr": inspected.stderr}
        save(boot / ("container" + suffix + ".json"), state)
        if inspected.returncode == 0:
            container = json.loads(inspected.stdout)[0]
            if container["Image"] != image or container["Config"].get("Labels", {}).get("spark.qualification-token") != token:
                raise RuntimeError("cannot reach or validate owned gate container cleanup; leave unowned state untouched")
            removed = subprocess.run(["docker", "rm", "-f", container["Id"]], text=True, capture_output=True)
            save(boot / ("cleanup" + suffix + ".json"), {"container_id": container["Id"], "exit": removed.returncode,
                                                       "stdout": removed.stdout, "stderr": removed.stderr})
            if removed.returncode:
                raise RuntimeError("cannot reach Docker for owned gate container cleanup")
        elif inspected.returncode != 1 or "no such" not in (inspected.stderr + inspected.stdout).lower():
            raise RuntimeError("cannot reach Docker to verify owned gate container cleanup")
        return state
    except BaseException as error:
        raise RuntimeError("cannot reach or validate owned gate container cleanup: " + str(error)) from error


def owned_container_run(argv, stream, boot, image, token):
    try:
        result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
    finally:
        state = cleanup_owned_container(boot, image, token, argv[argv.index("--name") + 1])
    return result, state


def host(output, source, image):
    assert not output.exists(), "preserve previous gate output"
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == HEAD
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True).strip() == TREE
    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
    assert inspected["Id"] == image
    assert inspected["Config"]["Labels"]["local.spark-ds41f.tilelang.tree"] == TREE
    plan = load(HERE / "gpu-gate-plan.json")
    output.mkdir(parents=True)
    staged = output / "canonical-tests"
    inputs = (plan["stage_files_from_canonical_source_readonly_only"] +
              plan.get("static_source_assertion_assets", []) + plan.get("dynamic_test_helper_assets", []))
    for record in inputs:
        path = source / record["source_relative"]
        assert path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], path
        target = staged / record["source_relative"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    support = output / "test-support"
    for record in plan["isolated_example_import_assets"]:
        path = source / record["source_relative"]
        assert path.stat().st_size == record["bytes"] and sha(path) == record["sha256"]
        assert record["source_relative"].startswith("examples/")
        target = support / record["source_relative"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    result = {"image": image, "tilelang_head": HEAD, "tilelang_tree": TREE,
              "plan_sha256": sha(HERE / "gpu-gate-plan.json"), "inputs": inputs, "boots": [], "passed": False}
    try:
        for number in (1, 2):
            boot = output / ("boot" + str(number))
            boot.mkdir()
            name = "compiler-c42-features-" + str(number)
            token = uuid.uuid4().hex
            save(boot / "ownership.json", {"name": name, "token": token, "image": image})
            argv = ["docker", "run", "--name", name, "--label", "spark.qualification-token=" + token,
                    "--cidfile", str((boot / "container.cid").resolve()),
                    "--gpus", "all", "--network", "none", "--ipc=host",
                    "--memory=32g", "--memory-swap=32g", "--ulimit=core=0", "--ulimit=memlock=-1:-1",
                    "-e", "TILELANG_CACHE_DIR=/out/tilelang-cache", "-e", "CUDA_CACHE_PATH=/out/cuda-cache",
                    "-e", "MAX_JOBS=2", "-e", "DO_NOT_TRACK=1", "-e", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                    "-e", "TORCH_EXTENSIONS_DIR=/out/torch-extensions", "-e", "PYTHONDONTWRITEBYTECODE=1",
                    "-v", str(staged.resolve()) + ":/tests:ro", "-v", str(boot.resolve()) + ":/out",
                    "-v", str(support.resolve()) + ":/test-support:ro",
                    "-w", "/out", "--entrypoint", "python3"]
            for file in ("compiler_gate.py", "gpu-gate-plan.json", "authored_standalone.py"):
                argv += ["-v", str(HERE / file) + ":/packet/" + file + ":ro"]
            argv += [image, "/packet/compiler_gate.py", "--inside", "--output", "/out", "--tests", "/tests"]
            with (output / ("boot" + str(number) + ".log")).open("x") as stream:
                run, state = owned_container_run(argv, stream, boot, image, token)
            row = {"name": name, "argv": argv, "exit": run.returncode}
            result["boots"].append(row)
            assert run.returncode == 0 and state["exit"] == 0, row
            container = json.loads(state["stdout"])[0]
            assert container["Image"] == image and not container["State"]["OOMKilled"]
            row["container_id"] = container["Id"]
            row["receipt"] = load(boot / "receipt.json")
            assert row["receipt"]["passed"]
        assert result["boots"][0]["container_id"] != result["boots"][1]["container_id"]
        assert result["boots"][0]["receipt"]["installed_identity"] == result["boots"][1]["receipt"]["installed_identity"]
        result["passed"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        save(output / "receipt.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--pytest-child", action="store_true")
    parser.add_argument("--prepare-child", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--tests", type=Path)
    parser.add_argument("--image")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.pytest_child or args.prepare_child:
        before = installed_identity()
        assert before == load(Path("/out/installed-identity.json"))
        print(json.dumps({"installed_identity": before}), flush=True)
        test_support()
        if args.pytest_child:
            import pytest
            arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
            xml = next(x.split("=", 1)[1] for x in arguments if x.startswith("--junitxml="))
            try:
                code = pytest.main(arguments)
            finally:
                after = installed_identity()
                save(Path(xml).with_name(Path(xml).stem + "-identity.json"), {"before": before, "after": after})
                assert before == after
            raise SystemExit(code)
        try:
            sys.argv = ["/packet/authored_standalone.py", "prepare"]
            runpy.run_path(sys.argv[0], run_name="__main__")
        finally:
            after = installed_identity()
            save(Path("/out/prepare-identity.json"), {"before": before, "after": after})
            assert before == after
    elif args.inside:
        assert args.output and args.tests
        inside(args.output, args.tests, load(HERE / "gpu-gate-plan.json"))
    else:
        assert args.source and args.image
        host(args.output, args.source, args.image)


if __name__ == "__main__":
    main()
