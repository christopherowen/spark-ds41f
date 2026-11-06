"""Offline trust-boundary controls; stdlib mocks only, no compiler/GPU import.

Authored but not executed locally. These exercise the real draft gate helpers;
they do not repeat the22 prepared-method tests or qualify an installed image.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c45_feature_gate_controls_target", HERE / "compiler_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

control_spec = importlib.util.spec_from_file_location("c45_build_control_target", HERE / "c45_build_control.py")
build_control = importlib.util.module_from_spec(control_spec)
control_spec.loader.exec_module(build_control)

cache_spec = importlib.util.spec_from_file_location("c45_cache_target", HERE / "c45_cache.py")
c45_cache = importlib.util.module_from_spec(cache_spec)
cache_spec.loader.exec_module(c45_cache)

mgmt_spec = importlib.util.spec_from_file_location("c45_management_target", HERE / "c45_management.py")
c45_management = importlib.util.module_from_spec(mgmt_spec)
mgmt_spec.loader.exec_module(c45_management)


class GateControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def test_repository_and_container_packet_roots(self):
        self.assertEqual(gate.repository_root(Path("/packet")), Path("/"))
        self.assertEqual(gate.repository_root(Path("/repo/experiments/c45")), Path("/repo"))

    def authority_fixture(self):
        root = self.directory / "repo"
        path = root / "results/private/candidate-rounds/candidate-45-image/pre-feature-authority.json"
        binding = {"installed_source_sha256": {"tilelang/jit/__init__.py": gate.JIT_SHA}}
        libraries = {"/installed/tilelang/lib/libtilelang.so": "a" * 64,
                     "/installed/tilelang/lib/libtvm_compiler.so": "b" * 64}
        identity = {"passed": True,
                    "provider": {"module_version": gate.VERSION, "distribution_version": gate.VERSION,
                                 "module": "/installed/tilelang/__init__.py", "module_sha256": "c" * 64,
                                 "tilelang_loader": "/installed/tilelang/lib/libtilelang.so",
                                 "tvm_loader": "/installed/tilelang/lib/libtvm_compiler.so",
                                 "installed_source_sha256": binding["installed_source_sha256"],
                                 "loaded_native_libraries": libraries, "installed_library_inventory": libraries},
                    "cache": {"base_key": {"version": gate.VERSION, "platform": "aarch64"},
                              "namespace": gate.NAMESPACE, "root": "/c/release"}}
        value = {"schema": "c45-compiler-feature-authority/1", "candidate": 45,
                 "stage": "release-and-topk-passed-features-pending", "passed": True,
                 "image": "sha256:" + "1" * 64, "commit": "2" * 40,
                 "source": {"vllm_head": gate.VLLM_HEAD, "vllm_tree": gate.VLLM_TREE,
                            "tilelang_head": gate.HEAD, "tilelang_tree": gate.TREE,
                            "tilelang_version": gate.VERSION, "tilelang_patchset_sha256": gate.PATCHSET},
                 "source_binding": binding, "gate": {"passed": True, "failed": 0, "errors": 0},
                 "native_gate_audit": {"passed": True, "files": gate.RELEASE_FILES,
                                       "expected_files": gate.RELEASE_FILES, "passed_cases": 1,
                                       "failed_cases": 0, "error_cases": 0, "results_sha256": "3" * 64},
                 "steps": [{"log": name, "exit": 0, "ended": "recorded-terminal-time"}
                           for name in gate.PREREQUISITE_LOGS],
                 "topk_initcheck": {"passed": True, "source_sha256": gate.TOPK_SHA},
                 "release_cache": {"namespace_absent_before_import": True, "namespace": gate.NAMESPACE,
                                   "version": gate.VERSION, "host_root": "/host/cache", "container_root": "/c/release"},
                 "installed_compiler_identity": identity}
        return root, path, binding, value

    def validate_authority(self, value, binding=None, expected_sha=None):
        root, path, original_binding, _ = self.authority_fixture()
        self.write_json(path, value)
        with mock.patch.object(gate, "ROOT", root):
            return gate.authority(path, expected_sha or gate.sha(path), binding or original_binding)

    def test_valid_passed_source_bound_prerequisite_authority(self):
        _, _, _, value = self.authority_fixture()
        self.assertEqual(self.validate_authority(value)["image"], value["image"])

    def test_authority_rejects_tag_wrong_source_or_unfinished_prerequisites(self):
        _, _, _, original = self.authority_fixture()
        variants = []
        for key, changed in (("image", "spark-ds41f:candidate-45"), ("passed", False), ("candidate", 42)):
            value = copy.deepcopy(original); value[key] = changed; variants.append(value)
        value = copy.deepcopy(original); value["source"]["tilelang_tree"] = "f" * 40; variants.append(value)
        value = copy.deepcopy(original); value["source"]["tilelang_version"] = "0.1.15"; variants.append(value)
        value = copy.deepcopy(original); value["steps"][0]["exit"] = 1; variants.append(value)
        value = copy.deepcopy(original); value["steps"].append(value["steps"][-1]); variants.append(value)
        value = copy.deepcopy(original); value["native_gate_audit"]["failed_cases"] = 1; variants.append(value)
        value = copy.deepcopy(original); value["native_gate_audit"]["files"] -= 1; variants.append(value)
        value = copy.deepcopy(original); value["release_cache"]["namespace_absent_before_import"] = False; variants.append(value)
        value = copy.deepcopy(original); value["installed_compiler_identity"]["passed"] = False; variants.append(value)
        value = copy.deepcopy(original); value["installed_compiler_identity"]["provider"]["tvm_loader"] = "/unrelated/libtvm_compiler.so"; variants.append(value)
        value = copy.deepcopy(original); value["installed_compiler_identity"]["cache"]["namespace"] = "old/linux-aarch64"; variants.append(value)
        for value in variants:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.validate_authority(value)

    def test_authority_rejects_changed_bytes_or_source_binding(self):
        _, _, binding, value = self.authority_fixture()
        with self.assertRaises(ValueError):
            self.validate_authority(value, expected_sha="0" * 64)
        changed = dict(binding, unexpected="changed")
        with self.assertRaises(ValueError):
            self.validate_authority(value, binding=changed)

    def test_authority_rejects_historical_output_path(self):
        root, _, binding, value = self.authority_fixture()
        path = root / "results/private/candidate-rounds/candidate-42-image/receipt.json"
        self.write_json(path, value)
        with mock.patch.object(gate, "ROOT", root), self.assertRaises(ValueError):
            gate.authority(path, gate.sha(path), binding)

    def test_plan_preserves_exact_original23_and_adds_exact_control_asset(self):
        plan = gate.load(HERE / "gpu-gate-plan.json")
        assets = gate.verify_plan(plan)
        self.assertEqual(len(assets), 23)
        self.assertEqual(len({row["source_relative"] for row in assets} | {plan["prepared_control_asset"]["source_relative"]}), 24)
        self.assertFalse(plan["dispatch_ready"])

    def test_plan_rejects_omission_changed_asset_selector_or_new_control(self):
        original = gate.load(HERE / "gpu-gate-plan.json")
        variants = []
        value = copy.deepcopy(original); value["stage_files_from_canonical_source_readonly_only"].pop(); variants.append(value)
        value = copy.deepcopy(original); value["static_source_assertion_assets"][0]["sha256"] = "0" * 64; variants.append(value)
        value = copy.deepcopy(original); value["retained_feature_pytest_selectors"].pop(); variants.append(value)
        value = copy.deepcopy(original); value["fix_specific"].pop(); variants.append(value)
        value = copy.deepcopy(original); value["prepared_control_asset"]["sha256"] = "0" * 64; variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                gate.verify_plan(value)

    def test_source_binding_rejects_wrong_head_tree_or_dirty_source_before_staging(self):
        plan = gate.load(HERE / "gpu-gate-plan.json")
        for responses in (("0" * 40,), (gate.HEAD, "0" * 40), (gate.HEAD, gate.TREE, " M tilelang/jit/__init__.py")):
            with self.subTest(responses=responses), \
                 mock.patch.object(gate.subprocess, "check_output", side_effect=responses), self.assertRaises(ValueError):
                gate.source_binding(Path("/unused-canonical-source"), plan)

    def cache_environment(self, output):
        return {"TILELANG_CACHE_DIR": str(output / "tilelang-cache"),
                "CUDA_CACHE_PATH": str(output / "cuda-cache")}

    def test_distinct_initially_absent_caches_are_recorded_without_creation(self):
        for name in ("first", "repeat"):
            output = self.directory / name; output.mkdir()
            with mock.patch.dict(os.environ, self.cache_environment(output)):
                gate.absent_caches(output)
            record = gate.load(output / "cache-before-import.json")
            self.assertTrue(all(row["absent"] for row in record.values()))
            self.assertFalse((output / "tilelang-cache").exists())

    def test_stale_cache_refusal_preserves_native_files(self):
        output = self.directory / "stale"; output.mkdir()
        cache = output / "tilelang-cache"; cache.mkdir()
        sentinel = cache / "original"; sentinel.write_text("preserve\n")
        with mock.patch.dict(os.environ, self.cache_environment(output)), self.assertRaises(ValueError):
            gate.absent_caches(output)
        self.assertEqual(sentinel.read_text(), "preserve\n")
        self.assertFalse(gate.load(output / "cache-before-import.json")["tilelang"]["absent"])

    def test_cache_environment_mismatch_is_refused(self):
        output = self.directory / "wrong-env"; output.mkdir()
        with mock.patch.dict(os.environ, {"TILELANG_CACHE_DIR": "/other", "CUDA_CACHE_PATH": "/other"}), self.assertRaises(ValueError):
            gate.absent_caches(output)

    def identity_fixture(self):
        site = self.directory / "site"
        module = site / "tilelang/__init__.py"
        paths = [module, site / "tilelang/jit/__init__.py", site / "tilelang/cache/kernel_cache.py",
                 site / "tilelang/lib/libtilelang.so", site / "tilelang/lib/libtvm_compiler.so"]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(path.name + " fixture\n")
        state = types.SimpleNamespace(base={"version": gate.VERSION, "platform": "aarch64"},
                                      namespace=gate.NAMESPACE, stamp=False)
        cache_root = self.directory / "actual-cache"
        fake_cache = type("KernelCache", (), {
            "_get_base_key": staticmethod(lambda: state.base),
            "_get_cache_namespace": staticmethod(lambda: state.namespace),
            "_get_namespace_root": staticmethod(lambda: str(cache_root / state.namespace))})
        tilelang = types.ModuleType("tilelang")
        tilelang.__path__ = [str(module.parent)]
        tilelang.__file__, tilelang.__version__, tilelang._LIB_PATH = str(module), gate.VERSION, str(paths[-2])
        tilelang.env = types.SimpleNamespace(should_use_kernel_cache_lib_stamp=lambda: state.stamp)
        cache_module = types.ModuleType("tilelang.cache"); cache_module.__path__ = []
        kernel_module = types.ModuleType("tilelang.cache.kernel_cache"); kernel_module.KernelCache = fake_cache
        torch = types.ModuleType("torch"); torch.cuda = types.SimpleNamespace(get_device_capability=lambda: (12, 1))
        tvm = types.ModuleType("tvm"); tvm.__path__ = []
        tvm_base = types.ModuleType("tvm.base"); tvm_base._LIB = types.SimpleNamespace(_name=str(paths[-1])); tvm.base = tvm_base
        dist = types.SimpleNamespace(version=gate.VERSION,
                                     files=[str(path.relative_to(site)) for path in paths],
                                     locate_file=lambda relative: site / relative)
        modules = {"tilelang": tilelang, "tilelang.cache": cache_module,
                   "tilelang.cache.kernel_cache": kernel_module, "torch": torch, "tvm": tvm, "tvm.base": tvm_base}
        maps = "\n".join("1-2 r-xp 0 0:0 0 " + str(path) for path in paths[-2:])
        binding = {"installed_source_sha256": {str(path.relative_to(site)): gate.sha(path) for path in paths[1:3]}}
        return types.SimpleNamespace(module=tilelang, dist=dist, state=state, modules=modules,
                                     maps=maps, binding=binding, cache_root=cache_root, paths=paths)

    def installed_identity(self, fixture):
        original_is_file, original_read_text = Path.is_file, Path.read_text
        def is_file(path):
            return True if str(path) == "/.dockerenv" else original_is_file(path)
        def read_text(path, *args, **kwargs):
            return fixture.maps if str(path) == "/proc/self/maps" else original_read_text(path, *args, **kwargs)
        with mock.patch.dict(sys.modules, fixture.modules), \
             mock.patch.object(gate.importlib.metadata, "distribution", return_value=fixture.dist), \
             mock.patch.object(Path, "is_file", is_file), mock.patch.object(Path, "read_text", read_text), \
             mock.patch.dict(os.environ, {"TILELANG_CACHE_DIR": str(fixture.cache_root)}):
            return gate.installed_identity(fixture.binding)

    def test_installed_identity_binds_actual_sources_selected_libraries_and_namespace(self):
        fixture = self.identity_fixture()
        result = self.installed_identity(fixture)
        self.assertEqual(result["provider"]["installed_source_sha256"], fixture.binding["installed_source_sha256"])
        self.assertEqual(len(result["provider"]["loaded_native_libraries"]), 2)
        self.assertEqual(result["cache"]["base_key"], fixture.state.base)

    def test_installed_identity_rejects_module_distribution_source_and_cache_mismatch(self):
        for kind in ("module-version", "dist-version", "module-path", "jit-hash", "base-key", "namespace", "stamp"):
            with self.subTest(kind=kind):
                fixture = self.identity_fixture()
                if kind == "module-version": fixture.module.__version__ = "0.1.15"
                elif kind == "dist-version": fixture.dist.version = "0.1.15"
                elif kind == "module-path": fixture.module.__file__ = str(self.directory / "other/__init__.py")
                elif kind == "jit-hash": fixture.binding["installed_source_sha256"]["tilelang/jit/__init__.py"] = "0" * 64
                elif kind == "base-key": fixture.state.base = {"version": "0.1.15", "platform": "aarch64"}
                elif kind == "namespace": fixture.state.namespace = "old/linux-aarch64"
                else: fixture.state.stamp = True
                with self.assertRaises(ValueError):
                    self.installed_identity(fixture)

    def test_installed_identity_rejects_unmapped_foreign_or_uninventoried_selected_dso(self):
        for kind in ("unmapped", "foreign", "uninventoried"):
            with self.subTest(kind=kind):
                fixture = self.identity_fixture()
                if kind == "unmapped": fixture.maps = fixture.maps.splitlines()[-1]
                elif kind == "foreign": fixture.maps += "\n1-2 r-xp 0 0:0 0 /unrelated/libtilelang.so"
                else: fixture.dist.files.remove(str(fixture.paths[-2].relative_to(self.directory / "site")))
                with self.assertRaises(ValueError):
                    self.installed_identity(fixture)

    def test_host_refuses_draft_or_missing_control_without_any_subprocess(self):
        with mock.patch.object(gate.subprocess, "check_output") as command:
            with self.assertRaises(ValueError):
                gate.host(self.directory / "out", Path("/unused"), Path("/unused"), "0" * 64)
            with self.assertRaises(ValueError):
                gate.host(self.directory / "out", Path("/unused"), Path("/unused"), "0" * 64, control=mock.Mock())
        command.assert_not_called()

    def test_existing_output_refusal_preserves_original_receipt(self):
        output = self.directory / "prior-output"; output.mkdir()
        original = output / "receipt.json"; original.write_text("original native verdict\n")
        plan = copy.deepcopy(gate.load(HERE / "gpu-gate-plan.json")); plan["dispatch_ready"] = True
        with mock.patch.object(gate, "load", return_value=plan), \
             mock.patch.object(gate.subprocess, "check_output") as command, self.assertRaises(ValueError):
            gate.host(output, Path("/unused"), Path("/unused"), "0" * 64, control=mock.Mock())
        command.assert_not_called()
        self.assertEqual(original.read_text(), "original native verdict\n")

    def test_host_rejects_inspected_image_or_source_labels_before_launch(self):
        for wrong in ("image", "labels"):
            with self.subTest(wrong=wrong):
                _, _, binding, value = self.authority_fixture()
                plan = copy.deepcopy(gate.load(HERE / "gpu-gate-plan.json")); plan["dispatch_ready"] = True
                labels = {"local.spark-ds41f.tilelang.tree": gate.TREE,
                          "local.spark-ds41f.tilelang.patch-head": gate.HEAD,
                          "local.spark-ds41f.vllm.tree": gate.VLLM_TREE,
                          "local.spark-ds41f.vllm.patch-head": gate.VLLM_HEAD}
                inspected = {"Id": value["image"], "Config": {"Labels": labels}}
                if wrong == "image": inspected["Id"] = "sha256:" + "f" * 64
                else: labels["local.spark-ds41f.tilelang.tree"] = "f" * 40
                def command(argv, **kwargs):
                    if argv[0] == "git":
                        return "" if "status" in argv else value["commit"] + "\n"
                    if argv[:3] == ["docker", "image", "inspect"]:
                        return json.dumps([inspected])
                    self.fail("unexpected command: " + repr(argv))
                control = mock.Mock()
                with mock.patch.object(gate, "load", return_value=plan), \
                     mock.patch.object(gate, "source_binding", return_value=binding), \
                     mock.patch.object(gate, "authority", return_value=value), \
                     mock.patch.object(gate.subprocess, "check_output", side_effect=command), \
                     mock.patch.object(gate.subprocess, "run") as launched, self.assertRaises(ValueError):
                    gate.host(self.directory / ("out-" + wrong), Path("/unused"), Path("/unused"), "0" * 64,
                              control=control)
                launched.assert_not_called()
                control.run_container.assert_not_called()

    def test_build_control_binds_locks_scope_and_owned_container_stops(self):
        locks = {}
        for name in ("gpu", "cpu-work", "cpu-offline"):
            path = self.directory / f"{name}.lock"
            handle = path.open("a+")
            self.addCleanup(handle.close)
            locks[name] = handle
        commit, window, holder, token = "2" * 40, "dgx1-test", "spark-lab: exact C45", "a" * 32
        state = {"passed": True, "commit": commit, "window": window, "holder": holder, "owner_token": token}
        watch = mock.Mock()
        management = c45_management.ManagementLatch()
        receipt = {}
        control = build_control.BuildControl(
            commit=commit, window=window, holder=holder, token=token, locks=locks,
            watch=watch, scope_check=lambda: dict(state), management=management,
            receipt=receipt, output=self.directory / "build-out", save=lambda: None)
        scope = control.assert_scope(dict(state))
        self.assertEqual(set(scope["held_locks"]), {"gpu", "cpu-work", "cpu-offline"})
        watch.check.assert_called_once()
        with self.assertRaises(RuntimeError):
            control.assert_scope(dict(state, owner_token="b" * 32))
        with self.assertRaises(ValueError):
            control.register_container("foreign-container", "tok", "sha256:" + "1" * 64)
        image = "sha256:" + "1" * 64
        control.register_container("compiler-c45-features-1", "own-tok", image)
        inspected = {"Id": "c" * 64, "Image": image, "Config": {"Labels": {"spark.qualification-token": "own-tok"}}}
        with mock.patch.object(build_control.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 0, json.dumps([inspected]), ""),
            subprocess.CompletedProcess([], 0, "c" * 64 + "\n", ""),
        ]) as command:
            control.stop_owned_containers()
        self.assertEqual(command.call_args_list[-1].args[0], ["docker", "stop", "--time", "15", "c" * 64])
        self.assertFalse(management.latched)

    def test_serving_cache_mapping_and_rank_collector(self):
        root = HERE.parents[1]
        for rel in ("experiments/2026-10-26-spark-ds41f-candidate/candidate-45/tp3-qualification.json",
                    "experiments/2026-10-26-spark-ds41f-candidate/candidate-45/tp3-qualification-dev.json"):
            cluster = json.loads((root / rel).read_text())
            mapping = c45_cache.mapped_namespace(cluster, lambda _cluster, src: "/home/swank/" + src.removeprefix("${HOME}/"))
            self.assertEqual(mapping["host_root"], "/home/swank/.cache/vllm/tilelang")
            self.assertEqual(mapping["namespace"], c45_cache.NAMESPACE)
        management = mock.Mock()
        management.ssh.side_effect = [json.dumps({"passed": True, "node": n}) for n in ("dgx1", "dgx2", "dgx3")]
        out = self.directory / "serving-first-use.json"
        result = c45_cache.collect(management, lambda: None, "profile.json", "sha256:" + "1" * 64, "2" * 40, out)
        self.assertTrue(result["passed"])
        self.assertEqual(set(result["ranks"]), {"dgx1", "dgx2", "dgx3"})
        with self.assertRaises(FileExistsError):
            c45_cache.collect(management, lambda: None, "profile.json", "sha256:" + "1" * 64, "2" * 40, out)

    def test_native_failure_exit_and_original_log_are_preserved(self):
        output = self.directory / "failed-native"; output.mkdir()
        identity = {"provider": {"fixture": "actual-provider"}}
        def failed(argv, stdout, **kwargs):
            stdout.write("original native failure\n")
            return types.SimpleNamespace(returncode=7)
        with mock.patch.dict(os.environ, self.cache_environment(output)), \
             mock.patch.object(gate, "installed_identity", return_value=identity), \
             mock.patch.object(gate, "test_support"), \
             mock.patch.object(gate.subprocess, "run", side_effect=failed), self.assertRaises(ValueError):
            gate.inside(output, self.directory / "tests", gate.load(HERE / "gpu-gate-plan.json"), {},
                        {"installed_compiler_identity": identity})
        receipt = gate.load(output / "receipt.json")
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["groups"][0]["exit"], 7)
        self.assertEqual((output / "retained-features.log").read_text(), "original native failure\n")

    def test_cleanup_removes_only_exact_image_token_and_id(self):
        for wrong in (None, "image", "token"):
            with self.subTest(wrong=wrong):
                boot = self.directory / ("cleanup-" + str(wrong)); boot.mkdir()
                cid, image, token = "a" * 64, "sha256:" + "b" * 64, "own-token"
                (boot / "container.cid").write_text(cid)
                state = {"Id": cid, "Image": image, "Config": {"Labels": {"spark.qualification-token": token}}}
                if wrong == "image": state["Image"] = "sha256:" + "c" * 64
                if wrong == "token": state["Config"]["Labels"]["spark.qualification-token"] = "foreign"
                responses = [subprocess.CompletedProcess([], 0, json.dumps([state]), ""),
                             subprocess.CompletedProcess([], 0, cid + "\n", "")]
                with mock.patch.object(gate.subprocess, "run", side_effect=responses) as command:
                    if wrong:
                        with self.assertRaises(ValueError):
                            gate.cleanup_owned(boot, image, token, "owned-name")
                        self.assertEqual(command.call_count, 1)
                    else:
                        gate.cleanup_owned(boot, image, token, "owned-name")
                        self.assertEqual(command.call_args_list[-1].args[0], ["docker", "rm", "-f", cid])
                        self.assertEqual(gate.load(boot / "cleanup.json")["container_id"], cid)
                self.assertEqual(gate.load(boot / "container.json")["stdout"], json.dumps([state]))

    def test_unreachable_cleanup_is_preserved_and_does_not_remove_anything(self):
        boot = self.directory / "unreachable"; boot.mkdir()
        response = subprocess.CompletedProcess([], 255, "", "transport unavailable")
        with mock.patch.object(gate.subprocess, "run", return_value=response) as command, self.assertRaises(ValueError):
            gate.cleanup_owned(boot, "sha256:" + "1" * 64, "token", "owned-name")
        self.assertEqual(command.call_count, 1)
        self.assertEqual(gate.load(boot / "container.json")["exit"], 255)


if __name__ == "__main__":
    unittest.main()
