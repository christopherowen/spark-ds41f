"""Reject incomplete compiler evidence before any image can enter serving."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


controls = module("compiler_round_controls", "test_controls.py")
gate = module("compiler_native_gate", "compiler_gate.py")


class CompilerAuthorityTests(unittest.TestCase):
    def test_missing_failed_stale_or_partial_compiler_gate_cannot_reach_serving(self):
        for kind in ("missing", "failed", "wrong_image", "wrong_tree", "wrong_head", "stale_plan",
                     "one_boot", "failed_boot", "incomplete_boot", "wrong_source_label"):
            image = controls.passing_image()
            value = image["compiler_features"]
            if kind == "missing": del image["compiler_features"]
            elif kind == "failed": value["passed"] = False
            elif kind == "wrong_image": value["image"] = "sha256:" + "c" * 64
            elif kind == "wrong_tree": value["tilelang_tree"] = "d" * 40
            elif kind == "wrong_head": image["source"]["tilelang_head"] = "e" * 40
            elif kind == "stale_plan": value["plan_sha256"] = "f" * 64
            elif kind == "one_boot": value["boots"].pop()
            elif kind == "failed_boot": value["boots"][1]["exit"] = 1
            elif kind == "incomplete_boot": value["boots"][1]["receipt"]["passed"] = False
            elif kind == "wrong_source_label": image["source"]["expected_labels"]["local.spark-ds41f.tilelang.tree"] = "0" * 40
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                controls.round_control.validate_image_gate(image)

    def test_steps_must_be_complete_and_ordered(self):
        for kind in ("missing", "reordered", "duplicate", "failed"):
            image = controls.passing_image()
            if kind == "missing": image["steps"].pop()
            elif kind == "reordered": image["steps"].reverse()
            elif kind == "duplicate": image["steps"].append(copy.deepcopy(image["steps"][-1]))
            elif kind == "failed": image["steps"][-1]["exit"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                controls.round_control.validate_image_gate(image)


class CompilerOutcomesTests(unittest.TestCase):
    def outcome(self, cases, minimum=1, allowed=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.xml"
            path.write_text("<testsuites><testsuite>" + cases + "</testsuite></testsuites>")
            return gate.junit(path, minimum, allowed)

    def test_empty_failed_errored_or_unexpectedly_skipped_gates_are_rejected(self):
        for cases in ("", '<testcase name="a"><failure/></testcase>',
                      '<testcase name="a"><error/></testcase>',
                      '<testcase name="a"/><testcase name="b"><skipped message="missing compiler"/></testcase>'):
            with self.subTest(cases=cases), self.assertRaises(AssertionError):
                self.outcome(cases, allowed=True)

    def test_only_declared_hardware_skips_are_allowed_beside_real_passes(self):
        cases = ('<testcase name="a"/><testcase name="test_nvf4_mma_block_scale_fragment_a_unrolls_k_atoms_and_packs_scales[policy]">'
                 '<skipped message="Requires CUDA compute eq 12.0, but have 12.1"/></testcase>')
        with self.assertRaises(AssertionError): self.outcome(cases)
        result = self.outcome(cases, allowed=True)
        self.assertEqual(result["passed"], 1)
        self.assertEqual(len(result["skipped"]), 1)

    def test_success_cannot_hide_missing_required_cases(self):
        with self.assertRaises(AssertionError): self.outcome('<testcase name="a"/>', minimum=2)


class OwnedContainerTests(unittest.TestCase):
    def state(self, token="owned"):
        return subprocess.CompletedProcess([], 0, json.dumps([{"Id": "container-id", "Image": "image-id",
                                            "Config": {"Labels": {"spark.qualification-token": token}}}]), "")

    def test_interruption_still_removes_only_the_created_owned_container(self):
        with tempfile.TemporaryDirectory() as directory:
            boot = Path(directory)
            with mock.patch.object(gate.subprocess, "run", side_effect=[KeyboardInterrupt(), self.state(),
                                   subprocess.CompletedProcess([], 0, "removed", "")]) as run:
                with self.assertRaises(KeyboardInterrupt):
                    gate.owned_container_run(["docker", "run", "--name", "gate-name"], io.StringIO(), boot, "image-id", "owned")
                self.assertEqual(run.call_args_list[-1].args[0], ["docker", "rm", "-f", "container-id"])

    def test_collision_with_another_owner_is_never_removed(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(gate.subprocess, "run", return_value=self.state("foreign")) as run:
            with self.assertRaisesRegex(RuntimeError, "cannot reach or validate"):
                gate.cleanup_owned_container(Path(directory), "image-id", "owned", "gate-name")
            self.assertEqual(run.call_count, 1)

    def test_empty_or_partial_cidfile_checks_owned_name_before_claiming_absence(self):
        for value in ("", "incomplete"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                boot = Path(directory)
                (boot / "container.cid").write_text(value)
                with mock.patch.object(gate.subprocess, "run", side_effect=[self.state(),
                                       subprocess.CompletedProcess([], 0, "removed", "")]) as run:
                    gate.cleanup_owned_container(boot, "image-id", "owned", "gate-name")
                self.assertEqual(run.call_args_list[0].args[0], ["docker", "inspect", "gate-name"])
                self.assertEqual(run.call_args_list[1].args[0], ["docker", "rm", "-f", "container-id"])

    def test_failed_removal_or_malformed_inspection_requires_review(self):
        for values in ([self.state(), subprocess.CompletedProcess([], 1, "", "daemon lost")],
                       [subprocess.CompletedProcess([], 0, "not-json", "")]):
            with self.subTest(values=values), tempfile.TemporaryDirectory() as directory, \
                    mock.patch.object(gate.subprocess, "run", side_effect=values), \
                    self.assertRaisesRegex(RuntimeError, "cannot reach or validate"):
                gate.cleanup_owned_container(Path(directory), "image-id", "owned", "gate-name")


class TestInputIsolationTests(unittest.TestCase):
    def test_canonical_examples_support_does_not_import_an_unrelated_regular_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            support = root / "support"
            package = support / "examples/dequantize_gemm/quantize"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("value = 'canonical-support'\n")
            foreign = root / "foreign/examples"
            foreign.mkdir(parents=True)
            (foreign / "__init__.py").write_text("raise RuntimeError('unrelated vLLM examples imported')\n")
            previous = {k: v for k, v in sys.modules.items() if k == "examples" or k.startswith("examples.")}
            try:
                for key in previous: del sys.modules[key]
                with mock.patch.object(sys, "path", [str(root / "foreign"), *sys.path]):
                    gate.isolate_examples(support)
                    import examples.dequantize_gemm.quantize as quantize
                    self.assertEqual(quantize.value, "canonical-support")
                    self.assertTrue(Path(quantize.__file__).is_relative_to(support))
            finally:
                for key in list(sys.modules):
                    if key == "examples" or key.startswith("examples."): del sys.modules[key]
                sys.modules.update(previous)



if __name__ == "__main__":
    unittest.main()
