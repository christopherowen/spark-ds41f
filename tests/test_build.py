"""Host-independent tests for the build directory identity."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.machinery
import importlib.util
import io
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
spec = importlib.util.spec_from_loader("spark", loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)


class BuildDirectoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = spark.build_inputs(spark.read_json("upstreams.lock.json"))

    def test_inputs_come_from_the_lock_and_source_manifest(self) -> None:
        upstreams = spark.read_json("upstreams.lock.json")
        manifest = spark.read_json(upstreams["source_manifest"])
        for name in spark.BUILD_PROJECTS:
            self.assertEqual(self.inputs[name]["revision"], upstreams["sources"][name]["revision"])
            self.assertEqual(self.inputs[name]["expected_tree"], manifest[name]["expected_tree"])
            self.assertEqual(self.inputs[name]["patch_head"], manifest[name]["patch_head"])

    def test_directory_is_stable_and_follows_every_input(self) -> None:
        directory = spark.build_directory(self.inputs)
        self.assertEqual(directory, spark.build_directory(copy.deepcopy(self.inputs)))
        self.assertEqual(directory.parent, ROOT / ".work/build")
        self.assertTrue(directory.name.startswith(
            f"vllm-{self.inputs['vllm']['revision'][:12]}-b12x-{self.inputs['b12x']['revision'][:12]}-"
        ))
        changed = copy.deepcopy(self.inputs)
        changed["cutlass"]["revision"] = "0" * 40
        self.assertNotEqual(directory, spark.build_directory(changed))

    def test_tree_digest_ignores_git_metadata(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.py").write_text("x = 1\n")
            before = spark.tree_digest(root)
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
            self.assertEqual(before, spark.tree_digest(root))
            (root / "a.py").write_text("x = 2\n")
            self.assertNotEqual(before, spark.tree_digest(root))

    def test_normalize_modes_uses_git_modes(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "d").mkdir(mode=0o775)
            plain = root / "d" / "plain.txt"
            plain.write_text("x")
            plain.chmod(0o664)
            tool = root / "tool.sh"
            tool.write_text("#!/bin/sh\n")
            tool.chmod(0o775)
            spark.normalize_modes(root)
            self.assertEqual(plain.stat().st_mode & 0o777, 0o644)
            self.assertEqual(tool.stat().st_mode & 0o777, 0o755)
            self.assertEqual((root / "d").stat().st_mode & 0o777, 0o755)


class WheelRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = spark.read_json("upstreams.lock.json")
        self.manifest = spark.read_json(self.lock["source_manifest"])
        self.lock["build"] = {"dockerfile": "docker/Dockerfile.vllm032"}
        self.lock["sources"]["b12x"] = {
            "kind": "pypi-wheel-set", "version": "1.5.1",
            "lockfile": "requirements/vllm032-aarch64.txt",
        }
        self.manifest["b12x"] = {"version": "1.5.1"}
        self.lock["sources"]["cutlass_dsl"]["lockfile"] = "requirements/cutlass-dsl-4.8-aarch64.txt"

    def inputs(self):
        with patch.object(spark, "read_json", return_value=self.manifest):
            return spark.build_inputs(self.lock)

    def test_package_runtime_skips_git_source_and_exports_locked_wheels(self):
        inputs = self.inputs()
        self.assertNotIn("b12x", spark.build_projects(self.lock["sources"]))
        self.assertNotIn("b12x", spark.build_projects(inputs))
        self.assertNotIn("b12x-source", spark.build_contexts(inputs))
        self.assertIn("runtime-wheels", spark.build_contexts(inputs))
        self.assertEqual(inputs["b12x"]["version"], "1.5.1")
        self.assertEqual(len(inputs["b12x"]["wheel_hashes"]), 6)
        self.assertIn("-b12x-1.5.1-", spark.build_directory(inputs).name)
        self.assertEqual(inputs["dockerfile"], "docker/Dockerfile.vllm032")
        for key in ("lock_sha256", "wheel_hashes"):
            changed = copy.deepcopy(inputs)
            changed["b12x"][key] = "changed"
            self.assertNotEqual(spark.build_directory(inputs), spark.build_directory(changed))
        changed = copy.deepcopy(inputs)
        changed["dockerfile_sha256"] = "changed"
        self.assertNotEqual(spark.build_directory(inputs), spark.build_directory(changed))

    def test_wheel_mode_keeps_vllm_and_nccl_required(self):
        for name in ("vllm", "nccl"):
            sources = copy.deepcopy(self.lock["sources"])
            del sources[name]
            with self.assertRaisesRegex(SystemExit, f"required build source {name}"):
                spark.build_projects(sources)

    def test_cutlass_wheels_follow_selected_lock_without_changing_legacy(self):
        inputs = self.inputs()
        self.assertEqual(inputs["cutlass_dsl_wheels"], spark.expected_wheel_hashes("requirements/cutlass-dsl-4.8-aarch64.txt"))
        self.assertNotEqual(inputs["cutlass_dsl_wheels"], spark.expected_wheel_hashes())

    def test_image_plan_uses_selected_dockerfile_and_wheel_context(self):
        inputs = self.inputs()
        cluster = {"container": {"image": "test:v032"}}
        args = argparse.Namespace(apply=False, tag=None)
        output = io.StringIO()
        with (patch.object(spark, "configuration", return_value=(cluster, {}, self.lock)),
              patch.object(spark, "build_inputs", return_value=inputs),
              patch.object(spark, "build_problems", return_value=[]),
              patch.object(spark, "repository_dirty", return_value=False),
              patch.object(spark, "repository_revision", return_value="a" * 40),
              contextlib.redirect_stdout(output)):
            self.assertEqual(spark.command_build_image(args), 0)
        command = shlex.split(output.getvalue().splitlines()[0])
        self.assertEqual(command[command.index("--file") + 1], str(ROOT / "docker/Dockerfile.vllm032"))
        contexts = [command[i + 1] for i, part in enumerate(command) if part == "--build-context"]
        self.assertTrue(any(context.startswith("runtime-wheels=") for context in contexts))
        self.assertFalse(any(context.startswith("b12x-source=") for context in contexts))

    def test_manifest_package_version_must_match(self):
        self.manifest["b12x"]["version"] = "1.5.0"
        with self.assertRaisesRegex(SystemExit, "wheel version"):
            self.inputs()

    def test_package_version_must_match_the_hashed_lock(self):
        self.lock["sources"]["b12x"]["version"] = "1.5.0"
        self.manifest["b12x"]["version"] = "1.5.0"
        with self.assertRaisesRegex(SystemExit, "lock must pin"):
            self.inputs()

    def test_dockerfile_cannot_escape_repository(self):
        for path in ("/tmp/Dockerfile", "../Dockerfile"):
            self.lock["build"]["dockerfile"] = path
            with self.assertRaisesRegex(SystemExit, "inside the repository"):
                self.inputs()

    def test_runtime_wheels_are_verified_before_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "b12x-1.5.1-py3-none-any.whl"
            wheel.write_bytes(b"locked wheel")
            expected = {"b12x": hashlib.sha256(wheel.read_bytes()).hexdigest()}
            self.assertEqual(spark.wheel_problems(root, expected), [])
            wheel.write_bytes(b"changed")
            self.assertIn("differs from the lock hash", spark.wheel_problems(root, expected)[0])
            wheel.unlink()
            self.assertIn("missing b12x", spark.wheel_problems(root, expected)[0])
            (root / "unlocked.whl").write_bytes(b"extra")
            self.assertTrue(any("unlocked file" in p for p in spark.wheel_problems(root, expected)))

    def test_runtime_wheels_download_verify_and_reuse(self):
        payload = b"verified wheel"
        expected = {"b12x": hashlib.sha256(payload).hexdigest()}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            def download(command):
                destination = Path(command[command.index("--dest") + 1])
                (destination / "b12x-1.5.1-py3-none-any.whl").write_bytes(payload)
            with (patch.object(spark, "expected_wheel_hashes", return_value=expected),
                  patch.object(spark, "checked", side_effect=download) as checked):
                spark.prepare_wheels(directory, self.lock, runtime=True)
                spark.prepare_wheels(directory, self.lock, runtime=True)
            self.assertEqual(checked.call_count, 1)
            command = checked.call_args.args[0]
            self.assertIn("--require-hashes", command)
            self.assertEqual(command[-1], str(ROOT / self.lock["sources"]["b12x"]["lockfile"]))
            self.assertEqual(spark.wheel_problems(directory / "context/runtime-wheels", expected), [])

    def test_download_failure_does_not_replace_verified_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            destination = directory / "context/runtime-wheels"
            destination.mkdir(parents=True)
            marker = destination / "old.whl"
            marker.write_bytes(b"previous context")
            with patch.object(spark, "checked", side_effect=RuntimeError("download failed")):
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    spark.prepare_wheels(directory, self.lock, runtime=True)
            self.assertEqual(marker.read_bytes(), b"previous context")
            self.assertEqual(list(destination.parent.glob(".wheels.*")), [])


if __name__ == "__main__":
    unittest.main()
