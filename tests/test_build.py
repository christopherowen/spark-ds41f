"""Host-independent tests for the build directory identity."""

from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
import unittest
from pathlib import Path

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


class LockWithoutB12xTest(unittest.TestCase):
    """A lock without B12X (the release after next, experiments/2026-10-26-retire-b12x)
    builds no B12X project, context, stage or label, and its smoke checks B12X is absent;
    the promoted lock, which lists B12X, builds as before."""

    def setUp(self) -> None:
        self.lock = spark.read_json("upstreams.lock.json")
        self.without = copy.deepcopy(self.lock)
        del self.without["sources"]["b12x"]

    def plan(self, lock: dict) -> str:
        import argparse
        import contextlib
        import io
        from unittest import mock

        output = io.StringIO()
        cluster = spark.read_json("config/cluster.json")
        with mock.patch.object(spark, "configuration", return_value=(cluster, None, lock)), \
                mock.patch.object(spark, "build_problems", return_value=[]), \
                mock.patch.object(spark, "repository_dirty", return_value=""), \
                mock.patch.object(spark, "repository_revision", return_value="0" * 40), \
                contextlib.redirect_stdout(output):
            spark.command_build_image(argparse.Namespace(cluster_config="c.json", tag=None, apply=False))
        return output.getvalue()

    def test_inputs_contexts_and_directory_leave_b12x_out(self) -> None:
        promoted = spark.build_inputs(self.lock)
        inputs = spark.build_inputs(self.without)
        self.assertEqual(inputs, {name: value for name, value in promoted.items() if name != "b12x"})
        self.assertNotIn("b12x", spark.build_projects(inputs))
        self.assertNotIn("b12x-source", spark.build_contexts(inputs))
        self.assertIn("b12x-source", spark.build_contexts(promoted))
        name = spark.build_directory(inputs).name
        self.assertRegex(name, rf"^vllm-{inputs['vllm']['revision'][:12]}-[0-9a-f]{{8}}$")
        self.assertIn(f"-b12x-{promoted['b12x']['revision'][:12]}-", spark.build_directory(promoted).name)

    def test_the_tilelang_runtime_builds_on_the_core_stage(self) -> None:
        promoted = spark.build_inputs(self.lock)
        inputs = spark.build_inputs(self.without)
        self.assertEqual(spark.tilelang_base_stage(promoted), "runtime")
        self.assertEqual(spark.tilelang_base_stage(inputs), "runtime-core")
        self.assertEqual(spark.runtime_stage(inputs), spark.runtime_stage(promoted))
        # Without B12X the image is TileLang's: a lock without either is refused.
        bare = {name: value for name, value in inputs.items()
                if name not in (*spark.OPTIONAL_BUILD_PROJECTS, *spark.INDEPENDENT_BUILD_PROJECTS)}
        with self.assertRaises(SystemExit):
            spark.runtime_stage(bare)

    def test_image_plan_has_no_b12x_context_or_arguments(self) -> None:
        promoted, without = self.plan(self.lock), self.plan(self.without)
        for expected in ("TILELANG_BASE_STAGE=runtime ", "--build-context b12x-source=", "B12X_TREE="):
            self.assertIn(expected, promoted + " ")
        self.assertIn("TILELANG_BASE_STAGE=runtime-core", without)
        for absent in ("b12x-source", "B12X_"):
            self.assertNotIn(absent, without)

    def test_smoke_imports_no_b12x_module_and_checks_it_is_absent(self) -> None:
        promoted, without = spark.smoke_script(self.lock), spark.smoke_script(self.without)
        compile(without, "smoke", "exec")
        self.assertIn(spark.B12X_MLA_IMPORT, promoted)
        self.assertNotIn("find_spec", promoted)
        self.assertNotIn("b12x_mla_sparse", without)
        self.assertNotIn("b12x_roce_all_reduce", without)
        for expected in ('find_spec("b12x") is None', "import vllm.models.deepseek_v4_1.common.engram",
                         "import vllm.models.deepseek_v4_1.nvidia.model", "import tilelang\n"):
            self.assertIn(expected, without)

    def test_only_the_b12x_stage_reads_the_b12x_context(self) -> None:
        stages = {}
        for block in (ROOT / "docker/Dockerfile").read_text().split("\nFROM ")[1:]:
            header = block.splitlines()[0]
            code = "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))
            stages[header.split(" AS ")[-1] if " AS " in header else "final"] = code
        self.assertEqual([name for name, code in stages.items() if "b12x-source" in code], ["runtime"])
        self.assertTrue(stages["runtime"].startswith("runtime-core AS runtime"))
        self.assertTrue(stages["runtime-tilelang"].startswith("${TILELANG_BASE_STAGE} AS runtime-tilelang"))
        self.assertNotIn("b12x", stages["runtime-core"].lower())


if __name__ == "__main__":
    unittest.main()
