"""Host-independent tests for the build directory identity."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import spark_ds41f as spark


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

    def plan(self, lock: dict, max_compile_jobs=None) -> str:
        import argparse
        import contextlib
        import io
        from unittest import mock

        output = io.StringIO()
        cluster = spark.read_json("config/cluster.json")
        if "b12x" not in lock.get("sources", {}):
            # A profile of a lock without B12X expects no B12X tree label.
            cluster["container"]["expected_labels"].pop("local.spark-ds41f.b12x.tree", None)
        with mock.patch.object(spark, "configuration", return_value=(cluster, None, lock)), \
                mock.patch.object(spark, "build_problems", return_value=[]), \
                mock.patch.object(spark, "repository_dirty", return_value=""), \
                mock.patch.object(spark, "repository_revision", return_value="0" * 40), \
                contextlib.redirect_stdout(output):
            spark.command_build_image(argparse.Namespace(cluster_config="c.json", tag=None, apply=False,
                                                         max_compile_jobs=max_compile_jobs))
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

    def test_opt_in_job_cap_covers_nccl_without_changing_default(self) -> None:
        original, capped = self.plan(self.lock), self.plan(self.lock, 2)
        self.assertNotIn("NCCL_COMPILE_JOBS=", original)
        self.assertIn("NCCL_COMPILE_JOBS=2", capped)
        self.assertIn("VLLM_COMPILE_JOBS=", capped)
        self.assertIn("TILELANG_COMPILE_JOBS=", capped)
        self.assertIn("at most 2; sized from MemAvailable", capped)


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
        self.assertIn("VLLM_NATIVE_CACHE_DIR=/opt/spark-ds41f/native", stages["runtime-core"])
        self.assertIn("from vllm.model_executor.model_loader.direct_io import _native as direct_native", stages["runtime-core"])
        self.assertIn("from vllm.models.deepseek_v4_1.engram_rows import _native as engram_native", stages["runtime-core"])

    def test_build_commands_refuse_misaligned_arm_notes_and_labels(self) -> None:
        import argparse
        import contextlib
        import io
        from unittest import mock

        lock = spark.read_json("experiments/2026-10-26-spark-ds41f-candidate/candidate-33/upstreams.lock.json")
        cluster = spark.read_cluster("experiments/2026-10-26-spark-ds41f-candidate/candidate-33/tp4.json")
        cluster["notes"] = "candidate-33 at tp4-ring: vLLM 0568..0636 (patch head 2c2d396f, tree 88468fc7)."
        cluster["container"]["expected_labels"]["local.spark-ds41f.vllm.tree"] = "041f8e3c" + "0" * 32

        for fn, args in (
            (spark.command_build_prepare, argparse.Namespace(cluster_config="c.json", only=None, lock=False)),
            (spark.command_build_check, argparse.Namespace(cluster_config="c.json")),
            (spark.command_build_image, argparse.Namespace(cluster_config="c.json", tag=None, apply=False)),
        ):
            output = io.StringIO()
            with mock.patch.object(spark, "configuration", return_value=(cluster, None, lock)), \
                    mock.patch.object(spark, "build_problems", return_value=[]), \
                    contextlib.redirect_stdout(output):
                code = fn(args)
            self.assertEqual(code, 1, fn.__name__)
            text = output.getvalue()
            self.assertIn("local.spark-ds41f.vllm.tree", text)
            self.assertIn("arm notes cite vllm tree 88468fc7, expected 354fd6e6", text)
            self.assertIn("arm notes cite vLLM 0568..0636", text)


class BuildResourcePolicyTest(unittest.TestCase):
    def test_cap_preserves_cpu_sizing_and_memory_floor(self) -> None:
        with mock.patch.object(spark.os, "sched_getaffinity", return_value=set(range(24))):
            self.assertEqual(spark.compile_jobs(64), 5)
            self.assertEqual(spark.compile_jobs(64, 2), 2)
            self.assertEqual(spark.compile_jobs(64, 20), 5)
            with self.assertRaises(SystemExit):
                spark.compile_jobs(63, 1)
        with mock.patch.object(spark.os, "sched_getaffinity", return_value=set(range(6))):
            self.assertEqual(spark.compile_jobs(96, 2), 1)

    def test_invalid_caps_fail_before_build_configuration_or_dispatch(self) -> None:
        import argparse
        for cap in (0, -1, True, "2", 1.5):
            with self.subTest(cap=cap), mock.patch.object(spark, "configuration") as configure:
                with self.assertRaises(SystemExit):
                    spark.command_build_image(argparse.Namespace(max_compile_jobs=cap))
                configure.assert_not_called()

    def test_cli_requires_a_positive_integer_and_keeps_default_optional(self) -> None:
        import contextlib
        import io
        self.assertIsNone(spark.parser().parse_args(["build", "image"]).max_compile_jobs)
        self.assertEqual(spark.parser().parse_args(["build", "image", "--max-compile-jobs", "2"]).max_compile_jobs, 2)
        for value in ("0", "-1", "1.5", "bad"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                spark.parser().parse_args(["build", "image", "--max-compile-jobs", value])

    def test_actual_build_argv_receipt_and_smoke_bind_effective_cap(self) -> None:
        import argparse
        import contextlib
        import io
        lock = spark.read_json("upstreams.lock.json")
        inputs = spark.build_inputs(lock)
        cluster = spark.read_json("config/cluster.json")
        labels = {f"local.spark-ds41f.{name}.tree": inputs[name]["expected_tree"]
                  for name in spark.build_projects(inputs)}
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            directory = root / "build"
            directory.mkdir()
            stack.enter_context(mock.patch.object(spark, "ROOT", root))
            lock_file = stack.enter_context((root / "build.lock").open("w"))
            lock_path = mock.Mock(spec=Path)
            lock_path.open.return_value = lock_file
            stack.enter_context(mock.patch.object(spark, "BUILD_LOCK", lock_path))
            returns = {"configuration": (cluster, None, lock), "build_inputs": inputs,
                       "build_directory": directory, "read_json": {}, "arm_source_problems": [],
                       "build_problems": [], "repository_dirty": "", "repository_revision": "b" * 40,
                       "require_idle_build_host": None, "available_memory_gib": 96, "smoke_script": "pass"}
            for name, value in returns.items():
                stack.enter_context(mock.patch.object(spark, name, return_value=value))
            stack.enter_context(mock.patch.object(spark.os, "sched_getaffinity", return_value=set(range(24))))
            stack.enter_context(mock.patch.object(spark, "image_labels", side_effect=[None, {"id": "sha256:" + "a" * 64, "labels": labels}]))
            build = stack.enter_context(mock.patch.object(spark, "run_memory_guarded", return_value=90))
            smoke = stack.enter_context(mock.patch.object(spark, "smoke_image"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(spark.command_build_image(argparse.Namespace(cluster_config="named.json", tag="example:capped", apply=True, max_compile_jobs=2)), 0)
            argv = build.call_args.args[0]
            for value in ("VLLM_COMPILE_JOBS=2", "TILELANG_COMPILE_JOBS=2", "NCCL_COMPILE_JOBS=2"):
                self.assertIn(value, argv)
            self.assertEqual(build.call_args.args[1], spark.BUILD_RESERVE_GIB)
            self.assertEqual(smoke.call_args.kwargs["max_cpus"], 2)
            receipt = json.loads(next((directory / "images").glob("*.json")).read_text())
            self.assertEqual({k: receipt[k] for k in ("compile_jobs", "max_compile_jobs", "nccl_compile_jobs", "smoke_cpu_limit")},
                             {"compile_jobs": 2, "max_compile_jobs": 2, "nccl_compile_jobs": 2, "smoke_cpu_limit": 2})



class BuildEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root_patch = mock.patch.object(spark, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.relative = "experiments/candidate-99"
        self.directory = self.root / self.relative
        self.directory.mkdir(parents=True)
        self.lock_path = f"{self.relative}/upstreams.lock.json"
        self.old = "1" * 40
        self.new = "2" * 40
        self.write("source.json", {"vllm": {
            "base_revision": self.new, "patch_head": self.new,
            "expected_tree": self.new, "patchset_sha256": "old fingerprint",
        }})
        self.write("upstreams.lock.json", {
            "source_manifest": f"{self.relative}/source.json",
            "sources": {
                "vllm": {"revision": self.new, "patch_head": self.old,
                         "expected_tree": self.old, "patchset_sha256": "old fingerprint"},
                "nccl": {"revision": self.old},
            },
        })

    def write(self, name: str, value: dict) -> Path:
        path = self.directory / name
        path.write_text(json.dumps(value, indent=1) + "\n")
        return path

    def test_refresh_keeps_result_identity_and_verdict_bytes(self) -> None:
        results = self.write("results.json", {
            "patch_head": self.old, "expected_tree": self.old,
            "patchset_sha256": "measured fingerprint", "total_patches": 10,
            "image_digest": "sha256:measured-image",
            "kernel_local_verdict": {"passed": True, "seconds": 191.06},
        })
        verdict = self.write("verdict.json", {"passed": True, "image": "sha256:measured-image"})
        before = {path: path.read_bytes() for path in (results, verdict)}

        updated = spark.sync_lock_and_arms(self.lock_path)

        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)
            self.assertNotIn(path.relative_to(self.root).as_posix(), updated)
        # The new inputs still advance, so the old receipt remains visibly different.
        lock = json.loads((self.directory / "upstreams.lock.json").read_text())
        self.assertEqual(lock["sources"]["vllm"]["expected_tree"], self.new)
        self.assertNotEqual(json.loads(results.read_text())["expected_tree"],
                            lock["sources"]["vllm"]["expected_tree"])

    def test_sibling_receipt_notes_are_not_treated_as_arm_configuration(self) -> None:
        notes = f"Measured vLLM patch head {self.old[:8]}, tree {self.old[:8]}."
        receipt = self.write("results.json", {
            "notes": notes, "container": {"expected_labels": {
                "local.spark-ds41f.vllm.tree": self.old,
            }}, "serve_args": [], "passed": True,
        })
        verdict = self.write("verdict.json", {"notes": notes, "passed": True})
        measurements = self.write("measurements.json", {"notes": notes, "seconds": 191.06})
        before = {path: path.read_bytes() for path in (receipt, verdict, measurements)}
        arm = self.write("tp3.json", {
            "notes": notes, "upstreams_config": self.lock_path, "serve_args": [],
            "container": {"expected_labels": {"local.spark-ds41f.vllm.tree": self.old}},
        })
        overlay = self.write("tp3-dev.json", {
            "notes": notes, "extends": {"config": f"{self.relative}/tp3.json"},
        })

        updated = spark.sync_lock_and_arms(self.lock_path)

        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)
            self.assertNotIn(path.relative_to(self.root).as_posix(), updated)
        for path in (arm, overlay):
            self.assertIn(path.relative_to(self.root).as_posix(), updated)
            self.assertIn(self.new[:8], json.loads(path.read_text())["notes"])
        self.assertEqual(json.loads(arm.read_text())["container"]["expected_labels"]
                         ["local.spark-ds41f.vllm.tree"], self.new)

    def test_overlay_refresh_updates_nested_labels_and_respects_its_lock(self) -> None:
        notes = f"vLLM patch head {self.old[:8]}, tree {self.old[:8]}."
        overlay = {
            "notes": notes, "extends": {"config": "releases/previous/cluster.json"},
            "set": {"upstreams_config": self.lock_path,
                    "container": {"expected_labels": {"local.spark-ds41f.vllm.tree": self.old}}},
        }
        current = self.write("tp3.json", overlay)
        other = copy.deepcopy(overlay)
        other["set"]["upstreams_config"] = "experiments/another/upstreams.lock.json"
        unrelated = self.write("tp3-other.json", other)
        before = unrelated.read_bytes()

        updated = spark.sync_lock_and_arms(
            self.lock_path, arm_paths=[unrelated.relative_to(self.root).as_posix()])

        refreshed = json.loads(current.read_text())
        self.assertIn(current.relative_to(self.root).as_posix(), updated)
        self.assertEqual(refreshed["set"]["container"]["expected_labels"]
                         ["local.spark-ds41f.vllm.tree"], self.new)
        self.assertIn(self.new[:8], refreshed["notes"])
        self.assertEqual(unrelated.read_bytes(), before)
        self.assertNotIn(unrelated.relative_to(self.root).as_posix(), updated)


class PackageStructureTest(unittest.TestCase):
    def test_spark_ds41f_submodules_and_thin_entrypoints(self) -> None:
        from spark_ds41f import bench, build, cli, cluster, config, doctor, lab

        self.assertTrue(callable(cli.main))
        self.assertTrue(callable(config.read_cluster))
        self.assertTrue(callable(doctor.local_doctor))
        self.assertTrue(callable(build.build_inputs))
        self.assertTrue(callable(cluster.deployment_order))
        self.assertTrue(callable(bench.command_bench))
        self.assertTrue(callable(lab.kernel_local))
        self.assertLessEqual(len((ROOT / "bin" / "spark").read_text().splitlines()), 15)
        self.assertLessEqual(len((ROOT / "scripts" / "lab.py").read_text().splitlines()), 15)
        self.assertTrue((ROOT / "pyproject.toml").is_file())


if __name__ == "__main__":
    unittest.main()
