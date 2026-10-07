"""The kernel backend policy: B12X by default, TileLang only with its sources."""

import argparse
import contextlib
import copy
import glob
import importlib.machinery
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark_kernel_backend_test", str(ROOT / "bin/spark"))
spec = importlib.util.spec_from_loader(loader.name, loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)
kernel_backend = spark.kernel_backend

CANDIDATE = "experiments/2026-10-03-tilelang-kernels"
# Experiments that run the TileLang family and may therefore build its sources.
TILELANG_EXPERIMENTS = (
    CANDIDATE,
    "experiments/2026-10-03-packed-bf16-head/tilelang-tp4",
    "experiments/2026-10-03-tilelang-tp4-performance",
    "experiments/2026-10-04-tilelang-1m",
    "experiments/2026-10-04-tilelang-gate-router",
    "experiments/2026-10-04-tilelang-mhc",
    "experiments/2026-10-04-tilelang-router",
    "experiments/2026-10-04-tilelang-sparknet",
    "experiments/2026-10-04-tilelang-vocab-heads",
    "experiments/2026-10-05-tilelang-r6",
    "experiments/2026-10-07-dspark-verification",
)
FLAGS = ("--attention-backend", "--linear-backend", "--moe-backend")
# Each TileLang profile and the B12X configuration it mirrors, with that
# configuration's lock and an example node map of its topology. The TP3 profile
# mirrors r5o, the configuration promoted when it was made.
R5O = "experiments/2026-10-05-r5p-promotion/r5o"
# r5p's B12X TP3 configuration and lock, kept when r6 promoted the TileLang family.
R5P_TP3 = "experiments/2026-10-05-tilelang-r6/r5p/cluster-64k.json"
R5P_LOCK = "experiments/2026-10-05-tilelang-r6/r5p/upstreams.lock.json"
TWINS = {
    "tp3": (f"{R5O}/cluster.json", f"{R5O}/upstreams.lock.json", "config/nodes.example.json"),
    "tp4": ("experiments/2026-10-03-collective-contract/candidate.json",
            "experiments/2026-10-03-collective-contract/upstreams.lock.json",
            "config/examples/nodes-ring4.json"),
}
# Per-boot artifacts a TileLang arm keeps apart from its twin's: pinned DSpark
# cost curves are keyed by shapes only (so each kernel image pins its own), and
# profiler traces by directory.
ARTIFACT_DIRS = {"tp4": ("ring4-collective-20261003", "ring4-tilelang-20261003")}
COST_DIRS = {"tp4": ("ring4-collective-nativeheads-20261003", "ring4-tilelang-v5-nativeheads-20261003")}


def speculative(cluster: dict) -> dict:
    return json.loads(spark.topology.argument(cluster, "--speculative-config"))


class KernelBackendTest(unittest.TestCase):
    def setUp(self):
        self.nodes = spark.read_json("config/nodes.example.json")
        head = self.nodes["nodes"][0]["management_ip"]
        self.base = spark.read_json(R5P_TP3)
        self.base["distributed"]["master_addr"] = head
        self.lock = spark.read_json(R5P_LOCK)
        self.candidate = spark.read_json(f"{CANDIDATE}/tp3/cluster.json")
        self.candidate["distributed"]["master_addr"] = head
        self.candidate_lock = spark.read_json(f"{CANDIDATE}/tp3/upstreams.lock.json")

    def twin(self, name):
        """(TileLang profile, its lock, the B12X configuration it mirrors, that lock, node map)."""
        config, lock, node_file = TWINS[name]
        nodes = spark.read_json(node_file)
        head = next(node for node in nodes["nodes"] if node.get("head"))["management_ip"]
        candidate = spark.read_json(f"{CANDIDATE}/{name}/cluster.json")
        base = spark.read_json(config)
        for cluster in (candidate, base):
            cluster["distributed"]["master_addr"] = head
        return (candidate, spark.read_json(f"{CANDIDATE}/{name}/upstreams.lock.json"),
                base, spark.read_json(lock), nodes)

    def ready(self):
        """The candidate, with a copy of its source manifest that the test may edit."""
        lock = copy.deepcopy(self.candidate_lock)
        manifest = copy.deepcopy(spark.read_json(lock["source_manifest"]))
        cluster = copy.deepcopy(self.candidate)
        read = spark.read_json
        patch = mock.patch.object(
            spark, "read_json", lambda path: manifest if path == lock["source_manifest"] else read(path)
        )
        return cluster, lock, manifest, patch

    def errors(self, cluster, lock):
        return [p for p in spark.local_doctor(cluster, self.nodes, lock) if not isinstance(p, spark.Warn)]

    def test_promoted_profiles_run_tilelang(self):
        lock = spark.read_json("upstreams.lock.json")
        for name in ("config/cluster.json", "config/cluster-64k.json", "config/cluster-4k.json"):
            with self.subTest(name=name):
                cluster = spark.read_json(name)
                cluster["distributed"]["master_addr"] = self.base["distributed"]["master_addr"]
                self.assertEqual(cluster["kernel_backend"], "tilelang")
                self.assertEqual(cluster["environment"][kernel_backend.ENVIRONMENT], "tilelang")
                self.assertEqual(kernel_backend.problems(cluster), [])
                self.assertEqual(self.errors(cluster, lock), [])

    def test_promoted_tp4_profile_runs_tilelang(self):
        # config/cluster-tp4.json reads the site's git-ignored ring map; check it
        # against the example map of the same topology.
        lock = spark.read_json("upstreams.lock.json")
        nodes = spark.read_json("config/examples/nodes-ring4.json")
        cluster = spark.read_json("config/cluster-tp4.json")
        self.assertEqual(cluster["nodes_config"], "config/nodes-ring4.local.json")
        self.assertNotIn("upstreams_config", cluster)
        self.assertEqual(cluster["promoted_baseline"], spark.read_json("config/cluster.json")["promoted_baseline"])
        cluster["distributed"]["master_addr"] = next(n for n in nodes["nodes"] if n.get("head"))["management_ip"]
        self.assertEqual(cluster["kernel_backend"], "tilelang")
        self.assertEqual(spark.topology.argument(cluster, "--tensor-parallel-size"), "4")
        self.assertEqual(spark.topology.argument(cluster, "--max-model-len"), "1048576")
        self.assertEqual(spark.topology.transport(cluster), "oneshot-ring4")
        self.assertNotIn("--profiler-config", cluster["serve_args"])
        errors = [p for p in spark.local_doctor(cluster, nodes, lock) if not isinstance(p, spark.Warn)]
        self.assertEqual(errors, [])

    def test_absent_backend_means_b12x(self):
        # Every configuration recorded before r6 omits kernel_backend.
        self.assertNotIn("kernel_backend", self.base)
        self.assertEqual(kernel_backend.backend(self.base), "b12x")
        self.assertNotIn(kernel_backend.ENVIRONMENT, self.base["environment"])
        self.assertEqual(kernel_backend.problems(self.base), [])
        self.assertEqual(self.errors(self.base, self.lock), [])
        # Naming the default explicitly changes nothing that is launched.
        explicit = copy.deepcopy(self.base)
        explicit["kernel_backend"] = "b12x"
        self.assertEqual(kernel_backend.problems(explicit), [])
        for node in self.nodes["nodes"]:
            command = spark.rendered_docker_command(self.base, node)
            self.assertEqual(spark.rendered_docker_command(explicit, node), command)
            self.assertFalse(any(kernel_backend.ENVIRONMENT in part for part in command))

    def test_b12x_profile_requires_no_tilelang_sources(self):
        self.assertEqual(kernel_backend.source_problems(self.base, self.lock, {}), [])
        inputs = spark.build_inputs(self.lock)
        for name in kernel_backend.SOURCES:
            self.assertNotIn(name, inputs)
        self.assertEqual(spark.build_contexts(inputs), spark.BUILD_CONTEXTS)
        self.assertEqual(spark.smoke_script(self.lock), spark.SMOKE_SCRIPT)

    def test_tilelang_candidate_passes_doctor(self):
        cluster, lock, _, patch = self.ready()
        with patch:
            self.assertEqual(self.errors(cluster, lock), [])
        for name in TWINS:
            with self.subTest(topology=name):
                candidate, candidate_lock, base, base_lock, nodes = self.twin(name)
                for config, upstreams in ((candidate, candidate_lock), (base, base_lock)):
                    errors = [p for p in spark.local_doctor(config, nodes, upstreams)
                              if not isinstance(p, spark.Warn)]
                    self.assertEqual(errors, [])
                self.assertEqual(kernel_backend.backend(candidate), "tilelang")
                self.assertFalse(candidate["deployment"]["launch_enabled"])
        self.assertEqual(kernel_backend.backend(cluster), "tilelang")
        self.assertFalse(cluster["deployment"]["launch_enabled"])
        for flag in FLAGS:
            self.assertEqual(spark.topology.argument(cluster, flag),
                             kernel_backend.BACKENDS["tilelang"][flag])
        self.assertEqual(speculative(cluster)["attention_backend"], "TILELANG")
        for node in self.nodes["nodes"]:
            env = spark.expected_environment(cluster, node)
            self.assertEqual(env[kernel_backend.ENVIRONMENT], "tilelang")
            self.assertEqual(env["VLLM_PLUGINS"], "b12x_loader")
            self.assertEqual(spark.topology.argument(cluster, "--load-format"), "b12x")

    def test_candidates_match_their_b12x_configuration_except_the_policy(self):
        for name in TWINS:
            with self.subTest(topology=name):
                candidate, candidate_lock, base, base_lock, _ = self.twin(name)
                labels = candidate["container"]["expected_labels"]
                for source in ("b12x", "nccl"):
                    self.assertEqual(labels[kernel_backend.label(source)],
                                     base["container"]["expected_labels"][kernel_backend.label(source)])
                    self.assertEqual(candidate_lock["sources"][source], base_lock["sources"][source])
                self.assertEqual(spark.read_json(candidate_lock["source_manifest"])["vllm"]["base_revision"],
                                 spark.read_json(base_lock["source_manifest"])["vllm"]["base_revision"])
                # The TileLang series is the twin's series plus the TileLang patch.
                series = [Path(candidate_lock["sources"]["vllm"]["patch_series"]).parent / patch
                          for patch in spark.read_series(ROOT / candidate_lock["sources"]["vllm"]["patch_series"])]
                twin = [Path(base_lock["sources"]["vllm"]["patch_series"]).parent / patch
                        for patch in spark.read_series(ROOT / base_lock["sources"]["vllm"]["patch_series"])]
                resolved = [(ROOT / path).resolve() for path in series]
                self.assertEqual(resolved[:-1], [(ROOT / path).resolve() for path in twin])
                self.assertEqual(resolved[-1].name, "0027-deepseek-v41-tilelang-kernels.patch")
                before, after = ARTIFACT_DIRS.get(name, (None, None))
                for cluster in (candidate, base):
                    for flag in FLAGS:
                        spark.topology.set_argument(cluster, flag, "-")
                    config = speculative(cluster)
                    config["attention_backend"] = "-"
                    spark.topology.set_argument(cluster, "--speculative-config", json.dumps(config))
                    for key in ("kernel_backend", "upstreams_config"):
                        cluster.pop(key, None)
                    for key in ("image", "expected_labels"):
                        cluster["container"].pop(key)
                    for key in ("launch_enabled", "branch"):
                        cluster["deployment"].pop(key, None)
                    cluster["environment"].pop(kernel_backend.ENVIRONMENT, None)
                    if before:
                        twin_cost, own_cost = COST_DIRS[name]
                        cluster["environment"]["SPARK3_DSPARK_COST_DIR"] = (
                            cluster["environment"]["SPARK3_DSPARK_COST_DIR"].replace(own_cost, twin_cost))
                        profiler = spark.topology.argument(cluster, "--profiler-config")
                        spark.topology.set_argument(cluster, "--profiler-config", profiler.replace(after, before))
                self.assertEqual(candidate, base)

    def test_candidate_build_inputs_record_every_source(self):
        images = set()
        for name in TWINS:
            with self.subTest(topology=name):
                candidate, lock, base, _, _ = self.twin(name)
                inputs = spark.build_inputs(lock)
                manifest = spark.read_json(lock["source_manifest"])
                self.assertEqual(inputs["vllm"]["expected_tree"], manifest["vllm"]["expected_tree"])
                self.assertEqual(candidate["container"]["expected_labels"][kernel_backend.label("vllm")],
                                 manifest["vllm"]["expected_tree"])
                self.assertEqual(manifest["vllm"]["patchset_sha256"], spark.patchset_sha256(lock["sources"]["vllm"]))
                self.assertEqual(inputs["tilelang"]["expected_tree"], "1697ad52fab8379ce78f9abc547c7b1e02cd9c5e")
                self.assertIn("3rdparty/tvm", inputs["tilelang"]["submodules"])
                self.assertEqual(inputs["tile_kernels"]["patch_head"], inputs["tile_kernels"]["revision"])
                self.assertNotEqual(candidate["container"]["image"], base["container"]["image"])
                images.add(candidate["container"]["image"])
        self.assertEqual(len(images), len(TWINS))

    def test_tuning_catalog_mirrors_the_b12x_profiles(self):
        # The transport settings match the promoted catalog; the model layout
        # matches the B12X configuration each profile mirrors.
        profiles = spark.transport_profiles
        b12x = spark.read_json(profiles.DEFAULT_PROFILES)["profiles"]
        catalog = f"{CANDIDATE}/profiles.json"
        self.assertEqual(set(spark.read_json(catalog)["profiles"]), set(b12x))
        for name in b12x:
            with self.subTest(topology=name):
                profile, resolved = profiles.resolve(ROOT, catalog, name)
                self.assertEqual(profile["base_config"], f"{CANDIDATE}/{name}/cluster.json")
                self.assertEqual(resolved, spark.read_json(profile["base_config"]))
                for key in ("node_count", "transport", "rocenante", "nccl"):
                    self.assertEqual(profile[key], b12x[name][key])
                layout = spark.model_layout.describe(ROOT, resolved, profiles.size_bytes(
                    resolved["environment"]["VLLM_ROCE_ALLREDUCE_MAX_SIZE"]))
                twin = spark.read_json(TWINS[name][0])
                twin_layout = spark.model_layout.describe(ROOT, twin, profiles.size_bytes(
                    twin["environment"]["VLLM_ROCE_ALLREDUCE_MAX_SIZE"]))
                for key in ("model_dimensions", "drafter", "scheduled_rows", "transport_layout"):
                    self.assertEqual(layout[key], twin_layout[key])
                self.assertFalse(layout["kernel_scratch"]["compact_moe_n64_path"])
                self.assertIsNone(layout["kernel_scratch"]["compact_moe_intermediate_width_when_selected"])

    def test_missing_patch_is_reported_without_crashing(self):
        series = Path(self.candidate_lock["sources"]["vllm"]["patch_series"])
        with tempfile.NamedTemporaryFile("w", dir=ROOT / series.parent, suffix=".series") as handle:
            handle.write((ROOT / series).read_text() + "0099-not-written-yet.patch\n")
            handle.flush()
            lock = copy.deepcopy(self.candidate_lock)
            lock["sources"]["vllm"]["patch_series"] = (series.parent / Path(handle.name).name).as_posix()
            errors = self.errors(self.candidate, lock)
            self.assertTrue(any("missing patch 0099-not-written-yet.patch" in e for e in errors))
            with self.assertRaises(SystemExit) as raised:
                spark.build_inputs(lock)
            self.assertIn("missing patch", str(raised.exception))
            inputs = spark.build_inputs(lock, ["tilelang", "tile_kernels"])
            self.assertIn("missing_patches", inputs["vllm"])

    def test_missing_capability_fails(self):
        cluster, lock, manifest, patch = self.ready()
        manifest["vllm"]["capabilities"] = []
        with patch:
            self.assertTrue(any("capability tilelang-kernels" in e for e in self.errors(cluster, lock)))

    def test_missing_or_mismatched_labels_fail(self):
        for name in ("vllm", *kernel_backend.SOURCES):
            label = kernel_backend.label(name)
            for value in (None, "0" * 40):
                with self.subTest(label=label, value=value):
                    cluster, lock, _, patch = self.ready()
                    labels = cluster["container"]["expected_labels"]
                    if value is None:
                        labels.pop(label)
                    else:
                        labels[label] = value
                    with patch:
                        self.assertTrue(any(label in e for e in self.errors(cluster, lock)))

    def test_lock_without_tilelang_sources_fails(self):
        cluster, _, _, _ = self.ready()
        errors = kernel_backend.source_problems(cluster, self.lock, spark.read_json(self.lock["source_manifest"]))
        for name in kernel_backend.SOURCES:
            self.assertTrue(any(f"source {name}" in e for e in errors))
        lock = copy.deepcopy(self.candidate_lock)
        lock["sources"].pop("tile_kernels")
        self.assertTrue(any("built together" in e for e in spark.local_doctor(self.base, self.nodes, lock)))

    def test_serve_arguments_must_match_the_backend(self):
        for flag in FLAGS:
            with self.subTest(flag=flag):
                cluster = copy.deepcopy(self.candidate)
                spark.topology.set_argument(cluster, flag, kernel_backend.BACKENDS["b12x"][flag])
                self.assertTrue(any(flag in e for e in kernel_backend.problems(cluster)))
                base = copy.deepcopy(self.base)
                spark.topology.set_argument(base, flag, kernel_backend.BACKENDS["tilelang"][flag])
                self.assertTrue(any(flag in e for e in kernel_backend.problems(base)))

    def test_speculative_attention_backend_must_match(self):
        for cluster, wrong in ((copy.deepcopy(self.candidate), "B12X"), (copy.deepcopy(self.base), "TILELANG")):
            config = speculative(cluster)
            config["attention_backend"] = wrong
            spark.topology.set_argument(cluster, "--speculative-config", json.dumps(config))
            self.assertTrue(any("speculative attention_backend" in e for e in kernel_backend.problems(cluster)))

    def test_environment_must_match_the_backend(self):
        for cluster, wrong in ((self.candidate, "b12x"), (self.candidate, None), (self.base, "tilelang")):
            with self.subTest(backend=kernel_backend.backend(cluster), value=wrong):
                changed = copy.deepcopy(cluster)
                if wrong is None:
                    changed["environment"].pop(kernel_backend.ENVIRONMENT, None)
                else:
                    changed["environment"][kernel_backend.ENVIRONMENT] = wrong
                self.assertTrue(any(kernel_backend.ENVIRONMENT in e for e in kernel_backend.problems(changed)))
        for value in (None, "/tmp/tilelang", "/cache", "cache/tilelang"):
            with self.subTest(cache=value):
                changed = copy.deepcopy(self.candidate)
                if value is None:
                    changed["environment"].pop(kernel_backend.CACHE_ENVIRONMENT)
                else:
                    changed["environment"][kernel_backend.CACHE_ENVIRONMENT] = value
                self.assertTrue(any(kernel_backend.CACHE_ENVIRONMENT in e for e in kernel_backend.problems(changed)))
        unknown = copy.deepcopy(self.base)
        unknown["kernel_backend"] = "triton"
        self.assertTrue(any("unknown kernel_backend" in e for e in kernel_backend.problems(unknown)))

    def test_build_inputs_include_tilelang_only_when_listed(self):
        _, lock, _, patch = self.ready()
        with patch:
            inputs = spark.build_inputs(lock)
        self.assertEqual(spark.build_projects(inputs), (*spark.BUILD_PROJECTS, *kernel_backend.SOURCES))
        self.assertEqual(inputs["tilelang"]["version"], "0.1.15")
        self.assertEqual(inputs["tile_kernels"]["version"], "2.0.0")
        self.assertEqual(spark.build_contexts(inputs)[-2:], ("tilelang-source", "tile_kernels-source"))
        self.assertNotEqual(spark.build_directory(inputs), spark.build_directory(spark.build_inputs(self.lock)))

    def test_image_plan_selects_the_tilelang_stage_only_for_its_sources(self):
        cluster, lock, _, patch = self.ready()

        def plan(selected, upstreams):
            output = io.StringIO()
            with mock.patch.object(spark, "configuration", return_value=(selected, self.nodes, upstreams)), \
                    mock.patch.object(spark, "build_problems", return_value=[]), \
                    mock.patch.object(spark, "repository_dirty", return_value=""), \
                    mock.patch.object(spark, "repository_revision", return_value="0" * 40), \
                    contextlib.redirect_stdout(output):
                spark.command_build_image(argparse.Namespace(cluster_config="c.json", tag=None, apply=False))
            return output.getvalue()

        with patch:
            tilelang = plan(cluster, lock)
        for expected in ("RUNTIME_STAGE=runtime-tilelang", "TILELANG_TREE=1697ad52fab8379ce78f9abc547c7b1e02cd9c5e",
                         "TILELANG_VERSION=0.1.15", "TILE_KERNELS_VERSION=2.0.0", "TILELANG_COMPILE_JOBS=",
                         "--build-context tilelang-source=", "--build-context tile_kernels-source="):
            self.assertIn(expected, tilelang)
        b12x = plan(self.base, self.lock)
        for absent in ("RUNTIME_STAGE", "TILELANG", "TILE_KERNELS", "tilelang-source", "tile_kernels-source"):
            self.assertNotIn(absent, b12x)

    def test_sparknet_builds_on_the_tilelang_runtime(self):
        experiment = "experiments/2026-10-04-tilelang-1m"
        lock = spark.read_json(f"{experiment}/upstreams.lock.json")
        cluster = spark.read_json(f"{experiment}/candidate.json")
        cluster["distributed"]["master_addr"] = self.nodes["nodes"][0]["management_ip"]
        inputs = spark.build_inputs(lock)
        self.assertEqual(spark.build_projects(inputs)[-1], "sparknet")
        self.assertEqual(inputs["sparknet"]["version"], "0.2.0")
        self.assertEqual(spark.build_contexts(inputs)[-1], "sparknet-source")
        self.assertEqual(spark.runtime_stage(inputs), "runtime-tilelang-sparknet")
        without_tilelang = {name: value for name, value in inputs.items() if name not in kernel_backend.SOURCES}
        with self.assertRaises(SystemExit):
            spark.runtime_stage(without_tilelang)
        output = io.StringIO()
        with mock.patch.object(spark, "configuration", return_value=(cluster, self.nodes, lock)), \
                mock.patch.object(spark, "build_problems", return_value=[]), \
                mock.patch.object(spark, "repository_dirty", return_value=""), \
                mock.patch.object(spark, "repository_revision", return_value="0" * 40), \
                contextlib.redirect_stdout(output):
            spark.command_build_image(argparse.Namespace(cluster_config="c.json", tag=None, apply=False))
        for expected in ("RUNTIME_STAGE=runtime-tilelang-sparknet", "SPARKNET_VERSION=0.2.0",
                         "SPARKNET_TREE=bd4191b3241383e8ecb705ad291f2da5b064370f",
                         "--build-context sparknet-source=", "--build-context tilelang-source="):
            self.assertIn(expected, output.getvalue())
        script = spark.smoke_script(lock)
        compile(script, "smoke", "exec")
        for expected in ("from sparknet.integration.vllm import SparknetOneShotAllReduce",
                         "version(\"dgx-spark-networking\") == '0.2.0'", "roce_abi_version()"):
            self.assertIn(expected, script)

    def test_smoke_imports_tilelang_and_the_capability_module(self):
        _, lock, manifest, patch = self.ready()
        with patch:
            script = spark.smoke_script(lock)
            compile(script, "smoke", "exec")
            self.assertTrue(script.startswith(spark.SMOKE_SCRIPT))
            for expected in ("import tilelang\n", "import tile_kernels", "== '0.1.15'",
                             "import vllm.models.deepseek_v4_1.tilelang"):
                self.assertIn(expected, script)
            manifest["vllm"]["capabilities"] = []
            self.assertNotIn("deepseek_v4_1.tilelang", spark.smoke_script(lock))

    def test_existing_profiles_and_locks_are_unaffected(self):
        for path in sorted(glob.glob(str(ROOT / "experiments/**/*.json"), recursive=True)):
            relative = Path(path).relative_to(ROOT).as_posix()
            if relative.startswith(TILELANG_EXPERIMENTS):
                continue
            data = json.loads(Path(path).read_text())
            if not isinstance(data, dict):
                continue
            if "sources" in data and "source_manifest" in data:
                with self.subTest(lock=relative):
                    self.assertEqual(spark.build_projects(data["sources"]), spark.BUILD_PROJECTS)
                    inputs = spark.build_inputs(data)
                    self.assertEqual(spark.build_contexts(inputs), spark.BUILD_CONTEXTS)
            elif "serve_args" in data and "container" in data and "kernel_backend" not in data:
                with self.subTest(profile=relative):
                    self.assertEqual(kernel_backend.source_problems(data, {"sources": {}}, {}), [])
                    if spark.topology.argument(data, "--attention-backend") == "B12X":
                        self.assertEqual(kernel_backend.problems(data), [])


if __name__ == "__main__":
    unittest.main()
