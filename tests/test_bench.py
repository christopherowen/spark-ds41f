"""Host-independent tests for bench's conditions, identity and reference rules (Sonar measures;
its own tests cover the measurements)."""

from __future__ import annotations

import argparse
import contextlib
import http.server
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import spark_ds41f as spark


class ThermalTest(unittest.TestCase):
    def test_parse_thermal_sample(self) -> None:
        sample = spark.parse_thermal("62,34,2405,14.25,0,0,27131561741,0,64900")
        self.assertEqual(sample["gpu_c"], 62.0)
        self.assertEqual(sample["headroom_c"], 34.0)
        self.assertEqual(sample["thermal_slowdown_us"], 0.0)
        self.assertEqual(sample["power_cap_us"], 27131561741.0)
        self.assertAlmostEqual(sample["system_c"], 64.9)
        missing = spark.parse_thermal("[N/A],[N/A],2405,14.25")
        self.assertIsNone(missing["gpu_c"])
        self.assertIsNone(missing["system_c"])


class BenchAbortCleanupTest(unittest.TestCase):
    """An aborted live bench still stops its fan curve and saves its report."""

    def test_aborted_bench_saves_its_report_with_and_without_fan_control(self) -> None:
        for fan_control in (True, False):
            with self.subTest(fan_control=fan_control), tempfile.TemporaryDirectory() as tmp:
                cluster = spark.read_cluster("config/cluster.json", "tp4-ring")
                cluster["host"]["fan_control"] = fan_control
                nodes = spark.read_json("config/examples/nodes-ring4.json")
                options = spark.parser().parse_args(
                    ["--cluster-config", "config/cluster.json", "bench", "--compare", "none", "--output", tmp])
                identity = {"live_warnings": [], "live_problems": []}
                bench = mock.Mock(discarded=[])
                bench.guard.thermal_summary.return_value = {}
                bench.guard.close.return_value = {}
                matrix = spark.sonar.module("matrix")
                with mock.patch.object(spark, "configuration", return_value=(cluster, nodes, {})), \
                        mock.patch.object(spark, "bench_identity", return_value=identity), \
                        mock.patch.object(matrix, "Sampler", return_value=bench), \
                        mock.patch.object(spark, "cool_nodes", side_effect=RuntimeError("no zone")), \
                        mock.patch.object(spark, "run_ssh", side_effect=AssertionError("contacted a node")), \
                        contextlib.redirect_stdout(io.StringIO()):
                    status = options.func(options)
                self.assertEqual(status, 1)
                report = json.loads((Path(tmp) / "bench.json").read_text())
                self.assertEqual(report["aborted"], "cooling failed: no zone")
                self.assertEqual(report["workload"]["concurrency"], [1, 2, 4, 8, 16])
                # The deployment's reasoning switch reaches every reasoning-off case.
                self.assertEqual(report["workload"]["reasoning_off"], {"chat_template_kwargs": {"thinking": False}})
                self.assertEqual(report["workload"]["memory_floor_gib"], float(cluster["host"]["memguard_gib"]) + 1)


class ReferenceShapeTest(unittest.TestCase):
    TP3_REFERENCE = "manifests/benchmarks/2026-10-12-karmic-kraken-r6h.json"
    TP4_REFERENCE = "manifests/suites/2026-10-15-karmic-kraken-r6n-tp4"
    # tp4-ring was last released on r6n: its qualified recipe is frozen with r6s's promotion.
    TP4_RECIPE = "experiments/2026-10-16-r6s/r6n/cluster.json"
    TP4_BENCH = "manifests/benchmarks/2026-10-13-karmic-kraken-r6k-tp4.json"
    OLD_REFERENCE = "manifests/benchmarks/2026-10-02-karmic-kraken-r5o-64k.json"
    TP4_MISMATCH = "was measured with nodes 3, transport oneshot-direct; this {} has nodes 4, transport oneshot-ring4"
    # The profiles read the site's git-ignored node maps; use the example map
    # of the same size.
    EXAMPLE_NODES = {
        "1": "config/examples/nodes-tp1.json",
        "2": "config/examples/nodes-direct2.json",
        "3": "config/nodes.example.json",
        "4": "config/examples/nodes-ring4.json",
        "5": "config/examples/nodes-switched5.json",
        "6": "config/examples/nodes-switched6.json",
        "7": "config/examples/nodes-switched7.json",
        "8": "config/examples/nodes-switched8.json",
    }

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def profile(self, name: str, shape: str | None = None) -> tuple[dict, dict]:
        cluster = spark.read_cluster(name, shape)
        nodes_file = spark.NODE_MAP_EXAMPLES.get(
            cluster.get("nodes_config"),
            self.EXAMPLE_NODES[spark.serve_arg(cluster, "--tensor-parallel-size")],
        )
        return cluster, spark.read_json(nodes_file)

    def report(self, cluster: dict, nodes: dict) -> str:
        """A report with no suites, measured on the profile's shape."""
        path = Path(self.tmp.name) / "bench.json"
        identity = {
            "topology": nodes,
            "transport": spark.topology.transport(cluster),
            "kernel_backend": spark.kernel_backend.backend(cluster),
        }
        path.write_text(json.dumps({"identity": identity, "suites": {}}))
        return str(path)

    def bench(self, name: str, cluster: dict, nodes: dict, *argv: str) -> tuple[int, str]:
        options = spark.parser().parse_args(["--cluster-config", name, "bench", *argv])
        with mock.patch.object(spark, "configuration", return_value=(cluster, nodes, {})), \
                mock.patch.object(spark, "bench_identity", side_effect=AssertionError("contacted the cluster")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            status = options.func(options)
        return status, output.getvalue()

    def test_qualified_shapes_name_a_reference_of_their_shape(self) -> None:
        recipe = spark.read_json("config/cluster.json")
        for shape, entry in sorted(recipe["shapes"].items()):
            with self.subTest(shape=shape):
                cluster, nodes = self.profile("config/cluster.json", shape)
                if "benchmark_reference" not in entry:
                    # Not yet benched with the recipe's image: no reference, and no production start.
                    self.assertNotIn("benchmark_reference", cluster)
                    continue
                reference, named = spark.benchmark_reference(cluster)
                self.assertTrue(named and (reference.is_file() or spark.is_suite_run(reference)))
                relative = reference.relative_to(ROOT)
                self.assertIn(relative.parent.as_posix(), ("manifests/benchmarks", "manifests/suites"))
                # A reference is a baseline's release run of the shape: the promoted baseline's,
                # or an earlier one's when the shape's path did not change (r6v's TP3 path is r6u's).
                baseline = relative.stem.rsplit("-", 1)[0]
                self.assertTrue((ROOT / "manifests" / "baselines" / f"{baseline}.json").is_file(), baseline)
                identity = spark.reference_identity(reference)
                self.assertEqual(
                    spark.shape_differences(spark.profile_shape(cluster, nodes), spark.bench_shape(identity)),
                    ([], []),
                )
                self.assertEqual(spark.benchmark_reference_problems(cluster, nodes), [])
        # All promoted shapes name a benchmark_reference (including the 4 KiB-page triangle).
        self.assertIn("benchmark_reference", recipe["shapes"]["tp3-triangle"])
        self.assertIn("benchmark_reference", recipe["shapes"]["tp4-ring"])
        self.assertIn("benchmark_reference", recipe["shapes"]["tp3-triangle-4k"])

    def test_tp4_profile_compares_with_the_tp4_run(self) -> None:
        cluster, nodes = self.profile(self.TP4_RECIPE, "tp4-ring")
        self.assertEqual(cluster["benchmark_reference"], self.TP4_REFERENCE)
        # The reference is a suite run: the old bench runs without a comparison (setup's verify step).
        status, output = self.bench(self.TP4_RECIPE, cluster, nodes, "--report", self.report(cluster, nodes))
        self.assertEqual(status, 0)
        self.assertIn("is a suite run; this bench compares with nothing", output)
        self.assertNotIn("reference:", output)
        status, output = self.bench(self.TP4_RECIPE, cluster, nodes, "--report", self.report(cluster, nodes),
                                    "--compare", self.TP4_BENCH)
        self.assertEqual(status, 0)
        self.assertIn(f"reference: {ROOT / self.TP4_BENCH}", output)
        self.assertNotIn("WARN", output)
        self.assertNotIn("ERROR", output)
        cluster["benchmark_reference"] = "manifests/benchmarks/missing.json"
        with self.assertRaisesRegex(SystemExit, "benchmark_reference does not exist: manifests/benchmarks/missing.json"):
            self.bench(self.TP4_RECIPE, cluster, nodes, "--report", self.report(cluster, nodes))

    def suite_run(self, cluster: dict, nodes: dict, name: str, suite_name: str = "release") -> Path:
        """A run of the deployment's suite, recorded on the profile's shape."""
        suite = spark.sonar.module("suite")
        definition = spark.sonar.load_suite(suite_name)
        run = Path(self.tmp.name) / name
        run.mkdir(exist_ok=True)
        identity = {"topology": nodes, "transport": spark.topology.transport(cluster),
                    "kernel_backend": spark.kernel_backend.backend(cluster)}
        (run / "suite.json").write_text(json.dumps({"suite": definition, "suite_sha256": suite.digest(definition),
                                                    "steps": [], "metadata": {"identity": identity}}))
        return run

    def test_suite_bench_needs_the_dev_rpc_for_the_grid(self) -> None:
        cluster, nodes = self.profile("config/cluster.json", "tp4-ring")
        status, output = self.bench("config/cluster.json", cluster, nodes, "--suite", "release")
        self.assertEqual(status, spark.SUITE_REFUSED)
        self.assertIn("needs the dev-mode collective RPC (VLLM_SERVER_DEV_MODE=1)", output)

    def test_suite_bench_refuses_a_default_reference_of_another_suite(self) -> None:
        # tp4-ring's reference is r6n's release run, from before the Poisson step.
        cluster, nodes = self.profile(self.TP4_RECIPE, "tp4-ring")
        cluster["environment"]["VLLM_SERVER_DEV_MODE"] = "1"
        status, output = self.bench(self.TP4_RECIPE, cluster, nodes, "--suite", "release")
        self.assertEqual(status, spark.SUITE_REFUSED)
        self.assertIn("a changed suite is a new suite", output)
        self.assertEqual(spark.bench_shape(spark.reference_identity(ROOT / self.TP4_REFERENCE)),
                         {"nodes": 4, "transport": "oneshot-ring4", "kernel_backend": "tilelang"})

    def test_suite_reference_shape(self) -> None:
        cluster, nodes = self.profile("config/cluster.json", "tp4-ring")
        tp3, tp3_nodes = self.profile("config/cluster.json", "tp3-triangle")
        suite = spark.sonar.module("suite")
        definition = spark.sonar.load_suite("release")
        same, other = self.suite_run(cluster, nodes, "tp4"), self.suite_run(tp3, tp3_nodes, "tp3")
        args = spark.parser().parse_args(["bench", "--suite", "release", "--compare", str(same)])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(spark.suite_reference(args, cluster, nodes, definition, suite.digest(definition)), (same, None))
        self.assertEqual(output.getvalue(), "")
        # An explicit run of another shape warns; a default one is refused.
        args.compare = str(other)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(spark.suite_reference(args, cluster, nodes, definition, suite.digest(definition))[0], other)
        self.assertIn("WARN: reference", output.getvalue())
        args.compare = None
        with mock.patch.object(spark, "benchmark_reference", return_value=(other, True)):
            path, refusal = spark.suite_reference(args, cluster, nodes, definition, suite.digest(definition))
        self.assertIsNone(path)
        self.assertIn("nodes 3", refusal)
        args.compare = str(Path(self.tmp.name))
        self.assertIn("is not a suite run directory",
                      spark.suite_reference(args, cluster, nodes, definition, suite.digest(definition))[1])

    def test_suite_bench_runs_under_bench_conditions(self) -> None:
        """Cooling, the fan curve and the memory watch around the suite; the identity in its
        suite.json; conditions.json beside it; the curve stopped even when a step fails."""
        cluster, nodes = self.profile("config/cluster.json", "tp4-ring")
        cluster["environment"]["VLLM_SERVER_DEV_MODE"] = "1"
        out = Path(self.tmp.name) / "run"
        calls: list = []

        class Watch:
            def __init__(self, *args): calls.append("watch")
            def wait_ready(self): calls.append("ready")
            def thermal_summary(self): return {"dgx1": {"thermal_slowdown_ms": 0}}
            def close(self): return {"dgx1": {"min_available_gib": 20.0}}

        class Child:
            def __init__(self, argv, cwd, env):
                metadata = json.loads(Path(argv[argv.index("--metadata") + 1]).read_text())
                out.mkdir(parents=True, exist_ok=True)
                (out / "suite.json").write_text(json.dumps({"metadata": metadata}))
                # Sonar runs the deployment's own suite file, from the pinned checkout.
                self.test.assertEqual(argv[1:4], ["-m", "sonar", "suite"])
                self.test.assertEqual(argv[argv.index("--suite") + 1], str(ROOT / "config/suites/release.json"))
                self.test.assertIn(str(spark.sonar.checkout()), env["PYTHONPATH"])
                calls.append(Path(argv[argv.index("--suite") + 1]).stem)
            def poll(self): return 0
            def wait(self): return self.code

        Child.test = self
        spark.sonar.checkout()  # verified now: the Popen below is the suite's child only
        for code, expected in ((0, 0), (1, spark.SUITE_FAILED)):
            with self.subTest(code=code):
                calls.clear()
                Child.code = code
                options = spark.parser().parse_args(["--cluster-config", "config/cluster.json", "bench", "--suite",
                                                     "release", "--compare", "none", "--output", str(out)])
                with mock.patch.object(spark, "configuration", return_value=(cluster, nodes, {})), \
                        mock.patch.object(spark, "bench_identity", return_value={"transport": "oneshot-ring4",
                                                                               "live_problems": [], "live_warnings": []}), \
                        mock.patch.object(spark, "cool_nodes", side_effect=lambda *a, **k: calls.append("cool") or []), \
                        mock.patch.object(spark, "start_fan_curve", side_effect=lambda *a, **k: calls.append("fan") or []), \
                        mock.patch.object(spark, "stop_fan_curve", side_effect=lambda *a, **k: calls.append("fan stop") or []), \
                        mock.patch.object(spark, "MemoryWatch", Watch), \
                        mock.patch.object(spark.subprocess, "Popen", Child), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(options.func(options), expected)
                self.assertEqual(calls, ["watch", "cool", "fan", "ready", "release", "fan stop"])
                self.assertEqual(json.loads((out / "suite.json").read_text())["metadata"]["identity"]["transport"],
                                 "oneshot-ring4")
                conditions = json.loads((out / "conditions.json").read_text())
                self.assertEqual((conditions["halted"], conditions["memory"]["dgx1"]["min_available_gib"]), (None, 20.0))

    def test_default_reference_of_another_shape_is_refused(self) -> None:
        # Without its own reference the TP4 profile falls back to the baseline's
        # three-node run. Refuse it both live, before contacting the cluster,
        # and for a saved report.
        cluster, nodes = self.profile("config/cluster.json", "tp4-ring")
        cluster.pop("benchmark_reference", None)  # a ring not yet benched with the image
        # On r6h's baseline, whose default reference is its three-node run.
        cluster["promoted_baseline"] = Path(self.TP3_REFERENCE).stem
        for argv in ([], ["--report", self.report(cluster, nodes)]):
            with self.subTest(argv=argv):
                status, output = self.bench("config/cluster.json", cluster, nodes, *argv)
                self.assertEqual(status, 1)
                self.assertIn(f"ERROR: reference {self.TP3_REFERENCE} {self.TP4_MISMATCH.format('run')}\n", output)
                self.assertIn("refusing to compare with a default reference of another shape", output)
                self.assertIn("--compare PATH", output)

    def test_explicit_reference_of_another_shape_only_warns(self) -> None:
        cluster, nodes = self.profile("config/cluster.json", "tp4-ring")
        report = self.report(cluster, nodes)
        status, output = self.bench(
            "config/cluster.json", cluster, nodes, "--report", report, "--compare", self.TP3_REFERENCE
        )
        self.assertEqual(status, 0)
        self.assertIn(f"WARN: reference {self.TP3_REFERENCE} {self.TP4_MISMATCH.format('run')}\n", output)
        self.assertIn(f"reference: {ROOT / self.TP3_REFERENCE}", output)
        self.assertNotIn("ERROR", output)
        status, output = self.bench("config/cluster.json", cluster, nodes, "--report", report, "--compare", "none")
        self.assertEqual(status, 0)
        self.assertNotIn("reference", output)

    def test_kernel_backend_counts_and_unrecorded_fields_only_warn(self) -> None:
        cluster, nodes = self.profile("config/cluster.json", "tp3-triangle")
        r5p = spark.read_json("manifests/benchmarks/2026-10-05-karmic-kraken-r5p.json")["identity"]
        self.assertEqual(
            spark.shape_differences(spark.profile_shape(cluster, nodes), spark.bench_shape(r5p)),
            (["transport", "kernel_backend"], []),
        )
        # Reports before r5p record neither the node map nor the transport, and
        # those before r6 no kernel_backend, which means B12X.
        self.assertEqual(spark.bench_shape({}), {"nodes": None, "transport": None, "kernel_backend": "b12x"})
        name = "experiments/2026-10-05-tilelang-r6/r5p/cluster-64k.json"
        b12x, nodes = self.profile(name)
        b12x["promoted_baseline"] = "2026-10-02-karmic-kraken-r5o-64k"
        status, output = self.bench(name, b12x, nodes, "--report", self.report(b12x, nodes))
        self.assertEqual(status, 0)
        self.assertIn(f"WARN: cannot compare nodes, transport with reference {self.OLD_REFERENCE}: not recorded\n", output)
        self.assertNotIn("ERROR", output)

    def test_doctor_checks_a_named_reference(self) -> None:
        cluster, nodes = self.profile("config/cluster.json", "tp4-ring")
        self.assertEqual(spark.benchmark_reference_problems(cluster, nodes), [])
        cluster["benchmark_reference"] = self.TP3_REFERENCE
        self.assertEqual(
            spark.benchmark_reference_problems(cluster, nodes),
            [f"benchmark_reference {self.TP3_REFERENCE} {self.TP4_MISMATCH.format('profile')}"],
        )
        cluster["benchmark_reference"] = self.OLD_REFERENCE
        problems = spark.benchmark_reference_problems(cluster, nodes)
        self.assertEqual(spark.split_findings(problems), (
            [f"benchmark_reference {self.OLD_REFERENCE} was measured with kernel_backend b12x; "
             "this profile has kernel_backend tilelang"],
            [f"benchmark_reference {self.OLD_REFERENCE} does not record nodes, transport; bench cannot check them"],
        ))
        cluster["benchmark_reference"] = "manifests/benchmarks/missing.json"
        self.assertEqual(
            spark.benchmark_reference_problems(cluster, nodes),
            ["benchmark_reference does not exist: manifests/benchmarks/missing.json"],
        )
        for raw in ("", "/tmp/bench.json", "../bench.json", 7):
            with self.subTest(raw=raw):
                cluster["benchmark_reference"] = raw
                self.assertEqual(
                    spark.benchmark_reference_problems(cluster, nodes),
                    ["benchmark_reference must be a bench.json or suite run directory relative to the repository root"],
                )
        del cluster["benchmark_reference"]
        self.assertEqual(spark.benchmark_reference_problems(cluster, nodes), [])

    def test_take_fan_control_retries_transient_floor_failure(self) -> None:
        nodes = {"nodes": [{"name": "dgx3"}]}
        node = nodes["nodes"][0]
        outcomes = iter([0, 0, 1, 1, 0])
        seen: list[tuple[str, ...]] = []

        def fake_ssh(_nodes, _node, *command):
            seen.append(command)
            return mock.Mock(returncode=next(outcomes))

        with mock.patch.object(spark, "run_ssh", side_effect=fake_ssh):
            result = spark.take_fan_control(nodes, node, attempts=3, retry_delay_s=0)

        self.assertEqual(result, {"service": True, "floor": True})
        self.assertEqual(
            seen,
            [
                ("systemctl", "is-active", "--quiet", spark.FAN_SERVICE),
                ("sudo", "-n", "systemctl", "stop", spark.FAN_SERVICE),
                ("sudo", "-n", "dgx-fan-control", "set-state", "12"),
                ("systemctl", "is-active", "--quiet", spark.FAN_SERVICE),
                ("sudo", "-n", "dgx-fan-control", "set-state", "12"),
            ],
        )

    def test_distinct_decode_cases_vary_by_stream_without_changing_reference_prompts(self) -> None:
        matrix = spark.sonar.module("matrix")
        self.assertEqual(matrix.DECODE_CASE_GROUPS["distinct"], ["prose-distinct", "code-distinct"])
        self.assertEqual(set(matrix.DECODE_PROMPTS), {"prose", "code"})
        for case in matrix.DECODE_CASE_GROUPS["distinct"]:
            payloads = [payload for _, payload in matrix.decode_payloads("m", case, 16, spark.sonar.REASONING_OFF)]
            self.assertEqual(len({payload["messages"][0]["content"] for payload in payloads}), 16, case)
            for payload in payloads:
                self.assertEqual(payload["temperature"], 0)
                self.assertEqual(payload["seed"], 42)
                self.assertNotIn("min_tokens", payload)
                self.assertNotIn("chat_template_kwargs", payload)

    def test_suite_prefill_includes_262144_under_256k_and_skips_larger(self) -> None:
        matrix = spark.sonar.module("matrix")
        all_sizes = [32768, 131072, 262144, 524288, 1000000]
        for max_model_len, expected in ((262144, [32768, 131072, 262144]), (1048576, all_sizes)):
            seen: list[str] = []

            class FakeBench:
                def run(self, label, payloads, **kwargs):
                    seen.append(label)
                    return {"requests": [{"ok": True, "prompt_tokens": 100, "ttft_s": 0.1}]}

            options = argparse.Namespace(
                prefill_text="source",
                prefill_sizes=all_sizes,
                max_model_len=max_model_len,
                prefill_repeats=1,
                model="m",
            )
            result = matrix.suite_prefill(FakeBench(), options, random.Random(0))
            self.assertEqual(sorted(int(k) for k in result["points"]), expected)
            self.assertEqual(sorted(int(lbl.split()[-1]) for lbl in seen), expected)

    def test_sonar_suite_and_prefill_cool_when_configured_and_keep_digest(self) -> None:
        sb_prefill = spark.sonar.module("prefill")
        sb_suite = spark.sonar.module("suite")

        recorded = json.loads(
            (ROOT / "manifests/suites/2026-11-07-spark-ds41f-1107-tp4/suite.json").read_text()
        )
        loaded = spark.sonar.load_suite("release")
        self.assertEqual(sb_suite.digest(loaded), recorded["suite_sha256"])

        calls: list[tuple[str, float, float]] = []
        fake_cfg = ({"host": {}, "environment": {}}, {"nodes": [{"name": "dgx1"}]}, {})

        def fake_cool(nodes, targets, below_c, timeout_s, **kwargs):
            calls.append(("cool", below_c, timeout_s))
            return []

        spec = json.dumps({"cluster_config": "config/cluster.json", "cool_below": 55, "cool_timeout": 900})
        mini_suite = {
            "name": "mini",
            "steps": [
                {"id": "model", "command": "model", "args": {}},
                {"id": "grid", "command": "engine", "args": {}},
                {"id": "prefill", "command": "prefill", "args": {"sizes": [32768, 131072], "repeats": 1}},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {"SPARK_BENCH_COOL": spec}), \
                mock.patch("spark_ds41f.config.configuration", return_value=fake_cfg), \
                mock.patch("spark_ds41f.bench.fan_control_enabled", return_value=True), \
                mock.patch("spark_ds41f.bench.cool_nodes", side_effect=fake_cool), \
                mock.patch.object(sb_prefill, "document", return_value=("doc", 32000)), \
                mock.patch.object(
                    sb_prefill.client,
                    "stream_chat",
                    return_value={"first": 2.0, "sent": 1.0, "prompt_tokens": 32000, "error": None},
                ):
            def fake_main(argv):
                calls.append(("step:" + argv[0], 0.0, 0.0))
                if argv[0] == "prefill":
                    sb_prefill.run("http://127.0.0.1:8000", "m", [32768, 131072], repeats=1, vllm=False)
                return 0

            with contextlib.redirect_stdout(io.StringIO()):
                sb_suite.run(fake_main, "http://127.0.0.1:8000", mini_suite, tmp)

        # step 0 (model) starts uncooled (bench_suite cools right before Popen);
        # step 1 (grid) cools before starting; step 2 (prefill) cools before each size sample.
        self.assertEqual(
            [c[0] for c in calls],
            ["step:model", "cool", "step:engine", "step:prefill", "cool", "cool"],
        )

    def test_real_rows_shapes_and_subprocess_cli(self) -> None:
        engine = spark.sonar.module("engine")
        shapes = engine.default_shapes(5, 1048576, real_rows=True)
        labels = [engine.label(s) for s in shapes]
        self.assertIn("decode 1x3 real rows @1016K", labels)
        self.assertIn("prefill 8192 real @256K",
                      [engine.label(s) for s in engine.default_shapes(5, 1048576, prefill=True, real_rows=True)])
        self.assertFalse(set(labels) & {engine.label(s) for s in engine.default_shapes(5, 1048576)})
        self.assertTrue(all(s["context_len"] % 4096 == 0 for s in shapes))

        state = {"real_rows": True, "seen_fill": []}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/collective_rpc":
                    spec = json.loads(body["args"][0])
                    state["seen_fill"].append(spec.get("fill_real_context"))
                    timed = [
                        {
                            "shape": s,
                            "step_ms": [10.0, 10.1, 10.2],
                            "real_rows": bool(s.get("real_rows")) if state["real_rows"] else False,
                        }
                        for s in spec["shapes"]
                    ]
                    payload = json.dumps({"results": [{"rank": 0, "settle": 1, "shapes": timed}]}).encode()
                else:
                    payload = b"{}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            raw = engine.run(url, shapes[:2], replays=3, require_idle=False, fill_real_context=False)
            self.assertTrue(all(timed["real_rows"] for timed in raw["grid"]))
            self.assertEqual(state["seen_fill"], [False])
            state["real_rows"] = False
            with self.assertRaisesRegex(RuntimeError, "padding rows for decode 1x2 real rows @4K"):
                engine.run(url, shapes[:2], replays=3, require_idle=False)
        finally:
            server.shutdown()
            server.server_close()

        help_proc = subprocess.run(
            spark.sonar.argv("engine", "--help"),
            env=spark.sonar.environment(),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--real-rows", help_proc.stdout)
        self.assertIn("--no-fill-real-context", help_proc.stdout)


class BetweenBenchCoolingTest(unittest.TestCase):
    """Cooling runs between suites and before every individual prefill sample."""

    def test_command_bench_cools_between_suites_and_before_each_prefill_sample(self) -> None:
        matrix = spark.sonar.module("matrix")
        with tempfile.TemporaryDirectory() as tmp:
            cluster = spark.read_cluster("config/cluster.json", "tp4-ring")
            nodes = spark.read_json("config/examples/nodes-ring4.json")
            options = spark.parser().parse_args([
                "--cluster-config", "config/cluster.json", "bench",
                "--compare", "none",
                "--suites", "quality,decode,prefill",
                "--prefill-sizes", "32768,131072",
                "--prefill-repeats", "2",
                "--output", tmp,
            ])
            identity = {"live_warnings": [], "live_problems": []}
            events: list[str] = []

            class Watch:
                def __init__(self, *args): pass
                def wait_ready(self): pass
                def check(self): pass
                def thermal_summary(self): return {}
                def close(self): return {}

            def fake_cool(*args, **kwargs):
                events.append("cool")
                return [{"node": "dgx1", "final_c": 50.0}]

            def fake_quality(bench, opts, rng):
                events.append("quality")
                return {"ok": True, "passed": 1, "runs": []}

            def fake_decode(bench, opts, rng):
                events.append("decode")
                return {"points": {}}

            def fake_prefill_run(self_bench, label, payloads, **kwargs):
                events.append(f"run:{label}")
                return {
                    "wall_s": 1.0,
                    "requests": [{"ok": True, "prompt_tokens": 100, "ttft_s": 0.1}],
                    "metrics": {},
                    "peak": {},
                }

            orig_cool_phase = matrix.Sampler.cool_phase

            def spy_cool_phase(self_bench, phase):
                events.append(f"phase:{phase}")
                return orig_cool_phase(self_bench, phase)

            def spy_wait_idle(self_bench, timeout=300):
                events.append("wait_idle")
                return {}

            with mock.patch.object(spark, "configuration", return_value=(cluster, nodes, {})), \
                    mock.patch.object(spark, "bench_identity", return_value=identity), \
                    mock.patch.object(spark, "MemoryWatch", Watch), \
                    mock.patch.object(spark, "cool_nodes", side_effect=fake_cool), \
                    mock.patch.object(spark, "start_fan_curve", return_value=[]), \
                    mock.patch.object(spark, "fan_curve_problems", return_value=[]), \
                    mock.patch.object(spark, "stop_fan_curve", return_value=[]), \
                    mock.patch.object(matrix.Sampler, "cool_phase", spy_cool_phase), \
                    mock.patch.object(matrix.Sampler, "wait_idle", spy_wait_idle), \
                    mock.patch.object(matrix.Sampler, "run", fake_prefill_run), \
                    mock.patch.dict(matrix.SUITE_FUNCTIONS, {
                        "quality": fake_quality,
                        "decode": fake_decode,
                    }), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = options.func(options)

            self.assertEqual(status, 0)
            report = json.loads((Path(tmp) / "bench.json").read_text())
            self.assertEqual(events[0:4], ["cool", "quality", "wait_idle", "phase:suite:decode"])
            self.assertEqual(events[4:6], ["cool", "decode"])
            prefill_phases = [entry["phase"] for entry in report["between_cooling"]]
            self.assertEqual(prefill_phases[0], "suite:decode")
            self.assertEqual(len(prefill_phases), 5)
            self.assertEqual([p.split(":")[0] for p in prefill_phases[1:]], ["prefill"] * 4)
            self.assertTrue(any(p.endswith(":r1") for p in prefill_phases[1:]))
            self.assertTrue(any(p.endswith(":r2") for p in prefill_phases[3:]))

    def test_between_bench_cooling_failure_aborts_and_stops_fan_curve(self) -> None:
        matrix = spark.sonar.module("matrix")
        with tempfile.TemporaryDirectory() as tmp:
            cluster = spark.read_cluster("config/cluster.json", "tp4-ring")
            nodes = spark.read_json("config/examples/nodes-ring4.json")
            options = spark.parser().parse_args([
                "--cluster-config", "config/cluster.json", "bench",
                "--compare", "none",
                "--suites", "quality,decode",
                "--output", tmp,
            ])
            identity = {"live_warnings": [], "live_problems": []}
            calls = {"cool": 0, "stop_fan": 0}

            class Watch:
                def __init__(self, *args): pass
                def wait_ready(self): pass
                def check(self): pass
                def thermal_summary(self): return {}
                def close(self): return {}

            def fake_cool(*args, **kwargs):
                calls["cool"] += 1
                if calls["cool"] == 2:
                    raise RuntimeError("sensor timeout")
                return [{"node": "dgx1", "final_c": 50.0}]

            with mock.patch.object(spark, "configuration", return_value=(cluster, nodes, {})), \
                    mock.patch.object(spark, "bench_identity", return_value=identity), \
                    mock.patch.object(spark, "MemoryWatch", Watch), \
                    mock.patch.object(spark, "cool_nodes", side_effect=fake_cool), \
                    mock.patch.object(spark, "start_fan_curve", return_value=[{"node": "dgx1", "started": True}]), \
                    mock.patch.object(spark, "fan_curve_problems", return_value=[]), \
                    mock.patch.object(spark, "stop_fan_curve", side_effect=lambda *a, **k: calls.__setitem__("stop_fan", calls["stop_fan"] + 1) or []), \
                    mock.patch.object(matrix.Sampler, "wait_idle", return_value={}), \
                    mock.patch.dict(matrix.SUITE_FUNCTIONS, {
                        "quality": lambda b, o, r: {"ok": True, "passed": 1, "runs": []},
                        "decode": lambda b, o, r: self.fail("decode should not run after cooling failure"),
                    }), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = options.func(options)

            self.assertEqual(status, 1)
            self.assertEqual(calls["stop_fan"], 1)
            report = json.loads((Path(tmp) / "bench.json").read_text())
            self.assertEqual(report["aborted"], "cooling failed (suite:decode): sensor timeout")
            self.assertIn("quality", report["suites"])
            self.assertNotIn("decode", report["suites"])


if __name__ == "__main__":
    unittest.main()
