"""Host-independent tests for the experiment window runner (scripts/lab.py)."""

from __future__ import annotations

import contextlib
import io
import importlib.machinery
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("lab", str(ROOT / "scripts" / "lab.py"))
spec = importlib.util.spec_from_loader("lab", loader)
lab = importlib.util.module_from_spec(spec)
loader.exec_module(lab)
# Host-independent: read the shipped example, not the site's git-ignored config/nodes.json.
lab.spark.site_nodes = lambda: lab.spark.read_json("config/nodes.example.json")
# Never read or write the real hold and request files, even when the suite runs on dgx1.
_STATE = tempfile.TemporaryDirectory()
lab.HOLD = Path(_STATE.name) / "spark-hold.json"
lab.LEGACY_HOLD = Path(_STATE.name) / "spark3-hold.json"
lab.REQUEST = Path(_STATE.name) / "spark-request.json"
lab.LEGACY_REQUEST = Path(_STATE.name) / "spark3-request.json"

SPEC = {
    "experiment": "experiments/2026-09-29-determinism",
    "run": "lab0",
    "jobs": [
        {"kind": "measure", "profile": "lean", "bracket": True,
         "arms": [{"config": "cluster-r5o-pin.json", "label": "r5o"},
                  {"config": "cluster-detm-r5o-ref4d-b4144-pin.json", "label": "ref4d-b4144"}]},
        {"kind": "validate", "config": "cluster-detm-r5o-ref4d-b4144-trace8.json", "name": "ref4d-b4144",
         "chunk": 4096, "tier": "quick"},
        {"kind": "kernel", "bundles": [{"bundle": "b/mhc", "node": "dgx2"}]},
    ],
}


class HoldTest(unittest.TestCase):
    def test_new_hold_is_ours_and_carries_its_cap(self) -> None:
        hold = lab.new_hold(30, "run lab0", at=1_000_000.0)
        self.assertTrue(lab.hold_is_ours(hold))
        self.assertEqual(lab.parse_time(hold["expected_end"]) - lab.parse_time(hold["since"]), 1800)
        self.assertAlmostEqual(lab.heartbeat_age(hold, at=1_000_090.0), 90, delta=1)

    def test_a_window_without_a_cap_never_closes_on_the_clock(self) -> None:
        hold = lab.new_hold(0, "queue", at=1_000_000.0)
        self.assertIsNone(hold["expected_end"])
        self.assertEqual(lab.DEFAULT_MINUTES, 0)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(lab, "REQUEST", Path(directory) / "request.json"), \
                mock.patch.object(lab, "LEGACY_REQUEST", Path(directory) / "legacy-request.json"):
            self.assertIsNone(lab.window_should_close(hold, at=1_000_000.0 + 7 * 86400))
            lab.REQUEST.write_text("{}")
            self.assertIn("requested", lab.window_should_close(hold, at=1_000_060.0))

    def test_foreign_or_missing_holds_are_not_ours(self) -> None:
        self.assertFalse(lab.hold_is_ours(None))
        self.assertFalse(lab.hold_is_ours({"holder": "codex tool-eval-bench"}))

    def test_window_closes_on_request_or_cap(self) -> None:
        hold = lab.new_hold(10, "", at=1_000_000.0)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(lab, "REQUEST", Path(directory) / "request.json"), \
                mock.patch.object(lab, "LEGACY_REQUEST", Path(directory) / "legacy-request.json"):
            self.assertIsNone(lab.window_should_close(hold, at=1_000_060.0))
            self.assertEqual(lab.window_should_close(hold, at=1_000_000.0 + 601), "time cap reached")
            lab.REQUEST.write_text("{}")
            self.assertIn("requested", lab.window_should_close(hold, at=1_000_060.0))

    def test_former_hold_and_request_names_still_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(lab, "HOLD", Path(directory) / "spark-hold.json"), \
                mock.patch.object(lab, "LEGACY_HOLD", Path(directory) / "spark3-hold.json"), \
                mock.patch.object(lab, "REQUEST", Path(directory) / "spark-request.json"), \
                mock.patch.object(lab, "LEGACY_REQUEST", Path(directory) / "spark3-request.json"):
            self.assertIsNone(lab.read_hold())
            lab.LEGACY_HOLD.write_text('{"holder": "another agent"}')
            self.assertEqual(lab.read_hold()["holder"], "another agent")
            self.assertFalse(lab.hold_is_ours(lab.read_hold()))
            hold = lab.new_hold(10, "", at=1_000_000.0)
            self.assertIsNone(lab.window_should_close(hold, at=1_000_060.0))
            lab.LEGACY_REQUEST.write_text("{}")
            self.assertIn("spark3-request.json", lab.window_should_close(hold, at=1_000_060.0))

    def test_read_hold_tolerates_damage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hold.json"
            self.assertIsNone(lab.read_hold(path))
            path.write_text("{not json")
            self.assertFalse(lab.hold_is_ours(lab.read_hold(path)))


class PlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.steps = lab.plan(SPEC)

    def test_measure_brackets_the_first_arm_and_never_stops_between_arms(self) -> None:
        boots = [s for s in self.steps if s["kind"] == "boot"]
        self.assertEqual([b["label"] for b in boots[:3]], ["r5o", "ref4d-b4144", "r5o-end"])
        first_validate = next(i for i, s in enumerate(self.steps) if s.get("label") == "ref4d-b4144" and
                              s["kind"] == "boot" and "trace8" in s["config"])
        self.assertNotIn("stop", [s["kind"] for s in self.steps[:first_validate]])

    def test_measure_ends_with_the_comparison_table(self) -> None:
        table = next(s for s in self.steps if s["kind"] == "table")
        self.assertEqual(table["argv"][1:], ["lab0", "r5o", "ref4d-b4144", "r5o-end"])

    def test_lean_profile_and_result_layout(self) -> None:
        bench = next(s for s in self.steps if s["kind"] == "cli" and s["label"] == "ref4d-b4144")
        self.assertIn("1024,16384", bench["argv"])
        self.assertEqual(bench["argv"][-1], "results/private/bench/lab0-ref4d-b4144")
        outputs = {s["out"] for s in self.steps if s["kind"] == "script" and s.get("label") == "r5o"}
        self.assertIn("results/private/determinism/lab0/c1-distinct-r5o.jsonl", outputs)
        self.assertIn("results/private/determinism/lab0/mixed-r5o.jsonl", outputs)
        c1 = next(s for s in self.steps if s["kind"] == "script" and "c1-distinct-r5o" in s["out"])
        self.assertEqual(c1["argv"][-2:], ["--prompts", "12"])

    def test_quick_validation_restarts_without_mixes_then_analyses_stopped(self) -> None:
        validate = self.steps[next(i for i, s in enumerate(self.steps) if s["kind"] == "fresh_inventory") - 1:]
        kinds = [s["kind"] for s in validate]
        self.assertEqual(kinds.count("boot"), 2)
        scripts = [" ".join(s["argv"]) for s in validate if s["kind"] == "script"]
        self.assertFalse(any("trace_mixes" in s for s in scripts))
        self.assertTrue(any("--suffix -bootB" in s and "--scenarios cache_long,distinct" in s for s in scripts))
        self.assertTrue(all("--chunk 4096" in s for s in scripts if "scenario_trace" in s))
        analyze_at = kinds.index("analyze")
        self.assertEqual(kinds[analyze_at - 1], "stop")
        self.assertEqual(validate[analyze_at]["dirs"], ["c8", "scenarios"])

    def test_full_validation_keeps_mixes_and_every_scenario(self) -> None:
        full = dict(SPEC, jobs=[dict(SPEC["jobs"][1], tier="full")])
        scripts = [" ".join(s["argv"]) for s in lab.plan(full) if s["kind"] == "script"]
        self.assertTrue(any("trace_mixes" in s for s in scripts))
        self.assertFalse(any("--scenarios" in s for s in scripts))

    def test_kernel_jobs_run_with_the_cluster_stopped(self) -> None:
        kernel_at = [s["kind"] for s in self.steps].index("kernel")
        self.assertEqual(self.steps[kernel_at - 1]["kind"], "stop")

    def test_unknown_job_kind_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            lab.plan(dict(SPEC, jobs=[{"kind": "mystery"}]))

    def test_kernel_cooling_honours_the_site_fan_policy(self) -> None:
        """A no-fan site must not be made to take a fan floor before a kernel job.

        The same site flag command_bench passes has to reach cool_nodes here: on a
        board that cannot select a fan floor, cool_nodes raises and the job dies.
        """
        step = {"kind": "kernel", "out": "results/private/lab/r",
                "bundles": [{"bundle": "bundles/mhc", "node": "dgx2"}]}
        seen = {}

        def cool(nodes, targets, below_c, timeout_s, poll_s=None, fan_control=True):
            seen["fan_control"] = fan_control
            return []

        cluster = json.loads((ROOT / "config/cluster.json").read_text())
        cluster["host"]["fan_control"] = False
        nodes = json.loads((ROOT / "config/nodes.example.json").read_text())
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(lab, "ROOT", Path(directory)), \
                mock.patch.object(lab.spark.repository_path, "__call__", create=True), \
                mock.patch.object(lab, "production_configuration", return_value=(cluster, nodes, None)), \
                mock.patch.object(lab.spark, "cool_nodes", side_effect=cool), \
                mock.patch.object(lab.spark, "run_ssh", return_value=subprocess.CompletedProcess([], 0, "{}", "")), \
                mock.patch.object(lab.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            bundle = Path(directory) / "bundles" / "mhc"
            bundle.mkdir(parents=True)
            (bundle / "candidate.json").write_text(json.dumps({"sync": []}))
            lab.run_kernel_bundles(step)
        self.assertIs(seen.get("fan_control"), False)


class BootTimeTest(unittest.TestCase):
    def test_boot_rejects_a_different_physical_topology_before_launch(self) -> None:
        from unittest import mock

        promoted = {"nodes": [{"rank": 0}, {"rank": 1}, {"rank": 2}]}
        candidate = {"nodes": [*promoted["nodes"], {"rank": 3}]}
        with mock.patch.object(lab.spark, "configuration", return_value=({}, candidate, {})), \
             mock.patch.object(lab, "nodes_config", return_value=promoted), \
             mock.patch.object(lab, "spark_cli") as launch, \
             contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertFalse(lab.boot("ring4.json"))
            launch.assert_not_called()
        self.assertIn("lab windows require the promoted node topology", log.getvalue())

    def test_ready_seconds_come_from_the_cluster_ready_line(self) -> None:
        lines = ["dgx1: steady memguard active", "cluster ready; memory guards are active on all nodes (+116.9s)"]
        self.assertEqual(lab.ready_seconds(lines), 116.9)
        self.assertIsNone(lab.ready_seconds(["boot failed"]))


def setUpModule() -> None:
    # These tests ran against the default profile when it was TP3 on config/nodes.json;
    # the recipe resolves the same shape for them.
    patcher = mock.patch.object(lab.spark.cluster_recipe, "site_shape", return_value="tp3-triangle")
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


# A production profile other than the recipe: the archived TP4 r6k file (a
# site restores an archived release this way while a shape is unqualified).
TP4 = "experiments/2026-10-14-one-recipe/tp4-r6k-production.json"
RING4 = {"nodes": [{"name": f"dgx{rank + 1}", "rank": rank, "head": rank == 0,
                    "management_ip": f"192.0.2.{71 + rank}"} for rank in range(4)]}


def fake_configuration(args=None):
    """The TP4 profile's node map is a site file (git-ignored); stand in for it with a ring of four."""
    config = getattr(args, "cluster_config", lab.spark.DEFAULT_CLUSTER_CONFIG)
    nodes = RING4 if "tp4" in config else lab.spark.site_nodes()
    return {"host": {"home": "/home/x"}, "deployment": {"path": "{home}/projects/spark-ds41f"}}, nodes, {}


def no_subprocesses(*args, **kwargs):
    raise AssertionError(f"unexpected subprocess: {args}")


class ProductionProfileTest(unittest.TestCase):
    """The profile a window restores is selectable, recorded in the hold and used by every cluster action."""

    def setUp(self) -> None:
        # main() sets the module's profile; restore it after every test.
        patcher = mock.patch.object(lab, "PRODUCTION_CONFIG", lab.spark.DEFAULT_CLUSTER_CONFIG)
        patcher.start()
        self.addCleanup(patcher.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.state = Path(directory.name)
        for name, file in (("HOLD", "spark-hold.json"), ("LEGACY_HOLD", "spark3-hold.json"),
                           ("REQUEST", "spark-request.json"), ("LEGACY_REQUEST", "spark3-request.json")):
            patcher = mock.patch.object(lab, name, self.state / file)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(lab.spark, "configuration", side_effect=fake_configuration)
        patcher.start()
        self.addCleanup(patcher.stop)

    def main(self, *argv: str) -> tuple[int, str]:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = lab.main(list(argv))
        return code, out.getvalue()

    def hold(self, production: str | None, heartbeat_at: float | None = None) -> dict:
        hold = lab.new_hold(60, "test", at=heartbeat_at)
        if production is None:
            del hold["production_config"]  # a window opened before the field existed
        else:
            hold["production_config"] = production
        lab.write_json_atomic(lab.HOLD, hold)
        return hold

    def test_the_window_record_wins_then_the_option_then_the_default(self) -> None:
        self.assertEqual(lab.resolve_production(None, None), "config/cluster.json")
        self.assertEqual(lab.resolve_production("./experiments/2026-10-14-one-recipe/tp4-r6k-production.json", None), TP4)
        ours = lab.new_hold(10, "", at=1_000_000.0)
        ours["production_config"] = TP4
        self.assertEqual(lab.resolve_production(None, ours), TP4)
        self.assertEqual(lab.resolve_production(TP4, ours), TP4)
        with self.assertRaises(SystemExit) as refused:
            lab.resolve_production("config/cluster.json", ours)
        self.assertIn("restores experiments/2026-10-14-one-recipe/tp4-r6k-production.json", str(refused.exception))
        foreign = {"holder": "codex tool-eval-bench", "production_config": TP4}
        self.assertEqual(lab.resolve_production(None, foreign), "config/cluster.json")
        legacy = {k: v for k, v in ours.items() if k != "production_config"}
        self.assertEqual(lab.resolve_production(None, legacy), "config/cluster.json")
        self.assertEqual(lab.resolve_production(TP4, legacy), TP4)
        with self.assertRaises(SystemExit):
            lab.resolve_production("config/no-such-profile.json", None)

    def test_the_hold_records_the_profile_beside_the_shared_fields(self) -> None:
        with mock.patch.object(lab, "PRODUCTION_CONFIG", TP4):
            hold = lab.new_hold(30, "run lab0", at=1_000_000.0)
        self.assertEqual(hold["production_config"], TP4)
        for field in ("holder", "since", "expected_end", "heartbeat", "rule"):  # AGENTS.md, Shared cluster windows
            self.assertIn(field, hold)

    def test_stop_targets_the_profile_and_keeps_containers_unless_asked(self) -> None:
        lab.PRODUCTION_CONFIG = TP4
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertTrue(lab.stop_cluster(dry=True))
            self.assertTrue(lab.stop_cluster(dry=True, remove=True))
        kept, removed = [line for line in out.getvalue().splitlines() if "$" in line]
        self.assertTrue(kept.endswith("--cluster-config experiments/2026-10-14-one-recipe/tp4-r6k-production.json cluster stop --apply --parallel"))
        self.assertTrue(removed.endswith("cluster stop --remove --apply --parallel"))
        steps = lab.plan(dict(SPEC, jobs=[SPEC["jobs"][2], dict(SPEC["jobs"][2], remove_stopped=True)]))
        stops = [lab.describe_step(step) for step in steps if step["kind"] == "stop"]
        self.assertEqual(stops, ["bin/spark --cluster-config experiments/2026-10-14-one-recipe/tp4-r6k-production.json cluster stop --apply --parallel",
                                 "bin/spark --cluster-config experiments/2026-10-14-one-recipe/tp4-r6k-production.json cluster stop --remove "
                                 "--apply --parallel"])

    def test_doctor_live_checks_the_profile(self) -> None:
        lab.PRODUCTION_CONFIG = TP4
        serving = subprocess.CompletedProcess([], 0, stdout="configuration OK; live cluster matches\n")
        with mock.patch.object(lab.subprocess, "run", return_value=serving) as run:
            self.assertTrue(lab.production_live())
        argv = run.call_args.args[0]
        self.assertEqual(argv[2:], ["--cluster-config", TP4, "doctor", "--live"])
        down = subprocess.CompletedProcess([], 1, stdout="ERROR: dgx4: container is not running\n")
        problems = []
        with mock.patch.object(lab.subprocess, "run", return_value=down):
            self.assertFalse(lab.production_live(problems))
        self.assertEqual(problems, ["ERROR: dgx4: container is not running"])

    def test_boot_compares_candidates_with_the_profile_node_map(self) -> None:
        lab.PRODUCTION_CONFIG = TP4
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertTrue(lab.boot("experiments/x/cluster-tp4-candidate.json", dry=True))
            self.assertFalse(lab.boot("config/cluster.json", dry=True))
        self.assertIn("--cluster-config experiments/x/cluster-tp4-candidate.json cluster start --replace --change-shape --apply",
                      out.getvalue())
        self.assertIn("lab windows require the promoted node topology (experiments/2026-10-14-one-recipe/tp4-r6k-production.json)", out.getvalue())
        self.assertEqual(out.getvalue().count("cluster start"), 1)

    def test_dry_run_shows_the_profile_for_every_cluster_action(self) -> None:
        spec = self.state / "spec.json"
        spec.write_text(json.dumps(SPEC))
        # SPEC's arms predate the current locks; their own checks are not this test's subject.
        with mock.patch.object(lab, "arm_problems", return_value=[]):
            code, out = self.main("run", str(spec), "--dry-run", "--production-config", TP4)
        self.assertEqual(code, 0)
        self.assertIn("steps; production experiments/2026-10-14-one-recipe/tp4-r6k-production.json", out.splitlines()[0])
        self.assertIn("bin/spark --cluster-config experiments/2026-10-14-one-recipe/tp4-r6k-production.json cluster stop --apply --parallel", out)
        self.assertNotIn("--remove", out)
        code, out = self.main("window", "open", "--dry-run", "--production-config", TP4)
        self.assertIn("window restoring experiments/2026-10-14-one-recipe/tp4-r6k-production.json", out)
        self.assertFalse(lab.HOLD.exists())

    def test_close_restores_the_profile_the_window_recorded(self) -> None:
        self.hold(TP4)
        code, out = self.main("window", "close", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("--cluster-config experiments/2026-10-14-one-recipe/tp4-r6k-production.json cluster start --replace --change-shape --apply", out)
        self.assertTrue(lab.HOLD.exists())
        with self.assertRaises(SystemExit):
            self.main("window", "close", "--dry-run", "--production-config", "config/cluster.json")

    def test_a_window_opened_before_the_record_closes_on_the_option_or_default(self) -> None:
        self.hold(None)
        _, out = self.main("window", "close", "--dry-run")
        self.assertIn("--cluster-config config/cluster.json cluster start", out)
        _, out = self.main("window", "close", "--dry-run", "--production-config", TP4)
        self.assertIn("--cluster-config experiments/2026-10-14-one-recipe/tp4-r6k-production.json cluster start", out)

    def test_open_refuses_unless_the_profile_is_live(self) -> None:
        def down(problems=None):
            problems.append("ERROR: dgx4: container is not running")
            return False

        with mock.patch.object(lab, "published_problems", return_value=[]), \
                mock.patch.object(lab, "production_live", side_effect=down), \
                mock.patch.object(lab, "idle_for") as idle, \
                mock.patch.object(lab.subprocess, "Popen", side_effect=no_subprocesses), \
                self.assertRaises(SystemExit) as refused:
            lab.window_open(30, "test")
        message = str(refused.exception)
        self.assertIn("config/cluster.json is not the live cluster", message)
        self.assertIn("--production-config", message)
        self.assertIn("dgx4: container is not running", message)
        idle.assert_not_called()
        self.assertFalse(lab.HOLD.exists())

    def test_open_records_the_profile_and_hands_it_to_the_watchdog(self) -> None:
        lab.PRODUCTION_CONFIG = TP4
        with mock.patch.object(lab, "published_problems", return_value=[]), \
                mock.patch.object(lab, "production_live", return_value=True), \
                mock.patch.object(lab, "idle_for", return_value=True), \
                mock.patch.object(lab, "ROOT", self.state), \
                mock.patch.object(lab.subprocess, "run", side_effect=no_subprocesses), \
                mock.patch.object(lab.subprocess, "Popen") as popen, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            lab.window_open(30, "test")
        self.assertEqual(lab.read_hold()["production_config"], TP4)
        self.assertEqual(popen.call_args.args[0][-3:], ["watchdog", "--production-config", TP4])
        self.assertIn("restores experiments/2026-10-14-one-recipe/tp4-r6k-production.json", out.getvalue())

    def test_watchdog_restores_the_recorded_profile(self) -> None:
        self.hold(TP4, heartbeat_at=1_000_000.0)  # stale
        with mock.patch.object(lab, "spark_cli", return_value=0) as cli, \
                mock.patch.object(lab, "production_live", side_effect=[False, True]), \
                mock.patch.object(lab.subprocess, "run", side_effect=no_subprocesses), \
                mock.patch.object(lab.subprocess, "Popen", side_effect=no_subprocesses):
            code, out = self.main("watchdog", "--once")
        self.assertEqual(code, 0)
        self.assertEqual(cli.call_args.args, ("--cluster-config", TP4, "cluster", "start", "--replace", "--change-shape", "--apply"))
        self.assertFalse(lab.HOLD.exists())
        self.assertIn("window closed; hold removed", out)

    def test_publish_guard_and_sync_cover_every_profile_node(self) -> None:
        lab.PRODUCTION_CONFIG = TP4
        head = subprocess.CompletedProcess([], 0, stdout="abc1234\n")

        def checkout(nodes, node, *command):
            return subprocess.CompletedProcess([], 0, stdout="old5678\n" if node["name"] == "dgx4" else "abc1234\n")

        with mock.patch.object(lab.subprocess, "run", return_value=head), \
                mock.patch.object(lab.spark, "run_ssh", side_effect=checkout) as ssh:
            problems = lab.published_problems()
        self.assertEqual([call.args[1]["name"] for call in ssh.call_args_list], ["dgx2", "dgx3", "dgx4"])
        self.assertEqual(problems, ["dgx4 checkout old5678 differs from abc1234"])
        with mock.patch.object(lab.subprocess, "run", return_value=head), \
                mock.patch.object(lab, "spark_cli", return_value=0) as cli, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(lab.sync_checkouts())
        self.assertEqual(cli.call_args.args, ("--cluster-config", TP4, "cluster", "sync", "--apply"))


class ProfilePlanTest(unittest.TestCase):
    def test_profile_job_traces_each_workload_then_compares_stopped(self) -> None:
        spec = {"experiment": "e", "run": "costs3", "jobs": [{
            "kind": "profile", "workloads": ["decode", "prefill"],
            "arms": [{"config": "cluster-r5o-pin-prof.json", "label": "r5o"},
                     {"config": "cluster-cand-prof.json", "label": "cand"}]}]}
        steps = lab.plan(spec)
        kinds = [s["kind"] for s in steps]
        self.assertEqual(kinds, ["boot", "curves", "profile", "profile", "boot", "curves", "profile", "profile", "stop", "costs"])
        self.assertEqual(steps[3]["argv"][-2:], ["--tokens", "16384"])
        self.assertEqual(steps[1]["out"], "results/private/determinism/costs3/curves-r5o.json")
        self.assertEqual(steps[-1]["labels"], ["r5o", "cand"])
        self.assertEqual(steps[-1]["out"], "results/private/determinism/costs3")

    def test_profile_workload_arguments_replace_defaults_without_changing_other_workloads(self) -> None:
        spec = {"experiment": "e", "run": "long-prefill", "jobs": [{
            "kind": "profile", "workloads": ["decode", "prefill"],
            "workload_args": {"prefill": ["--tokens", "131072"]},
            "arms": [{"config": "cluster-prof.json", "label": "reference"}]}]}
        steps = lab.plan(spec)
        profiles = [step for step in steps if step["kind"] == "profile"]
        self.assertEqual(profiles[0]["argv"][-2:], ["--tokens", "128"])
        self.assertEqual(profiles[1]["argv"][-2:], ["--tokens", "131072"])

    def test_failed_profile_stops_before_collecting_or_running_another_workload(self) -> None:
        step = {"config": "profile.json", "label": "candidate", "workload": "prefill",
                "argv": ["profile.py"], "out": "trace"}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(lab, "ROOT", Path(directory)), \
                mock.patch.object(lab, "profiler_dir", return_value="/cache/profile"), \
                mock.patch.object(lab, "nodes_config", return_value={"nodes": [{"name": "dgx1"}]}), \
                mock.patch.object(lab, "production_configuration", return_value=({}, {}, {})), \
                mock.patch.object(lab.spark, "repository_path", return_value="/repo"), \
                mock.patch.object(lab.spark, "run_ssh", return_value=mock.Mock(returncode=0)), \
                mock.patch.object(lab.subprocess, "run", return_value=mock.Mock(returncode=1)) as run:
            with self.assertRaisesRegex(RuntimeError, "prefill failed with exit 1"):
                lab.profile_workload(step)
            self.assertEqual(run.call_count, 1)


class QueueTest(unittest.TestCase):
    def test_add_validates_and_orders_specs(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(lab, "QUEUE", Path(directory) / "queue"), \
                mock.patch.object(lab, "arm_problems", return_value=[]):
            good = Path(directory) / "a.json"
            good.write_text(json.dumps(SPEC))
            with contextlib.redirect_stdout(io.StringIO()) as log:
                lab.queue_add(str(good))
            self.assertIn("QUEUE added", log.getvalue())
            bad = Path(directory) / "b.json"
            bad.write_text(json.dumps(dict(SPEC, jobs=[{"kind": "mystery"}])))
            with self.assertRaises(SystemExit):
                lab.queue_add(str(bad))
            self.assertEqual([p.name.split("-", 1)[1] for p in lab.queued()], ["a.json"])

    def test_sync_is_a_job(self) -> None:
        steps = lab.plan(dict(SPEC, jobs=[{"kind": "sync"}]))
        self.assertEqual([s["kind"] for s in steps], ["sync"])

    def test_gpu_clocks_jobs(self) -> None:
        steps = lab.plan(dict(SPEC, jobs=[{"kind": "gpu_clocks", "lock_mhz": [2400, 2400]},
                                          {"kind": "gpu_clocks", "reset": True}]))
        self.assertEqual([s["lock"] for s in steps], [[2400, 2400], None])
        self.assertEqual(lab.describe_step(steps[0]), "lock GPU SM clocks to 2400-2400 MHz on every node")
        self.assertEqual(lab.describe_step(steps[1]), "reset GPU clocks on every node")
        for bad in ([2400, 2000], [0, 2400]):
            with self.assertRaises(SystemExit):
                lab.plan(dict(SPEC, jobs=[{"kind": "gpu_clocks", "lock_mhz": bad}]))


class NewCommitTest(unittest.TestCase):
    def test_a_sync_to_another_commit_restarts_the_job(self) -> None:
        """The steps after the sync never run on the code this process loaded; a sync that
        leaves the commit where it was continues the job."""
        steps = [{"kind": "sync"}, {"kind": "boot", "config": "arm.json", "label": "arm"}]

        def run(commit_after_sync):
            with mock.patch.object(lab, "plan", return_value=steps), \
                    mock.patch.object(lab, "read_hold", return_value={}), \
                    mock.patch.object(lab, "hold_is_ours", return_value=True), \
                    mock.patch.object(lab, "window_should_close", return_value=None), \
                    mock.patch.object(lab, "production_configuration",
                                      return_value=({"host": {"home": "/h"}}, None, None)), \
                    mock.patch.object(lab, "beat"), \
                    mock.patch.object(lab, "sync_checkouts", return_value=True), \
                    mock.patch.object(lab, "checkout_commit", return_value=commit_after_sync), \
                    mock.patch.object(lab, "COMMIT_AT_START", "loaded"), \
                    mock.patch.object(lab, "plan_problems", return_value=[]), \
                    mock.patch.object(lab.spark, "read_cluster", side_effect=RuntimeError("boot step")), \
                    mock.patch.object(lab, "boot") as boot, \
                    mock.patch.object(lab, "window_close") as close, \
                    contextlib.redirect_stdout(io.StringIO()):
                code = lab.execute({"run": "r"}, dry=False, keep_open=True, close_on_failure=False)
            return code, boot, close

        code, boot, close = run("synced")
        self.assertEqual(code, lab.RESTART_ON_NEW_COMMIT)
        boot.assert_not_called()
        close.assert_not_called()
        code, _, _ = run("loaded")  # same commit: the job goes on (to a boot step that fails here)
        self.assertEqual(code, 1)


class PlanProblemsTest(unittest.TestCase):
    """Mistakes that would cost a boot are found from the spec alone."""

    def test_an_arm_cluster_start_would_refuse_is_named(self) -> None:
        # r6o's first gate stopped at its boot: its lock still named r6n's own baseline.
        arm = "experiments/2026-10-16-r6o/r6o-tp3.json"
        real = lab.spark.configuration

        def stale(args):
            cluster, nodes, upstreams = real(args)
            return cluster, nodes, dict(upstreams, generated_from_baseline="2026-10-13-karmic-kraken-r6k")

        with mock.patch.object(lab.spark, "configuration", stale):
            problems = lab.plan_problems([{"kind": "boot", "config": arm}])
        self.assertIn(f"{arm}: cluster and upstream locks name different baselines", problems)
        self.assertNotIn(f"{arm}: cluster and upstream locks name different baselines",
                         lab.plan_problems([{"kind": "boot", "config": arm}]))

    def measure(self, extras: list) -> dict:
        return {"experiment": "experiments/2026-10-14-servebench", "run": "r", "jobs": [
            {"kind": "measure", "bench": False, "extras": extras,
             "arms": [{"config": "experiments/2026-10-14-servebench/engine-r6k.json", "label": "a"}]}]}

    def test_good_servebench_arguments_pass(self) -> None:
        spec = self.measure([["scripts/servebench_run.py", ["--mode", "closed", "--concurrency", "16"], "c16"],
                             ["scripts/servebench_engine.py", ["--replays", "100"], "grid"]])
        self.assertEqual(lab.plan_problems(lab.plan(spec)), [])

    def test_extras_name_each_arm(self) -> None:
        spec = self.measure([["scripts/servebench_engine.py", ["--label", "{label}", "--output", "out/{run}-{label}.json"], "g"]])
        spec["jobs"][0]["arms"].append({"config": "experiments/2026-10-14-servebench/engine-r6h.json", "label": "b"})
        with mock.patch.object(lab, "head_url", return_value="http://h:8000"):
            outputs = [s["argv"][-1] for s in lab.plan(spec) if s["kind"] == "script"]
        self.assertEqual(outputs, ["out/r-a.json", "out/r-b.json"])

    def test_telemetry_switch(self) -> None:
        spec = self.measure([["scripts/servebench_run.py", ["--mode", "closed"], "c"]])
        with mock.patch.object(lab, "head_url", return_value="http://h:8000"):
            on = [s for s in lab.plan(spec) if s["kind"] == "script"]
            spec["jobs"][0]["telemetry"] = False
            off = [s for s in lab.plan(spec) if s["kind"] == "script"]
        self.assertEqual((on[0]["telemetry"], off[0]["telemetry"]), (True, False))

    def test_suite_job(self) -> None:
        spec = {"experiment": "experiments/2026-10-14-servebench", "run": "s", "jobs": [
            {"kind": "measure", "suite": "screen",
             "arms": [{"config": "experiments/2026-10-14-servebench/engine-r6k.json", "label": "cand"},
                      {"config": "experiments/2026-10-14-servebench/engine-r6h.json", "label": "base"}]}]}
        with mock.patch.object(lab, "head_url", return_value="http://h:8000"):
            steps = lab.plan(spec)
            self.assertEqual(lab.plan_problems(steps), [])
        # Each arm runs the suite through bin/spark bench (cooling, fan curve, identity).
        benches = [s for s in steps if s["kind"] == "cli"]
        self.assertEqual(benches[0]["argv"],
                         ["--cluster-config", "experiments/2026-10-14-servebench/engine-r6k.json", "bench",
                          "--suite", "screen", "--compare", "none", "--output", "results/private/suites/s/cand",
                          "--label", "cand"])
        self.assertEqual(benches[1]["argv"][-2:], ["--corpus", "results/private/suites/s/cand/model.json.gz"])
        self.assertEqual(benches[0]["telemetry_out"], "results/private/determinism/s/suite-cand")
        self.assertEqual(steps[-1]["argv"], ["scripts/servebench_suite_table.py", "s", "cand", "base"])
        spec["jobs"][0]["suite"] = "no-such-suite"
        with mock.patch.object(lab, "head_url", return_value="http://h:8000"):
            self.assertTrue(lab.plan_problems(lab.plan(spec)))
        # The engine grid needs the dev-mode collective RPC: refused before any boot.
        spec["jobs"][0]["suite"] = "screen"
        spec["jobs"][0]["arms"][1]["config"] = "experiments/2026-10-15-r6n/r6n-release.json"
        with mock.patch.object(lab, "head_url", return_value="http://h:8000"):
            problems = lab.plan_problems(lab.plan(spec))
        self.assertEqual(problems, ["suite screen times the engine grid, but experiments/2026-10-15-r6n/r6n-release.json "
                                    "lacks VLLM_SERVER_DEV_MODE=1 (the dev-mode collective RPC)"])

    def test_suite_job_with_a_companion_suite(self) -> None:
        """A list of suites: the release suite first, a companion (single-stream) on the same
        boot in its own run directory and table; the release suite's outputs are unchanged."""
        spec = {"experiment": "experiments/2026-10-14-servebench", "run": "s", "jobs": [
            {"kind": "measure", "suite": ["screen", "single-stream"],
             "arms": [{"config": "experiments/2026-10-14-servebench/engine-r6k.json", "label": "cand"},
                      {"config": "experiments/2026-10-14-servebench/engine-r6h.json", "label": "base"}]}]}
        with mock.patch.object(lab, "head_url", return_value="http://h:8000"):
            steps = lab.plan(spec)
            self.assertEqual(lab.plan_problems(steps), [])
        self.assertEqual([s["kind"] for s in steps],
                         ["boot", "curves", "cli", "cli", "boot", "curves", "cli", "cli", "table", "table"])
        benches = [s for s in steps if s["kind"] == "cli"]
        self.assertEqual([b["suite"] for b in benches], ["screen", "single-stream"] * 2)
        self.assertEqual(benches[0]["argv"][-4:], ["--output", "results/private/suites/s/cand", "--label", "cand"])
        self.assertEqual(benches[1]["argv"][-4:],
                         ["--output", "results/private/suites/s-single-stream/cand", "--label", "cand"])
        self.assertEqual(benches[2]["argv"][-2:], ["--corpus", "results/private/suites/s/cand/model.json.gz"])
        self.assertNotIn("--corpus", benches[3]["argv"])
        self.assertEqual(benches[3]["telemetry_out"], "results/private/determinism/s/suite-base-single-stream")
        tables = [s for s in steps if s["kind"] == "table"]
        self.assertEqual([t["argv"][1] for t in tables], ["s", "s-single-stream"])
        self.assertEqual([t["out"] for t in tables],
                         ["results/private/lab/s-table.txt", "results/private/lab/s-single-stream-table.txt"])

    def test_bad_arguments_missing_files_and_configs(self) -> None:
        spec = self.measure([["scripts/servebench_run.py", ["--mode", "sideways"], "c16"],
                             ["no_such_script.py", [], "x"]])
        spec["jobs"][0]["arms"].append({"config": "missing.json", "label": "b"})
        problems = lab.plan_problems(lab.plan(spec))
        self.assertTrue(any("servebench_run.py" in p and "invalid choice" in p for p in problems), problems)
        self.assertTrue(any("no_such_script.py does not exist" in p for p in problems), problems)
        self.assertTrue(any("missing.json does not exist" in p for p in problems), problems)

    def test_queue_rejects_without_a_sync_and_warns_with_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(lab, "QUEUE", Path(directory) / "q"):
            bad = self.measure([["scripts/servebench_run.py", ["--mode", "sideways"], "c16"]])
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(bad))
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                lab.queue_add(str(path))
            path.write_text(json.dumps(dict(bad, jobs=[{"kind": "sync"}, *bad["jobs"]])))
            with contextlib.redirect_stderr(io.StringIO()) as warned, contextlib.redirect_stdout(io.StringIO()):
                lab.queue_add(str(path))
            self.assertIn("checked again after the run's sync", warned.getvalue())


class TelemetryTest(unittest.TestCase):
    """Node telemetry around a measured step: summarized, saved, never fatal."""

    LINES = "".join(f"2026/10/14 07:00:{i:02d}.000, {2400 + i}, 4266, {50 + i}.0, 60, 0x0000000000000000\n"
                    for i in range(10)) + "2026/10/14 07:00:10.000, 1100, 4266, 30.0, 61, 0x0000000000000004\n"

    def test_summary(self) -> None:
        summary = lab.summarize_telemetry(self.LINES + "garbage line\n")
        self.assertEqual(summary["samples"], 11)
        self.assertEqual(summary["clocks.sm"]["min"], 1100)
        self.assertEqual(summary["clocks.sm"]["max"], 2409)
        self.assertEqual(summary["event_reasons"], {"0x0000000000000000": 10, "0x0000000000000004": 1})

    def test_step_writes_beside_its_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = mock.Mock()
            process.communicate.return_value = (self.LINES, "")
            marker = Path(directory) / "locked.json"
            marker.write_text(json.dumps({"lock_mhz": [2400, 2400]}))
            with mock.patch.object(lab, "ROOT", Path(directory)), \
                    mock.patch.object(lab, "GPU_CLOCKS_MARKER", marker), \
                    mock.patch.object(lab, "nodes_config", return_value={"ssh_user": "u", "nodes": [{"name": "n1"}, {"name": "n2"}]}), \
                    mock.patch.object(lab.spark, "ssh_options", return_value=[]), \
                    mock.patch.object(lab.subprocess, "Popen", return_value=process) as popen:
                with lab.Telemetry("results/x/c16-arm.jsonl"):
                    pass
            saved = json.loads((Path(directory) / "results/x/c16-arm.jsonl.telemetry.json").read_text())
        self.assertEqual(sorted(saved["nodes"]), ["n1", "n2"])
        self.assertEqual(saved["gpu_clocks_lock"], {"lock_mhz": [2400, 2400]})
        self.assertEqual(saved["nodes"]["n1"]["samples"], 11)
        self.assertEqual(popen.call_count, 4)  # clocks and threads on each node
        commands = [call.args[0] for call in popen.call_args_list]
        self.assertTrue(any("nvidia-smi" in c[-1] for c in commands))
        self.assertTrue(any(c[-4:-2] == ["python3", "-"] for c in commands))  # the thread sampler
        process.terminate.assert_called()

    def test_thread_summary(self) -> None:
        lines = [json.dumps({"fast": [5, 6]})]
        for step in range(3):
            lines.append(json.dumps({"t": 100.0 + 2 * step, "threads": {
                "11": ["VLLM::Worker_TP", 1.8 * step, 5],
                "12": ["engram", 0.5 * step, 1],
                "13": ["idle", 0.01 * step, 2]}}))
        summary = lab.summarize_threads("\n".join(lines))
        self.assertEqual(summary["window_s"], 4.0)
        self.assertEqual([(t["name"], t["core_share"], t["slow_share"]) for t in summary["busy_threads"]],
                         [("VLLM::Worker_TP", 0.9, 0.0), ("engram", 0.25, 1.0)])
        self.assertAlmostEqual(summary["slow_core_share"], 1.02 / 4.62, places=3)
        self.assertEqual(lab.summarize_threads("not json")["samples"], 0)

    def test_buddy_free(self) -> None:
        text = ("Node 0, zone      DMA      2      0      0      0      0      1\n"
                "Node 0, zone   Normal      4      0      0      0      0      3\n")
        free = lab.buddy_free(text, 65536)
        self.assertEqual(free["free_gib_2mib_blocks"], round(4 * 2 ** 21 / 2 ** 30, 3))
        self.assertEqual(free["free_gib_by_block"]["64 KiB"], round(6 * 65536 / 2 ** 30, 3))

    def test_boot_step_memory_options(self) -> None:
        self.assertNotIn("memory_out", lab.boot_step("r", {"label": "a"}, "c.json"))
        step = lab.boot_step("r", {"label": "a", "compact_memory": True}, "c.json")
        self.assertEqual((step["memory_out"], step["compact"]), ("results/private/determinism/r/memory-a.json", True))
        self.assertIn("compact memory first", lab.describe_step(step))
        self.assertIn("memory snapshot first", lab.describe_step(lab.boot_step("r", {"label": "b", "memory_snapshot": True}, "c.json")))

    def test_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(lab, "ROOT", Path(directory)):
            run = Path(directory) / "results/private/determinism/r"
            run.mkdir(parents=True)
            (Path(directory) / "results/private/lab").mkdir(parents=True)
            (Path(directory) / "results/private/lab/r-table.txt").write_text("TABLE\n")
            (run / "suite-a.jsonl.telemetry.json").write_text(json.dumps({
                "seconds": 12.0, "gpu_clocks_lock": {"lock_mhz": [2489, 2489]},
                "nodes": {"n1": lab.summarize_telemetry(self.LINES)}}))
            text = lab.report("r")
        self.assertIn("TABLE", text)
        self.assertIn("suite-a.jsonl: 12 s, clocks locked to 2489-2489 MHz", text)
        self.assertIn("n1: SM 1100/2404/2408 MHz (min 1100), 54 W, 61 °C max", text)
        self.assertIn("clock events {'0x0000000000000004': 1}", text)

    def test_real_processes_close_cleanly(self) -> None:
        """Real pipes, as on the cluster: the sampler's stdin is written and closed at start."""
        real = subprocess.Popen

        def local(command, **kwargs):
            script = "import sys; sys.stdin.read(); print('2026/10/14 07:00:00.000, 2400, 4266, 40.0, 60, 0x0')"
            return real([sys.executable, "-c", script], **kwargs)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(lab, "ROOT", Path(directory)), mock.patch.object(lab, "log"), \
                mock.patch.object(lab, "GPU_CLOCKS_MARKER", Path(directory) / "none.json"), \
                mock.patch.object(lab, "nodes_config", return_value={"ssh_user": "u", "nodes": [{"name": "n1"}]}), \
                mock.patch.object(lab.spark, "ssh_options", return_value=[]), \
                mock.patch.object(lab.subprocess, "Popen", side_effect=local):
            with lab.Telemetry("results/z/out.jsonl") as telemetry:
                time.sleep(0.3)
                # Python 3.12's communicate() flushes a closed stdin and raises (3.13 does not):
                # the sampler's stdin must be released once the script is sent.
                self.assertTrue(all(process.stdin is None for process in telemetry.threads.values()))
            saved = json.loads((Path(directory) / "results/z/out.jsonl.telemetry.json").read_text())
        self.assertNotIn("error", saved["nodes"]["n1"])
        self.assertIn("threads", saved["nodes"]["n1"])

    def test_engine_logs_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(lab, "ROOT", Path(directory)), \
                mock.patch.object(lab, "log"), \
                mock.patch.object(lab, "nodes_config", return_value={"nodes": [{"name": "n1"}, {"name": "n2"}]}), \
                mock.patch.object(lab.spark, "run_ssh", return_value=mock.Mock(stdout="Traceback: boom\n", stderr="")):
            lab.save_engine_logs("results/r/goodput.jsonl")
            saved = (Path(directory) / "results/r/goodput.jsonl.engine-n2.log").read_text()
        self.assertIn("boom", saved)

    def test_never_fails_the_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(lab, "ROOT", Path(directory)), mock.patch.object(lab, "log"), \
                mock.patch.object(lab, "nodes_config", side_effect=RuntimeError("no map")):
            with lab.Telemetry("results/y/out.jsonl"):
                ran = True
            self.assertTrue(ran)
            self.assertTrue((Path(directory) / "results/y/out.jsonl.telemetry.json").exists())


class GpuClocksTest(unittest.TestCase):
    """A lock applies to every node or to none, and restoring production resets it."""

    NODES = {"nodes": [{"name": "n1"}, {"name": "n2"}]}

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.patches = [mock.patch.object(lab, "GPU_CLOCKS_MARKER", Path(self.directory.name) / "locked.json"),
                        mock.patch.object(lab, "nodes_config", return_value=self.NODES),
                        mock.patch.object(lab, "log")]
        for patch in self.patches:
            patch.start()
        self.calls = []

    def tearDown(self) -> None:
        for patch in self.patches:
            patch.stop()
        self.directory.cleanup()

    def ssh(self, fail_on: str | None = None):
        def run(nodes, node, *command):
            self.calls.append((node["name"], command))
            failed = fail_on == node["name"] and "-lgc" in command
            return mock.Mock(returncode=1 if failed else 0, stdout="2400 MHz, 3003 MHz", stderr="no")
        return run

    def test_lock_and_reset(self) -> None:
        with mock.patch.object(lab.spark, "run_ssh", self.ssh()):
            self.assertTrue(lab.set_gpu_clocks([2400, 2400]))
            self.assertTrue(lab.GPU_CLOCKS_MARKER.exists())
            self.assertTrue(lab.set_gpu_clocks(None))
        self.assertFalse(lab.GPU_CLOCKS_MARKER.exists())
        sudo = [command for _, command in self.calls if command[0] == "sudo"]
        self.assertEqual(sudo, [("sudo", "-n", "nvidia-smi", "-lgc", "2400,2400")] * 2
                         + [("sudo", "-n", "nvidia-smi", "-rgc")] * 2)

    def test_partial_lock_resets_every_node(self) -> None:
        with mock.patch.object(lab.spark, "run_ssh", self.ssh(fail_on="n2")):
            self.assertFalse(lab.set_gpu_clocks([2400, 2400]))
        self.assertFalse(lab.GPU_CLOCKS_MARKER.exists())
        resets = [name for name, command in self.calls if command[-1] == "-rgc"]
        self.assertEqual(resets, ["n1", "n2"])

    def test_restore_resets_a_lock_first(self) -> None:
        lab.GPU_CLOCKS_MARKER.write_text("{}")
        with mock.patch.object(lab, "set_gpu_clocks") as clocks, \
                mock.patch.object(lab, "production_live", return_value=True):
            self.assertTrue(lab.restore_production())
        clocks.assert_called_once_with(None)


class QueueFailureTest(unittest.TestCase):
    """A failed job costs only itself in the queue; the window still ends on its own reasons."""

    def _execute(self, close_on_failure: bool, should_close=None):
        closed = []
        spec = {"run": "r", "experiment": "experiments/x", "jobs": []}
        step = {"kind": "kernel", "bundles": [], "out": "results/private/lab/r"}
        with mock.patch.object(lab, "plan", return_value=[step]), \
                mock.patch.object(lab, "read_hold", return_value={"holder": "spark-lab: queue"}), \
                mock.patch.object(lab, "hold_is_ours", return_value=True), \
                mock.patch.object(lab, "window_should_close", return_value=should_close), \
                mock.patch.object(lab, "production_configuration",
                                  return_value=({"host": {"home": "/h"}}, [], None)), \
                mock.patch.object(lab, "beat"), mock.patch.object(lab, "log"), \
                mock.patch.object(lab, "run_kernel_bundles", side_effect=RuntimeError("boom")), \
                mock.patch.object(lab, "window_close", side_effect=lambda reason, dry=False: closed.append(reason)):
            code = lab.execute(spec, dry=False, keep_open=True, close_on_failure=close_on_failure)
        return code, closed

    def test_queue_job_failure_keeps_the_window(self) -> None:
        code, closed = self._execute(close_on_failure=False)
        self.assertEqual(code, 1)
        self.assertEqual(closed, [])

    def test_run_failure_still_closes_the_window(self) -> None:
        code, closed = self._execute(close_on_failure=True)
        self.assertEqual(code, 1)
        self.assertEqual(len(closed), 1)
        self.assertIn("raised RuntimeError", closed[0])

    def test_window_reasons_close_even_in_the_queue(self) -> None:
        code, closed = self._execute(close_on_failure=False, should_close="cluster requested (x)")
        self.assertEqual(code, 1)
        self.assertEqual(closed, ["cluster requested (x)"])

    def test_queue_runs_the_next_spec_after_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue"
            queue.mkdir()
            for name in ("20260101T000000-a.json", "20260101T000001-b.json"):
                (queue / name).write_text(json.dumps({"run": name, "jobs": []}))
            ran, closed = [], []

            def execute(spec, dry, keep_open, close_on_failure=True):
                ran.append((spec["run"], close_on_failure))
                return 1 if len(ran) == 1 else 0

            idle = iter([None, None, None, "time cap reached"])
            with mock.patch.object(lab, "QUEUE", queue), \
                    mock.patch.object(lab, "window_open"), \
                    mock.patch.object(lab, "read_hold", return_value={"holder": "spark-lab: queue"}), \
                    mock.patch.object(lab, "hold_is_ours", return_value=True), \
                    mock.patch.object(lab, "window_should_close", side_effect=lambda hold: next(idle)), \
                    mock.patch.object(lab, "execute", side_effect=execute), \
                    mock.patch.object(lab, "beat"), mock.patch.object(lab, "log"), \
                    mock.patch.object(lab.time, "sleep"), \
                    mock.patch.object(lab, "window_close", side_effect=lambda reason, dry=False: closed.append(reason)):
                code = lab.queue_run(idle_minutes=5, minutes=60)
            self.assertEqual(code, 0)
            self.assertEqual(ran, [("20260101T000000-a.json", False), ("20260101T000001-b.json", False)])
            self.assertEqual([p.name for p in (queue / "failed").iterdir()], ["20260101T000000-a.json"])
            self.assertEqual([p.name for p in (queue / "done").iterdir()], ["20260101T000001-b.json"])
            self.assertEqual(closed, ["time cap reached"])


    def test_an_idle_queue_holds_the_window_without_an_idle_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory) / "queue"
            queue.mkdir()
            closed, logs = [], []
            # Idle polls a day apart, then a request: only the request closes the window.
            clock = iter(1_000_000.0 + 86400 * i for i in range(100))
            reasons = iter([None, None, None, "cluster requested (x)"])
            with mock.patch.object(lab, "QUEUE", queue), \
                    mock.patch.object(lab, "window_open"), \
                    mock.patch.object(lab, "read_hold", return_value={"holder": "spark-lab: queue"}), \
                    mock.patch.object(lab, "hold_is_ours", return_value=True), \
                    mock.patch.object(lab, "window_should_close", side_effect=lambda hold: next(reasons)), \
                    mock.patch.object(lab, "execute") as execute, \
                    mock.patch.object(lab, "beat"), mock.patch.object(lab, "log", side_effect=logs.append), \
                    mock.patch.object(lab.time, "sleep"), mock.patch.object(lab.time, "time", side_effect=clock), \
                    mock.patch.object(lab, "window_close", side_effect=lambda reason, dry=False: closed.append(reason)):
                code = lab.queue_run(idle_minutes=0, minutes=0)
            self.assertEqual(code, 0)
            execute.assert_not_called()
            self.assertEqual(closed, ["cluster requested (x)"])
            self.assertIn("QUEUE idle (held for the next spec)", logs)


class CustomWorkloadTest(unittest.TestCase):
    def test_job_workloads_may_name_a_repository_script(self) -> None:
        job = {"kind": "measure", "arms": [{"config": "cluster-a.json", "label": "a"}],
               "extras": [["scripts/distinct_streams.py", ["--streams", "4"], "c4-distinct"],
                          ["own.py", [], "own"]]}
        scripts = [s["argv"][0] for s in lab.measure_steps({"experiment": "experiments/x", "run": "r"}, job)
                   if s["kind"] == "script"]
        self.assertEqual(scripts, ["scripts/distinct_streams.py", "experiments/x/own.py"])

    def test_a_job_may_skip_the_bench_and_name_its_own_workloads(self) -> None:
        spec = {"experiment": "experiments/x", "run": "s1", "jobs": [{
            "kind": "measure", "bench": False,
            "extras": [["hol_latency.py", ["--rounds", "2"], "hol"]],
            "arms": [{"config": "cluster-a.json", "label": "a"}]}]}
        steps = lab.plan(spec)
        self.assertEqual([s["kind"] for s in steps], ["boot", "curves", "script", "table"])
        self.assertEqual(steps[2]["out"], "results/private/determinism/s1/hol-a.jsonl")
        self.assertEqual(steps[3]["summaries"], ["results/private/determinism/s1/hol-a.jsonl"])


class CurvesTest(unittest.TestCase):
    def test_curve_table_compares_arms_at_fixed_token_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(lab, "ROOT", Path(directory)):
            base = Path(directory) / "curves-r5o.json"
            base.write_text(json.dumps({"verify": [[1, 10.0], [6, 12.0], [48, 20.0]], "draft": [[1, 2.0]]}))
            cand = Path(directory) / "curves-cand.json"
            cand.write_text(json.dumps({"verify": [[1, 11.0], [6, 12.0], [48, 19.0]], "draft": [[1, 2.0]]}))
            table = lab.curves_table(["curves-r5o.json", "curves-cand.json", "curves-missing.json"])
        self.assertIn("1:11.00 (+10.0%)", table)
        self.assertIn("48:19.00 (-5.0%)", table)
        self.assertEqual(table.count("\n"), 2)

    def test_padded_and_real_row_curves_never_compare(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(lab, "ROOT", Path(directory)):
            (Path(directory) / "curves-a.json").write_text(json.dumps({"verify": [[1, 10.0]], "draft": []}))
            (Path(directory) / "curves-b.json").write_text(
                json.dumps({"rows": "real", "verify": [[1, 15.0]], "draft": []}))
            (Path(directory) / "curves-c.json").write_text(
                json.dumps({"rows": "real", "verify": [[1, 16.5]], "draft": []}))
            table = lab.curves_table(["curves-a.json", "curves-b.json", "curves-c.json"])
        self.assertIn("b              real   1:15.00", table)
        self.assertNotIn("+50.0%", table)
        self.assertIn("1:16.50 (+10.0%)", table)


class OverlayTest(unittest.TestCase):
    def test_fewer_fences_than_the_image_is_a_problem(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / "stale.py"
            stale.write_text("x = 1\n")
            fresh = Path(directory) / "fresh.py"
            fresh.write_text("cute.arch.fence_proxy('async.shared')\n")
            image = Path(directory) / "image.py"
            image.write_text("cute.arch.fence_proxy('async.shared')\n")
            mounts = [[str(stale), "/opt/spark3/candidate/b12x/b12x/gemm/a.py", "ro"],
                      [str(fresh), "/opt/spark3/candidate/b12x/b12x/gemm/b.py", "ro"],
                      [str(stale), "/opt/spark3/candidate/vllm/vllm/x.py", "ro"]]
            problems = lab.fence_problems(mounts, "img", "/home/x", read_image=lambda path: image)
        self.assertEqual(len(problems), 1)
        self.assertIn("stale.py", problems[0])


class KernelLabTest(unittest.TestCase):
    CANDIDATE = {"image": "img:tag", "env": {"B12X_AUTOTUNE": "0"}, "argv": ["/r/replay.py", "/cap"],
                 "mounts": [["~/captures", "/cap"]]}

    def test_overlay_files_mount_at_the_candidate_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            target = bundle / "overlay" / "b12x" / "norm" / "mhc" / "_kernels.py"
            target.parent.mkdir(parents=True)
            target.write_text("")
            vllm = bundle / "overlay" / "vllm" / "models" / "deepseek_v4_1" / "b12x_layers.py"
            vllm.parent.mkdir(parents=True)
            vllm.write_text("")
            command = lab.bundle_command(bundle, self.CANDIDATE, bundle / "out")
        joined = " ".join(command)
        self.assertIn(":/opt/spark3/candidate/b12x/b12x/norm/mhc/_kernels.py:ro", joined)
        self.assertIn(":/opt/spark3/candidate/vllm/vllm/models/deepseek_v4_1/b12x_layers.py:ro", joined)
        self.assertIn("B12X_AUTOTUNE=0", command)
        self.assertEqual(command[-3:], ["img:tag", "/r/replay.py", "/cap"])

    def test_display_carveout_bundles_get_the_serving_drm_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            plain = lab.bundle_command(bundle, self.CANDIDATE, bundle / "out")
            carveout = lab.bundle_command(bundle, dict(self.CANDIDATE, display_carveout=True), bundle / "out")
        self.assertFalse([arg for arg in plain if "dri" in arg or "226" in arg])
        serving = lab.spark.read_cluster("config/cluster.json", "tp3-triangle")["container"]["docker_run_args"]
        for arg in ("--mount=type=bind,source=/dev/dri/by-path/pci-000f:01:00.0-card,target=/dev/dri/card0",
                    "--device-cgroup-rule=c 226:* rw"):
            self.assertIn(arg, carveout)
            self.assertIn(arg, serving)

    def test_io_uring_bundles_get_the_serving_io_uring_allowance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            plain = lab.bundle_command(bundle, self.CANDIDATE, bundle / "out")
            uring = lab.bundle_command(bundle, dict(self.CANDIDATE, io_uring=True), bundle / "out")
        self.assertFalse([arg for arg in plain if "seccomp" in arg or "memlock" in arg])
        serving = json.loads((ROOT / "config/cluster.json").read_text())["container"]["docker_run_args"]
        for arg in ("--security-opt=seccomp=unconfined", "--ulimit=memlock=-1:-1"):
            self.assertIn(arg, uring)
            self.assertIn(arg, serving)

    def test_verdict_passes_only_invariant_and_bit_equal(self) -> None:
        good = [json.dumps({"op": "mhc production layer 1 pre", "groups": 2}),
                json.dumps({"op": "mhc candidate layer 1 pre", "groups": 1}),
                json.dumps({"bits": "tf32x-a vs tf32-s40 layer 1 pre", "equal_at": [1, 2], "differs_at": []}),
                json.dumps({"timing_us": "candidate layer 1 pre", "1": 10.0})]
        self.assertTrue(lab.verdict_from_lines(good, self.CANDIDATE)["passed"])
        variant = good + [json.dumps({"op": "mhc tf32x-b layer 1 pre", "groups": 2})]
        self.assertEqual(lab.verdict_from_lines(variant, self.CANDIDATE)["batch_variant"],
                         ["mhc tf32x-b layer 1 pre"])
        unequal = good + [json.dumps({"bits": "x vs y", "equal_at": [1], "differs_at": [2]})]
        self.assertFalse(lab.verdict_from_lines(unequal, self.CANDIDATE)["passed"])
        crashed = good + ["Traceback (most recent call last):"]
        self.assertFalse(lab.verdict_from_lines(crashed, self.CANDIDATE)["passed"])
        self.assertFalse(lab.verdict_from_lines([], self.CANDIDATE)["passed"])

    def test_suite_bundles_pass_on_passing_tests_only(self) -> None:
        suite = dict(self.CANDIDATE, verdict="exit")
        good = ["tests/kernels/moe/test_x.py ....", "===== 18 passed, 1 warning in 40.2s ====="]
        verdict = lab.verdict_from_lines(good, suite)
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["summary"], "18 passed, 1 warning in 40.2s")
        failed = ["FAILED tests/kernels/moe/test_x.py::test_a[1] - AssertionError: rows differ",
                  "===== 1 failed, 17 passed in 41.0s ====="]
        verdict = lab.verdict_from_lines(failed, suite)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["failed"], ["tests/kernels/moe/test_x.py::test_a[1]"])
        for lines in ([], ["===== 3 skipped in 1.0s ====="], ["===== 2 passed, 1 error in 3.0s ====="]):
            with self.subTest(lines=lines):
                self.assertFalse(lab.verdict_from_lines(lines, suite)["passed"])
        a = dict(lab.verdict_from_lines(failed, suite), bundle="b")
        self.assertFalse(lab.verdicts_agree([a, dict(lab.verdict_from_lines(good, suite), bundle="b")]))
        self.assertTrue(lab.verdicts_agree([a, dict(a, node="dgx4")]))

    def test_nodes_must_agree_on_groups_and_bits(self) -> None:
        a = {"bundle": "b", "groups": [{"op": "x", "groups": 1}], "bits": [{"bits": "p", "differs_at": []}]}
        b = dict(a, node="dgx2")
        self.assertTrue(lab.verdicts_agree([a, b]))
        c = dict(a, groups=[{"op": "x", "groups": 2}])
        self.assertFalse(lab.verdicts_agree([a, c]))


class AnalysisTest(unittest.TestCase):
    def test_summary_counts_rows_and_cross_boot_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            records = [
                {"request": "a-r0", "against": "a-r0-bootB", "rows_compared": 10, "rows_differing": 0,
                 "outputs_equal": True},
                {"request": "a-r0", "against": "a-r1", "rows_compared": 5, "rows_differing": 0,
                 "outputs_equal": True},
                {"request": "a-r0", "rows": 12, "recomputed_mismatch": []},
            ]
            (out / "analysis4-scenarios-dgx1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            summary = lab.summarize_analysis(out)
            self.assertTrue(summary["passed"])
            entry = summary["analysis4-scenarios-dgx1"]
            self.assertEqual((entry["pairs"], entry["rows"], entry["cross_boot_pairs"]), (2, 15, 1))
            records[1]["rows_differing"] = 3
            (out / "analysis4-scenarios-dgx1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            self.assertFalse(lab.summarize_analysis(out)["passed"])

    def test_no_analysis_files_is_not_a_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(lab.summarize_analysis(Path(directory))["passed"])


if __name__ == "__main__":
    unittest.main()


class TransferRouteTest(unittest.TestCase):
    def test_bulk_copies_prefer_the_transfer_host(self) -> None:
        nodes = {"ssh_user": "swank"}
        target = lab.spark.transfer_target
        self.assertEqual(target(nodes, {"name": "dgx3", "transfer_host": "dgx3-cx7"}), "swank@dgx3-cx7")
        self.assertEqual(target(nodes, {"name": "dgx4", "ssh_host": "10.0.1.74"}), "swank@10.0.1.74")
        self.assertEqual(target(nodes, {"name": "dgx2"}), "swank@dgx2")


class StepFailureTest(unittest.TestCase):
    def test_a_step_that_raises_fails_the_run_and_closes_the_window(self) -> None:
        spec = {"experiment": "experiments/2026-09-29-determinism", "run": "t0",
                "jobs": [{"kind": "kernel", "bundles": [{"bundle": "b/mhc", "node": "dgx2"}]}]}
        with mock.patch.object(lab, "read_hold", return_value={"holder": "test"}), \
                mock.patch.object(lab, "hold_is_ours", return_value=True), \
                mock.patch.object(lab, "window_should_close", return_value=None), \
                mock.patch.object(lab, "beat"), \
                mock.patch.object(lab, "production_configuration", return_value=({"host": {"home": "/h"}}, {}, {})), \
                mock.patch.object(lab, "container_running", return_value=False), \
                mock.patch.object(lab, "run_kernel_bundles", side_effect=RuntimeError("cannot read dgx3")), \
                mock.patch.object(lab, "window_close", return_value=True) as close, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = lab.execute(spec, dry=False, keep_open=True)
        self.assertEqual(code, 1)
        close.assert_called_once()
        self.assertIn("raised RuntimeError: cannot read dgx3", close.call_args.args[0])


    def test_a_failing_bench_ends_the_job_before_its_scripts(self) -> None:
        spec = {"experiment": "experiments/x", "run": "t1",
                "jobs": [{"kind": "measure", "profile": "lean", "arms": [{"config": "a.json", "label": "a"}]}]}
        with mock.patch.object(lab, "read_hold", return_value={"holder": "test"}), \
                mock.patch.object(lab, "hold_is_ours", return_value=True), \
                mock.patch.object(lab, "window_should_close", return_value=None), \
                mock.patch.object(lab, "beat"), \
                mock.patch.object(lab, "fence_problems", return_value=[]), \
                mock.patch.object(lab, "overlay_manifest", return_value=[]), \
                mock.patch.object(lab, "boot", return_value=True), \
                mock.patch.object(lab, "save_curves"), \
                mock.patch.object(lab, "spark_cli", return_value=1), \
                mock.patch.object(lab, "plan_problems", return_value=[]), \
                mock.patch.object(lab, "Telemetry", return_value=contextlib.nullcontext()), \
                mock.patch.object(lab, "run_script") as script, \
                mock.patch.object(lab.spark, "read_cluster", return_value={"container": {"mounts": [], "image": "i"}}), \
                mock.patch.object(lab, "production_configuration", return_value=(
                    {"host": {"home": "/h"}}, {"nodes": [{"name": "dgx1", "head": True, "management_ip": "1"}]}, {})), \
                mock.patch.object(lab, "window_close", return_value=True) as close, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = lab.execute(spec, dry=False, keep_open=True)
        self.assertEqual(code, 1)
        script.assert_not_called()
        self.assertIn("bench a exited 1", close.call_args.args[0])

    def test_a_failed_boot_keeps_every_node_s_log(self) -> None:
        spec = {"experiment": "experiments/x", "run": "t2",
                "jobs": [{"kind": "measure", "profile": "lean", "arms": [{"config": "a.json", "label": "a"}]}]}
        with mock.patch.object(lab, "read_hold", return_value={"holder": "test"}), \
                mock.patch.object(lab, "hold_is_ours", return_value=True), \
                mock.patch.object(lab, "window_should_close", return_value=None), \
                mock.patch.object(lab, "beat"), \
                mock.patch.object(lab, "fence_problems", return_value=[]), \
                mock.patch.object(lab, "overlay_manifest", return_value=[]), \
                mock.patch.object(lab, "boot", return_value=False), \
                mock.patch.object(lab, "save_engine_logs") as saved, \
                mock.patch.object(lab, "plan_problems", return_value=[]), \
                mock.patch.object(lab.spark, "read_cluster", return_value={"container": {"mounts": [], "image": "i"}}), \
                mock.patch.object(lab, "production_configuration", return_value=(
                    {"host": {"home": "/h"}}, {"nodes": [{"name": "dgx1", "head": True, "management_ip": "1"}]}, {})), \
                mock.patch.object(lab, "window_close", return_value=True) as close, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = lab.execute(spec, dry=False, keep_open=True)
        self.assertEqual(code, 1)
        saved.assert_called_once_with("results/private/lab/t2/boot-a")
        self.assertIn("boot of experiments/x/a.json failed", close.call_args.args[0])

    def test_the_lean_bench_does_not_override_its_live_check(self) -> None:
        self.assertNotIn("--allow-mismatch", lab.BENCH_BASE)


class AnalysisFailureTest(unittest.TestCase):
    """A validate run whose analysis found differences fails the run, like a raised step.

    ``execute`` collects ``failed`` only where a handler breaks, so an analyze step that only
    logged PASSED/DIFFERENCES left a run that found differences exiting 0 (and its queue job
    recorded as done). The analyzer's own pass flag is the evidence; it is not a self-report.
    """

    def _execute(self, passed, close_on_failure=True, later_step=True):
        steps = [{"kind": "analyze", "out": "results/private/lab/r", "dirs": ["c8"], "config": "c.json"}]
        if later_step:
            steps.append({"kind": "sync"})
        closed, ran = [], []
        with mock.patch.object(lab, "plan", return_value=steps), \
                mock.patch.object(lab, "read_hold", return_value={"holder": "test"}), \
                mock.patch.object(lab, "hold_is_ours", return_value=True), \
                mock.patch.object(lab, "window_should_close", return_value=None), \
                mock.patch.object(lab, "beat"), \
                mock.patch.object(lab, "production_configuration", return_value=({"host": {"home": "/h"}}, [], None)), \
                mock.patch.object(lab, "analyze", return_value={"passed": passed}), \
                mock.patch.object(lab, "sync_checkouts", side_effect=lambda: ran.append("sync") or True), \
                mock.patch.object(lab, "window_close",
                                  side_effect=lambda reason, dry=False: closed.append(reason)):
            code = lab.execute({"experiment": "experiments/x", "run": "r", "jobs": []}, dry=False,
                               keep_open=True, close_on_failure=close_on_failure)
        return code, closed, ran

    def test_analysis_differences_fail_the_run(self) -> None:
        code, closed, _ = self._execute(passed=False)
        self.assertEqual(code, 1)
        self.assertEqual(len(closed), 1)
        self.assertIn("found differences", closed[0])

    def test_analysis_differences_stop_the_run_before_the_next_step(self) -> None:
        # Without the break the run would exit 1 only at the end, having already run the steps
        # that follow a validation nobody should trust.
        _, _, ran = self._execute(passed=False)
        self.assertEqual(ran, [])

    def test_analysis_differences_keep_the_window_open_for_the_queue(self) -> None:
        code, closed, _ = self._execute(passed=False, close_on_failure=False)
        self.assertEqual(code, 1)
        self.assertEqual(closed, [])

    def test_a_passing_analysis_lets_the_run_continue(self) -> None:
        code, closed, ran = self._execute(passed=True)
        self.assertEqual(code, 0)
        self.assertEqual(closed, [])
        self.assertEqual(ran, ["sync"])


class FabricTest(unittest.TestCase):
    def test_plan_stops_serving_then_runs_on_every_node(self):
        steps = lab.plan({"experiment": "experiments/x", "run": "r", "jobs": [
            {"kind": "fabric", "config": "arm.json", "script": "bench.py", "args": ["--rows", 205]}]})
        self.assertEqual([s["kind"] for s in steps], ["stop", "fabric"])
        self.assertEqual(steps[1]["config"], "experiments/x/arm.json")
        self.assertEqual(steps[1]["script"], "experiments/x/bench.py")  # a bare name is the experiment's
        self.assertEqual(steps[1]["args"], ["--rows", "205"])
        self.assertEqual(steps[1]["label"], "bench")

    def test_command_is_the_probe_container_with_the_arm_mounts(self):
        cluster = {"container": {"image": "img", "mounts": [["{home}/o/a.py", "/opt/a.py", "ro"]]},
                   "host": {"home": "/h"}}
        probe = ["docker", "run", "--env", "A=1", "--volume", "/repo/scripts/probe_collectives.py:/probe.py:ro",
                 "img", "--signal=TERM", "600s", "python3", "/probe.py", "--rank", "1", "--world-size", "4",
                 "--master-addr", "m", "--master-port", "29581"]
        with mock.patch.object(lab.spark, "collective_probe_command", return_value=probe), \
                mock.patch.object(lab.spark, "repository_path", return_value="/repo"), \
                mock.patch.object(lab.spark.topology, "transport", return_value="oneshot-ring4"):
            command = lab.fabric_command(cluster, {}, {}, "exp/bench.py", ["--rows", "205"])
        self.assertEqual(command, [
            "docker", "run", "--env", "A=1", "--volume", "/repo/exp/bench.py:/fabric.py:ro",
            "--volume", "/h/o/a.py:/opt/a.py:ro", "img", "--signal=TERM", "600s",
            "python3", "/fabric.py", "--rank", "1", "--world-size", "4", "--master-addr", "m",
            "--master-port", "29581", "--rows", "205"])

    def test_environment_overrides_the_arms_variables(self):
        cluster = {"container": {"image": "img", "mounts": []}, "host": {"home": "/h"}}
        probe = ["docker", "run", "--env", "A=1", "--env", "B=2", "img", "python3", "/probe.py", "--rank", "0"]
        with mock.patch.object(lab.spark, "collective_probe_command", return_value=probe), \
                mock.patch.object(lab.spark, "repository_path", return_value="/repo"), \
                mock.patch.object(lab.spark.topology, "transport", return_value="oneshot-ring4"):
            command = lab.fabric_command(cluster, {}, {}, "s.py", [], {"B": "3", "C": "4"})
        self.assertEqual(command, ["docker", "run", "--env", "A=1", "--env", "B=3", "--env", "C=4", "img",
                                   "python3", "/fabric.py", "--rank", "0"])


class ImageAndBenchTest(unittest.TestCase):
    def test_image_job_stops_serving_then_builds_and_copies(self):
        steps = lab.plan({"experiment": "experiments/x", "run": "r", "jobs": [
            {"kind": "image", "config": "arm.json", "node": "dgx4", "copies": [["dgx4", "10.14.1.1"]]}]})
        self.assertEqual([s["kind"] for s in steps], ["stop", "image"])
        self.assertEqual(steps[1]["config"], "experiments/x/arm.json")
        self.assertEqual(steps[1]["copies"], [["dgx4", "10.14.1.1"]])

    def test_bench_args_replace_the_screen_bench(self):
        steps = lab.plan({"experiment": "experiments/x", "run": "r", "jobs": [
            {"kind": "measure", "extras": [], "bench_args": ["--suites", "quality,decode", "--concurrency", "1,16"],
             "arms": [{"config": "a.json", "label": "a"}]}]})
        bench = next(s for s in steps if s["kind"] == "cli")["argv"]
        self.assertEqual(bench[:7], ["--cluster-config", "experiments/x/a.json", "bench", "--suites",
                                     "quality,decode", "--concurrency", "1,16"])
        self.assertNotIn("--allow-mismatch", bench)


class ReleaseProfileTest(unittest.TestCase):
    def test_release_profile_builds_its_bench_and_table(self) -> None:
        spec = {"experiment": "experiments/x", "run": "rel", "jobs": [{
            "kind": "measure", "profile": "release", "bracket": False,
            "arms": [{"config": "base.json", "label": "base"}, {"config": "cand.json", "label": "cand"}]}]}
        steps = lab.plan(spec)
        bench = [s for s in steps if s["kind"] == "cli"]
        self.assertEqual(len(bench), 2)
        self.assertIn("quality,decode,prefill", bench[0]["argv"])
        self.assertEqual(steps[-1]["argv"][:4], ["scripts/release_table.py", "rel", "base", "cand"])
        stems = {s["out"].rsplit("/", 1)[1] for s in steps if s["kind"] == "script"}
        for stem in ("determinism", "sweep", "c1-distinct", "c4-distinct", "c8-distinct", "c16-distinct",
                     "ttft", "mixed"):
            self.assertIn(f"{stem}-cand.jsonl", stems)

    def test_release_prefill_sizes_fit_a_1m_profile(self) -> None:
        args = lab.PROFILES["release"]["bench_args"]
        sizes = [int(x) for x in args[args.index("--prefill-sizes") + 1].split(",")]
        self.assertTrue(all(size < 1048576 - 64 for size in sizes))
        self.assertIn(1000000, sizes)

    def test_release_table_judges_against_the_first_arm(self) -> None:
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            det = root / "results/private/determinism/rel"
            det.mkdir(parents=True)
            for arm, tps, prefill in (("base", 200.0, 5000.0), ("cand", 220.0, 5005.0)):
                bench = root / f"results/private/bench/rel-{arm}"
                bench.mkdir(parents=True)
                (bench / "bench.json").write_text(json.dumps({"suites": {
                    "quality": {"passed": 5, "runs": [1, 2, 3, 4, 5], "ok": True},
                    "prefill": {"points": {"32768": {"prefill_tps": {"n": 3, "mean": prefill, "sd": 10.0}}}},
                    "decode": {"points": {}}}}))
                (det / f"c16-distinct-{arm}.jsonl").write_text(json.dumps(
                    {"summary": "c16", "tps_mean": tps, "ci95_pct": 2.0}) + "\n")
                (det / f"sweep-{arm}.jsonl").write_text("".join(
                    json.dumps({"prompt": p, "tps": 80.0 + p + (5 if arm == "cand" else 0),
                                "accepted_per_draft": 1.5}) + "\n" for p in range(16)))
            out = subprocess.run([sys.executable, str(ROOT / "scripts/release_table.py"), "rel", "base", "cand"],
                                 cwd=root, text=True, capture_output=True, check=True).stdout
        self.assertIn("16 different prompts at once tok/s", out)
        self.assertRegex(out, r"16 different prompts at once tok/s.*\+10\.0%.*faster")
        self.assertRegex(out, r"32,768 tokens.*same")
        self.assertRegex(out, r"acceptance sweep tok/s.*paired over 16 prompts.*faster")
        self.assertIn("5/5", out)


class ReleaseTP3ProfileTest(unittest.TestCase):
    def test_release_tp3_stays_within_tp3_limits(self) -> None:
        args = lab.PROFILES["release-tp3"]["bench_args"]
        self.assertEqual(args[args.index("--concurrency") + 1], "1,2,4,8")
        sizes = [int(x) for x in args[args.index("--prefill-sizes") + 1].split(",")]
        self.assertTrue(all(size < 524288 - 64 for size in sizes))
        stems = [extra[2] for extra in lab.PROFILES["release-tp3"]["extras"]]
        self.assertNotIn("c16-distinct", stems)
        self.assertIn("c8-distinct", stems)
        self.assertEqual(lab.PROFILES["release-tp3"]["table"], "scripts/release_table.py")


class OverlayArmTest(unittest.TestCase):
    """The lab reads every arm resolved: an overlay arm has no container or serve_args of its own."""

    def test_the_lab_reads_overlay_arms_resolved(self) -> None:
        files = {"arm.json": {"extends": {"config": "config/cluster.json", "shape": "tp4-ring"},
                              "set": {"container": {"image": "vllm-ds41f-kkref:overlay"}}}}
        read = lab.spark.read_json
        with mock.patch.object(lab.spark, "read_json", lambda path: files[path] if path in files else read(path)):
            arm = lab.spark.read_cluster("arm.json")
            self.assertEqual(arm["container"]["image"], "vllm-ds41f-kkref:overlay")
            self.assertTrue(arm["container"]["mounts"])
            files["profiled.json"] = {
                "extends": {"config": "arm.json"},
                "set": {"serve_args": [*arm["serve_args"], "--profiler-config",
                                       json.dumps({"torch_profiler_dir": "/cache/profile"})]}}
            self.assertEqual(lab.profiler_dir("profiled.json"), "/cache/profile")
