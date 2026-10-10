"""Host-independent tests for the experiment window runner (scripts/lab.py)."""

from __future__ import annotations

import contextlib
import io
import importlib.machinery
import importlib.util
import json
import subprocess
import tempfile
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


TP4 = "config/cluster-tp4.json"
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
        self.assertEqual(lab.resolve_production("./config/cluster-tp4.json", None), TP4)
        ours = lab.new_hold(10, "", at=1_000_000.0)
        ours["production_config"] = TP4
        self.assertEqual(lab.resolve_production(None, ours), TP4)
        self.assertEqual(lab.resolve_production(TP4, ours), TP4)
        with self.assertRaises(SystemExit) as refused:
            lab.resolve_production("config/cluster.json", ours)
        self.assertIn("restores config/cluster-tp4.json", str(refused.exception))
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
        self.assertTrue(kept.endswith("--cluster-config config/cluster-tp4.json cluster stop --apply --parallel"))
        self.assertTrue(removed.endswith("cluster stop --remove --apply --parallel"))
        steps = lab.plan(dict(SPEC, jobs=[SPEC["jobs"][2], dict(SPEC["jobs"][2], remove_stopped=True)]))
        stops = [lab.describe_step(step) for step in steps if step["kind"] == "stop"]
        self.assertEqual(stops, ["bin/spark --cluster-config config/cluster-tp4.json cluster stop --apply --parallel",
                                 "bin/spark --cluster-config config/cluster-tp4.json cluster stop --remove "
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
        self.assertIn("--cluster-config experiments/x/cluster-tp4-candidate.json cluster start --replace --apply",
                      out.getvalue())
        self.assertIn("lab windows require the promoted node topology (config/cluster-tp4.json)", out.getvalue())
        self.assertEqual(out.getvalue().count("cluster start"), 1)

    def test_dry_run_shows_the_profile_for_every_cluster_action(self) -> None:
        spec = self.state / "spec.json"
        spec.write_text(json.dumps(SPEC))
        code, out = self.main("run", str(spec), "--dry-run", "--production-config", TP4)
        self.assertEqual(code, 0)
        self.assertIn("steps; production config/cluster-tp4.json", out.splitlines()[0])
        self.assertIn("bin/spark --cluster-config config/cluster-tp4.json cluster stop --apply --parallel", out)
        self.assertNotIn("--remove", out)
        code, out = self.main("window", "open", "--dry-run", "--production-config", TP4)
        self.assertIn("window restoring config/cluster-tp4.json", out)
        self.assertFalse(lab.HOLD.exists())

    def test_close_restores_the_profile_the_window_recorded(self) -> None:
        self.hold(TP4)
        code, out = self.main("window", "close", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("--cluster-config config/cluster-tp4.json cluster start --replace --apply", out)
        self.assertTrue(lab.HOLD.exists())
        with self.assertRaises(SystemExit):
            self.main("window", "close", "--dry-run", "--production-config", "config/cluster.json")

    def test_a_window_opened_before_the_record_closes_on_the_option_or_default(self) -> None:
        self.hold(None)
        _, out = self.main("window", "close", "--dry-run")
        self.assertIn("--cluster-config config/cluster.json cluster start", out)
        _, out = self.main("window", "close", "--dry-run", "--production-config", TP4)
        self.assertIn("--cluster-config config/cluster-tp4.json cluster start", out)

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
        self.assertIn("restores config/cluster-tp4.json", out.getvalue())

    def test_watchdog_restores_the_recorded_profile(self) -> None:
        self.hold(TP4, heartbeat_at=1_000_000.0)  # stale
        with mock.patch.object(lab, "spark_cli", return_value=0) as cli, \
                mock.patch.object(lab, "production_live", side_effect=[False, True]), \
                mock.patch.object(lab.subprocess, "run", side_effect=no_subprocesses), \
                mock.patch.object(lab.subprocess, "Popen", side_effect=no_subprocesses):
            code, out = self.main("watchdog", "--once")
        self.assertEqual(code, 0)
        self.assertEqual(cli.call_args.args, ("--cluster-config", TP4, "cluster", "start", "--replace", "--apply"))
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


class QueueTest(unittest.TestCase):
    def test_add_validates_and_orders_specs(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(lab, "QUEUE", Path(directory) / "queue"):
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
        serving = json.loads((ROOT / "config/cluster.json").read_text())["container"]["docker_run_args"]
        for arg in ("--mount=type=bind,source=/dev/dri/by-path/pci-000f:01:00.0-card,target=/dev/dri/card0",
                    "--device-cgroup-rule=c 226:* rw"):
            self.assertIn(arg, carveout)
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
