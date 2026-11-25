"""A site that is not the owner's: its own home directory, no lldpd, no fan control, no GLP file."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import spark_ds41f as spark


class NodeHomeTest(unittest.TestCase):
    def test_the_recipe_names_the_variable_not_a_user(self) -> None:
        self.assertEqual(spark.read_json("config/cluster.json")["host"]["home"], spark.HOME_VARIABLE)

    def test_a_written_path_is_kept(self) -> None:
        self.assertEqual(spark.node_home("/srv/spark", "ubuntu"), "/srv/spark")

    def test_on_a_node_the_variable_is_this_shell_home(self) -> None:
        with patch.object(spark.sys, "platform", "linux"), \
                patch.dict(spark.os.environ, {"HOME": "/data/home/j0n", "USER": "j0n"}):
            self.assertEqual(spark.node_home("$HOME", "j0n"), "/data/home/j0n")
            self.assertEqual(spark.node_home(None, None), "/data/home/j0n")
            self.assertEqual(spark.node_home("$HOME/work", ""), "/data/home/j0n/work")

    def test_a_controller_elsewhere_takes_the_node_user_home(self) -> None:
        with patch.object(spark.sys, "platform", "darwin"), \
                patch.dict(spark.os.environ, {"HOME": "/Users/someone", "USER": "ubuntu"}):
            self.assertEqual(spark.node_home("$HOME", "ubuntu"), "/home/ubuntu")
        with patch.object(spark.sys, "platform", "linux"), \
                patch.dict(spark.os.environ, {"HOME": "/home/operator", "USER": "operator"}):
            self.assertEqual(spark.node_home("$HOME", "ubuntu"), "/home/ubuntu")

    def test_checkout_and_mounts_follow_it(self) -> None:
        cluster = spark.read_cluster("config/cluster.json", "tp3-triangle-4k")
        with patch.object(spark, "node_home", return_value="/home/j0n"):
            self.assertEqual(spark.repository_path(cluster), "/home/j0n/projects/spark-ds41f")
            self.assertEqual(spark.host_path(cluster, "{checkout}/cache"), "/home/j0n/projects/spark-ds41f/cache")


class LldpdTest(unittest.TestCase):
    def test_a_running_daemon_and_an_older_report_pass(self) -> None:
        self.assertEqual(spark.lldpd_warnings("dgx1", {"lldpd": "active"}), [])
        self.assertEqual(spark.lldpd_warnings("dgx1", {}), [])

    def test_a_missing_daemon_warns_with_what_is_lost(self) -> None:
        for state in ("inactive", ""):
            with self.subTest(state=state):
                (warning,) = spark.lldpd_warnings("dgx2", {"lldpd": state})
                self.assertIsInstance(warning, spark.Warn)
                self.assertIn("dgx2: lldpd is not running", warning)
                self.assertIn("sparknet topology discover cannot read this node's cables", warning)
                self.assertIn("Serving, the launch preflight and bench do not use it", warning)
                self.assertIn("sudo apt install lldpd", warning)

    def test_only_directly_cabled_nodes_are_checked(self) -> None:
        nodes = {"nodes": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
        triangle = spark.read_cluster("config/cluster.json", "tp3-triangle-4k")
        self.assertTrue(spark.lldp_cabled(triangle, nodes))
        self.assertFalse(spark.lldp_cabled(triangle, {"nodes": nodes["nodes"][:1]}))
        switched = spark.read_cluster("config/cluster.json", "tp3-switched")
        self.assertFalse(spark.lldp_cabled(switched, nodes))
        self.assertEqual(spark.lldpd_warnings("a", {"lldpd": "inactive"}, cabled=False), [])


class CoolingWithoutFanControlTest(unittest.TestCase):
    ABSENT = {"service": False, "floor": False}

    def cool(self, temperatures, controls, timeout=60.0, fan_control=True):
        targets = [{"name": f"n{i}"} for i in range(len(controls) or 1)]
        readings = {target["name"]: list(temperatures[i::len(targets)]) for i, target in enumerate(targets)}
        with patch.object(spark, "hottest_zone_c", side_effect=lambda nodes, node: readings[node["name"]].pop(0)), \
                patch.object(spark, "take_fan_control",
                             side_effect=lambda nodes, node: controls[int(node["name"][1:])]) as take, \
                patch.object(spark, "restore_fan_control", return_value=None), \
                patch.object(spark.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()) as progress:
            records = spark.cool_nodes({}, targets, 55, timeout, fan_control=fan_control)
        return records, progress.getvalue(), take

    def test_it_waits_until_the_node_cools(self) -> None:
        records, progress, _ = self.cool([60.0, 57.0, 54.0], [self.ABSENT])
        self.assertEqual((records[0]["final_c"], records[0]["fan_floor"]), (54.0, None))
        self.assertNotIn("timed_out", records[0])
        self.assertIn("no fan control here, waiting until all are below 55 C", progress)
        self.assertNotIn("fans restored", progress)

    def test_a_node_that_stays_warm_is_measured_with_a_warning(self) -> None:
        records, progress, _ = self.cool([60.0, 58.0], [self.ABSENT], timeout=0.0)
        self.assertTrue(records[0]["timed_out"])
        self.assertIn("WARN: cooling: still at or above 55 C after 0 s without fan control on n0", progress)

    def test_a_site_that_declares_no_fan_control_waits_without_touching_fans(self) -> None:
        records, progress, take = self.cool([60.0, 50.0], [], fan_control=False)
        take.assert_not_called()
        self.assertEqual(records[0]["final_c"], 50.0)

    def test_one_node_without_it_does_not_stop_the_others(self) -> None:
        records, progress, _ = self.cool([60.0, 60.0, 50.0, 50.0], [{"service": True, "floor": True}, self.ABSENT])
        self.assertEqual([record["fan_floor"] for record in records], [12, None])
        self.assertIn("n1 without fan control", progress)

    def test_an_installed_control_that_fails_still_aborts(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cannot select maximum fan floor on n0"):
            self.cool([60.0], [{"service": True, "floor": False}])

    def test_the_curve_refuses_only_a_service_that_did_not_start(self) -> None:
        absent = {"node": "a", "service": "absent", "started": False}
        running = {"node": "b", "service": "active", "started": False}
        failed = {"node": "c", "service": "inactive", "started": False, "error": "x"}
        self.assertFalse(spark.fan_curve_failed([absent, running]))
        self.assertFalse(spark.fan_curve_failed([absent, absent]))
        self.assertTrue(spark.fan_curve_failed([absent, failed]))


class MissingGlpTest(unittest.TestCase):
    NODES = {"nodes": [{"name": "dgx1", "head": True}, {"name": "dgx2"}, {"name": "dgx3"}]}

    def profile(self) -> dict:
        cluster = spark.read_cluster("config/cluster.json", "tp3-triangle-4k")
        cluster["host"]["home"] = "/home/u"
        return cluster

    def served(self, cluster: dict, missing: set[str], unreachable: set[str] = frozenset()):
        def run_ssh(nodes, node, *argv):
            self.assertEqual(argv[:2], ("test", "-e"))
            code = 255 if node["name"] in unreachable else 1 if node["name"] in missing else 0
            return subprocess.CompletedProcess(argv, code, "", "")

        with patch.object(spark, "run_ssh", side_effect=run_ssh):
            return spark.without_missing_glp(cluster, self.NODES)

    def test_the_recipe_renders_the_steered_model(self) -> None:
        cluster = self.profile()
        self.assertEqual(spark.served_model_names(cluster),
                         ["deepseek-v4.1-flash", "deepseek-v4.1-flash-weightless"])

    def test_with_the_file_everywhere_nothing_changes(self) -> None:
        cluster = self.profile()
        served, warnings = self.served(cluster, set())
        self.assertIs(served, cluster)
        self.assertEqual(warnings, [])

    def test_a_node_without_the_file_serves_the_stock_model_with_a_warning(self) -> None:
        cluster = self.profile()
        served, (warning,) = self.served(cluster, {"dgx2", "dgx3"})
        self.assertIsInstance(warning, spark.Warn)
        self.assertIn("GLP steering file is not installed on dgx2, dgx3", warning)
        self.assertIn("serves deepseek-v4.1-flash only", warning)
        self.assertIn("hf download msuiche/", warning)
        self.assertNotIn("glp", served)
        self.assertEqual(spark.served_model_names(served), ["deepseek-v4.1-flash"])
        self.assertFalse({"WEIGHTLESS_STEER_PATH", "WEIGHTLESS_STEER_MODEL"} & set(served["environment"]))
        self.assertFalse([mount for mount in served["container"]["mounts"] if mount[1] == "/glp/vector.gguf"])
        self.assertEqual(spark.glp_problems(served), [])
        # The profile that was read is untouched, and the rest of the recipe is the same.
        self.assertIn("glp", cluster)
        rest = spark.cluster_recipe.without_glp(cluster)
        self.assertEqual(served, rest)

    def test_an_unreachable_node_is_left_to_the_other_checks(self) -> None:
        cluster = self.profile()
        served, warnings = self.served(cluster, set(), unreachable={"dgx3"})
        self.assertIs(served, cluster)
        self.assertEqual(warnings, [])

    def test_a_profile_without_steering_asks_no_node(self) -> None:
        cluster = spark.cluster_recipe.without_glp(self.profile())
        with patch.object(spark, "run_ssh", side_effect=AssertionError("contacted a node")):
            self.assertEqual(spark.without_missing_glp(cluster, self.NODES), (cluster, []))


if __name__ == "__main__":
    unittest.main()
