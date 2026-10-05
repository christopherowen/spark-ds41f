"""Cooling must restore the normal fan policy before a measurement can start."""
import contextlib
import io
import importlib.machinery
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

loader = importlib.machinery.SourceFileLoader(
    "spark_cooling_tests", str(Path(__file__).resolve().parents[1] / "bin/spark"))
spec = importlib.util.spec_from_loader(loader.name, loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)


class CoolingTest(unittest.TestCase):
    def test_intentional_restart_recovers_an_exhausted_start_budget(self):
        state = {"blocked": True, "active": False}

        def systemctl(nodes, node, *command):
            if "reset-failed" in command:
                state["blocked"] = False
            elif "start" in command:
                state["active"] = not state["blocked"]
                return subprocess.CompletedProcess(command, 0 if state["active"] else 1)
            elif "is-active" in command:
                return subprocess.CompletedProcess(command, 0 if state["active"] else 3)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(spark, "run_ssh", side_effect=systemctl):
            result = spark.restore_fan_control({}, {}, {"service": True, "floor": True})
        self.assertEqual(result, spark.FAN_SERVICE)
        self.assertTrue(state["active"])

    def test_successful_start_does_not_hide_a_failed_daemon(self):
        def systemctl(nodes, node, *command):
            return subprocess.CompletedProcess(command, 3 if "is-active" in command else 0)

        with patch.object(spark, "run_ssh", side_effect=systemctl):
            result = spark.restore_fan_control({}, {}, {"service": True, "floor": True})
        self.assertTrue(result.startswith("FAILED"))

    def test_failed_restore_aborts_even_after_temperatures_fall(self):
        with patch.object(spark, "hottest_zone_c", side_effect=[60.0, 50.0]), \
             patch.object(spark, "take_fan_control", return_value={"service": True, "floor": True}), \
             patch.object(spark, "restore_fan_control", return_value="FAILED: service"), \
             patch.object(spark.time, "sleep"), \
             contextlib.redirect_stdout(io.StringIO()) as progress:
            with self.assertRaisesRegex(RuntimeError, "cannot restore fan control"):
                spark.cool_nodes({}, [{"name": "node"}], 55, 60)
        self.assertIn("every node at fan floor 12", progress.getvalue())

    def test_failed_take_still_restores_the_original_control(self):
        with patch.object(spark, "hottest_zone_c", return_value=60.0), \
             patch.object(spark, "take_fan_control", return_value={"service": True, "floor": False}), \
             patch.object(spark, "restore_fan_control", return_value=spark.FAN_SERVICE) as restore, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "cannot select maximum fan floor"):
                spark.cool_nodes({}, [{"name": "node"}], 55, 60)
        restore.assert_called_once()


if __name__ == "__main__":
    unittest.main()
