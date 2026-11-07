"""CPU-only checks of fan-policy failure and restoration paths."""
import subprocess
import unittest

import spark_ds41f as spark
from cooling import helpers


class Host:
    def __init__(self, active=False, installed=True, fail_start=False, fail_stop=False):
        self.active = active
        self.installed = installed
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.commands = []
        self.temperature = 50000

    def __call__(self, nodes, node, *command):
        self.commands.append(command)
        code, stdout = 0, ""
        if command[:2] == ("sh", "-c"):
            stdout = str(self.temperature)
        elif command[:2] == ("systemctl", "is-active"):
            code = 0 if self.active else 3
        elif command[:2] == ("systemctl", "cat"):
            code = 0 if self.installed else 1
        elif command[:4] == ("sudo", "-n", "systemctl", "start"):
            code = int(self.fail_start)
            self.active = not self.fail_start
        elif command[:4] == ("sudo", "-n", "systemctl", "stop"):
            code = int(self.fail_stop)
            if not code:
                self.active = False
        return subprocess.CompletedProcess(command, code, stdout, "failure" if code == 1 else "")


class CoolingPolicy(unittest.TestCase):
    targets = [{"name": "kernel-host"}]

    def test_inactive_curve_started_and_restored_without_global_override(self):
        original_transport = spark.run_ssh
        host = Host()
        policy = helpers(host)
        records = policy.cool_nodes({}, self.targets, 55, 0)
        self.assertEqual(records[0]["start_c"], 50)
        curves = policy.start_fan_curve({}, self.targets)
        self.assertEqual(policy.fan_curve_problems(curves), [])
        self.assertTrue(host.active)
        self.assertEqual(policy.stop_fan_curve({}, self.targets, curves), [])
        self.assertFalse(host.active)
        self.assertIs(spark.run_ssh, original_transport)

    def test_preexisting_curve_is_left_running(self):
        host = Host(active=True)
        policy = helpers(host)
        records = policy.start_fan_curve({}, self.targets)
        self.assertFalse(records[0]["started"])
        self.assertEqual(policy.stop_fan_curve({}, self.targets, records), [])
        self.assertTrue(host.active)
        self.assertFalse(any("stop" in command for command in host.commands))

    def test_missing_or_failed_curve_blocks_measurement(self):
        for host in (Host(installed=False), Host(fail_start=True)):
            policy = helpers(host)
            records = policy.start_fan_curve({}, self.targets)
            self.assertTrue(policy.fan_curve_problems(records))
            self.assertFalse(host.active)

    def test_hot_timeout_restores_the_previous_active_curve(self):
        host = Host(active=True)
        host.temperature = 60000
        policy = helpers(host)
        with self.assertRaisesRegex(RuntimeError, "still at or above"):
            policy.cool_nodes({}, self.targets, 55, 0)
        self.assertTrue(host.active)
        self.assertTrue(any(command[-2:] == ("set-state", "12") for command in host.commands))

    def test_stop_failure_is_reported(self):
        host = Host(fail_stop=True)
        policy = helpers(host)
        records = policy.start_fan_curve({}, self.targets)
        self.assertTrue(policy.stop_fan_curve({}, self.targets, records))
        self.assertTrue(host.active)


if __name__ == "__main__":
    unittest.main()
