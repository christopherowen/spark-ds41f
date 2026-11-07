"""Host-independent tests for the degraded-node memory detector."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spark_ds41f import node_memory

NODES = {
    "nodes": [
        {"name": "spark1", "head": True},
        {"name": "spark2", "head": False},
    ]
}


class FakeSsh:
    """Stands in for run_ssh: docker logs answers, hostname answers."""

    def __init__(self, values: dict[str, float | None]):
        self.values = values
        self.calls: list[tuple] = []

    def __call__(self, nodes, node, *command):
        self.calls.append((node["name"], command))
        name = node["name"]

        class Result:
            returncode = 0
            stderr = ""

        result = Result()
        if command[:2] == ("docker", "logs"):
            value = self.values.get(name)
            if value is None:
                result.returncode = 1
                result.stdout = ""
            else:
                result.stdout = (
                    f"(Worker_TP0 pid=1) Initial free memory {value:.2f} GiB, reserved "
                    "4.91 GiB memory for KV Cache\n"
                )
        else:
            result.stdout = ""
        return result


def cluster(image="spark-ds41f:1107", **host):
    return {"container": {"name": "spark-ds41f", "image": image}, "host": dict(host)}


class NodeMemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "baseline.json"
        self._original_path = node_memory.baseline_path
        node_memory.baseline_path = lambda: self.path  # type: ignore[assignment]
        self.addCleanup(setattr, node_memory, "baseline_path", self._original_path)

    def run_check(self, values, **host):
        fake = FakeSsh(values)
        original = node_memory.run_ssh
        node_memory.run_ssh = fake  # type: ignore[assignment]
        try:
            return node_memory.check_nodes(cluster(**host), NODES, "2026-11-07T00:00:00Z")
        finally:
            node_memory.run_ssh = original

    def test_first_boot_records_a_baseline_and_stays_quiet(self):
        warnings = self.run_check({"spark1": 113.28, "spark2": 115.02})
        self.assertEqual(warnings, [])
        recorded = json.loads(self.path.read_text())
        self.assertEqual(recorded["spark1"]["best_gib"], 113.28)
        self.assertEqual(recorded["spark2"]["best_gib"], 115.02)

    def test_a_material_drop_warns_with_numbers_and_a_reboot_hint(self):
        self.run_check({"spark1": 113.28, "spark2": 115.02})
        warnings = self.run_check({"spark1": 108.55, "spark2": 115.02})
        self.assertEqual(len(warnings), 1)
        self.assertIn("spark1", warnings[0])
        self.assertIn("108.55 GiB", warnings[0])
        self.assertIn("4.73 GiB below", warnings[0])
        self.assertIn("113.28 GiB", warnings[0])
        self.assertIn("reboot", warnings[0])

    def test_a_small_drop_is_quiet(self):
        self.run_check({"spark1": 113.28, "spark2": 115.02})
        self.assertEqual(self.run_check({"spark1": 112.80, "spark2": 115.02}), [])

    def test_the_threshold_is_configurable(self):
        self.run_check({"spark1": 113.28, "spark2": 115.02})
        warnings = self.run_check(
            {"spark1": 112.60, "spark2": 115.02}, node_memory_drop_gib=0.5
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("0.68 GiB below", warnings[0])

    def test_a_new_high_water_mark_raises_the_baseline(self):
        self.run_check({"spark1": 110.00, "spark2": 115.02})
        self.run_check({"spark1": 113.28, "spark2": 115.02})
        recorded = json.loads(self.path.read_text())
        self.assertEqual(recorded["spark1"]["best_gib"], 113.28)

    def test_a_different_image_rebaselines_instead_of_warning(self):
        self.run_check({"spark1": 113.28, "spark2": 115.02})
        warnings = self.run_check(
            {"spark1": 108.55, "spark2": 115.02}, image="spark-ds41f:candidate-36"
        )
        self.assertEqual(warnings, [])
        recorded = json.loads(self.path.read_text())
        self.assertEqual(recorded["spark1"]["image"], "spark-ds41f:candidate-36")

    def test_unreadable_logs_are_skipped_not_guessed(self):
        warnings = self.run_check({"spark1": None, "spark2": 115.02})
        self.assertEqual(warnings, [])
        recorded = json.loads(self.path.read_text())
        self.assertNotIn("spark1", recorded)

    def test_the_watch_can_be_switched_off(self):
        self.run_check({"spark1": 113.28, "spark2": 115.02})
        self.assertEqual(
            self.run_check({"spark1": 100.00, "spark2": 115.02}, node_memory_watch=False),
            [],
        )

    def test_a_corrupt_baseline_is_replaced_rather_than_fatal(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not json")
        warnings = self.run_check({"spark1": 113.28, "spark2": 115.02})
        self.assertEqual(warnings, [])
        self.assertEqual(json.loads(self.path.read_text())["spark1"]["best_gib"], 113.28)

    def test_a_node_seen_for_the_first_time_baselines_quietly(self):
        self.run_check({"spark1": 113.28})
        warnings = self.run_check({"spark1": 113.28, "spark2": 114.60})
        self.assertEqual(warnings, [])
        self.assertEqual(json.loads(self.path.read_text())["spark2"]["best_gib"], 114.60)

    def test_the_colon_form_of_the_line_is_understood(self):
        fake = FakeSsh({"spark1": 113.28})
        result = fake(NODES, NODES["nodes"][0], "docker", "logs", "spark-ds41f")
        self.assertIn("Initial free memory 113.28 GiB", result.stdout)
        self.assertEqual(node_memory.parse_initial_free_memory(
            "Initial free memory: 109.59 GiB; Requested memory: 0.82"), 109.59)


if __name__ == "__main__":
    unittest.main()
