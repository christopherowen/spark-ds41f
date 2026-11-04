"""Safety cases for resuming an image-only scheduling handoff, without hardware."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("copy_round", Path(__file__).with_name("run_round.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GateResume(unittest.TestCase):
    def setUp(self):
        self.previous = {"ended": "2026-11-04T10:00:00Z", "restored": True, "commit": "build-commit",
                         "error": "RuntimeError: window changed or closure requested",
                         "steps": [{"log": "image-and-gate.log", "exit": 0}]}
        logs = ["build-prepare", "build-check", "build-image", "build-image-apply",
                "gate-manifest", "gate-bundle", "release-tests"]
        self.image = {"ended": "2026-11-04T09:59:00Z", "passed": True, "commit": "build-commit",
                      "image": module.IMAGE_ID, "gate": {"passed": True, "summary": "8000 passed, 20 skipped"},
                      "conditions": {"halted": [], "fan_stop_failures": []},
                      "steps": [{"log": name + ".log", "exit": 0} for name in logs]}

    def rejects(self, previous=None, image=None):
        with self.assertRaises(ValueError):
            module.validate_gate_resume(previous or self.previous, image or self.image)

    def test_clean_scheduling_handoff_can_reuse_its_exact_image(self):
        self.assertEqual(module.validate_gate_resume(self.previous, self.image)["image"], module.IMAGE_ID)

    def test_active_controller_is_rejected(self):
        self.previous.pop("ended"); self.rejects()

    def test_even_successful_timed_work_cannot_be_repeated(self):
        self.previous["steps"].append({"log": "functional.log", "exit": 0}); self.rejects()

    def test_failed_image_phase_is_not_a_scheduling_resume(self):
        self.previous["steps"][0]["exit"] = 1; self.rejects()

    def test_missing_restoration_or_latched_hardware_loss_is_rejected(self):
        for patch in [{"restored": False}, {"restore_error": "not healthy"}, {"management_loss_latched": True}]:
            previous = {**self.previous, **patch}
            self.rejects(previous=previous)

    def test_failed_or_unfinished_gate_is_rejected(self):
        for patch in [{"passed": False}, {"ended": None}, {"gate": {"passed": True, "summary": "1 failed, 8000 passed"}},
                      {"gate": {"passed": True, "summary": "0 passed, 100 skipped"}}]:
            self.rejects(image={**self.image, **patch})

    def test_any_failed_native_gate_step_is_rejected(self):
        image = copy.deepcopy(self.image); image["steps"][-1]["exit"] = 1; self.rejects(image=image)

    def test_changed_digest_or_lineage_is_rejected(self):
        self.rejects(image={**self.image, "image": "sha256:" + "0" * 64})
        self.rejects(image={**self.image, "commit": "other-build"})

    def test_guard_or_fan_restore_failure_is_rejected(self):
        for conditions in [{"halted": ["thermal guard"]}, {"fan_stop_failures": ["dgx4"]}]:
            self.rejects(image={**self.image, "conditions": conditions})


if __name__ == "__main__":
    unittest.main()
