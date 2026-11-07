"""Metadata failure controls; these do not execute or claim native GPU cases."""
import json
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c46_native


class NativeRegressionTest(unittest.TestCase):
    def test_recorded_guard_schema_rejects_missing_or_adverse_telemetry(self):
        # C43's good guard evidence is a metadata fixture only. Its failed
        # native release verdict remains failed and is not replayed here.
        root = Path(__file__).resolve().parents[2]
        path = root / "experiments/2026-11-05-frontend-lifetime-qualification/runs/2026-11-05-dgx/attempt-1/failed-control-native/builder-receipt.json"
        original = json.loads(path.read_text())["conditions"]
        self.assertTrue(c46_native.conditions_pass(original, ["dgx4"], 4))
        for field in ("missing-rank", "missing-field", "nonfinite", "swap", "slowdown"):
            with self.subTest(field=field):
                conditions = copy.deepcopy(original)
                if field == "missing-rank": conditions["thermal"] = {}
                elif field == "missing-field": conditions["thermal"]["dgx4"].pop("max_gpu_c")
                elif field == "nonfinite": conditions["thermal"]["dgx4"]["max_gpu_c"] = float("nan")
                elif field == "swap": conditions["memory"]["dgx4"]["swap_growth_gib"] = 0.01
                elif field == "slowdown": conditions["thermal"]["dgx4"]["thermal_slowdown_ms"] = 1
                self.assertFalse(c46_native.conditions_pass(conditions, ["dgx4"], 4))

    def fixture(self, root):
        rows = [{"file": name, "status": "ok", "passed": count, "failed": 0,
                 "error": 0, "skipped": 0, "failures": []} for name, count in c46_native.EXPECTED.items()]
        (root / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        (root / "chunk.xml").write_text('<testsuite><testcase classname="tests.v1.worker.test_gpu_verified_logprob_token_ids" '
                                       f'name="{c46_native.INVARIANT}"/></testsuite>')

    def test_failed_skipped_missing_or_duplicated_original_invariant_is_not_fixed(self):
        for outcome in ("failure", "error", "skipped", "missing", "duplicate"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); self.fixture(root)
                case = ('<testcase classname="tests.v1.worker.test_gpu_verified_logprob_token_ids" '
                        f'name="{c46_native.INVARIANT}">')
                content = "" if outcome == "missing" else (case + "</testcase>") * 2 if outcome == "duplicate" else case + f"<{outcome}/></testcase>"
                (root / "chunk.xml").write_text("<testsuite>" + content + "</testsuite>")
                self.assertFalse(c46_native.audit(root)["passed"])

    def test_missing_producer_numerical_or_frontend_cases_are_refused(self):
        for name in c46_native.EXPECTED:
            with self.subTest(file=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); self.fixture(root)
                path = root / "results.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                row = next(row for row in rows if row["file"] == name)
                row["passed"] -= 1; row["skipped"] = 1
                path.write_text("\n".join(json.dumps(row) for row in rows))
                self.assertFalse(c46_native.audit(root)["passed"])

    def test_complete_regression_metadata_passes_without_mutating_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.fixture(root)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            result = c46_native.audit(root)
            self.assertTrue(result["passed"])
            self.assertEqual(result["invariant_occurrences"], 1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})


if __name__ == "__main__":
    unittest.main()
