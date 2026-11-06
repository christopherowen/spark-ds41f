"""Offline failure proofs for C46's pre-timing and completion barriers."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import c46_functional as functional
import c46_performance as performance
from servebench import suite
from spark_ds41f import functional_evidence


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


class FunctionalBarrierTest(unittest.TestCase):
    def test_zero_exit_and_forged_pass_do_not_hide_false_live_evidence(self):
        spec = json.loads((HERE / "functional.json").read_text())
        contract = json.loads((HERE / "control-readiness.json").read_text())["semantic_validator"]["contract"]
        for forged in (False, True):
            with self.subTest(forged=forged), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                image = {"image": "sha256:" + "a" * 64, "source": {"expected_labels": {}}}
                write(root / functional.QUALIFIED, image)
                collector = root / "scripts/collect_runtime.py"
                collector.parent.mkdir(parents=True)
                collector.write_text("# source-bound collector fixture\n")
                label = spec["jobs"][0]["arms"][0]["label"]
                output = root / "results/private/determinism" / spec["run"] / f"determinism-{label}.jsonl"
                write(output, {"summary": "temperature-0 determinism", "identical": False, "logprobs": True})
                write(Path(str(output) + ".telemetry.json"), {})
                record = functional_evidence.evaluate(output, spec["jobs"][0]["evidence"]["determinism"])
                record["process_exit"] = 0
                if forged:
                    record.update(required_evidence_passed=True, errors=[], status="requirements-passed")
                write(Path(str(output) + ".evidence.json"), record)
                with mock.patch.object(functional, "ROOT", root), self.assertRaisesRegex(ValueError, "declared evidence failed"):
                    functional.validate(root, spec, image, contract)

    def test_fault_scan_catches_chunk_boundary_without_loading_whole_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engine.log"
            path.write_bytes(b"a" * 65534 + b"CUDA error" + b"b" * 65536)
            digest, fault = functional.engine_log(path)
            self.assertTrue(fault)
            self.assertEqual(digest, functional.sha(path.read_bytes()))

    def test_malformed_json_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('human diagnostic\n{"summary": "ok"}\n{"broken"\n')
            with self.assertRaisesRegex(ValueError, "malformed JSON"):
                functional.records(path)


class PerformanceCompletionTest(unittest.TestCase):
    CONTRACT = "d" * 64
    def fixture(self, root):
        spec = json.loads((HERE / "performance.json").read_text())
        commit = "b" * 40
        image = {"image": "sha256:" + "a" * 64, "source": {"expected_labels": {"tree": "c46"}}}
        directories = []
        for boot, arm in enumerate(spec["jobs"][0]["arms"]):
            for number, name in enumerate(spec["jobs"][0]["suite"]):
                suffix = "" if not number else "-" + name
                path = root / "results/private/suites" / (spec["run"] + suffix) / arm["label"]
                definition = suite.load(name)
                identity = {"live_problems": [], "client_dirty": False, "client_commit": commit,
                            "cluster_config": arm["config"], "nodes": {}}
                for node in performance.NODES:
                    identity["nodes"][node] = {"image_id": image["image"], "checkout": commit,
                        "image_labels": {"tree": "c46"}, "image_layers": ["same"],
                        "image_content_sha256": "same", "container_started_at": "boot-" + str(boot)}
                write(path / "suite.json", {"suite": definition, "suite_sha256": suite.digest(definition),
                    "finished_utc": "finished", "metadata": {"identity": identity},
                    "steps": [{"id": s["id"], "command": s["command"], "exit": 0,
                               "output": suite.output_name(s)} for s in definition["steps"]]})
                if name == "release-tp3":
                    index = json.loads((path / "suite.json").read_text())
                    index["model_evidence"] = {"output": "model.json.gz.evidence.json",
                        "required_evidence_passed": True, "contract_sha256": self.CONTRACT}
                    write(path / "suite.json", index)
                    write(path / "model.json.gz.evidence.json", {"required_evidence_passed": True,
                        "process_exit": 0, "errors": [], "contract": {"sha256": self.CONTRACT}})
                write(path / "conditions.json", {"halted": None, "fan_stop_failures": [],
                    "memory": {n: {"min_available_gib": 6, "swap_growth_gib": 0} for n in performance.NODES},
                    "thermal": {n: {"thermal_slowdown_ms": 0, "power_cap_ms": 0,
                                    "max_gpu_c": 62, "min_headroom_c": 25, "max_system_c": 70, "max_power_w": 99}
                                for n in performance.NODES},
                    "cooling": [{"node": n, "final_c": 50} for n in performance.NODES],
                    "fan_curve": [{"node": n} for n in performance.NODES]})
                write(root / "results/private/determinism" / spec["run"] / f"suite-{arm['label']}{suffix}.telemetry.json",
                      {"nodes": {n: {"samples": 1} for n in performance.NODES}})
                directories.append(path)
            write(root / "results/private/determinism" / spec["run"] / f"startup-memory-{arm['label']}.json",
                  {n: {"samples": 1, "min_available_gib": 6} for n in performance.NODES})
        return spec, image, commit, directories

    def test_native_failed_or_incomplete_suite_cannot_be_a_completed_round(self):
        for failure in ("failed-step", "missing-step", "unfinished", "wrong-image", "halted", "reused-boot",
                        "missing-model-barrier", "wrong-model-contract", "false-model-barrier"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spec, image, commit, paths = self.fixture(root)
                if failure == "halted":
                    p = paths[0] / "conditions.json"
                    d = json.loads(p.read_text()); d["halted"] = "memory telemetry stopped reporting"; write(p, d)
                elif failure == "reused-boot":
                    for path in paths[3:]:
                        p = path / "suite.json"; d = json.loads(p.read_text())
                        for node in d["metadata"]["identity"]["nodes"].values():
                            node["container_started_at"] = "boot-0"
                        write(p, d)
                else:
                    p = paths[0] / "suite.json"; d = json.loads(p.read_text())
                    if failure == "failed-step": d["steps"][0]["exit"] = 1
                    elif failure == "missing-step": d["steps"].pop()
                    elif failure == "unfinished": d.pop("finished_utc")
                    elif failure == "wrong-image": d["metadata"]["identity"]["nodes"]["dgx2"]["image_id"] = "wrong"
                    elif failure == "missing-model-barrier": d.pop("model_evidence")
                    elif failure == "wrong-model-contract": d["model_evidence"]["contract_sha256"] = "e" * 64
                    elif failure == "false-model-barrier": d["model_evidence"]["required_evidence_passed"] = False
                    write(p, d)
                with mock.patch.object(performance, "ROOT", root), self.assertRaises(ValueError):
                    performance.validate(spec, image, commit, self.CONTRACT)

    def test_complete_six_suite_two_boot_cohort_has26_ordered_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, image, commit, _ = self.fixture(root)
            with mock.patch.object(performance, "ROOT", root):
                result = performance.validate(spec, image, commit, self.CONTRACT)
            self.assertTrue(result["passed"])
            self.assertEqual((result["suite_count"], result["ordered_step_count"]), (6, 26))


if __name__ == "__main__":
    unittest.main()
