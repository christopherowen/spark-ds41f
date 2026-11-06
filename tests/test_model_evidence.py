"""CPU-only fake-step coverage for an opt-in existing-model suite barrier."""
import argparse
import copy
import hashlib
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from servebench import __main__ as cli, behaviour, model_evidence as evidence, suite


class ModelBarrierTest(unittest.TestCase):
    DEFINITION = {"name": "model-barrier-test", "version": 1, "steps": [
        {"id": "model", "command": "model", "args": {"limit": 2, "max_tokens": 2, "top": 2, "concurrency": 2}},
        {"id": "grid", "command": "engine", "args": {}},
        {"id": "c1", "command": "run", "args": {}}]}

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.model()
        refs = []
        for name in ("saved-a", "saved-b"):
            path = self.root / (name + ".json.gz")
            cli.write_json(str(path), self.data)
            raw = path.read_bytes()
            refs.append({"label": name, "path": str(path), "bytes": len(raw),
                         "sha256": hashlib.sha256(raw).hexdigest(),
                         "boot_started_at": {node: f"boot-{name}-{node}" for node in ("node1", "node2", "node3")}})
        self.contract = {"schema": evidence.SCHEMA, "suite": self.DEFINITION["name"],
            "suite_sha256": suite.digest(self.DEFINITION), "step": "model",
            "candidate": {"client_commit": "a" * 40, "image_id": "sha256:" + "b" * 64,
                "vllm_head": "c" * 40, "vllm_tree": "d" * 40,
                "nodes": ["node1", "node2", "node3"], "profiles": {"candidate": "candidate.json"}},
            "reference_source": {"client_commit": "e" * 40, "image_id": "sha256:" + "f" * 64,
                                 "vllm_head": "1" * 40, "vllm_tree": "2" * 40},
            "references": refs, "corpus_sha256": self.data["corpus"]["sha256"], "positions": 4,
            "workload_shape": {"prompt_count": 2, "concurrency": 2, "max_tokens": 2, "fixed_length": False, "top": 2}}
        candidate = self.contract["candidate"]
        self.metadata = {"identity": {"client_commit": candidate["client_commit"], "client_dirty": False,
            "cluster_config": "candidate.json", "live_problems": [],
            "nodes": {name: {"image_id": candidate["image_id"], "checkout": candidate["client_commit"],
                       "image_labels": {"local.spark-ds41f.vllm.tree": candidate["vllm_tree"],
                                        "local.spark-ds41f.vllm.patch-head": candidate["vllm_head"]}}
                      for name in candidate["nodes"]}}}
        self.binding = self.bind()

    @staticmethod
    def model():
        sequences = [{"id": i, "prompt_ids": [10 + i], "output_ids": [1, 2]} for i in range(2)]
        scores = [{"id": i, "tokens": [1, 2], "logprob": [-0.2, -0.3], "rank": [1, 1],
                   "top_ids": [[1, 2], [2, 1]], "top_logprobs": [[-0.2, -1.7], [-0.3, -1.4]]}
                  for i in range(2)]
        return {"schema": behaviour.SCHEMA, "workload": {"prompt_count": 2,
            "prompts_sha256": "0" * 64, "concurrency": 2, "max_tokens": 2,
            "fixed_length": False, "top": 2, "extra_body": {}},
            "generation": {"sequences": sequences}, "scores": scores,
            "corpus": {"source": "self", "sha256": behaviour.corpus_digest(sequences), "sequences": 2}}

    def bind(self):
        path = self.root / "contract.json"
        provenance = {"execution_commit": self.contract["reference_source"]["client_commit"],
            "image": self.contract["reference_source"]["image_id"],
            "performance_analysis": {"path": "saved-analysis.json", "sha256": "3" * 64},
            "export_receipt": {"path": "saved-export.json", "sha256": "4" * 64},
            "references": [{"label": ref["label"], "boot_started_at": ref["boot_started_at"],
                "artifact": {"source": ref["path"], "size": ref["bytes"], "sha256": ref["sha256"]}}
                for ref in self.contract["references"]]}
        provenance_path = self.root / "provenance.json"
        data = json.dumps(provenance).encode()
        provenance_path.write_bytes(data)
        self.contract["reference_provenance"] = {"path": str(provenance_path), "sha256": hashlib.sha256(data).hexdigest()}
        raw = json.dumps(self.contract).encode()
        path.write_bytes(raw)
        return evidence.load_binding(path, hashlib.sha256(raw).hexdigest())

    def run_suite(self, result=None, model_exit=0, binding=True):
        calls = []
        def fake(argv):
            calls.append(argv[0])
            if argv[0] == "model":
                cli.write_json(argv[argv.index("--output") + 1], result if result is not None else self.data)
                return model_exit
            return 0
        out = self.root / ("run-" + str(len(list(self.root.glob("run-*")))))
        index = suite.run(fake, "http://unused", self.DEFINITION, str(out), label="candidate",
                          metadata=self.metadata, log=lambda _: None,
                          model_evidence=self.binding if binding else None)
        return calls, index, out

    def test_valid_existing_model_runs_once_and_keeps_suite_definition(self):
        original = copy.deepcopy(self.DEFINITION)
        calls, index, out = self.run_suite()
        self.assertEqual(calls, ["model", "engine", "run"])
        self.assertEqual(index["suite_sha256"], suite.digest(original))
        self.assertEqual(index["suite"], original)
        self.assertEqual(self.DEFINITION, original)
        self.assertEqual([step["exit"] for step in index["steps"]], [0, 0, 0])
        self.assertTrue(index["model_evidence"]["required_evidence_passed"])
        self.assertTrue((out / "model.json.gz.evidence.json").is_file())

    def test_failed_model_process_stops_before_grid_and_preserves_native_exit(self):
        calls, index, out = self.run_suite(model_exit=1)
        self.assertEqual(calls, ["model"])
        self.assertEqual(index["steps"][0]["exit"], 1)
        self.assertTrue(index["stopped_before_timing"])
        report = json.loads((out / "model.json.gz.evidence.json").read_text())
        self.assertEqual(report["process_exit"], 1)
        self.assertFalse(report["required_evidence_passed"])
        self.assertIn("artifact", report)

    def test_zero_exit_wrong_model_semantics_stops_before_grid(self):
        for change in ("tokens", "score", "missing", "nonfinite"):
            data = copy.deepcopy(self.data)
            if change == "tokens": data["generation"]["sequences"][0]["output_ids"][0] = 9
            if change == "score": data["scores"][0]["logprob"][0] -= 0.1
            if change == "missing": data["scores"].pop()
            if change == "nonfinite": data["scores"][0]["logprob"][0] = float("nan")
            with self.subTest(change=change):
                calls, index, out = self.run_suite(result=data)
                self.assertEqual(calls, ["model"])
                self.assertEqual(index["steps"][0]["exit"], 0)
                self.assertFalse(index["model_evidence"]["required_evidence_passed"])
                self.assertTrue((out / "model.json.gz").is_file())

    def test_default_suite_behavior_still_runs_remaining_steps_after_model_failure(self):
        calls, index, _ = self.run_suite(model_exit=1, binding=False)
        self.assertEqual(calls, ["model", "engine", "run"])
        self.assertNotIn("model_evidence", index)

    def test_saved_reference_hash_source_or_missing_reference_refuses_before_model(self):
        for change in ("hash", "source", "missing"):
            with self.subTest(change=change):
                metadata, binding = copy.deepcopy(self.metadata), copy.deepcopy(self.binding)
                if change == "hash": binding["contract"]["references"][0]["sha256"] = "9" * 64
                if change == "source": metadata["identity"]["nodes"]["node1"]["image_id"] = "sha256:" + "9" * 64
                if change == "missing": binding["contract"]["references"][0]["path"] += ".absent"
                callback = mock.Mock(return_value=0)
                with self.assertRaises((OSError, ValueError)):
                    suite.run(callback, "http://unused", self.DEFINITION,
                        str(self.root / ("preflight-" + change)), label="candidate", metadata=metadata,
                        model_evidence=binding, log=lambda _: None)
                callback.assert_not_called()

    def test_different_saved_boot_cannot_be_hidden_by_selecting_one_reference(self):
        second = self.contract["references"][1]
        changed = copy.deepcopy(self.data)
        changed["scores"][0]["logprob"][0] -= 0.1
        path = Path(second["path"])
        cli.write_json(str(path), changed)
        raw = path.read_bytes()
        second.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        binding = self.bind()
        with self.assertRaisesRegex(ValueError, "reference boots differ"):
            evidence.prepare(binding, self.DEFINITION, suite.digest(self.DEFINITION), "candidate", self.metadata)

    def test_prior_opted_in_model_evidence_is_never_overwritten(self):
        _, _, out = self.run_suite()
        callback = mock.Mock(return_value=0)
        with self.assertRaises(FileExistsError):
            suite.run(callback, "http://unused", self.DEFINITION, str(out), label="candidate",
                metadata=self.metadata, model_evidence=self.binding, log=lambda _: None)
        callback.assert_not_called()

    def test_opt_in_environment_requires_exact_contract_hash_and_cli_returns_failure(self):
        environment = {evidence.CONTRACT_ENV: self.binding["path"], evidence.HASH_ENV: self.binding["sha256"]}
        with mock.patch.dict(os.environ, environment):
            self.assertEqual(evidence.from_environment()["sha256"], self.binding["sha256"])
        with mock.patch.dict(os.environ, {evidence.CONTRACT_ENV: self.binding["path"], evidence.HASH_ENV: "9" * 64}):
            with self.assertRaises(ValueError): evidence.from_environment()
        definition, metadata = self.root / "definition.json", self.root / "metadata.json"
        definition.write_text(json.dumps(self.DEFINITION)); metadata.write_text(json.dumps(self.metadata))
        data = copy.deepcopy(self.data); data["scores"][0]["logprob"][0] -= 0.1
        def fake(argv):
            self.assertEqual(argv[0], "model")
            cli.write_json(argv[argv.index("--output") + 1], data)
            return 0
        args = argparse.Namespace(suite=str(definition), metadata=str(metadata), url="http://unused",
                                  out=str(self.root / "cli-run"), label="candidate", corpus=None)
        with mock.patch.dict(os.environ, environment), mock.patch.object(cli, "main", side_effect=fake):
            self.assertEqual(cli.command_suite(args), 1)

    def test_full_c46_builtin_plan_stays_26_operations_and_release_digest_is_unchanged(self):
        definitions = [suite.load(name) for name in ("release-tp3", "single-stream", "sampled")]
        self.assertEqual(2 * sum(len(definition["steps"]) for definition in definitions), 26)
        self.assertEqual(suite.digest(definitions[0]), "4f54f4b56f8d323473c7725d428caa2a4a76e04b9e728cb374fd092115669f03")

    def test_rank_changes_are_observed_without_a_new_equality_requirement(self):
        data = copy.deepcopy(self.data)
        data["scores"][0]["rank"] = [7, 8]
        calls, index, out = self.run_suite(result=data)
        self.assertEqual(calls, ["model", "engine", "run"])
        self.assertTrue(index["model_evidence"]["required_evidence_passed"])
        report = json.loads((out / "model.json.gz.evidence.json").read_text())
        self.assertNotEqual(report["observed"]["rank"]["sha256"],
                            report["references"][0]["observed"]["rank"]["sha256"])

    def test_score_fingerprint_matches_existing_numeric_equality_and_order(self):
        base, candidate = copy.deepcopy(self.data), copy.deepcopy(self.data)
        base["scores"][0]["logprob"] = [0, 1]
        candidate["scores"][0]["logprob"] = [-0.0, 1.0]
        base["scores"][0]["top_logprobs"][0] = [0, 1]
        candidate["scores"][0]["top_logprobs"][0] = [-0.0, 1.0]
        candidate["scores"][0]["rank"] = [99, None]
        self.assertEqual(behaviour.compare_scores(base["scores"], candidate["scores"])["identical_fraction"], 1)
        self.assertEqual(evidence.fingerprints(base)["scores_sha256"], evidence.fingerprints(candidate)["scores_sha256"])
        candidate["scores"][0]["top_ids"][0].reverse()
        self.assertLess(behaviour.compare_scores(base["scores"], candidate["scores"])["identical_fraction"], 1)
        self.assertNotEqual(evidence.fingerprints(base)["scores_sha256"], evidence.fingerprints(candidate)["scores_sha256"])
        candidate = copy.deepcopy(base); candidate["scores"][0]["rank"] = []
        self.assertEqual(evidence.fingerprints(base)["scores_sha256"], evidence.fingerprints(candidate)["scores_sha256"])
        self.assertFalse(evidence.rank_observation(candidate)["shape_valid"])

    def test_model_suite_mismatch_refuses_but_model_free_companion_is_allowed(self):
        other = copy.deepcopy(self.DEFINITION); other["name"] = "unexpected-model-suite"
        callback = mock.Mock(return_value=0)
        with self.assertRaisesRegex(ValueError, "suite definition differs"):
            suite.run(callback, "http://unused", other, str(self.root / "mismatch"),
                      label="candidate", metadata=self.metadata, model_evidence=self.binding, log=lambda _: None)
        callback.assert_not_called()
        companion = {"name": "companion", "steps": [{"id": "c1", "command": "run", "args": {}}]}
        index = suite.run(callback, "http://unused", companion, str(self.root / "companion"),
                          label="candidate", metadata=self.metadata, model_evidence=self.binding, log=lambda _: None)
        self.assertEqual(len(index["steps"]), 1)
        self.assertNotIn("model_evidence", index)

    def test_encoded_size_is_checked_before_json_and_decoded_bytes_are_capped(self):
        path = self.root / "over.json"
        path.write_bytes(b"{}" + b" " * 100)
        with mock.patch.object(evidence, "MAX_ENCODED_BYTES", 16), mock.patch.object(evidence.json, "load") as parse:
            with self.assertRaisesRegex(ValueError, "before parsing"):
                evidence.read_model(path)
            parse.assert_not_called()
        path = self.root / "expanded.json.gz"
        path.write_bytes(gzip.compress(b" " * 100 + b"{}"))
        with mock.patch.object(evidence, "MAX_DECODED_BYTES", 16):
            with self.assertRaisesRegex(ValueError, "decoded model byte limit"):
                evidence.read_model(path)

    def test_truncated_and_deep_json_return_bounded_worker_failure_receipts(self):
        for name, data in (("truncated", gzip.compress(json.dumps(self.data).encode())[:-8]),
                           ("deep", gzip.compress(b"[" * 2000 + b"]" * 2000))):
            path = self.root / (name + ".json.gz"); path.write_bytes(data)
            with self.subTest(name=name), self.assertRaises(evidence.ParserFailure) as caught:
                evidence.bounded_fingerprints(path)
            self.assertNotEqual(caught.exception.receipt["exit"], 0)
            self.assertIn("actual_limits", caught.exception.receipt)
            self.assertIn("EOFError" if name == "truncated" else "RecursionError", caught.exception.receipt["parse_error"])

    def test_evaluate_records_eof_memory_and_recursion_failures_without_hash_scan(self):
        prepared = evidence.prepare(self.binding, self.DEFINITION, suite.digest(self.DEFINITION), "candidate", self.metadata)
        path = self.root / "native.json.gz"; cli.write_json(str(path), self.data)
        for error in (EOFError("truncated"), MemoryError("bounded"), RecursionError("nested")):
            with self.subTest(error=type(error).__name__), mock.patch.object(evidence, "bounded_fingerprints", side_effect=error):
                report = evidence.evaluate(path, 0, prepared)
            self.assertFalse(report["required_evidence_passed"])
            self.assertEqual(report["process_exit"], 0)
            self.assertIsNone(report["artifact"]["sha256"])

    def test_native_outcome_is_persisted_before_evaluator_failure(self):
        def fail_after_receipt(output, code, _prepared):
            index = json.loads((Path(output).parent / "suite.json").read_text())
            self.assertEqual(index["steps"][0]["exit"], code)
            raise MemoryError("forced evaluator failure")
        with mock.patch.object(evidence, "evaluate", side_effect=fail_after_receipt):
            calls, index, out = self.run_suite()
        self.assertEqual(calls, ["model"])
        self.assertEqual(index["steps"][0]["exit"], 0)
        report = json.loads((out / "model.json.gz.evidence.json").read_text())
        self.assertFalse(report["required_evidence_passed"])
        self.assertEqual(report["process_exit"], 0)

    def test_parser_limits_apply_only_to_child_and_reference_boots_are_distinct(self):
        import resource
        affinity, address_space = os.sched_getaffinity(0), resource.getrlimit(resource.RLIMIT_AS)
        calls, _, out = self.run_suite()
        self.assertEqual(calls, ["model", "engine", "run"])
        self.assertEqual(os.sched_getaffinity(0), affinity)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_AS), address_space)
        report = json.loads((out / "model.json.gz.evidence.json").read_text())
        limits = report["parser_worker"]["actual_limits"]
        self.assertEqual(len(limits["actual_affinity"]), 1)
        self.assertTrue(0 < limits["actual_address_space"][0] == limits["actual_address_space"][1] <= evidence.PARSER_AS_BYTES)
        self.assertTrue(0 < limits["actual_cpu_seconds"][0] <= evidence.PARSER_CPU_SECONDS)
        self.assertTrue(limits["actual_cpu_seconds"][0] <= limits["actual_cpu_seconds"][1] <= evidence.PARSER_CPU_SECONDS + 5)
        self.contract["references"][1]["boot_started_at"] = self.contract["references"][0]["boot_started_at"].copy()
        with self.assertRaisesRegex(ValueError, "reuse a rank boot"):
            self.bind()

    def test_worker_timeout_kills_only_its_process_group_and_returns_failure(self):
        process = mock.Mock(pid=99123, returncode=-9)
        process.poll.side_effect = [None, -9]
        with mock.patch.object(evidence.subprocess, "Popen", return_value=process), \
                mock.patch.object(evidence, "_bounded_output", side_effect=TimeoutError("bounded wall")), \
                mock.patch.object(evidence.os, "killpg") as kill:
            with self.assertRaises(evidence.ParserFailure) as caught:
                evidence.bounded_fingerprints(self.root / "unread.json.gz")
        kill.assert_called_once_with(99123, evidence.signal.SIGKILL)
        self.assertEqual(caught.exception.receipt["exit"], -9)
        self.assertEqual(caught.exception.receipt["requested_limits"]["wall_seconds"], evidence.PARSER_WALL_SECONDS)

    def test_published_reference_provenance_hash_cannot_be_replaced(self):
        Path(self.contract["reference_provenance"]["path"]).write_text('{}')
        with self.assertRaisesRegex(ValueError, "provenance bytes changed"):
            evidence.load_binding(self.binding["path"], self.binding["sha256"])


if __name__ == "__main__":
    unittest.main()
