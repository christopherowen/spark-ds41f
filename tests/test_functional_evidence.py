"""Required collector verdicts stop timing; observations retain their meaning."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from spark_ds41f import functional_evidence as evidence, lab

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sampled_collector", ROOT / evidence.SAMPLED_COLLECTOR)
sampled = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampled)

REQUIRED = {"kind": "summary", "label": "check", "require": {"identical": True}}
SAMPLED = {"kind": "sampled", "batch_invariance": "observe", "isolated_repeat": "require",
           "tokens": 192, "seed": 20261024, "cases": 8}


def sampled_rows(batch=True, repeat=False):
    scenarios = {"c8": {"differing": [0, 2] if batch else []},
                 "again": {"differing": [1] if repeat else []}}
    return [dict(scenario=name, **row, sha256=("b" if name == "c8" else "c") * 64 if row["differing"] else "a" * 64)
            for name, row in scenarios.items()] + [
        {"summary": "seeded temperature-1 digest", "identical": not (batch or repeat),
         "seed": 20261024, "tokens": 192, "scenarios": scenarios, "reference_sha256": "a" * 64}]


class VerdictTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def evaluate(self, rows, contract=REQUIRED):
        path = self.root / "gate.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return evidence.evaluate(path, contract)

    def test_required_false_is_not_process_success(self):
        self.assertFalse(self.evaluate([{"summary": "check", "identical": False}])["required_evidence_passed"])
        self.assertTrue(self.evaluate([{"summary": "check", "identical": True}])["required_evidence_passed"])

    def test_missing_ambiguous_malformed_and_wrongly_typed_verdicts_fail(self):
        for rows in ([], [{"summary": "check"}], [{"summary": "check", "identical": "true"}],
                     [{"summary": "check", "identical": 1}],
                     [{"summary": "check", "identical": True}] * 2):
            with self.subTest(rows=rows):
                self.assertFalse(self.evaluate(rows)["required_evidence_passed"])
        for text in ('{"summary":', 'null\n', '{"summary":"check","identical":NaN}\n'):
            path = self.root / "broken.jsonl"; path.write_text(text)
            self.assertFalse(evidence.evaluate(path, REQUIRED)["required_evidence_passed"])
        self.assertFalse(evidence.evaluate(self.root / "absent", REQUIRED)["required_evidence_passed"])

    def test_observational_batch_difference_is_neither_error_nor_invariance_pass(self):
        result = self.evaluate(sampled_rows(), SAMPLED)
        self.assertTrue(result["required_evidence_passed"])
        self.assertEqual(result["required"], {"isolated_repeat": {"matches": True, "differing": []}})
        self.assertEqual(result["observed"], {"batch_invariance": {"matches": False, "differing": [0, 2]}})
        strict = dict(SAMPLED, batch_invariance="require")
        self.assertFalse(self.evaluate(sampled_rows(), strict)["required_evidence_passed"])

    def test_required_isolated_repeat_cannot_be_excused_as_observed_batch_variation(self):
        result = self.evaluate(sampled_rows(repeat=True), SAMPLED)
        self.assertFalse(result["required_evidence_passed"])
        self.assertIn("required sampled isolated_repeat failed", result["errors"])

    def test_sampled_evidence_must_be_complete_and_consistent(self):
        for change in ("missing_row", "duplicate_row", "bad_digest", "wrong_seed", "wrong_tokens", "false_pass", "bad_index"):
            rows = sampled_rows()
            if change == "missing_row": rows.pop(0)
            if change == "duplicate_row": rows.insert(0, copy.deepcopy(rows[0]))
            if change == "bad_digest": rows[0]["sha256"] = "bad"
            if change == "wrong_seed": rows[-1]["seed"] += 1
            if change == "wrong_tokens": rows[-1]["tokens"] += 1
            if change == "false_pass": rows[-1]["identical"] = True
            if change == "bad_index": rows[0]["differing"] = [99]
            with self.subTest(change=change):
                self.assertFalse(self.evaluate(rows, SAMPLED)["required_evidence_passed"])
        self.assertFalse(self.evaluate(sampled_rows(), dict(SAMPLED, record_requests=True))["required_evidence_passed"])

    def test_contracts_are_explicit_and_reject_typos_before_boot(self):
        for contract in ({}, {"kind": "summary"}, dict(SAMPLED, isolated_repeat="ignore"),
                         dict(SAMPLED, require_batch=True), dict(SAMPLED, tokens=True)):
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                evidence.validate_contract(contract)
        with self.assertRaises(ValueError):
            evidence.contracts_for_extras({"evidence": {"typo": REQUIRED}}, [["a.py", [], "a"]])


class AdvancementTests(unittest.TestCase):
    def test_real_zero_exit_collector_failure_stops_before_timing(self):
        for output in ('{"summary":"check","identical":false}', '{"summary":"check"}', '{broken'):
            with self.subTest(output=output), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "collector.py").write_text("print(" + repr(output) + ")\n")
                step = {"kind": "script", "argv": ["collector.py"], "out": "gate.jsonl",
                        "label": "candidate", "telemetry": False, "evidence": REQUIRED}
                timing = {"kind": "cli", "argv": ["bench"], "label": "candidate"}
                with mock.patch.object(lab, "ROOT", root), mock.patch.object(lab, "plan", return_value=[step, timing]), \
                        mock.patch.object(lab, "read_hold", return_value={"holder": "spark-lab: fixture"}), \
                        mock.patch.object(lab, "window_should_close", return_value=None), mock.patch.object(lab, "beat"), \
                        mock.patch.object(lab, "save_engine_logs"), mock.patch.object(lab, "window_close") as close, \
                        mock.patch.object(lab, "spark_cli") as timed, mock.patch.object(lab, "Telemetry", return_value=contextlib.nullcontext()), \
                        contextlib.redirect_stdout(io.StringIO()):
                    code = lab.execute({"run": "fixture"}, dry=False, keep_open=True)
                self.assertEqual(code, 1)
                timed.assert_not_called()
                close.assert_called_once()
                report = json.loads((root / "gate.jsonl.evidence.json").read_text())
                self.assertEqual(report["process_exit"], 0)
                self.assertFalse(report["required_evidence_passed"])

    def test_observation_only_false_is_retained_and_does_not_change_collector_semantics(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(lab, "ROOT", Path(directory)):
            root = Path(directory); (root / "collector.py").write_text('print(\'{"summary":"check","identical":false}\')\n')
            step = {"argv": ["collector.py"], "out": "gate.jsonl",
                    "evidence": {"kind": "summary", "label": "check", "observe": ["identical"]}}
            self.assertEqual(lab.run_script(step), 0)
            report = json.loads((root / "gate.jsonl.evidence.json").read_text())
            self.assertEqual(report["status"], "observed")
            self.assertEqual(report["observed"], {"identical": False})
            with self.assertRaises(FileExistsError): lab.run_script(step)

    def test_active_plan_records_sampled_requests_and_declares_the_historical_contract(self):
        spec = json.loads((ROOT / "experiments/2026-11-05-frontend-lifetime-qualification/functional.json").read_text())
        with mock.patch.object(lab, "head_url", return_value="http://fixture:8000"):
            steps = lab.plan(spec)
        gates = [step for step in steps if "evidence" in step]
        self.assertEqual(len(gates), 8)
        sampled_steps = [step for step in gates if step["evidence"]["kind"] == "sampled"]
        self.assertEqual(len(sampled_steps), 2)
        for step in sampled_steps:
            self.assertEqual(step["evidence"]["batch_invariance"], "observe")
            self.assertEqual(step["evidence"]["isolated_repeat"], "require")
            self.assertIn("--record-dir", step["argv"])

    def test_contract_cannot_be_placed_after_the_default_timing_step(self):
        spec = json.loads((ROOT / "experiments/2026-11-05-frontend-lifetime-qualification/functional.json").read_text())
        spec["jobs"][0].pop("bench")
        with self.assertRaisesRegex(ValueError, "bench:false"):
            lab.plan(spec)

    def test_existing_request_records_are_rejected_before_boot(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(lab, "ROOT", Path(directory)):
            root = Path(directory)
            (root / "collector.py").touch()
            (root / "sampled.requests").mkdir()
            step = {"kind": "script", "argv": ["collector.py"], "out": "sampled.jsonl",
                    "evidence": dict(SAMPLED, record_requests=True)}
            self.assertTrue(any("prior required evidence" in issue for issue in lab.plan_problems([step])))


class Response(io.BytesIO):
    status = 200
    headers = {"x-request-id": "server-request", "authorization": "must-not-be-recorded"}


class RecorderTests(unittest.TestCase):
    def test_complete_recording_links_arrays_to_observed_digests_and_detects_tampering(self):
        def respond(request, **kwargs):
            if isinstance(request, str):
                return Response(b'{"data":[{"id":"fixture"}]}')
            body = json.loads(request.data)
            batch_difference = threading.current_thread() is not threading.main_thread() and body["seed"] == 20261024
            payload = {"id": "response-" + request.get_header("X-request-id"),
                       "usage": {"prompt_tokens": 12},
                       "choices": [{"token_ids": [4, 5], "logprobs": {"content": [
                           {"token": "a", "logprob": -1.1 if batch_difference else -1.0},
                           {"token": "b", "logprob": -2.0}]}}]}
            return Response(json.dumps(payload).encode())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "sampled.jsonl"; records = output.with_suffix(".requests")
            argv = ["sampled_digest.py", "http://fixture", "--tokens", "2", "--record-dir", str(records)]
            with mock.patch.object(sampled.urllib.request, "urlopen", side_effect=respond), \
                    mock.patch.object(sampled, "__name__", "sampled_fixture"), \
                    mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()) as stream:
                sampled.main()
            output.write_text(stream.getvalue())
            contract = dict(SAMPLED, tokens=2, record_requests=True)
            result = evidence.evaluate(output, contract)
            self.assertTrue(result["required_evidence_passed"], result)
            self.assertEqual(result["observed"]["batch_invariance"]["differing"], [0])
            rows = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual(rows[0]["token_differing"], [])
            self.assertEqual(rows[0]["logprob_differing"], [0])
            self.assertEqual(result["request_records"]["requests"], 24)
            changed = records / "c8-0.json"
            changed.write_text(changed.read_text() + " ")
            result = evidence.evaluate(output, contract)
            self.assertFalse(result["required_evidence_passed"])
            self.assertIn("request evidence hash differs", result["errors"][0])

    def test_token_and_logprob_differences_are_distinct(self):
        reference = [{"tokens": ["a", "b"], "logprobs": [-1.0, -2.0]}] * 3
        values = [{"tokens": ["a", "c"], "logprobs": [-1.0, -2.0]},
                  {"tokens": ["a", "b"], "logprobs": [-1.0, -2.1]}, reference[2]]
        report = sampled.compare(values, reference)
        self.assertEqual(report["differing"], [0, 1])
        self.assertEqual(report["token_differing"], [0])
        self.assertEqual(report["logprob_differing"], [1])
        self.assertEqual(report["cases"][0]["first_token_diff"], 1)
        self.assertIsNone(report["cases"][0]["first_logprob_diff"])

    def test_private_recording_keeps_identity_controls_arrays_and_unavailable_rng_dimensions(self):
        payload = {"id": "chatcmpl-response", "model": "fixture", "usage": {"prompt_tokens": 12},
                   "choices": [{"token_ids": [4, 5], "logprobs": {"content": [
                       {"token": "a", "logprob": -1.0}, {"token": "b", "logprob": -2.0}]}}]}
        def respond(request, **kwargs):
            value = {"data": [{"id": "fixture"}]} if isinstance(request, str) else payload
            return Response(json.dumps(value).encode())
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(sampled.urllib.request, "urlopen", side_effect=respond):
            records = Path(directory) / "private"
            client = sampled.Client("http://user:password@fixture:8000", records)
            value = client.complete("prompt", 2, 9, scenario="alone", case=0)
            self.assertEqual(value, {"tokens": ["a", "b"], "logprobs": [-1.0, -2.0]})
            record = json.loads((records / "alone-0.json").read_text())
            self.assertEqual(record["native_token_ids"], [4, 5])
            self.assertEqual(record["response_id"], "chatcmpl-response")
            self.assertEqual(record["request"]["seed"], 9)
            self.assertEqual(record["output_indexes"], [0, 1])
            self.assertEqual(record["prompt_tokens"], 12)
            self.assertIsNone(record["rng_identity"]["native_logit_positions"])
            self.assertIsNone(record["rng_identity"]["local_speculative_steps"])
            self.assertNotIn("password", record["endpoint"])
            self.assertNotIn("authorization", record["response_headers"])
            self.assertNotIn("must-not-be-recorded", (records / "alone-0.json").read_text())
            with self.assertRaises(FileExistsError): sampled.Client("http://fixture", records)

    def test_recorded_http_failure_preserves_request_identity(self):
        error = sampled.urllib.error.HTTPError("http://fixture", 400, "bad", {}, io.BytesIO(b'{"error":"bad request"}'))
        responses = [Response(b'{"data":[{"id":"fixture"}]}'), error]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(sampled.urllib.request, "urlopen", side_effect=responses):
            records = Path(directory) / "private"; client = sampled.Client("http://fixture", records)
            with self.assertRaises(sampled.urllib.error.HTTPError):
                client.complete("prompt", 2, 9, scenario="alone", case=0)
            record = json.loads((records / "alone-0.json").read_text())
            self.assertEqual(record["http_status"], 400)
            self.assertIn("HTTPError", record["error"])
            self.assertEqual(record["request"]["seed"], 9)
            self.assertTrue(record["client_request_id"])


if __name__ == "__main__":
    unittest.main()
