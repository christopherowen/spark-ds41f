import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import run_hardware as runner
import tool_schema_gate

SAMPLED_SPEC = importlib.util.spec_from_file_location(
    "sampled_collector", runner.ROOT / runner.evidence.SAMPLED_COLLECTOR)
sampled = importlib.util.module_from_spec(SAMPLED_SPEC)
SAMPLED_SPEC.loader.exec_module(sampled)


class FakeResponse(io.BytesIO):
    status = 200
    headers = {"x-request-id": "server-request"}


class GateTests(unittest.TestCase):
    def test_predeclared_evidence_contracts_are_valid_and_complete(self):
        runner.validate_declared_contracts()
        self.assertEqual(runner.SAMPLED_CONTRACT["batch_invariance"], "observe")
        self.assertEqual(runner.SAMPLED_CONTRACT["isolated_repeat"], "require")
        self.assertTrue(runner.SAMPLED_CONTRACT["record_requests"])
        with mock.patch.object(runner.lab, "head_url", return_value="http://fixture:8000"):
            argv = runner.gate_argv("sampled", runner.evidence.SAMPLED_COLLECTOR,
                                    ["--tokens", "192"], Path("/tmp/row010-a-sampled.jsonl"))
        self.assertEqual(argv[-2:], ["--record-dir", "/tmp/row010-a-sampled.requests"])

    def test_one_arm_reservation_finishes_current_arm_then_yields_without_a_request(self):
        self.assertIsNone(runner.handoff_reason(0, 1, "pending request"))
        self.assertEqual(runner.handoff_reason(1, 1, None), runner.ARM_LIMIT)
        self.assertIsNone(runner.handoff_reason(1, 2, None))
        self.assertEqual(runner.handoff_reason(1, None, "pending request"), runner.HANDOFF_REQUEST)

    def test_parent_corpus_requires_the_saved_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.gz"
            expected = hashlib.sha256(b"saved corpus").hexdigest()
            with self.assertRaises(FileNotFoundError):
                runner.verify_corpus(path, expected)
            path.write_bytes(b"saved corpus")
            self.assertEqual(runner.verify_corpus(path, expected), path)
            path.write_bytes(b"different corpus")
            with self.assertRaises(ValueError):
                runner.verify_corpus(path, expected)

    def test_resume_accepts_only_restored_ordered_handoff_at_identical_inputs(self):
        arms = [{"label": "row010-a", "config": "row010.json"},
                {"label": "control035-a", "config": "control.json"}]
        identity = {"commit": "a" * 40, "stage": "screen", "gate_sources": {"gate": "hash"},
                    "evidence_contracts": runner.EVIDENCE_CONTRACTS,
                    "images": {"dgx1": "digest"}, "restore_profile": runner.RESTORE}
        previous = {**identity, "completed": False, "restored": True,
                    "error": "RuntimeError: handoff requested after completed arm",
                    "completed_arms": [{"label": "row010-a", "config": f"{runner.EXPERIMENT}/row010.json"}]}
        self.assertEqual(runner.resume_arms(previous, identity, arms), previous["completed_arms"])
        self.assertEqual(runner.resume_arms({**previous, "error": f"RuntimeError: {runner.ARM_LIMIT}"},
                                           identity, arms), previous["completed_arms"])
        for change in ({"completed": True}, {"restored": False}, {"error": "gate failed"},
                       {"commit": "b" * 40}, {"evidence_contracts": {}}, {"gate_sources": {}}, {"images": {}},
                       {"completed_arms": []}, {"completed_arms": [{"label": "control035-a"}]},
                       {"restore_error": "failed"}):
            with self.assertRaises(ValueError):
                runner.resume_arms({**previous, **change}, identity, arms)

    def test_tool_and_schema_content_must_match_and_have_native_ids(self):
        expected = {"label": "record-0", "value": 7}
        message = {"content": json.dumps(expected), "tool_calls": [{"function": {
            "name": "record", "arguments": json.dumps(expected)}}]}
        choice = {"message": message, "token_ids": [1, 2, 3], "finish_reason": "stop"}
        reply = {"choices": [choice]}
        for mode in ("schema", "tool"):
            self.assertEqual(tool_schema_gate.normalize(reply, mode, expected)["value"], expected)
            with self.assertRaises(ValueError):
                tool_schema_gate.normalize(reply, mode, {"label": "wrong", "value": 7})
        with self.assertRaises(ValueError):
            tool_schema_gate.normalize({"choices": [{**choice, "token_ids": None}]}, "schema", expected)
        with self.assertRaises(ValueError):
            tool_schema_gate.normalize({"choices": [{**choice, "finish_reason": "length"}]}, "tool", expected)

    def test_tool_schema_gate_enforces_isolated_repeat_and_observes_t1_batch_tokens(self):
        def make_post(fail_repeat=False, fail_t0_batch=False):
            calls = []
            def fake_post(base, body):
                if body.get("max_tokens") == -1:
                    raise tool_schema_gate.urllib.error.HTTPError(base, 400, "bad", {}, io.BytesIO(b"{}"))
                if body.get("max_tokens") == 1:
                    return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
                index = body["seed"] - 20261106
                expected = {"label": f"record-{index}", "value": index + 7}
                batched = threading.current_thread() is not threading.main_thread()
                calls.append((body["temperature"], batched, index))
                repeat_count = sum(1 for t, b, i in calls if t == body["temperature"] and not b and i == index)
                tokens = [10, 20, index]
                if body["temperature"] == 1.0 and batched:
                    tokens = [10, 21, index]
                if fail_t0_batch and body["temperature"] == 0.0 and batched:
                    tokens = [10, 99, index]
                if fail_repeat and body["temperature"] == 1.0 and not batched and repeat_count > 1:
                    tokens = [10, 88, index]
                message = {"content": json.dumps(expected), "tool_calls": [{"function": {
                    "name": "record", "arguments": json.dumps(expected)}}]}
                return {"choices": [{"message": message, "token_ids": tokens, "finish_reason": "stop"}]}
            return fake_post

        with mock.patch.object(tool_schema_gate.urllib.request, "urlopen",
                               return_value=FakeResponse(b'{"data":[{"id":"fixture"}]}')):
            with mock.patch.object(tool_schema_gate, "post", side_effect=make_post()), \
                    contextlib.redirect_stdout(io.StringIO()) as stream:
                signatures = tool_schema_gate.run("http://fixture")
            text = stream.getvalue() + json.dumps({"summary": "tool-schema gate", "passed": True,
                                                   "signatures": signatures}) + "\n"
            parsed = runner.gate_summary(text, "tool-schema")
            self.assertEqual(len(parsed), 13)
            self.assertEqual(parsed["schema/t0.0/c1"], parsed["schema/t0.0/c8"])
            self.assertNotEqual(parsed["schema/t1.0/c1"], parsed["schema/t1.0/c8"])
            for kwargs in ({"fail_repeat": True}, {"fail_t0_batch": True}):
                with mock.patch.object(tool_schema_gate, "post", side_effect=make_post(**kwargs)), \
                        contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    tool_schema_gate.run("http://fixture")

    def test_resolved_profiles_change_only_row_cost(self):
        def normalized(name):
            config = runner.lab.spark.read_cluster(f"{runner.EXPERIMENT}/{name}.json")
            config.pop("notes", None)
            index = config["serve_args"].index("--speculative-config") + 1
            speculative = json.loads(config["serve_args"][index])
            cost = speculative.pop("adaptive_verification_row_cost")
            config["serve_args"][index] = speculative
            return config, cost
        control, cost = normalized("control")
        self.assertEqual(cost, .035)
        for name, expected in (("row010", .01), ("row020", .02)):
            candidate, cost = normalized(name)
            self.assertEqual(cost, expected)
            self.assertEqual(candidate, control)

    def test_false_summary_is_failure_even_when_process_exits_zero(self):
        with self.assertRaises(ValueError):
            runner.gate_summary(json.dumps({"summary": "greedy", "identical": False}), "greedy")

    def test_missing_or_duplicate_summary_fails(self):
        for text in ("", "{}", '{"summary":"x"}\n{"summary":"x"}'):
            with self.assertRaises(ValueError):
                runner.gate_summary(text, "greedy")

    def test_control_batch_mismatch_or_failure_fails_while_penalties_sampled_c8_is_observed(self):
        for result in ({"failed": ["json"], "digests": {"json/alone": "a" * 64}},
                       {"failed": [], "digests": {"json/alone": "a" * 64, "json/c8": "b" * 64}}):
            with self.assertRaises(ValueError):
                runner.gate_summary(json.dumps({"summary": "controls digests", **result}), "controls")
        digests = {
            "penalties/alone": "a" * 64,
            "penalties/c8": "a" * 64,
            "penalties-sampled/alone": "b" * 64,
            "penalties-sampled/c8": "c" * 64,
            "min-p/rejected": "d" * 64,
            "logit-bias/rejected": "d" * 64,
            "bad-words/alone": "e" * 64,
            "bad-words/c8": "e" * 64,
            "thinking-budget/alone": "f" * 64,
            "thinking-budget/c8": "f" * 64,
            "logprob-ids/alone": "1" * 64,
            "logprob-ids/c8": "1" * 64,
        }
        parsed = runner.gate_summary(json.dumps({"summary": "controls digests", "tokens": 96,
                                                 "seed": 20261024, "failed": [], "digests": digests}), "controls")
        self.assertEqual(parsed, digests)
        with self.assertRaises(ValueError):
            runner.gate_summary(json.dumps({"summary": "controls digests", "tokens": 96, "seed": 20261024,
                                            "failed": [], "digests": {**digests, "penalties/c8": "9" * 64}}),
                                "controls")

    def test_sampled_gate_requires_isolated_repeat_and_records_c8_with_per_case_arrays(self):
        def make_responder(repeat_differs=False):
            calls = []
            def respond(request, **kwargs):
                if isinstance(request, str):
                    return FakeResponse(b'{"data":[{"id":"fixture"}]}')
                body = json.loads(request.data)
                batched = threading.current_thread() is not threading.main_thread()
                calls.append((batched, body["seed"]))
                solo_calls = sum(1 for b, s in calls if not b and s == body["seed"])
                token = "b" if (batched and body["seed"] in (20261024, 20261026)) else "a"
                logprob = -1.2 if (batched and body["seed"] == 20261027) else -1.0
                if repeat_differs and not batched and solo_calls > 1 and body["seed"] == 20261024:
                    token = "c"
                content = [{"token": token, "logprob": logprob}] + [
                    {"token": "x", "logprob": -2.0} for _ in range(191)]
                payload = {"id": "response-" + request.get_header("X-request-id"),
                           "usage": {"prompt_tokens": 12},
                           "choices": [{"token_ids": list(range(192)), "logprobs": {"content": content}}]}
                return FakeResponse(json.dumps(payload).encode())
            return respond

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "row010-a-sampled.jsonl"
            records = output.with_suffix(".requests")
            argv = ["sampled_digest.py", "http://fixture", "--tokens", "192", "--record-dir", str(records)]
            with mock.patch.object(sampled.urllib.request, "urlopen", side_effect=make_responder()), \
                    mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()) as stream:
                sampled.main()
            output.write_text(stream.getvalue())
            summary, report = runner.evaluate_gate(output, "sampled", 0, [])
            self.assertTrue(report["required_evidence_passed"])
            self.assertEqual(summary["reference_sha256"], summary["isolated_repeat_sha256"])
            self.assertEqual(summary["observed_c8_differing"], [0, 2, 3])
            self.assertEqual(summary["observed_c8_token_differing"], [0, 2])
            self.assertEqual(summary["observed_c8_logprob_differing"], [3])
            self.assertEqual(report["request_records"]["requests"], 24)
            case0 = json.loads((records / "c8-0.json").read_text())
            self.assertEqual(len(case0["result"]["tokens"]), 192)
            self.assertEqual(len(case0["result"]["logprobs"]), 192)

            # A failed isolated repeat must fail the gate even when the process exits 0.
            failed_out = root / "failed-sampled.jsonl"
            failed_records = failed_out.with_suffix(".requests")
            argv[-1] = str(failed_records)
            with mock.patch.object(sampled.urllib.request, "urlopen", side_effect=make_responder(repeat_differs=True)), \
                    mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()) as stream:
                sampled.main()
            failed_out.write_text(stream.getvalue())
            with self.assertRaisesRegex(RuntimeError, "required sampled isolated_repeat failed"):
                runner.evaluate_gate(failed_out, "sampled", 0, [])
            saved = json.loads(Path(f"{failed_out}.evidence.json").read_text())
            self.assertFalse(saved["required_evidence_passed"])

            # Unrecorded sampled output without .requests/ is rejected by evaluate_gate.
            unrecorded = root / "unrecorded-sampled.jsonl"
            unrecorded.write_text(output.read_text())
            with self.assertRaisesRegex(RuntimeError, "request recording is missing"):
                runner.evaluate_gate(unrecorded, "sampled", 0, [])

    def test_cross_arm_and_same_config_signatures_distinguish_row_cost_from_repeat_boots(self):
        base = {
            "greedy": {"reference_sha256": "a" * 64, "long_sha256": "b" * 64,
                       "reference_text_sha256": "c" * 64, "long_text_sha256": "d" * 64},
            "greedy-fast": {"reference_text_sha256": "c" * 64, "long_text_sha256": "d" * 64},
            "sampled": {"reference_sha256": "1" * 64, "isolated_repeat_sha256": "1" * 64,
                        "observed_c8_sha256": "2" * 64, "observed_c8_differing": [0, 2]},
            "controls": {"penalties/alone": "3" * 64, "penalties/c8": "3" * 64,
                         "penalties-sampled/alone": "4" * 64, "penalties-sampled/c8": "5" * 64},
            "tool-schema": {"schema/t0.0/c1": "6" * 64, "schema/t0.0/c8": "6" * 64,
                            "schema/t1.0/c1": "7" * 64, "schema/t1.0/c8": "8" * 64,
                            "invalid_request_status": 400},
        }
        repeat_boot = copy.deepcopy(base) if "copy" in globals() else json.loads(json.dumps(base))
        repeat_boot["sampled"]["observed_c8_sha256"] = "9" * 64
        repeat_boot["controls"]["penalties-sampled/c8"] = "0" * 64
        repeat_boot["tool-schema"]["schema/t1.0/c8"] = "f" * 64
        self.assertEqual(runner.same_config_signature(base), runner.same_config_signature(repeat_boot))

        other_row_cost = json.loads(json.dumps(repeat_boot))
        other_row_cost["sampled"]["reference_sha256"] = "e" * 64
        other_row_cost["controls"]["penalties-sampled/alone"] = "d" * 64
        self.assertEqual(runner.cross_arm_signature(base), runner.cross_arm_signature(other_row_cost))
        self.assertNotEqual(runner.same_config_signature(base), runner.same_config_signature(other_row_cost))

    def test_greedy_rejects_silent_logprob_fallback_and_missing_native_comparison(self):
        scenarios = ["c8", "staggered", "mixed", "cached", "again", "mixed15"]
        result = {"summary": "temperature-0 determinism", "identical": True, "logprobs": True, "tokens": 192,
                  "scenarios": {name: {} for name in scenarios},
                  **{key: "a" * 64 for key in ("reference_sha256", "long_sha256",
                                               "reference_text_sha256", "long_text_sha256")}}
        def log(summary, matched=True):
            return "\n".join([*(json.dumps({"scenario": name, "requests": [{"logprobs_equal": matched}]})
                                for name in scenarios), json.dumps(summary)])
        self.assertEqual(len(runner.gate_summary(log(result), "greedy")), 4)
        self.assertEqual(len(runner.gate_summary(log({**result, "logprobs": False}, None), "greedy-fast")), 2)
        for summary, matched in (({**result, "logprobs": False}, True), (result, None),
                                 ({**result, "scenarios": {}}, True)):
            with self.assertRaises(ValueError):
                runner.gate_summary(log(summary, matched), "greedy")

    def test_foreign_holder_window_or_restore_profile_is_not_ours(self):
        ours = {"holder": runner.HOLDER, "window": "window-1", "production_config": runner.RESTORE}
        self.assertTrue(runner.owns(ours, "window-1"))
        for change in ({"holder": "spark-lab: another task"}, {"window": "window-2"},
                       {"production_config": "config/cluster.json"}):
            self.assertFalse(runner.owns({**ours, **change}, "window-1"))
        self.assertFalse(runner.owns(None, "window-1"))

    def test_actual_start_shape_change_spelling_refused(self):
        for text in ("live: running\nshape:   CHANGES nodes", "live: running\n--apply would refuse: no"):
            with self.assertRaises(ValueError):
                runner.validate_plan(text, "start")

    def test_stop_format_and_start_shape_are_distinguished(self):
        runner.validate_plan("live: running\ndgx1: stop exact container", "stop")
        runner.validate_plan("live: running\nshape: unchanged", "start")
        with self.assertRaises(ValueError):
            runner.validate_plan("live: running", "start")


if __name__ == "__main__":
    unittest.main()
