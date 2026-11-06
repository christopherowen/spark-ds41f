import json
import unittest

import run_hardware as runner
import tool_schema_gate


class GateTests(unittest.TestCase):
    def test_resume_accepts_only_restored_ordered_handoff_at_identical_inputs(self):
        arms = [{"label": "row010-a", "config": "row010.json"},
                {"label": "control035-a", "config": "control.json"}]
        identity = {"commit": "a" * 40, "stage": "screen", "gate_sources": {"gate": "hash"},
                    "images": {"dgx1": "digest"}, "restore_profile": runner.RESTORE}
        previous = {**identity, "completed": False, "restored": True,
                    "error": "RuntimeError: handoff requested after completed arm",
                    "completed_arms": [{"label": "row010-a", "config": f"{runner.EXPERIMENT}/row010.json"}]}
        self.assertEqual(runner.resume_arms(previous, identity, arms), previous["completed_arms"])
        for change in ({"completed": True}, {"restored": False}, {"error": "gate failed"},
                       {"commit": "b" * 40}, {"gate_sources": {}}, {"images": {}},
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

    def test_control_batch_mismatch_or_failure_fails(self):
        for result in ({"failed": ["json"], "digests": {"json/alone": "a"}},
                       {"failed": [], "digests": {"json/alone": "a", "json/c8": "b"}}):
            with self.assertRaises(ValueError):
                runner.gate_summary(json.dumps({"summary": "controls", **result}), "controls")

    def test_valid_signature_is_preserved_for_cross_arm_comparison(self):
        text = json.dumps({"summary": "sampled", "identical": True,
                           "reference_sha256": "a" * 64})
        self.assertEqual(runner.gate_summary(text, "sampled"), {"reference_sha256": "a" * 64})

    def test_greedy_rejects_silent_logprob_fallback_and_missing_native_comparison(self):
        scenarios = ["c8", "staggered", "mixed", "cached", "again", "mixed15"]
        result = {"summary": "greedy", "identical": True, "logprobs": True,
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
