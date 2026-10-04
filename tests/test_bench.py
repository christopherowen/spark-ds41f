"""Host-independent tests for the benchmark statistics and client parsing."""

from __future__ import annotations

import argparse
import http.server
import random
import tempfile
import importlib.machinery
import importlib.util
import json
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark3", str(ROOT / "bin" / "spark3"))
spec = importlib.util.spec_from_loader("spark3", loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)


class StatisticsTest(unittest.TestCase):
    def test_describe_reports_student_interval(self) -> None:
        summary = spark3.describe([10.0, 12.0, 14.0])
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["mean"], 12.0)
        self.assertEqual(summary["median"], 12.0)
        self.assertEqual(summary["sd"], 2.0)
        # t(2) = 4.303; 4.303 * 2 / sqrt(3)
        self.assertAlmostEqual(summary["ci95"], 4.969, places=3)
        self.assertAlmostEqual(summary["ci95_pct"], 41.41, places=2)

    def test_describe_single_value_has_no_interval(self) -> None:
        summary = spark3.describe([5.0])
        self.assertIsNone(summary["ci95"])
        self.assertEqual(spark3.describe([]), {"n": 0})

    def test_difference_is_signed_percent_with_interval(self) -> None:
        reference = {"n": 30, "mean": 100.0, "sd": 5.0}
        slower = {"n": 30, "mean": 90.0, "sd": 5.0}
        change, low, high = spark3.difference(slower, reference)
        self.assertAlmostEqual(change, -10.0)
        self.assertLess(high, 0)
        self.assertGreater(low, -13)
        same = spark3.difference({"n": 30, "mean": 100.5, "sd": 5.0}, reference)
        self.assertLess(same[1], 0)
        self.assertGreater(same[2], 0)

    def test_t_quantile_is_conservative_between_table_rows(self) -> None:
        self.assertEqual(spark3.t_95(11), spark3.t_95(10))
        self.assertEqual(spark3.t_95(500), 1.980)


class ParsingTest(unittest.TestCase):
    def test_parse_metrics_sums_label_sets_and_ignores_similar_names(self) -> None:
        text = "\n".join(
            [
                "# HELP vllm:request_success_total Count",
                'vllm:request_success_total{finished_reason="stop"} 7.0',
                'vllm:request_success_total{finished_reason="length"} 3.0',
                'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 655.0',
                'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 247.0',
                'vllm:num_requests_running{engine="0"} 2.0',
            ]
        )
        values = spark3.parse_metrics(text)
        self.assertEqual(values["vllm:request_success_total"], 10.0)
        self.assertEqual(values["vllm:spec_decode_num_accepted_tokens_total"], 655.0)
        self.assertEqual(values["vllm:num_requests_running"], 2.0)
        self.assertEqual(values["vllm:num_preemptions_total"], 0.0)

    def test_parse_thermal_sample(self) -> None:
        sample = spark3.parse_thermal("62,34,2405,14.25,0,0,27131561741,0,64900")
        self.assertEqual(sample["gpu_c"], 62.0)
        self.assertEqual(sample["headroom_c"], 34.0)
        self.assertEqual(sample["thermal_slowdown_us"], 0.0)
        self.assertEqual(sample["power_cap_us"], 27131561741.0)
        self.assertAlmostEqual(sample["system_c"], 64.9)
        missing = spark3.parse_thermal("[N/A],[N/A],2405,14.25")
        self.assertIsNone(missing["gpu_c"])
        self.assertIsNone(missing["system_c"])

    def test_novel_text_is_reproducible_and_does_not_repeat(self) -> None:
        text = spark3.novel_text(4000, 7)
        self.assertEqual(text, spark3.novel_text(4000, 7))
        words = text.split()[2:]
        trigrams = list(zip(words, words[1:], words[2:]))
        self.assertGreater(len(set(trigrams)), 0.99 * len(trigrams))

    def test_decode_cases_keep_the_reference_prompts(self) -> None:
        for case, prompt in spark3.DECODE_PROMPTS.items():
            self.assertEqual(spark3.DECODE_CASES[case], spark3.DecodeCase(prompt))
            payloads = spark3.decode_payloads("m", case, 4)
            self.assertEqual([payload for _, payload in payloads], [payloads[0][1]] * 4)
            self.assertEqual(payloads[0][1]["temperature"], 0)
            self.assertEqual(payloads[0][1]["seed"], 42)
            self.assertNotIn("min_tokens", payloads[0][1])
        self.assertEqual(spark3.DECODE_CASES["code-nothink"].template_kwargs, {"thinking": False})

    def test_portable_cases_fix_the_length_and_vary_by_stream(self) -> None:
        self.assertEqual(
            spark3.DECODE_CASE_GROUPS["portable"],
            ["count", "explain", "tasks", "rows", "math", "chat", "essay", "story", "chat-sampled"],
        )
        for case in spark3.DECODE_CASE_GROUPS["portable"]:
            payloads = [payload for _, payload in spark3.decode_payloads("m", case, 8)]
            self.assertEqual(len({payload["messages"][0]["content"] for payload in payloads}), 8, case)
            for payload in payloads:
                self.assertEqual(payload["min_tokens"], 256)
                self.assertEqual(payload["max_tokens"], 256)
                self.assertTrue(payload["ignore_eos"])
                self.assertEqual(payload["chat_template_kwargs"], {"thinking": False})
        # One stream gets the exact prompt; code streams get distinct tasks,
        # tagged once the tasks repeat.
        single = spark3.decode_payloads("m", "count", 1)[0][1]
        self.assertEqual(single["messages"][0]["content"], spark3.PORTABLE_CASES["count"])
        tasks = [payload["messages"][0]["content"] for _, payload in spark3.decode_payloads("m", "tasks", 10)]
        self.assertEqual(tasks[0], spark3.PORTABLE_CODE_TASKS[0])
        self.assertEqual(tasks[9], f"[stream 10]\n{spark3.PORTABLE_CODE_TASKS[1]}")
        sampled = [payload for _, payload in spark3.decode_payloads("m", "chat-sampled", 2)]
        self.assertEqual([(p["temperature"], p["top_p"], p["seed"]) for p in sampled], [(0.7, 0.95, 42), (0.7, 0.95, 43)])

    def test_decode_window_excludes_prefill_and_start_stagger(self) -> None:
        requests = [
            {"ok": True, "first_s": 0.2, "last_s": 2.2, "completion_tokens": 101},
            {"ok": True, "first_s": 0.4, "last_s": 3.2, "completion_tokens": 201},
            {"ok": False, "first_s": 0.1, "last_s": 9.0, "completion_tokens": 50},
        ]
        self.assertEqual(spark3.decode_window_tps(requests), 100.0)
        self.assertIsNone(spark3.decode_window_tps([{"ok": True, "first_s": None, "last_s": None, "completion_tokens": 9}]))

    def test_source_text_is_real_reproducible_text(self) -> None:
        text = spark3.source_text(20000, 3)
        self.assertEqual(text, spark3.source_text(20000, 3))
        self.assertNotEqual(text, spark3.source_text(20000, 4))
        self.assertTrue(text.startswith("Document 3:\n"))
        self.assertEqual(len(text), len("Document 3:\n") + int(20000 * spark3.SOURCE_CHARS_PER_TOKEN))
        self.assertIn("def ", text)

    def test_assess_lru(self) -> None:
        good = "```python\nclass LRU:\n    def get(self, key):\n        pass\n    def put(self, key, value):\n        pass\n```"
        self.assertEqual(spark3.assess_lru(good), "pass")
        self.assertEqual(spark3.assess_lru("no code"), "missing Python code block")
        self.assertIn("syntax error", spark3.assess_lru("```python\nclass (:\n```"))
        wrong = good.replace("key, value", "k, v")
        self.assertIn("signature", spark3.assess_lru(wrong))


HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None
needs_jsonschema = unittest.skipUnless(HAS_JSONSCHEMA, "python-jsonschema is not installed")


class PrecisionSetTest(unittest.TestCase):
    def test_precision_cases_use_the_1k_budget_without_forcing_length(self) -> None:
        self.assertEqual(spark3.DECODE_CASE_GROUPS["precision"], ["prose-1k", "code-1k", "json-1k"])
        for case in spark3.DECODE_CASE_GROUPS["precision"]:
            spec = spark3.DECODE_CASES[case]
            self.assertEqual(
                (spec.max_tokens, spec.min_samples, spec.max_samples, spec.converge_on, spec.fixed_length),
                (1024, 5, 12, "step_ms", False), case,
            )
            payloads = [payload for _, payload in spark3.decode_payloads("m", case, 8)]
            self.assertEqual(len({payload["messages"][0]["content"] for payload in payloads}), 8, case)
            for payload in payloads:
                self.assertEqual(payload["max_tokens"], 1024)
                self.assertNotIn("min_tokens", payload)
                self.assertNotIn("ignore_eos", payload)
                self.assertEqual(payload["temperature"], 0)
                self.assertEqual(payload["chat_template_kwargs"], {"thinking": False})
        self.assertEqual(len(spark3.DECODE_CASES["json-1k"].json_schema), 8)
        self.assertEqual(len(set(spark3.PRECISION_JSON_PROMPTS)), 8)
        # Every prompt names every key of its own schema, so prompt and check cannot drift.
        for prompt, schema in zip(spark3.PRECISION_JSON_PROMPTS, spark3.PRECISION_JSON_SCHEMAS):
            for key in schema["properties"]:
                self.assertIn(f"{key} (", prompt)
        self.assertEqual(spark3.json_schema_for(spark3.DECODE_CASES["json-1k"], 9), spark3.PRECISION_JSON_SCHEMAS[1])
        # The portable protocol and the reference prompts are untouched.
        rows = spark3.decode_payloads("m", "rows", 1)[0][1]
        self.assertEqual((rows["max_tokens"], rows["min_tokens"]), (256, 256))
        self.assertEqual(spark3.DECODE_CASES["prose"], spark3.DecodeCase(spark3.DECODE_PROMPTS["prose"]))

    def test_structured_output_case_is_opt_in(self) -> None:
        self.assertNotIn("json-schema-1k", spark3.DECODE_CASE_GROUPS["precision"])
        payload = spark3.decode_payloads("m", "json-schema-1k", 1)[0][1]
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["response_format"]["json_schema"]["schema"]["type"], "array")
        self.assertEqual(spark3.DECODE_CASES["json-schema-1k"].json_schema, payload["response_format"]["json_schema"]["schema"]["items"])
        self.assertIn("tags (an array of two strings)", payload["messages"][0]["content"])
        self.assertNotIn("response_format", spark3.decode_payloads("m", "json-1k", 1)[0][1])

    def test_schema_key_text_describes_types_counts_and_ranges(self) -> None:
        schema = spark3.record_schema(
            a={"type": "string"}, b={"type": "integer"}, c=spark3.percent(), d=spark3.strings(3),
            e={"type": "array", "items": spark3.record_schema(x={"type": "boolean"})},
        )
        self.assertEqual(
            spark3.schema_key_text(schema),
            "a (a string), b (an integer), c (a number from 0 to 100), d (an array of three strings) "
            "and e (an array of objects with x (a boolean))",
        )

    @needs_jsonschema
    def test_json_schema_violations_cover_the_record_shapes(self) -> None:
        books = spark3.PRECISION_JSON_SCHEMAS[0]
        good = {"title": "T", "author": "A", "year": 1999, "genres": ["x", "y"], "isbn": "1", "pages": 10,
                "available": True, "rating": 4.5}
        self.assertEqual(spark3.json_schema_violations(good, books), [])
        bad = dict(good, year="1999", genres=["x"], rating=7, available=1, extra=None)
        del bad["pages"]
        found = spark3.json_schema_violations(bad, books)
        self.assertEqual(len(found), 6)
        # Root-level findings come first; their mutual order varies by jsonschema version.
        self.assertEqual(set(found[:2]), {"$: Additional properties are not allowed ('extra' was unexpected)",
                                          "$: 'pages' is a required property"})
        self.assertIn("$.available: 1 is not of type 'boolean'", found)
        self.assertIn("$.genres: ['x'] is too short", found)
        self.assertIn("$.rating: 7 is greater than the maximum of 5", found)
        self.assertIn("$.year: '1999' is not of type 'integer'", found)
        # Booleans are not integers; integers are numbers.
        self.assertEqual(spark3.json_schema_violations(True, {"type": "integer"}), ["$: True is not of type 'integer'"])
        self.assertEqual(spark3.json_schema_violations(3, {"type": "number"}), [])
        invoice = {"invoiceId": "i", "customer": "c", "issuedOn": "d", "dueOn": "d", "paid": False,
                   "lineItems": [{"description": "x", "amountCents": 5}, {"description": "y", "amountCents": "5"}]}
        self.assertEqual(spark3.json_schema_violations(invoice, spark3.PRECISION_JSON_SCHEMAS[6]),
                         ["$.lineItems[1].amountCents: '5' is not of type 'integer'"])
        # Every benchmark schema is itself a valid Draft 2020-12 schema.
        for schema in (*spark3.PRECISION_JSON_SCHEMAS, spark3.PRECISION_JSON_SCHEMA["schema"]):
            spark3.schema_validator(schema)

    @needs_jsonschema
    def test_json_progress_scores_objects_against_the_schema_and_saves_outputs(self) -> None:
        schema = spark3.record_schema(a={"type": "integer"})
        text = '[{"a": 1}, {"a": "two"}, {"a": 3'
        scored = spark3.json_array_progress(text, schema)
        self.assertEqual(
            {key: scored[key] for key in ("json_objects", "json_prefix_valid", "json_valid_objects", "json_violation")},
            {"json_objects": 2, "json_prefix_valid": True, "json_valid_objects": 1,
             "json_violation": "$[1].a: 'two' is not of type 'integer'"},
        )
        self.assertEqual(json.loads(scored["json_prefix"]), [{"a": 1}, {"a": "two"}])
        broken = spark3.json_array_progress('[{"a": }', schema)
        self.assertEqual((broken["json_objects"], broken["json_prefix_valid"], broken["json_valid_objects"]), (0, False, 0))
        self.assertNotIn("json_prefix", broken)
        with tempfile.TemporaryDirectory() as tmp:
            request = {"content": text, **scored}
            spark3.save_output(Path(tmp), "json-1k-c2", "sample03", 1, request)
            folder = Path(tmp) / "outputs" / "json-1k-c2"
            self.assertEqual((folder / "sample03-stream2.txt").read_text(), text)
            self.assertEqual(json.loads((folder / "sample03-stream2.json").read_text()), [{"a": 1}, {"a": "two"}])
            spark3.save_output(Path(tmp), "json-1k-c2", "warmup", 0, {"content": "no json", **spark3.json_array_progress("no json", schema)})
            self.assertFalse((folder / "warmup-stream1.json").exists())
            spark3.save_output(None, "json-1k-c2", "sample01", 0, request)

    def test_json_array_progress_scores_truncated_arrays(self) -> None:
        text = '[{"a": 1, "t": ["x", "y]"]}, {"a": "}", "n": {"k": 2}}, {"a": 3'
        scored = spark3.json_array_progress(text)
        self.assertEqual((scored["json_objects"], scored["json_prefix_valid"]), (2, True))
        self.assertEqual(scored["json_prefix"], '[{"a": 1, "t": ["x", "y]"]}, {"a": "}", "n": {"k": 2}}]')
        self.assertEqual(spark3.json_array_progress("Sure! Here is"), {"json_objects": 0, "json_prefix_valid": False})
        # The decoder stops at the first element that is not JSON; what came before still counts.
        partial = spark3.json_array_progress('[{"a": 1}, {"b": }]')
        self.assertEqual((partial["json_objects"], partial["json_prefix_valid"], partial["json_prefix"]), (1, True, '[{"a": 1}]'))
        nothing = spark3.json_array_progress('[{"a": }')
        self.assertEqual((nothing["json_objects"], nothing["json_prefix_valid"]), (0, False))
        self.assertNotIn("json_prefix", nothing)
        mixed = spark3.json_array_progress('["s", 1, {"a": 1}]')
        self.assertEqual((mixed["json_objects"], mixed["json_prefix_valid"]), (1, True))
        escaped = spark3.json_array_progress('[{"q": "\\\\"}, {"r": "\\"}"}')
        self.assertEqual((escaped["json_objects"], escaped["json_prefix_valid"]), (2, True))

    def test_sample_metrics_reports_primaries_and_both_throughputs(self) -> None:
        requests = [
            {"ok": True, "first_s": 0.5, "last_s": 10.5, "completion_tokens": 1024, "decode_tps": 102.3,
             "ttft_s": 0.5, "output_sha256": "a", "content": '[{"a": 1}'},
            {"ok": True, "first_s": 0.6, "last_s": 10.5, "completion_tokens": 1024, "decode_tps": 103.3,
             "ttft_s": 0.6, "output_sha256": "b"},
        ]
        metrics = {"spec_decode_num_drafts": 500, "spec_decode_num_draft_tokens": 2000,
                   "spec_decode_num_accepted_tokens": 1548}
        entry = spark3.sample_metrics({"wall_s": 11.0, "requests": requests, "metrics": metrics}, 2)
        self.assertEqual(entry["accepted_per_draft"], 3.096)
        self.assertEqual(entry["accepted_per_verified"], 0.774)
        self.assertEqual(entry["tokens_per_step"], 4.096)
        self.assertEqual(entry["verified_per_draft"], 4.0)
        # Two streams, 2046 tokens after their first tokens, over a 10 s window.
        self.assertEqual(entry["decode_window_tps"], 204.6)
        self.assertAlmostEqual(entry["step_ms"], 1000 * 4.096 * 2 / 204.6, places=2)
        self.assertAlmostEqual(entry["derived_tps"], 204.6, places=1)
        self.assertEqual(entry["tps"], round(2048 / 11.0, 3))
        self.assertNotIn("content", entry["requests"][0])
        single = spark3.sample_metrics({"wall_s": 11.0, "requests": requests[:1], "metrics": metrics}, 1)
        self.assertAlmostEqual(single["step_ms"], 1000 * 4.096 / 102.3, places=2)
        self.assertEqual(spark3.convergence_values([entry, single], "step_ms"), [entry["step_ms"], single["step_ms"]])
        self.assertEqual(spark3.convergence_values([entry, single], "tps"), [entry["tps"], single["tps"]])
        # A failed stream leaves the step time undefined rather than wrong.
        broken = [dict(requests[0]), dict(requests[1], ok=False)]
        self.assertIsNone(spark3.sample_metrics({"wall_s": 11.0, "requests": broken, "metrics": metrics}, 2)["step_ms"])
        # Older reports kept only single-stream requests; their step times still derive.
        legacy = [{"accepted_per_draft": 3.0, "requests": [{"ok": True, "decode_tps": 100.0}]}]
        self.assertEqual(spark3.sample_step_times(legacy), [40.0])

    def test_decode_plan_applies_case_floors_and_cli_override(self) -> None:
        options = argparse.Namespace(min_samples=3, max_samples=4, converge_on=None)
        spec, concurrency, metric, minimum, maximum = spark3.decode_plan(options, "json-1k-c8")
        self.assertEqual((spec, concurrency, metric, minimum, maximum), (spark3.DECODE_CASES["json-1k"], 8, "step_ms", 5, 12))
        self.assertEqual(spark3.decode_plan(options, "prose-c1")[2:], ("tps", 3, 4))
        options.converge_on = "accepted_per_verified"
        self.assertEqual(spark3.decode_plan(options, "json-1k-c1")[2], "accepted_per_verified")
        options.min_samples, options.max_samples = 8, 20
        self.assertEqual(spark3.decode_plan(options, "json-1k-c1")[3:], (8, 20))

    @needs_jsonschema
    def test_suite_decode_applies_case_floors_and_scores_json(self) -> None:
        class StubBench:
            def __init__(self) -> None:
                self.calls: list[tuple[str, bool]] = []

            def run(self, label, payloads, concurrency=None, keep_content=False, retries=3):
                self.calls.append((label, keep_content))
                streams = len(payloads)
                requests = [
                    {"ok": True, "ttft_s": 0.3, "first_s": 0.3, "last_s": 10.3, "prompt_tokens": 50,
                     "completion_tokens": 1024, "decode_tps": 100.0, "output_sha256": f"{len(self.calls)}-{index}",
                     "reasoning_chars": 0, "content_chars": 20, "error": None,
                     "content": '[{"a": 1}, {"b": 2}, {"c": '}
                    for index in range(streams)
                ]
                metrics = {"spec_decode_num_drafts": 250 * streams, "spec_decode_num_draft_tokens": 1000 * streams,
                           "spec_decode_num_accepted_tokens": 774 * streams}
                return {"wall_s": 10.6, "requests": requests, "metrics": metrics, "peak": {}}

        with tempfile.TemporaryDirectory() as tmp:
            options = argparse.Namespace(decode_cases=["json-1k"], concurrency=[1, 2], model="m",
                                         min_samples=3, max_samples=4, precision=2.0, converge_on=None,
                                         output_dir=Path(tmp))
            bench = StubBench()
            report = spark3.suite_decode(bench, options, random.Random(0))
            saved = sorted(path.name for path in (Path(tmp) / "outputs" / "json-1k-c2").iterdir())
        self.assertIn("warmup-stream2.txt", saved)
        self.assertIn("sample05-stream1.json", saved)
        self.assertEqual(len(saved), 2 * 2 * 6)
        point = report["points"]["json-1k-c2"]
        # The stub's objects lack the books keys, so none conform.
        self.assertEqual(point["json_schema_valid"], 0.0)
        self.assertRegex(point["json_violation_example"], r"^\$\[0\]: .*(required property|Additional properties)")
        # Constant samples converge at once, so the case floor of five decides.
        self.assertEqual(point["step_ms"]["n"], 5)
        self.assertEqual(point["converge_on"], "step_ms")
        self.assertEqual(point["accepted_per_verified"]["mean"], 0.774)
        self.assertEqual(point["tokens_per_step"]["mean"], 4.096)
        self.assertEqual(point["derived_tps"]["n"], 5)
        self.assertEqual(point["json_objects"]["mean"], 2.0)
        self.assertEqual(point["json_prefix_valid"], 1.0)
        self.assertEqual(point["full_length_outputs"], 10)
        self.assertEqual(point["max_tokens"], 1024)
        self.assertTrue(all(keep for _, keep in bench.calls))
        self.assertNotIn("content", report["samples"]["json-1k-c2"][0]["requests"][0])
        self.assertEqual(len(bench.calls), 2 + 5 * 2)

    def test_primary_rows_show_both_throughputs_and_warn_on_short_outputs(self) -> None:
        stat = lambda mean: {"n": 5, "mean": mean, "ci95_pct": 1.5}
        points = {
            "json-1k-c2": {
                "step_ms": stat(40.0), "accepted_per_verified": stat(0.77), "tokens_per_step": stat(4.1),
                "tps": stat(190.0), "decode_window_tps": stat(204.6), "derived_tps": stat(205.0),
                "max_tokens": 1024, "full_length_outputs": 9, "failed_requests": 0,
                "json_objects": stat(22.4), "json_prefix_valid": 0.9, "json_schema_valid": 0.98,
                "json_violation_example": "$[3].year: expected integer, got str",
            },
            "prose-c1": {"tps": stat(57.0)},
        }
        lines = spark3.decode_primary_rows(points)
        self.assertTrue(lines[0].startswith("decode primaries"))
        self.assertIn("40.00 ±1.5%", lines[1])
        self.assertIn("190.0 ±1.5% /", lines[1])
        self.assertIn("205.0 ±1.5%", lines[1])
        self.assertIn("WARN json-1k-c2: 1 of 10 outputs stopped before 1024 tokens", lines[2])
        self.assertIn("JSON: 22.4 ±1.5% completed objects per output, prefix parses 0.9, objects conform 0.98", lines[3])
        self.assertIn("first violation: $[3].year: expected integer, got str", lines[4])
        self.assertIn("WARN json-1k-c2: some outputs are not JSON arrays", lines[5])
        self.assertEqual(len(lines), 6)


class ComplianceSuiteTest(unittest.TestCase):
    def test_compliance_is_explicit_and_full_excludes_it(self) -> None:
        self.assertIn("compliance", spark3.BENCH_SUITES)
        self.assertNotIn("compliance", spark3.FULL_SUITES)
        self.assertEqual(spark3.BENCH_SUITE_FUNCTIONS["compliance"], spark3.suite_compliance)

    def test_compliance_payloads_ask_for_a_complete_array_with_room(self) -> None:
        plain = [payload for _, payload in spark3.compliance_payloads("m", False)]
        strict = [payload for _, payload in spark3.compliance_payloads("m", True)]
        self.assertEqual(len(plain), len(spark3.PRECISION_JSON_RECORDS))
        for payload in plain:
            self.assertEqual(payload["max_tokens"], 4096)
            self.assertNotIn("min_tokens", payload)
            self.assertNotIn("response_format", payload)
            self.assertIn("array of 12 invented", payload["messages"][0]["content"])
            self.assertEqual(payload["chat_template_kwargs"], {"thinking": False})
        schema = strict[0]["response_format"]["json_schema"]["schema"]
        self.assertEqual((schema["type"], schema["minItems"], schema["maxItems"]), ("array", 12, 12))
        self.assertEqual(schema["items"], spark3.PRECISION_JSON_SCHEMAS[0])

    def test_used_whole_budget_prefers_the_finish_reason(self) -> None:
        self.assertTrue(spark3.used_whole_budget({"finish_reason": "length", "completion_tokens": 10}, 1024))
        self.assertFalse(spark3.used_whole_budget({"finish_reason": "stop", "completion_tokens": 1024}, 1024))
        self.assertTrue(spark3.used_whole_budget({"completion_tokens": 1024}, 1024))
        self.assertFalse(spark3.used_whole_budget({"completion_tokens": 900}, 1024))

    @needs_jsonschema
    def test_assess_json_records_requires_a_clean_complete_conforming_array(self) -> None:
        schema = spark3.record_schema(a={"type": "integer"})
        good = json.dumps([{"a": index} for index in range(3)])
        self.assertEqual(spark3.assess_json_records(good, schema, 3)["ok"], True)
        short = spark3.assess_json_records(json.dumps([{"a": 1}]), schema, 3)
        self.assertEqual((short["ok"], short["objects"], short["violation"]), (False, 1, "1 objects, 3 requested"))
        fenced = spark3.assess_json_records(f"```json\n{good}\n```", schema, 3)
        self.assertEqual((fenced["ok"], fenced["fenced"], fenced["conforming"]), (False, True, 3))
        bad = spark3.assess_json_records(json.dumps([{"a": 1}, {"a": "x"}, {"a": 3}]), schema, 3)
        self.assertEqual((bad["ok"], bad["conforming"], bad["violation"]), (False, 2, "$[1].a: 'x' is not of type 'integer'"))
        self.assertFalse(spark3.assess_json_records("Sure thing", schema, 3)["parsed"])
        self.assertEqual(spark3.assess_json_records('{"a": 1}', schema, 1)["violation"], "top level is not an array")

    @needs_jsonschema
    def test_suite_compliance_reports_both_arms_and_saves_outputs(self) -> None:
        class StubBench:
            def __init__(self) -> None:
                self.labels: list[str] = []

            def run(self, label, payloads, concurrency=None, keep_content=False, retries=3):
                self.labels.append(label)
                requests = []
                for index, (subject, schema) in enumerate(spark3.PRECISION_JSON_RECORDS):
                    content = json.dumps([
                        {key: ({"string": "s", "integer": 1, "number": 2.5, "boolean": True}.get(prop.get("type"))
                               if prop.get("type") != "array" else
                               (["x", "y", "z"][: prop["minItems"]] if prop["items"].get("type") == "string"
                                else [{"description": "d", "amountCents": 1}] * prop["minItems"]))
                         for key, prop in schema["properties"].items()}
                        for _ in range(spark3.COMPLIANCE_OBJECTS)
                    ])
                    finish = "length" if (index == 0 and "constrained" not in label) else "stop"
                    requests.append({"ok": True, "content": content, "finish_reason": finish, "completion_tokens": 700,
                                     "elapsed_s": 9.5 + index, "error": None})
                metrics = {"spec_decode_num_drafts": 100, "spec_decode_num_draft_tokens": 400,
                           "spec_decode_num_accepted_tokens": 300}
                return {"wall_s": 30.0, "requests": requests, "metrics": metrics, "peak": {}}

        with tempfile.TemporaryDirectory() as tmp:
            options = argparse.Namespace(model="m", output_dir=Path(tmp))
            bench = StubBench()
            report = spark3.suite_compliance(bench, options, random.Random(0))
            saved = sorted(p.name for p in (Path(tmp) / "outputs" / "compliance-constrained").iterdir())
        self.assertEqual(bench.labels, ["compliance prompt_only", "compliance constrained"])
        self.assertEqual(report["arms"]["prompt_only"]["passed"], len(spark3.PRECISION_JSON_RECORDS) - 1)
        self.assertEqual(report["arms"]["prompt_only"]["requests"][0]["violation"], "hit the token budget")
        self.assertEqual(report["arms"]["constrained"]["passed"], len(spark3.PRECISION_JSON_RECORDS))
        self.assertEqual(report["arms"]["constrained"]["accepted_per_verified"], 0.75)
        self.assertFalse(report["ok"])
        self.assertIn("library-books.txt", saved)
        lines, _ = spark3.bench_rows({"suites": {"compliance": report}}, None, 3.0)
        self.assertTrue(lines[0].startswith("compliance prompt_only: 7/8 complete and conforming"))
        self.assertIn("  library books: hit the token budget", lines)


class AdmissionPayloadTest(unittest.TestCase):
    def capture(self, forced):
        class Captured(Exception):
            pass

        class Bench:
            def run(self, label, payloads):
                self.label, self.payloads = label, payloads
                raise Captured

        bench = Bench()
        options = spark3.argparse.Namespace(
            model="m", admission_tokens=4000, max_model_len=4096,
            admission_output_tokens=1024, admission_force_length=forced,
        )
        with self.assertRaises(Captured):
            spark3.suite_admission(bench, options, spark3.random.Random(42))
        self.assertEqual(bench.label, "admission 4 x 2048")
        self.assertEqual(len(bench.payloads), 4)
        self.assertEqual(len({p["cache_salt"] for _, p in bench.payloads}), 4)
        return [p for _, p in bench.payloads]

    def test_sustained_admission_reserves_output_and_keeps_requests_alive(self):
        for payload in self.capture(True):
            self.assertEqual(payload["max_tokens"], 1024)
            self.assertEqual(payload["min_tokens"], 1024)
            self.assertTrue(payload["ignore_eos"])

    def test_ordinary_admission_allows_natural_stop(self):
        for payload in self.capture(False):
            self.assertEqual(payload["max_tokens"], 1024)
            self.assertNotIn("min_tokens", payload)
            self.assertNotIn("ignore_eos", payload)
        options = spark3.parser().parse_args(["bench"])
        self.assertEqual(options.admission_output_tokens, 256)
        self.assertFalse(options.admission_force_length)


class StreamTest(unittest.TestCase):
    def serve(self, events: list[dict]) -> str:
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(body.encode())

            def log_message(self, *args) -> None:
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def test_chat_stream_keeps_content_and_usage(self) -> None:
        url = self.serve(
            [
                {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
                {"choices": [{"delta": {"reasoning_content": "think "}}]},
                {"choices": [{"delta": {"content": "hello"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
            ]
        )
        result = spark3.stream_request(url, {"model": "m"}, threading.Event(), keep_content=True)
        self.assertTrue(result["ok"])
        self.assertEqual(result["finish_reason"], "stop")
        self.assertIn("finish_reason", spark3.compact([result])[0])
        self.assertEqual(result["content"], "hello")
        self.assertEqual(result["completion_tokens"], 3)
        self.assertEqual(result["prompt_tokens"], 5)
        self.assertEqual(result["reasoning_chars"], 6)
        self.assertEqual(result["content_chars"], 5)
        self.assertEqual(spark3.reasoning_share([result]), 0.545)
        # The last token is the last emitted text, not the usage chunk.
        self.assertLessEqual(result["first_at"], result["last_at"])

    def test_completion_stream_counts_an_empty_first_token(self) -> None:
        url = self.serve(
            [
                {"choices": [{"text": ""}]},
                {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 1}},
            ]
        )
        result = spark3.stream_request(url, {"model": "m"}, threading.Event())
        self.assertTrue(result["ok"])
        self.assertIsNotNone(result["ttft_s"])
        self.assertIsNone(result["finish_reason"])

    def test_failed_request_is_recorded(self) -> None:
        result = spark3.stream_request("http://127.0.0.1:9", {"model": "m"}, threading.Event())
        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["error"])


if __name__ == "__main__":
    unittest.main()
