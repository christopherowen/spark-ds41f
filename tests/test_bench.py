"""Host-independent tests for the benchmark statistics and client parsing."""

from __future__ import annotations

import argparse
import contextlib
import http.server
import io
import random
import tempfile
import importlib.machinery
import importlib.util
import json
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
spec = importlib.util.spec_from_loader("spark", loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)


class StatisticsTest(unittest.TestCase):
    def test_describe_reports_student_interval(self) -> None:
        summary = spark.describe([10.0, 12.0, 14.0])
        self.assertEqual(summary["n"], 3)
        self.assertEqual(summary["mean"], 12.0)
        self.assertEqual(summary["median"], 12.0)
        self.assertEqual(summary["sd"], 2.0)
        # t(2) = 4.303; 4.303 * 2 / sqrt(3)
        self.assertAlmostEqual(summary["ci95"], 4.969, places=3)
        self.assertAlmostEqual(summary["ci95_pct"], 41.41, places=2)

    def test_describe_single_value_has_no_interval(self) -> None:
        summary = spark.describe([5.0])
        self.assertIsNone(summary["ci95"])
        self.assertEqual(spark.describe([]), {"n": 0})

    def test_difference_is_signed_percent_with_interval(self) -> None:
        reference = {"n": 30, "mean": 100.0, "sd": 5.0}
        slower = {"n": 30, "mean": 90.0, "sd": 5.0}
        change, low, high = spark.difference(slower, reference)
        self.assertAlmostEqual(change, -10.0)
        self.assertLess(high, 0)
        self.assertGreater(low, -13)
        same = spark.difference({"n": 30, "mean": 100.5, "sd": 5.0}, reference)
        self.assertLess(same[1], 0)
        self.assertGreater(same[2], 0)

    def test_t_quantile_is_conservative_between_table_rows(self) -> None:
        self.assertEqual(spark.t_95(11), spark.t_95(10))
        self.assertEqual(spark.t_95(500), 1.980)


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
        values = spark.parse_metrics(text)
        self.assertEqual(values["vllm:request_success_total"], 10.0)
        self.assertEqual(values["vllm:spec_decode_num_accepted_tokens_total"], 655.0)
        self.assertEqual(values["vllm:num_requests_running"], 2.0)
        self.assertEqual(values["vllm:num_preemptions_total"], 0.0)

    def test_parse_thermal_sample(self) -> None:
        sample = spark.parse_thermal("62,34,2405,14.25,0,0,27131561741,0,64900")
        self.assertEqual(sample["gpu_c"], 62.0)
        self.assertEqual(sample["headroom_c"], 34.0)
        self.assertEqual(sample["thermal_slowdown_us"], 0.0)
        self.assertEqual(sample["power_cap_us"], 27131561741.0)
        self.assertAlmostEqual(sample["system_c"], 64.9)
        missing = spark.parse_thermal("[N/A],[N/A],2405,14.25")
        self.assertIsNone(missing["gpu_c"])
        self.assertIsNone(missing["system_c"])

    def test_novel_text_is_reproducible_and_does_not_repeat(self) -> None:
        text = spark.novel_text(4000, 7)
        self.assertEqual(text, spark.novel_text(4000, 7))
        words = text.split()[2:]
        trigrams = list(zip(words, words[1:], words[2:]))
        self.assertGreater(len(set(trigrams)), 0.99 * len(trigrams))

    def test_decode_cases_keep_the_reference_prompts(self) -> None:
        for case, prompt in spark.DECODE_PROMPTS.items():
            self.assertEqual(spark.DECODE_CASES[case], spark.DecodeCase(prompt))
            payloads = spark.decode_payloads("m", case, 4)
            self.assertEqual([payload for _, payload in payloads], [payloads[0][1]] * 4)
            self.assertEqual(payloads[0][1]["temperature"], 0)
            self.assertEqual(payloads[0][1]["seed"], 42)
            self.assertNotIn("min_tokens", payloads[0][1])
        self.assertEqual(spark.DECODE_CASES["code-nothink"].template_kwargs, {"thinking": False})

    def test_portable_cases_fix_the_length_and_vary_by_stream(self) -> None:
        self.assertEqual(
            spark.DECODE_CASE_GROUPS["portable"],
            ["count", "explain", "tasks", "rows", "math", "chat", "essay", "story", "chat-sampled"],
        )
        for case in spark.DECODE_CASE_GROUPS["portable"]:
            payloads = [payload for _, payload in spark.decode_payloads("m", case, 8)]
            self.assertEqual(len({payload["messages"][0]["content"] for payload in payloads}), 8, case)
            for payload in payloads:
                self.assertEqual(payload["min_tokens"], 256)
                self.assertEqual(payload["max_tokens"], 256)
                self.assertTrue(payload["ignore_eos"])
                self.assertEqual(payload["chat_template_kwargs"], {"thinking": False})
        # One stream gets the exact prompt; code streams get distinct tasks,
        # tagged once the tasks repeat.
        single = spark.decode_payloads("m", "count", 1)[0][1]
        self.assertEqual(single["messages"][0]["content"], spark.PORTABLE_CASES["count"])
        tasks = [payload["messages"][0]["content"] for _, payload in spark.decode_payloads("m", "tasks", 10)]
        self.assertEqual(tasks[0], spark.PORTABLE_CODE_TASKS[0])
        self.assertEqual(tasks[9], f"[stream 10]\n{spark.PORTABLE_CODE_TASKS[1]}")
        sampled = [payload for _, payload in spark.decode_payloads("m", "chat-sampled", 2)]
        self.assertEqual([(p["temperature"], p["top_p"], p["seed"]) for p in sampled], [(0.7, 0.95, 42), (0.7, 0.95, 43)])

    def test_decode_window_excludes_prefill_and_start_stagger(self) -> None:
        requests = [
            {"ok": True, "first_s": 0.2, "last_s": 2.2, "completion_tokens": 101},
            {"ok": True, "first_s": 0.4, "last_s": 3.2, "completion_tokens": 201},
            {"ok": False, "first_s": 0.1, "last_s": 9.0, "completion_tokens": 50},
        ]
        self.assertEqual(spark.decode_window_tps(requests), 100.0)
        self.assertIsNone(spark.decode_window_tps([{"ok": True, "first_s": None, "last_s": None, "completion_tokens": 9}]))

    def test_source_text_is_real_reproducible_text(self) -> None:
        text = spark.source_text(20000, 3)
        self.assertEqual(text, spark.source_text(20000, 3))
        self.assertNotEqual(text, spark.source_text(20000, 4))
        self.assertTrue(text.startswith("Document 3:\n"))
        self.assertEqual(len(text), len("Document 3:\n") + int(20000 * spark.SOURCE_CHARS_PER_TOKEN))
        self.assertIn("def ", text)

    def test_assess_lru(self) -> None:
        good = "```python\nclass LRU:\n    def get(self, key):\n        pass\n    def put(self, key, value):\n        pass\n```"
        self.assertEqual(spark.assess_lru(good), "pass")
        self.assertEqual(spark.assess_lru("no code"), "missing Python code block")
        self.assertIn("syntax error", spark.assess_lru("```python\nclass (:\n```"))
        wrong = good.replace("key, value", "k, v")
        self.assertIn("signature", spark.assess_lru(wrong))


HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None
needs_jsonschema = unittest.skipUnless(HAS_JSONSCHEMA, "python-jsonschema is not installed")


class PrecisionSetTest(unittest.TestCase):
    def test_precision_cases_use_the_1k_budget_without_forcing_length(self) -> None:
        self.assertEqual(spark.DECODE_CASE_GROUPS["precision"], ["prose-1k", "code-1k", "json-1k"])
        for case in spark.DECODE_CASE_GROUPS["precision"]:
            spec = spark.DECODE_CASES[case]
            self.assertEqual(
                (spec.max_tokens, spec.min_samples, spec.max_samples, spec.converge_on, spec.fixed_length),
                (1024, 5, 12, "step_ms", False), case,
            )
            payloads = [payload for _, payload in spark.decode_payloads("m", case, 8)]
            self.assertEqual(len({payload["messages"][0]["content"] for payload in payloads}), 8, case)
            for payload in payloads:
                self.assertEqual(payload["max_tokens"], 1024)
                self.assertNotIn("min_tokens", payload)
                self.assertNotIn("ignore_eos", payload)
                self.assertEqual(payload["temperature"], 0)
                self.assertEqual(payload["chat_template_kwargs"], {"thinking": False})
        self.assertEqual(len(spark.DECODE_CASES["json-1k"].json_schema), 8)
        # Code tasks are sized past the budget: implementation, tests and a CLI each.
        self.assertEqual(len(set(spark.PRECISION_CODE_TASKS)), 8)
        self.assertNotEqual(spark.DECODE_CASES["code-1k"].prompt, spark.PORTABLE_CODE_TASKS)
        for task in spark.PRECISION_CODE_TASKS:
            self.assertIn("unittest suite", task)
            self.assertIn("argparse", task)
        self.assertEqual(len(set(spark.PRECISION_JSON_PROMPTS)), 8)
        # Every prompt names every key of its own schema, so prompt and check cannot drift.
        for prompt, schema in zip(spark.PRECISION_JSON_PROMPTS, spark.PRECISION_JSON_SCHEMAS):
            for key in schema["properties"]:
                self.assertIn(f"{key} (", prompt)
        self.assertEqual(spark.json_schema_for(spark.DECODE_CASES["json-1k"], 9), spark.PRECISION_JSON_SCHEMAS[1])
        # The portable protocol and the reference prompts are untouched.
        rows = spark.decode_payloads("m", "rows", 1)[0][1]
        self.assertEqual((rows["max_tokens"], rows["min_tokens"]), (256, 256))
        self.assertEqual(spark.DECODE_CASES["prose"], spark.DecodeCase(spark.DECODE_PROMPTS["prose"]))

    def test_structured_output_case_is_opt_in(self) -> None:
        self.assertNotIn("json-schema-1k", spark.DECODE_CASE_GROUPS["precision"])
        payload = spark.decode_payloads("m", "json-schema-1k", 1)[0][1]
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["response_format"]["json_schema"]["schema"]["type"], "array")
        self.assertEqual(spark.DECODE_CASES["json-schema-1k"].json_schema, payload["response_format"]["json_schema"]["schema"]["items"])
        self.assertIn("tags (an array of two strings)", payload["messages"][0]["content"])
        self.assertNotIn("response_format", spark.decode_payloads("m", "json-1k", 1)[0][1])

    def test_schema_key_text_describes_types_counts_and_ranges(self) -> None:
        schema = spark.record_schema(
            a={"type": "string"}, b={"type": "integer"}, c=spark.percent(), d=spark.strings(3),
            e={"type": "array", "items": spark.record_schema(x={"type": "boolean"})},
        )
        self.assertEqual(
            spark.schema_key_text(schema),
            "a (a string), b (an integer), c (a number from 0 to 100), d (an array of three strings) "
            "and e (an array of objects with x (a boolean))",
        )

    @needs_jsonschema
    def test_json_schema_violations_cover_the_record_shapes(self) -> None:
        books = spark.PRECISION_JSON_SCHEMAS[0]
        good = {"title": "T", "author": "A", "year": 1999, "genres": ["x", "y"], "isbn": "1", "pages": 10,
                "available": True, "rating": 4.5}
        self.assertEqual(spark.json_schema_violations(good, books), [])
        bad = dict(good, year="1999", genres=["x"], rating=7, available=1, extra=None)
        del bad["pages"]
        found = spark.json_schema_violations(bad, books)
        self.assertEqual(len(found), 6)
        # Root-level findings come first; their mutual order varies by jsonschema version.
        self.assertEqual(set(found[:2]), {"$: Additional properties are not allowed ('extra' was unexpected)",
                                          "$: 'pages' is a required property"})
        self.assertIn("$.available: 1 is not of type 'boolean'", found)
        self.assertIn("$.genres: ['x'] is too short", found)
        self.assertIn("$.rating: 7 is greater than the maximum of 5", found)
        self.assertIn("$.year: '1999' is not of type 'integer'", found)
        # Booleans are not integers; integers are numbers.
        self.assertEqual(spark.json_schema_violations(True, {"type": "integer"}), ["$: True is not of type 'integer'"])
        self.assertEqual(spark.json_schema_violations(3, {"type": "number"}), [])
        invoice = {"invoiceId": "i", "customer": "c", "issuedOn": "d", "dueOn": "d", "paid": False,
                   "lineItems": [{"description": "x", "amountCents": 5}, {"description": "y", "amountCents": "5"}]}
        self.assertEqual(spark.json_schema_violations(invoice, spark.PRECISION_JSON_SCHEMAS[6]),
                         ["$.lineItems[1].amountCents: '5' is not of type 'integer'"])
        # Every benchmark schema is itself a valid Draft 2020-12 schema.
        for schema in (*spark.PRECISION_JSON_SCHEMAS, spark.PRECISION_JSON_SCHEMA["schema"]):
            spark.schema_validator(schema)

    @needs_jsonschema
    def test_json_progress_scores_objects_against_the_schema_and_saves_outputs(self) -> None:
        schema = spark.record_schema(a={"type": "integer"})
        text = '[{"a": 1}, {"a": "two"}, {"a": 3'
        scored = spark.json_array_progress(text, schema)
        self.assertEqual(
            {key: scored[key] for key in ("json_objects", "json_prefix_valid", "json_valid_objects", "json_violation")},
            {"json_objects": 2, "json_prefix_valid": True, "json_valid_objects": 1,
             "json_violation": "$[1].a: 'two' is not of type 'integer'"},
        )
        self.assertEqual(json.loads(scored["json_prefix"]), [{"a": 1}, {"a": "two"}])
        broken = spark.json_array_progress('[{"a": }', schema)
        self.assertEqual((broken["json_objects"], broken["json_prefix_valid"], broken["json_valid_objects"]), (0, False, 0))
        self.assertNotIn("json_prefix", broken)
        with tempfile.TemporaryDirectory() as tmp:
            request = {"content": text, **scored}
            spark.save_output(Path(tmp), "json-1k-c2", "sample03", 1, request)
            folder = Path(tmp) / "outputs" / "json-1k-c2"
            self.assertEqual((folder / "sample03-stream2.txt").read_text(), text)
            self.assertEqual(json.loads((folder / "sample03-stream2.json").read_text()), [{"a": 1}, {"a": "two"}])
            spark.save_output(Path(tmp), "json-1k-c2", "warmup", 0, {"content": "no json", **spark.json_array_progress("no json", schema)})
            self.assertFalse((folder / "warmup-stream1.json").exists())
            spark.save_output(None, "json-1k-c2", "sample01", 0, request)

    def test_json_array_progress_scores_truncated_arrays(self) -> None:
        text = '[{"a": 1, "t": ["x", "y]"]}, {"a": "}", "n": {"k": 2}}, {"a": 3'
        scored = spark.json_array_progress(text)
        self.assertEqual((scored["json_objects"], scored["json_prefix_valid"]), (2, True))
        self.assertEqual(scored["json_prefix"], '[{"a": 1, "t": ["x", "y]"]}, {"a": "}", "n": {"k": 2}}]')
        self.assertEqual(spark.json_array_progress("Sure! Here is"), {"json_objects": 0, "json_prefix_valid": False})
        # The decoder stops at the first element that is not JSON; what came before still counts.
        partial = spark.json_array_progress('[{"a": 1}, {"b": }]')
        self.assertEqual((partial["json_objects"], partial["json_prefix_valid"], partial["json_prefix"]), (1, True, '[{"a": 1}]'))
        nothing = spark.json_array_progress('[{"a": }')
        self.assertEqual((nothing["json_objects"], nothing["json_prefix_valid"]), (0, False))
        self.assertNotIn("json_prefix", nothing)
        mixed = spark.json_array_progress('["s", 1, {"a": 1}]')
        self.assertEqual((mixed["json_objects"], mixed["json_prefix_valid"]), (1, True))
        escaped = spark.json_array_progress('[{"q": "\\\\"}, {"r": "\\"}"}')
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
        entry = spark.sample_metrics({"wall_s": 11.0, "requests": requests, "metrics": metrics}, 2)
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
        single = spark.sample_metrics({"wall_s": 11.0, "requests": requests[:1], "metrics": metrics}, 1)
        self.assertAlmostEqual(single["step_ms"], 1000 * 4.096 / 102.3, places=2)
        self.assertEqual(spark.convergence_values([entry, single], "step_ms"), [entry["step_ms"], single["step_ms"]])
        self.assertEqual(spark.convergence_values([entry, single], "tps"), [entry["tps"], single["tps"]])
        # A failed stream leaves the step time undefined rather than wrong.
        broken = [dict(requests[0]), dict(requests[1], ok=False)]
        self.assertIsNone(spark.sample_metrics({"wall_s": 11.0, "requests": broken, "metrics": metrics}, 2)["step_ms"])
        # Older reports kept only single-stream requests; their step times still derive.
        legacy = [{"accepted_per_draft": 3.0, "requests": [{"ok": True, "decode_tps": 100.0}]}]
        self.assertEqual(spark.sample_step_times(legacy), [40.0])

    def test_decode_plan_applies_case_floors_and_cli_override(self) -> None:
        options = argparse.Namespace(min_samples=3, max_samples=4, converge_on=None)
        spec, concurrency, metric, minimum, maximum = spark.decode_plan(options, "json-1k-c8")
        self.assertEqual((spec, concurrency, metric, minimum, maximum), (spark.DECODE_CASES["json-1k"], 8, "step_ms", 5, 12))
        self.assertEqual(spark.decode_plan(options, "prose-c1")[2:], ("tps", 3, 4))
        options.converge_on = "accepted_per_verified"
        self.assertEqual(spark.decode_plan(options, "json-1k-c1")[2], "accepted_per_verified")
        options.min_samples, options.max_samples = 8, 20
        self.assertEqual(spark.decode_plan(options, "json-1k-c1")[3:], (8, 20))

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
            with contextlib.redirect_stdout(io.StringIO()) as progress:
                report = spark.suite_decode(bench, options, random.Random(0))
            saved = sorted(path.name for path in (Path(tmp) / "outputs" / "json-1k-c2").iterdir())
        self.assertIn("all points converged", progress.getvalue())
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
        lines = spark.decode_primary_rows(points)
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
        self.assertIn("compliance", spark.BENCH_SUITES)
        self.assertNotIn("compliance", spark.FULL_SUITES)
        self.assertEqual(spark.BENCH_SUITE_FUNCTIONS["compliance"], spark.suite_compliance)

    def test_compliance_payloads_ask_for_a_complete_array_with_room(self) -> None:
        plain = [payload for _, payload in spark.compliance_payloads("m", False)]
        strict = [payload for _, payload in spark.compliance_payloads("m", True)]
        self.assertEqual(len(plain), len(spark.PRECISION_JSON_RECORDS))
        for payload in plain:
            self.assertEqual(payload["max_tokens"], 4096)
            self.assertNotIn("min_tokens", payload)
            self.assertNotIn("response_format", payload)
            self.assertIn("array of 12 invented", payload["messages"][0]["content"])
            self.assertEqual(payload["chat_template_kwargs"], {"thinking": False})
        schema = strict[0]["response_format"]["json_schema"]["schema"]
        self.assertEqual((schema["type"], schema["minItems"], schema["maxItems"]), ("array", 12, 12))
        self.assertEqual(schema["items"], spark.PRECISION_JSON_SCHEMAS[0])

    def test_used_whole_budget_prefers_the_finish_reason(self) -> None:
        self.assertTrue(spark.used_whole_budget({"finish_reason": "length", "completion_tokens": 10}, 1024))
        self.assertFalse(spark.used_whole_budget({"finish_reason": "stop", "completion_tokens": 1024}, 1024))
        self.assertTrue(spark.used_whole_budget({"completion_tokens": 1024}, 1024))
        self.assertFalse(spark.used_whole_budget({"completion_tokens": 900}, 1024))

    @needs_jsonschema
    def test_assess_json_records_requires_a_clean_complete_conforming_array(self) -> None:
        schema = spark.record_schema(a={"type": "integer"})
        good = json.dumps([{"a": index} for index in range(3)])
        self.assertEqual(spark.assess_json_records(good, schema, 3)["ok"], True)
        short = spark.assess_json_records(json.dumps([{"a": 1}]), schema, 3)
        self.assertEqual((short["ok"], short["objects"], short["violation"]), (False, 1, "1 objects, 3 requested"))
        fenced = spark.assess_json_records(f"```json\n{good}\n```", schema, 3)
        self.assertEqual((fenced["ok"], fenced["fenced"], fenced["conforming"]), (False, True, 3))
        bad = spark.assess_json_records(json.dumps([{"a": 1}, {"a": "x"}, {"a": 3}]), schema, 3)
        self.assertEqual((bad["ok"], bad["conforming"], bad["violation"]), (False, 2, "$[1].a: 'x' is not of type 'integer'"))
        self.assertFalse(spark.assess_json_records("Sure thing", schema, 3)["parsed"])
        self.assertEqual(spark.assess_json_records('{"a": 1}', schema, 1)["violation"], "top level is not an array")

    @needs_jsonschema
    def test_suite_compliance_reports_both_arms_and_saves_outputs(self) -> None:
        class StubBench:
            def __init__(self) -> None:
                self.labels: list[str] = []

            def run(self, label, payloads, concurrency=None, keep_content=False, retries=3):
                self.labels.append(label)
                requests = []
                for index, (subject, schema) in enumerate(spark.PRECISION_JSON_RECORDS):
                    content = json.dumps([
                        {key: ({"string": "s", "integer": 1, "number": 2.5, "boolean": True}.get(prop.get("type"))
                               if prop.get("type") != "array" else
                               (["x", "y", "z"][: prop["minItems"]] if prop["items"].get("type") == "string"
                                else [{"description": "d", "amountCents": 1}] * prop["minItems"]))
                         for key, prop in schema["properties"].items()}
                        for _ in range(spark.COMPLIANCE_OBJECTS)
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
            with contextlib.redirect_stdout(io.StringIO()) as progress:
                report = spark.suite_compliance(bench, options, random.Random(0))
            saved = sorted(p.name for p in (Path(tmp) / "outputs" / "compliance-constrained").iterdir())
        self.assertIn("compliance prompt_only library books: hit the token budget", progress.getvalue())
        self.assertEqual(bench.labels, ["compliance prompt_only", "compliance constrained"])
        self.assertEqual(report["arms"]["prompt_only"]["passed"], len(spark.PRECISION_JSON_RECORDS) - 1)
        self.assertEqual(report["arms"]["prompt_only"]["requests"][0]["violation"], "hit the token budget")
        self.assertEqual(report["arms"]["constrained"]["passed"], len(spark.PRECISION_JSON_RECORDS))
        self.assertEqual(report["arms"]["constrained"]["accepted_per_verified"], 0.75)
        self.assertFalse(report["ok"])
        self.assertIn("library-books.txt", saved)
        lines, _ = spark.bench_rows({"suites": {"compliance": report}}, None, 3.0)
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
        options = spark.argparse.Namespace(
            model="m", admission_tokens=4000, max_model_len=4096,
            admission_output_tokens=1024, admission_force_length=forced,
        )
        with self.assertRaises(Captured):
            spark.suite_admission(bench, options, spark.random.Random(42))
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
        options = spark.parser().parse_args(["bench"])
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
        result = spark.stream_request(url, {"model": "m"}, threading.Event(), keep_content=True)
        self.assertTrue(result["ok"])
        self.assertEqual(result["finish_reason"], "stop")
        self.assertIn("finish_reason", spark.compact([result])[0])
        self.assertEqual(result["content"], "hello")
        self.assertEqual(result["completion_tokens"], 3)
        self.assertEqual(result["prompt_tokens"], 5)
        self.assertEqual(result["reasoning_chars"], 6)
        self.assertEqual(result["content_chars"], 5)
        self.assertEqual(spark.reasoning_share([result]), 0.545)
        # The last token is the last emitted text, not the usage chunk.
        self.assertLessEqual(result["first_at"], result["last_at"])

    def test_completion_stream_counts_an_empty_first_token(self) -> None:
        url = self.serve(
            [
                {"choices": [{"text": ""}]},
                {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 1}},
            ]
        )
        result = spark.stream_request(url, {"model": "m"}, threading.Event())
        self.assertTrue(result["ok"])
        self.assertIsNotNone(result["ttft_s"])
        self.assertIsNone(result["finish_reason"])

    def test_failed_request_is_recorded(self) -> None:
        result = spark.stream_request("http://127.0.0.1:9", {"model": "m"}, threading.Event())
        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["error"])


class ReferenceShapeTest(unittest.TestCase):
    TP3_REFERENCE = "manifests/benchmarks/2026-10-05-karmic-kraken-r6.json"
    TP4_REFERENCE = "manifests/benchmarks/2026-10-10-karmic-kraken-r6d-tp4.json"
    OLD_REFERENCE = "manifests/benchmarks/2026-10-02-karmic-kraken-r5o-64k.json"
    TP4_MISMATCH = "was measured with nodes 3, transport oneshot-direct; this {} has nodes 4, transport oneshot-ring4"
    # The profiles read the site's git-ignored node maps; use the example map
    # of the same size.
    EXAMPLE_NODES = {"3": "config/nodes.example.json", "4": "config/examples/nodes-ring4.json"}

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def profile(self, name: str) -> tuple[dict, dict]:
        cluster = spark.read_json(name)
        return cluster, spark.read_json(self.EXAMPLE_NODES[spark.serve_arg(cluster, "--tensor-parallel-size")])

    def report(self, cluster: dict, nodes: dict) -> str:
        """A report with no suites, measured on the profile's shape."""
        path = Path(self.tmp.name) / "bench.json"
        identity = {
            "topology": nodes,
            "transport": spark.topology.transport(cluster),
            "kernel_backend": spark.kernel_backend.backend(cluster),
        }
        path.write_text(json.dumps({"identity": identity, "suites": {}}))
        return str(path)

    def bench(self, name: str, cluster: dict, nodes: dict, *argv: str) -> tuple[int, str]:
        options = spark.parser().parse_args(["--cluster-config", name, "bench", *argv])
        with mock.patch.object(spark, "configuration", return_value=(cluster, nodes, {})), \
                mock.patch.object(spark, "bench_identity", side_effect=AssertionError("contacted the cluster")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            status = options.func(options)
        return status, output.getvalue()

    def test_promoted_profiles_default_to_a_reference_of_their_shape(self) -> None:
        for path in sorted((ROOT / "config").glob("cluster*.json")):
            name = path.relative_to(ROOT).as_posix()
            with self.subTest(profile=name):
                cluster, nodes = self.profile(name)
                reference, _ = spark.benchmark_reference(cluster)
                self.assertTrue(reference.is_file())
                relative = reference.relative_to(ROOT)
                self.assertEqual(relative.parent.as_posix(), "manifests/benchmarks")
                # A promotion that moves promoted_baseline moves the reference too.
                self.assertTrue(relative.stem.startswith(cluster["promoted_baseline"]))
                identity = json.loads(reference.read_text())["identity"]
                self.assertEqual(
                    spark.shape_differences(spark.profile_shape(cluster, nodes), spark.bench_shape(identity)),
                    ([], []),
                )
                self.assertEqual(spark.benchmark_reference_problems(cluster, nodes), [])

    def test_tp4_profile_compares_with_the_tp4_run(self) -> None:
        cluster, nodes = self.profile("config/cluster-tp4.json")
        self.assertEqual(cluster["benchmark_reference"], self.TP4_REFERENCE)
        status, output = self.bench("config/cluster-tp4.json", cluster, nodes, "--report", self.report(cluster, nodes))
        self.assertEqual(status, 0)
        self.assertIn(f"reference: {ROOT / self.TP4_REFERENCE}", output)
        self.assertNotIn("WARN", output)
        self.assertNotIn("ERROR", output)
        cluster["benchmark_reference"] = "manifests/benchmarks/missing.json"
        with self.assertRaisesRegex(SystemExit, "benchmark_reference does not exist: manifests/benchmarks/missing.json"):
            self.bench("config/cluster-tp4.json", cluster, nodes, "--report", self.report(cluster, nodes))

    def test_default_reference_of_another_shape_is_refused(self) -> None:
        # Without its own reference the TP4 profile falls back to the baseline's
        # three-node run. Refuse it both live, before contacting the cluster,
        # and for a saved report.
        cluster, nodes = self.profile("config/cluster-tp4.json")
        del cluster["benchmark_reference"]
        # On the TP3 profiles' baseline, whose default reference is its three-node run.
        cluster["promoted_baseline"] = spark.read_json("config/cluster.json")["promoted_baseline"]
        for argv in ([], ["--report", self.report(cluster, nodes)]):
            with self.subTest(argv=argv):
                status, output = self.bench("config/cluster-tp4.json", cluster, nodes, *argv)
                self.assertEqual(status, 1)
                self.assertIn(f"ERROR: reference {self.TP3_REFERENCE} {self.TP4_MISMATCH.format('run')}\n", output)
                self.assertIn("refusing to compare with a default reference of another shape", output)
                self.assertIn("--compare PATH", output)

    def test_explicit_reference_of_another_shape_only_warns(self) -> None:
        cluster, nodes = self.profile("config/cluster-tp4.json")
        report = self.report(cluster, nodes)
        status, output = self.bench(
            "config/cluster-tp4.json", cluster, nodes, "--report", report, "--compare", self.TP3_REFERENCE
        )
        self.assertEqual(status, 0)
        self.assertIn(f"WARN: reference {self.TP3_REFERENCE} {self.TP4_MISMATCH.format('run')}\n", output)
        self.assertIn(f"reference: {ROOT / self.TP3_REFERENCE}", output)
        self.assertNotIn("ERROR", output)
        status, output = self.bench("config/cluster-tp4.json", cluster, nodes, "--report", report, "--compare", "none")
        self.assertEqual(status, 0)
        self.assertNotIn("reference", output)

    def test_kernel_backend_counts_and_unrecorded_fields_only_warn(self) -> None:
        cluster, nodes = self.profile("config/cluster.json")
        r5p = spark.read_json("manifests/benchmarks/2026-10-05-karmic-kraken-r5p.json")["identity"]
        self.assertEqual(
            spark.shape_differences(spark.profile_shape(cluster, nodes), spark.bench_shape(r5p)),
            (["transport", "kernel_backend"], []),
        )
        # Reports before r5p record neither the node map nor the transport, and
        # those before r6 no kernel_backend, which means B12X.
        self.assertEqual(spark.bench_shape({}), {"nodes": None, "transport": None, "kernel_backend": "b12x"})
        name = "experiments/2026-10-05-tilelang-r6/r5p/cluster-64k.json"
        b12x, nodes = self.profile(name)
        b12x["promoted_baseline"] = "2026-10-02-karmic-kraken-r5o-64k"
        status, output = self.bench(name, b12x, nodes, "--report", self.report(b12x, nodes))
        self.assertEqual(status, 0)
        self.assertIn(f"WARN: cannot compare nodes, transport with reference {self.OLD_REFERENCE}: not recorded\n", output)
        self.assertNotIn("ERROR", output)

    def test_doctor_checks_a_named_reference(self) -> None:
        cluster, nodes = self.profile("config/cluster-tp4.json")
        self.assertEqual(spark.benchmark_reference_problems(cluster, nodes), [])
        cluster["benchmark_reference"] = self.TP3_REFERENCE
        self.assertEqual(
            spark.benchmark_reference_problems(cluster, nodes),
            [f"benchmark_reference {self.TP3_REFERENCE} {self.TP4_MISMATCH.format('profile')}"],
        )
        cluster["benchmark_reference"] = self.OLD_REFERENCE
        problems = spark.benchmark_reference_problems(cluster, nodes)
        self.assertEqual(spark.split_findings(problems), (
            [f"benchmark_reference {self.OLD_REFERENCE} was measured with kernel_backend b12x; "
             "this profile has kernel_backend tilelang"],
            [f"benchmark_reference {self.OLD_REFERENCE} does not record nodes, transport; bench cannot check them"],
        ))
        cluster["benchmark_reference"] = "manifests/benchmarks/missing.json"
        self.assertEqual(
            spark.benchmark_reference_problems(cluster, nodes),
            ["benchmark_reference does not exist: manifests/benchmarks/missing.json"],
        )
        for raw in ("", "/tmp/bench.json", "../bench.json", 7):
            with self.subTest(raw=raw):
                cluster["benchmark_reference"] = raw
                self.assertEqual(
                    spark.benchmark_reference_problems(cluster, nodes),
                    ["benchmark_reference must be a bench.json path relative to the repository root"],
                )
        del cluster["benchmark_reference"]
        self.assertEqual(spark.benchmark_reference_problems(cluster, nodes), [])


if __name__ == "__main__":
    unittest.main()
