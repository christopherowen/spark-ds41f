"""Host-independent tests for the benchmark statistics and client parsing."""

from __future__ import annotations

import http.server
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
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
            ]
        )
        result = spark3.stream_request(url, {"model": "m"}, threading.Event(), keep_content=True)
        self.assertTrue(result["ok"])
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

    def test_failed_request_is_recorded(self) -> None:
        result = spark3.stream_request("http://127.0.0.1:9", {"model": "m"}, threading.Event())
        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["error"])


if __name__ == "__main__":
    unittest.main()
