"""Host-independent tests for the live-traffic workload summary."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
spec = importlib.util.spec_from_loader("spark", loader)
spark = importlib.util.module_from_spec(spec)
loader.exec_module(spark)

MODEL = 'engine="0",model_name="m"'


def exposition(requests: int, prompt: float, cached: float, steps: tuple[int, int]) -> str:
    """A small /metrics page; steps are (decode-only, carrying prefill)."""
    decode_only, prefill = steps
    return "\n".join([
        "# TYPE vllm:request_success_total counter",
        f'vllm:request_success_total{{{MODEL},finished_reason="stop"}} {requests}',
        f'vllm:request_success_total{{{MODEL},finished_reason="length"}} 1',
        f"vllm:prompt_tokens_total{{{MODEL}}} {prompt}",
        f"vllm:prompt_tokens_cached_total{{{MODEL}}} {cached}",
        f"vllm:generation_tokens_total{{{MODEL}}} {requests * 100}",
        f"vllm:spec_decode_num_drafts_total{{{MODEL}}} {requests * 10}",
        f"vllm:spec_decode_num_draft_tokens_total{{{MODEL}}} {requests * 40}",
        f"vllm:spec_decode_num_accepted_tokens_total{{{MODEL}}} {requests * 20}",
        f'vllm:spec_decode_num_draft_tokens_per_pos_total{{{MODEL},position="0"}} {requests * 10}',
        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{{MODEL},position="0"}} {requests * 8}',
        f'vllm:spec_decode_num_draft_tokens_per_pos_total{{{MODEL},position="1"}} {requests * 10}',
        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{{MODEL},position="1"}} {requests * 5}',
        f'vllm:tool_call_parser_invocations_total{{{MODEL},mode="streaming",outcome="tool_call",request_type="chat_completions"}} {requests}',
        f'vllm:tool_call_parser_invocations_total{{{MODEL},mode="streaming",outcome="no_tool_call",request_type="chat_completions"}} 3',
        f'vllm:iteration_tokens_total_bucket{{{MODEL},le="8.0"}} {decode_only}',
        f'vllm:iteration_tokens_total_bucket{{{MODEL},le="64.0"}} {decode_only}',
        f'vllm:iteration_tokens_total_bucket{{{MODEL},le="4096.0"}} {decode_only + prefill}',
        f'vllm:iteration_tokens_total_bucket{{{MODEL},le="+Inf"}} {decode_only + prefill}',
        f"vllm:iteration_tokens_total_count{{{MODEL}}} {decode_only + prefill}",
        f"vllm:num_requests_running{{{MODEL}}} 2",
        "",
    ])


class MetricSeriesTest(unittest.TestCase):
    def test_parse_drops_engine_and_model_and_keeps_other_labels(self) -> None:
        series = spark.parse_metric_series(exposition(4, 1000, 250, (90, 10)))
        self.assertEqual(series[("vllm:request_success_total", (("finished_reason", "stop"),))], 4.0)
        self.assertEqual(series[("vllm:prompt_tokens_total", ())], 1000.0)
        self.assertEqual(spark.series_sum(series, "vllm:request_success_total"), 5.0)
        self.assertEqual(
            spark.series_sum(series, "vllm:tool_call_parser_invocations_total", outcome="tool_call"), 4.0
        )

    def test_histogram_quantile_interpolates_inside_the_crossing_bucket(self) -> None:
        series = spark.parse_metric_series("\n".join([
            'h_bucket{le="1.0"} 0',
            'h_bucket{le="2.0"} 50',
            'h_bucket{le="4.0"} 100',
            'h_bucket{le="+Inf"} 100',
        ]))
        self.assertAlmostEqual(spark.histogram_quantile(series, "h", 0.5), 2.0)
        self.assertAlmostEqual(spark.histogram_quantile(series, "h", 0.75), 3.0)
        self.assertAlmostEqual(spark.histogram_quantile(series, "h", 0.25), 1.5)
        self.assertIsNone(spark.histogram_quantile(series, "missing", 0.5))

    def test_quantile_in_the_open_bucket_reports_its_lower_bound(self) -> None:
        series = spark.parse_metric_series('h_bucket{le="10.0"} 1\nh_bucket{le="+Inf"} 4')
        self.assertEqual(spark.histogram_quantile(series, "h", 0.9), 10.0)


class WorkloadSummaryTest(unittest.TestCase):
    def test_window_delta_summary(self) -> None:
        before = spark.parse_metric_series(exposition(2, 1000, 200, (50, 5)))
        after = spark.parse_metric_series(exposition(6, 9000, 6200, (150, 30)))
        delta = {key: value - before.get(key, 0.0) for key, value in after.items()}
        summary = spark.workload_summary(delta, 1800.0, {"vllm:num_requests_running": [1.0, 3.0]})
        self.assertEqual(summary["requests"]["finished"], 4)
        self.assertEqual(summary["requests"]["by_reason"], {"stop": 4})
        self.assertEqual(summary["requests"]["per_hour"], 8.0)
        self.assertEqual(summary["prompt"]["tokens"], 8000)
        self.assertEqual(summary["prompt"]["cached_share"], 0.75)
        self.assertEqual(summary["output"]["tokens"], 400)
        self.assertEqual(summary["speculative"]["accepted_per_draft"], 2.0)
        self.assertEqual(summary["speculative"]["verified_per_draft"], 4.0)
        self.assertEqual(summary["speculative"]["acceptance_by_position"], [0.8, 0.5])
        self.assertEqual(summary["steps"]["count"], 125)
        self.assertEqual(summary["steps"]["carrying_prefill_share"], 0.2)
        self.assertEqual(summary["load"]["num_requests_running"], {"max": 3.0, "mean": 2.0})


if __name__ == "__main__":
    unittest.main()
