"""Host-independent tests for the AgentX driver (scripts/agentx.py) and lab.py's agentx job."""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import io
import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: str):
    loader = importlib.machinery.SourceFileLoader(name, str(ROOT / path))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    loader.exec_module(module)
    return module


agentx = load("agentx_under_test", "scripts/agentx.py")
lab = load("lab_for_agentx", "scripts/lab.py")
# Host-independent: the shipped example node map, never the real hold and request files.
lab.spark.site_nodes = lambda: lab.spark.read_json("config/nodes.example.json")
_STATE = tempfile.TemporaryDirectory()
lab.HOLD = Path(_STATE.name) / "spark-hold.json"
lab.LEGACY_HOLD = Path(_STATE.name) / "spark3-hold.json"
lab.REQUEST = Path(_STATE.name) / "spark-request.json"
lab.LEGACY_REQUEST = Path(_STATE.name) / "spark3-request.json"

EXPERIMENT = "experiments/2026-10-25-agentx"
ARM = f"{EXPERIMENT}/production-agentx.json"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
HEAD = {"nodes": [{"name": "dgx1", "head": True, "management_ip": "10.0.0.1"}]}
SPEC = {"experiment": EXPERIMENT, "run": "ax",
        "jobs": [{"kind": "agentx", "arm": {"config": "production-agentx.json", "label": "p"}, "points": [8, 16]}]}

METRICS = """# HELP vllm:prefix_cache_hits Prefix cache hits, in terms of number of cached tokens.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 900.0
vllm:prefix_cache_hits_created{engine="0",model_name="m"} 1.7e+09
vllm:prompt_tokens_by_source_total{engine="0",model_name="m",source="local_compute"} 100.0
vllm:prompt_tokens_by_source_total{engine="0",model_name="m",source="local_cache_hit"} 900.0
vllm:request_prefill_kv_computed_tokens_bucket{engine="0",le="1024.0",model_name="m"} 3.0
vllm:iteration_tokens_total_bucket{engine="0",le="1.0",model_name="m"} 7.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 4.0
vllm:other_engine_value{engine="1",model_name="m"} NaN
python_gc_objects_collected_total{generation="0"} 5.0
"""


def series_row(t: float, queries: float, hits: float, kv: float, waiting: float, preempt: float) -> dict:
    return {"t": t, "m": {"vllm:prefix_cache_queries_total": queries, "vllm:prefix_cache_hits_total": hits,
                          "vllm:prompt_tokens_total": queries, "vllm:prompt_tokens_cached_total": hits,
                          'vllm:prompt_tokens_by_source_total{source="local_compute"}': queries - hits,
                          "vllm:num_preemptions_total": preempt, "vllm:kv_cache_usage_perc": kv,
                          "vllm:num_requests_running": 4.0, "vllm:num_requests_waiting": waiting,
                          "vllm:request_success_total": queries / 1000,
                          "vllm:request_prefill_time_seconds_sum": (queries - hits) / 5000,
                          "vllm:request_prefill_time_seconds_count": queries / 1000,
                          "vllm:request_prefill_kv_computed_tokens_sum": queries - hits,
                          "vllm:spec_decode_num_drafts_total": queries / 10,
                          "vllm:spec_decode_num_accepted_tokens_total": queries / 10 * 1.5,
                          "vllm:spec_decode_num_draft_tokens_total": queries / 10 * 2.0,
                          "vllm:generation_tokens_total": queries / 100}}


def write_run(directory: Path) -> None:
    """A finished point as the harness and the sampler leave it (the harness's nesting kept)."""
    harness = directory / "harness" / "run"
    harness.mkdir(parents=True)
    start_ns, end_ns = 1_000_000_000_000_000_000, 1_000_000_100_000_000_000  # t = 1e9 .. 1e9 + 100 s
    (harness / "profile_export_aiperf.json").write_text(json.dumps({
        "metadata": {"scenario": "agentx", "submission_valid": True, "submission_invalid_reasons": []},
        "request_count": {"unit": "requests", "avg": 40.0},
        "error_request_count": None,
        "time_to_first_token": {"unit": "ms", "avg": 9000.0, "p50": 8000.0, "p90": 30000.0, "p99": 41000.0},
        "inter_token_latency": {"unit": "ms", "avg": 20.0, "p50": 19.0, "p90": 25.0, "p99": 40.0},
        "output_token_throughput_per_user": {"unit": "tokens/sec/user", "avg": 50.0, "p10": 40.0, "p50": 52.0,
                                             "p90": 60.0},
        "output_token_throughput": {"unit": "tokens/sec", "avg": 300.0},
        "theoretical_prefix_cache_hit": {"unit": "%", "avg": 97.5},
        # As harness 1.0.7 writes them: empty, so the window comes from the records.
        "min_request_timestamp": None, "max_response_timestamp": None,
        "was_cancelled": False, "error_summary": []}))
    records = []
    for i in range(10):  # TTFT = 0.5 s + 0.2 s per thousand uncached tokens
        uncached = 10000.0 * i
        records.append({"metadata": {"benchmark_phase": "profiling", "request_start_ns": start_ns + i * 1_000_000_000,
                                     "request_end_ns": end_ns - (9 - i) * 1_000_000_000},
                        "metrics": {"time_to_first_token": {"value": 500.0 + 0.2 * uncached, "unit": "ms"},
                                    "usage_prompt_tokens": {"value": 100000.0, "unit": "tokens"},
                                    "usage_prompt_cache_read_tokens": {"value": 100000.0 - uncached,
                                                                       "unit": "tokens"}}})
    records.append({"metadata": {"benchmark_phase": "warmup", "request_start_ns": start_ns - 10 ** 12,
                                 "request_end_ns": start_ns - 10 ** 11},
                    "metrics": {"time_to_first_token": {"value": 1e6, "unit": "ms"}}})
    records.append({"metadata": {"benchmark_phase": "profiling", "request_start_ns": start_ns + 10 ** 9,
                                 "request_end_ns": start_ns + 2 * 10 ** 9}, "metrics": {}, "error": {"code": 500}})
    (harness / "profile_export.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    rows = [series_row(1e9 - 10, 0, 0, 0.0, 0, 0),  # warm-up, before the window
            series_row(1e9 - 2, 10_000, 5_000, 0.1, 0, 0),
            series_row(1e9 + 50, 60_000, 50_000, 0.5, 2, 1),
            series_row(1e9 + 101, 110_000, 95_000, 0.9, 6, 3),
            series_row(1e9 + 200, 900_000, 100_000, 1.0, 9, 9)]  # after the window
    (directory / "server-series.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows) + '{"t": 5, "error": "refused"}\n')


class HarnessCommandTest(unittest.TestCase):
    def test_the_scenario_defaults_stay_and_nothing_trusts_remote_code(self) -> None:
        command = agentx.harness_command("http://h:8001", "m", "org/name", REVISION, 8, "out/harness")
        self.assertEqual(command[:2], [str(agentx.VENV / "bin" / "agentx"), "profile"])
        pairs = dict(zip(command[2::1], command[3::1]))
        self.assertEqual(pairs["--public-dataset"], "semianalysis_cc_traces_weka_062126")
        self.assertEqual(pairs["--random-seed"], "42")
        self.assertEqual(pairs["--benchmark-duration"], "3600")
        self.assertEqual((pairs["--tokenizer"], pairs["--tokenizer-revision"]), ("org/name", REVISION))
        self.assertEqual((pairs["--workers-max"], pairs["--warmup-grace-period"]), ("8", "3600"))
        for forbidden in ("--tokenizer-trust-remote-code", "--max-context-length", "--unsafe-override",
                          "--scenario", "--warmup-requests-per-lane", "--trajectory-start-min-ratio"):
            self.assertNotIn(forbidden, command)
        with self.assertRaises(SystemExit):
            agentx.harness_command("u", "m", "t", REVISION, 8, "o", dataset="semianalysis_cc_traces_weka")

    def test_the_client_runs_offline_on_the_free_cores_inside_a_memory_cap(self) -> None:
        env = agentx.harness_environment({"PATH": "/bin"}, cache=Path("/c"), hub="/hub")
        self.assertEqual({k: env[k] for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")},
                         {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        self.assertEqual((env["AIPERF_DATASET_MMAP_CACHE_DIR"], env["HF_HUB_CACHE"]), ("/c", "/hub"))
        self.assertEqual(env["AIPERF_HTTP_KEEPALIVE_TIMEOUT"], "4")  # below vLLM's 5 s keep-alive
        self.assertEqual(agentx.client_cpus("5-9,15-19", set(range(20))), "0,1,2,3,4,10,11,12,13,14")
        with self.assertRaises(SystemExit):
            agentx.client_cpus("0-3", {0, 1, 2, 3})
        self.assertEqual(agentx.confined(["agentx"], "0,1", "12G"),
                         ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=12G",
                          "-p", "MemorySwapMax=0", "taskset", "-c", "0,1", "agentx"])
        self.assertEqual(agentx.confined(["agentx"], None, None), ["agentx"])

    def test_every_pinned_dataset_names_a_revision(self) -> None:
        for name, (repo, revision) in agentx.DATASETS.items():
            self.assertTrue(repo.startswith("semianalysisai/cc-traces-weka-062126"), name)
            self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_the_requirements_pin_the_harness_that_was_read(self) -> None:
        text = agentx.REQUIREMENTS.read_text()
        self.assertIn(f"agentx-harness=={agentx.HARNESS_VERSION} \\", text)
        # The wheel whose source was reviewed (experiments/2026-10-25-agentx/README.md).
        self.assertIn("sha256:a67dc18b37ab8bf25771f9f677811102df8cf669238b1f55d8b052b062be553c", text)
        for line in text.splitlines():
            if line and not line.startswith((" ", "#", "--index-url")):
                self.assertRegex(line, r"^[A-Za-z0-9_.\-\[\]]+==[^ ]+ \\$")


class MetricsTest(unittest.TestCase):
    def test_parse_keeps_vllm_series_and_the_labels_that_matter(self) -> None:
        parsed = agentx.parse_metrics(METRICS)
        self.assertEqual(parsed["vllm:prefix_cache_hits_total"], 900.0)
        self.assertEqual(parsed['vllm:prompt_tokens_by_source_total{source="local_compute"}'], 100.0)
        self.assertEqual(parsed['vllm:request_prefill_kv_computed_tokens_bucket{le="1024.0"}'], 3.0)
        self.assertEqual(parsed['vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"}'], 4.0)
        self.assertNotIn("vllm:prefix_cache_hits_created", parsed)
        self.assertFalse(any(k.startswith("vllm:iteration_tokens_total_bucket") for k in parsed))
        self.assertFalse(any(k.startswith("python_") or "other_engine" in k for k in parsed))

    def test_window_delta_and_gauges(self) -> None:
        rows = [series_row(t, 1000 * t, 900 * t, t / 10, t, 0) for t in range(10)]
        span = agentx.window(rows, 2.5, 6.5)
        self.assertEqual([row["t"] for row in span], [2, 3, 4, 5, 6, 7])
        self.assertEqual(agentx.delta(span, "vllm:prefix_cache_queries_total"), 5000)
        self.assertEqual(agentx.delta(span, "vllm:prompt_tokens_by_source_total"), 500)  # summed over sources
        self.assertEqual(agentx.gauge(span, "vllm:num_requests_waiting"), {"mean": 4.5, "p50": 4, "p90": 7, "max": 7})
        self.assertIsNone(agentx.delta(span[:1], "vllm:prefix_cache_queries_total"))
        self.assertEqual(agentx.window([], 0, 1), [])

    def test_sampler_writes_a_line_per_scrape_and_survives_a_dead_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.jsonl"
            with agentx.Sampler("http://127.0.0.1:9", path, interval=0.05):
                pass
            lines = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertGreaterEqual(len(lines), 2)
        self.assertTrue(all("error" in line and "t" in line for line in lines))


class SummaryTest(unittest.TestCase):
    def test_a_point_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_run(out)
            (out / "run.json").write_text(json.dumps({"exit": 0, "client_memory_peak_bytes": 3 * 2 ** 30}))
            result = agentx.summarize(out)
            self.assertEqual(json.loads((out / "agentx.json").read_text()), result)
            table = agentx.table("r", ["p-c8", "p-c16"], base=out.parent)
        harness, server = result["harness"], result["server_profiling"]
        self.assertEqual((harness["submission_valid"], harness["requests"], harness["errors"]), (True, 40.0, 0))
        self.assertEqual(harness["ttft_ms"], {"p50": 8000.0, "p90": 30000.0, "p99": 41000.0, "avg": 9000.0})
        self.assertEqual(harness["interactivity_tok_s"]["p50"], 52.0)
        # The profiling window [1e9, 1e9 + 100 s] is bracketed by the samples at -2 s and +101 s.
        self.assertEqual(server["samples"], 3)
        self.assertEqual(harness["profiling_window_utc"], ["2001-09-09T01:46:40Z", "2001-09-09T01:48:20Z"])
        self.assertEqual(server["prefix_cache"], {"queries": 100_000, "hits": 90_000, "hit_rate": 0.9})
        self.assertEqual(server["prompt_tokens"]["computed"], 10_000)
        self.assertEqual(server["preemptions"], 3)
        self.assertEqual(server["kv_cache_usage"]["max"], 0.9)
        self.assertEqual(server["waiting"]["max"], 6)
        self.assertEqual(server["prefill"]["tok_per_request_s"], 5000.0)
        self.assertEqual(server["speculative"], {"accepted_per_draft": 1.5, "verified_per_draft": 2.0})
        self.assertEqual(result["server_whole_run"]["samples"], 5)
        cost = result["miss_cost"]
        self.assertEqual(cost["requests"], 10)  # warm-up and errored records left out
        self.assertEqual(cost["ttft_vs_uncached"]["s_per_1k_tokens"], 0.2)
        self.assertEqual(cost["ttft_vs_uncached"]["intercept_s"], 0.5)
        self.assertEqual(cost["ttft_vs_uncached"]["r2"], 1.0)
        self.assertEqual(cost["prompt_tokens_p50"], 100000.0)
        self.assertEqual(cost["misses_over_half"], 4)  # 60k-90k of a 100k prompt: a lost prefix
        self.assertIn("p-c8", table)
        self.assertIn("(no result)", table)

    def test_a_run_without_harness_output_still_summarizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = agentx.summarize(Path(directory))
        self.assertEqual(result["harness"], {})
        self.assertEqual(result["miss_cost"], {"requests": 0})
        self.assertEqual(result["server_profiling"], {"samples": 0})


class StubTest(unittest.TestCase):
    def test_the_stub_streams_the_usage_the_harness_reads(self) -> None:
        server = agentx.stub_server(0, "m")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=5) as response:
                models = json.load(response)
            body = json.dumps({"model": "m", "stream": True, "max_completion_tokens": 3,
                               "messages": [{"role": "user", "content": "x" * 400}]}).encode()
            request = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=5) as response:
                events = [line[6:] for line in response.read().decode().splitlines() if line.startswith("data: ")]
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(models["data"][0]["id"], "m")
        self.assertEqual(events[-1], "[DONE]")
        usage = json.loads(events[-2])["usage"]
        self.assertEqual(usage, {"prompt_tokens": 100, "completion_tokens": 3, "total_tokens": 103,
                                 "prompt_tokens_details": {"cached_tokens": 90}})
        text = "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-2])
        self.assertEqual(text, "x x x")


class LabJobTest(unittest.TestCase):
    def test_each_point_boots_checks_the_traffic_then_replays(self) -> None:
        steps = lab.plan(SPEC)
        self.assertEqual([s["kind"] for s in steps],
                         ["agentx_prepare", "boot", "clean", "agentx", "boot", "clean", "agentx", "table"])
        self.assertEqual((steps[0]["dataset"], steps[0]["seed"]), ("semianalysis_cc_traces_weka_062126", 42))
        self.assertEqual([s["label"] for s in steps if s["kind"] == "boot"], ["p-c8", "p-c16"])
        point = steps[3]
        self.assertEqual((point["concurrency"], point["duration"], point["seed"], point["dataset"]),
                         (8, 3600, 42, "semianalysis_cc_traces_weka_062126"))
        self.assertEqual(point["out"], "results/private/agentx/ax/p-c8")
        self.assertEqual(steps[2]["seconds"], 15)
        self.assertEqual(steps[-1]["argv"], ["scripts/agentx.py", "table", "ax", "p-c8", "p-c16"])
        for points in ([], [8, 8], [0]):
            bad = json.loads(json.dumps(SPEC))
            bad["jobs"][0]["points"] = points
            with self.assertRaises(SystemExit):
                lab.plan(bad)
        bad = json.loads(json.dumps(SPEC))
        bad["jobs"][0]["dataset"] = "my_local_traces"
        with self.assertRaises(SystemExit):
            lab.plan(bad)

    def test_the_command_targets_the_arm_and_wraps_it_in_workload(self) -> None:
        step = lab.plan(SPEC)[3]
        with mock.patch.object(lab, "nodes_config", return_value=HEAD):
            command = lab.agentx_command(step)
            prepare = lab.agentx_prepare_command(lab.plan(SPEC)[0])
        self.assertEqual(command[1:8], ["bin/spark", "--cluster-config", ARM, "workload", "--interval", "30",
                                        "--label"])
        inner = command[command.index("--") + 1:]
        self.assertEqual(inner[1:4], ["scripts/agentx.py", "run", "http://10.0.0.1:8001"])
        pairs = dict(zip(inner, inner[1:]))
        self.assertEqual(pairs["--model"], "deepseek-v4.1-flash")
        self.assertEqual((pairs["--tokenizer"], pairs["--tokenizer-revision"]),
                         ("deepseek-ai/DeepSeek-V4.1-Flash-1017", REVISION))
        self.assertEqual(pairs["--hf-hub-cache"], "/home/swank/.cache/huggingface/hub")
        self.assertEqual(pairs["--avoid-cpus"], "5-9,15-19")
        self.assertIn("--require-prepared", inner)
        self.assertNotIn("--", inner)  # no harness arguments unless the job names them
        # The rebuild runs inside the window on every core (serving is stopped for it).
        self.assertEqual(prepare[1:4], ["scripts/agentx.py", "prepare", "--in-window"])
        self.assertNotIn("--avoid-cpus", prepare)

    def test_clean_means_nothing_finished_ran_or_waited(self) -> None:
        quiet = {"requests": {"finished": 0}, "load": {"num_requests_running": {"max": 0.0},
                                                       "num_requests_waiting": {"max": 0.0}}}
        self.assertEqual(lab.clean_problems(quiet), [])
        busy = {"requests": {"finished": 2}, "load": {"num_requests_running": {"max": 1.0}}}
        self.assertEqual(lab.clean_problems(busy), ["2 requests finished", "num_requests_running reached 1"])

    def test_problems_name_a_missing_install_dataset_or_tokenizer(self) -> None:
        step = lab.plan(SPEC)[0]
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(lab.agentx, "VENV", Path(directory) / "venv"), \
                mock.patch.object(lab.agentx, "HOME", Path(directory)), \
                mock.patch.object(lab, "nodes_config", return_value=HEAD), \
                mock.patch.object(lab.spark.cluster_recipe, "model_snapshot", return_value=(
                    f"{directory}/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/{REVISION}")):
            problems = lab.agentx_problems(step)
            self.assertEqual([p.split(" ")[:4] for p in problems],
                             [["the", "AgentX", "harness", "is"], ["semianalysis_cc_traces_weka_062126", "is", "not", "fetched"],
                              ["the", "tokenizer", f"{directory}/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/"
                               f"{REVISION}/tokenizer.json", "is"]])
            (Path(directory) / "venv" / "bin").mkdir(parents=True)
            (Path(directory) / "venv" / "bin" / "agentx").write_text("")
            for path in lab.agentx.fetched_paths(step["dataset"], f"{directory}/hub"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            tokenizer = Path(directory) / "hub" / "models--deepseek-ai--DeepSeek-V4.1-Flash-1017" / "snapshots" / REVISION
            tokenizer.mkdir(parents=True)
            (tokenizer / "tokenizer.json").write_text("{}")
            self.assertEqual(lab.agentx_problems(step), [])

    def execute(self, clean: dict, codes: list[int], prepared: list[bool] | None = None,
                running: bool = True) -> tuple[int, mock.Mock, mock.Mock, list]:
        calls = []
        prepared = [True] if prepared is None else prepared

        def run_logged(argv, log_path):
            calls.append(log_path)
            if argv[0] == "prepare":
                return codes.pop(0)
            if "clean" in argv[argv.index("--label") + 1]:
                out = Path(argv[argv.index("--json") + 1])
                return 0 if out.name else 1
            return codes.pop(0)

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            patch = lambda name, **kw: stack.enter_context(mock.patch.object(lab, name, **kw))
            patch("ROOT", new=Path(directory))
            patch("read_hold", return_value={"holder": "test"})
            patch("hold_is_ours", return_value=True)
            patch("window_should_close", return_value=None)
            patch("beat")
            patch("plan_problems", return_value=[])
            patch("overlay_manifest", return_value=[])
            logs = patch("save_engine_logs")
            patch("Telemetry", return_value=contextlib.nullcontext())
            patch("run_logged", side_effect=run_logged)
            patch("agentx_command", side_effect=lambda step: ["spark", "--label", step["label"]])
            patch("agentx_prepare_command", return_value=["prepare"])
            patch("agentx_prepared", side_effect=lambda step: prepared.pop(0))
            patch("container_running", return_value=running)
            stop = patch("stop_cluster", return_value=True)
            boot = patch("boot", return_value=True)
            patch("run_script", return_value=0)
            patch("production_configuration", return_value=(
                {"host": {"home": "/h"}}, {"nodes": [{"name": "dgx1", "head": True, "management_ip": "1"}]}, {}))
            close = patch("window_close", return_value=True)
            stack.enter_context(mock.patch.object(lab.spark, "read_cluster",
                                                  return_value={"container": {"mounts": [], "image": "i"}}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            for label in ("p-c8", "p-c16"):
                path = Path(directory) / "results/private/agentx/ax" / label / "clean.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(clean))
            (Path(directory) / "scripts").mkdir()
            (Path(directory) / "results/private/lab").mkdir(parents=True)
            (Path(directory) / "results/private/lab/ax-table.txt").write_text("table\n")
            code = lab.execute(SPEC, dry=False, keep_open=True)
        self.stopped, self.boots = stop.call_count, boot.call_count
        return code, close, logs, calls

    def test_a_failed_point_keeps_its_evidence_and_the_next_point_runs(self) -> None:
        quiet = {"requests": {"finished": 0}, "load": {}}
        code, close, logs, calls = self.execute(quiet, [1, 0])
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["results/private/agentx/ax/p-c8/clean.json.log", "results/private/agentx/ax/p-c8/lab.log",
                                 "results/private/agentx/ax/p-c16/clean.json.log",
                                 "results/private/agentx/ax/p-c16/lab.log"])
        logs.assert_called_once()
        self.assertIn("agentx points failed: p-c8", close.call_args.args[0])

    def test_a_host_without_the_rebuild_stops_serving_and_rebuilds_first(self) -> None:
        quiet = {"requests": {"finished": 0}, "load": {}}
        code, close, logs, calls = self.execute(quiet, [0, 0, 0], prepared=[False, True])
        self.assertEqual(code, 0)
        self.assertEqual(calls[0], "results/private/agentx/ax/prepare.log")
        self.assertEqual((self.stopped, self.boots), (1, 2))
        # A failed rebuild ends the job before any boot; the window restores production.
        code, close, logs, calls = self.execute(quiet, [1], prepared=[False, False])
        self.assertEqual((code, calls, self.boots), (1, ["results/private/agentx/ax/prepare.log"], 0))
        self.assertIn("the AgentX rebuild failed (exit 1)", close.call_args.args[0])

    def test_traffic_before_a_point_ends_the_job(self) -> None:
        code, close, logs, calls = self.execute({"requests": {"finished": 3}, "load": {}}, [0, 0])
        self.assertEqual(code, 1)
        self.assertEqual(calls, ["results/private/agentx/ax/p-c8/clean.json.log"])
        self.assertIn("not idle before the point (3 requests finished)", close.call_args.args[0])


class ArmTest(unittest.TestCase):
    def test_the_arm_is_production_but_its_port_and_usage_details(self) -> None:
        """The arm copies the recipe's serve_args; it must follow the recipe exactly otherwise."""
        recipe = lab.spark.read_cluster("config/cluster.json", "tp4-ring")
        arm = lab.spark.read_cluster(ARM)
        expected = [("8001" if previous == "--port" else value) for previous, value in
                    zip([None, *recipe["serve_args"]], recipe["serve_args"])]
        if "--enable-prompt-tokens-details" not in expected:
            expected.insert(expected.index("--enable-prefix-caching") + 1, "--enable-prompt-tokens-details")
        self.assertEqual(arm["serve_args"], expected)
        self.assertEqual(arm["deployment"]["ready_url"], "http://127.0.0.1:8001/v1/models")
        for key in set(recipe) | set(arm):
            if key in ("serve_args", "deployment", "shape", "benchmark_reference", "notes"):
                continue
            self.assertEqual(arm.get(key), recipe.get(key), key)
        self.assertEqual({k: v for k, v in arm["deployment"].items() if k != "ready_url"},
                         {k: v for k, v in recipe["deployment"].items() if k != "ready_url"})
        self.assertEqual(lab.arm_problems(ARM), [])

    def test_the_committed_spec_plans(self) -> None:
        spec = json.loads((ROOT / EXPERIMENT / "agentx-production.json").read_text())
        steps = lab.plan(spec)
        self.assertEqual([s["concurrency"] for s in steps if s["kind"] == "agentx"], [8, 16, 4])
        self.assertTrue(all(s["duration"] == 3600 for s in steps if s["kind"] == "agentx"))


if __name__ == "__main__":
    unittest.main()
