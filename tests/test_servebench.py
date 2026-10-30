"""servebench against a fake streaming server (no cluster, no GPU)."""

import json
import math
import os
import statistics
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.server as t_http
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from servebench import __main__ as cli  # noqa: E402
from servebench import behaviour, client, engine, load, prefill, prom, prompts, report, stall, stats, suite  # noqa: E402

CHUNK_TOKENS = 2      # tokens per streamed chunk (a "step" of the fake engine)
CHUNK_DELAY_S = 0.004
LONG_PROMPT = 2000    # characters: a prompt longer than this prefills (FakeServer.prefill_stall_s)
VOCAB = 1000


def next_token(token: int) -> int:
    """The fake model's greedy choice after ``token``."""
    return (token * 7 + 3) % VOCAB


def fake_distribution(previous: int, top: int, variant: bool) -> list[tuple[int, float]]:
    """Top ``top`` (token, log-probability) after ``previous``: the greedy token first."""
    probabilities = [0.5 ** (k + 1) for k in range(top)]
    if variant:  # move a little mass from the greedy token to the runner-up
        probabilities[0] -= 0.004
        probabilities[1] += 0.004
    return [((next_token(previous) + 37 * k) % VOCAB, math.log(p)) for k, p in enumerate(probabilities)]


class _ThreadingServer(ThreadingHTTPServer):
    # The clients open many connections at once; with socketserver's listen backlog of 5,
    # macOS resets the connections beyond it (Linux queues them).
    request_queue_size = 128


class FakeServer:
    """Streams max_tokens / 2 chunks of two tokens each; counts steps and tokens."""

    def __init__(self, step_scale: float = 1.0, variant: bool = False, prefill_stall_s: float = 0.0):
        self.step_scale = step_scale  # engine grid: multiplies every step time
        self.variant = variant        # model behaviour: a build whose numbers differ slightly
        # A prompt longer than LONG_PROMPT characters prefills for this long, and every
        # stream's next chunk waits for it, as decode steps wait for a prefill chunk.
        self.prefill_stall_s = prefill_stall_s
        self.stall_until = 0.0
        self.resets = 0
        self.seen: list[tuple[str, int]] = []  # (cache salt, prompt characters) per streamed request
        self.lock = threading.Lock()
        self.counters = {"generation": 0, "steps": 0, "step_tokens": 0, "drafts": 0, "accepted": 0,
                         "pos0": 0, "pos1": 0, "finished": 0, "chunk_steps": 0}
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/v1/models":
                    body = json.dumps({"data": [{"id": "fake-model", "max_model_len": 1048576}]}).encode()
                elif self.path == "/metrics":
                    with fake.lock:
                        c = dict(fake.counters)
                    body = (
                        "# HELP vllm:generation_tokens_total x\n"
                        f'vllm:generation_tokens_total{{model_name="fake"}} {c["generation"]}\n'
                        f'vllm:iteration_tokens_total_count{{engine="0"}} {c["steps"]}\n'
                        f'vllm:iteration_tokens_total_sum{{engine="0"}} {c["step_tokens"]}\n'
                        f'vllm:iteration_tokens_total_bucket{{engine="0",le="1024.0"}} {c["steps"]}\n'
                        f'vllm:iteration_tokens_total_bucket{{engine="0",le="+Inf"}} '
                        f'{c["steps"] + c["chunk_steps"]}\n'
                        f'vllm:spec_decode_num_drafts_total{{engine="0"}} {c["drafts"]}\n'
                        f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {c["accepted"]}\n'
                        f'vllm:spec_decode_num_draft_tokens_per_pos_total{{engine="0",position="0"}} {c["drafts"]}\n'
                        f'vllm:spec_decode_num_draft_tokens_per_pos_total{{engine="0",position="1"}} {c["drafts"]}\n'
                        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="0"}} {c["pos0"]}\n'
                        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="1"}} {c["pos1"]}\n'
                        f'vllm:request_success_total{{finished_reason="length"}} {c["finished"]}\n'
                        'vllm:kv_cache_usage_perc{engine="0"} NaN\n').encode()
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path in ("/collective_rpc", "/reset_prefix_cache"):
                    if self.path == "/reset_prefix_cache":
                        fake.resets += 1
                        body = b""
                    else:
                        spec = json.loads(request["args"][0])
                        workers = []
                        for rank in (0, 1):
                            shapes = []
                            for i, shape in enumerate(spec["shapes"]):
                                base = fake.step_scale * (10 + shape["num_tokens"] / 10 + shape["context_len"] / 1e5)
                                samples = [base * (1 + 0.002 * ((k * 7 + i) % 5 - 2)) for k in range(spec["replays"])]
                                shapes.append({"shape": shape, "step_ms": samples,
                                               "forward_ms": [v * 0.8 for v in samples],
                                               "drafter_ms": [v * 0.2 for v in samples], "modes": []})
                            workers.append({"rank": rank, "replays": spec["replays"], "shapes": shapes})
                        body = json.dumps({"results": workers}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/tokenize":
                    body = json.dumps({"count": len(request["prompt"]) // 4}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/v1/completions" or not request.get("stream"):
                    body = json.dumps(fake.answer(self.path, request)).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                with fake.lock:
                    fake.seen.append((request.get("cache_salt"), len(request["messages"][0]["content"])))
                    long_prompt = fake.prefill_stall_s and len(request["messages"][0]["content"]) > LONG_PROMPT
                    if long_prompt:
                        fake.stall_until = time.monotonic() + fake.prefill_stall_s
                        fake.counters["chunk_steps"] += 1
                per_chunk = request.get("stream_options", {}).get("continuous_usage_stats", False)
                total = request["max_tokens"]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def send(event):
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()

                try:
                    send({"choices": [{"delta": {"role": "assistant"}}]})
                    self.stream(send, total, per_chunk)
                except (BrokenPipeError, ConnectionResetError):  # the client stopped the stream
                    pass

            def stream(self, send, total, per_chunk):
                produced = 0
                while produced < total:
                    time.sleep(max(CHUNK_DELAY_S, fake.stall_until - time.monotonic()))
                    produced += CHUNK_TOKENS
                    with fake.lock:
                        fake.counters["generation"] += CHUNK_TOKENS
                        fake.counters["steps"] += 1
                        fake.counters["step_tokens"] += CHUNK_TOKENS
                        fake.counters["drafts"] += 1
                        fake.counters["accepted"] += CHUNK_TOKENS - 1
                    event = {"choices": [{"delta": {"content": "ab"}}]}
                    if per_chunk:
                        event["usage"] = {"prompt_tokens": 7, "completion_tokens": produced}
                    send(event)
                with fake.lock:
                    fake.counters["finished"] += 1
                send({"choices": [{"delta": {}, "finish_reason": "length"}],
                      "usage": {"prompt_tokens": 7, "completion_tokens": produced}})
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = _ThreadingServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def answer(self, path: str, request: dict) -> dict:
        """Non-streaming replies: greedy generation with token ids, or prompt scoring."""
        if path == "/v1/completions":
            tokens, top = request["prompt"], request["prompt_logprobs"]
            rows = [None]
            for j in range(1, len(tokens)):
                pairs = fake_distribution(tokens[j - 1], top, self.variant)
                entry = {str(t): {"logprob": lp, "rank": k + 1, "decoded_token": "x"}
                         for k, (t, lp) in enumerate(pairs)}
                entry.setdefault(str(tokens[j]), {"logprob": -9.0, "rank": 500, "decoded_token": "y"})
                rows.append(entry)
            return {"choices": [{"text": "x", "prompt_logprobs": rows}]}
        text = request["messages"][0]["content"]
        prompt_ids = [ord(ch) % VOCAB for ch in text[:12]]
        output, token = [], prompt_ids[-1]
        length = request["max_tokens"] if request.get("ignore_eos") else min(request["max_tokens"], 6 + len(text) % 5)
        for i in range(length):
            token = next_token(token)
            if self.variant and len(text) % 3 == 0 and i == 4:
                token = (token + 1) % VOCAB  # this build's greedy choice differs here
            output.append(token)
        with self.lock:
            self.counters["generation"] += length
            self.counters["drafts"] += length // 2
            self.counters["accepted"] += length // 2 + length // 4
            self.counters["pos0"] += length // 2
            self.counters["pos1"] += length // 4
        return {"prompt_token_ids": prompt_ids,
                "choices": [{"message": {"content": "x"}, "token_ids": output,
                             "finish_reason": "length" if length == request["max_tokens"] else "stop"}]}

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class StatsTest(unittest.TestCase):
    def test_t95(self):
        self.assertAlmostEqual(stats.t95(1), 12.706)
        self.assertAlmostEqual(stats.t95(10), 2.228)
        self.assertTrue(2.228 > stats.t95(11) > 2.179)
        self.assertAlmostEqual(stats.t95(10_000), 1.96, places=2)

    def test_describe_and_percentiles(self):
        d = stats.describe([10.0, 12.0, 11.0, 13.0])
        self.assertEqual(d["n"], 4)
        self.assertAlmostEqual(d["mean"], 11.5)
        self.assertAlmostEqual(d["ci95"], 3.182 * d["sd"] / 2)
        p = stats.percentiles(list(range(1, 101)))
        self.assertEqual((p["p50"], p["p90"], p["p99"]), (50, 90, 99))

    def test_t_quantile_and_trim(self):
        for df, exact in ((10, 2.228), (30, 2.042), (120, 1.980)):
            self.assertAlmostEqual(stats.t_quantile(0.975, df), exact, places=2)
        self.assertGreater(stats.t_quantile(1 - 0.025 / 18, 60), 3.0)  # Bonferroni over 18
        kept, dropped = stats.trim_slow([21.4, 21.5, 21.3, 31.6, 21.2, 21.6, 32.0, 21.4])
        self.assertEqual((dropped, max(kept)), (2, 21.6))
        self.assertEqual(stats.trim_slow([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])[1], 0)  # no spread: keep all

    def test_median_interval_is_exact(self):
        ten = stats.median_interval(list(range(10, 0, -1)))
        # Ten values: the 2nd and 9th order statistics cover 1 - 2 * 11/1024.
        self.assertEqual((ten["median"], ten["low"], ten["high"]), (5.5, 2, 9))
        self.assertAlmostEqual(ten["coverage"], 1 - 22 / 1024)
        self.assertEqual(stats.median_interval([1, 2, 3, 4, 5])["low"], None)  # no 95% pair below six
        self.assertAlmostEqual(stats.median_interval([1, 2, 3, 4, 5, 6])["coverage"], 1 - 2 / 64)
        self.assertIsNone(stats.median_interval([]))
        hundred = stats.median_interval([float(v) for v in range(100)])
        self.assertEqual((hundred["low"], hundred["high"]), (39.0, 60.0))
        self.assertGreaterEqual(hundred["coverage"], 0.95)

    def test_welch_and_verdict(self):
        same = stats.welch([10, 10.1, 9.9, 10.0], [10.05, 9.95, 10.0, 10.1])
        self.assertEqual(stats.verdict(same["change_pct"], same["ci95_pct"], True, 3), "same")
        faster = stats.welch([10, 10.1, 9.9, 10.0], [11, 11.1, 10.9, 11.0])
        self.assertEqual(stats.verdict(faster["change_pct"], faster["ci95_pct"], True, 3), "better")
        self.assertEqual(stats.verdict(5.0, None, False, 3), "worse (no interval)")


class PromTest(unittest.TestCase):
    def test_parse_sums_series_and_skips_nan(self):
        text = ('# TYPE x counter\nx{a="1"} 2\nx{a="2"} 3\ny 1.5\nz{q="}"} NaN\n'
                'h_count{le="+Inf"} 4 1700000000\n')
        self.assertEqual(prom.parse(text), {"x": 5.0, "y": 1.5, "h_count": 4.0})

    def test_window_splits_rate_into_steps_and_tokens_per_step(self):
        start = {"steps": 100, "generation_tokens": 300, "drafts": 100, "accepted_tokens": 150}
        end = {"steps": 200, "generation_tokens": 600, "drafts": 200, "accepted_tokens": 330}
        out = prom.window(start, end, 2.0)
        self.assertEqual(out["steps_per_s"], 50)
        self.assertEqual(out["output_tokens_per_step"], 3)
        self.assertEqual(out["generation_tps"], 150)
        self.assertAlmostEqual(out["accepted_per_draft"], 1.8)


    def test_positions_scrape_and_window(self):
        fake = FakeServer()
        try:
            behaviour.generate(fake.url, "fake-model", prompts.default_prompts()[:2], concurrency=1, max_tokens=8)
            scraped = prom.scrape(fake.url, prom.VLLM)
            counters = dict(fake.counters)
        finally:
            fake.close()
        self.assertEqual({k: scraped[k] for k in ("accepted_at_0", "accepted_at_1", "verified_at_0", "verified_at_1")},
                         {"accepted_at_0": counters["pos0"], "accepted_at_1": counters["pos1"],
                          "verified_at_0": counters["drafts"], "verified_at_1": counters["drafts"]})
        self.assertGreater(counters["pos1"], 0)
        start = {"drafts": 0, "accepted_at_0": 0, "accepted_at_1": 0, "verified_at_0": 0, "verified_at_1": 0}
        end = {"drafts": 10, "accepted_at_0": 8, "accepted_at_1": 5, "verified_at_0": 10, "verified_at_1": 7}
        window = prom.window(start, end, 1.0)
        self.assertEqual((window["survival_at_0"], window["survival_at_1"]), (0.8, 0.5))
        self.assertEqual((window["coverage_at_0"], window["coverage_at_1"]), (1.0, 0.7))


class PromptsTest(unittest.TestCase):
    def test_default_set_is_256_distinct_and_stable(self):
        p = prompts.default_prompts()
        self.assertEqual(len(p), 256)
        self.assertEqual(len({x["text"] for x in p}), 256)
        self.assertEqual(prompts.digest(p), prompts.digest(prompts.default_prompts()))


class LiveTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeServer()

    def tearDown(self):
        self.fake.close()

    def test_stream_chat_exact_and_approximate_token_times(self):
        body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
        exact = client.stream_chat(self.fake.url, body, continuous_usage=True)
        self.assertIsNone(exact["error"])
        self.assertEqual([n for _, n in exact["events"]], [2, 4, 6, 8])
        self.assertEqual(exact["token_times"], "exact")
        self.assertEqual(exact["completion_tokens"], 8)
        rough = client.stream_chat(self.fake.url, body, continuous_usage=False)
        self.assertEqual(rough["token_times"], "approximate")
        self.assertEqual([n for _, n in rough["events"]], [2, 4, 6, 8])
        middle = (exact["events"][1][0] + exact["events"][2][0]) / 2
        self.assertEqual(client.tokens_at(exact, middle), 4)
        self.assertEqual(client.tokens_at(exact, exact["sent"]), 0)

    def test_stream_error_event_is_recorded(self):
        class Handler(t_http.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'data: {"error": {"message": "max_tokens is too large", "code": 400}}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
        server = t_http.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            record = client.stream_chat(f"http://127.0.0.1:{server.server_address[1]}", {"model": "m", "messages": []})
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(record["error"], "server: max_tokens is too large")

    def test_closed_loop_window_counts_tokens_and_steps(self):
        spec = load.Load(mode="closed", concurrency=3, warmup_s=0.3, duration_s=0.9, batches=3,
                         max_tokens=16)
        raw = load.run(self.fake.url, "fake-model", prompts.default_prompts(), spec, log=lambda m: None)
        summary = report.summarize(raw)
        self.assertEqual(summary["requests"]["failed"], 0)
        self.assertEqual(summary["output_tps"]["n"], 3)
        self.assertAlmostEqual(summary["inflight_mean"], 3, delta=0.3)
        # Client-side tokens in the window match the server's generation counter.
        server_tps = summary["server"]["generation_tps"]["mean"]
        self.assertAlmostEqual(summary["output_tps"]["mean"], server_tps, delta=0.15 * server_tps)
        self.assertAlmostEqual(summary["server"]["output_tokens_per_step"]["mean"], CHUNK_TOKENS, places=6)
        self.assertAlmostEqual(summary["server"]["accepted_per_draft"]["mean"], CHUNK_TOKENS - 1, places=6)
        self.assertGreater(summary["latency"]["tpot_ms"]["p50"], 0)
        self.assertEqual(summary["requests"]["foreign"], 0)

    def test_someone_elses_requests_are_foreign(self):
        """Requests the server finishes beyond the run's own (another client on the same
        server) are counted, so the run's speeds are not taken for the build's."""
        spec = load.Load(mode="closed", concurrency=2, warmup_s=0.2, duration_s=0.6, batches=2,
                         max_tokens=8)

        def visitor():
            time.sleep(0.4)
            with self.fake.lock:
                self.fake.counters["finished"] += 5

        threading.Thread(target=visitor, daemon=True).start()
        raw = load.run(self.fake.url, "fake-model", prompts.default_prompts(), spec, log=lambda m: None)
        self.assertEqual(report.summarize(raw)["requests"]["foreign"], 5)

    def test_context_streams_keep_their_documents(self):
        server = FakeServer()
        try:
            spec = load.Load(mode="closed", concurrency=2, warmup_s=0.1, duration_s=0.4, batches=2,
                             max_tokens=8, context_tokens=2000)
            raw = load.run(server.url, "fake-model", prompts.load("default"), spec, log=lambda m: None)
        finally:
            server.close()
        self.assertEqual([d["slot"] for d in raw["documents"]], [0, 1])
        for doc in raw["documents"]:
            self.assertLess(abs(doc["tokens"] - 2000), 20)
            self.assertIsNone(doc["prime_error"])
        salts = {salt for salt, _ in server.seen}
        self.assertEqual(len(salts), 2)  # priming and every question reuse the stream's salt
        self.assertTrue(all(chars > 7000 for _, chars in server.seen))
        self.assertGreater(len(server.seen), 4)

    def test_source_text_is_seeded_and_sized(self):
        a, b = prompts.source_text(1000, 3), prompts.source_text(1000, 3)
        self.assertEqual(a, b)
        self.assertNotEqual(a, prompts.source_text(1000, 4))
        self.assertTrue(a.startswith("Document 3:\n"))
        self.assertEqual(len(a), len("Document 3:\n") + int(1000 * prompts.SOURCE_CHARS_PER_TOKEN))

    def test_poisson_arrivals_follow_the_rate(self):
        spec = load.Load(mode="poisson", rate=40.0, warmup_s=0.2, duration_s=1.0, batches=2,
                         max_tokens=4, seed=3)
        raw = load.run(self.fake.url, "fake-model", prompts.default_prompts(), spec, log=lambda m: None)
        sent = len(raw["records"])
        expected = 40.0 * 1.2
        self.assertLess(abs(sent - expected), 4 * math.sqrt(expected))
        summary = report.summarize(raw, slo_ttft_s=10.0, slo_tpot_ms=1000.0)
        self.assertEqual(summary["latency"]["slo"]["attainment_pct"], 100.0)

    def test_cli_run_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = []
            for label in ("base", "candidate"):
                path = os.path.join(tmp, f"{label}.json")
                code = cli.main(["run", self.fake.url, "--concurrency", "2", "--warmup", "0.2",
                                 "--duration", "0.6", "--batches", "3", "--max-tokens", "8",
                                 "--label", label, "--output", path, "--save-requests"])
                self.assertEqual(code, 0)
                with open(path, encoding="utf-8") as handle:
                    outputs.append(json.load(handle))
            self.assertEqual(outputs[0]["schema"], "servebench/1")
            self.assertIn("events", outputs[0]["requests"][0])
            result = report.compare(outputs[0], outputs[1])
            self.assertFalse(result["warnings"])
            metrics = {row["metric"] for row in result["rows"]}
            self.assertIn("output tok/s", metrics)
            self.assertIn("engine steps/s", metrics)
            self.assertIn("output tokens/step", metrics)
            # Every run keeps its request table, on the run's clock with its window.
            table = outputs[0]["request_table"]
            self.assertEqual(table["columns"], list(report.TABLE_COLUMNS))
            self.assertEqual(table["window"], [0.2, 0.8])
            row = dict(zip(table["columns"], table["rows"][0]))
            self.assertEqual((row["index"], row["completion_tokens"], row["first_tokens"], row["chunks"]),
                             (0, 8, CHUNK_TOKENS, 8 // CHUNK_TOKENS))
            self.assertLess(row["sent"], row["first"])
            self.assertLessEqual(row["first"], row["last"])
            self.assertEqual(row["prompt"], outputs[1]["request_table"]["rows"][0][1])  # the same seeded order
            paired = result["paired"]
            self.assertGreater(paired["pairs"], 0)
            self.assertEqual(paired["same_chunks"], paired["pairs"])
            self.assertIn("decode tok/s, paired", metrics)
            self.assertIn("paired:", report.format_comparison(result, "base", "candidate"))


class PairedTest(unittest.TestCase):
    """Requests paired by index: text that varies by prompt cancels."""

    @staticmethod
    def record(rates, ttfts, *, window=(10.0, 1000.0), prompt=lambda i: f"d{i:03d}", errors=(), tokens=256):
        rows, now = [], 10.0
        for i, (rate, ttft) in enumerate(zip(rates, ttfts)):
            first = now + ttft
            last = first + (tokens - 1) / rate
            rows.append([i, prompt(i), 0, now, first, last, last + 0.001, 20, tokens, 1, 110,
                         "boom" if i in errors else None])
            now = last + 0.01
        batches = [statistics.fmean(rates[k:k + 8]) for k in range(0, len(rates), 8)]
        return {"workload": {"mode": "closed", "concurrency": 1, "prompts_sha256": "x"},
                "summary": {"output_tps": stats.describe(batches) | {"batches": batches},
                            "latency": {"ttft_s": {"p50": 0.13, "p99": 0.16}}},
                "request_table": {"columns": list(report.TABLE_COLUMNS), "window": list(window), "rows": rows}}

    # Eighty prompts whose decode speed spreads ±20% with the text, and their TTFTs.
    TEXT = [75.0 * (1 + 0.2 * math.sin(1.7 * i)) for i in range(80)]
    TTFT = [0.13 * (1 + 0.1 * math.cos(i)) for i in range(80)]

    def jitter(self, scale: float, seed: int) -> list[float]:
        return [1 + scale * math.sin(seed + 2.3 * i) for i in range(80)]

    def test_a_small_change_resolves_and_the_same_build_does_not(self):
        base = self.record(self.TEXT, self.TTFT)
        faster = self.record([v * 1.005 * j for v, j in zip(self.TEXT, self.jitter(0.0005, 1))],
                          [t * 0.98 for t in self.TTFT])
        again = self.record([v * j for v, j in zip(self.TEXT, self.jitter(0.0005, 2))], self.TTFT)
        result = report.paired(base, faster)
        self.assertEqual((result["pairs"], result["prompt_mismatches"]), (80, 0))
        self.assertAlmostEqual(result["decode_tps"]["change_pct"], 0.5, delta=0.06)
        self.assertGreater(result["decode_tps"]["low_pct"], 0.3)
        self.assertAlmostEqual(result["ttft_s"]["change_pct"], -2.0, delta=0.01)
        rows = {row["metric"]: row for row in report.compare(base, faster)["rows"]}
        self.assertEqual((rows["decode tok/s, paired"]["verdict"], rows["TTFT, paired"]["verdict"]), ("better", "better"))
        self.assertEqual(rows["output tok/s"]["verdict"], "same")  # the window interval holds the text
        same = report.paired(base, again)["decode_tps"]
        self.assertLess(same["low_pct"], 0)
        self.assertGreater(same["high_pct"], 0)
        order = [row["metric"] for row in report.compare(base, faster)["rows"]]
        self.assertEqual(order[order.index("output tok/s") + 1], "decode tok/s, paired")

    def test_only_both_windows_matching_prompts_and_successes_pair(self):
        base = self.record(self.TEXT, self.TTFT, errors={50})
        other = self.record(self.TEXT, self.TTFT, window=(30.0, 1000.0), prompt=lambda i: "x" if i == 40 else f"d{i:03d}")
        result = report.paired(base, other)
        early = sum(1 for row in other["request_table"]["rows"] if row[3] < 30.0)
        self.assertEqual(result["prompt_mismatches"], 1)
        self.assertEqual(result["pairs"], 80 - 1 - early)
        self.assertEqual(result["decode_tps"]["n"], 80 - 1 - early - 1)  # the failed request has no rate
        self.assertIsNone(report.paired(base, {k: v for k, v in other.items() if k != "request_table"}))

    def test_runs_without_tables_pair_by_window(self):
        base = self.record(self.TEXT, self.TTFT)
        cand = self.record([v * 1.02 for v in self.TEXT], self.TTFT)
        for side in (base, cand):
            del side["request_table"]
        result = report.compare(base, cand)
        rows = {row["metric"]: row for row in result["rows"]}
        self.assertNotIn("decode tok/s, paired", rows)
        self.assertAlmostEqual(rows["output tok/s, window-paired"]["change_pct"], 2.0, places=6)
        self.assertEqual(rows["output tok/s, window-paired"]["pairs"], 10)
        self.assertIn("window-paired", result["note"])

    def test_rows_a_step(self):
        card = report.rows_a_step({"server": {
            "output_tokens_per_step": {"mean": 2.28}, "accepted_per_draft": {"mean": 1.28},
            "draft_tokens_per_draft": {"mean": 3.09}, "survival_at_0": {"mean": 0.75},
            "coverage_at_0": {"mean": 1.0}, "survival_at_1": {"mean": 0.52}, "coverage_at_1": {"mean": 0.9}}})
        self.assertAlmostEqual(card["acceptance_length"], 2.28)
        self.assertAlmostEqual(card["decoding_requests_per_step"], 1.0)
        self.assertAlmostEqual(card["verified_rows_per_step"], 4.09)
        self.assertAlmostEqual(card["row_yield"], 2.28 / 4.09)
        self.assertAlmostEqual(card["wasted_rows_per_step"], 4.09 - 2.28)
        self.assertEqual([p["position"] for p in card["positions"]], [0, 1])
        self.assertAlmostEqual(card["positions"][0]["conditional_acceptance"], 0.75)
        self.assertAlmostEqual(card["positions"][0]["verified_yield"], 0.75)
        self.assertAlmostEqual(card["positions"][1]["conditional_acceptance"], 0.52 / 0.75)
        self.assertAlmostEqual(card["positions"][1]["verified_yield"], 0.52 / 0.9)
        self.assertEqual(report.rows_a_step({"server": {}})["positions"], [])
        text = suite.format_accounting({"cand": {"c1": card}, "ref": {"c1": report.rows_a_step({"server": {}})}})
        self.assertIn("0:0.750/1.000 1:0.520/0.900", text)
        self.assertIn("[cond/yield 0:0.750/0.750 1:0.693/0.578]", text)
        self.assertIn("(no speculative counters)", text)

    def test_spec_decomposition_identities_and_formatting(self):
        base_summary = {"server": {
            "steps_per_s": {"mean": 32.0},
            "output_tokens_per_step": {"mean": 18.24},
            "accepted_per_draft": {"mean": 1.28},
            "draft_tokens_per_draft": {"mean": 3.00},
            "survival_at_0": {"mean": 0.75}, "coverage_at_0": {"mean": 1.00},
            "survival_at_1": {"mean": 0.53}, "coverage_at_1": {"mean": 0.90},
        }}
        cand_summary = {"server": {
            "steps_per_s": {"mean": 34.4},
            "output_tokens_per_step": {"mean": 19.04},
            "accepted_per_draft": {"mean": 1.38},
            "draft_tokens_per_draft": {"mean": 2.60},
            "survival_at_0": {"mean": 0.78}, "coverage_at_0": {"mean": 1.00},
            "survival_at_1": {"mean": 0.60}, "coverage_at_1": {"mean": 0.85},
        }}
        decomp = report.spec_decomposition(base_summary, cand_summary)
        self.assertIsNotNone(decomp)
        self.assertAlmostEqual(
            decomp["server_tps"]["log_change"],
            decomp["acceptance_length"]["log_change"]
            + decomp["steps_per_s"]["log_change"]
            + decomp["requests_per_step"]["log_change"],
            places=12,
        )
        tokens_log_change = math.log(19.04 / 18.24)
        self.assertAlmostEqual(
            tokens_log_change,
            decomp["verified_rows_per_step"]["log_change"] + decomp["row_yield"]["log_change"],
            places=12,
        )
        self.assertEqual(len(decomp["positions"]), 2)
        self.assertAlmostEqual(decomp["positions"][1]["survival_delta"], 0.07)
        self.assertAlmostEqual(decomp["positions"][1]["coverage_delta"], -0.05)
        self.assertAlmostEqual(
            decomp["positions"][1]["conditional_acceptance_delta"],
            (0.60 / 0.78) - (0.53 / 0.75),
        )
        self.assertAlmostEqual(
            decomp["positions"][1]["verified_yield_delta"],
            (0.60 / 0.85) - (0.53 / 0.90),
        )
        base_run = {"workload": {"mode": "closed", "concurrency": 8, "prompts_sha256": "x"},
                    "summary": base_summary | {"output_tps": {"mean": 583.68, "batches": [583.0, 584.0]}}}
        cand_run = {"workload": {"mode": "closed", "concurrency": 8, "prompts_sha256": "x"},
                    "summary": cand_summary | {"output_tps": {"mean": 654.98, "batches": [654.0, 656.0]}}}
        compared = report.compare(base_run, cand_run)
        self.assertIn("spec_decomposition", compared)
        formatted = report.format_comparison(compared, "base", "cand")
        self.assertIn("speculative log-tps:", formatted)
        self.assertIn("row yield", formatted)


class StallTest(unittest.TestCase):
    def test_gaps_split_at_the_prefill(self):
        events = [[0.0, 1], [1.0, 2], [2.0, 3], [5.0, 4], [6.0, 5], [7.0, 6]]
        self.assertEqual(stall.split_gaps(events, 2.5, 5.5), ([1.0, 1.0], [3.0, 1.0]))
        self.assertEqual(stall.split_gaps(events[:1], 0.5, 1.0), ([], []))

    def test_admission_is_the_gaps_before_the_first_chunk_step(self):
        gaps = [(0.0, 0.08), (0.08, 0.07), (0.15, 2.2), (2.35, 0.08), (2.43, 2.1)]
        self.assertEqual(stall.admission(gaps, 0.08), 2)
        self.assertEqual(stall.admission(gaps[:2], 0.08), 0)
        self.assertEqual(stall.admission(gaps, None), 0)

    def test_long_share_counts_the_steps_a_chunk_held(self):
        self.assertEqual(stall.long_share([0.1, 2.2, 0.1, 2.3], 0.1), 0.5)
        self.assertIsNone(stall.long_share([], 0.1))
        self.assertIsNone(stall.long_share([2.0], None))

    def test_engine_window_counts_steps_by_size(self):
        def scrape(small, total):
            return (f'vllm:iteration_tokens_total_bucket{{engine="0",le="64.0"}} {small}\n'
                    f'vllm:iteration_tokens_total_bucket{{engine="0",le="1024.0"}} {small}\n'
                    f'vllm:iteration_tokens_total_bucket{{engine="0",le="+Inf"}} {total}\n')
        window = stall.engine_window(scrape(100, 110), scrape(116, 134))
        self.assertEqual(window["steps"], 24)
        self.assertEqual(window["buckets"], [["64.0", 16], ["1024.0", 16], ["+Inf", 24]])
        self.assertIsNone(stall.engine_window(None, scrape(1, 1)))

    def test_comparison_skips_metrics_an_older_record_lacks(self):
        def row(**samples):
            return {"size": 3000, "prompt_tokens": 3000, "samples": samples,
                    **{key: stats.describe(values) for key, values in samples.items()}}
        old = row(stall_s=[2.2, 2.3], during_p50_s=[0.8, 2.1])
        new = row(stall_s=[2.2, 2.3], during_mean_s=[1.2, 1.2], long_share=[0.5, 0.5])
        metrics = [row["metric"] for row in stall.compare([old], [new])["rows"]]
        self.assertEqual(metrics, ["worst decode wait during 3,000 prefill (s)"])

    def test_streams_wait_for_a_long_prefill(self):
        fake = FakeServer(prefill_stall_s=0.3)
        try:
            samples = stall.run(fake.url, "fake-model", [3000], streams=3, max_tokens=200000, warmup_s=0.2,
                                tail_s=0.1, repeats=2)
            for sample in samples:
                self.assertIsNone(sample["error"])
                self.assertEqual(sample["covering"], 3)
                self.assertGreaterEqual(sample["ttft_s"], 0.29)
                self.assertGreaterEqual(sample["stall_s"], 0.25)
                self.assertLess(sample["before_s"]["p50"], 0.05)
                self.assertGreater(sample["long_share"], 0)
                self.assertEqual(sum(not gap[3] for gap in sample["gaps"]), sample["during_s"]["n"])
                stream, start, length, admitted = max(sample["gaps"], key=lambda gap: gap[2])
                self.assertFalse(admitted)
                self.assertIn(stream, range(3))
                self.assertLess(start, sample["ttft_s"])
                self.assertEqual(length, round(sample["stall_s"], 4))
                self.assertGreaterEqual(sample["engine"]["steps"], 1)
            rows = stall.summarize(samples)
            self.assertEqual(rows[0]["failed"], 0)
            self.assertEqual(rows[0]["stall_s"]["n"], 2)
            verdicts = {row["metric"]: row["verdict"] for row in stall.compare(rows, rows)["rows"]}
            self.assertEqual(set(verdicts.values()), {"same"})
            self.assertIn("worst decode wait during 3,000 prefill (s)", verdicts)
            self.assertIn("decode step mean during 3,000 prefill (s)", verdicts)
            self.assertNotIn("decode step p50 during 3,000 prefill (s)", verdicts)
            # The streams were stopped at the document's first token, not run to their budget.
            self.assertLess(fake.counters["generation"], 100000)
        finally:
            fake.close()

    def test_probes_wait_for_the_prefill(self):
        fake = FakeServer(prefill_stall_s=0.4)
        try:
            sample = stall.run(fake.url, "fake-model", [3000], streams=2, max_tokens=200000, warmup_s=0.2,
                               tail_s=0.6, repeats=1, probes=(0.1, 0.5), probe_tokens=40,
                               probe_max_tokens=4)[0]
            self.assertIsNone(sample["error"])
            during, after = sample["probes"]
            self.assertTrue(during["during"])
            self.assertGreaterEqual(during["ttft_s"], 0.25)  # waited for the document's prefill
            self.assertFalse(after["during"])
            self.assertLess(after["ttft_s"], 0.2)
            self.assertEqual(sample["probe_ttft_s"]["n"], 1)
            self.assertEqual(sample["probe_ttft_max_s"], during["ttft_s"])
            rows = stall.summarize([sample, sample])
            self.assertEqual(rows[0]["probe_ttft_p50_s"]["n"], 2)
            metrics = {row["metric"] for row in stall.compare(rows, rows)["rows"]}
            self.assertIn("short request TTFT p50 during 3,000 prefill (s)", metrics)
            # Each probe was cold and short.
            salts = [salt for salt, chars in fake.seen if chars < LONG_PROMPT and salt]
            self.assertGreaterEqual(len(set(salts)), 2)
        finally:
            fake.close()

    def test_short_streams_fail_the_repeat(self):
        fake = FakeServer(prefill_stall_s=0.3)
        try:
            sample = stall.run(fake.url, "fake-model", [3000], streams=2, max_tokens=8, warmup_s=0.2,
                               tail_s=0.1, repeats=1)[0]
            self.assertEqual(sample["covering"], 0)
            self.assertIn("did not decode across the prefill", sample["error"])
            self.assertEqual(stall.summarize([sample])[0]["failed"], 1)
        finally:
            fake.close()

    def test_cli_stall_compare_and_suite_arguments(self):
        fake = FakeServer(prefill_stall_s=0.2)
        try:
            with tempfile.TemporaryDirectory() as directory:
                out = os.path.join(directory, "stall.json")
                argv = ["stall", fake.url, "--sizes", "3000", "--streams", "2", "--max-tokens", "200000",
                        "--warmup", "0.1", "--tail", "0.1", "--repeats", "2", "--output", out]
                self.assertEqual(cli.main(argv), 0)
                result = json.loads(Path(out).read_text())
                self.assertEqual(result["schema"], stall.SCHEMA)
                self.assertEqual(result["workload"]["streams"], 2)
                self.assertNotIn("samples", result)
                self.assertIn("samples", result["stall"][0])
                self.assertEqual(cli.main(["compare", out, out]), 0)
                self.assertEqual(cli.main([*argv, "--save-samples"]), 0)
                self.assertIn("samples", json.loads(Path(out).read_text()))
            step = {"id": "stall", "command": "stall",
                    "args": {"sizes": [131072], "streams": 8, "max_tokens": 16384, "warmup": 20, "repeats": 3}}
            parsed = cli.build_parser().parse_args(suite.arguments(step, "http://h:8000", "o.json", "x", None))
            self.assertEqual((parsed.sizes, parsed.streams, parsed.max_tokens, parsed.warmup),
                             ("131072", 8, 16384, 20.0))
        finally:
            fake.close()

    def test_a_suite_runs_and_compares_a_stall_step(self):
        fake = FakeServer(prefill_stall_s=0.2)
        steps = [{"id": "stall", "command": "stall",
                  "args": {"sizes": [3000], "streams": 2, "max_tokens": 200000, "warmup": 0.1, "tail": 0.1,
                           "repeats": 2}}]
        definition = {"name": "stall-only", "version": 1, "description": "a stall step", "steps": steps}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "stall.json")
                Path(path).write_text(json.dumps(definition))
                self.assertEqual(suite.load(path)["steps"], steps)
                base, cand = os.path.join(tmp, "base"), os.path.join(tmp, "cand")
                for out in (base, cand):
                    self.assertEqual(cli.main(["suite", fake.url, "--suite", path, "--out", out]), 0)
                result = suite.compare(cli.read_json, base, cand, lambda b, c: cli.compare_results(b, c, 3.0))
                self.assertEqual(result["steps"]["stall"][0], stall.SCHEMA)
                self.assertEqual(len(result["steps"]["stall"][1]["rows"]), 7)
        finally:
            fake.close()

    def test_a_stopped_stream_is_not_an_error(self):
        fake = FakeServer()
        try:
            stop = threading.Event()
            body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 200000}
            timer = threading.Timer(0.1, stop.set)
            timer.start()
            record = client.stream_chat(fake.url, body, continuous_usage=True, stop=stop)
            self.assertTrue(record["stopped"])
            self.assertIsNone(record["error"])
            self.assertGreater(len(record["events"]), 1)
        finally:
            fake.close()


class EngineTest(unittest.TestCase):
    def test_default_grid_reaches_the_longest_context(self):
        shapes = engine.default_shapes(drafts=5, max_context=1048576)
        labels = [engine.label(s) for s in shapes]
        self.assertIn("decode 16x96 rows @4K", labels)  # full depth
        self.assertIn("decode 16x32 rows @4K", labels)  # depth 1, as serving runs under load
        self.assertIn("decode 1x3 rows @1016K", labels)  # 8K below the 1M maximum
        self.assertFalse(any(s.get("prefill") for s in shapes))
        self.assertEqual(len(labels), len(set(labels)))
        self.assertIn("prefill 8192 @256K", [engine.label(s) for s in engine.default_shapes(5, 1048576, prefill=True)])
        self.assertNotIn("decode 1x3 rows @1016K", [engine.label(s) for s in engine.default_shapes(5, 262144)])
        # Three nodes serve eight streams: no shape of more requests than the server takes.
        tp3 = engine.default_shapes(5, 524288, max_requests=8)
        self.assertEqual(max(s["num_reqs"] for s in tp3), 8)
        self.assertIn("decode 1x3 rows @504K", [engine.label(s) for s in tp3])
        self.assertEqual(len(tp3), 15)  # 6 at depths 1-2, 4 at full depth, 2 at 64K, 2 at 256K, 1 longest

    def test_grid_run_summary_and_compare(self):
        fast, slow = FakeServer(1.0), FakeServer(1.05)
        try:
            runs = []
            for server in (fast, slow):
                shapes = engine.default_shapes(5, engine.max_model_len(server.url))
                raw = engine.run(server.url, shapes, replays=12)
                self.assertTrue(raw["prefix_cache_reset"])
                self.assertEqual(server.resets, 1)
                self.assertEqual(raw["rank"], 0)
                runs.append({"engine": engine.summarize(raw)})
            row = runs[0]["engine"][0]
            self.assertEqual(row["step_ms"]["n"], 12)
            self.assertLess(row["step_ms"]["ci95_pct"], 0.5)
            self.assertIn("forward_ms", row)
            result = engine.compare(runs[0], runs[1])
            self.assertEqual({r["verdict"] for r in result["rows"]}, {"worse"})
            self.assertTrue(result["rows"][-1]["metric"].startswith("decode geometric mean"))
            self.assertIn("Bonferroni", result["note"])
            self.assertAlmostEqual(result["rows"][0]["change_pct"], 5.0, places=1)
        finally:
            fast.close()
            slow.close()

    def test_spread_across_boots(self):
        def grid(label, scale):
            rows = [{"label": f"s{i}", "shape": {"num_tokens": i}, "samples": [scale * (10 + i)] * 6} for i in range(3)]
            return {"label": label, "engine": rows}
        one = [grid("a1", 1.000), grid("a2", 1.004), grid("a3", 0.996)]
        result = engine.spread(one)
        self.assertAlmostEqual(result["range_pct"], 0.8, places=2)
        self.assertAlmostEqual(result["sd_pct"], 0.4, places=2)
        two = [grid("b1", 0.980), grid("b2", 0.984), grid("b3", 0.976)]
        compared = engine.spread(one, two)
        self.assertAlmostEqual(compared["difference"]["change_pct"], -2.0, places=1)
        self.assertEqual(compared["difference"]["verdict"], "better")
        self.assertIn("difference of means", engine.format_spread(compared))

    def test_grid_refuses_a_busy_server(self):
        busy = FakeServer()
        try:
            original = prom.scrape
            prom.scrape = lambda url, names, timeout=10.0: {"requests_running": 2.0}
            with self.assertRaises(RuntimeError):
                engine.run(busy.url, engine.default_shapes(), replays=2)
        finally:
            prom.scrape = original
            busy.close()


class BehaviourTest(unittest.TestCase):
    def test_by_label_sums_one_label(self):
        text = ('m{engine="0",position="0"} 3\nm{engine="1",position="0"} 2\n'
                'm{engine="0",position="1"} 4\nm{engine="0",xposition="1"} 9\nother{position="0"} 7\n')
        self.assertEqual(prom.by_label(text, "m", "position"), {"0": 5.0, "1": 4.0})

    def test_kl_divergence(self):
        half = math.log(0.5)
        self.assertEqual(behaviour.kl_divergence({1: half, 2: half}, {1: half, 2: half}, half), 0.0)
        kl = behaviour.kl_divergence({1: half, 2: half}, {1: math.log(0.25), 2: math.log(0.75)}, -9.0)
        self.assertAlmostEqual(kl, 0.5 * math.log(2) + 0.5 * math.log(0.5 / 0.75))
        # the unlisted rest of the vocabulary is one bucket
        kl = behaviour.kl_divergence({1: math.log(0.9)}, {1: math.log(0.8)}, -9.0)
        self.assertAlmostEqual(kl, 0.9 * math.log(0.9 / 0.8) + 0.1 * math.log(0.1 / 0.2))

    def test_model_runs_self_agreement_and_compare(self):
        base, again, other = FakeServer(), FakeServer(), FakeServer(variant=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                paths = {name: os.path.join(tmp, f"{name}.json.gz") for name in ("base", "again", "other")}
                self.assertEqual(cli.main(["model", base.url, "--limit", "12", "--top", "5",
                                           "--label", "base", "--output", paths["base"]]), 0)
                for name, server in (("again", again), ("other", other)):
                    self.assertEqual(cli.main(["model", server.url, "--limit", "12", "--top", "5",
                                               "--corpus", paths["base"], "--label", name,
                                               "--output", paths[name]]), 0)
                runs = {name: cli.read_json(path) for name, path in paths.items()}
            self.assertEqual(runs["base"]["self_agreement"]["rank1_fraction"], 1.0)
            accepted = runs["base"]["generation"]["acceptance"]
            self.assertEqual([row["position"] for row in accepted["per_position"]], [0, 1])
            self.assertGreater(accepted["acceptance_length"], 1.0)
            self.assertEqual(runs["again"]["corpus"]["sha256"], runs["base"]["corpus"]["sha256"])

            same = behaviour.compare(runs["base"], runs["again"])
            self.assertEqual(same["warnings"], [])
            self.assertEqual(same["generation"]["identical"], 12)
            self.assertEqual(same["scores"]["identical_fraction"], 1.0)
            self.assertEqual(same["scores"]["kl"]["max"], 0.0)

            diff = behaviour.compare(runs["base"], runs["other"])
            self.assertLess(diff["generation"]["identical"], 12)
            self.assertEqual(diff["generation"]["divergence"]["p50"], 4)
            scores = diff["scores"]
            self.assertEqual((scores["skipped"], scores["identical_fraction"], scores["top1_agreement"]), (0, 0.0, 1.0))
            self.assertGreater(scores["kl"]["mean"], 0.0)
            self.assertLess(scores["kl"]["mean"], 1e-3)
            self.assertIn("top-1 agreement", behaviour.format_comparison(diff, "base", "other"))
        finally:
            for server in (base, again, other):
                server.close()


class SuiteTest(unittest.TestCase):
    TINY = {"name": "tiny", "version": 1, "steps": [
        {"id": "model", "command": "model", "args": {"limit": 4, "top": 3, "concurrency": 2}},
        {"id": "grid", "command": "engine", "args": {"replays": 4, "settle": 1}},
        {"id": "c2", "command": "run", "args": {"mode": "closed", "concurrency": 2, "warmup": 0.1,
                                                 "duration": 0.4, "batches": 2, "max_tokens": 8,
                                                 "extra_body": {"chat_template_kwargs": {"thinking": False}}}},
        {"id": "prefill", "command": "prefill", "args": {"sizes": [2000, 4000], "repeats": 2}}]}

    def test_builtin_suites_parse(self):
        for name in ("screen", "release", "screen-tp3", "release-tp3"):
            definition = suite.load(name)
            for step in definition["steps"]:
                with self.subTest(suite=name, step=step["id"]):
                    cli.build_parser().parse_args(suite.arguments(step, "http://x", "/tmp/o.json", "l", "/tmp/c.json.gz"))

    def test_tp3_suites_fit_three_nodes(self):
        """Eight streams and 512K of context: no step asks for more."""
        for name in ("screen-tp3", "release-tp3"):
            for step in suite.load(name)["steps"]:
                with self.subTest(suite=name, step=step["id"]):
                    args = step["args"]
                    self.assertLessEqual(args.get("concurrency", 1), 8)
                    if step["command"] == "engine":
                        self.assertEqual(args["max_requests"], 8)
                    self.assertTrue(all(size <= 524288 - 1024 for size in args.get("sizes", [])))

    def test_release_measures_open_loop_goodput(self):
        poisson = next(s for s in suite.load("release")["steps"] if s["id"] == "poisson")
        self.assertEqual((poisson["args"]["mode"], poisson["args"]["slo_ttft"], poisson["args"]["slo_tpot"]),
                         ("poisson", 2, 100))

    def test_dev_rpc_need(self):
        self.assertTrue(suite.needs_dev_rpc(self.TINY))
        self.assertFalse(suite.needs_dev_rpc({"name": "x", "steps": [s for s in self.TINY["steps"] if s["id"] != "grid"]}))

    def test_arguments(self):
        step = {"id": "p", "command": "prefill", "args": {"sizes": [1, 2], "repeats": 3, "no_vllm": True,
                                                          "extra_body": {"a": 1}, "label_ignored": None}}
        self.assertEqual(suite.arguments(step, "U", "O", "arm", None),
                         ["prefill", "U", "--output", "O", "--label", "arm-p", "--sizes", "1,2", "--repeats", "3",
                          "--no-vllm", "--extra-body", '{"a": 1}'])
        model = {"id": "m", "command": "model", "args": {}}
        self.assertIn("--corpus", suite.arguments(model, "U", "O", None, "C"))
        self.assertEqual(suite.check({"name": "x", "steps": [{"id": "a", "command": "dance"}]}),
                         ["step a: command must be one of model, engine, run, prefill, stall"])

    def test_floor_verdicts(self):
        def outcome(grid, tps):
            return {"steps": {"grid": ("servebench-engine/1", {"rows": [
                        {"metric": "decode geometric mean (20)", "change_pct": grid, "ci95_pct": 0.15}]}),
                    "c16": ("servebench/1", {"rows": [
                        {"metric": "output tok/s", "change_pct": tps, "ci95_pct": 0.9},
                        {"metric": "TTFT p50 s", "change_pct": 9.0, "ci95_pct": None}]})}}
        rows = suite.with_floor(outcome(-0.36, 2.5), outcome(-0.44, 0.1))
        verdicts = {row["metric"]: row["verdict"] for row in rows}
        self.assertEqual(verdicts, {"decode geometric mean (20)": "same", "output tok/s": "better"})
        rows = suite.with_floor(outcome(-1.2, -2.0), outcome(0.2, 0.1))
        self.assertEqual([row["verdict"] for row in rows], ["better", "worse"])  # faster steps; fewer tok/s
        self.assertIn("against the run's own repeat", suite.format_floor(rows, "base", "cand", "again"))

    def test_paired_rows_take_the_recorded_floor(self):
        def outcome(change, half=0.05, repeat_window=0.1):
            return {"steps": {"c1": ("servebench/1", {"rows": [
                {"metric": "output tok/s", "change_pct": repeat_window, "ci95_pct": 3.7},
                {"metric": "decode tok/s, paired", "change_pct": change, "ci95_pct": half,
                 "low_pct": change - half, "high_pct": change + half, "higher_is_better": True}]}),
                "poisson": ("servebench/1", {"rows": [
                    {"metric": "decode tok/s, paired", "change_pct": change, "ci95_pct": half,
                     "low_pct": change - half, "high_pct": change + half, "higher_is_better": True}]})}}
        floors = {"c1": 0.35}
        # One candidate boot: floor 0.35, margin ~0.354. The repeat's own 0.02% would have
        # called the 0.3% "better".
        rows = suite.with_floor(outcome(0.30), outcome(0.02), floors=floors)
        paired = next(r for r in rows if r["metric"] == "decode tok/s, paired")
        self.assertEqual((paired["step"], paired["boots"], paired["verdict"]), ("c1", 1, "same"))
        self.assertAlmostEqual(paired["floor_pct"], 0.35)
        self.assertAlmostEqual(paired["margin_pct"], math.hypot(0.05, 0.35))
        self.assertNotIn("poisson", {r["step"] for r in rows})  # no recorded floor, no paired verdict
        # Both boots: the mean change against a floor smaller by sqrt(3/4).
        rows = suite.with_floor(outcome(0.40), outcome(0.02), floors=floors, reference_to_repeat=outcome(0.30))
        paired = next(r for r in rows if r["metric"] == "decode tok/s, paired")
        self.assertEqual(paired["boots"], 2)
        self.assertAlmostEqual(paired["change_pct"], 0.35)
        self.assertAlmostEqual(paired["floor_pct"], 0.35 * math.sqrt(0.75))
        self.assertEqual(paired["verdict"], "better")
        # A noisy repeat widens the floor.
        rows = suite.with_floor(outcome(0.40), outcome(0.6), floors=floors, reference_to_repeat=outcome(0.30))
        self.assertEqual(next(r for r in rows if r["metric"] == "decode tok/s, paired")["verdict"], "same")
        rows = suite.with_floor(outcome(-1.0), outcome(0.02), floors=floors, reference_to_repeat=outcome(-0.8))
        self.assertEqual(next(r for r in rows if r["metric"] == "decode tok/s, paired")["verdict"], "worse")
        text = suite.format_floor(rows, "ref", "cand", "cand-again")
        self.assertIn("both candidate boots; window tok/s +0.10% ±3.70%", text)
        self.assertIn("in quadrature", text)

    def test_suite_runs_and_compares(self):
        servers = [FakeServer(), FakeServer(variant=True)]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                definition = os.path.join(tmp, "tiny.json")
                Path(definition).write_text(json.dumps(self.TINY))
                base, cand = os.path.join(tmp, "base"), os.path.join(tmp, "cand")
                metadata = os.path.join(tmp, "metadata.json")
                Path(metadata).write_text(json.dumps({"identity": {"transport": "oneshot-ring4"}}))
                self.assertEqual(cli.main(["suite", servers[0].url, "--suite", definition, "--out", base, "--label", "base",
                                           "--metadata", metadata]), 0)
                self.assertEqual(cli.main(["suite", servers[1].url, "--suite", definition, "--out", cand, "--label", "cand",
                                           "--corpus", os.path.join(base, "model.json.gz")]), 0)
                index = json.loads(Path(base, "suite.json").read_text())
                self.assertEqual([s["exit"] for s in index["steps"]], [0, 0, 0, 0])
                self.assertEqual(index["suite_sha256"], suite.digest(self.TINY))
                self.assertEqual(index["metadata"], {"identity": {"transport": "oneshot-ring4"}})
                self.assertNotIn("metadata", json.loads(Path(cand, "suite.json").read_text()))
                result = suite.compare(cli.read_json, base, cand, lambda b, c: cli.compare_results(b, c, 3.0))
                self.assertEqual(list(result["steps"]), ["model", "grid", "c2", "prefill"])
                self.assertEqual(result["steps"]["model"][0], behaviour.SCHEMA)
                self.assertEqual(result["steps"]["prefill"][0], prefill.SCHEMA)
                self.assertEqual(len(result["steps"]["prefill"][1]["rows"]), 2)
                # A reference kept without the model step's text compares everything else.
                os.rename(os.path.join(base, "model.json.gz"), os.path.join(tmp, "model.json.gz"))
                kept = suite.compare(cli.read_json, base, cand, lambda b, c: cli.compare_results(b, c, 3.0))
                self.assertIn("not kept", kept["steps"]["model"]["skipped"])
                self.assertEqual(kept["steps"]["prefill"][0], prefill.SCHEMA)
                changed = json.loads(Path(cand, "suite.json").read_text())
                changed["suite_sha256"] = "0" * 64
                Path(cand, "suite.json").write_text(json.dumps(changed))
                with self.assertRaises(SystemExit):
                    suite.compare(cli.read_json, base, cand, lambda b, c: cli.compare_results(b, c, 3.0))
        finally:
            for server in servers:
                server.close()


class CompareTest(unittest.TestCase):
    def run_summary(self, batches, tokens_per_step):
        return {"workload": {"mode": "closed", "concurrency": 16, "prompts_sha256": "x"},
                "summary": {"output_tps": stats.describe(batches) | {"batches": batches},
                            "server": {"output_tokens_per_step": stats.describe(tokens_per_step)},
                            "server_batches": {"output_tokens_per_step": tokens_per_step},
                            "latency": {"ttft_s": {"p50": 0.5}, "tpot_ms": {"p50": 60.0},
                                        "per_user_tps": {"p50": 16.0}}}}

    def test_verdicts_and_workload_warning(self):
        base = self.run_summary([200, 202, 198, 201], [3.0, 3.1, 2.9, 3.0])
        cand = self.run_summary([220, 221, 219, 222], [3.0, 3.05, 2.95, 3.0])
        cand["workload"]["concurrency"] = 8
        result = report.compare(base, cand)
        rows = {row["metric"]: row for row in result["rows"]}
        self.assertEqual(rows["output tok/s"]["verdict"], "better")
        self.assertEqual(rows["output tokens/step"]["verdict"], "same")
        self.assertEqual(rows["TTFT p50 s"]["verdict"], "same (no interval)")
        self.assertTrue(any("concurrency" in w for w in result["warnings"]))
        self.assertIn("output tok/s", report.format_comparison(result, "base", "cand"))


if __name__ == "__main__":
    unittest.main()
