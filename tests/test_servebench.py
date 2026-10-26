"""servebench against a fake streaming server (no cluster, no GPU)."""

import json
import math
import os
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
                         "pos0": 0, "pos1": 0, "finished": 0}
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


class StallTest(unittest.TestCase):
    def test_gaps_split_at_the_prefill(self):
        events = [[0.0, 1], [1.0, 2], [2.0, 3], [5.0, 4], [6.0, 5], [7.0, 6]]
        self.assertEqual(stall.split_gaps(events, 2.5, 5.5), ([1.0, 1.0], [3.0, 1.0]))
        self.assertEqual(stall.split_gaps(events[:1], 0.5, 1.0), ([], []))

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
            rows = stall.summarize(samples)
            self.assertEqual(rows[0]["failed"], 0)
            self.assertEqual(rows[0]["stall_s"]["n"], 2)
            verdicts = {row["metric"]: row["verdict"] for row in stall.compare(rows, rows)["rows"]}
            self.assertEqual(set(verdicts.values()), {"same"})
            self.assertIn("worst decode wait during 3,000 prefill (s)", verdicts)
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
                self.assertEqual(cli.main(["compare", out, out]), 0)
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
                self.assertEqual(len(result["steps"]["stall"][1]["rows"]), 4)
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
