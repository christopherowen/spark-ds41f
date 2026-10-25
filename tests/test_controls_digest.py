"""controls_digest.py fails on a broken server: only vLLM's by-design HTTP 400 rejections
(min_p and logit_bias under speculative decoding) count as an expected outcome."""
import http.server
import json
import subprocess
import sys
import threading
import unittest
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[1]
          / "experiments/2026-10-24-tilelang-completion/controls_digest.py")


def serve(status_for):
    """A stub server answering /v1/models and, for chat requests, ``status_for(body)``."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, {"data": [{"id": "m"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            code = status_for(body)
            if code != 200:
                self.reply(code, {"error": {"message": f"status {code}"}})
                return
            self.reply(200, {"choices": [{"message": {"content": "x"}, "logprobs": None,
                                          "finish_reason": "length"}]})

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run(status_for):
    server = serve(status_for)
    try:
        return subprocess.run([sys.executable, str(SCRIPT), f"http://127.0.0.1:{server.server_port}",
                               "--tokens", "2"], capture_output=True, text=True, timeout=120)
    finally:
        server.shutdown()
        server.server_close()


class ControlsDigestTest(unittest.TestCase):
    def test_server_errors_fail_the_run(self):
        result = run(lambda body: 500)
        self.assertEqual(result.returncode, 1)
        summary = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(len(summary["failed"]), 7)

    def test_expected_rejections_pass(self):
        result = run(lambda body: 400 if "min_p" in body or "logit_bias" in body else 200)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(summary["failed"], [])
        self.assertIn("min-p/rejected", summary["digests"])

    def test_a_400_elsewhere_fails(self):
        result = run(lambda body: 400 if "bad_words" in body else 200)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["failed"], ["bad-words"])

    def test_a_concurrent_failure_is_caught(self):
        calls = {"n": 0}

        def status(body):
            # The probe and the eight alone pass; the concurrent eight fail.
            if "repetition_penalty" in body and body["temperature"] == 0.0:
                calls["n"] += 1
                return 500 if calls["n"] > 9 else 200
            return 200

        result = run(status)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["failed"], ["penalties"])


if __name__ == "__main__":
    unittest.main()
