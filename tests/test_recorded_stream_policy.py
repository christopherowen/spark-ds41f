"""Match independently captured historical collector records without inference."""

import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from servebench import client

HERE = Path(__file__).resolve().parents[1] / "experiments/2026-11-04-lossless-memory/collector"
FIXTURE = json.loads((HERE / "recorded-events.json").read_text())


class Response:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def __iter__(self):
        return iter([("data: " + json.dumps(event) + "\n\n").encode() for event in self.events]
                    + [b"data: [DONE]\n\n"])


class RecordedStreamTests(unittest.TestCase):
    def observe(self, case, policy):
        wires = []
        def request(value, **kwargs):
            wires.append({"url": value.full_url, "method": value.method, "body": json.loads(value.data)})
            return Response(case["events"])
        ticks = iter(float(i) for i in range(100))
        with patch.dict(os.environ, {client.STREAM_POLICY_ENV: policy}), \
                patch.object(client.urllib.request, "urlopen", request):
            try:
                result = client.stream_chat("http://fixture", case["payload"],
                                            continuous_usage=case["continuous_usage"], clock=lambda: next(ticks))
            except Exception as error:
                result = {"raised": f"{type(error).__name__}: {error}"}
        return result, wires

    def test_matches_every_historical_record_and_wire_request(self):
        self.assertEqual(FIXTURE["reference_client_commit"], "19efdd6fe1f11a159c5f1989aaa3433682325ab2")
        self.assertEqual(FIXTURE["inference_requests"], 0)
        for case in FIXTURE["cases"]:
            with self.subTest(case=case["name"]):
                result, wires = self.observe(case, client.RECORDED_STREAM_POLICY)
                self.assertEqual(result, case["expected_record"])
                self.assertEqual(wires, case["expected_wire"])

    def test_current_policy_keeps_progress_events_and_identical_wire_requests(self):
        case = FIXTURE["cases"][0]
        result, wires = self.observe(case, "current")
        self.assertEqual([count for _, count in result["events"]], [1, 2, 3, 4])
        self.assertEqual([count for _, count in case["expected_record"]["events"]], [2, 4])
        self.assertEqual(wires, case["expected_wire"])

    def test_unknown_policy_fails_before_a_request(self):
        with patch.dict(os.environ, {client.STREAM_POLICY_ENV: "unknown"}), \
                patch.object(client.urllib.request, "urlopen") as request:
            with self.assertRaises(ValueError):
                client.stream_chat("http://fixture", {})
            request.assert_not_called()

    def test_default_is_current(self):
        with patch.dict(os.environ):
            os.environ.pop(client.STREAM_POLICY_ENV, None)
            self.assertEqual(client.stream_policy(), "current")


if __name__ == "__main__":
    unittest.main()
