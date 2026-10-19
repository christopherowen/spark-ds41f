"""Profiler workloads must measure their claimed target shape and always stop."""
import io
import json
import unittest
from unittest import mock
from servebench import profile


class OneRowTest(unittest.TestCase):
    def test_raw_token_cold_requests_and_stop_on_error(self):
        calls = []
        def post(url, body, timeout):
            calls.append((url, body))
            return {"usage": {"prompt_tokens": 1}}
        def models():
            return io.BytesIO(json.dumps({"data": [{"id": "model"}]}).encode())
        with mock.patch.object(profile.prom, "require_idle"), \
             mock.patch.object(profile.urllib.request, "urlopen", side_effect=lambda *a, **k: models()), \
             mock.patch.object(profile.engine, "_post", side_effect=post):
            self.assertEqual(profile.one_row("http://server", requests=24)["requests"], 24)
        payloads = [body for url, body in calls if url.endswith("/v1/completions")]
        self.assertEqual(len(payloads), 26)
        self.assertTrue(all(p["prompt"] == [1] and p["max_tokens"] == 1 for p in payloads))
        self.assertEqual(len({p["cache_salt"] for p in payloads}), 26)
        self.assertTrue(calls[-1][0].endswith("/stop_profile"))
        calls.clear()
        def failure(url, body, timeout):
            result = post(url, body, timeout)
            if len(calls) == 4:
                raise RuntimeError("forward failed")
            return result
        with mock.patch.object(profile.prom, "require_idle"), \
             mock.patch.object(profile.urllib.request, "urlopen", side_effect=lambda *a, **k: models()), \
             mock.patch.object(profile.engine, "_post", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                profile.one_row("http://server", requests=24)
        self.assertTrue(calls[-1][0].endswith("/stop_profile"))


class CedAttributionTest(unittest.TestCase):
    def test_excludes_encoder_and_draft_and_refuses_partial_groups(self):
        def routed(layers, tiles, start=0):
            return [dict(cat='kernel', name='moe_grouped_fp8_fp4_gemm_kernel',
                         ts=start + i * 1000, dur=100,
                         args={'grid': [12 if i % 2 == 0 else 40, tiles, 1]})
                    for i in range(layers * 2)]
        events = routed(3, 1143) + routed(20, 408, 10000) + routed(3, 15, 60000)
        self.assertEqual(profile.ced_moe_costs(events, rows=128)[0]['routed_ffn_ms'], 4)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            profile.ced_moe_costs(events + routed(1, 408, 90000), rows=128)
        events = routed(40, 6) + [dict(cat='kernel', name='_prepare_dflash_inputs_kernel', ts=90000, dur=1)] + routed(3, 6, 100000)
        self.assertEqual(profile.ced_moe_costs(events, rows=1)[0]['routed_ffn_ms'], 4)
