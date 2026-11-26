"""Tests for GLP serve-time alpha override, reader/bench provenance, and tool_gate scoring."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import recipe  # noqa: E402
import spark_ds41f as spark  # noqa: E402
import tool_gate  # noqa: E402
from spark_ds41f import config as spark_config  # noqa: E402


class GlpAlphaRecipeAndReaderTest(unittest.TestCase):
    def setUp(self):
        self.recipe = json.loads((ROOT / "config/cluster.json").read_text(encoding="utf-8"))

    def _resolved(self, mutate=None) -> dict:
        candidate = copy.deepcopy(self.recipe)
        if mutate is not None:
            mutate(candidate)
        return recipe.resolve(candidate, "tp4-ring")

    def test_glp_alpha_renders_env_and_reader_command(self):
        base = self._resolved()
        self.assertEqual(base["environment"]["WEIGHTLESS_STEER_ALPHA"], "2.0")
        base_cmd = spark_config.glp_reader_command(base, "/host/v.gguf")
        self.assertIn("--env", base_cmd)
        self.assertIn("WEIGHTLESS_STEER_ALPHA=2.0", base_cmd)

        unpinned = self._resolved(lambda r: r["glp"].pop("alpha", None))
        self.assertNotIn("WEIGHTLESS_STEER_ALPHA", unpinned["environment"])
        self.assertNotIn("--env", spark_config.glp_reader_command(unpinned, "/host/v.gguf"))

        dosed = self._resolved(lambda r: r["glp"].update(alpha=1.5))
        self.assertEqual(dosed["environment"]["WEIGHTLESS_STEER_ALPHA"], "1.5")
        cmd = spark_config.glp_reader_command(dosed, "/host/v.gguf")
        self.assertIn("--env", cmd)
        self.assertIn("WEIGHTLESS_STEER_ALPHA=1.5", cmd)

    def test_glp_alpha_rejects_invalid_and_hand_written_values(self):
        for bad in ("2.0", 0, -1, True, float("inf"), float("nan")):
            with self.subTest(bad=bad), self.assertRaisesRegex(SystemExit, "glp alpha"):
                self._resolved(lambda r, b=bad: r["glp"].update(alpha=b))

        def hand_written(r):
            r["environment"]["WEIGHTLESS_STEER_ALPHA"] = "2.0"

        with self.assertRaisesRegex(SystemExit, "does not set them itself"):
            self._resolved(hand_written)

    def test_glp_alpha_overlay_set_and_unset(self):
        base = self._resolved()
        dosed = recipe.overlay({"extends": {}, "set": {"glp": {"alpha": 1.5}}}, base)
        self.assertEqual(dosed["environment"]["WEIGHTLESS_STEER_ALPHA"], "1.5")
        cleared = recipe.overlay({"extends": {}, "unset": ["glp"]}, dosed)
        self.assertNotIn("WEIGHTLESS_STEER_ALPHA", cleared["environment"])

    def test_glp_generation_defaults_render_validate_and_unset(self):
        base = self._resolved()
        self.assertEqual(
            json.loads(base["environment"]["WEIGHTLESS_STEER_GENERATION_DEFAULTS"]),
            {"temperature": 0.3, "top_p": 0.95},
        )
        unset_only = recipe.overlay({"extends": {}, "unset": ["glp.generation_defaults"]}, base)
        self.assertNotIn("WEIGHTLESS_STEER_GENERATION_DEFAULTS", unset_only["environment"])

        dosed = recipe.overlay(
            {"extends": {}, "set": {"glp": {"generation_defaults": {"temperature": 0.3, "top_p": 0.95}}}},
            base,
        )
        self.assertEqual(
            dosed["environment"]["WEIGHTLESS_STEER_GENERATION_DEFAULTS"],
            '{"temperature":0.3,"top_p":0.95}',
        )
        # Stock --override-generation-config remains untouched at DeepSeek 1.0/1.0 default.
        idx = dosed["serve_args"].index("--override-generation-config") + 1
        self.assertEqual(json.loads(dosed["serve_args"][idx]), {"temperature": 1.0, "top_p": 1.0})

        cleared = recipe.overlay({"extends": {}, "unset": ["glp"]}, dosed)
        self.assertNotIn("WEIGHTLESS_STEER_GENERATION_DEFAULTS", cleared["environment"])

        for bad in (
            "0.3",
            {"temperature": -0.1},
            {"top_p": 0},
            {"top_p": 1.2},
            {"unknown": 1},
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(SystemExit, "glp generation_defaults"):
                self._resolved(lambda r, b=bad: r["glp"].update(generation_defaults=b))

        def hand_written(r):
            r["environment"]["WEIGHTLESS_STEER_GENERATION_DEFAULTS"] = '{"temperature":0.3}'

        with self.assertRaisesRegex(SystemExit, "does not set them itself"):
            self._resolved(hand_written)


class BenchIdentityGlpTest(unittest.TestCase):
    def test_bench_identity_records_glp_config_and_boot_lines(self):
        cluster = spark_config.read_cluster("config/cluster.json", "tp4-ring")
        cluster["glp"]["alpha"] = 2.0
        nodes = json.loads((ROOT / "config/examples/nodes-ring4.json").read_text(encoding="utf-8"))
        boot_line = (
            "(Worker_TP0 pid=1) INFO [glp.py:493] GLP steering on "
            "(hc_fold_pre_engram, spec_version=3): alpha=2.0 (alpha_default=3.5, source=override), "
            "layers=27 [11..37], file_sha256=24e97d41a9a4f77a, content_sha256=abcdef0123456789"
        )

        def fake_ssh(_nodes, _node, *command):
            if command[0] == "sh" and "docker logs" in command[-1]:
                return subprocess.CompletedProcess(command, 0, boot_line + "\n", "")
            if command[:2] == ("docker", "inspect"):
                return subprocess.CompletedProcess(command, 0, "sha256:1111 2026-11-22T00:00:00Z\n", "")
            return subprocess.CompletedProcess(command, 0, "a" * 40 + "\n", "")

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cluster.json"
            cfg_path.write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(spark, "run_ssh", side_effect=fake_ssh),
                mock.patch.object(spark, "read_running_image", return_value={"Config": {"Labels": {}}}),
                mock.patch.object(spark, "image_content_identity", return_value={}),
                mock.patch.object(spark, "running_image_problems", return_value=[]),
                mock.patch.object(spark, "live_doctor", return_value=[]),
                mock.patch.object(spark, "repository_revision", return_value="a" * 40),
                mock.patch.object(spark, "repository_dirty", return_value=""),
                mock.patch.object(spark, "repository_path", return_value="/deployment"),
            ):
                ident = spark.bench_identity(cluster, nodes, "config/cluster.json", cfg_path)

        self.assertIn("glp", ident)
        self.assertEqual(ident["glp"]["alpha"], 2.0)
        self.assertEqual(ident["glp"]["sha256"], cluster["glp"]["sha256"])
        self.assertEqual(ident["glp"]["boot_lines"], [boot_line])


class ToolGateScoringTest(unittest.TestCase):
    def test_scenarios_are_well_formed(self):
        scenarios = tool_gate.build_scenarios()
        self.assertGreaterEqual(len(scenarios), 20)
        ids = [s["id"] for s in scenarios]
        self.assertEqual(len(ids), len(set(ids)))

    def test_derailment_signatures_are_caught(self):
        sc = {
            "id": "t",
            "kind": "tool_call",
            "expect_names": ["get_weather"],
            "required_keys": {"get_weather": ["city"]},
        }
        # Clean pass
        ok = tool_gate.score_response(
            sc,
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}}],
                },
            },
        )
        self.assertTrue(ok["strict_pass"])
        self.assertEqual(ok["derailments"], [])

        # Vanish with finish_reason=length
        vanish = tool_gate.score_response(
            sc, {"finish_reason": "length", "message": {"content": "", "tool_calls": []}}
        )
        self.assertTrue(any("missing_tool_call" in d for d in vanish["derailments"]))

        # DSML leak in content
        leak = tool_gate.score_response(
            {"id": "p", "kind": "plain"},
            {"finish_reason": "stop", "message": {"content": "oops <｜DSML｜ invoke name=\"x\">"}},
        )
        self.assertIn("dsml_leak_in_content", leak["derailments"])

        # Malformed JSON + duplicate + empty args
        bad = tool_gate.score_response(
            sc,
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "get_weather", "arguments": "{bad"}},
                        {"function": {"name": "get_weather", "arguments": "{}"}},
                        {"function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}},
                        {"function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}},
                    ],
                },
            },
        )
        self.assertIn("malformed_json_args:get_weather", bad["derailments"])
        self.assertIn("empty_args:get_weather", bad["derailments"])
        self.assertIn("duplicate_call:get_weatherx2", bad["derailments"])

        # Plain text finish_reason != stop
        runaway = tool_gate.score_response(
            {"id": "p", "kind": "plain"},
            {"finish_reason": "length", "message": {"content": "truncated"}},
        )
        self.assertTrue(any("finish_reason='length'" in d for d in runaway["derailments"]))


if __name__ == "__main__":
    unittest.main()
