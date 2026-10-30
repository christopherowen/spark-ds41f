"""Unit tests for the per-type L2 weight prefetch schedule (owner COMP/WO and
Engram ENGRAM/NEXT windows) ported onto spark-ds41f:1030.
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "2026-10-30-l2-prefetch-pertype"

VLLM_FILES = (
    "vllm/models/deepseek_v4_1/attention.py",
    "vllm/models/deepseek_v4_1/common/engram.py",
    "vllm/models/deepseek_v4_1/l2_prefetch.py",
    "vllm/models/deepseek_v4_1/nvidia/model.py",
)


class L2PrefetchPertypeTest(unittest.TestCase):
    def test_overlay_and_bundle_parity_and_ast_contracts(self) -> None:
        bundle_dir = EXP_DIR / "bundles" / "l2-prefetch-pertype"
        for rel in VLLM_FILES:
            overlay_file = EXP_DIR / "overlay" / rel
            bundle_file = bundle_dir / "overlay" / rel
            self.assertTrue(overlay_file.is_file(), f"missing {overlay_file}")
            self.assertTrue(bundle_file.is_file(), f"missing {bundle_file}")
            src = overlay_file.read_text(encoding="utf-8")
            self.assertEqual(src, bundle_file.read_text(encoding="utf-8"))
            ast.parse(src, filename=rel)

        attn_src = (EXP_DIR / "overlay" / VLLM_FILES[0]).read_text(encoding="utf-8")
        engram_src = (EXP_DIR / "overlay" / VLLM_FILES[1]).read_text(encoding="utf-8")
        l2pf_src = (EXP_DIR / "overlay" / VLLM_FILES[2]).read_text(encoding="utf-8")
        model_src = (EXP_DIR / "overlay" / VLLM_FILES[3]).read_text(encoding="utf-8")

        # attention.py: preserves 1030's fused _front and adds _issue_wo_prefetch
        self.assertIn("front = self._front(qr_kv, positions, sp_rows)", attn_src)
        self.assertIn("def _issue_wo_prefetch(self, rows, *, after):", attn_src)
        self.assertIn('self._issue_wo_prefetch(rows, after="compressor")', attn_src)
        self.assertIn('self._issue_wo_prefetch(rows, after="q-b")', attn_src)

        # common/engram.py: project_rows issues after_projection between local_product and all_gather
        self.assertIn("def project_rows(wkv, rows, after_projection=None):", engram_src)
        self.assertIn("self.l2_prefetch_after_projection = None", engram_src)
        self.assertIn(
            "project_rows(self.wkv, rows, self.l2_prefetch_after_projection)",
            engram_src,
        )

        # l2_prefetch.py & nvidia/model.py: COMP and ENGRAM windows
        self.assertIn("COMP  an owner layer's compressor projection", l2pf_src)
        self.assertIn("ENGRAM  before the MoE all-reduce ahead of an Engram layer", l2pf_src)
        self.assertIn('attn._l2pf_comp = l2_prefetch.make_plan(comp, l2_prefetch.BUDGET_WO, device)', model_src)
        self.assertIn('l2_prefetch.linear_segments("engram.wkv", engram.wkv)', model_src)
        self.assertIn("engram.l2_prefetch_after_projection = (", model_src)
        self.assertIn("hooked = plan_engram if plan_engram is not None else plan_next", model_src)

    def test_series_and_overlay_configs(self) -> None:
        series = (EXP_DIR / "vllm" / "series").read_text(encoding="utf-8")
        patches = list((EXP_DIR / "vllm").glob("0568-*.patch"))
        self.assertEqual(len(patches), 1)
        self.assertIn(patches[0].name, series)

        for topo in ("tp3", "tp4"):
            cfg = json.loads(
                (EXP_DIR / f"l2pf-pertype-{topo}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cfg["extends"]["config"], f"config/production-{topo}.json")
            mounts = cfg["set"]["container"]["mounts"]
            self.assertEqual(len(mounts), 1 + len(VLLM_FILES))
            dev_cfg = json.loads(
                (EXP_DIR / f"l2pf-pertype-{topo}-dev.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                dev_cfg["extends"]["config"],
                f"experiments/2026-10-30-l2-prefetch-pertype/l2pf-pertype-{topo}.json",
            )

        verdict = json.loads((EXP_DIR / "verdict.json").read_text(encoding="utf-8"))
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["exit"], 0)
        self.assertEqual(verdict["failed"], [])


if __name__ == "__main__":
    unittest.main()
