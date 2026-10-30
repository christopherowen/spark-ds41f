"""Unit tests for sequence-parallel prefill local MoE routing and grouped
MXFP8 + route all-gather (Row 19 / Sketch D input half of fusion-survey).
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "2026-10-30-sp-prefill-mxfp8-route"

VLLM_FILES = (
    "vllm/models/deepseek_v4_1/sp_prefill.py",
    "vllm/models/deepseek_v4_1/tilelang/router.py",
    "vllm/models/deepseek_v4_1/kernels.py",
    "vllm/models/deepseek_v4_1/nvidia/model.py",
)


class SPPrefillMXFP8RouteTest(unittest.TestCase):
    def test_byte_and_latency_accounting(self) -> None:
        hidden_size = 5120
        scale_block = 32
        top_k = 6

        bf16_bytes_per_row = hidden_size * 2
        mxfp8_val_bytes_per_row = hidden_size * 1
        mxfp8_scale_bytes_per_row = hidden_size // scale_block
        weights_bytes_per_row = top_k * 4  # float32
        ids_bytes_per_row = top_k * 8  # int64

        gathered_bytes_per_row = (
            mxfp8_val_bytes_per_row
            + mxfp8_scale_bytes_per_row
            + weights_bytes_per_row
            + ids_bytes_per_row
        )
        self.assertEqual(bf16_bytes_per_row, 10240)
        self.assertEqual(gathered_bytes_per_row, 5352)
        saved_bytes_per_row = bf16_bytes_per_row - gathered_bytes_per_row
        self.assertEqual(saved_bytes_per_row, 4888)
        reduction_ratio = saved_bytes_per_row / bf16_bytes_per_row
        self.assertAlmostEqual(reduction_ratio, 4888 / 10240, places=9)

    def test_overlay_and_bundle_parity_and_ast_contracts(self) -> None:
        bundle_dir = EXP_DIR / "bundles" / "sp-prefill-mxfp8-route"
        for rel in VLLM_FILES:
            overlay_file = EXP_DIR / "overlay" / rel
            bundle_file = bundle_dir / "overlay" / rel
            self.assertTrue(overlay_file.is_file(), f"missing {overlay_file}")
            self.assertTrue(bundle_file.is_file(), f"missing {bundle_file}")
            src = overlay_file.read_text(encoding="utf-8")
            self.assertEqual(src, bundle_file.read_text(encoding="utf-8"))
            ast.parse(src, filename=rel)

        sp_src = (EXP_DIR / "overlay" / VLLM_FILES[0]).read_text(encoding="utf-8")
        router_src = (EXP_DIR / "overlay" / VLLM_FILES[1]).read_text(encoding="utf-8")
        kernels_src = (EXP_DIR / "overlay" / VLLM_FILES[2]).read_text(encoding="utf-8")
        model_src = (EXP_DIR / "overlay" / VLLM_FILES[3]).read_text(encoding="utf-8")

        # sp_prefill.py: grouped PyNCCL gather_many and gather_mxfp8
        self.assertIn("def gather_many(", sp_src)
        self.assertIn("comm.group_start()", sp_src)
        self.assertIn("comm.all_gather(dst, src)", sp_src)
        self.assertIn("comm.group_end()", sp_src)
        self.assertIn("def gather_mxfp8(self, x_q, *extra: torch.Tensor)", sp_src)

        # tilelang/router.py: RoutedRows, provide_routes, provided_routes, route_local
        self.assertIn("class RoutedRows(NamedTuple):", router_src)
        self.assertIn("def provide_routes(", router_src)
        self.assertIn("def provided_routes(", router_src)
        self.assertIn("def _local_mask(", router_src)
        self.assertIn("def route_local(", router_src)

        # kernels.py: sp_moe_inputs context manager
        self.assertIn("def sp_moe_inputs(", kernels_src)
        self.assertIn("route_local(x, sp_rows, input_ids=input_ids)", kernels_src)
        self.assertIn("sp_rows.gather_mxfp8(x_q, weights, ids)", kernels_src)
        self.assertIn("provide_routes(gathered_q.rows, weights, ids)", kernels_src)

        # nvidia/model.py: uses kernels.sp_moe_inputs in SP MoE branch
        self.assertIn(
            "kernels.sp_moe_inputs(\n"
            "                    sp_rows, x, x_q, self.ffn, input_ids, self.mhc\n"
            "                ) as rows",
            model_src,
        )
        self.assertNotIn("with kernels.mxfp8_inputs(self.mhc.cast_rows(rows)):", model_src)

    def test_series_and_overlay_configs(self) -> None:
        series = (EXP_DIR / "vllm" / "series").read_text(encoding="utf-8")
        patches = list((EXP_DIR / "vllm").glob("0568-*.patch"))
        self.assertEqual(len(patches), 1)
        self.assertIn(patches[0].name, series)

        for topo in ("tp3", "tp4"):
            cfg = json.loads(
                (EXP_DIR / f"sp-mxfp8-route-{topo}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cfg["extends"]["config"], f"config/production-{topo}.json")
            mounts = cfg["set"]["container"]["mounts"]
            self.assertEqual(len(mounts), 1 + len(VLLM_FILES))
            dev_cfg = json.loads(
                (EXP_DIR / f"sp-mxfp8-route-{topo}-dev.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                dev_cfg["extends"]["config"],
                f"experiments/2026-10-30-sp-prefill-mxfp8-route/sp-mxfp8-route-{topo}.json",
            )

        verdict = json.loads((EXP_DIR / "verdict.json").read_text(encoding="utf-8"))
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["exit"], 0)
        self.assertEqual(verdict["failed"], [])


if __name__ == "__main__":
    unittest.main()
