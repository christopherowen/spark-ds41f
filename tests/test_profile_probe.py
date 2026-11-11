"""CED attribution in profiler traces must refuse unfamiliar geometry and partial groups."""
import importlib.util
from pathlib import Path
import unittest

_spec = importlib.util.spec_from_file_location("ced_profile", Path(__file__).resolve().parents[1] / "scripts/ced_profile.py")
profile = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(profile)


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


if __name__ == "__main__":
    unittest.main()
