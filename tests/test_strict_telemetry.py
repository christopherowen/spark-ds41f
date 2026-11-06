"""Opt-in telemetry-loss proofs without SSH, GPU queries or serving operations."""
import threading
import unittest
from unittest import mock
from spark_ds41f import bench


class StrictTelemetryTest(unittest.TestCase):
    def watch(self, strict=True):
        watch = bench.MemoryWatch.__new__(bench.MemoryWatch)
        watch.lock = threading.Lock()
        watch.state = {"dgx1": {"last": 100.0}}
        watch.thermal = {"dgx1": {}}
        watch.thermal_last = {"dgx1": None}
        watch.thermal_errors = {}
        watch.strict_thermal = strict
        watch.floor_gib = 3
        watch.reasons = []
        watch.halt = watch.reasons.append
        return watch

    def sample(self):
        return bench.parse_thermal("62,25,2405,99,0,0,100,0,64900")

    def test_complete_stream_passes_without_new_queries(self):
        watch = self.watch()
        with mock.patch.object(bench.time, "monotonic", return_value=100.0):
            watch._temperature("dgx1", self.sample())
            watch.wait_ready()
            watch.check()
        self.assertEqual(watch.reasons, [])

    def test_invalid_temperature_latches_after_a_valid_sample_and_recovery(self):
        for text in ("[N/A],[N/A],2405,99", "nan,25,2405,99,0,0,100,0,64900"):
            with self.subTest(text=text), mock.patch.object(bench.time, "monotonic", return_value=100.0):
                watch = self.watch()
                watch._temperature("dgx1", self.sample())
                watch._temperature("dgx1", bench.parse_thermal(text))
                watch._temperature("dgx1", self.sample())
                with self.assertRaisesRegex(bench.BenchAbort, "thermal telemetry"):
                    watch.check()
                self.assertTrue(watch.reasons)

    def test_stale_thermal_data_is_not_masked_by_fresh_memory(self):
        watch = self.watch(); watch.thermal_last["dgx1"] = 54.0
        with mock.patch.object(bench.time, "monotonic", return_value=100.0):
            with self.assertRaisesRegex(bench.BenchAbort, "thermal telemetry stopped"):
                watch.check()
            watch._temperature("dgx1", self.sample())
            with self.assertRaisesRegex(bench.BenchAbort, "thermal telemetry stopped"):
                watch.check()

    def test_strict_readiness_requires_initial_thermal_sample(self):
        watch = self.watch()
        with self.assertRaisesRegex(bench.BenchAbort, "thermal"):
            watch.wait_ready(timeout=0)

    def test_legacy_default_does_not_add_a_new_historical_requirement(self):
        watch = self.watch(strict=False)
        with mock.patch.object(bench.time, "monotonic", return_value=100.0):
            watch._temperature("dgx1", bench.parse_thermal("[N/A],[N/A],2405,99"))
            watch.wait_ready()
            watch.check()
        self.assertFalse(watch.reasons)
        self.assertFalse(watch.thermal_errors)


if __name__ == "__main__":
    unittest.main()
