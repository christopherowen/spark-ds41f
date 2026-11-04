"""Complete-byte cache checks, including special encodings and corrupted scales."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys
import os
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "experiments/2026-11-04-lossless-memory/engram/verify_cache.py"
spec = importlib.util.spec_from_file_location("verify_engram_cache", PATH)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class CacheVerificationTests(unittest.TestCase):
    def setUp(self):
        memory = patch.object(verify, "available_bytes", return_value=8 * 1024**3)
        memory.start()
        self.addCleanup(memory.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.values = [bytes((value + row) % 256 for value in range(256)) for row in range(4099)]
        self.scales = [row.to_bytes(8, "little") for row in range(4099)]
        w, s = self.root / "weight", self.root / "scale"
        w.write_bytes(b"prefix1" + b"".join(self.values))
        s.write_bytes(b"prefix2" + b"".join(self.scales))
        self.weight, self.scale = (w, 7), (s, 7)
        self.packed = self.root / "packed"
        self.starts, self.lengths = [0, 128, 4000], [17, 125, 99]
        self.expected = b"".join(self.values[row] + self.scales[row]
                                 for start, count in zip(self.starts, self.lengths)
                                 for row in range(start, start + count))
        self.packed.write_bytes(self.expected)

    def check(self):
        return verify.verify_rows(self.weight, self.scale, self.packed,
                                  self.starts, self.lengths, chunk_rows=7)

    def test_all_bytes_and_uneven_owned_ranges_cross_chunk_and_page_boundaries(self):
        result = self.check()
        self.assertTrue(result["all_byte_exact"])
        self.assertEqual(result["verified_bytes"], len(self.expected))
        self.assertEqual(result["packed_sha256"], hashlib.sha256(self.expected).hexdigest())

    def test_value_and_scale_corruption_are_refused_without_float_conversion(self):
        for offset in (127, 256, len(self.expected) - 1):
            data = bytearray(self.expected)
            data[offset] ^= 1
            self.packed.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "packed bytes differ"):
                self.check()

    def test_standard_library_path_checks_the_same_complete_bytes(self):
        with patch.dict("sys.modules", {"numpy": None}):
            result = self.check()
        self.assertEqual(result["packed_sha256"], hashlib.sha256(self.expected).hexdigest())

    def test_trailing_data_and_short_source_are_refused(self):
        self.packed.write_bytes(self.expected + b"\0")
        with self.assertRaisesRegex(ValueError, "wrong length"):
            self.check()
        self.packed.write_bytes(self.expected)
        self.scale[0].write_bytes(b"prefix2")
        with self.assertRaisesRegex(ValueError, "short source/cache read"):
            self.check()

    def test_host_memory_floor_stops_before_allocating_another_chunk(self):
        with patch.object(verify, "available_bytes", return_value=3 * 1024**3):
            with self.assertRaisesRegex(ValueError, "4 GiB host memory floor"):
                self.check()

    def test_live_wrapper_refuses_foreign_or_missing_window_before_any_node_call(self):
        path = PATH.with_name("check_live_cache.py")
        spec = importlib.util.spec_from_file_location("check_live_engram_cache", path)
        live = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(live)
        argv = [str(path), "http://unused", "--config", "unused", "--run", "fixture", "--label", "fixture"]
        for hold in (None, {"holder": "foreign", "production_config": live.RESTORE}):
            with patch.object(sys, "argv", argv), patch.object(live.lab, "read_hold", return_value=hold), \
                    patch.object(live.lab, "requests_in_flight", side_effect=AssertionError("unexpected node call")):
                with self.assertRaisesRegex(SystemExit, "no node operation"):
                    live.main()

    def test_reader_file_identity_survives_different_mount_path_spellings(self):
        alias = self.root / "another-mount-name"
        os.link(self.packed, alias)
        proc = self.root / "proc"
        fd = proc / "999999/fd/7"
        fd.parent.mkdir(parents=True)
        fd.symlink_to(alias)
        with patch.object(verify, "Path", return_value=proc):
            owners = verify.open_by_serving_process([self.packed])
        self.assertEqual(owners[str(self.packed)][0]["observed_path"], str(alias))

    def test_reader_holding_a_replaced_file_does_not_qualify_the_new_cache(self):
        alias = self.root / "old-open-file"
        os.link(self.packed, alias)
        proc = self.root / "proc"
        fd = proc / "999999/fd/7"
        fd.parent.mkdir(parents=True)
        fd.symlink_to(alias)
        self.packed.unlink()
        self.packed.write_bytes(self.expected)
        with patch.object(verify, "Path", return_value=proc):
            with self.assertRaisesRegex(ValueError, "does not hold"):
                verify.open_by_serving_process([self.packed])


if __name__ == "__main__":
    unittest.main()
