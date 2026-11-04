"""Lossless field decoding and malformed-metadata gates; no GPU calls."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import random
import struct
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / "experiments/2026-11-04-lossless-memory"
with patch.object(sys, "path", [str(HERE), *sys.path]):
    import fp8_results as receipts
    spec = importlib.util.spec_from_file_location("fixed_frame", HERE / "fp8_fixed_frame.py")
    frame = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = frame
    spec.loader.exec_module(frame)


class FixedFieldsTests(unittest.TestCase):
    def test_compressed_all_encodings_and_exception_chunk_boundaries(self):
        rng = random.Random(41)
        raw = bytearray(b"\x30" * frame.TILE_BYTES)
        for origin in (0, 127, 1023, 4095, 8191, frame.TILE_BYTES - 256):
            raw[origin:origin + 256] = bytes(range(256))
        for index in range(0, frame.TILE_BYTES, 7):
            raw[index] = rng.randrange(256)
        tile = frame.pack(bytes(raw))
        self.assertEqual(tile.mode, 1)
        self.assertEqual(frame.unpack(tile), raw)
        self.assertEqual(frame.pack(bytes(raw)), tile)
        self.assertEqual(tile.stored_bytes, len(tile.payload) + 9)

    def test_single_exponent_and_all_special_byte_patterns(self):
        for value in (0, 128, 127, 255, 1, 129):
            raw = bytes([value]) * frame.TILE_BYTES
            tile = frame.pack(raw)
            self.assertEqual(tile.mode, 1)
            self.assertEqual(frame.unpack(tile), raw)

    def test_unprofitable_tile_is_raw_without_unused_metadata(self):
        raw = bytes(range(256)) * 64
        tile = frame.pack(raw)
        self.assertEqual(tile, frame.Tile(0, raw))
        self.assertEqual(tile.stored_bytes, frame.TILE_BYTES + 9)

    def test_truncated_and_noncanonical_metadata_refused(self):
        tile = frame.pack(b"\x30" * (frame.TILE_BYTES - 256) + bytes(range(256)))
        bad = [replace(tile, mode=2), replace(tile, payload=tile.payload[:-1]),
               replace(tile, payload=tile.payload + b"\0"), frame.Tile(0, b""),
               replace(tile, payload=b"BAD!" + tile.payload[4:])]
        for offset in (6, 8, 10, 12, frame.OFFSETS, frame.OFFSETS + 2):
            data = bytearray(tile.payload)
            data[offset] ^= 1
            bad.append(replace(tile, payload=bytes(data)))
        data = bytearray(tile.payload)
        struct.pack_into("<H", data, 4, 0)  # repeated codebook exponents
        bad.append(replace(tile, payload=bytes(data)))
        for value in bad:
            with self.subTest(value=value.payload[:16]):
                with self.assertRaises(ValueError):
                    frame.unpack(value)

    def test_fixed_receipts_require_their_own_source_and_exactness_gate(self):
        native = {"status": "measured microbenchmark; serving remains unqualified",
                  "gpu_byte_exact": True, receipts.FIXED_EXACT_FLAG: True,
                  "gpu_consumer_checksums_exact": True,
                  "threads": 128, "model_revision": "checkpoint", "inputs": [{"sha256": "sample"}],
                  "sources_sha256": receipts.source_hashes(receipts.FIXED_SOURCES),
                  "decoded_bytes_per_replay": 128 * 1024**2,
                  "host_peak_rss_kib": 1024, "peak_cuda_allocated_bytes": 1024,
                  "peak_cuda_reserved_bytes": 1024, "kernel_ms": {"raw": [1] * 5, "encoded": [2] * 5},
                  "allocated_input_bytes": {"raw": 128 * 1024**2, "encoded": 120 * 1024**2}}
        verdict = {"passed": True, "exit": 0, "image": "image", "errors": [], "failed": []}
        options = {"lanes": 128, "image": "image", "revision": "checkpoint", "axis": "threads",
                   "exact_flag": receipts.FIXED_EXACT_FLAG, "sources": receipts.FIXED_SOURCES}
        receipts.validate(native, verdict, **options)
        bad = dict(native)
        bad.pop(receipts.FIXED_EXACT_FLAG)
        bad["gpu_all_encodings_and_long_codes_exact"] = True
        with self.assertRaises(ValueError):
            receipts.validate(bad, verdict, **options)
        bad = dict(native, sources_sha256=receipts.source_hashes())
        with self.assertRaises(ValueError):
            receipts.validate(bad, verdict, **options)
        summary = receipts.summarize([native, native], axis="threads", exact_flag=receipts.FIXED_EXACT_FLAG)
        self.assertEqual(summary["decision"], "reject this microdecoder layout")
        self.assertEqual(summary["threads"], 128)


if __name__ == "__main__":
    unittest.main()
