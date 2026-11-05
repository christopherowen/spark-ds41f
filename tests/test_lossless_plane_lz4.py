"""Exact bit coordinates and structural bounds before any CUDA measurement."""

from pathlib import Path
import random
import sys
import unittest

HERE = Path(__file__).resolve().parents[1] / "experiments/2026-11-04-lossless-memory"
sys.path.insert(0, str(HERE))
import fp8_plane_lz4_frame as frame
from memory_codec import LZ4


class PlaneLz4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.lz4 = LZ4()
        except (RuntimeError, OSError):
            cls.lz4 = None

    def test_all_encodings_and_raw_fallback_are_exact(self):
        if self.lz4 is None:
            self.skipTest("liblz4 is not installed")
        for raw in (bytes(range(256)) * 64, bytes(frame.TILE_BYTES),
                    random.Random(17).randbytes(frame.TILE_BYTES)):
            for chunk in (256, 512, 1024, 2048):
                tile = frame.pack(raw, self.lz4, chunk)
                self.assertEqual(frame.unpack(tile, self.lz4), raw)
                self.assertEqual(tile, frame.pack(raw, self.lz4, chunk))
                self.assertEqual(tile.stored_bytes, len(tile.payload) + 9)
        self.assertEqual(tile.mode, 0)

    def test_each_bit_coordinate_transposes_and_the_operation_is_its_inverse(self):
        for plane in range(8):
            for value in range(8):
                word = 1 << (8 * plane + value)
                self.assertEqual(frame.transpose8(word), 1 << (8 * value + plane))
        generator = random.Random(41)
        for _ in range(20):
            word = generator.getrandbits(64)
            self.assertEqual(frame.transpose8(frame.transpose8(word)), word)

    def test_header_extents_and_padding_cannot_be_silently_accepted(self):
        if self.lz4 is None:
            self.skipTest("liblz4 is not installed")
        tile = frame.pack(bytes(range(256)) * 64, self.lz4)
        self.assertEqual(tile.mode, 1)
        mutations = []
        for offset in (0, 4, 6, 8, 12, frame.HEADER.size):
            payload = bytearray(tile.payload)
            payload[offset] ^= 1
            mutations.append(bytes(payload))
        mutations.extend([tile.payload[:-1], tile.payload + bytes(16)])
        offset, flags = frame.ENTRY.unpack_from(tile.payload, frame.HEADER.size)
        end = offset + (flags & ~frame.RAW)
        if end % 16:
            payload = bytearray(tile.payload)
            payload[end] = 1
            mutations.append(bytes(payload))
        for payload in mutations:
            with self.assertRaises(ValueError):
                frame.unpack(frame.Tile(1, payload), self.lz4)


if __name__ == "__main__":
    unittest.main()
