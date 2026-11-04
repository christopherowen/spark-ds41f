"""Exact-byte, malformed-input and publication tests for the memory screens."""

import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import struct
import tempfile
import unittest
from unittest import mock

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments/2026-11-04-lossless-memory"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codec = load("memory_codec")
store = load("cold_store")
collector = load("collect_weights")


class CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.lz4 = codec.LZ4()
        except RuntimeError:
            cls.lz4 = None

    def test_fp8_fields_preserve_every_encoding_and_odd_lengths(self):
        raw = bytes(range(256)) * 2
        for n in (0, 1, 3, 7, 8, 9, 31, 255, 256, 511, 512):
            with self.subTest(n=n):
                self.assertEqual(codec.join_fp8(codec.split_fp8(raw[:n]), n), raw[:n])

    def test_bitplanes_preserve_every_encoding_and_tail(self):
        raw = bytes(range(256))
        for n in (0, 1, 3, 7, 8, 9, 31, 255, 256):
            self.assertEqual(codec.unbitplanes(codec.bitplanes(raw[:n]), n), raw[:n])

    def test_transforms_refuse_nonzero_padding(self):
        with self.assertRaises(ValueError):
            codec.unpack_nibbles(b"\xf1", 1)
        with self.assertRaises(ValueError):
            codec.unbitplanes(b"\x80" * 8, 1)

    def test_huffman_preserves_all_exponents_and_special_encodings(self):
        raw = bytes(range(256)) + b"\x80\x00\x7f\xff" * 1024
        self.assertEqual(codec.unhuffman_fp8(codec.huffman_fp8(raw), len(raw)), raw)

    def test_huffman_single_symbol_and_short_blocks(self):
        for raw in (b"", b"\x80", b"\xff" * 17, b"\x00" * 1024):
            self.assertEqual(codec.unhuffman_fp8(codec.huffman_fp8(raw), len(raw)), raw)

    def test_huffman_rejects_oversubscribed_tree_and_trailing_data(self):
        with self.assertRaises(ValueError):
            codec.canonical_codes(bytes([1] * 16))
        raw = b"\x30" * 64
        with self.assertRaises(ValueError):
            codec.unhuffman_fp8(codec.huffman_fp8(raw) + b"\0", len(raw))

    def test_all_modes_round_trip_adversarial_bytes_deterministically(self):
        rng = random.Random(9841)
        cases = (b"", bytes(range(256)) * 4, b"\xff" * 4096,
                 rng.randbytes(4093), bytes(rng.choice([0, 48, 56, 128]) for _ in range(4093)))
        for mode in codec.MODES:
            if "lz4" in mode and not self.lz4:
                continue
            for raw in cases:
                with self.subTest(mode=mode, size=len(raw)):
                    frame = codec.encode(raw, mode, self.lz4)
                    self.assertEqual(codec.decode(frame, self.lz4), raw)
                    self.assertEqual(frame, codec.encode(raw, mode, self.lz4))

    def test_native_lz4_is_exercised(self):
        if not self.lz4:
            self.skipTest("liblz4 is not installed")
        raw = b"abcdef" * 1024
        frame = codec.encode(raw, "lz4", self.lz4)
        self.assertEqual(codec.HEADER.unpack_from(frame)[2], codec.MODES["lz4"])
        self.assertLess(len(frame), len(raw))

    def test_each_compressed_representation_is_exercised(self):
        for mode in codec.MODES:
            if mode == "raw" or ("lz4" in mode and not self.lz4):
                continue
            frame = codec.encode(b"\x30" * 4096, mode, self.lz4)
            self.assertEqual(codec.HEADER.unpack_from(frame)[2], codec.MODES[mode])

    def test_raw_representation_is_explicit_when_coding_does_not_pay(self):
        for mode in codec.MODES:
            if "lz4" in mode and not self.lz4:
                continue
            frame = codec.encode(b"\x7f", mode, self.lz4)
            self.assertEqual(codec.HEADER.unpack_from(frame)[2], 0)

    def test_corruption_and_truncation_fail_before_returning_bytes(self):
        frame = codec.encode(b"hello" * 500, "fp8-exponent-huffman")
        for bad in (frame[:-1], frame + b"\0", frame[:4],
                    frame[:-1] + bytes([frame[-1] ^ 1]), b"BAD!" + frame[4:]):
            with self.assertRaises(ValueError):
                codec.decode(bad)

    def test_oversized_header_fails_before_allocation(self):
        frame = codec.HEADER.pack(b"LMB1", 1, 0, 0, codec.MAX_BLOCK + 1, 0, b"\0" * 32)
        with self.assertRaises(ValueError):
            codec.decode(frame)

    def test_missing_decoder_is_an_error(self):
        with self.assertRaises(RuntimeError):
            codec.encode(b"hello", "lz4")


IDENTITY = {"model_revision": "checkpoint-a", "tokenizer_revision": "tokenizer-a",
            "vllm_tree": "tree-a", "tp_size": 3, "rank": 0,
            "cache_layout": "grouped-cache-v1", "steering_digest": "stock"}
BOUNDARY = {"num_tokens": 1024, "draft_prefix_len": 1023,
            "groups": [0, 1, 2], "auxiliary_state": ["target", "draft"],
            "capture_fence": "synchronized_written_regions", "complete": True}
PREFIX = hashlib.sha256(b"the exact token and multimodal prefix identity").hexdigest()


class ColdStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cache = store.ColdStore(self.root, IDENTITY, quota_bytes=131072,
                                    max_checkpoint_bytes=65536)
        self.records = [({"name": "target/group0/page0", "dtype": "uint8",
                          "logical_page": 0}, bytes(range(256)) * 4),
                        ({"name": "draft/auxiliary", "dtype": "uint8"}, b"\xff" * 512)]

    def tearDown(self):
        self.temporary.cleanup()

    def put(self, **kwargs):
        return self.cache.put(PREFIX, self.records, boundary=BOUNDARY,
                              mode="fp8-exponent-huffman", **kwargs)

    def test_checkpoint_preserves_every_record_descriptor_and_boundary(self):
        result = self.put()
        boundary, records = self.cache.get(PREFIX)
        self.assertEqual(records, self.records)
        self.assertEqual(boundary, BOUNDARY)
        self.assertFalse(result["published_to_vllm"])

    def test_model_rank_layout_and_steering_have_separate_namespaces(self):
        self.put()
        for field, value in (("rank", 1), ("model_revision", "checkpoint-b"),
                             ("cache_layout", "different"), ("steering_digest", "steered")):
            identity = {**IDENTITY, field: value}
            other = store.ColdStore(self.root, identity, quota_bytes=131072,
                                    max_checkpoint_bytes=65536)
            with self.assertRaises(FileNotFoundError):
                other.get(PREFIX)

    def test_incomplete_or_unfenced_checkpoint_is_refused(self):
        for changed in ({"complete": False}, {"capture_fence": "none"},
                        {"draft_prefix_len": 1025}):
            with self.assertRaises(ValueError):
                self.cache.put(PREFIX, self.records, boundary={**BOUNDARY, **changed}, mode="raw")

    def test_corrupted_metadata_payload_and_trailing_bytes_are_refused(self):
        self.put()
        path = self.cache.path(PREFIX)
        original = path.read_bytes()
        for pos in (store.FILE_HEADER.size + 1, len(original) - 1):
            bad = bytearray(original)
            bad[pos] ^= 1
            path.write_bytes(bad)
            with self.assertRaises(ValueError):
                self.cache.get(PREFIX)
        path.write_bytes(original + b"\0")
        with self.assertRaises(ValueError):
            self.cache.get(PREFIX)

    def test_failed_publication_keeps_previous_checkpoint_and_cleans_temporary(self):
        self.put()
        original = self.cache.path(PREFIX).read_bytes()
        with mock.patch.object(store.os, "replace", side_effect=OSError("injected publication failure")):
            with self.assertRaises(OSError):
                self.put()
        self.assertEqual(self.cache.path(PREFIX).read_bytes(), original)
        self.assertEqual(list(self.root.glob(".checkpoint-*")), [])

    def test_quota_is_checked_before_publication(self):
        cache = store.ColdStore(self.root, IDENTITY, quota_bytes=6000, max_checkpoint_bytes=5000)
        first = cache.put(PREFIX, self.records, boundary=BOUNDARY, mode="raw")
        self.assertGreater(first["stored_bytes"], 1500)
        key = hashlib.sha256(b"another prefix").hexdigest()
        cache.put(key, self.records, boundary=BOUNDARY, mode="raw")
        key2 = hashlib.sha256(b"third prefix").hexdigest()
        with self.assertRaisesRegex(ValueError, "quota"):
            cache.put(key2, self.records, boundary=BOUNDARY, mode="raw")
        self.assertFalse(cache.path(key2).exists())

    def test_path_traversal_and_duplicate_records_are_refused(self):
        with self.assertRaises(ValueError):
            self.cache.path("../not-a-digest")
        with self.assertRaises(ValueError):
            self.cache.put(PREFIX, self.records * 2, boundary=BOUNDARY, mode="raw")


class CollectorTests(unittest.TestCase):
    def test_rectangular_tiles_are_not_contiguous_file_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "revision"
            snapshot.mkdir()
            # Each row has a different value; taking 16 KiB contiguously would
            # read only half the intended rows and fail this assertion.
            data = b"".join(bytes([row % 256]) * 256 for row in range(256))
            header = json.dumps({"layers.0.attn.wq_b.weight": {
                "dtype": "F8_E4M3", "shape": [256, 256],
                "data_offsets": [0, len(data)]}}).encode()
            (snapshot / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + data)
            result = collector.collect(snapshot, "revision", 1, 2)
            import base64
            first = base64.b64decode(result["samples"][0]["data_base64"])
            self.assertEqual(first, b"".join(bytes([row]) * 128 for row in range(128)))
            self.assertEqual(result["sample_bytes"], 32768)
            self.assertEqual(result["samples"][0]["family"], "wq_b")

    def test_wrong_revision_is_refused_before_reading(self):
        with self.assertRaises(ValueError):
            collector.collect(Path("wrong-revision"), "expected-revision", 1, 1)


if __name__ == "__main__":
    unittest.main()
