#!/usr/bin/env python3
"""Bounded, byte-exact CPU reference codecs for independent memory blocks.

These are screening codecs, not serving kernels. CPU timings do not establish
GPU throughput. No floating-point interpretation or rounding is performed.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import heapq
from pathlib import Path
import struct
from collections import Counter

MAX_BLOCK = 16 * 1024 * 1024
HEADER = struct.Struct("<4sBBHII32s")
MODES = {"raw": 0, "lz4": 1, "bitplane-lz4": 2,
         "fp8-fields-lz4": 3, "fp8-exponent-huffman": 4}


class LZ4:
    def __init__(self, path: str | None = None):
        self.path = path or ctypes.util.find_library("lz4")
        if not self.path:
            raise RuntimeError("liblz4 is required; no substitute codec is selected")
        self.lib = ctypes.CDLL(self.path)
        binary = Path(self.path)
        if not binary.is_file() and Path("/proc/self/maps").is_file():
            candidates = [line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                          if "/liblz4.so" in line]
            if candidates:
                binary = Path(candidates[0])
        self.binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest() if binary.is_file() else None
        self.lib.LZ4_versionNumber.restype = ctypes.c_int
        self.version = self.lib.LZ4_versionNumber()
        self.lib.LZ4_compressBound.argtypes = [ctypes.c_int]
        self.lib.LZ4_compressBound.restype = ctypes.c_int
        self.lib.LZ4_compress_default.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                                                ctypes.c_int, ctypes.c_int]
        self.lib.LZ4_compress_default.restype = ctypes.c_int
        self.lib.LZ4_decompress_safe.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                                               ctypes.c_int, ctypes.c_int]
        self.lib.LZ4_decompress_safe.restype = ctypes.c_int

    def compress(self, raw: bytes) -> bytes:
        if len(raw) > MAX_BLOCK:
            raise ValueError("LZ4 input exceeds the bounded block contract")
        bound = self.lib.LZ4_compressBound(len(raw))
        dst = ctypes.create_string_buffer(bound)
        n = self.lib.LZ4_compress_default(raw, dst, len(raw), bound)
        if n <= 0:
            raise ValueError("LZ4 compression failed")
        return dst.raw[:n]

    def decompress(self, payload: bytes, size: int) -> bytes:
        if not 0 <= size <= MAX_BLOCK or len(payload) > MAX_BLOCK + MAX_BLOCK // 255 + 32:
            raise ValueError("LZ4 block exceeds the bounded contract")
        dst = ctypes.create_string_buffer(max(size, 1))
        n = self.lib.LZ4_decompress_safe(payload, dst, len(payload), size)
        if n != size:
            raise ValueError(f"LZ4 decoded {n} bytes, expected {size}")
        return dst.raw[:n]


def pack_nibbles(values: bytes) -> bytes:
    return bytes(values[i] | ((values[i + 1] if i + 1 < len(values) else 0) << 4)
                 for i in range(0, len(values), 2))


def unpack_nibbles(raw: bytes, size: int) -> bytes:
    if len(raw) != (size + 1) // 2 or (size % 2 and raw[-1] >> 4):
        raise ValueError("invalid nibble payload or padding")
    return bytes((raw[i // 2] >> (4 * (i % 2))) & 15 for i in range(size))


def split_fp8(raw: bytes) -> bytes:
    """Exact E4M3 field permutation, including signed zero and special encodings."""
    low = bytes((v & 7) | ((v >> 4) & 8) for v in raw)
    exponent = bytes((v >> 3) & 15 for v in raw)
    return pack_nibbles(low) + pack_nibbles(exponent)


def join_fp8(raw: bytes, size: int) -> bytes:
    half = (size + 1) // 2
    if len(raw) != 2 * half:
        raise ValueError("invalid FP8 field length")
    low = unpack_nibbles(raw[:half], size)
    exp = unpack_nibbles(raw[half:], size)
    return bytes((a & 7) | ((a & 8) << 4) | (b << 3) for a, b in zip(low, exp))


def bitplanes(raw: bytes) -> bytes:
    stride = (len(raw) + 7) // 8
    out = bytearray(8 * stride)
    for i, value in enumerate(raw):
        for bit in range(8):
            out[bit * stride + i // 8] |= ((value >> bit) & 1) << (i % 8)
    return bytes(out)


def unbitplanes(raw: bytes, size: int) -> bytes:
    stride = (size + 7) // 8
    if len(raw) != 8 * stride:
        raise ValueError("invalid bitplane length")
    if size % 8 and any(raw[(b + 1) * stride - 1] >> (size % 8) for b in range(8)):
        raise ValueError("nonzero bitplane padding")
    return bytes(sum(((raw[b * stride + i // 8] >> (i % 8)) & 1) << b
                     for b in range(8)) for i in range(size))


def canonical_codes(lengths: bytes) -> dict[int, tuple[int, int]]:
    if len(lengths) != 16 or any(n > 15 for n in lengths):
        raise ValueError("invalid Huffman lengths")
    pairs = sorted((n, symbol) for symbol, n in enumerate(lengths) if n)
    code, previous = 0, 0
    result = {}
    for n, symbol in pairs:
        code <<= n - previous
        if code >= 1 << n:
            raise ValueError("oversubscribed Huffman tree")
        result[symbol] = code, n
        code += 1
        previous = n
    return result


def huffman_fp8(raw: bytes) -> bytes:
    counts = Counter((v >> 3) & 15 for v in raw)
    heap = [(count, symbol, (symbol,)) for symbol, count in counts.items()]
    heapq.heapify(heap)
    lengths = bytearray(16)
    serial = 16
    while len(heap) > 1:
        ca, _, a = heapq.heappop(heap)
        cb, _, b = heapq.heappop(heap)
        for symbol in a + b:
            lengths[symbol] += 1
        heapq.heappush(heap, (ca + cb, serial, a + b))
        serial += 1
    if heap and not any(lengths):
        lengths[heap[0][2][0]] = 1
    codes = canonical_codes(bytes(lengths))
    output, acc, bits = bytearray(), 0, 0
    for v in raw:
        code, length = codes[(v >> 3) & 15]
        acc = (acc << length) | code
        bits += length
        while bits >= 8:
            bits -= 8
            output.append((acc >> bits) & 255)
            acc &= (1 << bits) - 1
    if bits:
        output.append(acc << (8 - bits))
    low = bytes((v & 7) | ((v >> 4) & 8) for v in raw)
    return bytes(lengths) + pack_nibbles(low) + output


def unhuffman_fp8(payload: bytes, size: int) -> bytes:
    half = (size + 1) // 2
    if len(payload) < 16 + half:
        raise ValueError("truncated Huffman fields")
    codes = canonical_codes(payload[:16])
    low = unpack_nibbles(payload[16:16 + half], size)
    table = {(n, code): symbol for symbol, (code, n) in codes.items()}
    stream = payload[16 + half:]
    out, code, width, used = bytearray(), 0, 0, 0
    for byte in stream:
        for bit in range(7, -1, -1):
            if len(out) == size:
                break
            code = (code << 1) | ((byte >> bit) & 1)
            width += 1
            used += 1
            if width > 15:
                raise ValueError("invalid Huffman code")
            if (width, code) in table:
                a = low[len(out)]
                out.append((a & 7) | ((a & 8) << 4) | (table[width, code] << 3))
                code, width = 0, 0
    if len(out) != size or len(stream) != (used + 7) // 8:
        raise ValueError("incomplete Huffman block or trailing bytes")
    if used % 8 and stream[-1] & ((1 << (8 - used % 8)) - 1):
        raise ValueError("nonzero Huffman padding")
    return bytes(out)


def encode(raw: bytes, mode: str, lz4: LZ4 | None = None) -> bytes:
    if len(raw) > MAX_BLOCK or mode not in MODES:
        raise ValueError("unsupported mode or oversized block")
    selected = MODES[mode]
    if mode == "raw":
        payload = raw
    elif mode == "fp8-exponent-huffman":
        payload = huffman_fp8(raw)
    else:
        if lz4 is None:
            raise RuntimeError("the requested codec requires liblz4")
        source = bitplanes(raw) if mode == "bitplane-lz4" else (
            split_fp8(raw) if mode == "fp8-fields-lz4" else raw)
        payload = lz4.compress(source)
    # An explicit RAW representation is part of the format, not an error fallback.
    if len(payload) >= len(raw):
        selected, payload = MODES["raw"], raw
    result = HEADER.pack(b"LMB1", 1, selected, 0, len(raw), len(payload),
                         hashlib.sha256(raw).digest()) + payload
    if decode(result, lz4) != raw:
        raise ValueError("encoded block failed its byte-exact round trip")
    return result


def decode(frame: bytes, lz4: LZ4 | None = None) -> bytes:
    if len(frame) < HEADER.size:
        raise ValueError("truncated block header")
    magic, version, mode, reserved, size, count, digest = HEADER.unpack_from(frame)
    if magic != b"LMB1" or version != 1 or reserved or mode not in MODES.values():
        raise ValueError("unsupported block header")
    if size > MAX_BLOCK or count > size or count != len(frame) - HEADER.size:
        raise ValueError("block length exceeds its bounded contract")
    payload = frame[HEADER.size:]
    if mode == 0:
        raw = payload
    elif mode == 4:
        raw = unhuffman_fp8(payload, size)
    else:
        if lz4 is None:
            raise RuntimeError("the encoded block requires liblz4")
        transformed = (8 * ((size + 7) // 8) if mode == 2 else
                       2 * ((size + 1) // 2) if mode == 3 else size)
        data = lz4.decompress(payload, transformed)
        raw = unbitplanes(data, size) if mode == 2 else join_fp8(data, size) if mode == 3 else data
    if len(raw) != size or hashlib.sha256(raw).digest() != digest:
        raise ValueError("decoded bytes failed integrity verification")
    return raw
