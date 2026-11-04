"""Exact fixed-field FP8 tiles with bounded exception chunks; probe ABI only.

Split the original byte into four sign/mantissa bits and four exponent bits.
Two-bit tags name the three most frequent exponents or an unchanged four-bit
exception. No floating-point operation is used. Integrity is checked against
the original checkpoint before upload; this is not a persisted storage format.
"""

from dataclasses import dataclass
import struct

from memory_codec import pack_nibbles, unpack_nibbles

TILE_BYTES = 16384
CHUNK = 128
HEADER = struct.Struct("<4sHHHHI")
LOW = HEADER.size
TAGS = LOW + TILE_BYTES // 2
OFFSETS = TAGS + TILE_BYTES // 4
EXCEPTIONS = OFFSETS + TILE_BYTES // CHUNK * 2


@dataclass(frozen=True)
class Tile:
    mode: int
    payload: bytes

    @property
    def stored_bytes(self):
        # Exactly the payload, one uint64 offset and one byte mode on device.
        return len(self.payload) + 9


def pack(raw: bytes) -> Tile:
    if len(raw) != TILE_BYTES:
        raise ValueError("the probe requires a complete N128/K128 byte tile")
    exponents = bytes((value >> 3) & 15 for value in raw)
    counts = [exponents.count(symbol) for symbol in range(16)]
    common = sorted(range(16), key=lambda symbol: (-counts[symbol], symbol))[:3]
    tags, exceptions, offsets = bytearray(), bytearray(), []
    tag_codes = {value: tag for tag, value in enumerate(common)}
    for index in range(0, TILE_BYTES, 4):
        if index % CHUNK == 0:
            offsets.append(len(exceptions))
        word = 0
        for j in range(4):
            exponent = exponents[index + j]
            tag = tag_codes.get(exponent, 3)
            word |= tag << (2 * j)
            if tag == 3:
                exceptions.append(exponent)
        tags.append(word)
    book = sum(value << (4 * i) for i, value in enumerate(common))
    header = HEADER.pack(b"F3E1", book, CHUNK, len(exceptions), 0, TILE_BYTES)
    low = pack_nibbles(bytes((value & 7) | ((value >> 4) & 8) for value in raw))
    payload = (header + low + tags + struct.pack("<128H", *offsets)
               + pack_nibbles(exceptions))
    payload += bytes(-len(payload) % 16)
    tile = Tile(1, payload) if len(payload) < TILE_BYTES else Tile(0, raw)
    if unpack(tile) != raw:
        raise ValueError("fixed-field encoding failed its exact round trip")
    return tile


def unpack(tile: Tile) -> bytes:
    if tile.mode == 0:
        if len(tile.payload) != TILE_BYTES:
            raise ValueError("invalid raw tile length")
        return tile.payload
    if tile.mode != 1 or not EXCEPTIONS <= len(tile.payload) < TILE_BYTES:
        raise ValueError("invalid fixed-field tile extent or mode")
    magic, book, chunk, count, reserved, size = HEADER.unpack_from(tile.payload)
    common = [(book >> (4 * i)) & 15 for i in range(3)]
    end = EXCEPTIONS + (count + 1) // 2
    if (magic != b"F3E1" or book >> 12 or chunk != CHUNK or reserved
            or size != TILE_BYTES or len(set(common)) != 3 or count > TILE_BYTES
            or len(tile.payload) != (end + 15) // 16 * 16
            or any(tile.payload[end:])):
        raise ValueError("invalid fixed-field metadata or padding")
    low = unpack_nibbles(tile.payload[LOW:TAGS], TILE_BYTES)
    exceptions = unpack_nibbles(tile.payload[EXCEPTIONS:end], count)
    offsets = struct.unpack_from("<128H", tile.payload, OFFSETS)
    result, position = bytearray(TILE_BYTES), 0
    for index in range(TILE_BYTES):
        if index % CHUNK == 0 and offsets[index // CHUNK] != position:
            raise ValueError("exception chunk offset disagrees with tags")
        tag = (tile.payload[TAGS + index // 4] >> (2 * (index % 4))) & 3
        if tag == 3:
            if position == count:
                raise ValueError("truncated exception stream")
            exponent = exceptions[position]
            position += 1
            if exponent in common:
                raise ValueError("noncanonical common exponent in exception stream")
        else:
            exponent = common[tag]
        value = low[index]
        result[index] = (value & 7) | ((value & 8) << 4) | (exponent << 3)
    if position != count:
        raise ValueError("unused exception nibbles")
    return bytes(result)
