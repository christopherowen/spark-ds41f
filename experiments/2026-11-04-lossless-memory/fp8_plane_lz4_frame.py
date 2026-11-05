"""Eight bounded bitplane LZ4 streams per FP8 tile; probe ABI, not persistent storage."""

from dataclasses import dataclass
import struct

from memory_codec import bitplanes, unbitplanes

TILE_BYTES = 16384
PLANE_BYTES = TILE_BYTES // 8
HEADER = struct.Struct("<4sHHII")
ENTRY = struct.Struct("<HH")
DATA = HEADER.size + 8 * ENTRY.size
RAW = 0x8000


@dataclass(frozen=True)
class Tile:
    mode: int
    payload: bytes

    @property
    def stored_bytes(self):
        return len(self.payload) + 9  # uint64 offset and uint8 mode on device


def pack(raw, lz4, chunk_bytes=PLANE_BYTES):
    if len(raw) != TILE_BYTES or chunk_bytes not in (256, 512, 1024, 2048):
        raise ValueError("the probe requires a complete N128/K128 tile")
    planes = bitplanes(raw)
    entries, payload = bytearray(), bytearray()
    data = HEADER.size + TILE_BYTES // chunk_bytes * ENTRY.size
    for start in range(0, TILE_BYTES, chunk_bytes):
        source = planes[start:start + chunk_bytes]
        compressed = lz4.compress(source)
        # Equal aligned size buys no bandwidth; keep that plane RAW.
        coded = (len(compressed) + 15) // 16 * 16 < chunk_bytes
        block = compressed if coded else source
        entries.extend(ENTRY.pack(data + len(payload), len(block) | (0 if coded else RAW)))
        payload.extend(block)
        payload.extend(bytes(-len(payload) % 16))
    magic = b"P4L1" if chunk_bytes == PLANE_BYTES else b"P4L2"
    frame = HEADER.pack(magic, 8, chunk_bytes, TILE_BYTES, 0) + entries + payload
    tile = Tile(1, bytes(frame)) if len(frame) < TILE_BYTES else Tile(0, raw)
    if unpack(tile, lz4) != raw:
        raise ValueError("plane-LZ4 encoding failed exact reconstruction")
    return tile


def unpack(tile, lz4):
    if tile.mode == 0:
        if len(tile.payload) != TILE_BYTES:
            raise ValueError("invalid RAW tile extent")
        return tile.payload
    if tile.mode != 1 or not HEADER.size <= len(tile.payload) < TILE_BYTES:
        raise ValueError("invalid plane-LZ4 tile extent or mode")
    magic, count, chunk, size, reserved = HEADER.unpack_from(tile.payload)
    if (count != 8 or size != TILE_BYTES or reserved or chunk not in (256, 512, 1024, 2048)
            or magic != (b"P4L1" if chunk == PLANE_BYTES else b"P4L2")):
        raise ValueError("invalid plane-LZ4 header")
    streams = TILE_BYTES // chunk
    position, planes = HEADER.size + streams * ENTRY.size, bytearray()
    if len(tile.payload) < position:
        raise ValueError("truncated plane-LZ4 directory")
    for stream in range(streams):
        offset, flags = ENTRY.unpack_from(tile.payload, HEADER.size + stream * ENTRY.size)
        length = flags & ~RAW
        raw = bool(flags & RAW)
        end = offset + length
        padded_end = (end + 15) // 16 * 16
        if (offset != position or not 0 < length <= chunk
                or raw and length != chunk or not raw and padded_end - offset >= chunk
                or padded_end > len(tile.payload) or any(tile.payload[end:padded_end])):
            raise ValueError("invalid plane-LZ4 stream extent or padding")
        block = tile.payload[offset:end]
        planes.extend(block if raw else lz4.decompress(block, chunk))
        position = padded_end
    if position != len(tile.payload):
        raise ValueError("unused plane-LZ4 payload")
    return unbitplanes(planes, TILE_BYTES)


def transpose8(word):
    # Exchange address-coordinate bits (0,3), (1,4), (2,5). A bit at
    # plane*8+value moves to value*8+plane, with no arithmetic on FP8.
    for shift, mask in ((7, 0x00AA00AA00AA00AA), (14, 0x0000CCCC0000CCCC),
                        (28, 0x00000000F0F0F0F0)):
        difference = (word ^ (word >> shift)) & mask
        word ^= difference ^ (difference << shift)
    return word
