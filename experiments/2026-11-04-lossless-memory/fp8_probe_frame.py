"""Exact parallel FP8 streams for a GPU decoder screen, not a serving ABI."""

from dataclasses import dataclass
import struct

from memory_codec import canonical_codes, huffman_fp8, pack_nibbles, unpack_nibbles

TILE_BYTES = 16384


@dataclass(frozen=True)
class Tile:
    mode: int
    lanes: int
    payload: bytes
    low: bytes = b""
    tree: bytes = b""
    quick: bytes = b""
    offsets: tuple[int, ...] = ()

    @property
    def stored_bytes(self):
        return 16 + len(self.payload) + len(self.low) + len(self.tree) + len(self.quick) + 4 * len(self.offsets)


def pack(raw: bytes, lanes: int) -> Tile:
    if len(raw) != TILE_BYTES or lanes not in (32, 64, 128):
        raise ValueError("the probe requires N128/K128 tiles and 32/64/128 lanes")
    lengths = huffman_fp8(raw)[:16]
    codes = canonical_codes(lengths)
    children = [[255, 255]]
    for symbol, (code, length) in codes.items():
        node = 0
        for position in range(length - 1, -1, -1):
            bit = (code >> position) & 1
            if position == 0:
                children[node][bit] = 128 | symbol
            else:
                if children[node][bit] == 255:
                    children[node][bit] = len(children)
                    children.append([255, 255])
                node = children[node][bit]
    tree = bytes(value for pair in children for value in pair).ljust(32, b"\xff")
    if len(tree) != 32:
        raise ValueError("exponent tree exceeds the bounded 16-symbol contract")
    quick = []
    for prefix in range(256):
        node, entry = 0, 0
        for position in range(8):
            node = tree[2 * node + ((prefix >> (7 - position)) & 1)]
            if node == 255:
                break
            if node & 128:
                entry = ((position + 1) << 4) | (node & 15)
                break
        quick.append(entry)
    streams, offsets = bytearray(), []
    for lane in range(lanes):
        offsets.append(len(streams))
        acc, bits = 0, 0
        for index in range(lane, TILE_BYTES, lanes):
            code, length = codes[(raw[index] >> 3) & 15]
            acc = (acc << length) | code
            bits += length
            while bits >= 8:
                bits -= 8
                streams.append((acc >> bits) & 255)
                acc &= (1 << bits) - 1
        if bits:
            streams.append(acc << (8 - bits))
        streams.extend(b"\0\0\0")  # bounded three-byte GPU lookahead, per lane
    low = pack_nibbles(bytes((v & 7) | ((v >> 4) & 8) for v in raw))
    tile = Tile(1, lanes, bytes(streams), low, tree,
                struct.pack("<256H", *quick), tuple(offsets))
    if tile.stored_bytes >= 16 + len(raw):
        tile = Tile(0, lanes, raw)
    if unpack(tile) != raw:
        raise ValueError("parallel-stream encoding failed its exact round trip")
    return tile


def unpack(tile: Tile) -> bytes:
    if tile.lanes not in (32, 64, 128):
        raise ValueError("invalid lane count")
    if tile.mode == 0:
        if len(tile.payload) != TILE_BYTES or tile.low or tile.offsets or tile.tree or tile.quick:
            raise ValueError("invalid raw tile")
        return tile.payload
    if (tile.mode != 1 or len(tile.tree) != 32 or len(tile.quick) != 512 or
            len(tile.offsets) != tile.lanes or tile.offsets[0] != 0 or
            list(tile.offsets) != sorted(set(tile.offsets))):
        raise ValueError("invalid parallel tile metadata")
    low = unpack_nibbles(tile.low, TILE_BYTES)
    out = bytearray(TILE_BYTES)
    for lane, start in enumerate(tile.offsets):
        end = tile.offsets[lane + 1] if lane + 1 < tile.lanes else len(tile.payload)
        if not 0 <= start < end - 3 <= len(tile.payload) - 3:
            raise ValueError("invalid stream extent")
        data = tile.payload[start:end - 3]
        if tile.payload[end - 3:end] != b"\0\0\0":
            raise ValueError("invalid lookahead padding")
        position = 0
        for index in range(lane, TILE_BYTES, tile.lanes):
            initial = position
            node = 0
            for _ in range(15):
                if position >= len(data) * 8:
                    raise ValueError("truncated exponent stream")
                bit = (data[position // 8] >> (7 - position % 8)) & 1
                position += 1
                node = tile.tree[2 * node + bit]
                if node == 255:
                    raise ValueError("invalid exponent code")
                if node & 128:
                    break
            else:
                raise ValueError("exponent code exceeds 15 bits")
            byte = start + initial // 8
            word = int.from_bytes(tile.payload[byte:byte + 3], "big")
            prefix = (word >> (16 - initial % 8)) & 255
            entry = struct.unpack_from("<H", tile.quick, prefix * 2)[0]
            if entry and (entry >> 4 != position - initial or entry & 15 != node & 15):
                raise ValueError("GPU prefix lookup disagrees with the exact exponent tree")
            a = low[index]
            out[index] = (a & 7) | ((a & 8) << 4) | ((node & 15) << 3)
        if len(data) != (position + 7) // 8:
            raise ValueError("trailing exponent bytes")
        if position % 8 and data[-1] & ((1 << (8 - position % 8)) - 1):
            raise ValueError("nonzero exponent padding")
    return bytes(out)
