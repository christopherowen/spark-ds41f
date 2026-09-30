#!/usr/bin/env python3
"""Write the fatbin embedded in each B12X compile-cache object next to it.

usage: extract_fatbins.py obj.o [...]   (then: cuobjdump -sass obj.fatbin > obj.sass)

B12X's compile cache (B12X_COMPILE_CACHE_DIR, <key>.o beside <key>.json) stores
host ELF objects with the device code as an embedded fatbin; cuobjdump does not
find it in the .o itself. The fatbin starts at magic 0xBA55ED50, followed by a
16-byte header whose last field is the payload size.
"""
import struct
import sys

for path in sys.argv[1:]:
    data = open(path, "rb").read()
    at, n = data.find(b"\x50\xed\x55\xba"), 0
    while at >= 0:
        _magic, _version, header, size = struct.unpack_from("<IHHQ", data, at)
        n += 1
        out = path[:-2] + (".fatbin" if n == 1 else f".{n}.fatbin")
        open(out, "wb").write(data[at:at + header + size])
        at = data.find(b"\x50\xed\x55\xba", at + header + size)
    if n != 1:
        print(f"{path}: {n} fatbins", file=sys.stderr)
