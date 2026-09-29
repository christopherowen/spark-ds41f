#!/usr/bin/env python3
"""Probe the GB10 UEFI display reserve through DRM dumb buffers.

For each size: create a dumb buffer on card0, map it, write every page, and
report MemFree/MemAvailable movement, then destroy it. A buffer carved from
the hidden reserve leaves both unchanged; one backed by ordinary RAM drops
MemFree by its size. Stops at the first size that allocates and maps.
"""
import fcntl
import mmap
import os
import struct
import sys
import time

DRM_IOCTL_GET_CAP = 0xC010640C
DRM_IOCTL_MODE_CREATE_DUMB = 0xC02064B2
DRM_IOCTL_MODE_MAP_DUMB = 0xC01064B3
DRM_IOCTL_MODE_DESTROY_DUMB = 0xC00464B4
DRM_CAP_DUMB_BUFFER = 0x1
ROW = 4096 * 4  # 4096 px x 32 bpp
MIB = 1 << 20


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            out[key] = int(value.split()[0]) * 1024
    return out


def gib(n):
    return f"{n / (1 << 30):.3f} GiB"


def main():
    sizes = [int(s) for s in sys.argv[1:]] or [2048, 2040, 2032, 2016, 2000, 1984, 1952, 1920, 1792, 1536, 1024, 512]
    fd = os.open("/dev/dri/card0", os.O_RDWR | os.O_CLOEXEC)
    cap = bytearray(struct.pack("QQ", DRM_CAP_DUMB_BUFFER, 0))
    fcntl.ioctl(fd, DRM_IOCTL_GET_CAP, cap)
    print("dumb buffer cap:", struct.unpack("QQ", cap)[1])
    for mib_size in sizes:
        size = mib_size * MIB
        height = size // ROW
        req = bytearray(struct.pack("IIIIIIQ", height, 4096, 32, 0, 0, 0, 0))
        before = meminfo()
        try:
            fcntl.ioctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, req)
        except OSError as exc:
            print(f"{mib_size} MiB: create failed: {exc}")
            continue
        _, _, _, _, handle, pitch, got = struct.unpack("IIIIIIQ", req)
        try:
            mreq = bytearray(struct.pack("IIQ", handle, 0, 0))
            fcntl.ioctl(fd, DRM_IOCTL_MODE_MAP_DUMB, mreq)
            offset = struct.unpack("IIQ", mreq)[2]
            buf = mmap.mmap(fd, got, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)
            t0 = time.time()
            chunk = b"\xa5" * (16 * MIB)
            for pos in range(0, got, len(chunk)):
                buf[pos:pos + len(chunk)] = chunk[: got - pos]
            wrote = time.time() - t0
            ok = buf[got - 1] == 0xA5 and buf[0] == 0xA5
            after = meminfo()
            print(
                f"{mib_size} MiB: handle={handle} pitch={pitch} size={got} offset={offset:#x} "
                f"write {got / wrote / 1e9:.2f} GB/s ok={ok} "
                f"MemFree {gib(after['MemFree'] - before['MemFree'])} "
                f"MemAvailable {gib(after['MemAvailable'] - before['MemAvailable'])} "
                f"Shmem {gib(after['Shmem'] - before['Shmem'])}"
            )
            buf.close()
        finally:
            fcntl.ioctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, bytearray(struct.pack("I", handle)))
        settled = meminfo()
        print(f"  after destroy: MemFree {gib(settled['MemFree'] - before['MemFree'])} vs start")
        break
    os.close(fd)


if __name__ == "__main__":
    main()
