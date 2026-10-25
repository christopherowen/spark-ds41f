#!/usr/bin/env python3
"""What L2 cache policies do on this GPU (GB10, SM121), and what a persisting-L2 set-aside
changes: whether a buffer's lines survive a stream several times the L2, by how the
buffer was loaded and how the stream reads, at set-aside sizes from 0 to the device's
maximum. No model, no serving: one process, one GPU.

The buffer P stands for a prefetch window's weights (the shared expert's ~6 MB, a whole
16 MB window); the stream X (96 MB) for a decode layer's routed experts. P is loaded by a
default read, by an evict-last read, or by the L2 prefetch kernel the windows use (the
TileLang ``l2_prefetch_kernel``: ``cp.async.bulk.prefetch.L2`` with an evict-last
policy). X is read with the default policy or evict-first. P is then read again and
timed; its survival is (cold - t) / (cold - hot), from the times of reading P after an
L2 scrub (cold) and right after reading it (hot). All loads skip L1 (``L1::no_allocate``).

The set-aside's cost: a 16 MB working set read twice by default loads, with and without
an evict-last 20 MB buffer prefetched first (stale persisting lines, as prefill would
find them after decode steps), as the second read's survival.

Output: one JSON line per measurement, then a table. Exit status 0 unless a kernel or the
driver fails.
"""

import json
import statistics
import time

import tilelang
import tilelang.language as T
import torch
from cuda.bindings import driver as cu
from vllm.models.glm5next.nvidia.tilelang.l2_prefetch import l2_prefetch_kernel

MB = 1 << 20
STREAM_MB = 96
PROTECTED_MB = (4, 8, 16)
SETASIDE_MB = (0, 4, 8, 12, 16)
REPEATS = 5
GRID, BLOCK = 192, 256
PREFETCH = (2, 128, 4096)  # the recipe's grid on TP4, block, chunk bytes

# L1-bypassing 16-byte loads, by L2 policy. createpolicy has no side effects; the
# loads do (volatile), so nothing is dropped.
_HEADER = r"""
__device__ __forceinline__ unsigned l2probe_hinted(long long base, long long bytes,
                                                   long long first, long long stride,
                                                   unsigned long long policy) {
  unsigned acc = 0;
  for (long long offset = first * 16; offset < bytes; offset += stride * 16) {
    unsigned a, b, c, d;
    asm volatile(
        "ld.global.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(base + offset), "l"(policy));
    acc ^= a ^ b ^ c ^ d;
  }
  return acc;
}

__device__ __forceinline__ unsigned l2probe_default(long long base, long long bytes,
                                                    long long first, long long stride) {
  unsigned acc = 0;
  for (long long offset = first * 16; offset < bytes; offset += stride * 16) {
    unsigned a, b, c, d;
    asm volatile("ld.global.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(base + offset));
    acc ^= a ^ b ^ c ^ d;
  }
  return acc;
}

__device__ __forceinline__ unsigned l2probe_evict_first(long long base, long long bytes,
                                                        long long first, long long stride) {
  unsigned long long policy;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return l2probe_hinted(base, bytes, first, stride, policy);
}

__device__ __forceinline__ unsigned l2probe_evict_last(long long base, long long bytes,
                                                       long long first, long long stride) {
  unsigned long long policy;
  asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
  return l2probe_hinted(base, bytes, first, stride, policy);
}
"""


@tilelang.jit(target="cuda")
def read_kernel(policy: str, grid: int = GRID, block: int = BLOCK):
    """Every thread reads its 16-byte chunks of ``segs``' (address, bytes) range with
    ``policy`` (default, evict_first or evict_last) and stores their XOR."""
    threads = grid * block
    function = f"l2probe_{policy}"

    @T.prim_func
    def l2_probe_read(
        segs: T.Tensor((2,), T.int64), sink: T.Tensor((threads,), T.uint32)
    ):
        with T.Kernel(grid, threads=block) as bx:
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            sink[bx * block + tx] = T.call_extern(
                "uint32",
                function,
                segs[0],
                segs[1],
                T.cast(bx, T.int64) * block + T.cast(tx, T.int64),
                T.int64(threads),
            )

    return l2_probe_read


def check(result, what: str):
    if result[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {result[0]}")
    return result[1] if len(result) > 1 else None


class Device:
    def __init__(self):
        torch.cuda.init()
        index = torch.cuda.current_device()
        self.device = check(cu.cuDeviceGet(index), "cuDeviceGet")
        attribute = cu.CUdevice_attribute
        self.l2 = check(
            cu.cuDeviceGetAttribute(
                attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, self.device
            ),
            "L2 size",
        )
        self.max_persisting = check(
            cu.cuDeviceGetAttribute(
                attribute.CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, self.device
            ),
            "max persisting L2",
        )
        self.context = check(
            cu.cuDevicePrimaryCtxRetain(self.device), "primary context"
        )

    def setaside(self, nbytes: int) -> int:
        """Set the persisting-L2 set-aside of the primary context (torch's); clear every
        persisting line; return the size the driver applied."""
        check(cu.cuCtxPushCurrent(self.context), "cuCtxPushCurrent")
        try:
            limit = cu.CUlimit.CU_LIMIT_PERSISTING_L2_CACHE_SIZE
            check(cu.cuCtxSetLimit(limit, nbytes), "cuCtxSetLimit")
            check(cu.cuCtxResetPersistingL2Cache(), "cuCtxResetPersistingL2Cache")
            return int(check(cu.cuCtxGetLimit(limit), "cuCtxGetLimit"))
        finally:
            cu.cuCtxPopCurrent()

    def reset(self) -> None:
        check(cu.cuCtxPushCurrent(self.context), "cuCtxPushCurrent")
        try:
            check(cu.cuCtxResetPersistingL2Cache(), "cuCtxResetPersistingL2Cache")
        finally:
            cu.cuCtxPopCurrent()


class Buffer:
    def __init__(self, mb: int):
        self.data = torch.randint(
            0, 1 << 30, (mb * MB // 4,), dtype=torch.int32, device="cuda"
        )
        nbytes = self.data.numel() * 4
        self.segs = torch.tensor(
            [self.data.data_ptr(), nbytes], dtype=torch.int64, device="cuda"
        )


SINK = None
READ = {}
PREFETCHER = None


def read(buffer: Buffer, policy: str = "default") -> float:
    """Read ``buffer`` once with ``policy``; return the microseconds it took."""
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    READ[policy](buffer.segs, SINK)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000


def prefetch(buffer: Buffer) -> None:
    """The windows' L2 prefetch of ``buffer`` (evict-last bulk prefetches), landed."""
    PREFETCHER(buffer.segs)
    torch.cuda.synchronize()
    time.sleep(0.005)  # bulk prefetches complete after the kernel


def survival(t: float, cold: float, hot: float) -> float:
    return max(0.0, min(1.0, (cold - t) / (cold - hot))) if cold > hot else float("nan")


def main() -> None:
    global SINK, PREFETCHER
    device = Device()
    print(
        json.dumps(
            {"l2_bytes": device.l2, "max_persisting_bytes": device.max_persisting}
        ),
        flush=True,
    )
    SINK = torch.empty(GRID * BLOCK, dtype=torch.uint32, device="cuda")
    for policy in ("default", "evict_first", "evict_last"):
        READ[policy] = read_kernel(policy)
    PREFETCHER = l2_prefetch_kernel(*PREFETCH)
    scrub, stream = Buffer(256), Buffer(STREAM_MB)
    protected = {mb: Buffer(mb) for mb in PROTECTED_MB}
    working, stale = Buffer(16), Buffer(20)
    sizes = sorted(
        {min(mb * MB, device.max_persisting) for mb in SETASIDE_MB}
        | {device.max_persisting}
    )

    def cold_read(buffer):
        device.reset()
        read(scrub)
        return read(buffer)

    rows = []
    for nbytes in sizes:
        applied = device.setaside(nbytes)
        for mb, buffer in protected.items():
            cold = statistics.median(cold_read(buffer) for _ in range(REPEATS))
            hot = statistics.median(
                (read(buffer), read(buffer))[1] for _ in range(REPEATS)
            )
            for load in ("read", "read evict_last", "prefetch"):
                for policy in ("default", "evict_first"):
                    times = []
                    for _ in range(REPEATS):
                        device.reset()
                        read(scrub)
                        if load == "prefetch":
                            prefetch(buffer)
                        else:
                            read(buffer, "default" if load == "read" else "evict_last")
                        read(stream, policy)
                        times.append(read(buffer))
                    t = statistics.median(times)
                    row = {
                        "setaside_mb": applied / MB,
                        "protected_mb": mb,
                        "load": load,
                        "stream": policy,
                        "us": round(t, 1),
                        "cold_us": round(cold, 1),
                        "hot_us": round(hot, 1),
                        "survival": round(survival(t, cold, hot), 3),
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        # The set-aside's cost to default reuse, with stale persisting lines resident.
        for with_stale in (False, True):
            times = []
            for _ in range(REPEATS):
                device.reset()
                read(scrub)
                if with_stale:
                    prefetch(stale)
                read(working)
                times.append(read(working))
            cold = statistics.median(cold_read(working) for _ in range(REPEATS))
            hot = statistics.median(
                (read(working), read(working))[1] for _ in range(REPEATS)
            )
            t = statistics.median(times)
            row = {
                "setaside_mb": applied / MB,
                "reuse_16mb_after_stale_20mb": with_stale,
                "us": round(t, 1),
                "survival": round(survival(t, cold, hot), 3),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    stream_us = {
        policy: statistics.median(read(stream, policy) for _ in range(REPEATS))
        for policy in ("default", "evict_first", "evict_last")
    }
    print(json.dumps({"stream_96mb_us": stream_us}), flush=True)
    device.setaside(0)
    print(
        f"\nL2 {device.l2 / MB:.1f} MB, max set-aside {device.max_persisting / MB:.1f} MB"
    )
    print("survival of P after the 96 MB stream (1 = all lines hit, 0 = none):")
    for row in rows:
        if "load" in row:
            print(
                f"  set-aside {row['setaside_mb']:5.1f} MB  P {row['protected_mb']:3d} MB  "
                f"{row['load']:16} stream {row['stream']:11}  {row['survival']:.2f}"
            )
        else:
            print(
                f"  set-aside {row['setaside_mb']:5.1f} MB  16 MB reuse, stale 20 MB "
                f"{'yes' if row['reuse_16mb_after_stale_20mb'] else 'no ':3}  "
                f"{row['survival']:.2f}"
            )


if __name__ == "__main__":
    main()
