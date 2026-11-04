# SPDX-License-Identifier: Apache-2.0
"""Bounded, lossless transport for the existing TileLang IKET phase pass."""

import atexit
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
_ACTIVE = None


class Collector:
    def __init__(
        self,
        out,
        ring_bytes=8 << 20,
        chunk_bytes=64 << 20,
        writer_delay_us=0,
        fail_after_records=0,
        uniform_registers=False,
        managed=True,
    ):
        import torch

        torch.cuda.init()
        self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / "sources").mkdir()
        self.cuda = ctypes.CDLL("libcudart.so.13")
        self.cuda.cudaHostAlloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self.cuda.cudaHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self.cuda.cudaFreeHost.argtypes = [ctypes.c_void_p]
        self.cuda.cudaMallocManaged.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self.cuda.cudaFree.argtypes = [ctypes.c_void_p]
        self.cuda.cudaDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        for attribute in (86, 89) if managed else (19, 86):
            value = ctypes.c_int()
            self.check(
                self.cuda.cudaDeviceGetAttribute(
                    ctypes.byref(value), attribute, torch.cuda.current_device()
                )
            )
            if not value.value:
                raise RuntimeError(
                    f"device lacks mapped/system-atomic capability {attribute}"
                )
        if ring_bytes < 128 or chunk_bytes < 128:
            raise ValueError("ring and file chunks must hold at least one record")
        self.capacity = (ring_bytes - 64) // 64
        self.size = 64 + self.capacity * 64
        self.host = ctypes.c_void_p()
        self.device = ctypes.c_void_p()
        if managed:
            self.check(
                self.cuda.cudaMallocManaged(ctypes.byref(self.host), self.size, 1)
            )
            self.device.value = self.host.value
            self.free_memory = self.cuda.cudaFree
        else:
            self.check(self.cuda.cudaHostAlloc(ctypes.byref(self.host), self.size, 2))
            self.check(
                self.cuda.cudaHostGetDevicePointer(
                    ctypes.byref(self.device), self.host, 0
                )
            )
            self.free_memory = self.cuda.cudaFreeHost
        ctypes.memset(self.host, 0, self.size)
        self.header = (ctypes.c_uint64 * 8).from_address(self.host.value)
        self.header[0] = self.capacity
        for index in range(self.capacity):
            ctypes.c_uint64.from_address(
                self.host.value + 64 + index * 64
            ).value = index
        library = self.out / "writer.so"
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O2",
                "-shared",
                "-fPIC",
                "-pthread",
                str(HERE / "writer.cpp"),
                "-lz",
                "-o",
                str(library),
            ],
            check=True,
        )
        self.lib = ctypes.CDLL(str(library))
        self.lib.stream_start.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self.lib.stream_start.restype = ctypes.c_void_p
        self.lib.stream_abort.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.stream_finish.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.handle = self.lib.stream_start(
            self.host,
            os.fsencode(self.out),
            chunk_bytes,
            writer_delay_us,
            fail_after_records,
        )
        if not self.handle:
            code = self.header[3]
            self.free_memory(self.host)
            raise OSError(code, "trace writer could not start")
        self.closed = False
        self.uniform_registers = uniform_registers
        self.write_json(
            "identity.json",
            {
                "format": "SPKIKET1",
                "format_version": 2,
                "record_bytes": 64,
                "writer_source_sha256": hashlib.sha256(
                    (HERE / "writer.cpp").read_bytes()
                ).hexdigest(),
                "writer_binary_sha256": hashlib.sha256(
                    library.read_bytes()
                ).hexdigest(),
                "device_source_sha256": hashlib.sha256(
                    (HERE / "device.cuh").read_bytes()
                ).hexdigest(),
                "ring_bytes": self.size,
                "capacity_records": self.capacity,
                "chunk_bytes": chunk_bytes,
                "writer_delay_us": writer_delay_us,
                "fail_after_records": fail_after_records,
                "device": torch.cuda.get_device_name(),
                "pid": os.getpid(),
                "uniform_registers": uniform_registers,
                "allocation": "managed" if managed else "mapped-host",
                "scope": "TileLang lane-zero events; not arbitrary SASS tracing",
            },
        )
        atexit.register(self._at_exit)

    def _at_exit(self):
        try:
            self.close()
        except Exception as error:
            print(f"Streaming trace failed: {error}", file=sys.stderr, flush=True)
            os._exit(1)

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"CUDA runtime error {code}")

    def write_json(self, name, data):
        (self.out / name).write_text(json.dumps(data, indent=2) + "\n")

    def close(self):
        if self.closed:
            return
        self.closed = True
        import torch

        sync_error = None
        try:
            torch.cuda.synchronize()
        except Exception as error:
            sync_error = str(error)
            # Stop the reader if a failed producer left an unpublished ticket.
            self.lib.stream_abort(self.host, self.header[3] or 125)
        written, chunks = ctypes.c_uint64(), ctypes.c_uint64()
        status = self.lib.stream_finish(
            self.handle, ctypes.byref(written), ctypes.byref(chunks)
        )
        self.write_json(
            "summary.json",
            {
                "reserved_records": self.header[1],
                "consumed_records": self.header[2],
                "written_records": written.value,
                "chunks": chunks.value,
                "producer_waits": self.header[4],
                "status": status,
                "cuda_error": sync_error,
                "complete": not status
                and not sync_error
                and written.value == self.header[1],
            },
        )
        # Compiled kernels embed this pointer. Keep the bounded allocation alive
        # until process exit; stop=1 makes subsequent calls skip recording.
        if status or sync_error:
            raise RuntimeError(
                f"incomplete streaming capture: {os.strerror(status) if status else sync_error}; writer status {status}; receipts: {self.out}"
            )

    def postproc(self, code, target):
        from vllm.models.deepseek_v4_1.tilelang import iket

        ctas, iters, skip, granularity, warps, payloads, _ = iket._sampling
        kernels = list(iket._kernels(code))
        if not kernels:
            return code
        if skip and all(re.search(skip, name) for name, *_ in kernels):
            return code
        original_digest = hashlib.sha256(code.encode()).hexdigest()
        redistribution = re.findall(
            r"tl::warpgroup_reg_(?:dealloc|alloc)<[0-9]+>\(\);", code
        )
        if self.uniform_registers:
            code = re.sub(
                r"tl::warpgroup_reg_(?:dealloc|alloc)<[0-9]+>\(\);",
                "/* streaming: uniform register allocation */",
                code,
            )
        digest = hashlib.sha256(code.encode()).hexdigest()
        annotated, sites = iket.annotate(
            code,
            lambda name, number: str(number),
            ctas,
            iters,
            granularity,
            warps,
            payloads,
        )
        annotated = annotated.replace(iket._SETTLE, "")
        inserts = []
        for name, _, start, end in iket._kernels(annotated):
            number = int(hashlib.sha256((digest + name).encode()).hexdigest()[:16], 16)
            stem = f"{name}.{digest[:12]}"
            manifest = {
                "kernel": name,
                "kernel_id": number,
                "ctas": ctas,
                "iters": iters,
                "warps": warps,
                "granularity": granularity,
                "payloads": payloads,
                "original_code_sha256": original_digest,
                "register_redistribution": "disabled"
                if redistribution and self.uniform_registers
                else "unchanged",
                "sites": [
                    dict(
                        s,
                        range_id=__import__("zlib").crc32(s["range"].encode())
                        & 0xFFFFFFFF
                        or 1,
                    )
                    if "range" in s
                    else s
                    for s in sites
                    if s["kernel"] == name
                ],
            }
            self.write_json("sources/" + stem + ".json", manifest)
            # One elected active lane per warp emits each event. %gridid identifies
            # real launches, including graph replays, without host launch hooks.
            entry = (
                f"\n  const unsigned long long __spark_trace_kernel = {number}ULL;\n"
            )
            entry += "  spark_trace_emit(__spark_trace_kernel, 0, 0, 0);\n"
            inserts.append((start + 1, start + 1, entry))
            inserts.append(
                (
                    end - 1,
                    end - 1,
                    "\n  spark_trace_emit(__spark_trace_kernel, 3, 0, 0);\n",
                )
            )
            for match in re.finditer(r"\breturn\s*;", annotated[start + 1 : end - 1]):
                inserts.append(
                    (
                        start + 1 + match.start(),
                        start + 1 + match.end(),
                        "{ spark_trace_emit(__spark_trace_kernel, 3, 0, 0); return; }",
                    )
                )
        for position, until, text in sorted(inserts, reverse=True):
            annotated = annotated[:position] + text + annotated[until:]
        header = (
            f"#define SPARK_TRACE_ADDRESS {self.device.value}ULL\n"
            + (HERE / "device.cuh").read_text()
        )
        source = header + "\n" + annotated
        (self.out / "sources" / (kernels[0][0] + "." + digest[:12] + ".cu")).write_text(
            source
        )
        return source


def enable(
    out,
    *,
    ring_bytes=8 << 20,
    chunk_bytes=64 << 20,
    writer_delay_us=0,
    fail_after_records=0,
    ctas=1,
    iters=1,
    warps=1,
    payloads=True,
    patched=False,
    uniform_registers=False,
    managed=True,
):
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("one collector per process")
    if os.environ.get("CUDA_INJECTION64_PATH") or os.environ.get(
        "VLLM_DS41_IKET_CACHE_DIR"
    ):
        raise RuntimeError(
            "streaming requires its own collector and no persistent instrumented cache"
        )
    from vllm.models.deepseek_v4_1.tilelang import iket

    collector = Collector(
        out,
        ring_bytes,
        chunk_bytes,
        writer_delay_us,
        fail_after_records,
        uniform_registers,
        managed,
    )
    if patched:
        from .phase import load

        iket = load(collector.out)
    iket._postproc = collector.postproc
    iket.enable(
        out, ctas=ctas, iters=iters, warps=warps, granularity="phase", payloads=payloads
    )
    _ACTIVE = collector
    return collector
