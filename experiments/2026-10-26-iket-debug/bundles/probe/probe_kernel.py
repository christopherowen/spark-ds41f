"""IKET on GB10: three small TileLang kernels compiled under iket.session.

usage: probe_kernel.py OUT_DIR

- ``add``: a warp-local range around marks (TileLang's minimal example);
- ``payload``: int32 and float32 runtime payloads (separate session, payloads on);
- ``smem``: a kernel whose shared buffer starts at the dynamic shared base. It
  records that base's shared-window address and checks every element after an
  IKET mark. IKET's placeholder store writes shared address 0x20; run outside the
  profiler, that store lands wherever 0x20 is, so the check tells whether 0x20 is
  in the per-block reservation (below the dynamic base) or in the kernel's data.

Each kernel's CUDA source and SASS go to OUT_DIR; one JSON line per kernel to stdout.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import torch

import tilelang
import tilelang.language as T
from tilelang.tools.cuda import iket

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
N = 1024
THREADS = 128


def add_kernel(n):
    @T.prim_func
    def iket_probe_add(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32), C: T.Tensor((n,), T.float32)):
        with T.Kernel(T.ceildiv(n, THREADS), threads=THREADS) as bx, iket.range("kernel_total"):
            for i in T.Parallel(THREADS):
                idx = bx * THREADS + i
                if idx < n:
                    iket.mark("before_store")
                    C[idx] = A[idx] + B[idx]
                    iket.mark("after_store")

    return iket_probe_add


def payload_kernel(n):
    @T.prim_func
    def iket_probe_payload(A: T.Tensor((n,), T.float32), C: T.Tensor((n,), T.float32)):
        with T.Kernel(T.ceildiv(n, THREADS), threads=THREADS) as bx, iket.range("block_total"):
            iket.mark("block_enter", payload=iket.payload(bx, dtype="int32"))
            for i in T.Parallel(THREADS):
                idx = bx * THREADS + i
                if idx < n:
                    iket.mark("value", payload=iket.payload(A[idx], dtype="float32"))
                    C[idx] = A[idx] * 2.0

    return iket_probe_payload


def smem_kernel(n, instrumented):
    @T.prim_func
    def iket_probe_smem(A: T.Tensor((n,), T.float32), C: T.Tensor((n,), T.float32), Base: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=n):
            S = T.alloc_shared((n,), T.float32)
            for i in T.Parallel(n):
                S[i] = A[i]
            if instrumented:
                iket.mark("after_fill")
            for i in T.Parallel(n):
                C[i] = S[(i + 1) % n]
            Base[0] = T.call_extern("int32", "__cvta_generic_to_shared", T.address_of(S[0]))

    return iket_probe_smem


def sass(kernel, name):
    lib = getattr(kernel.adapter, "libpath", None)
    record = {"libpath": str(lib)}
    if not lib:
        return record
    out = subprocess.run(["cuobjdump", "-sass", str(lib)], text=True, capture_output=True)
    (OUT / f"{name}.sass").write_text(out.stdout + out.stderr)
    lines = [line.strip() for line in out.stdout.splitlines() if re.search(r"PMTRIG|STS|SR_CgaCtaId|SR_GLOBALTIMER|CS2R|S2UR|S2R", line)]
    record["sass_lines"] = lines[:40]
    record["arch"] = sorted(set(re.findall(r"arch = (sm_\w+)", out.stdout)))
    return record


def report(name, kernel, ok, extra=None):
    source = kernel.get_kernel_source()
    (OUT / f"{name}.cu").write_text(source)
    record = {
        "kernel": name,
        "ok": bool(ok),
        "instrumented": "__iket_meta_info" in source and "TL_IKET_EVENT" in source,
        "cluster_rank_instruction": "%cluster_ctarank" in source,
        **sass(kernel, name),
        **(extra or {}),
    }
    print(json.dumps(record), flush=True)


def main():
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    print(json.dumps({"device": props.name, "capability": f"{props.major}.{props.minor}"}), flush=True)

    with iket.session(output_dir=OUT):
        add = tilelang.compile(add_kernel(N), out_idx=-1, target="cuda", execution_backend="cython")
        smem = tilelang.compile(smem_kernel(256, True), out_idx=[1, 2], target="cuda", execution_backend="cython")
        print(json.dumps({"event_table": iket.event_table()}), flush=True)
    with iket.session(output_dir=OUT, runtime_payloads=True):
        payload = tilelang.compile(payload_kernel(N), out_idx=-1, target="cuda", execution_backend="cython")
    plain_smem = tilelang.compile(smem_kernel(256, False), out_idx=[1, 2], target="cuda", execution_backend="cython")

    a = torch.arange(N, device="cuda", dtype=torch.float32)
    b = torch.full((N,), 2.0, device="cuda", dtype=torch.float32)
    c = add(a, b)
    torch.cuda.synchronize()
    report("add", add, torch.equal(c, a + b), {"target": str(add.target)})

    c = payload(a)
    torch.cuda.synchronize()
    report("payload", payload, torch.equal(c, a * 2.0))

    s = torch.arange(256, device="cuda", dtype=torch.float32) + 1000.0
    expected = torch.roll(s, -1)
    for name, kernel in (("smem", smem), ("smem_plain", plain_smem)):
        c, base = kernel(s)
        torch.cuda.synchronize()
        bad = (c != expected).nonzero().flatten().tolist()
        report(name, kernel, not bad, {"dynamic_shared_base": int(base.item()), "wrong_elements": bad[:16]})


if __name__ == "__main__":
    main()
