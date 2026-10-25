#!/usr/bin/env python3
"""The SASS of the evict-first GEMM's hinted K-tail copy, which GB10 rejects.

evict_first_cause.py: at the TP4 down shape (384x576, a 64-wide K tail) the GEMM faults
(Xid 13, "Illegal Instruction Parameter") whenever the tail's 8-byte cp.async carries the
L2 cache hint, and runs with the release's bits without it; the 16-byte hinted copy of
the scale codes runs. The same 8-byte hinted cp.async runs alone (cp_async_hint_diag.py).
This compiles, without launching, the down GEMM with the helper on the tail only, with
the unhinted helper, and with the helper on the codes only, plus the standalone 8-byte
hinted kernel, and prints each one's LDGSTS instructions with the instructions before
them (cuobjdump -sass of the compiled library).

Mounts: /b/moe_release.py (the release's moe.py), /b/moe_evict_first.py (the change's).
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

RELEASE = Path("/b/moe_release.py").read_text()
CHANGE = Path("/b/moe_evict_first.py").read_text()
TMA_HINT = '                eviction_policy="evict_first",\n'
HELPER = CHANGE.replace(TMA_HINT, "")
HINT16 = """        "{\\n"
        "  .reg .b64 policy;\\n"
        "  createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\\n"
        "  cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, policy;\\n"
        "}\\n" ::"r"(dst), "l"(gmem) : "memory");"""
HINT_N = """        "{\\n"
        "  .reg .b64 policy;\\n"
        "  createpolicy.fractional.L2::evict_first.b64 policy, 1.0;\\n"
        "  cp.async.ca.shared.global.L2::cache_hint [%0], [%1], %2, policy;\\n"
        "}\\n" ::"r"(dst), "l"(gmem), "n"(N) : "memory");"""
TAIL_HELPER = """                        _cp_async_evict_first(
                            T.access_ptr(B_shared[s, n, g * 16], "w", 8),"""
TAIL_RELEASE = """                        T.ptx_cp_async(
                            T.access_ptr(B_shared[s, n, g * 16], "w", 8),"""
CODES_HELPER = """                _cp_async_evict_first(
                    T.access_ptr(SFB_shared[s, tx * 8], "w", 8),
                    T.access_ptr(
                        SFB[(expert * n_blocks + bx) * k_blocks + ko, tx * 8], "r", 8
                    ),
                    16,  # eight 2-byte codes"""
CODES_RELEASE = """                T.ptx_cp_async(
                    T.access_ptr(SFB_shared[s, tx * 8], "w", 8),
                    T.access_ptr(
                        SFB[(expert * n_blocks + bx) * k_blocks + ko, tx * 8], "r", 8
                    ),
                    8,  # elements: eight 2-byte codes, one 16-byte copy"""
for text, piece in (
    (HELPER, HINT16),
    (HELPER, HINT_N),
    (HELPER, TAIL_HELPER),
    (HELPER, CODES_HELPER),
    (RELEASE, TAIL_RELEASE),
    (RELEASE, CODES_RELEASE),
):
    assert text.count(piece) == 1, piece[:60]
PLAIN = HELPER.replace(
    HINT16,
    '        "cp.async.cg.shared.global [%0], [%1], 16;\\n" ::"r"(dst), "l"(gmem) : "memory");',
).replace(
    HINT_N,
    '        "cp.async.ca.shared.global [%0], [%1], %2;\\n" ::"r"(dst), "l"(gmem), "n"(N) : "memory");',
)
GEMMS = {
    "tail hinted (faults)": HELPER.replace(CODES_HELPER, CODES_RELEASE),
    "helper, no hint (runs)": PLAIN,
    "codes hinted (runs)": HELPER.replace(TAIL_HELPER, TAIL_RELEASE),
}

COMPILE = r"""
import importlib.util, sys
path = sys.argv[1]
package = "vllm.models.deepseek_v4_1.tilelang"
spec = importlib.util.spec_from_file_location(package + ".moe_variant", path)
moe = importlib.util.module_from_spec(spec)
moe.__package__ = package
spec.loader.exec_module(moe)
moe.grouped_fp8_fp4_gemm(32, 384, 576, 6, False, 32, 128, coded_scales=True)
"""

STANDALONE = r'''
import tilelang
import tilelang.language as T
HEADER = """
__device__ __forceinline__ void diag_copy(const void* smem, const void* gmem) {
  unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  unsigned long long p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  asm volatile("cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 8, %2;" ::"r"(d), "l"(gmem), "l"(p) : "memory");
}
"""
@tilelang.jit(target="cuda")
def kernel(threads: int = 128):
    words = threads * 2
    @T.prim_func
    def diag_cp_async(src: T.Tensor((words,), T.int32), dst: T.Tensor((words,), T.int32)):
        with T.Kernel(1, threads=threads) as bx:
            T.import_source(HEADER)
            shared = T.alloc_shared((words,), T.int32)
            tx = T.get_thread_binding()
            T.call_extern("handle", "diag_copy", T.access_ptr(shared[tx * 2], "w", 2), T.access_ptr(src[tx * 2], "r", 2))
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.sync_threads()
            for i in T.serial(2):
                dst[tx * 2 + i] = shared[tx * 2 + i]
    return diag_cp_async
kernel()
'''


def sass(cache: Path, function: str) -> list[str]:
    tool = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    lines = []
    for lib in sorted(
        p for p in cache.rglob("*") if p.suffix in (".so", ".cubin", ".fatbin")
    ):
        out = subprocess.run(
            [tool, "-sass", str(lib)], capture_output=True, text=True
        ).stdout
        if function in out:
            lines = out.splitlines()
            break
    else:
        print(
            "  no SASS found; files:",
            [str(p.relative_to(cache)) for p in cache.rglob("*") if p.is_file()][:40],
        )
    return lines


def show(lines: list[str], function: str) -> None:
    start = next(
        (i for i, line in enumerate(lines) if "Function" in line and function in line),
        None,
    )
    if start is None:
        print("  function not found")
        return
    body = []
    for line in lines[start + 1 :]:
        if "Function :" in line:
            break
        body.append(line)
    code = [line.strip() for line in body if re.search(r"/\*[0-9a-f]{4,}\*/", line)]
    hits = [
        i
        for i, line in enumerate(code)
        if "LDGSTS" in line or "R2UR" in line or "UBLKCP" in line
    ]
    print(f"  {len(code)} instructions, {sum('LDGSTS' in c for c in code)} LDGSTS")
    shown = set()
    for i in hits:
        for j in range(max(0, i - 3), i + 1):
            if j not in shown and ("LDGSTS" in code[j] or j >= i - 3):
                shown.add(j)
                print("   ", code[j][:160])
        print("    ...")


def main() -> int:
    for index, (name, module) in enumerate(GEMMS.items()):
        cache = Path(f"/tmp/tl_sass_{index}")
        shutil.rmtree(cache, ignore_errors=True)
        path = Path(f"/tmp/moe_sass_{index}.py")
        path.write_text(module)
        script = Path(f"/tmp/moe_sass_compile_{index}.py")
        script.write_text(COMPILE)
        env = {**os.environ, "TILELANG_CACHE_DIR": str(cache)}
        run = subprocess.run(
            [sys.executable, str(script), str(path)],
            capture_output=True,
            text=True,
            env=env,
        )
        print(f"\n===== {name}: compile exit {run.returncode}", flush=True)
        if run.returncode:
            print(run.stderr[-1500:])
            continue
        show(sass(cache, "moe_grouped_fp8_fp4_gemm"), "moe_grouped_fp8_fp4_gemm")
    cache = Path("/tmp/tl_sass_standalone")
    shutil.rmtree(cache, ignore_errors=True)
    script = Path("/tmp/moe_sass_standalone.py")
    script.write_text(STANDALONE)
    env = {**os.environ, "TILELANG_CACHE_DIR": str(cache)}
    run = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env
    )
    print(
        f"\n===== standalone 8-byte hinted cp.async (runs): compile exit {run.returncode}",
        flush=True,
    )
    if run.returncode:
        print(run.stderr[-1500:])
    else:
        show(sass(cache, "diag_cp_async"), "diag_cp_async")
    return 0


if __name__ == "__main__":
    sys.exit(main())
