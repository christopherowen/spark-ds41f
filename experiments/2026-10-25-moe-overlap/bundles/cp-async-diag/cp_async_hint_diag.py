#!/usr/bin/env python3
"""Which form of a hinted cp.async is illegal on GB10 (SM121).

evict_first_diag.py traced the evict-first GEMM's CUDA_ERROR_ILLEGAL_INSTRUCTION to its
cp.async helper (createpolicy, then cp.async ... .L2::cache_hint), not the TMA tiles'
hint. Each variant here is one kernel that copies 16 (or 8) bytes a thread from global to
shared memory with one cp.async form, waits, and writes the shared bytes back; it runs in
its own process and its bytes are checked against the source:

- plain: cp.async.cg without a hint (TileLang's form; the control);
- hint: .L2::cache_hint with a createpolicy evict_first word (the change's form);
- hint, src-size: the same with an explicit source size operand;
- hint, evict_normal: a createpolicy evict_normal word;
- hint, constant: TileLang's constant TMA word for evict_first (0x12F0000000000000);
- hint, 8 bytes: cp.async.ca, 8 bytes, createpolicy evict_first (the K tails' form).

It also prints the createpolicy words of this GPU.
"""
import json
import subprocess
import sys
from pathlib import Path

POLICY_FIRST = (
    'asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));'
)
POLICY_NORMAL = (
    'asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));'
)
POLICY_CONSTANT = "p = 0x12F0000000000000ull;"
COPY16 = '"cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(s) : "memory"'
HINT16 = (
    '"cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" '
    '::"r"(d), "l"(s), "l"(p) : "memory"'
)
HINT16_SIZE = (
    '"cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, 16, %2;" '
    '::"r"(d), "l"(s), "l"(p) : "memory"'
)
HINT8 = (
    '"cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 8, %2;" '
    '::"r"(d), "l"(s), "l"(p) : "memory"'
)
VARIANTS = {
    "plain": (16, "", COPY16),
    "hint": (16, POLICY_FIRST, HINT16),
    "hint, src-size": (16, POLICY_FIRST, HINT16_SIZE),
    "hint, evict_normal": (16, POLICY_NORMAL, HINT16),
    "hint, constant": (16, POLICY_CONSTANT, HINT16),
    "hint, 8 bytes": (8, POLICY_FIRST, HINT8),
}

KERNEL = r'''
import sys
import tilelang
import tilelang.language as T
import torch

BYTES = {nbytes}
HEADER = """
__device__ __forceinline__ void diag_copy(const void* smem, const void* gmem) {{
  unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const void* s = gmem;
  unsigned long long p = 0;
  {policy}
  (void)p;
  asm volatile({copy});
}}
"""


@tilelang.jit(target="cuda")
def kernel(threads: int = 128):
    words = threads * BYTES // 4

    @T.prim_func
    def diag_cp_async(src: T.Tensor((words,), T.int32), dst: T.Tensor((words,), T.int32)):
        with T.Kernel(1, threads=threads) as bx:
            T.import_source(HEADER)
            shared = T.alloc_shared((words,), T.int32)
            tx = T.get_thread_binding()
            T.call_extern(
                "handle",
                "diag_copy",
                T.access_ptr(shared[tx * (BYTES // 4)], "w", BYTES // 4),
                T.access_ptr(src[tx * (BYTES // 4)], "r", BYTES // 4),
            )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.sync_threads()
            for i in T.serial(BYTES // 4):
                dst[tx * (BYTES // 4) + i] = shared[tx * (BYTES // 4) + i]

    return diag_cp_async


src = torch.arange(128 * BYTES // 4, dtype=torch.int32, device="cuda")
dst = torch.zeros_like(src)
kernel()(src, dst)
torch.cuda.synchronize()
sys.exit(0 if torch.equal(src, dst) else 3)
'''

POLICIES = r'''
import json
import tilelang
import tilelang.language as T
import torch

HEADER = """
__device__ __forceinline__ long long diag_policy(int kind) {
  unsigned long long p;
  if (kind == 0) asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
  else if (kind == 1) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  else asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return (long long)p;
}
"""


@tilelang.jit(target="cuda")
def policies():
    @T.prim_func
    def diag_policies(out: T.Tensor((3,), T.int64)):
        with T.Kernel(1, threads=32) as bx:
            T.import_source(HEADER)
            tx = T.get_thread_binding()
            if tx < 3:
                out[tx] = T.call_extern("int64", "diag_policy", tx)

    return diag_policies


out = torch.zeros(3, dtype=torch.int64, device="cuda")
policies()(out)
torch.cuda.synchronize()
names = ("evict_normal", "evict_first", "evict_last")
print(json.dumps({n: hex(v & 0xFFFFFFFFFFFFFFFF) for n, v in zip(names, out.tolist())}))
'''


def run(source: str, name: str) -> subprocess.CompletedProcess:
    path = Path(f"/tmp/cp_async_diag_{name}.py")
    path.write_text(source)
    return subprocess.run([sys.executable, str(path)], capture_output=True, text=True)


def main() -> int:
    words = run(POLICIES, "policies")
    print("createpolicy words:", words.stdout.strip() or words.stderr.strip()[-1500:])
    for index, (name, (nbytes, policy, copy)) in enumerate(VARIANTS.items()):
        source = KERNEL.format(nbytes=nbytes, policy=policy, copy=copy)
        result = run(source, str(index))
        outcome = {0: "ran, bytes equal", 3: "ran, bytes differ"}.get(result.returncode)
        if outcome is None:
            lines = [line for line in result.stderr.splitlines() if "Error" in line]
            outcome = f"failed: {lines[-1] if lines else result.stderr[-300:]}"
        print(json.dumps({"variant": name, "outcome": outcome}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
