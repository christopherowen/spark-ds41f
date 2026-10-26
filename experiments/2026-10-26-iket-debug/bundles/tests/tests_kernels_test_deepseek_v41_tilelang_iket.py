# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DS4.1 IKET debug mode: the phase pass on CUDA source, and the switch.

The pass tests need no GPU. The mode's tests run in subprocesses, because the mode
is process-wide: on, a kernel computes the same bits and is never cached where the
deployment's kernels are; off, nothing is imported and no callback is registered.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from vllm.models.deepseek_v4_1.tilelang import iket


def _token(name, range_id):
    return f"__tl_iket_v1_{name}"


KERNEL = """\
#include <tl_templates/cuda/copy.h>
extern "C" __global__ void k_kernel(const float* A, float* C);
extern "C" __global__ void __launch_bounds__(128) k_kernel(const float* A, float* C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  int e = ((int*)A)[blockIdx.x];
  if (e < 0) {
    return;
  }
  __syncthreads();
  #pragma unroll
  for (int s = 0; s < 2; ++s) {
    tl::cp_async_gs<16>(buf_dyn_shmem + s * 16, A + s * 4);
    tl::cp_async_commit();
  }
  for (int ko = 0; ko < 4; ++ko) {
    tl::cp_async_wait<1>();
    __syncthreads();
    tl::mma_sync(acc, a, b);
    float t = acc[0] * 2.0f;
    tl::mma_sync(acc, a, b);
  }
  if (threadIdx.x < 32) {
    __syncthreads();
    acc[1] = 0.0f;
  }
  C[threadIdx.x] = acc[0];
  C[threadIdx.x + 128] += acc[1];
}
"""


def _ranges(code):
    out, sites = iket.annotate(code, _token)
    return out, {s["range"]: s for s in sites}


def test_phases_and_runs():
    out, sites = _ranges(KERNEL)
    assert list(sites) == [
        "load.global",
        "wait.sync",
        "load.cp_async",
        "wait.cp_async",
        "wait.sync#2",
        "compute.mma",
        "wait.sync#3",
        "store.global",
    ]
    # The run of MMAs keeps the phaseless statement between them.
    assert sites["compute.mma"]["last_line"] - sites["compute.mma"]["first_line"] == 2
    # A compound global write is a store, as a plain one is.
    assert sites["store.global"]["last_line"] - sites["store.global"]["first_line"] == 1
    assert out.count("TL_IKET_EVENT(31);") == len(sites)


def test_ranges_close_on_every_path():
    out, sites = _ranges(KERNEL)
    lines = out.splitlines()
    depth = 0
    for line in lines:
        if "TL_IKET_EVENT(1," in line:
            depth += 1
        elif "TL_IKET_EVENT(31)" in line:
            depth -= 1
        # No range is open at a return; none nests in another.
        assert depth in (0, 1)
        if line.strip() == "return;":
            assert depth == 0
    assert depth == 0
    # The wait inside the warp-0 branch is its statement alone, inside the branch.
    branch = lines.index("  if (threadIdx.x < 32) {")
    assert (
        lines[branch + 1]
        .strip()
        .startswith('TL_IKET_EVENT(1, "__tl_iket_v1_wait.sync#3")')
    )


def test_range_ends_wait_for_barriers():
    # A deferred-blocking barrier holds a warp only at its next memory access: each
    # range end loads shared memory before it reads the timer.
    out, sites = _ranges(KERNEL)
    lines = [line.strip() for line in out.splitlines()]
    ends = [i for i, line in enumerate(lines) if line == "TL_IKET_EVENT(31);"]
    assert len(ends) == len(sites)
    assert all(lines[i - 1] == "TL_IKET_SETTLE();" for i in ends)
    assert out.startswith("#define TL_IKET_SETTLE() asm volatile(")


def test_pragma_stays_with_its_loop():
    out, _ = _ranges(KERNEL)
    lines = out.splitlines()
    pragma = lines.index("  #pragma unroll")
    assert (
        lines[pragma - 1].strip() == 'TL_IKET_EVENT(1, "__tl_iket_v1_load.cp_async");'
    )
    assert lines[pragma + 1].startswith("  for (int s = 0;")


def test_sampled_ctas_guard_every_event():
    out, sites = iket.annotate(KERNEL, _token, ctas=8)
    lines = out.splitlines()
    body = next(
        i for i, line in enumerate(lines) if line.endswith("{") and "k_kernel" in line
    )
    assert lines[body + 1].startswith("  const bool __tl_iket_cta = ")
    assert "gridDim.x * gridDim.y * gridDim.z / 8u" in lines[body + 1]
    events = [line.strip() for line in lines if "TL_IKET_EVENT(" in line]
    assert len(events) == 2 * len(sites)
    # Each in a loop of zero or one iterations, which ptxas does not predicate.
    assert all(
        e.startswith("TL_IKET_WHEN(__tl_iket_cta) TL_IKET_EVENT(") for e in events
    )
    assert "#define TL_IKET_WHEN(C) " in out
    # Unsampled, the kernel declares nothing and its events need no loop.
    unsampled = iket.annotate(KERNEL, _token)[0]
    assert "__tl_iket_cta" not in unsampled
    assert "TL_IKET_WHEN" not in unsampled


def test_sampled_iterations_guard_loop_ranges():
    out, sites = iket.annotate(KERNEL, _token, iters=2)
    # Iterations 1 and 3 of four: the middle of each half.
    block = "max(1, (int)(((4) - (0) + 1) / 2))"
    stride = f"((int)((ko) - (0)) % {block} == {block} / 2)"
    lines = [line.strip() for line in out.splitlines()]
    in_loop = [s["range"] for s in sites if s["loops"]]
    assert in_loop == ["wait.cp_async", "wait.sync#2", "compute.mma"]
    # Each in-loop range: its start and end events, and its settle.
    events = [line for line in lines if line.startswith(f"TL_IKET_WHEN({stride}) ")]
    assert len(events) == 2 * len(in_loop)
    settles = [line for line in lines if line == f"if ({stride}) TL_IKET_SETTLE();"]
    assert len(settles) == len(in_loop)
    # Ranges outside the loop, and the unrolled loop wrapped whole, have no condition;
    # none depends on the lane.
    assert 'TL_IKET_EVENT(1, "__tl_iket_v1_load.cp_async");' in lines
    assert "laneid" not in out
    both = iket.annotate(KERNEL, _token, ctas=4, iters=2)[0]
    assert f"TL_IKET_WHEN(__tl_iket_cta && {stride}) TL_IKET_EVENT(1," in both


def test_while_loop_ranges_have_no_bound():
    # TVM's while loop has no variable to sample: a range in it records in every
    # iteration, so the kernel's bound on a warp's events is None.
    code = KERNEL.replace(
        "  C[threadIdx.x] = acc[0];\n",
        "  int t = 0;\n  while (1) {\n    if (!((t < e))) { break; }\n"
        "    C[t] = acc[0];\n    t = (t + 1);\n  }\n  C[threadIdx.x] = acc[0];\n",
    )
    _, sites = iket.annotate(code, _token, iters=1)
    assert [s["unsampled_loops"] for s in sites if "unsampled_loops" in s] == [
        ["while (1)"]
    ]
    assert {s["events_per_warp_bound"] for s in sites} == {None}
    # Without it, each range opens and closes once in the sampled iteration.
    sites = iket.annotate(KERNEL, _token, iters=1)[1]
    assert {s["events_per_warp_bound"] for s in sites} == {2 * len(sites)}


def test_module_without_kernels_is_unchanged():
    code = (
        "#include <tl_templates/cuda/copy.h>\n__device__ int f(int x) { return x; }\n"
    )
    assert iket.annotate(code, _token) == (code, [])


def test_shared_lines_are_not_range_ends():
    code = KERNEL.replace(
        "  __syncthreads();\n  #pragma", "  __syncthreads(); // a comment\n  #pragma"
    )
    _, sites = _ranges(code)
    # The commented sync shares its line with the comment: untraced, not split.
    assert [s["statement"] for s in sites.values()].count("__syncthreads();") == 2


def test_names_fold_past_thirty():
    body = "".join(f"  __syncthreads();\n  C[{i}] = A[{i}];\n" for i in range(20))
    code = (
        'extern "C" __global__ void __launch_bounds__(32, 1) many_kernel('
        "const float* __restrict__ A, float* __restrict__ C) {\n" + body + "}\n"
    )
    out, sites = iket.annotate(code, _token)
    # 40 sites would take 40 names; their first names take two.
    assert len(sites) == 40 and {s["range"] for s in sites} == {
        "wait.sync",
        "store.global",
    }
    assert all(s["traced"] for s in sites)
    assert out.count("TL_IKET_EVENT(31);") == 40


def test_preprocessor_conditional_skips_the_kernel():
    code = KERNEL.replace(
        "  __syncthreads();\n  #pragma", "#if 1\n  __syncthreads();\n#endif\n  #pragma"
    )
    out, sites = iket.annotate(code, _token)
    assert out == code and sites == [
        {"kernel": "k_kernel", "skipped": "a preprocessor conditional in its body"}
    ]


# ---------------------------------------------------------------- the mode

GPU = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="the IKET mode compiles CUDA kernels",
)

KERNEL_SCRIPT = """
import json, os, sys
import torch
import tilelang
import tilelang.language as T

ON = sys.argv[1] == "on"
if ON:
    from vllm.models.deepseek_v4_1.tilelang import iket
    iket.enable(sys.argv[2])

@tilelang.jit
def scale_add(n, threads=128):
    @T.prim_func
    def iket_test_scale_add(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32),
                            C: T.Tensor((n,), T.float32)):
        with T.Kernel(T.ceildiv(n, threads), threads=threads) as bx:
            S = T.alloc_shared((threads,), T.float32)
            for i in T.Parallel(threads):
                S[i] = A[bx * threads + i]
            for i in T.Parallel(threads):
                C[bx * threads + i] = S[(i + 1) % threads] * 2.0 + B[bx * threads + i]
    return iket_test_scale_add

from vllm.models.deepseek_v4_1.tilelang import prepare
prepare.declare(lambda: [prepare.call(scale_add, 1024)])
prepare.freeze("test")
kernel = scale_add(1024)
a = torch.arange(1024, device="cuda", dtype=torch.float32)
b = torch.ones(1024, device="cuda")
c = torch.empty_like(a)
kernel(a, b, c)
torch.cuda.synchronize()
import hashlib, tvm_ffi
from tilelang.env import env as tilelang_env
callback = "tilelang_callback_cuda_postproc"
print(json.dumps({
    "sum": c.double().sum().item(),
    "bits": hashlib.sha256(c.cpu().numpy().tobytes()).hexdigest(),
    "source_instrumented": "__iket_meta_info" in kernel.get_kernel_source(),
    # The modules the mode adds; CuTe DSL's own experimental IKET loads with vLLM.
    "modules": sorted(m for m in sys.modules if m.startswith("tilelang.tools.cuda")
                      or m == "vllm.models.deepseek_v4_1.tilelang.iket"),
    "callback": tvm_ffi.get_global_func(callback, allow_missing=True) is not None,
    "cache_enabled": tilelang_env.is_cache_enabled(),
    "cache_dir": os.environ.get("TILELANG_CACHE_DIR"),
}))
"""


def _script(tmp_path, source):
    """A script file: TileLang reads a kernel's Python source, which ``-c`` lacks."""
    path = tmp_path / f"script_{len(list(tmp_path.glob('script_*.py')))}.py"
    path.write_text(source)
    return str(path)


def _run(tmp_path, *args):
    env = dict(os.environ, TILELANG_CACHE_DIR=str(tmp_path / "deployment-cache"))
    process = subprocess.run(
        [sys.executable, _script(tmp_path, KERNEL_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert process.returncode == 0, process.stderr[-4000:]
    return json.loads(process.stdout.strip().splitlines()[-1])


@GPU
def test_on_matches_off_and_keeps_its_own_cache(tmp_path):
    off = _run(tmp_path, "off")
    cache = tmp_path / "deployment-cache"
    before = sorted(str(p.relative_to(cache)) for p in cache.rglob("*"))
    on = _run(tmp_path, "on", str(tmp_path / "iket"))
    # The on run neither read nor wrote the deployment's cache.
    assert (
        before and sorted(str(p.relative_to(cache)) for p in cache.rglob("*")) == before
    )
    assert off["bits"] == on["bits"] and off["sum"] == on["sum"]
    assert not off["source_instrumented"] and on["source_instrumented"]
    # Off imports nothing of IKET and registers no callback.
    assert off["modules"] == [] and not off["callback"] and off["cache_enabled"]
    assert on["callback"] and not on["cache_enabled"]
    assert on["cache_dir"] == str(tmp_path / "iket" / "tilelang-cache")
    # The instrumented kernel's source and sites, and nothing in the deployment's cache
    # from the on run (the off run's kernel is the only one there).
    manifests = list(
        (tmp_path / "iket" / "sources").glob("iket_test_scale_add_kernel.*.json")
    )
    assert len(manifests) == 1
    phases = {s["phase"] for s in json.loads(manifests[0].read_text())["sites"]}
    assert {"load", "wait", "store"} <= phases
    assert not (tmp_path / "iket" / "tilelang-cache").exists()


@GPU
def test_enable_refuses_after_a_compile(tmp_path):
    script = textwrap.dedent("""
        import sys, tilelang, tilelang.language as T
        @tilelang.jit
        def k(n):
            @T.prim_func
            def iket_test_early(A: T.Tensor((n,), T.float32)):
                with T.Kernel(1, threads=32):
                    A[0] = A[0] + 1.0
            return iket_test_early
        k(32)
        from vllm.models.deepseek_v4_1.tilelang import iket
        try:
            iket.enable(sys.argv[1])
        except RuntimeError as error:
            print("refused:", error)
    """)
    env = dict(os.environ, TILELANG_CACHE_DIR=str(tmp_path / "cache"))
    process = subprocess.run(
        [sys.executable, _script(tmp_path, script), str(tmp_path / "iket")],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert process.returncode == 0, process.stderr[-4000:]
    assert "refused: IKET mode must be on before TileLang compiles" in process.stdout
