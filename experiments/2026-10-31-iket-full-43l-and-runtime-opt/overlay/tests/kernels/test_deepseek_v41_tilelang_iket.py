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

import vllm
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
    # A deferred-blocking barrier holds a warp only at its next memory access:
    # only ranges containing a block barrier (a block sync or tl::AllReduce) load
    # shared memory before reading the timer.
    out, sites = _ranges(KERNEL)
    lines = [line.strip() for line in out.splitlines()]
    ends = [i for i, line in enumerate(lines) if line == "TL_IKET_EVENT(31);"]
    assert len(ends) == len(sites)
    settled_ranges = [name for name, site in sites.items() if site["settled"]]
    assert settled_ranges == ["wait.sync", "wait.sync#2", "wait.sync#3"]
    assert [lines[i - 1] == "TL_IKET_SETTLE();" for i in ends] == [
        site["settled"] for site in sites.values()
    ]
    assert out.startswith("#define TL_IKET_SETTLE() asm volatile(")

    # tl::AllReduce contains a block barrier in its C++ template, so its compute
    # range is also settled, whereas warp-only shuffles and barrier-free kernels
    # emit no settle macro at all.
    allreduce_code = (
        'extern "C" __global__ void ar_kernel(float* C) {\n'
        "  float v = tl::AllReduce<tl::SumOp, 128, 1>::run(C[threadIdx.x], red_buf);\n"
        "  C[threadIdx.x] = v;\n"
        "}\n"
    )
    ar_out, ar_sites = _ranges(allreduce_code)
    assert ar_sites["compute.reduce"]["settled"] is True
    assert ar_sites["store.global"]["settled"] is False
    assert ar_out.count("TL_IKET_SETTLE();") == 1

    no_barrier_code = (
        'extern "C" __global__ void copy_kernel(const float* A, float* C) {\n'
        "  float x = A[threadIdx.x];\n"
        "  C[threadIdx.x] = x;\n"
        "}\n"
    )
    nb_out, nb_sites = _ranges(no_barrier_code)
    assert not any(site["settled"] for site in nb_sites.values())
    assert "TL_IKET_SETTLE" not in nb_out

    # When a sampled loop ends with a block barrier, the first range after the loop
    # also settles before opening so the loop's final unsampled iteration's barrier
    # does not leak into the post-loop range's start timestamp.
    loop_tail_sync_code = (
        'extern "C" __global__ void tail_sync_kernel(const float* A, float* C) {\n'
        "  for (int n = 0; n < 4; ++n) {\n"
        "    tl::mma_sync(acc, a, b);\n"
        "    __syncthreads();\n"
        "  }\n"
        "  C[threadIdx.x] = acc[0];\n"
        "}\n"
    )
    lt_out, _ = iket.annotate(loop_tail_sync_code, _token, ctas=1, iters=1)
    lt_lines = [line.strip() for line in lt_out.splitlines()]
    store_start = next(
        i
        for i, line in enumerate(lt_lines)
        if "__tl_iket_v1_store.global" in line
    )
    assert lt_lines[store_start - 1] == "if (__tl_iket_cta) TL_IKET_SETTLE();"


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


def test_sampled_warps_and_payloads():
    out, sites = iket.annotate(KERNEL, _token, ctas=4, warps=2, payloads=True)
    lines = out.splitlines()
    body = next(
        i for i, line in enumerate(lines) if line.endswith("{") and "k_kernel" in line
    )
    assert lines[body + 1].startswith("  const bool __tl_iket_cta = ")
    assert lines[body + 2].startswith("  const bool __tl_iket_warp = ")
    assert "((blockDim.x * blockDim.y * blockDim.z + 31u) >> 5) / 2u" in lines[body + 2]
    starts = [line.strip() for line in lines if "TL_IKET_EVENT_PAYLOAD_U32(" in line]
    assert len(starts) == len(sites)
    assert all(
        s.startswith("TL_IKET_WHEN(__tl_iket_cta && __tl_iket_warp) ") for s in starts
    )
    # Loop ranges attach the innermost loop variable; outer ranges attach the CTA index.
    by_range = {s["range"]: s for s in sites}
    assert by_range["compute.mma"]["payload_expr"] == "ko"
    assert (
        by_range["load.global"]["payload_expr"]
        == "((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x)"
    )
    assert any("TL_IKET_EVENT_PAYLOAD_U32(1, ko," in s for s in starts)
    # Static load byte extraction records cp_async and TMA transaction sizes.
    assert by_range["load.cp_async"]["cp_async_bytes_per_thread"] == 16
    tma_code = (
        'extern "C" __global__ void tma_k(const float* A) {\n'
        "  loaded[0].expect_transaction(8192);\n"
        "  tl::tma_load(B_desc, loaded[0], smem, 0, 0, 0);\n"
        "}\n"
    )
    _, tma_sites = iket.annotate(tma_code, _token)
    assert tma_sites[0]["tma_bytes"] == 8192


def test_sampled_iterations_guard_loop_ranges():
    out, sites = iket.annotate(KERNEL, _token, iters=2)
    # Iterations 1 and 3 of four: the middle of each half.
    block = "max(1, (int)(((4) - (0) + 1) / 2))"
    stride = f"((int)((ko) - (0)) % {block} == {block} / 2)"
    lines = [line.strip() for line in out.splitlines()]
    in_loop = [s["range"] for s in sites if s["loops"]]
    # Each name carries the factor from its two sampled iterations to the loop's four.
    assert in_loop == ["wait.cp_async*2", "wait.sync#2*2", "compute.mma*2"]
    # The loop's trips scale its sampled iterations; a site outside it has none.
    assert {s["loop_trips"] for s in sites if s["loops"]} == {4}
    assert not any("loop_trips" in s for s in sites if not s["loops"])
    # Each in-loop range: its start and end events, and its settle on barrier ranges.
    events = [line for line in lines if line.startswith(f"TL_IKET_WHEN({stride}) ")]
    assert len(events) == 2 * len(in_loop)
    settles = [line for line in lines if line == f"if ({stride}) TL_IKET_SETTLE();"]
    assert len(settles) == sum(s["settled"] for s in sites if s["loops"]) == 1
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
    assert [s["loop_trips"] for s in sites if "unsampled_loops" in s] == [None]
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


def test_phase_granularity_merges_waits_and_internal_sync_blocks():
    fine_out, fine_sites = iket.annotate(KERNEL, _token, ctas=1, iters=1)
    phase_out, phase_sites = iket.annotate(
        KERNEL, _token, ctas=1, iters=1, granularity="phase"
    )
    assert [s["range"] for s in fine_sites] == [
        "load.global",
        "wait.sync",
        "load.cp_async",
        "wait.cp_async*4",
        "wait.sync#2*4",
        "compute.mma*4",
        "wait.sync#3",
        "store.global",
    ]
    assert [s["range"] for s in phase_sites] == [
        "load.global",
        "wait.sync",
        "load.cp_async",
        "wait*4",
        "compute.mma*4",
        "wait.sync#2",
        "store.global",
    ]
    assert phase_sites[3]["kinds"] == ["cp_async", "sync"]
    assert phase_sites[0]["events_per_warp_bound"] < fine_sites[0]["events_per_warp_bound"]

    mla_like = (
        'extern "C" __global__ void mla_kernel(const float* A, float* C) {\n'
        "  float q = A[threadIdx.x];\n"
        "  __syncthreads();\n"
        "  __syncthreads();\n"
        "  float k = A[threadIdx.x + 32];\n"
        "  for (int c = 0; c < 4; ++c) {\n"
        "    tl::cp_async_wait<0>();\n"
        "    __syncthreads();\n"
        "    tl::mma_sync(acc, a, b);\n"
        "    for (int i = 0; i < 2; ++i) {\n"
        "      __syncthreads();\n"
        "      m[i] = tl::AllReduce<tl::MaxOp, 256, 64>::run(m[i], ws);\n"
        "    }\n"
        "    __syncthreads();\n"
        "    tl::mma_sync(acc_o, p, v);\n"
        "  }\n"
        "  C[threadIdx.x] = acc_o[0];\n"
        "}\n"
    )
    _, mla_fine = iket.annotate(mla_like, _token, ctas=1, iters=1, granularity="fine")
    _, mla_phase = iket.annotate(mla_like, _token, ctas=1, iters=1, granularity="phase")
    assert len(mla_fine) == 12
    assert [s["range"] for s in mla_phase] == [
        "load.global",
        "wait*4",
        "compute*4",
        "store.global",
    ]
    assert mla_phase[0]["events_per_warp_bound"] == 8

    with pytest.raises(ValueError, match="unknown IKET granularity"):
        iket.annotate(KERNEL, _token, granularity="invalid")


def test_cache_digest_isolates_sampling_settings():
    d_all = iket._cache_digest((0, 0, "", "fine", 0, False, "/tmp/c"))
    d_sampled = iket._cache_digest((1, 1, "", "fine", 0, False, "/tmp/c"))
    d_phase = iket._cache_digest((1, 1, "", "phase", 0, False, "/tmp/c"))
    d_warps = iket._cache_digest((1, 1, "", "phase", 1, True, "/tmp/c"))
    assert len({d_all, d_sampled, d_phase, d_warps}) == 4
    assert d_sampled == iket._cache_digest((1, 1, "", "fine", 0, False, "/tmp/other"))


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
    cache_arg = sys.argv[3] if len(sys.argv) > 3 else None
    iket.enable(sys.argv[2], ctas=int(sys.argv[4]) if len(sys.argv) > 4 else 0, cache_dir=cache_arg)

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


def _env(cache_dir):
    """The subprocess imports the vLLM under test: a script's own directory, not the
    working directory, starts its path, and an installed vLLM may lack DS4.1."""
    source_root = os.path.dirname(os.path.dirname(os.path.abspath(vllm.__file__)))
    path = os.pathsep.join(p for p in (source_root, os.environ.get("PYTHONPATH")) if p)
    return dict(os.environ, PYTHONPATH=path, TILELANG_CACHE_DIR=str(cache_dir))


def _run(tmp_path, *args):
    env = _env(tmp_path / "deployment-cache")
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
def test_persistent_cache_dir_hits_and_mirrors_sources(tmp_path):
    shared_cache = tmp_path / "iket-cache"
    run1 = _run(tmp_path, "on", str(tmp_path / "iket-run1"), str(shared_cache), "1")
    assert run1["cache_enabled"] and run1["source_instrumented"]
    run2 = _run(tmp_path, "on", str(tmp_path / "iket-run2"), str(shared_cache), "1")
    assert run2["cache_enabled"] and run2["bits"] == run1["bits"]
    assert run2["cache_dir"] == run1["cache_dir"]
    manifests = list(
        (tmp_path / "iket-run2" / "sources").glob("iket_test_scale_add_kernel.*.json")
    )
    assert len(manifests) == 1
    run3 = _run(tmp_path, "on", str(tmp_path / "iket-run3"), str(shared_cache), "4")
    assert run3["cache_dir"] != run1["cache_dir"]


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
    env = _env(tmp_path / "cache")
    process = subprocess.run(
        [sys.executable, _script(tmp_path, script), str(tmp_path / "iket")],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert process.returncode == 0, process.stderr[-4000:]
    assert "refused: IKET mode must be on before TileLang compiles" in process.stdout
