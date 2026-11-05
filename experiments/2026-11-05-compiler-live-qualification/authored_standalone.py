"""Small compiler/kernel checks; this program never imports or starts vLLM."""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import traceback

import torch
import tilelang
import tilelang.language as T

OUT = Path("/out")
RESULTS = []


def check(name, function):
    try:
        details = function()
        RESULTS.append({"name": name, "status": "passed", "details": details})
    except Exception:
        RESULTS.append({"name": name, "status": "failed", "error": traceback.format_exc()})
    print(json.dumps(RESULTS[-1]), flush=True)


def timing(function):
    for _ in range(20):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            function()
    samples = []
    for _ in range(9):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))  # ms / 1000 launches = us / launch
    return {"mean_us": statistics.mean(samples), "ci95_us": 2.306 * statistics.stdev(samples) / 3, "samples_us": samples}


def load_test(pr, relative):
    path = Path(f"/review/variants/pr{pr}") / relative
    spec = importlib.util.spec_from_file_location("review_" + str(pr), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upstream_test(pr, relative, name, disable_ws=False):
    module = load_test(pr, relative)
    original_compile, original_call = tilelang.compile, tilelang.JITKernel.__call__
    captured = []
    last = {}

    def compile_kernel(*args, **kwargs):
        if disable_ws:
            kwargs["pass_configs"] = {**kwargs.get("pass_configs", {}), tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
        kernel = original_compile(*args, **kwargs)
        captured.append(kernel)
        return kernel

    def invoke(self, *args, **kwargs):
        last["call"] = (self, args, kwargs)
        return original_call(self, *args, **kwargs)

    tilelang.compile, tilelang.JITKernel.__call__ = compile_kernel, invoke
    try:
        getattr(module, name)()
    finally:
        tilelang.compile, tilelang.JITKernel.__call__ = original_compile, original_call
        for index, kernel in enumerate(captured):
            (OUT / f"{name}-ws-off-{disable_ws}-{index}.cu").write_text(kernel.get_kernel_source())
    result = {"upstream_test": relative + "::" + name, "compiled_kernels": len(captured), "forced_ws_off": disable_ws}
    if pr == 3661 and "call" in last:
        kernel, args, kwargs = last["call"]
        result["timing"] = timing(lambda: kernel(*args, **kwargs))
    return result


def clamp_kernel(size, dtype, fast):
    if fast:
        @T.prim_func
        def main(X: T.Tensor((size,), dtype), Y: T.Tensor((size,), dtype)):
            with T.Kernel(T.ceildiv(size, 256), threads=128) as bx:
                for j in T.Parallel(256):
                    i = bx * 256 + j
                    if i < size:
                        Y[i] = T.clamp_symmetric(X[i], T.cast(7.0, dtype))
    else:
        @T.prim_func
        def main(X: T.Tensor((size,), dtype), Y: T.Tensor((size,), dtype)):
            with T.Kernel(T.ceildiv(size, 256), threads=128) as bx:
                for j in T.Parallel(256):
                    i = bx * 256 + j
                    if i < size:
                        value = T.if_then_else(X[i] > 7.0, T.cast(7.0, dtype), X[i])
                        Y[i] = T.if_then_else(value < -7.0, T.cast(-7.0, dtype), value)
    return tilelang.compile(main, target="cuda")


def clamp(dtype):
    torch.manual_seed(41)
    x = (torch.randn(65536, device="cuda") * 14).to(getattr(torch, dtype))
    special = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e-38, -1e-38], device="cuda").to(x.dtype)
    x[:len(special)] = special
    if dtype == "float32":
        x.view(torch.int32)[0] = 0x7FC12345
    before = torch.empty_like(x)
    after = torch.empty_like(x)
    old, new = clamp_kernel(x.numel(), dtype, False), clamp_kernel(x.numel(), dtype, True)
    old(x, before)
    new(x, after)
    finite = ~torch.isnan(before)
    integer = torch.int32 if dtype == "float32" else torch.int16
    assert torch.equal(before.view(integer)[finite], after.view(integer)[finite])
    assert torch.equal(torch.isnan(before), torch.isnan(after))
    assert torch.equal(torch.signbit(before[x == 0]), torch.signbit(after[x == 0]))
    payload_difference = int((before.view(integer)[~finite] != after.view(integer)[~finite]).sum().item())
    (OUT / f"clamp-{dtype}.cu").write_text(new.get_kernel_source())
    return {"finite_bits_equal": True, "nan_propagation_equal": True, "signed_zero_equal": True,
            "nan_payload_differences": payload_difference, "compare_select": timing(lambda: old(x, before)),
            "symmetric": timing(lambda: new(x, after))}


def clamp_edges(dtype):
    bits = {
        "float32": [0x7FC12345, 0xFFC54321, 0x7FA12345, 0xFFA12345, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF,
                    0, 0x80000000, 0x7F800000, 0xFF800000, 0x40E00000, 0xC0E00000, 0x41000000, 0xC1000000],
        "float16": [0x7E21, 0xFE21, 0x7D21, 0xFD21, 1, 0x8001, 0x03FF, 0x83FF,
                    0, 0x8000, 0x7C00, 0xFC00, 0x4700, 0xC700, 0x4800, 0xC800],
        "bfloat16": [0x7FC1, 0xFFC1, 0x7FA1, 0xFFA1, 1, 0x8001, 0x007F, 0x807F,
                     0, 0x8000, 0x7F80, 0xFF80, 0x40E0, 0xC0E0, 0x4100, 0xC100],
    }[dtype]
    width = 32 if dtype == "float32" else 16
    integer = torch.int32 if width == 32 else torch.int16
    signed = [b if b < 2 ** (width - 1) else b - 2**width for b in bits]
    x = torch.tensor(signed, dtype=integer, device="cuda").view(getattr(torch, dtype))
    before, after = torch.empty_like(x), torch.empty_like(x)
    clamp_kernel(len(bits), dtype, False)(x, before)
    clamp_kernel(len(bits), dtype, True)(x, after)
    finite = ~torch.isnan(x)
    assert torch.equal(before.view(integer)[finite], after.view(integer)[finite])
    assert torch.equal(torch.isnan(before), torch.isnan(after))
    unsigned = lambda t: [int(v) & (2**width - 1) for v in t.view(integer).cpu().tolist()]
    old_bits, new_bits = unsigned(before), unsigned(after)
    return {"input_bits": bits, "compare_select_bits": old_bits, "symmetric_bits": new_bits,
            "finite_bits_equal": True, "nan_propagation_equal": True,
            "nan_output_bits_equal": old_bits[:4] == new_bits[:4],
            "old_preserves_input_nan_bits": old_bits[:4] == bits[:4],
            "new_preserves_input_nan_bits": new_bits[:4] == bits[:4]}


def reduce_kernel(rows, width, threads, op="sum", dtype="float32"):
    @T.prim_func
    def main(X: T.Tensor((rows, width), dtype), Y: T.Tensor((rows,), dtype)):
        with T.Kernel(rows, threads=threads) as row:
            values = T.alloc_fragment((width,), dtype)
            result = T.alloc_fragment((1,), dtype)
            T.copy(X[row, 0], values)
            if op == "sum":
                T.reduce_sum(values, result, dim=0)
            else:
                T.reduce_max(values, result, dim=0)
            Y[row] = result[0]
    return main


def reduction(width, threads, op):
    torch.manual_seed(17)
    x = torch.randn((64, width), device="cuda")
    y = torch.empty(64, device="cuda")
    kernel = tilelang.compile(reduce_kernel(64, width, threads, op), target="cuda")
    kernel(x, y)
    reference = x.sum(1) if op == "sum" else x.max(1).values
    torch.testing.assert_close(y, reference, rtol=1e-4, atol=1e-5)
    values = y.cpu().tolist()
    source = kernel.get_kernel_source()
    (OUT / f"reduce-{width}-{threads}-{op}.cu").write_text(source)
    return {"values": values, "output_sha256": hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest(),
            "max_abs_error_from_fp64": float((y.double() - (x.double().sum(1) if op == "sum" else x.double().max(1).values)).abs().max()),
            "timing": timing(lambda: kernel(x, y))}


def layout_kernel(layout, disable_tma):
    from tilelang.layout import make_linear_layout, make_swizzled_layout
    from tilelang.cuda.language import copy as cuda_copy

    @T.prim_func
    def main(A: T.Tensor((1024, 1024), T.float16), B: T.Tensor((1024, 1024), T.float16), C: T.Tensor((1024, 1024), T.float16)):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            a = T.alloc_shared((128, 32), T.float16)
            b = T.alloc_shared((32, 128), T.float16)
            acc = T.alloc_fragment((128, 128), T.float32)
            if layout == "linear":
                T.annotate_layout({a: make_linear_layout(a), b: make_linear_layout(b)})
            elif layout == "swizzled":
                T.annotate_layout({a: make_swizzled_layout(a), b: make_swizzled_layout(b)})
            T.clear(acc)
            for k in T.Pipelined(32, num_stages=3):
                cuda_copy(A[by * 128, k * 32], a, disable_tma=disable_tma)
                cuda_copy(B[k * 32, bx * 128], b, disable_tma=disable_tma)
                T.gemm(a, b, acc)
            T.copy(acc, C[by * 128, bx * 128])
    return main


def layout(layout_name, disable_tma):
    torch.manual_seed(29)
    a = torch.randint(-2, 3, (1024, 1024), device="cuda").half()
    b = torch.randint(-2, 3, (1024, 1024), device="cuda").half()
    expected = (a.float() @ b.float()).half()
    c = torch.empty_like(a)
    kernel = tilelang.compile(layout_kernel(layout_name, disable_tma), target="cuda")
    (OUT / f"layout-{layout_name}-{disable_tma}.cu").write_text(kernel.get_kernel_source())
    wrong = []
    for _ in range(3):
        c.fill_(float("nan"))
        kernel(a, b, c)
        wrong.append(int((c != expected).sum().item()))
    assert not any(wrong), f"wrong element counts: {wrong}"
    return {"wrong_elements": wrong, "inputs": "small exact integers, seed 29", "issue": 3571}


def prefetch(policy):
    @T.prim_func
    def main(X: T.Tensor((4096,), T.float32), Y: T.Tensor((4096,), T.float32)):
        with T.Kernel(16, threads=128) as bx:
            if T.get_thread_binding() == 0:
                T.prefetch_l2(X[bx * 256], eviction_policy=policy)
            for j in T.Parallel(256):
                Y[bx * 256 + j] = X[bx * 256 + j]
    kernel = tilelang.compile(main, target="cuda")
    x = torch.arange(4096, device="cuda", dtype=torch.float32)
    saved = x.clone()
    y = torch.empty_like(x)
    kernel(x, y)
    assert torch.equal(x, saved) and torch.equal(x, y)
    kernel.export_ptx(str(OUT / f"prefetch-{policy}.ptx"))
    return {"input_unchanged": True, "output_equal": True, "policy": policy}


def redux(arch, dtype):
    from tilelang.contrib import nvcc
    from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
    target = tilelang.tvm.target.Target({"kind": "cuda", "arch": arch})
    with target:
        result = tilelang.lower(reduce_kernel(4, 32, 32, "max", dtype), target=target, enable_device_compile=False)
    ptx = nvcc.compile_cuda(result.kernel_source, arch=arch,
                            options=["-std=c++17", "-I" + TILELANG_TEMPLATE_PATH, "-I" + CUTLASS_INCLUDE_DIR]).decode()
    (OUT / f"redux-{arch}-{dtype}.ptx").write_text(ptx)
    count = ptx.count("redux.sync")
    if dtype == "float32" and arch == "sm_121a":
        assert count == 0
    elif dtype == "int32" or arch == "sm_103a":
        assert count > 0
    return {"arch": arch, "dtype": dtype, "redux_instructions": count, "shuffle_instructions": ptx.count("shfl.sync")}


def prepare(strict=False):
    from tilelang.cache import freeze, thaw, frozen, KernelNotPreparedError
    if strict:
        os.environ["TILELANG_REQUIRE_EXPLICIT_COMPILE"] = "1"

    @tilelang.jit
    def build(size):
        @T.prim_func
        def main(X: T.Tensor((size,), T.float32), Y: T.Tensor((size,), T.float32)):
            with T.Kernel(T.ceildiv(size, 128), threads=128) as bx:
                for i in T.Parallel(128):
                    if bx * 128 + i < size:
                        Y[bx * 128 + i] = X[bx * 128 + i]
        return main

    progress = []
    prepared = tilelang.prepare([(build, (256,), {}), (build, (), {"size": 256})], num_workers=2,
                                progress=lambda done, total: progress.append([done, total]))
    assert prepared[0] is prepared[1]
    freeze("standalone compatibility check")
    try:
        assert frozen() == "standalone compatibility check"
        kernel = build(256)
        assert kernel is prepared[0]
        x = torch.arange(256, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        kernel(x, y)
        assert torch.equal(x, y)
        try:
            build(257)
        except KernelNotPreparedError as error:
            assert "build" in str(error) and "257" in str(error)
        else:
            raise AssertionError("a new specialization compiled after freeze")
    finally:
        thaw()
    assert frozen() is None
    return {"prepare_deduplicates": True, "prepared_executes_while_frozen": True,
            "missing_specialization_refused_by_name": True, "progress": progress, "strict": strict}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["clamp", "clamp-edges", "reductions", "layout", "prefetch", "redux", "ws3661", "ws3537", "prepare"])
    parser.add_argument("--layout", choices=["inferred", "swizzled", "linear"])
    parser.add_argument("--disable-tma", action="store_true")
    parser.add_argument("--strict-prepare", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"tilelang": tilelang.__file__, "version": tilelang.__version__, "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability()}), flush=True)
    if args.case == "clamp":
        for dtype in ("float32", "float16", "bfloat16"):
            check(dtype, lambda dtype=dtype: clamp(dtype))
    elif args.case == "clamp-edges":
        for dtype in ("float32", "float16", "bfloat16"):
            check(dtype, lambda dtype=dtype: clamp_edges(dtype))
    elif args.case == "reductions":
        for width, threads in ((256, 128), (1024, 256)):
            for op in ("sum", "max"):
                check(f"{width}-{threads}-{op}", lambda width=width, threads=threads, op=op: reduction(width, threads, op))
    elif args.case == "layout":
        for name in ([args.layout] if args.layout else ["inferred", "swizzled", "linear"]):
            for disabled in ([args.disable_tma] if args.layout else [False, True]):
                check(f"{name}-tma-disabled-{disabled}", lambda name=name, disabled=disabled: layout(name, disabled))
    elif args.case == "prefetch":
        for policy in (None, "evict_normal", "evict_last"):
            check(str(policy), lambda policy=policy: prefetch(policy))
    elif args.case == "redux":
        for arch in ("sm_121a", "sm_103a"):
            for dtype in ("float32", "int32"):
                check(f"{arch}-{dtype}", lambda arch=arch, dtype=dtype: redux(arch, dtype))
    elif args.case == "ws3661":
        for disabled in (False, True):
            check(f"ws-disabled-{disabled}", lambda disabled=disabled: upstream_test(3661,
                  "testing/python/issue/test_tilelang_issue_ws_simt_copy_full_producer_extent.py",
                  "test_ws_publishes_synchronous_fp8_conversion", disabled))
    elif args.case == "prepare":
        check("prepare-freeze", lambda: prepare(args.strict_prepare))
    else:
        for name in ("test_ws_generic_stage_reads_fence_before_release", "test_ws_mma_sync_stage_reads_fence_before_release", "test_ws_generic_stage_reads_fast_math_correctness"):
            check(name, lambda name=name: upstream_test(3537, "testing/python/issue/test_tilelang_issue_3529.py", name))
    (OUT / "standalone.json").write_text(json.dumps(RESULTS, indent=2) + "\n")
    raise SystemExit(int(any(x["status"] != "passed" for x in RESULTS)))


if __name__ == "__main__":
    main()
