# Shared helpers for candidate-51's own-mark harnesses (sp8, one GB10).
# GPU warm-up, eager loop, CUDA-graph replay (L2-hot back-to-back, and L2-cold single
# call), CUPTI kernel time, output hashes.
import hashlib
import os
import statistics
import sys
import time

import torch


def gpu_warmup(min_ms: float = 250.0) -> float:
    """Dense BF16 matmuls until at least ``min_ms`` of GPU time has elapsed."""
    a = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    total = 0.0
    while total < min_ms:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(8):
            a @ b
        e.record()
        torch.cuda.synchronize()
        total += s.elapsed_time(e)
    del a, b
    torch.cuda.empty_cache()
    print(f"[warm] {total:.1f} ms of GPU warm-up", flush=True)
    return round(total, 1)


def eager_us(fn, warmup=25, rep=150):
    """The PR scripts' bench(): back-to-back eager launches, mean per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return round(s.elapsed_time(e) * 1000.0 / rep, 2)


def _capture(fn, calls):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()
    return g


def _stats(samples):
    s = sorted(samples)
    return {
        "median_us": round(statistics.median(s), 2),
        "min_us": round(s[0], 2),
        "mean_us": round(statistics.mean(s), 2),
        "n": len(s),
    }


def graph_hot_us(fn, calls=20, warm=10, replays=30):
    """A captured graph of ``calls`` back-to-back calls, replayed; per call (L2-hot)."""
    g = _capture(fn, calls)
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(replays):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) * 1000.0 / calls)
    del g
    return _stats(samples)


_FLUSH = None


def graph_cold_us(fn, replays=30, warm=5):
    """A captured graph of one call, replayed after a 256 MiB write evicts L2."""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    g = _capture(fn, 1)
    for _ in range(warm):
        _FLUSH.zero_()
        g.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(replays):
        _FLUSH.zero_()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) * 1000.0)
    del g
    return _stats(samples)


def kernel_us(fn, calls=30, warm=5):
    """CUPTI device time of every kernel ``fn`` launches, summed, per call (eager)."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    names = {}
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0.0)
        if t and ev.count:
            names[ev.key[:60]] = ev.count
            total += t
    return {"per_call_us": round(total / calls, 2), "kernels": names}


def kernel_isolated_us(fn, calls=20, warm=3):
    """CUPTI device time of every kernel ``fn`` launches, summed, per call, with a
    synchronize after each call: no PDL overlap with a neighbouring launch is counted
    (kernel_us's back-to-back loop counts a PDL kernel's early-launch wait)."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
            torch.cuda.synchronize()
    total = 0.0
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0.0)
        if t and ev.count:
            total += t
    return round(total / calls, 2)


def h(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def env_info():
    return {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host_env_grid": os.environ.get("VLLM_L2_PREFETCH_GRID"),
        "argv": sys.argv,
    }
