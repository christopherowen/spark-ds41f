"""Both L2 prefetch families on the same plans: after flushing L2, prefetch a weight on
a side stream, wait for it, then time a full read of the weight. A family that fills L2
the same way gives the same read time. Prints one line per (size, family); exits 1 if the
TileLang read time is more than 3% slower than CuTe's at any size."""
import importlib
import os
import statistics
import sys

import torch

sys.path.insert(0, "/opt/spark3/candidate/vllm")
from vllm.models.glm5next.nvidia import l2_prefetch as l2pf  # noqa: E402
from vllm.models.glm5next.nvidia.l2_prefetch_tilelang import compile_launcher  # noqa: E402

GRID, BLOCK, CHUNK = 2, 128, 4096
dev = torch.device("cuda")
scratch = torch.empty(96 * 1024 * 1024, dtype=torch.uint8, device=dev)


def cute_launcher():
    os.environ["VLLM_L2_PREFETCH_KERNELS"] = "cute"
    importlib.reload(l2pf)
    l2pf._GRID = GRID
    return l2pf._get_launcher()


def read_time(weight, plan, launch, reps=30):
    side = torch.cuda.Stream()
    times = []
    for _ in range(reps):
        scratch.fill_(1)  # evict
        torch.cuda.synchronize()
        if launch is not None:
            side.wait_stream(torch.cuda.current_stream())
            launch(plan.segs, side.cuda_stream)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        weight.sum(dtype=torch.float32)
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) * 1000)
    return statistics.median(times)


families = {"none": None, "cute": cute_launcher(), "tilelang": compile_launcher(GRID, BLOCK, CHUNK)}
worst = 0.0
for mib in (4, 10, 18):
    weight = torch.randn(mib * 1024 * 1024 // 2, dtype=torch.bfloat16, device=dev)
    plan = l2pf.L2PrefetchPlan([l2pf.tensor_segment("w", weight)], dev)
    result = {name: read_time(weight, plan, launch) for name, launch in families.items()}
    ratio = result["tilelang"] / result["cute"]
    worst = max(worst, ratio)
    print(f"{mib:3d} MiB read after prefetch (us): none {result['none']:.1f}  cute {result['cute']:.1f}  "
          f"tilelang {result['tilelang']:.1f}  tilelang/cute {ratio:.3f}", flush=True)
sys.exit(0 if worst <= 1.03 else 1)
