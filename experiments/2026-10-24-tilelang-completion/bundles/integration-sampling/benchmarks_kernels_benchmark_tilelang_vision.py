# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's vision tower per image: TileLang (``nvidia/tilelang_vision.py``)
against B12X (``nvidia/b12x_vision.py``), at the checkpoint's geometry.

Both towers hold the same random weights (32 blocks of 1024 wide, 16 heads, 2816,
14-pixel patches, a 5120-wide aligner, capacity 9216 patches) and encode the same
images eagerly, as the model's multimodal path does: the ViT and the aligner, one
image a call. B12X is prepared as serving prepares it (its projections' plans for
one row and the capacity, autotuned unless ``--no-b12x-autotune``).

- End to end: at each grid, ``--samples`` fresh images; on each, ``--calls`` images
  of both towers back to back, alternating which goes first; the 95% interval of
  the per-image difference comes from the samples' differences. The outputs are
  compared (relative difference of the aligner's rows) before timing.
- Per op: one image of each tower under the CUDA profiler, device time by kernel
  name and by operation (projections, attention, RoPE, norms and adds, SwiGLU,
  merge, GELU, padding).
- ``--sweep`` (ungated): standalone launches of alternative TileLang tiles for every
  projection, the SwiGLU projection and attention at the largest grid, to pick the
  ``vision.LINEAR_TILE`` / ``SWIGLU_TILE`` / ``ATTENTION_TILE`` defaults.

The gate (``--gate``) is that TileLang is nowhere slower than B12X beyond noise: at
no grid is the interval's lower bound above zero.
"""

import argparse
import functools
import json
import math
import statistics
from collections.abc import Callable
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4_1.nvidia import tilelang_vision
from vllm.models.deepseek_v4_1.tilelang import vision

DEVICE = torch.device("cuda")
# DeepSeek-V4.1's vision_config (checkpoint 1017) and its text hidden size.
CONFIG = dict(
    vision_patch_size=14,
    vision_dim=1024,
    vision_n_heads=16,
    vision_inter_dim=2816,
    vision_n_layers=32,
    vision_rope_theta=10000.0,
    vision_downsample_ratio=3,
    hidden_size=5120,
    vision_max_n_token=1024,
)
# Grids of 14-pixel patches: the minimum-pixel image (544 x 544), a 4:3 photo, and
# 16:9 and square images at the 1024-token cap.
GRIDS = {
    "39x39 minimum pixels": (39, 39),
    "54x72 4:3": (54, 72),
    "72x128 16:9 at the cap": (72, 128),
    "96x96 square at the cap": (96, 96),
}
# Student's t (two-sided 95%) by degrees of freedom.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
_T95.update({8: 2.306, 9: 2.262, 11: 2.201, 15: 2.131, 19: 2.093})

# Device time by operation, from kernel names (TileLang's are the prim_funcs';
# B12X's by substring, best effort; the raw names are printed too).
_OPERATIONS = (
    ("attention", ("vision_attention", "attention", "fmha", "contiguous")),
    ("swiglu projection", ("vision_swiglu_linear",)),
    ("projections", ("vision_linear", "gemv", "gemm", "linear")),
    ("rope", ("vision_rope", "rope")),
    ("norms and adds", ("vision_add_norm", "rmsnorm", "norm", "add")),
    ("swiglu", ("swiglu", "pointwise")),
    ("merge", ("vision_spatial_merge", "merge")),
    ("gelu", ("gelu",)),
    ("padding", ("vision_pad_columns",)),
    ("vision support (b12x rope/merge/gelu)", ("vision",)),
)


def _operation(name: str) -> str:
    lowered = name.lower()
    for operation, keys in _OPERATIONS:
        if any(key.lower() in lowered for key in keys):
            return operation
    return "other"


def _init(module: torch.nn.Module, seed: int) -> None:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            noise = torch.randn(parameter.shape, generator=gen, device=DEVICE)
            if "norm" in name:
                parameter.copy_(1 + 0.1 * noise)
            elif name.endswith("bias"):
                parameter.copy_(0.1 * noise)
            else:
                parameter.copy_(noise / math.sqrt(parameter.shape[1]))


def towers(config: SimpleNamespace, autotune: bool):
    """The TileLang and B12X towers, the same weights, both prepared."""
    from b12x.preparation import PreparationSession

    from vllm.model_executor.warmup.b12x_prepare import _units_from_modules
    from vllm.models.deepseek_v4_1.nvidia import b12x_vision
    from vllm.utils.b12x import B12xWorkload

    with torch.device(DEVICE):
        ours = SimpleNamespace(
            vit=tilelang_vision.DeepseekV4ViT(config),
            aligner=tilelang_vision.DeepseekV4Aligner(config),
        )
        theirs = SimpleNamespace(
            vit=b12x_vision.DeepseekV4ViT(config),
            aligner=b12x_vision.DeepseekV4Aligner(config),
        )
    _init(ours.vit, 411)
    _init(ours.aligner, 412)
    theirs.vit.load_state_dict(ours.vit.state_dict())
    theirs.aligner.load_state_dict(ours.aligner.state_dict())
    ours.vit.declare_kernels()
    ours.aligner.declare_kernels()
    capacity = ours.vit.capacity
    workload = B12xWorkload(
        stage="weights",
        token_counts=(1,),
        fixed_token_counts=(),
        output_dtype=torch.bfloat16,
        max_tokens=capacity,
        max_seqs=1,
        max_model_len=capacity,
    )
    session = PreparationSession(device=DEVICE, autotune=autotune)
    units = [
        unit
        for provider in (theirs.vit, theirs.aligner)
        for unit in _units_from_modules(provider, workload)
    ]
    session.prepare(tuple(request for unit in units for request in unit.requests))
    theirs.session = session
    return ours, theirs


def encode(tower, patches: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return tower.aligner(tower.vit(patches, height, width), height, width)


def per_image_ms(run: Callable[[], object], calls: int) -> float:
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(calls):
        run()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / calls


def interval(differences: list[float]) -> list[float]:
    mean = statistics.fmean(differences)
    if len(differences) < 2:
        return [mean, mean]
    dof = len(differences) - 1
    # The nearest tabulated degrees of freedom at or below: conservative.
    t = _T95[max(k for k in _T95 if k <= dof)]
    half = t * statistics.stdev(differences) / math.sqrt(len(differences))
    return [mean - half, mean + half]


def end_to_end(ours, theirs, config, grid, samples: int, calls: int) -> dict:
    height, width = grid
    rows = height * width
    p = config.vision_patch_size
    ours_ms, theirs_ms, differences, agreement = [], [], [], []
    for sample in range(samples):
        gen = torch.Generator(device=DEVICE).manual_seed(sample)
        patches = torch.randn(rows, 3, p, p, generator=gen, device=DEVICE).bfloat16()
        ours_out = encode(ours, patches, height, width)
        theirs_out = encode(theirs, patches, height, width)
        agreement.append(
            (
                (ours_out.float() - theirs_out.float()).norm()
                / theirs_out.float().norm()
            ).item()
        )

        def run_ours(patches=patches):
            encode(ours, patches, height, width)

        def run_theirs(patches=patches):
            encode(theirs, patches, height, width)

        pair = (run_ours, run_theirs) if sample % 2 == 0 else (run_theirs, run_ours)
        times = {id(run): per_image_ms(run, calls) for run in pair}
        ours_ms.append(times[id(run_ours)])
        theirs_ms.append(times[id(run_theirs)])
        differences.append(times[id(run_ours)] - times[id(run_theirs)])
    return {
        "patches": rows,
        "tilelang_ms": statistics.median(ours_ms),
        "b12x_ms": statistics.median(theirs_ms),
        "difference_ms": statistics.fmean(differences),
        "interval_ms": interval(differences),
        "relative_difference_of_outputs": max(agreement),
    }


def profile(run: Callable[[], object]) -> dict[str, float]:
    """Device milliseconds by kernel name for one call of ``run``."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile

    run()
    torch.cuda.synchronize()
    with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    kernels: dict[str, float] = {}
    for event in prof.key_averages():
        if event.device_type != DeviceType.CUDA:
            continue
        kernels[event.key] = kernels.get(event.key, 0.0) + (
            event.self_device_time_total / 1000
        )
    return kernels


def per_op(ours, theirs, config, grid) -> dict:
    height, width = grid
    p = config.vision_patch_size
    gen = torch.Generator(device=DEVICE).manual_seed(1234)
    patches = torch.randn(
        height * width, 3, p, p, generator=gen, device=DEVICE
    ).bfloat16()
    result = {}
    for name, tower in (("tilelang", ours), ("b12x", theirs)):
        kernels = profile(lambda tower=tower: encode(tower, patches, height, width))
        operations: dict[str, float] = {}
        for kernel, ms in kernels.items():
            operation = _operation(kernel)
            operations[operation] = operations.get(operation, 0.0) + ms
        result[name] = {
            "total_ms": sum(kernels.values()),
            "by_operation_ms": dict(sorted(operations.items(), key=lambda i: -i[1])),
            "by_kernel_ms": dict(
                sorted(kernels.items(), key=lambda item: -item[1])[:24]
            ),
        }
    return result


# --- Tile sweep (ungated) ---------------------------------------------------------

# K tiles stay 64: each is a blocked FP32 partial sum, so they set the bits.
_LINEAR_TILES = [
    dict(block_M=128, block_N=128, num_stages=2, threads=256),
    dict(block_M=128, block_N=128, num_stages=3, threads=256),
    dict(block_M=128, block_N=128, num_stages=2, threads=128),
    dict(block_M=128, block_N=256, num_stages=2, threads=256),
    dict(block_M=256, block_N=128, num_stages=2, threads=256),
    dict(block_M=64, block_N=128, num_stages=3, threads=128),
    dict(block_M=128, block_N=64, num_stages=3, threads=128),
]
_SWIGLU_TILES = [
    dict(block_M=128, block_N=64, num_stages=2, threads=256),
    dict(block_M=128, block_N=64, num_stages=3, threads=256),
    dict(block_M=128, block_N=128, num_stages=2, threads=256),
    dict(block_M=64, block_N=64, num_stages=3, threads=128),
    dict(block_M=128, block_N=32, num_stages=3, threads=128),
]
_ATTENTION_TILES = [
    dict(block_M=128, block_N=64, num_stages=2, threads=128),
    dict(block_M=128, block_N=128, num_stages=1, threads=128),
    dict(block_M=128, block_N=128, num_stages=2, threads=256),
    dict(block_M=128, block_N=64, num_stages=3, threads=128),
    dict(block_M=64, block_N=64, num_stages=2, threads=128),
    dict(block_M=64, block_N=128, num_stages=2, threads=128),
]


def _median_ms(run: Callable[[], object], repeats: int = 10) -> float:
    run()
    times = []
    for _ in range(repeats):
        times.append(per_image_ms(run, 1))
    return statistics.median(times)


def sweep(config: SimpleNamespace, rows: int) -> dict:
    """Each candidate tile of each GEMM role and attention, launched alone on random
    data of ``rows`` patches (the aligner's projections on its merged rows)."""
    import tilelang

    hidden, inter = config.vision_dim, config.vision_inter_dim
    heads, out_dim = config.vision_n_heads, config.hidden_size
    r = config.vision_downsample_ratio
    merged = -(-rows // (r * r))
    patch = vision.padded_width(3 * config.vision_patch_size**2)
    linears = {
        "patch [1024, 640] + bias": (hidden, patch, True, False, rows),
        "qkv [3072, 1024] + bias": (3 * hidden, hidden, True, False, rows),
        "wo [1024, 1024] + bias": (hidden, hidden, True, False, rows),
        "mlp down [1024, 2816]": (hidden, inter, False, False, rows),
        "aligner w1 [5120, 9216] + bias + gelu": (
            out_dim,
            hidden * r * r,
            True,
            True,
            merged,
        ),
        "aligner w2 [5120, 5120] + bias": (out_dim, out_dim, True, False, merged),
    }
    candidates = []
    for name, (n, k, bias, gelu, m) in linears.items():
        for tile in _LINEAR_TILES:
            call = vision.linear_call(n, k, bias=bias, gelu=gelu, **tile)
            candidates.append((name, tile, call, m))
    for tile in _SWIGLU_TILES:
        call = vision.swiglu_call(inter, hidden, **tile)
        candidates.append(("mlp gate/up + swiglu [5632, 1024]", tile, call, rows))
    for tile in _ATTENTION_TILES:
        call = vision.attention_call(heads, hidden // heads, **tile)
        candidates.append(("attention 16 x 64", tile, call, rows))
    try:
        tilelang.prepare([call for _, _, call, _ in candidates])
    except Exception as error:  # noqa: BLE001 - a candidate may not fit; listed below
        print(json.dumps({"sweep compile": repr(error)}), flush=True)

    def tensor(*shape):
        return torch.randn(shape, device=DEVICE).bfloat16()

    results: dict[str, list] = {}
    for name, tile, call, m in candidates:
        function, args, _ = call
        try:
            kernel = vision.resolve(call)
        except Exception as error:  # noqa: BLE001 - a candidate may not compile
            results.setdefault(name, []).append({"tile": tile, "error": repr(error)})
            continue
        if function is vision.attention_kernel:
            tensors = (tensor(m, 3 * hidden), tensor(m, hidden))
        elif function is vision.swiglu_linear_kernel:
            tensors = (tensor(m, hidden), tensor(2 * inter, hidden), tensor(m, inter))
        else:
            n, k, bias = args[0], args[1], args[2]
            bias_tensor = (tensor(n),) if bias else ()
            tensors = (tensor(m, k), tensor(n, k), *bias_tensor, tensor(m, n))
        launch = functools.partial(kernel, *tensors)
        results.setdefault(name, []).append(
            {"tile": tile, "rows": m, "ms": _median_ms(launch)}
        )
    for entries in results.values():
        entries.sort(key=lambda entry: entry.get("ms", math.inf))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument(
        "--layers", type=int, default=CONFIG["vision_n_layers"], help="ViT blocks"
    )
    parser.add_argument(
        "--no-b12x-autotune",
        dest="b12x_autotune",
        action="store_false",
        help="prepare B12X with its default tiles (serving autotunes)",
    )
    parser.add_argument(
        "--grid",
        action="append",
        metavar="HxW",
        help="patch grids to time (default: the representative four)",
    )
    args = parser.parse_args()
    if args.grid:
        grids = {g: tuple(int(v) for v in g.split("x")) for g in args.grid}
    else:
        grids = GRIDS
    config = SimpleNamespace(**{**CONFIG, "vision_n_layers": args.layers})
    ours, theirs = towers(config, args.b12x_autotune)
    results = {"config": CONFIG | {"vision_n_layers": args.layers}, "grids": {}}
    slower = []
    for name, grid in grids.items():
        result = end_to_end(ours, theirs, config, grid, args.samples, args.calls)
        result["per_op"] = per_op(ours, theirs, config, grid)
        results["grids"][name] = result
        if result["interval_ms"][0] > 0:
            slower.append(name)
        print(json.dumps({name: result}, indent=1), flush=True)
    if args.sweep:
        rows = max(h * w for h, w in grids.values())
        results["sweep (ungated)"] = sweep(config, rows)
        print(json.dumps({"sweep": results["sweep (ungated)"]}, indent=1), flush=True)
    theirs.session.close()
    print(
        json.dumps({"summary": "TileLang vs B12X vision tower", "slower": slower}),
        flush=True,
    )
    if args.gate and slower:
        raise SystemExit(f"TileLang vision slower than B12X at {', '.join(slower)}")


if __name__ == "__main__":
    main()
