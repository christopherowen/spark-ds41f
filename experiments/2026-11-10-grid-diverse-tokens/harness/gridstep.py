"""The engine grid's 4K decode steps on one GB10 at TP4 rank-0 shapes, timed with the grid's
inputs (padding rows) or with servebench's grid version 2 token ids (real rows).

usage: gridstep.py --out FILE --inputs grid|sample|token0 [--shapes 1x2,...] [--replays 30]
                   [--profile] [--profile-replays 3]

The engine is step957.py's (40 target layers and the DSpark drafter's 3, one TP4 rank's
shapes, dummy weights, the recipe's environment), with the recipe's TP4 capture ladder
(16 streams x 6 rows: 1-4, 6, 8, every 4th to 32, every 8th to 96) so every grid shape
replays a full CUDA graph, as on the testbed. One engine_grid call times every shape as
servebench does: 2 warm-ups a shape, then ``replays`` rounds of alternating order, each
timed run after one settle run. Inputs:
- grid: no token ids (every row token 0 and padding: today's grid);
- sample: grid version 2's ids for each shape (servebench.engine.with_token_ids);
- token0: token 0 in every row, as real rows.

With ``--profile`` (under nsys --capture-range=cudaProfilerApi), each shape then runs once
more in its own NVTX range ("shape RxT": 1 warm-up, ``--profile-replays`` settle and timed
pairs), for the kernel split.

With ``--routes FILE`` (``--inputs sample`` only), the 40 target layers' routers keep their
kernels but each routed row's six expert ids are replaced, inside the captured graphs, by
the ids the checkpoint's own gates gave that row after its real context (ref_routes.py, via
routes_table.py): real text's expert spread on dummy weights. The ids come from a persistent
[40, rows, 6] buffer filled before each shape, so each shape is timed in its own engine_grid
call (2 warm-ups, ``replays`` settle and timed pairs), not in one alternating call.
"""

import argparse
import functools
import json
import os
import sys
import time

import step957 as s957

GRID_SHAPES = "1x2,4x8,8x16,16x32,1x3,4x12,8x24,16x48,1x6,2x12,4x24,8x48,12x72,16x96"


def ladder(max_size):
    sizes = [1, 2, 3, 4, 6, 8, *range(12, 33, 4), *range(40, max_size + 1, 8)]
    return [*(size for size in sizes if size < max_size), max_size]


ROUTES = {"buf": None, "made": 0, "prefixes": []}


def patch_routes(max_rows):
    """Target routers 0-39 (the first 40 built) return the buffer's ids for routed rows."""
    import torch
    from vllm.models.deepseek_v4_1.tilelang import router as R

    init, compute = R.TileLangGateRouter.__init__, R.TileLangGateRouter._compute_routing

    @functools.wraps(init)
    def wrapped_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        layer = ROUTES["made"]
        ROUTES["made"] += 1
        ROUTES["prefixes"].append(getattr(self.gate, "prefix", None))
        self._gf_layer = layer if layer < 40 else None

    @functools.wraps(compute)
    def wrapped_compute(self, *args, **kwargs):
        weights, ids = compute(self, *args, **kwargs)
        layer = getattr(self, "_gf_layer", None)
        if layer is None:
            return weights, ids
        if ROUTES["buf"] is None:
            ROUTES["buf"] = torch.full((40, max_rows, 6), -1, dtype=torch.int64, device=ids.device)
        rows = ids.shape[0]
        return weights, torch.where(ids >= 0, ROUTES["buf"][layer, :rows], ids)

    R.TileLangGateRouter.__init__ = wrapped_init
    R.TileLangGateRouter._compute_routing = wrapped_compute


def shapes_for(inputs, shapes, context):
    from servebench import engine as sb_engine

    out = []
    for r, t in shapes:
        shape = {"num_reqs": r, "num_tokens": t, "context_len": context}
        if inputs == "token0":
            shape["token_ids"] = [0] * t
        elif inputs == "sample":
            shape = sb_engine.with_token_ids([shape], 2)[0]
        out.append(shape)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", choices=("grid", "sample", "token0"), required=True)
    ap.add_argument("--shapes", default=GRID_SHAPES)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-replays", type=int, default=3)
    ap.add_argument("--routes", help="per-shape expert ids of real text (routes_table.py)")
    args = ap.parse_args()
    if args.routes and args.inputs != "sample":
        ap.error("--routes needs --inputs sample")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch

    torch.cuda.init()
    from omlib import gpu_warmup

    s957.zero_integer_dummies()
    s957.patch_compressor_tp(4)
    table = None
    if args.routes:
        with open(args.routes) as fh:
            table = json.load(fh)
        patch_routes(512)
    from servebench import engine as sb_engine
    from vllm import LLM

    overrides = {**s957.FULL, **s957.TP_SHAPES[4]}
    pairs = s957.parse_shapes(args.shapes)
    capture = ladder(96)
    started = time.monotonic()
    llm = LLM(
        model=s957.MODEL, tokenizer_mode="deepseek_v41", load_format="dummy", dtype="bfloat16", seed=0,
        tensor_parallel_size=1, block_size=256, max_model_len=1048576, max_num_seqs=16,
        max_num_batched_tokens=512, long_prefill_token_threshold=256, max_parallel_prefills=1,
        async_scheduling=True, kv_cache_memory_bytes=6 << 30, gpu_memory_utilization=0.85,
        enable_prefix_caching=True, attention_backend="TILELANG", linear_backend="tilelang",
        moe_backend="tilelang",
        speculative_config={
            "method": "dspark", "num_speculative_tokens": 5, "draft_tensor_parallel_size": 1,
            "attention_backend": "TILELANG", "draft_sample_method": "greedy",
            "rejection_sample_method": "block", "enable_adaptive_verification": True,
            "adaptive_verification_row_cost": 0.035,
        },
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"],
                            "cudagraph_capture_sizes": capture},
        generation_config="vllm", hf_overrides=functools.partial(s957.shrink, overrides=overrides),
        jit_monitor_mode="off", limit_mm_per_prompt={"image": 1}, mm_processor_cache_gb=0,
    )
    boot_s = time.monotonic() - started
    print(f"engine ready in {boot_s:.0f} s", flush=True)
    warm_ms = gpu_warmup()
    shapes = shapes_for(args.inputs, pairs, args.context)
    spec = {"shapes": shapes, "replays": args.replays, "warmup": 2, "settle": 1}
    injected = None
    if table is None:
        timed_shapes = llm.collective_rpc("engine_grid", args=(json.dumps(spec),))[0]["shapes"]
    else:
        assert ROUTES["made"] >= 40 and ROUTES["buf"] is not None, ROUTES["made"]
        timed_shapes, injected = [], {}
        for (r, t), shape in zip(pairs, shapes):
            ids = torch.tensor(table["shapes"][f"{r}x{t}"], dtype=torch.int64, device="cuda")  # [40, t, 6]
            with torch.inference_mode():  # the buffer was made inside the engine's inference mode
                ROUTES["buf"].fill_(-1)
                ROUTES["buf"][:, :t].copy_(ids)
            one = dict(spec, shapes=[shape])
            timed_shapes.append(llm.collective_rpc("engine_grid", args=(json.dumps(one),))[0]["shapes"][0])
            injected[f"{r}x{t}"] = round(sum(len(torch.unique(ids[layer])) for layer in range(40)) / 40, 2)
    rows = []
    for (r, t), timed in zip(pairs, timed_shapes):
        s = sorted(timed["step_ms"])
        print(f"{r:>3}x{t:<3} real_rows={timed.get('real_rows')} step median {s[len(s) // 2]:.3f} ms "
              f"min {s[0]:.3f}", flush=True)
        rows.append({"shape": f"{r}x{t}", "num_reqs": r, "num_tokens": t, "real_rows": timed.get("real_rows"),
                     "step_ms": timed["step_ms"], "forward_ms": timed["forward_ms"],
                     "drafter_ms": timed["drafter_ms"], "modes": timed["modes"]})
    profiled = None
    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        profiled = []
        for (r, t), shape in zip(pairs, shapes):
            if table is not None:
                with torch.inference_mode():
                    ROUTES["buf"].fill_(-1)
                    ROUTES["buf"][:, :t].copy_(torch.tensor(table["shapes"][f"{r}x{t}"], dtype=torch.int64,
                                                            device="cuda"))
            torch.cuda.nvtx.range_push(f"shape {r}x{t}")
            one = {"shapes": [shape], "replays": args.profile_replays, "warmup": 1, "settle": 1}
            got = llm.collective_rpc("engine_grid", args=(json.dumps(one),))[0]["shapes"][0]
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            profiled.append({"shape": f"{r}x{t}", "runs": 1 + 2 * args.profile_replays,
                             "step_ms": got["step_ms"], "real_rows": got.get("real_rows")})
        torch.cuda.cudart().cudaProfilerStop()
    from vllm import envs

    result = {
        "boot_s": round(boot_s, 1), "gpu_warmup_ms": warm_ms, "inputs": args.inputs, "tp_shape": 4,
        "context": args.context, "capture_sizes": capture, "text_overrides": overrides,
        "moe_skip_padding": bool(envs.VLLM_MOE_SKIP_PADDING),
        "grid_tokens_sha256": sb_engine.tokens_sha256(),
        "spec": {k: v for k, v in spec.items() if k != "shapes"}, "rows": rows, "profiled": profiled,
        "routes": None if table is None else {"file": args.routes, "source": table.get("source"),
                                              "experts_hit_mean_injected": injected,
                                              "routers_built": ROUTES["made"],
                                              "router_prefixes": ROUTES["prefixes"]},
        "memory": {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                   "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2)},
        "env": {k: os.environ.get(k) for k in sorted(os.environ)
                if k.startswith(("VLLM_", "SPARK_", "CUDA_DEVICE", "PYTORCH_CUDA"))},
        "argv": sys.argv, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {args.out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
