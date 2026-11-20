"""Expert routes of the engine grid's dummy steps on one GB10 at TP4 rank-0 shapes.

usage: census.py --out FILE [--shapes 1x2,...] [--context 4096] [--inputs grid,token0,sample]

Boots step957.py's engine (40 target layers and the DSpark drafter's 3, one TP4 rank's
shapes, dummy weights, the recipe's environment) eagerly and without adaptive verification
(which needs full CUDA graphs and does not touch a dummy step's routing), so a wrapper around
TileLangGateRouter._compute_routing sees every router call, and runs one engine_grid step
per shape and input through the real RPC path:
- grid: the grid as servebench runs it today (no token ids);
- token0: token 0 in every row, marked real rows (identical tokens);
- sample: servebench's grid version 2 ids (grid_tokens.json), marked real rows.
Per MoE layer and step: rows, rows routed (any expert id >= 0) and distinct experts hit.
Needs the patched arm (engine_grid token_ids) for token0 and sample.

Dummy weights: a real row's experts follow the dummy gate, so the sample's counts are a
random gate's spread. Whether padding rows are routed, and that identical rows meet the
same experts, does not depend on the weights. ref_routes.py gives real text's spread
through the checkpoint's own gates.
"""

import argparse
import functools
import json
import os
import sys
import time

import step957 as s957

GRID_SHAPES = "1x2,4x8,8x16,16x32,1x3,4x12,8x24,16x48,1x6,2x12,4x24,8x48,12x72,16x96"
CAPTURE = {"on": False, "calls": []}


def patch_router():
    import torch
    from vllm.models.deepseek_v4_1.tilelang import router as R

    orig = R.TileLangGateRouter._compute_routing

    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if CAPTURE["on"]:
            CAPTURE["calls"].append((id(self), out[1].detach().to("cpu", copy=True)))
        return out

    R.TileLangGateRouter._compute_routing = wrapped
    return torch


def router_names(worker):
    mr = worker.model_runner
    out = {}
    drafter = getattr(getattr(mr, "speculator", None), "model", None)
    for tag, model in (("target", mr.model), ("drafter", drafter)):
        if model is None:
            continue
        for name, mod in model.named_modules():
            router = getattr(mod, "router", None)
            if router is not None and type(router).__name__.endswith("Router"):
                out[id(router)] = f"{tag}:{name}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default=GRID_SHAPES)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--inputs", default="grid,token0,sample")
    args = ap.parse_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch

    torch.cuda.init()
    s957.zero_integer_dummies()
    s957.patch_compressor_tp(4)
    patch_router()
    from servebench import engine as sb_engine
    from vllm import LLM

    overrides = {**s957.FULL, **s957.TP_SHAPES[4]}
    shapes = s957.parse_shapes(args.shapes)
    started = time.monotonic()
    llm = LLM(
        model=s957.MODEL, tokenizer_mode="deepseek_v41", load_format="dummy", dtype="bfloat16", seed=0,
        tensor_parallel_size=1, block_size=256, max_model_len=1048576, max_num_seqs=16,
        max_num_batched_tokens=512, long_prefill_token_threshold=256, max_parallel_prefills=1,
        async_scheduling=True, kv_cache_memory_bytes=6 << 30, gpu_memory_utilization=0.85,
        enable_prefix_caching=True, attention_backend="TILELANG", linear_backend="tilelang",
        moe_backend="tilelang", enforce_eager=True,
        speculative_config={
            "method": "dspark", "num_speculative_tokens": 5, "draft_tensor_parallel_size": 1,
            "attention_backend": "TILELANG", "draft_sample_method": "greedy",
            # Adaptive verification needs full CUDA graphs; it prices and cuts verification
            # rows in serving steps and plays no part in a dummy step's routing.
            "rejection_sample_method": "block", "enable_adaptive_verification": False,
        },
        generation_config="vllm", hf_overrides=functools.partial(s957.shrink, overrides=overrides),
        jit_monitor_mode="off", limit_mm_per_prompt={"image": 1}, mm_processor_cache_gb=0,
    )
    boot_s = time.monotonic() - started
    print(f"engine ready in {boot_s:.0f} s", flush=True)
    names = llm.collective_rpc(router_names)[0]
    print(f"{len(names)} routers: {sorted(set(v.split(':')[0] for v in names.values()))}", flush=True)
    results = []
    for arm in args.inputs.split(","):
        for r, t in shapes:
            shape = {"num_reqs": r, "num_tokens": t, "context_len": args.context}
            if arm == "token0":
                shape["token_ids"] = [0] * t
            elif arm == "sample":
                shape = sb_engine.with_token_ids([shape], 2)[0]
            spec = {"shapes": [shape], "replays": 1, "warmup": 0, "settle": 0}
            CAPTURE["calls"].clear()
            CAPTURE["on"] = True
            reply = llm.collective_rpc("engine_grid", args=(json.dumps(spec),))[0]
            CAPTURE["on"] = False
            layers = []
            for rid, ids in CAPTURE["calls"]:
                valid = ids[ids >= 0]
                layers.append({"router": names.get(rid, f"router@{rid}"), "rows": int(ids.shape[0]),
                               "rows_routed": int((ids >= 0).any(dim=1).sum()),
                               "experts_hit": int(torch.unique(valid).numel())})
            target = [x["experts_hit"] for x in layers if x["router"].startswith("target")]
            drafter = [x["experts_hit"] for x in layers if x["router"].startswith("drafter")]
            print(f"{arm:7s} {r}x{t}: real_rows={reply['shapes'][0].get('real_rows')} "
                  f"{len(target)} target layers, experts hit mean "
                  f"{sum(target) / max(len(target), 1):.1f} (min {min(target, default=0)}, max {max(target, default=0)}); "
                  f"drafter {drafter}", flush=True)
            results.append({"input": arm, "shape": f"{r}x{t}", "num_reqs": r, "num_tokens": t,
                            "context_len": args.context, "real_rows": reply["shapes"][0].get("real_rows"),
                            "step_ms": reply["shapes"][0]["step_ms"], "layers": layers})
    from vllm import envs

    out = {"boot_s": round(boot_s, 1), "eager": True, "tp_shape": 4, "text_overrides": overrides,
           "moe_skip_padding": bool(envs.VLLM_MOE_SKIP_PADDING), "grid_tokens_sha256": sb_engine.tokens_sha256(),
           "results": results,
           "env": {k: os.environ.get(k) for k in sorted(os.environ)
                   if k.startswith(("VLLM_", "SPARK_", "CUDA_DEVICE", "PYTORCH_CUDA"))},
           "argv": sys.argv, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {args.out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
