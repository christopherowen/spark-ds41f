"""DS4.1 decode steps on one GB10 at TP4 rank shapes, with the compressor sharded as at TP4.

usage: step957.py --out FILE [--comp-tp 4] [--shapes 1x2,1x3,...] [--context 4096]
                  [--replays 10] [--no-comp-window] [--profile-shapes 1x2,1x6]
                  [--profile-replays 4] [--layers 40]

The engine runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0) at tensor-parallel
size 1 with one TP4 rank's shapes (16 attention heads, 2 output groups, a 576-wide expert
intermediate: experiments/2026-10-31-iket-full-43l-and-runtime-opt's TP_SHAPES[4]) and
dummy weights. At TP size 1 the compressor would always be replicated, so DeepseekCompressor
is built with the tensor-parallel world size it has at TP4 (``--comp-tp``, rank 0), as
tests/models/test_deepseek_v4_1_compressor.py builds its ranks: candidate-54's code then
shards and 12-bit packs ``fused_wkv_wgate`` (128 columns a part), and 0795's replicates it
in BF16. The all-gather of the sharded path is a local stand-in (rank 0's shard repeated:
one copy kernel), so this measures the rank's own kernels, not RoCE latency.

The steps are the engine grid's dummy runs (GPUModelRunner.engine_grid, as servebench's
``engine`` grid times them on the testbed): every shape warms up twice, then ``replays``
rounds alternate the shape order, each timed run after one settle run.
"""

import argparse
import functools
import json
import os
import sys
import time

MODEL = ("/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/"
         "1bae0e1088993b9b82de08f59de5be5a34e42170")
FULL = {
    "num_hidden_layers": 40,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    "kv_source_layer_ids": [2, 8, 14, 20],
    "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
    "candidate_source_layer_id": 20,
    "engram_layer_ids": [],
    "dspark_target_layer_ids": [37, 38, 39],
}
TP_SHAPES = {
    4: {"num_attention_heads": 16, "o_groups": 2, "moe_intermediate_size": 576},
    3: {"num_attention_heads": 24, "o_groups": 3, "moe_intermediate_size": 768},
}
DISPATCH = []
LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32]


def shrink(config, overrides):
    text = config.get_text_config()
    for key, value in overrides.items():
        setattr(text, key, value)
    return config


def parse_shapes(text):
    out = []
    for item in text.split(","):
        if item:
            r, t = item.split("x")
            out.append((int(r), int(t)))
    return out


def zero_integer_dummies():
    """vLLM's dummy loader seeds floating tensors and leaves integer ones (packed FP4
    experts, UE8M0 scales) as allocated; zero them so every boot loads the same bytes."""
    import torch
    from vllm.model_executor.model_loader import dummy_loader, weight_utils

    seeded = weight_utils.initialize_single_dummy_weight

    def initialize(param, *args, **kwargs):
        if param.device.type == "meta":
            return
        dummy_val = getattr(param, "dummy_weight_value", None)
        if (not torch.is_floating_point(param) and dummy_val is None) or dummy_val == 0:
            param.zero_()
            return
        seeded(param, *args, **kwargs)

    dummy_loader.initialize_single_dummy_weight = initialize
    weight_utils.initialize_single_dummy_weight = initialize


def patch_compressor_tp(tp):
    """Build DeepseekCompressor as one rank of a TP-``tp`` group sees it (rank 0)."""
    import torch
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    import vllm.models.deepseek_v4_1.compressor as comp

    orig_init = comp.DeepseekCompressor.__init__

    @functools.wraps(orig_init)
    def init(self, *args, **kwargs):
        saved = [(m, getattr(m, "get_tensor_model_parallel_world_size"),
                  getattr(m, "get_tensor_model_parallel_rank")) for m in (linear, parameter)]
        for m in (linear, parameter):
            m.get_tensor_model_parallel_world_size = lambda: tp
            m.get_tensor_model_parallel_rank = lambda: 0
        try:
            orig_init(self, *args, **kwargs)
        finally:
            for m, ws, rk in saved:
                m.get_tensor_model_parallel_world_size = ws
                m.get_tensor_model_parallel_rank = rk

    comp.DeepseekCompressor.__init__ = init

    def all_gather_standin(t, dim=0):
        assert dim == 0
        return t.repeat(tp, 1)

    comp.tensor_model_parallel_all_gather = all_gather_standin


WINDOWS = []


def patch_l2_windows(drop_comp):
    """Record every layer's L2 windows (build_l2_prefetch's description); with
    ``drop_comp``, drop the COMP window after building it (before any capture)."""
    from vllm.models.deepseek_v4_1.nvidia import model as m

    orig = m.DeepseekV4DecoderLayer.build_l2_prefetch

    @functools.wraps(orig)
    def build(self, nxt):
        out = orig(self, nxt)
        if drop_comp:
            self.attn._l2pf_comp = None
            out = out + " | COMP dropped"
        WINDOWS.append(out)
        return out

    m.DeepseekV4DecoderLayer.build_l2_prefetch = build


def patch_draft_dispatch_log():
    """Record each DSpark draft step's CUDA-graph dispatch: requests, tokens, mode."""
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as sp

    orig = sp.dispatch_cg_and_sync_dp

    @functools.wraps(orig)
    def dispatch(manager, num_reqs, num_tokens, *args, **kwargs):
        desc, sync = orig(manager, num_reqs, num_tokens, *args, **kwargs)
        DISPATCH.append((int(num_reqs), int(num_tokens), str(desc.cg_mode), int(desc.num_tokens)))
        return desc, sync

    sp.dispatch_cg_and_sync_dp = dispatch


def inspect_worker(worker):
    """Compressor layout and L2 prefetch windows of the loaded model (rank 0)."""
    import torch
    from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor

    model = worker.model_runner.model
    comps, comp_bytes, plans = [], 0, []
    for name, mod in model.named_modules():
        if isinstance(mod, DeepseekCompressor):
            layer = mod.fused_wkv_wgate
            nbytes = 0
            for attr in ("weight", "weight_exception_entries", "weight_row_offsets",
                         "packed_bf16_exceptions"):
                t = getattr(layer, attr, None)
                if isinstance(t, torch.Tensor):
                    nbytes += t.numel() * t.element_size()
            sp = getattr(mod, "_sp_weight", None)
            sp_bytes = (sp.numel() * sp.element_size()
                        if isinstance(sp, torch.Tensor) and sp.data_ptr() != layer.weight.data_ptr() else 0)
            nbytes += sp_bytes
            comp_bytes += nbytes
            comps.append({
                "name": name,
                "ratio": mod.compress_ratio,
                "layer_tp_size": int(getattr(layer, "tp_size", 1)),
                "packed": hasattr(layer, "packed_bf16_exponent_base"),
                "weight_shape": list(layer.weight.shape),
                "weight_dtype": str(layer.weight.dtype),
                "bytes": nbytes,
                "sp_weight_extra_bytes": sp_bytes,
                "sp_enabled": bool(getattr(mod, "_sp_enabled", False)),
            })
        comp_plan = getattr(mod, "_l2pf_comp", "absent")
        if comp_plan != "absent" and comp_plan is not None:
            plans.append({"name": name, "COMP": comp_plan.describe(),
                          "WO": mod._l2pf_wo.describe() if getattr(mod, "_l2pf_wo", None) else None})
    from vllm.models.deepseek_v4_1 import l2_prefetch

    return {"compressors": comps, "compressor_bytes": comp_bytes, "owner_plans": plans,
            "l2_prefetch_enabled": bool(l2_prefetch.ENABLED)}


def grid(llm, shapes, context, replays, warmup=2, settle=1):
    spec = {"shapes": [{"num_reqs": r, "context_len": context, "num_tokens": t}
                       for r, t in shapes],
            "replays": replays, "warmup": warmup, "settle": settle}
    reply = llm.collective_rpc("engine_grid", args=(json.dumps(spec),))[0]
    out = []
    for (r, t), row in zip(shapes, reply["shapes"]):
        out.append({"shape": f"{r}x{t}", "num_reqs": r, "num_tokens": t,
                    "step_ms": row["step_ms"], "forward_ms": row["forward_ms"],
                    "drafter_ms": row["drafter_ms"], "modes": row["modes"]})
    return spec, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp-shape", type=int, choices=(3, 4), default=4)
    ap.add_argument("--comp-tp", type=int, default=0, help="0: the --tp-shape")
    ap.add_argument("--max-num-seqs", type=int, default=0, help="0: 16 at TP4, 8 at TP3 (the recipe)")
    ap.add_argument("--log-draft-dispatch", action="store_true")
    ap.add_argument("--shapes", default="1x2,1x3,1x4,1x6,2x4,2x6,3x6,4x8,4x12,4x24,16x32")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--no-comp-window", action="store_true")
    ap.add_argument("--profile-shapes", default="")
    ap.add_argument("--profile-replays", type=int, default=4)
    ap.add_argument("--kv-bytes", type=int, default=6 << 30)
    ap.add_argument("--max-model-len", type=int, default=1048576)
    args = ap.parse_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch

    torch.cuda.init()
    from omlib import gpu_warmup

    zero_integer_dummies()
    if args.comp_tp == 0:
        args.comp_tp = args.tp_shape
    if args.log_draft_dispatch:
        patch_draft_dispatch_log()
    if args.comp_tp > 1:
        patch_compressor_tp(args.comp_tp)
    patch_l2_windows(args.no_comp_window)
    from vllm import LLM

    overrides = {**FULL, **TP_SHAPES[args.tp_shape]}
    shapes = parse_shapes(args.shapes)
    max_seqs = args.max_num_seqs or {4: 16, 3: 8}[args.tp_shape]
    max_seqs = max(max_seqs, max(r for r, _ in shapes))
    started = time.monotonic()
    llm = LLM(
        model=MODEL,
        tokenizer_mode="deepseek_v41",
        load_format="dummy",
        dtype="bfloat16",
        seed=0,
        tensor_parallel_size=1,
        block_size=256,
        max_model_len=args.max_model_len,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=512,
        long_prefill_token_threshold=256,
        max_parallel_prefills=1,
        async_scheduling=True,
        kv_cache_memory_bytes=args.kv_bytes,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        attention_backend="TILELANG",
        linear_backend="tilelang",
        moe_backend="tilelang",
        speculative_config={
            "method": "dspark", "num_speculative_tokens": 5,
            "draft_tensor_parallel_size": 1, "attention_backend": "TILELANG",
            "draft_sample_method": "greedy", "rejection_sample_method": "block",
            "enable_adaptive_verification": True, "adaptive_verification_row_cost": 0.035,
        },
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"],
                            "cudagraph_capture_sizes": LADDER},
        generation_config="vllm",
        hf_overrides=functools.partial(shrink, overrides=overrides),
        jit_monitor_mode="off",
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=0,
    )
    boot_s = time.monotonic() - started
    capture_dispatch = list(DISPATCH)
    DISPATCH.clear()
    print(f"engine ready in {boot_s:.0f} s", flush=True)
    info = llm.collective_rpc(inspect_worker)[0]
    print(json.dumps(info), flush=True)
    warm_ms = gpu_warmup()
    spec, rows = grid(llm, shapes, args.context, args.replays)
    for row in rows:
        s = sorted(row["step_ms"])
        print(f"{row['shape']:>6} step median {s[len(s)//2]:.3f} ms  min {s[0]:.3f}", flush=True)
    profiled = None
    if args.profile_shapes:
        pshapes = parse_shapes(args.profile_shapes)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        pspec, prow = grid(llm, pshapes, args.context, args.profile_replays, warmup=1, settle=0)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        profiled = {"spec": pspec, "rows": prow}
    from vllm import envs

    result = {
        "boot_s": round(boot_s, 1),
        "gpu_warmup_ms": warm_ms,
        "tp_shape": args.tp_shape,
        "comp_tp": args.comp_tp,
        "draft_dispatch": None if not args.log_draft_dispatch else {
            "at_startup": sorted({str(d) for d in capture_dispatch}),
            "in_grid": {str(k): DISPATCH.count(k) for k in sorted(set(DISPATCH))},
        },
        "no_comp_window": args.no_comp_window,
        "l2_windows": WINDOWS,
        "text_overrides": overrides,
        "capture_sizes": LADDER,
        "max_num_seqs": max_seqs,
        "kv_bytes": args.kv_bytes,
        "max_model_len": args.max_model_len,
        "spec": spec,
        "rows": rows,
        "profiled": profiled,
        "model": info,
        "memory": {"allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                   "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2)},
        "env": {k: os.environ.get(k) for k in sorted(os.environ)
                if k.startswith(("VLLM_", "SPARK_", "CUDA_DEVICE", "PYTORCH_CUDA"))},
        "argv": sys.argv,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {args.out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
