"""#957 part 1 (0795): the compressor projection one TP4 (and TP3) rank runs at decode.

usage: python3 /h/h957k.py --boot K --out FILE

Builds DeepseekCompressor ranks as tests/models/test_deepseek_v4_1_compressor.py does
(rank 0 of a TP group, world size patched in vllm.model_executor.layers.linear and
vllm.model_executor.parameter), with this container's code: candidate-54's shards and
12-bit packs fused_wkv_wgate at TP3/TP4 (then all-gathers and column-gathers), 0795's
replicates it in BF16. The all-gather is a local stand-in (rank 0's shard repeated, one
copy kernel), so the RoCE all-gather is not in these numbers. Times ``_project(x)``, the
whole decode projection of one owner layer, by CUDA-graph replay (20 calls a graph,
L2-hot), by one-call graph replay after a 256 MiB write evicts L2 (L2-cold), by CUPTI
kernel time, and eagerly. Hashes TP1's outputs (the reference both arms' tests use).
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

from omlib import _stats, eager_us, env_info, gpu_warmup, graph_cold_us, graph_hot_us, h, kernel_us

ROWS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
HIDDEN = 5120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import vllm.models.deepseek_v4_1.compressor as comp_mod
    import vllm.v1.worker.workspace as workspace
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor

    device = torch.device("cuda", 0)
    workspace._manager = workspace.WorkspaceManager(device)
    workspace.dbo_current_ubatch_id = lambda: 0
    capacity, requests = 160, 16
    warm = gpu_warmup()

    def make_config():
        return SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=capacity, max_num_seqs=requests),
            compilation_config=SimpleNamespace(static_forward_context={}),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
            speculative_config=None,
        )

    state = {"tp": 1}

    def all_gather_standin(t, dim=0):
        return t.repeat(state["tp"], 1)

    comp_mod.tensor_model_parallel_all_gather = all_gather_standin
    gen = torch.Generator(device=device).manual_seed(957)
    weights = {
        r: [(torch.randn((512, HIDDEN), generator=gen, device=device) * 0.02).bfloat16() for _ in range(r)]
        for r in (1, 2)
    }
    x = torch.randn((capacity, HIDDEN), generator=gen, device=device, dtype=torch.bfloat16)

    def build(ratio, tp):
        for mod in (linear, parameter):
            mod.get_tensor_model_parallel_rank = lambda: 0
            mod.get_tensor_model_parallel_world_size = lambda: tp
        c = DeepseekCompressor(make_config(), ratio, HIDDEN, 512, prefix=f"r{ratio}_{tp}").to(device)
        p = c.fused_wkv_wgate.weight
        for i, w in enumerate(weights[ratio]):
            p.weight_loader(p, w, i)
        c.fused_wkv_wgate.process_weights_after_loading()
        c.fused_wkv_wgate.quant_method.declare_layer_kernels(c.fused_wkv_wgate)
        c._ensure_capacity_buffers(device, capacity, requests)
        layer = c.fused_wkv_wgate
        nbytes = sum(
            t.numel() * t.element_size()
            for t in (getattr(layer, a, None) for a in ("weight", "weight_exception_entries", "weight_row_offsets"))
            if isinstance(t, torch.Tensor)
        )
        info = {
            "layer_tp_size": int(layer.tp_size),
            "packed": hasattr(layer, "packed_bf16_exponent_base"),
            "weight_shape": list(layer.weight.shape),
            "bytes": nbytes,
            "sp_enabled": bool(c._sp_enabled),
        }
        return c, info

    out = {"boot": args.boot, "env": env_info(), "gpu_warmup_ms": warm, "layouts": {}, "cells": {}, "tp1_hash": {}}
    with torch.no_grad(), workspace.use_workspace_lane(0):
        for ratio in (1, 2):
            tp1, info1 = build(ratio, 1)
            out["layouts"][f"r{ratio}_tp1"] = info1
            hs = {}
            for rows in (1, 7, 16, 64, 160):
                tp1._project(x[:rows])
                hs[str(rows)] = [h(tp1._values[:rows])] + ([h(tp1._gates[:rows])] if ratio == 2 else [])
            out["tp1_hash"][f"r{ratio}"] = hs
            del tp1
            for tp in (4, 3):
                state["tp"] = tp
                c, info = build(ratio, tp)
                out["layouts"][f"r{ratio}_tp{tp}"] = info
                for rows in ROWS:
                    inp = x[:rows]
                    fn = lambda: c._project(inp)  # noqa: E731
                    cell = {
                        "graph_hot": graph_hot_us(fn),
                        "graph_cold": graph_cold_us(fn),
                        "kernel": kernel_us(fn),
                        "eager_us": eager_us(fn),
                    }
                    out["cells"][f"r{ratio}_tp{tp}_m{rows}"] = cell
                    print(f"r{ratio} tp{tp} m{rows}: hot {cell['graph_hot']['median_us']} cold "
                          f"{cell['graph_cold']['median_us']} kern {cell['kernel']['per_call_us']} "
                          f"eager {cell['eager_us']}", flush=True)
                del c
                torch.cuda.empty_cache()
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
