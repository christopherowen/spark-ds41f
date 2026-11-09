#!/usr/bin/env python3
"""own-marks harness (candidate-49, ours, not the PR's): PR #996 ships no bench
script (its bench_sp_moe.py stayed in /tmp on sp5). This times the image's own
TileLangGateRouter.route_local on one GPU, as an SP rank would call it on the 20
encoder layers of one prefill step, and hashes its outputs so the 1107 and
candidate images can be compared bit for bit on the real rows.

usage: bench_996_route_local.py bench <out.json>
       bench_996_route_local.py infer <out.json>   (same calls under torch.inference_mode)
"""
import hashlib
import json
import sys
import time
import traceback

import torch

from vllm.models.deepseek_v4_1 import sp_prefill
from vllm.models.deepseek_v4_1.tilelang import linear
from vllm.models.deepseek_v4_1.tilelang import router as tilelang_router

LO = 129264
SCALE = 1.5
LAYERS = 20
DEV = "cuda"

# plain scratch, as the PR's tests do (no shared workspace outside a worker)
linear._scratch = lambda specs, scratch: [torch.empty(s, dtype=d, device=DEV) for s, d in specs]


class _Ctx:
    pass


CTX = {"ctx": None}
tilelang_router.is_forward_context_available = lambda: True
tilelang_router.get_forward_context = lambda: CTX["ctx"]


class _Gate(torch.nn.Module):
    def __init__(self, experts, hidden=5120, seed=0):
        from vllm.models.deepseek_v4_1.tilelang.linear import TileLangLinearMethod

        super().__init__()
        g = torch.Generator(device=DEV).manual_seed(seed)
        w = torch.randn(experts, hidden, device=DEV, generator=g) * 0.02
        self.weight = torch.nn.Parameter(w.to(torch.bfloat16), requires_grad=False)
        self.out_dtype = torch.float32
        self.quant_method = TileLangLinearMethod()
        self.quant_method.declare_layer_kernels(self)

    def forward(self, x):
        return self.quant_method.apply(self, x), None


def make_router(seed=1234):
    g = torch.Generator(device=DEV).manual_seed(seed)
    gate = _Gate(384, seed=seed)
    bias = torch.randn(384, device=DEV, generator=g) * 0.1
    bias_vl = torch.nn.Parameter(torch.randn(384, device=DEV, generator=g) * 0.1, requires_grad=False)
    r = tilelang_router.TileLangGateRouter(
        gate=gate,
        top_k=6,
        global_num_experts=384,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=SCALE,
        scoring_func="sqrtsoftplus",
        bias_vl=bias_vl,
        image_sentinel_lo=LO,
        image_sentinel_count=1,
    )
    if hasattr(r, "declare_kernels"):
        r.declare_kernels()
    return r


def step_inputs(T, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(T, 5120, device=DEV, generator=g).to(torch.bfloat16)
    ids = torch.randint(0, 1000, (T,), device=DEV, generator=g)
    ids[1::97] = LO
    return x, ids


def new_ctx(T):
    c = _Ctx()
    c.attn_metadata = None
    c.is_padding = torch.zeros(T, dtype=torch.bool, device=DEV)
    CTX["ctx"] = c
    return c


def h(t):
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


def med(x):
    x = sorted(x)
    return round(x[len(x) // 2], 3)


def bench(out_path):
    router = make_router()
    res = {"device": torch.cuda.get_device_name(0), "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "has_cache_kwargs": "context" in tilelang_router._local_mask.__code__.co_varnames, "rows": []}
    configs = []
    for tp in (3, 4):
        for T in (768, 1024, 2048, 4096, 6144, 8192, 16384):
            for rank in sorted({0, tp - 1}):
                configs.append((tp, T, rank))
    for tp, T, rank in configs:
        layout = sp_prefill.SPRows(T, tp, rank)
        x, ids = step_inputs(T, T * 10 + tp)
        local_x = layout.local_slice(x)
        first, cached, pass_us = [], [], []
        hashes = set()
        pad_ids_all_neg1 = None
        with torch.no_grad():
            for step in range(15):
                new_ctx(T)
                lat = []
                outs = []
                for layer in range(LAYERS):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    w, i = router.route_local(local_x, layout, input_ids=ids)
                    torch.cuda.synchronize()
                    lat.append((time.perf_counter() - t0) * 1e6)
                    if layer in (0, LAYERS - 1):
                        outs.append((w, i))
                if step >= 5:
                    first.append(lat[0])
                    cached.extend(lat[1:])
                real = layout.real_rows
                for w, i in outs:
                    hashes.add((h(w[:real].view(torch.int32)), h(i[:real])))
                if real < layout.local_rows:
                    pad_ids_all_neg1 = bool((outs[0][1][real:] == -1).all().item())
                # one pipelined 20-layer pass, synced once
                new_ctx(T)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for layer in range(LAYERS):
                    router.route_local(local_x, layout, input_ids=ids)
                torch.cuda.synchronize()
                if step >= 5:
                    pass_us.append((time.perf_counter() - t0) * 1e6 / LAYERS)
        row = {"tp": tp, "T": T, "rank": rank, "local_rows": layout.local_rows, "real_rows": layout.real_rows,
               "tail": layout.real_rows < layout.local_rows,
               "first_call_us_median": med(first), "cached_calls_us_median": med(cached),
               "pass_us_per_layer_median": med(pass_us),
               "out_hashes_real_rows": sorted(hashes), "outputs_stable_across_layers_and_steps": len(hashes) == 1,
               "pad_ids_all_minus1": pad_ids_all_neg1}
        res["rows"].append(row)
        print("row", json.dumps(row), flush=True)
    res["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print("SAVED", out_path, flush=True)


def infer(out_path):
    router = make_router()
    res = {"device": torch.cuda.get_device_name(0), "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "cases": []}
    for tp, T, rank in ((3, 1024, 0), (3, 1024, 2), (3, 205, 2), (4, 1024, 3), (3, 6144, 0)):
        case = {"tp": tp, "T": T, "rank": rank}
        try:
            with torch.inference_mode():
                layout = sp_prefill.SPRows(T, tp, rank)
                x, ids = step_inputs(T, T * 10 + tp)
                local_x = layout.local_slice(x)
                new_ctx(T)
                case["is_padding_is_inference"] = CTX["ctx"].is_padding.is_inference()
                for layer in range(3):
                    w, i = router.route_local(local_x, layout, input_ids=ids)
                torch.cuda.synchronize()
                case["result"] = "ok"
                case["out_hash_real_rows"] = [h(w[: layout.real_rows].view(torch.int32)), h(i[: layout.real_rows])]
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            case["result"] = "raised"
            case["exception"] = f"{type(exc).__name__}: {exc}"
            case["traceback_tail"] = traceback.format_exc().strip().splitlines()[-8:]
        res["cases"].append(case)
        print("case", json.dumps(case), flush=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print("SAVED", out_path, flush=True)


if __name__ == "__main__":
    {"bench": bench, "infer": infer}[sys.argv[1]](sys.argv[2])
