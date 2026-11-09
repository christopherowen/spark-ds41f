#!/usr/bin/env python3
"""rework harness for PR #996 r1 (ours, not the PR's). It runs the image's
TileLangGateRouter.route_local under torch.inference_mode, as the vLLM worker
runs every forward, with one new stand-in ForwardContext per step and 20 routing
layers a step, and compares it with 1107's uncached _local_mask (its body copied
verbatim below and swapped into the module for the reference arm). It reuses the
own-marks harness bench_996_route_local.py (router, inputs, stand-in context).

usage: exact_996.py exact <out.json>   every rank, TP 3 and 4, T 1024..16384:
                                       torch.equal of every layer against 1107's masks
       exact_996.py ibench <out.json>  per-call and 20-layer-pass timing, this image's
                                       _local_mask against 1107's, interleaved per step
"""
import json
import sys
import time

sys.path.insert(0, "/tmp")
import bench_996_route_local as b  # noqa: E402  (patches scratch and the context)
import torch  # noqa: E402

from vllm.models.deepseek_v4_1 import sp_prefill  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import router as tilelang_router  # noqa: E402

LAYERS = 20
PAD = 3  # the step's last rows pad its batch
CANDIDATE = tilelang_router._local_mask


def local_mask_1107(mask, rows, sp_rows, *, context=None, pad=False):
    """1107's _local_mask (vllm/models/deepseek_v4_1/tilelang/router.py at 1107),
    body verbatim; the keywords this image's callers pass are ignored."""
    if mask is None:
        return None
    if sp_rows is not None and mask.shape[0] != rows:
        return sp_rows.local_slice(mask[: sp_rows.num_tokens])
    return mask if mask.shape[0] == rows else mask[:rows]


def arm(name):
    tilelang_router._local_mask = local_mask_1107 if name == "1107" else CANDIDATE


def new_step(T):
    """A new forward: a new context with a new padding mask and new ids view."""
    ctx = b.new_ctx(T)
    ctx.is_padding[T - PAD :] = True
    return ctx


def same(a, c):
    (wa, ia), (wc, ic) = a, c
    return torch.equal(ia, ic) and torch.equal(wa.view(torch.int32), wc.view(torch.int32))


def exact(out_path):
    router = b.make_router()
    res = {"device": torch.cuda.get_device_name(0), "mode": "inference_mode",
           "has_cache_kwargs": "context" in CANDIDATE.__code__.co_varnames,
           "reads_version": "_version" in CANDIDATE.__code__.co_names,
           "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cells": []}
    with torch.inference_mode():
        for tp in (3, 4):
            for T in (1024, 2048, 4096, 6144, 8192, 16384):
                x, ids = b.step_inputs(T, T * 10 + tp)
                arm("candidate")
                new_step(T)
                full = router._compute_routing(None, x, None, input_ids=ids.clone())
                gathered = []
                for rank in range(tp):
                    layout = sp_prefill.SPRows(T, tp, rank)
                    local_x = layout.local_slice(x)
                    arm("1107")
                    new_step(T)
                    ref = router.route_local(local_x, layout, input_ids=ids.clone())
                    arm("candidate")
                    compared = equal = 0
                    entries = []
                    inference = None
                    for step in range(2):
                        ctx = new_step(T)
                        step_ids = ids.clone()
                        inference = ctx.is_padding.is_inference()
                        for layer in range(LAYERS):
                            out = router.route_local(local_x, layout, input_ids=step_ids)
                            compared += 1
                            equal += same(ref, out)
                        entries.append(len(getattr(ctx, "_ds41_local_masks", {})))
                    gathered.append(out)
                    real = layout.real_rows
                    cell = {"tp": tp, "T": T, "rank": rank, "local_rows": layout.local_rows,
                            "real_rows": real, "tail": real < layout.local_rows,
                            "path": "partials" if layout.local_rows <= 256 else "gate",
                            "is_padding_is_inference": inference,
                            "layers_compared": compared, "layers_torch_equal_to_1107": equal,
                            "local_mask_cache_entries_per_step": entries,
                            "padded_ids_all_minus1": bool((out[1][max(0, T - PAD - layout.start):] == -1).all().item())
                            if T - PAD < layout.start + layout.local_rows else None}
                    res["cells"].append(cell)
                    print("cell", json.dumps(cell), flush=True)
                w = torch.cat([g[0][: sp_prefill.SPRows(T, tp, r).real_rows] for r, g in enumerate(gathered)])
                i = torch.cat([g[1][: sp_prefill.SPRows(T, tp, r).real_rows] for r, g in enumerate(gathered)])
                fr = {"tp": tp, "T": T, "gathered_torch_equal_to_full_rows": same(full, (w, i))}
                res.setdefault("full_rows", []).append(fr)
                print("full", json.dumps(fr), flush=True)
    res["all_layers_equal"] = all(c["layers_compared"] == c["layers_torch_equal_to_1107"] for c in res["cells"])
    res["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print("ALL_LAYERS_EQUAL", res["all_layers_equal"], flush=True)
    print("SAVED", out_path, flush=True)


def ibench(out_path):
    router = b.make_router()
    res = {"device": torch.cuda.get_device_name(0), "mode": "inference_mode",
           "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}
    configs = [(tp, T, rank) for tp in (3, 4) for T in (768, 1024, 2048, 4096, 6144, 8192, 16384)
               for rank in sorted({0, tp - 1})]
    with torch.inference_mode():
        for tp, T, rank in configs:
            layout = sp_prefill.SPRows(T, tp, rank)
            x, ids = b.step_inputs(T, T * 10 + tp)
            local_x = layout.local_slice(x)
            stats = {a: {"first": [], "cached": [], "pass": []} for a in ("1107", "candidate")}
            for step in range(15):
                order = ("1107", "candidate") if step % 2 == 0 else ("candidate", "1107")
                for name in order:
                    arm(name)
                    new_step(T)
                    step_ids = ids.clone()
                    lat = []
                    for layer in range(LAYERS):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        router.route_local(local_x, layout, input_ids=step_ids)
                        torch.cuda.synchronize()
                        lat.append((time.perf_counter() - t0) * 1e6)
                    new_step(T)
                    step_ids = ids.clone()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for layer in range(LAYERS):
                        router.route_local(local_x, layout, input_ids=step_ids)
                    torch.cuda.synchronize()
                    p = (time.perf_counter() - t0) * 1e6 / LAYERS
                    if step >= 5:
                        stats[name]["first"].append(lat[0])
                        stats[name]["cached"].extend(lat[1:])
                        stats[name]["pass"].append(p)
            arm("candidate")
            row = {"tp": tp, "T": T, "rank": rank, "local_rows": layout.local_rows,
                   "real_rows": layout.real_rows, "tail": layout.real_rows < layout.local_rows}
            for name, s in stats.items():
                row[name] = {"first_call_us_median": b.med(s["first"]),
                             "cached_calls_us_median": b.med(s["cached"]),
                             "pass_us_per_layer_median": b.med(s["pass"])}
            row["cached_speedup_1107_over_candidate"] = round(
                row["1107"]["cached_calls_us_median"] / row["candidate"]["cached_calls_us_median"], 3)
            row["pass_speedup_1107_over_candidate"] = round(
                row["1107"]["pass_us_per_layer_median"] / row["candidate"]["pass_us_per_layer_median"], 3)
            res["rows"].append(row)
            print("row", json.dumps(row), flush=True)
    res["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print("SAVED", out_path, flush=True)


if __name__ == "__main__":
    {"exact": exact, "ibench": ibench}[sys.argv[1]](sys.argv[2])
