#!/usr/bin/env python3
"""rework r2 harness for PR #996 (ours, not the PR's): the host-side cost of one
routing layer's local masks, the part of route_local that _local_mask changes.

It times, per call, the mask lookups a later routing layer of a forward makes
(the first layer's mask kernel already ran): on the gate path
_routing_masks(local_rows, input_ids, ...) and on the partials path
_partials_mask(local_rows) plus _local_mask(input_ids). Three bodies of
_local_mask are swapped in and interleaved in one process: 1107's (verbatim),
round 1's (verbatim) and this image's (the mounted round 2). On a rank without
pad rows none of them launches a kernel, so the loop times host work only; on
the tail rank 1107's body launches its allocate-and-copy kernels every call.

usage: cpu_996.py <out.json>
"""
import json
import sys
import time

sys.path.insert(0, "/tmp")
import bench_996_route_local as b  # noqa: E402  (patches scratch and the context)
import torch  # noqa: E402

from vllm.models.deepseek_v4_1 import sp_prefill  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import router as R  # noqa: E402

LO = b.LO
CALLS = 2000
REPEATS = 7
CANDIDATE = R._local_mask


def local_mask_1107(mask, rows, sp_rows, *, context=None, pad=False):
    """1107's _local_mask, body verbatim; this image's keywords ignored."""
    if mask is None:
        return None
    if sp_rows is not None and mask.shape[0] != rows:
        return sp_rows.local_slice(mask[: sp_rows.num_tokens])
    return mask if mask.shape[0] == rows else mask[:rows]


def local_mask_r1(mask, rows, sp_rows, *, context=None, pad=False):
    """Round 1's _local_mask (rework-996 873df17), body verbatim."""
    if mask is None:
        return None
    if sp_rows is not None and mask.shape[0] != rows:
        if context is None and R.is_forward_context_available():
            context = R.get_forward_context()
        cache = None
        key = (
            id(mask),
            sp_rows.num_tokens,
            sp_rows.world_size,
            sp_rows.rank,
            rows,
            pad,
        )
        if context is not None:
            cache = getattr(context, "_ds41_local_masks", None)
            if cache is None:
                cache = {}
                context._ds41_local_masks = cache
            entry = cache.get(key)
            if entry is not None and entry[0] is mask:
                return entry[1]
        src = mask[: sp_rows.num_tokens]
        if pad and sp_rows.real_rows < sp_rows.local_rows:
            local = src.new_ones((sp_rows.local_rows, *src.shape[1:]))
            if sp_rows.real_rows:
                start = sp_rows.rank * sp_rows.local_rows
                local[: sp_rows.real_rows].copy_(src[start : start + sp_rows.real_rows])
        else:
            local = sp_rows.local_slice(src)
        if cache is not None:
            cache[key] = (mask, local)
        return local
    return mask if mask.shape[0] == rows else mask[:rows]


ARMS = {"1107": local_mask_1107, "r1": local_mask_r1, "r2": CANDIDATE}


def layer_masks(path, layout, ids):
    rows = layout.local_rows
    if path == "gate":
        return R._routing_masks(rows, ids, LO, 1, sp_rows=layout)
    return R._partials_mask(rows, sp_rows=layout), R._local_mask(ids, rows, layout)


def run(mode):
    out = []
    cells = [(3, 768, 0), (3, 768, 2), (3, 1024, 0), (3, 1024, 2), (3, 4096, 0), (3, 4096, 2),
             (3, 8192, 0), (3, 8192, 2), (4, 1024, 0), (4, 1024, 3), (4, 6144, 0), (4, 6144, 3)]
    for tp, T, rank in cells:
        layout = sp_prefill.SPRows(T, tp, rank)
        path = "partials" if layout.local_rows <= 256 else "gate"
        _, ids = b.step_inputs(T, T * 10 + tp)
        per = {a: [] for a in ARMS}
        for rep in range(REPEATS):
            order = list(ARMS) if rep % 2 == 0 else list(reversed(ARMS))
            for name in order:
                R._local_mask = ARMS[name]
                ctx = b.new_ctx(T)
                ctx.is_padding[T - 3 :] = True
                layer_masks(path, layout, ids)  # the forward's first layer
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(CALLS):
                    layer_masks(path, layout, ids)
                t1 = time.perf_counter()
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                per[name].append({"host_ns": (t1 - t0) * 1e9 / CALLS, "with_sync_ns": (t2 - t0) * 1e9 / CALLS})
        R._local_mask = CANDIDATE
        row = {"mode": mode, "tp": tp, "T": T, "rank": rank, "path": path,
               "local_rows": layout.local_rows, "real_rows": layout.real_rows,
               "tail": layout.real_rows < layout.local_rows}
        for name, runs in per.items():
            row[name] = {k: round(sorted(r[k] for r in runs)[len(runs) // 2], 1) for k in ("host_ns", "with_sync_ns")}
        out.append(row)
        print("row", json.dumps(row), flush=True)
    return out


def main(out_path):
    res = {"device": torch.cuda.get_device_name(0), "calls": CALLS, "repeats": REPEATS,
           "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": []}
    with torch.no_grad():
        res["rows"] += run("no_grad")
    with torch.inference_mode():
        res["rows"] += run("inference_mode")
    res["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print("SAVED", out_path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
