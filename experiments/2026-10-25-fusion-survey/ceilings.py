#!/usr/bin/env python3
"""Upper bounds of candidate fusions and overlaps, from rank 0's profiled traces.

usage: ceilings.py [TRACES] [--out ceilings.json]

TRACES is request-inventory-5's ports directory on dgx1
(results/private/determinism/request-inventory-5/ports): decode-trace/dgx1 holds
one torch-profiler trace a request type (greedy, sampled, ..., mixed-c8, in
profile_decode.py's order) and prefill-trace/dgx1 the 15,395-token prompt and the
image request.

Every kernel on the critical path gets two numbers:
- excl: the time it runs while no other kernel does (the L2 prefetch's side
  stream does not count; the shared expert overlapping the routed chain does);
- gap: the idle time just before it starts.

A fusion that folds kernel X into a neighbour saves at most excl(X) + gap(X),
plus the gap before the kernel that follows the chain. That is the ceiling: the
fused kernel is assumed to cost no more than the kernel it keeps. A decode step
is one target forward (the 40 layers, the head, sampling, the drafter's inputs)
plus the drafter forward that follows it; steps start at the target's
vocab_embedding_kernel. Gaps above 1.5 us mark eager (host-launched) work; in
the CUDA graphs kernels launch 0.2-0.7 us apart.

The traces were taken with Python stacks on, which slows the host: eager gaps
are upper bounds; graph replay and kernel times are not affected.
"""
import argparse
import gzip
import json
import os
from collections import defaultdict

DECODE = ("greedy", "sampled", "top-p", "top-k", "top-p-top-k", "structured", "logprobs",
          "prompt-logprobs", "penalties", "stop", "thinking", "n2", "mixed-c8")
PREFETCH = "l2_prefetch_kernel"
MARK = "vocab_embedding_kernel"
EAGER_GAP = 1.5
BIG = ("gemm_", "moe_grouped", "indexer_score", "oneshot_allreduce", "vocab_packed")
# per expert and rank at TP4: FC1 1152 x 5120 and FC2 5120 x 576 E2M1 values, one UE8M0 byte per 32
FC2_BYTES_PER_FC1_BYTE = (5120 * 576 // 2 + 5120 * 576 // 32) / (1152 * 5120 // 2 + 1152 * 5120 // 32)


def load(path):
    with gzip.open(path, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    ks = []
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            a = e.get("args", {})
            ks.append((e["ts"], e["dur"], e["name"], e.get("tid"), tuple(a.get("grid", ())), tuple(a.get("block", ()))))
    ks.sort()
    return ks


def rows_of(seq):
    """Non-prefetch kernels in start order, with exclusive time and idle gap."""
    ks = sorted((k for k in seq if k[2] != PREFETCH), key=lambda k: k[0])
    edges = sorted([(k[0], 1, i) for i, k in enumerate(ks)] + [(k[0] + k[1], -1, i) for i, k in enumerate(ks)],
                   key=lambda e: (e[0], e[1]))
    active, last, excl = set(), None, [0.0] * len(ks)
    for t, kind, i in edges:
        if last is not None and len(active) == 1:
            excl[next(iter(active))] += t - last
        (active.add if kind == 1 else active.discard)(i)
        last = t
    rows, end = [], None
    for i, k in enumerate(ks):
        rows.append(dict(ts=k[0], dur=k[1], name=k[2].replace("_kernel_kernel", "_kernel"), tid=k[3],
                         grid=k[4], excl=excl[i], gap=0.0 if end is None else max(0.0, k[0] - end)))
        end = k[0] + k[1] if end is None else max(end, k[0] + k[1])
    return rows


def step_pairs(ks):
    """(target rows, drafter rows, start, end) for each full decode step."""
    starts = [k[0] for k in ks if k[2] == MARK]
    segs = [[k for k in ks if starts[i] <= k[0] < starts[i + 1]] for i in range(len(starts) - 1)]
    moe = lambda s: sum(1 for k in s if k[2].startswith("moe_grouped"))
    out, i = [], 0
    while i + 2 < len(starts):
        if moe(segs[i]) == 80 and moe(segs[i + 1]) == 6:
            out.append((rows_of(segs[i]), rows_of(segs[i + 1]), starts[i], starts[i + 2]))
            i += 2
        else:
            i += 1
    return out


def nm(r):
    return r["name"]


def isn(r, *prefixes):
    return r["name"].startswith(prefixes)


def cost(r):
    return r["excl"] + r["gap"]


def pieces(rows):
    """Split at each one-shot all-reduce: one sublayer (attention or MoE) a piece."""
    out, cur = [], []
    for r in rows:
        cur.append(r)
        if isn(r, "oneshot_allreduce"):
            out.append(cur)
            cur = []
    out.append(cur)
    return out


def is_layer(p):
    return any(nm(r) == "sparse_mla_kernel" or isn(r, "moe_grouped") for r in p)


def first(rows, pred, start=0):
    return next((i for i in range(start, len(rows)) if pred(rows[i])), None)


def layer_ceilings(c, rows, seg):
    """Per-sublayer fusion candidates; c[key] accumulates microseconds."""
    absorb = lambda key, *rs: [c.__setitem__(seg + key, c[seg + key] + cost(r)) for r in rs if r is not None]
    border = lambda key, *rs: [c.__setitem__(seg + key, c[seg + key] + r["gap"]) for r in rs if r is not None]
    ps = pieces(rows)
    for p in ps:
        if not is_layer(p):
            continue
        attn = any(nm(r) == "sparse_mla_kernel" for r in p)
        for i, r in enumerate(p):
            if r["gap"] <= EAGER_GAP:
                c[seg + "pdl_graph_gaps"] += r["gap"]
            if not (isn(r, *BIG) or nm(r) == "sparse_mla_kernel"):
                c[seg + "layer_small_kernel_budget"] += cost(r)
            if isn(r, "gemm_", "moe_grouped") and r["excl"] > 5:
                prev = p[i - 1] if i else None
                hide = r["gap"]
                if prev is not None and not isn(prev, *BIG) and nm(prev) != "sparse_mla_kernel" and prev["tid"] == r["tid"]:
                    hide += min(prev["excl"], r["excl"])
                c[seg + "pdl_weight_prologue"] += hide
        if any(isn(r, "engram_rows_wait") for r in p):
            i0 = first(p, lambda r: isn(r, "engram_rows_wait"))
            i1 = first(p, lambda r: isn(r, "engram_gate"), i0)
            absorb("engram_chain_ahead_on_side_stream", *p[i0:i1])
        for r in p:
            if isn(r, "metadata_pages"):
                absorb("metadata_once_per_index_source", r)
        i_proj = first(p, lambda r: isn(r, "mhc_project_streams"))
        if i_proj is not None:
            if i_proj == 0:
                border("ar_into_mhc_entry", p[0])
            i_fin = first(p, lambda r: isn(r, "mhc_finalize"), i_proj)
            absorb("finalize_into_mhc_entry", p[i_fin])
            border("finalize_into_mhc_entry", p[i_fin + 1])
            for r in p[:i_proj]:
                if isn(r, "engram_rows_wait", "engram_rows_release"):
                    absorb("engram_rows_one_kernel", r)
                if isn(r, "engram_gate"):
                    absorb("engram_gate_into_mhc_entry", r)
                if isn(r, "mhc_post", "mhc_collapse"):
                    absorb("standalone_mhc_post", r)
        if attn:
            i_fin = first(p, lambda r: isn(r, "mhc_finalize"))
            if i_fin is not None and isn(p[i_fin + 1], "per_token_cast"):
                absorb("attn_finalize_emits_fp8", p[i_fin + 1])
                border("attn_finalize_emits_fp8", p[i_fin + 2])
            i_qa = first(p, lambda r: isn(r, "gemm_mxfp8") and r["grid"][0] == 28)
            i_qb = first(p, lambda r: isn(r, "gemm_mxfp8") and r["grid"][0] in (64, 128), (i_qa or 0) + 1)
            if i_qa is not None and i_qb is not None:
                mids = p[i_qa + 1:i_qb]
                absorb("qkv_norms_casts_one_kernel", *mids[1:])
                border("qkv_norms_casts_one_kernel", p[i_qb])
                absorb("qkv_norm_cast_into_qa_epilogue", *mids)
                border("qkv_norm_cast_into_qa_epilogue", p[i_qb])
            i_mla = first(p, lambda r: nm(r) == "sparse_mla_kernel")
            j = (i_qb + 1) if i_qb is not None else 0
            span = p[j:i_mla]
            has_indexer = any(isn(r, "indexer", "compress") for r in span)
            for k, r in enumerate(span):
                front = k < 3 or isn(r, "metadata_decode", "metadata_chunk")
                if isn(r, "metadata_chunk", "metadata_decode"):
                    absorb("metadata_once_per_index_source", r)
                elif isn(r, "rope", "cache_writer") and (front or not has_indexer):
                    absorb("rope_cache_into_attention", r)
                else:
                    c[seg + "_indexer_chain_total"] += cost(r)
                    if not isn(r, "gemm_mxfp8", "gemm_bf16_splitk", "indexer_score"):
                        absorb("indexer_small_kernels", r)
            border("rope_cache_into_attention", p[i_mla])
            if isn(p[i_mla + 1], "sparse_mla_kernel_1"):
                absorb("attn_out_chain_into_wo", p[i_mla + 1])
            i_woa = first(p, lambda r: isn(r, "gemm_fp8w_bf16_partials"), i_mla)
            if i_woa is not None:
                border("attn_out_chain_into_wo", p[i_woa])
                if isn(p[i_woa + 1], "gemm_splitk_reduce_cast"):
                    absorb("attn_out_chain_into_wo", p[i_woa + 1])
                    border("attn_out_chain_into_wo", p[i_woa + 2])
        else:
            i_disp = first(p, lambda r: isn(r, "moe_dispatch"))
            i_fin = first(p, lambda r: isn(r, "mhc_finalize"))
            rtid, stid = p[i_disp]["tid"], p[i_fin]["tid"]
            gx = p[i_fin + 1]["grid"][0]
            casts = [r for r in p if isn(r, "per_token_cast") and r["grid"][0] == gx]
            absorb("routed_act_quant_into_producer", *[r for r in casts if r["tid"] == rtid])
            absorb("ffn_finalize_emits_fp8_shared", *[r for r in casts if r["tid"] == stid])
            absorb("one_router_kernel", *[r for r in p if isn(r, "router_partials", "moe_dispatch")])
            fc1 = next(r for r in p if isn(r, "moe_grouped") and r["grid"][0] == 9)
            fc2 = next(r for r in p if isn(r, "moe_grouped") and r["grid"][0] == 40)
            absorb("swiglu_into_fc1", next(r for r in p if isn(r, "moe_swiglu_cast")))
            border("swiglu_into_fc1", fc2)
            absorb("combine_into_fc2", next(r for r in p if isn(r, "moe_combine")))
            border("pdl_moe_chain", fc1, fc2)
            shared = [r for r in p if r["tid"] == stid and r not in casts and isn(r, "gemm_mxfp8", "activation_swiglu", "per_token_cast")]
            absorb("shared_expert_into_routed_gemms", *shared)
            c[seg + "_fc1"] += fc1["excl"]
            c[seg + "_fc2"] += fc2["excl"]
            # FC2's MXFP4 weight and scales are half of FC1's bytes for the same experts:
            # what FC2 would save streaming them at FC1's rate
            c[seg + "fc2_at_fc1_byte_rate"] += max(0.0, fc2["excl"] - FC2_BYTES_PER_FC1_BYTE * fc1["excl"])
        if isn(p[-1], "oneshot_allreduce"):
            border("epilogue_into_ar_staging_gap", p[-1])
            c[seg + "_ar_excl"] += p[-1]["excl"]
            c[seg + "_ar_count"] += 1


def tail_ceilings(c, rows, seg):
    """Rows outside the layer pieces: embedding, head, sampling, draft and step preparation."""
    layer = {id(r) for p in pieces(rows) if is_layer(p) for r in p}
    for r in rows:
        if id(r) in layer:
            continue
        if isn(r, "vocab_packed"):
            c[seg + "_lm_head"] += r["excl"]
        elif isn(r, "oneshot_allgather", "ncclDev", "oneshot_allreduce"):
            c[seg + "_tail_collectives"] += cost(r)
        elif isn(r, "gemm_mxfp8", "gemm_bf16_kernel"):
            c[seg + "_tail_gemms"] += cost(r)
        else:
            c[seg + "tail_small_kernels"] += cost(r)
        if r["gap"] > EAGER_GAP:
            c[seg + "eager_gap_excess"] += r["gap"] - 0.5
        if seg == "D:" and isn(r, "partial_argmax", "finish_argmax", "select_first_argmax", "markov_anchor"):
            c[seg + "markov_argmax_small_kernels"] += cost(r)


def decode(ks, skip_first=1, last=None):
    sp = step_pairs(ks)[skip_first:last]
    c = defaultdict(float)
    for t, d, _, _ in sp:
        for seg, rows in (("T:", t), ("D:", d)):
            layer_ceilings(c, rows, seg)
            tail_ceilings(c, rows, seg)
    # the mHC boundary restructure: the all-reduce consumer does post, collapse, RMSNorm and
    # the MXFP8 cast; the projection partials and Sinkhorn move to a side stream
    for seg in ("T:", "D:"):
        c[seg + "mhc_boundary_restructure"] = sum(c[seg + k] for k in (
            "ar_into_mhc_entry", "finalize_into_mhc_entry", "attn_finalize_emits_fp8", "ffn_finalize_emits_fp8_shared",
            "routed_act_quant_into_producer"))
    n = len(sp)
    span = sum(b - a for _, _, a, b in sp) / n
    return {"steps": n, "step_ms": round(span / 1000, 3),
            "us_per_step": {k: round(v / n, 1) for k, v in sorted(c.items())},
            "percent": {k: round(100 * v / n / span, 2) for k, v in sorted(c.items())}}


def prefill(ks):
    """The 15,395-token prompt's forward (sequence parallel, one chunk)."""
    marks = [k for k in ks if k[2] == MARK]
    rows = rows_of([k for k in ks if marks[0][0] <= k[0] < marks[1][0]])
    span = marks[1][0] - marks[0][0]
    c = defaultdict(float)

    def until_entry(i):
        """The rows after rows[i] up to the next sublayer entry (or standalone mHC post)."""
        for x in rows[i + 1:]:
            if isn(x, "mhc_project_streams", "mhc_post", "mhc_collapse"):
                return
            yield x

    for i, r in enumerate(rows):
        prev = rows[i - 1] if i else None
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if isn(r, "moe_combine"):
            rs = copies = 0.0
            for x in until_entry(i):
                if isn(x, "Memcpy", "void at::"):
                    copies += cost(x)
                if isn(x, "ncclDev"):
                    rs += cost(x)
            c["combine_writes_rs_buffers"] += copies
            c["_moe_rs_exposed"] += rs
            # the exchange can hide under the combine (and the copies it replaces) at best
            c["moe_rs_under_combine"] += min(rs, r["excl"] + copies)
        if nm(r) == "sparse_mla_kernel":
            c["attn_rs_exposed_after_wo_slices"] += sum(cost(x) for x in until_entry(i) if isn(x, "ncclDev"))
        if isn(r, "ncclDevKernel_AllGather"):
            if prev is not None and isn(prev, "mhc_finalize"):
                c["_moe_gather_exposed"] += cost(r)
                c["_moe_entry_compute"] += rows[i - 2]["excl"] + prev["excl"]
            elif prev is not None and isn(prev, "norm_forward_and_per_token_cast"):
                c["_qkv_latent_gather_exposed"] += cost(r)  # attention layers without a compressor
            elif prev is not None and isn(prev, "engram_rows"):
                c["_engram_gather_exposed"] += cost(r)
            elif prev is not None and isn(prev, "indexer_topk"):
                c["_index_layer_gather_exposed"] += cost(r)
            else:
                c["_other_gather_exposed"] += cost(r)  # the embedding's and the forward's last gathers
        if isn(r, "per_token_cast") and r["grid"][0] > 20000:
            c["_moe_input_casts"] += cost(r)
        if isn(r, "gemm_bf16_kernel") and r["grid"][0] == 6:
            c["_router_gemm"] += cost(r)
        if isn(r, "moe_topk_gate"):
            c["_router_topk"] += cost(r)
        if isn(r, "moe_swiglu_cast"):
            c["swiglu_into_fc1"] += cost(r)
        if isn(r, "moe_combine"):
            c["_combine"] += cost(r)
        if isn(r, "metadata_decode"):
            c["_metadata_decode"] += cost(r)
        if isn(r, "mhc_finalize"):
            c["finalize_into_mhc_entry"] += cost(r)
            if nxt is not None and isn(nxt, "per_token_cast") and nxt["grid"][0] < 20000:
                c["attn_finalize_emits_fp8"] += cost(nxt)
        if isn(r, "split_columns") and nxt is not None and isn(nxt, "per_token_cast"):
            c["_attn_cast_after_gather"] += cost(nxt)
        if isn(r, "ncclDev"):
            c["_nccl_exposed"] += cost(r)
        if isn(r, "sparse_mla_kernel"):
            c["_sparse_mla"] += cost(r)
        if isn(r, "moe_grouped"):
            c["_moe_gemms"] += cost(r)
    fp8 = 1 - (5120 + 160) / (2 * 5120)  # E4M3 values plus UE8M0 scales against BF16 rows
    c["quantize_before_moe_gather"] = (fp8 * c["_moe_gather_exposed"] + (1 - 0.125) * c["_moe_input_casts"]
                                       + 0.75 * (c["_router_gemm"] + c["_router_topk"]))
    c["dedupe_moe_input_cast"] = 0.5 * c["_moe_input_casts"]
    c["quantize_before_attn_gather"] = fp8 * c["_qkv_latent_gather_exposed"] + 0.75 * c["_attn_cast_after_gather"]
    c["moe_gather_under_mhc_entry"] = min(c["_moe_gather_exposed"], c["_moe_entry_compute"])
    c["metadata_once_per_group"] = 0.75 * c["_metadata_decode"]
    return {"window_ms": round(span / 1000, 1),
            "ms": {k: round(v / 1000, 2) for k, v in sorted(c.items())},
            "percent": {k: round(100 * v / span, 2) for k, v in sorted(c.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="?", default="results/private/determinism/request-inventory-5/ports")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "ceilings.json"))
    args = ap.parse_args()
    dec = sorted(os.listdir(os.path.join(args.traces, "decode-trace/dgx1")))
    pre = sorted(os.listdir(os.path.join(args.traces, "prefill-trace/dgx1")))
    trace = dict(zip(DECODE, (os.path.join(args.traces, "decode-trace/dgx1", f) for f in dec)))
    out = {"source": args.traces, "decode": {}}
    # the first step of a window carries the request's prefill; sampled's step 24 holds a
    # 26 ms host stall at the request's end; mixed-c8's first two and last steps run
    # before or after the eight requests share the batch
    for name, (skip, last) in (("greedy", (1, None)), ("sampled", (1, 24)), ("mixed-c8", (2, 19))):
        out["decode"][name] = decode(load(trace[name]), skip, last)
    out["prefill-15k"] = prefill(load(os.path.join(args.traces, "prefill-trace/dgx1", pre[0])))
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=1)
        handle.write("\n")
    for name, d in out["decode"].items():
        print(f"{name}: {d['steps']} steps, {d['step_ms']} ms a step")
    print(f"prefill-15k: {out['prefill-15k']['window_ms']} ms")


if __name__ == "__main__":
    main()
