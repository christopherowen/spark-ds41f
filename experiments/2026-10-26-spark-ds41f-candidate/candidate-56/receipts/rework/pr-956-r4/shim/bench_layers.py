"""Per-layer SP prefill step time of DeepSeek V4.1 compressor layers on one GPU at TP4
rank shapes: #956's own mark for round 3 (candidate-56), on candidate-54 (without #957
part 1): the compressor is built as TP4 rank 0 builds it (attn_harness.py ``rank`` mode:
its packed column shard; the compressor's own all-gathers return W copies of the rank's
tensor), so the base's projection without SP and its column gather run at a TP4 rank's
shapes, and the candidate's bind() gathers and unpads the full ratio-2 weight. The
static memory each arm allocates when the layers declare their kernels (bind() included)
is recorded beside the times.

The layers are built exactly as ``attn_harness.py`` builds them (vLLM's own TileLang
attention under the served checkpoint's config, the compressor as a TP-W rank builds
it, random weights in checkpoint dtypes, real caches and metadata), with
``max_num_batched_tokens`` raised to the largest chunk timed. One request's prefill
chunk of T rows (no context) runs sequence-parallel: first every rank once until the
simulated all-gather reaches its fixed point (the real TP result), then the timed
loop runs one rank's forward after another (rank i % W at iteration i), each gather
returning the concatenation of every rank's recorded input. A CUDA event pair around
each forward times it; the GPU is synchronized after every iteration.

What it measures: a rank's own prefill work for the layer (projections, front,
compressor, index query, sparse MLA, cache writes), with each all-gather replaced by
a device copy of the gathered rows. Fabric time is not in it: on one GPU there is no
fabric. The rows each rank sends per gather are recorded, so the bytes on the wire per
row (base: the BF16 hidden state; #956: the MXFP8 q cast, SWA records and the
compressor's FP32 values and gates) come out beside the times.

The largest chunk is 65,535 rows: the sparse MLA extend kernel launches one CTA row per
query row (grid.y), and CUDA caps grid.y at 65,535, so a 65,536-row chunk cannot launch
in either arm (tp4-switched serves chunks of at most 16,384 rows).
"""

import json
import statistics
import sys
import time
import zlib

import torch

sys.path.insert(0, "/shim")
import attn_harness as h  # noqa: E402

DEV = h.DEV


def make_config(world: int, max_tokens: int):
    from types import SimpleNamespace

    from vllm.engine.arg_utils import EngineArgs

    heads, groups = h.per_rank_heads(world)
    args = EngineArgs(
        model=h.MODEL,
        tokenizer=h.MODEL,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        load_format="dummy",
        max_model_len=1048576,
        max_num_seqs=16,
        max_num_batched_tokens=max_tokens,
        block_size=256,
        attention_backend="TILELANG",
        linear_backend="tilelang",
        moe_backend="tilelang",
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"]},
        hf_overrides={"num_attention_heads": heads, "o_groups": groups},
        enforce_eager=False,
    )
    vc = args.create_engine_config()
    vc.speculative_config = SimpleNamespace(
        method="dspark",
        num_speculative_tokens=5,
        num_verification_tokens=16,
        parallel_drafting=False,
        use_dspark=lambda: True,
        use_dflash=lambda: False,
    )
    return vc


def time_layer(vc, mod, groups, world, T, warmup, iters):
    from vllm.forward_context import set_forward_context
    from vllm.models.deepseek_v4_1 import sp_prefill

    hf = vc.model_config.hf_config
    metadata, positions = h.build_metadata(vc, groups, [T], [0])
    g = torch.Generator(device=DEV)
    g.manual_seed(zlib.crc32(f"bench/{T}".encode()) % 100003)
    hidden = (torch.randn((T, hf.hidden_size), generator=g, device=DEV) * 0.5).bfloat16()
    for gr in groups:
        for t in gr.caches.values():
            t.zero_()
    locals_ = []
    for r in range(world):
        rows = sp_prefill.SPRows(T, world, r)
        local = torch.zeros((rows.local_rows, hf.hidden_size), dtype=torch.bfloat16, device=DEV)
        if rows.real_rows:
            local[: rows.real_rows].copy_(hidden[rows.start : rows.start + rows.real_rows])
        locals_.append((rows, local))
    times = {r: [] for r in range(world)}
    with set_forward_context(metadata, vc, num_tokens=T):
        fake = h.FakeTP(world)
        orig = sp_prefill.tensor_model_parallel_all_gather
        sp_prefill.tensor_model_parallel_all_gather = fake.all_gather
        try:
            passes = 0
            for _ in range(6):
                fake.changed = False
                for r, (rows, local) in enumerate(locals_):
                    fake.begin(r, rows.real_rows)
                    with sp_prefill.activate(rows):
                        mod.forward(positions, local)
                torch.cuda.synchronize()
                passes += 1
                if not fake.changed:
                    break
            gathers = list(fake.shapes)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for i in range(warmup + iters):
                r = i % world
                rows, local = locals_[r]
                fake.begin(r, rows.real_rows)
                torch.cuda.synchronize()
                with sp_prefill.activate(rows):
                    start.record()
                    mod.forward(positions, local)
                    end.record()
                torch.cuda.synchronize()
                if i >= warmup:
                    times[r].append(start.elapsed_time(end))
        finally:
            sp_prefill.tensor_model_parallel_all_gather = orig
    allt = [t for r in times for t in times[r]]
    itemsize = {"bfloat16": 2, "float32": 4, "int32": 4, "uint8": 1, "float8_e4m3fn": 1}
    rows0 = sp_prefill.SPRows(T, world, 0).local_rows
    bytes_per_row = sum(
        itemsize[dt] * (int(torch.tensor(shape[1:]).prod()) if len(shape) > 1 else 1)
        for shape, dt in gathers
        if shape[0] == rows0
    )
    return dict(
        T=T,
        passes=passes,
        converged=not fake.changed,
        gathers=gathers,
        gathered_bytes_per_row=bytes_per_row,
        ms_median=statistics.median(allt),
        ms_mean=statistics.fmean(allt),
        ms_stdev=statistics.stdev(allt) if len(allt) > 1 else 0.0,
        ms_min=min(allt),
        ms_per_rank_median={r: statistics.median(v) for r, v in times.items()},
        ms_all=allt,
    )


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", default="4096,8192,16384,32768,65535")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()
    t0 = time.time()
    chunks = [int(x) for x in args.chunks.split(",")]
    vc = make_config(args.world, max(chunks))
    h.init_dist(vc)
    mods, info = h.build_attention(vc, args.world, mode="rank")
    pages = max(96, -(-max(chunks) // 128) + 8)
    groups = h.bind_caches(vc, mods, pages_per_group=pages)
    print(f"built in {time.time() - t0:.1f}s; layers {json.dumps(info)}", flush=True)
    results = []
    for T in chunks:
        for layer, mod in mods.items():
            res = time_layer(vc, mod, groups, args.world, T, args.warmup, args.iters)
            res["layer"] = layer
            results.append(res)
            print(json.dumps({k: v for k, v in res.items() if k != "ms_all"}), flush=True)
    with open(f"{args.out}.json", "w") as f:
        json.dump(dict(arm=args.arm, world=args.world, compressor_mode="rank",
                       layers=info, results=results,
                       warmup=args.warmup, iters=args.iters,
                       seconds=round(time.time() - t0, 1)), f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
