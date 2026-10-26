"""The IKET prototype's hot kernels at TP3 decode shapes, on one GPU.

usage: iket_hot.py off|on --out DIR [--iket DIR] [--rows N] [--iters N] [--delay-us N]

- ``moe``: DS4.1's routed experts (MoeExpertsV41Plan(384, 6, 5120, 768), TP3's 768
  columns a rank): dispatch, the gate/up grouped FP8 x FP4 GEMM, the fused SwiGLU,
  the down GEMM and the combine;
- ``mla``: sparse MLA decode at TP3 (24 heads, a 128-entry window and 512 indexed
  entries), the ``split`` and ``decode`` routes;
- ``mhc``: a sublayer's mHC entry (``project_streams`` then ``finalize``, mode
  ``pre``);
- ``allreduce``: sparknet's one-shot all-reduce kernel at TP3 geometry (rank 0 of 3,
  two lanes a peer, 512 threads), on this GPU alone. A host thread stands in for the
  two peers' proxies: it pre-stages their payloads, waits for the kernel's doorbell,
  then (after ``--delay-us``) writes their flags, so the kernel's flag wait is the
  collective wait a decode step's all-reduce has.

Every DS4.1 kernel goes through declare and prepare (``prepare.declare`` outside a
runner prepares at once), then TileLang's cache is frozen before the first launch,
as the worker does. ``on`` enables the IKET mode first (``iket.enable(--iket)``).

Writes to ``--out``: ``result.json`` (per family the output tensors' SHA-256, the
per-call time in us over ``--iters`` launches after one warm launch, and the
modules and callback state), and for ``off`` every compiled kernel's CUDA source
under ``sources/`` (``<kernel>.<sha12>.cu``) with ``sources.json``.
"""

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import torch

DEVICE = "cuda"
ROWS = 8
HIDDEN, EXPERTS, TOPK, INTER = 5120, 384, 6, 768  # TP3 rank of DS4.1-Flash
HEADS, SWA_WIDTH, IDX_WIDTH, SWA_PAGE, IDX_PAGE = 24, 128, 512, 128, 128
RMS_EPS, HC_EPS, SINKHORN = 1e-20, 1e-6, 20
WORLD, LANES, THREADS, BLOCKS = 3, 2, 512, 8


def gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def digest(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def timed(call, iters):
    call()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    stop.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(stop) * 1000 / iters, 2)


# ---------------------------------------------------------------- routed experts


def moe_case(rows):
    from vllm.models.deepseek_v4_1.tilelang.linear import cast_mxfp8
    from vllm.models.deepseek_v4_1.tilelang.moe import MoeExpertsV41Plan, workspace_bytes

    plan = MoeExpertsV41Plan(EXPERTS, TOPK, HIDDEN, INTER, swiglu_limit=10.0)
    plan.declare_kernels()  # prepared and bound now, outside a runner
    g = gen(1)

    def experts(n, k):
        packed = torch.randint(0, 256, (EXPERTS, n, k // 2), generator=g, device=DEVICE, dtype=torch.uint8)
        exps = torch.randint(118, 122, (EXPERTS, n, k // 32), generator=g, device=DEVICE).to(torch.uint8)
        return packed, exps

    weights = plan.prepare_weights(*experts(2 * INTER, HIDDEN), *experts(HIDDEN, INTER))
    x = (torch.randn((rows, HIDDEN), generator=g, device=DEVICE) * 0.5).bfloat16()
    ids = torch.stack([torch.randperm(EXPERTS, generator=g, device=DEVICE)[:TOPK] for _ in range(rows)])
    ids = ids.to(torch.int32).contiguous()
    route_weights = (torch.rand((rows, TOPK), generator=g, device=DEVICE) + 0.1).contiguous()
    xq = cast_mxfp8(x)
    out = torch.empty_like(x)
    workspace = torch.empty(workspace_bytes(plan.workspace_specs(rows, torch.int32)),
                            dtype=torch.uint8, device=DEVICE)
    return (lambda: plan.run(xq, ids, route_weights, weights, out, workspace)), (out,)


# ---------------------------------------------------------------- sparse MLA


def mla_case(rows):
    from vllm.models.deepseek_v4_1.tilelang import prepare
    from vllm.models.deepseek_v4_1.tilelang.sparse_mla import (
        HEAD_DIM, IDX_GROUP, SWA_GROUP, SparseMlaV41Plan)

    args = (HEADS, SWA_WIDTH, IDX_WIDTH, IDX_PAGE)
    kwargs = dict(swa_page_size=SWA_PAGE, max_rows=64)
    prepare.declare(lambda: SparseMlaV41Plan.calls(*args, **kwargs))
    plan = SparseMlaV41Plan(*args, **kwargs)
    g = gen(2)
    swa_pages, idx_pages, context = 24, 40, 4096

    def pack_swa(kv):
        groups = kv.float().reshape(kv.shape[0], HEAD_DIM // SWA_GROUP, SWA_GROUP)
        exponent = torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0)).clamp(-126, 127)
        payload = (groups / torch.exp2(exponent).unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        return torch.cat([payload.view(torch.uint8).reshape(kv.shape[0], HEAD_DIM),
                          (exponent + 127).to(torch.uint8)], -1).contiguous()

    def pack_idx(kv):
        groups = kv.float().reshape(kv.shape[0], HEAD_DIM // IDX_GROUP, IDX_GROUP)
        scales = (groups.abs().amax(-1).clamp_min(6 * 2**-9) / 6.0).clamp(max=448).to(torch.float8_e4m3fn)
        normalized = groups / scales.float().unsqueeze(-1)
        lut = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=DEVICE)
        codes = (normalized.abs().clamp(max=6.0).unsqueeze(-1) - lut).abs().argmin(-1).to(torch.uint8)
        codes = (codes | (torch.signbit(normalized).to(torch.uint8) << 3)).reshape(kv.shape[0], HEAD_DIM)
        return torch.cat([codes[:, 0::2] | (codes[:, 1::2] << 4), scales.view(torch.uint8)], -1).contiguous()

    def paged(records, page):
        return records.view(records.shape[0] // page, -1).contiguous()

    def lists(width, extent, min_len):
        lengths = torch.randint(min_len, width + 1, (rows,), generator=g, device=DEVICE, dtype=torch.int32)
        chosen = torch.stack([torch.randperm(extent, generator=g, device=DEVICE)[:width] for _ in range(rows)])
        keep = torch.arange(width, device=DEVICE) < lengths[:, None]
        return torch.where(keep, chosen, -1).to(torch.int32).contiguous(), lengths

    q = torch.randn((rows, HEADS, HEAD_DIM), generator=g, device=DEVICE).bfloat16()
    sinks = torch.randn((HEADS,), generator=g, device=DEVICE)
    swa = paged(pack_swa(torch.randn((swa_pages * SWA_PAGE, HEAD_DIM), generator=g, device=DEVICE)), SWA_PAGE)
    idx = paged(pack_idx(torch.randn((idx_pages * IDX_PAGE, HEAD_DIM), generator=g, device=DEVICE)), IDX_PAGE)
    # Full windows and index lists, as a long decode context has.
    swa_idx, swa_len = lists(SWA_WIDTH, swa_pages * SWA_PAGE, SWA_WIDTH)
    idx_logical, idx_len = lists(IDX_WIDTH, context, IDX_WIDTH)
    table = torch.stack([torch.randperm(idx_pages, generator=g, device=DEVICE)[: context // IDX_PAGE]
                         for _ in range(rows)]).to(torch.int32)
    positions = torch.randint(0, 4096, (rows,), generator=g, device=DEVICE)
    exponents = torch.arange(0, 64, 2, device=DEVICE).double() / 64
    angles = torch.arange(4096, device=DEVICE).double()[:, None] * (1.0 / 10000**exponents)
    rope = torch.cat((angles.cos(), angles.sin()), dim=-1).float()
    outs = {}
    calls = []
    for kernel in [k for k in ("split", "decode") if k in plan.kernels]:
        out = torch.empty((rows, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
        scratch = [torch.empty(shape, dtype=dtype, device=DEVICE) for shape, dtype in plan.scratch_specs(rows, kernel)]
        outs[kernel] = out

        def call(kernel=kernel, out=out, scratch=scratch):
            plan.run(q, swa, swa_idx, swa_len, sinks, positions, rope, out, idx_cache=idx,
                     idx_indices=idx_logical, idx_lengths=idx_len, idx_page_table=table,
                     kernel=kernel, scratch=scratch)

        calls.append(call)
    return (lambda: [c() for c in calls]), tuple(outs.values())


# ---------------------------------------------------------------- mHC entry


def mhc_case(rows):
    from vllm.models.deepseek_v4_1.tilelang import mhc, prepare
    from vllm.models.deepseek_v4_1.tilelang.gemm import scale_words

    k = mhc.MULT * HIDDEN
    prepare.declare(lambda: [mhc._project_call(HIDDEN, "pre", 1),
                             mhc._finalize_call(HIDDEN, k, RMS_EPS, HC_EPS, SINKHORN)])
    project = mhc._project(HIDDEN, "pre", 1)
    finalize = mhc._finalize(HIDDEN, k, RMS_EPS, HC_EPS, SINKHORN)
    g = gen(3)
    splits = HIDDEN // mhc.BLOCK_H
    f32 = dict(device=DEVICE, dtype=torch.float32)
    residual = (torch.randn((rows, mhc.MULT, HIDDEN), generator=g, device=DEVICE)).bfloat16()
    fn = torch.randn((mhc.MIXES, k), generator=g, **f32) * 0.01
    scale = torch.rand((3,), generator=g, **f32) + 0.5
    base = torch.randn((mhc.MIXES,), generator=g, **f32) * 0.1
    norm = (torch.rand((HIDDEN,), generator=g, device=DEVICE) + 0.5).bfloat16()
    pre = torch.softmax(torch.randn((rows, mhc.MULT), generator=g, **f32), -1)
    post = torch.empty((rows, mhc.MULT), **f32)
    comb = torch.empty((rows, mhc.MULT, mhc.MULT), **f32)
    pre_out = torch.empty((rows, mhc.MULT), **f32)
    y = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
    y_q = torch.empty((rows, HIDDEN), dtype=torch.float8_e4m3fn, device=DEVICE)
    y_sf = torch.empty((rows, 4 * scale_words(HIDDEN)), dtype=torch.uint8, device=DEVICE)
    P = torch.empty((splits, rows, 1, mhc.MIXES), **f32)
    S = torch.empty((splits, rows, 1), **f32)
    C = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
    Q = torch.empty((splits, rows), **f32)
    source = residual.view(rows, k)
    mix_post = torch.empty((rows, mhc.MULT), **f32)
    mix_comb = torch.empty((rows, mhc.MULT, mhc.MULT), **f32)

    def call():
        # mode "pre" (as _mhc_pre takes it): the streams update in place.
        project(y.unsqueeze(0), source, mix_post, mix_comb, pre, fn, source, P, S, C, Q)
        finalize(P, S, C, Q, scale, base, norm, post, comb, pre_out, y, y_q, y_sf)

    return call, (post, comb, pre_out, y, y_q, y_sf)


# ---------------------------------------------------------------- one-shot all-reduce


def allreduce_case(rows, delay_us):
    from sparknet.oneshot import _oneshot_tilelang as oneshot
    from sparknet.oneshot import runtime
    from sparknet.oneshot._kernels import Launch
    from sparknet.oneshot._proxy import Layout

    nbytes = rows * HIDDEN * 2
    slot_bytes = runtime._align_up(max(nbytes, 1 << 20), runtime._SLOT_ALIGNMENT)
    layout = Layout(WORLD, slot_bytes)
    run = oneshot.get_launcher("bfloat16", WORLD, 0, THREADS, layout.slots, layout.flag_stride, LANES, 0)
    region = torch.zeros(layout.total_bytes, dtype=torch.uint8, pin_memory=True)
    classes = BLOCKS.bit_length()
    # sparknet 0.5.0 (the image's) adds two words after the stage and tail counters:
    # the rung doorbell's sequence and block 0's arrival, which the other blocks wait on.
    handoff = "rung_index" in Launch.__dataclass_fields__
    counters = torch.zeros(1 + 2 * classes + 2 * handoff, dtype=torch.int32, device=DEVICE)
    g = gen(4)
    parts = [torch.randn((rows, HIDDEN), generator=g, device=DEVICE).bfloat16() for _ in range(WORLD)]
    out = torch.empty_like(parts[0])
    raw = region.numpy()
    # The peers' payloads, in both slots of their receive areas.
    for peer in range(1, WORLD):
        payload = parts[peer].view(torch.uint8).reshape(-1).cpu().numpy()
        for slot in range(layout.slots):
            at = layout.recv_off + (peer * layout.slots + slot) * slot_bytes
            raw[at : at + nbytes] = payload
    ctrl = raw[layout.ctrl_off : layout.ctrl_off + 64].view("uint32")
    ctrl[runtime.CTRL_TIMEOUT_WORD] = 10_000_000  # us
    flags = raw.view("uint32")
    grid = runtime._grid_blocks(nbytes // 16, THREADS, BLOCKS)
    stage, tail = 1 + grid.bit_length() - 1, 1 + classes + grid.bit_length() - 1
    extra = dict(rung_index=1 + 2 * classes, arrival_index=2 + 2 * classes) if handoff else {}
    launch = Launch(input=parts[0].reshape(-1).view(torch.int32), output=out.reshape(-1).view(torch.int32),
                    size_packs=nbytes // 16, nbytes=nbytes, region=region, recv_off=layout.recv_off,
                    flag_off=layout.flag_off, send_off=layout.send_off, ctrl_off=layout.ctrl_off,
                    slot_bytes=slot_bytes, counters=counters, stage_index=stage, tail_index=tail, grid_x=grid,
                    **extra)
    def peers():
        """Each launch rings the doorbell with the next sequence number; answer it."""
        seq = 1
        while True:
            while int(ctrl[0]) != seq:
                pass
            if delay_us:
                until = time.perf_counter() + delay_us / 1e6
                while time.perf_counter() < until:
                    pass
            slot = seq & 1
            for peer in range(1, WORLD):
                for lane in range(LANES):
                    word = layout.flag_off + ((peer * layout.slots + slot) * LANES + lane) * layout.flag_stride
                    flags[word // 4] = seq
            seq += 1

    started = []

    def call():
        if not started:
            # It starts with the first launch, after the other families ran. The
            # polling thread holds the GIL between switches; switch often, so
            # launches are not held back by it.
            sys.setswitchinterval(1e-5)
            threading.Thread(target=peers, daemon=True).start()
            started.append(True)
        run(launch)
        torch.cuda.synchronize()

    total = parts[0].float()
    for p in parts[1:]:
        total = total + p.float()
    return call, (out,), total.bfloat16()


# ---------------------------------------------------------------- main


def sources(out_dir):
    """Every compiled kernel's CUDA source, from TileLang's in-memory kernel cache."""
    import tilelang.cache

    index = {}
    (out_dir / "sources").mkdir(parents=True, exist_ok=True)
    for cache in tilelang.cache._dispatch_map.values():
        for kernel in cache._memory_cache.values():
            source = kernel.get_kernel_source()
            name = f"{kernel.prim_func.attrs['global_symbol']}.{hashlib.sha256(source.encode()).hexdigest()[:12]}"
            (out_dir / "sources" / f"{name}.cu").write_text(source)
            index[name] = len(source)
    return dict(sorted(index.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("off", "on"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--iket")
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--delay-us", type=float, default=0.0)
    parser.add_argument("--only", nargs="*", default=["moe", "mla", "mhc", "allreduce"])
    parser.add_argument("--sample-ctas", type=int, default=0, help="VLLM_DS41_IKET_CTAS")
    parser.add_argument("--sample-iters", type=int, default=0, help="VLLM_DS41_IKET_ITERS")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    if args.mode == "on":
        from vllm.models.deepseek_v4_1.tilelang import iket

        iket.enable(args.iket, args.sample_ctas, args.sample_iters)
    from vllm.models.deepseek_v4_1.tilelang import prepare

    builders = {"moe": moe_case, "mla": mla_case, "mhc": mhc_case}
    cases = {}
    for name in args.only:
        started = time.monotonic()
        if name == "allreduce":
            call, outputs, expected = allreduce_case(args.rows, args.delay_us)
            cases[name] = (call, outputs, expected)
        else:
            call, outputs = builders[name](args.rows)
            cases[name] = (call, outputs, None)
        print(f"{name}: prepared in {time.monotonic() - started:.1f} s", flush=True)
    prepare.freeze("iket_hot.py: every kernel was prepared before the first launch")

    result = {"mode": args.mode, "rows": args.rows, "iters": args.iters, "families": {}}
    for name, (call, outputs, expected) in cases.items():
        call()
        torch.cuda.synchronize()
        family = {"sha256": digest(*outputs)}
        if expected is not None:
            family["rank_order_sum_equal"] = bool(torch.equal(outputs[0], expected))
        family["us_per_call"] = timed(call, args.iters)
        result["families"][name] = family
        print(name, json.dumps(family), flush=True)

    import tvm_ffi

    result["modules"] = {
        name: name in sys.modules
        for name in ("tilelang.tools.cuda.iket", "vllm.models.deepseek_v4_1.tilelang.iket")
    }
    result["postproc_callback"] = tvm_ffi.get_global_func("tilelang_callback_cuda_postproc",
                                                          allow_missing=True) is not None
    if args.mode == "off":
        result["sources"] = sources(out_dir)
    (out_dir / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    main()
