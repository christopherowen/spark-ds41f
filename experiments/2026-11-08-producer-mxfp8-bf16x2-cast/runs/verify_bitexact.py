import sys
import torch

from vllm.models.deepseek_v4_1.tilelang import (
    activation,
    columns,
    front,
    linear,
    mhc,
    moe,
    norm,
    quant_common,
)

mode = sys.argv[1]  # "ref", "cand_130", "cand_135"
ref_path = "/host_tmp/bitexact_ref.pt"

device = torch.device("cuda:0")
torch.manual_seed(20261108)


def make_edge_bf16(shape):
    x = torch.randn(shape, device=device, dtype=torch.bfloat16)
    if shape[0] >= 4:
        x[0].zero_()
        x[1].fill_(-0.0)
        x[2] = torch.randn(shape[1:], device=device, dtype=torch.bfloat16) * 1e-6
        x[3] = torch.randn(shape[1:], device=device, dtype=torch.bfloat16) * 500.0
    return x


def run_all(inputs):
    outputs = {}

    # 1. join_columns_and_cast_mxfp8
    for m, parts in inputs["join_columns"].items():
        out_bf16 = torch.empty(
            (m, sum(p.shape[1] for p in parts)), device=device, dtype=torch.bfloat16
        )
        cast = columns.join_columns_and_cast_mxfp8(parts, out_bf16)
        outputs[f"join_columns_m{m}_bf16"] = out_bf16.view(torch.uint16).cpu()
        outputs[f"join_columns_m{m}_vals"] = cast.values[:m].view(torch.uint8).cpu()
        outputs[f"join_columns_m{m}_sfs"] = cast.scales[:m].view(torch.uint8).cpu()

    # 2. cast_mxfp8 / per_token_cast
    for m, x in inputs["cast_mxfp8"].items():
        cast = linear.cast_mxfp8(x)
        outputs[f"cast_mxfp8_m{m}_vals"] = cast.values[:m].view(torch.uint8).cpu()
        outputs[f"cast_mxfp8_m{m}_sfs"] = cast.scales[:m].view(torch.uint8).cpu()
        vals2, sfs2 = quant_common.per_token_cast(
            x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True
        )
        outputs[f"per_token_cast_m{m}_vals"] = vals2.view(torch.uint8).cpu()
        outputs[f"per_token_cast_m{m}_sfs"] = sfs2.view(torch.uint8).cpu()

    # 3. activation_swiglu_clamp_cast
    act_mod = activation.SiluAndMulClampFP32(7.0, width=1536, mxfp8_output=True)
    for m, x in inputs["act_swiglu"].items():
        cast = act_mod.forward_mxfp8(x)
        outputs[f"act_swiglu_m{m}_vals"] = cast.values[:m].view(torch.uint8).cpu()
        outputs[f"act_swiglu_m{m}_sfs"] = cast.scales[:m].view(torch.uint8).cpu()

    # 4. moe_swiglu_cast
    fn_moe = moe.moe_swiglu_cast(768, 16, 7.0)
    for tag, (h, order, tile_expert, weights) in inputs["moe_swiglu"].items():
        rows = h.shape[0]
        hq = torch.empty((rows, 768), device=device, dtype=torch.float8_e4m3fn)
        h_sf = torch.empty((rows, 24), device=device, dtype=torch.uint8)
        fn_moe(h, hq, h_sf, order, weights, tile_expert)
        live = order >= 0
        outputs[f"moe_swiglu_{tag}_vals"] = hq[live].view(torch.uint8).cpu()
        outputs[f"moe_swiglu_{tag}_sfs"] = h_sf[live].view(torch.uint8).cpu()

    # 5. mhc._finalize
    fn_fin = mhc._finalize(5120, 20480, 1e-6, 1e-4, 20)
    for m, (P, S, C, Q, scale, base, weight) in inputs["mhc"].items():
        post = torch.empty((m, 4), device=device, dtype=torch.float32)
        comb = torch.empty((m, 4, 4), device=device, dtype=torch.float32)
        pre_out = torch.empty((m, 4), device=device, dtype=torch.float32)
        y = torch.empty((m, 5120), device=device, dtype=torch.bfloat16)
        yq = torch.empty((m, 5120), device=device, dtype=torch.float8_e4m3fn)
        ysf = torch.empty((m, 160), device=device, dtype=torch.uint8)
        fn_fin(P, S, C, Q, scale, base, weight, post, comb, pre_out, y, yq, ysf)
        outputs[f"mhc_m{m}_post"] = post.cpu()
        outputs[f"mhc_m{m}_comb"] = comb.cpu()
        outputs[f"mhc_m{m}_pre"] = pre_out.cpu()
        outputs[f"mhc_m{m}_y"] = y.view(torch.uint16).cpu()
        outputs[f"mhc_m{m}_yq"] = yq.view(torch.uint8).cpu()
        outputs[f"mhc_m{m}_ysf"] = ysf.view(torch.uint8).cpu()

    # 6. norm_forward_and_per_token_cast
    for m, (x, res, w) in inputs["norm"].items():
        (out_q, out_sf), out_bf16, _, _ = norm.norm_forward_and_per_token_cast(
            x=x.clone(),
            weight=w,
            eps=1e-6,
            fmt="e4m3",
            num_per_channels=32,
            residual=res.clone(),
            out_scale=1.0,
            use_tma_aligned_col_major_sf=False,
            round_sf=True,
            use_packed_ue8m0=True,
            with_out_bf16=True,
        )
        outputs[f"norm_m{m}_bf16"] = out_bf16.view(torch.uint16).cpu()
        outputs[f"norm_m{m}_vals"] = out_q.view(torch.uint8).cpu()
        outputs[f"norm_m{m}_sfs"] = out_sf.view(torch.uint8).cpu()

    # 7. attention_front
    for m, (qa_kv, q_w, kv_w, pos, cos_sin, slots) in inputs["front"].items():
        cache = torch.zeros((16, 64, 528), device=device, dtype=torch.uint8)
        qn = torch.empty((m, 1280), device=device, dtype=torch.bfloat16)
        q_values = torch.empty((m, 1280), device=device, dtype=torch.float8_e4m3fn)
        q_scales = torch.empty((m, 40), device=device, dtype=torch.uint8)
        kv = torch.empty((m, 512), device=device, dtype=torch.bfloat16)
        front.attention_front(
            qa_kv, q_w, kv_w, 1e-6, pos, cos_sin, cache, slots, 64, qn, q_values, q_scales, kv
        )
        outputs[f"front_m{m}_qn"] = qn.view(torch.uint16).cpu()
        outputs[f"front_m{m}_qvals"] = q_values.view(torch.uint8).cpu()
        outputs[f"front_m{m}_qscales"] = q_scales.view(torch.uint8).cpu()
        outputs[f"front_m{m}_kv"] = kv.view(torch.uint16).cpu()
        outputs[f"front_m{m}_cache"] = cache.cpu()

    torch.cuda.synchronize()
    return outputs


if mode == "ref":
    inputs = {
        "join_columns": {
            m: [make_edge_bf16((m, 5120)) for _ in range(3)]
            for m in (1, 4, 16, 64)
        },
        "cast_mxfp8": {
            m: make_edge_bf16((m, 5120)) for m in (1, 4, 16, 64, 256)
        },
        "act_swiglu": {
            m: make_edge_bf16((m, 3072)) for m in (1, 4, 16, 64, 256)
        },
        "moe_swiglu": {},
        "mhc": {},
        "norm": {
            m: (
                make_edge_bf16((m, 5120)),
                make_edge_bf16((m, 5120)),
                torch.randn(5120, device=device, dtype=torch.bfloat16),
            )
            for m in (1, 4, 16, 64)
        },
        "front": {},
    }
    for tiles in (2, 8):
        block_m = 16
        rows = tiles * block_m
        h = make_edge_bf16((rows, 1536))
        order = torch.arange(rows, device=device, dtype=torch.int32)
        order[-3:] = -1
        tile_expert = torch.arange(tiles, device=device, dtype=torch.int32)
        weights = torch.rand(rows, device=device, dtype=torch.float32) + 0.1
        inputs["moe_swiglu"][f"t{tiles}"] = (h, order, tile_expert, weights)

    for m in (1, 4, 16, 64):
        P = torch.randn((40, m, 1, 24), device=device, dtype=torch.float32) * 0.01
        S = torch.rand((40, m, 1), device=device, dtype=torch.float32) * 0.1 + 0.01
        C = make_edge_bf16((m, 5120))
        Q = torch.rand((40, m), device=device, dtype=torch.float32) * 0.1 + 0.01
        scale = torch.rand(3, device=device, dtype=torch.float32) * 0.1 + 0.01
        base = torch.randn(24, device=device, dtype=torch.float32) * 0.1
        weight = torch.randn(5120, device=device, dtype=torch.bfloat16)
        inputs["mhc"][m] = (P, S, C, Q, scale, base, weight)

    for m in (1, 4, 16, 64):
        qa_kv = make_edge_bf16((m, 1792))
        q_w = torch.randn(1280, device=device, dtype=torch.bfloat16)
        kv_w = torch.randn(512, device=device, dtype=torch.bfloat16)
        pos = torch.arange(m, device=device, dtype=torch.int64)
        cos_sin = torch.randn((256, 64), device=device, dtype=torch.float32)
        slots = torch.arange(m, device=device, dtype=torch.int64)
        inputs["front"][m] = (qa_kv, q_w, kv_w, pos, cos_sin, slots)

    ref_outputs = run_all(inputs)
    torch.save({"inputs": inputs, "outputs": ref_outputs}, ref_path)
    print(f"Saved {len(ref_outputs)} reference tensors to {ref_path}")
else:
    saved = torch.load(ref_path, map_location=device, weights_only=False)
    cand_outputs = run_all(saved["inputs"])
    mismatches = 0
    for k, ref_v in saved["outputs"].items():
        ref_v = ref_v.cpu()
        cand_v = cand_outputs[k]
        eq = torch.equal(ref_v, cand_v)
        if not eq:
            diff = (ref_v != cand_v).sum().item()
            print(f"MISMATCH [{mode}] {k}: {diff} / {ref_v.numel()} elements differ")
            mismatches += 1
    print(f"[{mode}] Checked {len(cand_outputs)} output tensors: {mismatches} mismatches")
    assert mismatches == 0, f"{mismatches} mismatches found!"
