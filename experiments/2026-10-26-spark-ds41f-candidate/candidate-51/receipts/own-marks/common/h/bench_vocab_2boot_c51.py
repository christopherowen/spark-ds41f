import hashlib
import json
import sys
import torch
import vllm.models.deepseek_v4_1.tilelang.vocab as vocab
from vllm.models.deepseek_v4_1.tilelang.vocab import (
    pack_bf16_weight,
    prepare_packed_head,
    project_head,
    project_head_greedy_pairs,
    row_tile,
)
from vllm.v1.worker.gpu.sample.tilelang_sampling import greedy_shard_pair

DEVICE = torch.device("cuda:0")
K = 5120
MS = [1, 6, 16, 32, 48, 64, 72, 96, 97, 104, 107, 112, 120, 128]  # c51 own marks: + 97, 107, 120
SHAPES = {
    "tp4": (32320, 32320),
    "tp3": (43136, 43072),
}


def bench_graph(fn, warmup=20, iters=200):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end) * 1000.0 / iters, 2)


def bench_eager(fn, warmup=20, iters=200):  # c51 own marks: eager loop beside the graphs
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return round(start.elapsed_time(end) * 1000.0 / iters, 2)


def sha(t):  # c51 own marks: output hash for the cross-image bit-exactness check
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def make_layer(packed, out_dtype):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(packed.values, requires_grad=False)
    layer.weight_exception_rows = torch.nn.Parameter(
        packed.exception_rows, requires_grad=False
    )
    layer.weight_exception_entries = torch.nn.Parameter(
        packed.exception_entries, requires_grad=False
    )
    layer.weight_row_offsets = torch.nn.Parameter(
        packed.row_offsets, requires_grad=False
    )
    layer.packed_bf16_in_features = K
    layer.packed_bf16_exponent_base = packed.exponent_base
    layer.packed_bf16_exceptions = packed.exceptions
    prepare_packed_head(layer, out_dtype=out_dtype)
    return layer


def run_shape(shape_name, n, org_n):
    gen = torch.Generator(device=DEVICE).manual_seed(20261108 + n)
    w = (torch.randn(n, K, generator=gen, device=DEVICE) * 0.02).bfloat16()
    w[0, 0] = 200.0
    w[100, 50] = -150.0
    packed = pack_bf16_weight(w, in_place=True)

    max_M = 128
    x_all = torch.randn(max_M, K, generator=gen, device=DEVICE, dtype=torch.bfloat16)
    num_tiles = (n + 31) // 32
    p_vals = torch.empty((max_M, num_tiles), dtype=torch.float32, device=DEVICE)
    p_idxs = torch.empty((max_M, num_tiles), dtype=torch.int32, device=DEVICE)
    sp_parts = 1024
    sp_vals = torch.empty((max_M, sp_parts), dtype=torch.float32, device=DEVICE)
    sp_idxs = torch.empty((max_M, sp_parts), dtype=torch.int32, device=DEVICE)

    results = {}

    for dtype_name, out_dtype in [
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
    ]:
        out_all = torch.empty((max_M, n), dtype=out_dtype, device=DEVICE)
        pairs_all = torch.empty((max_M, 4), dtype=torch.float32, device=DEVICE)

        # 1. base_1107 (TILE_CONFIGS[128] with num_stages=4, act_stages=2)
        vocab.TILE_CONFIGS[128] = dict(
            block_N=32, threads=256, step_groups=1, num_stages=4, act_stages=2
        )
        layer_base128 = make_layer(packed, out_dtype)

        # 2. candidate (TILE_CONFIGS[128] with num_stages=2)
        vocab.TILE_CONFIGS[128] = dict(
            block_N=32, threads=256, step_groups=1, num_stages=2
        )
        layer_cand128 = make_layer(packed, out_dtype)

        ker64 = layer_cand128.tilelang_vocab[64]
        # c51 own marks: with #997's certified screen in the base, tilelang_vocab_greedy[64]
        # holds the screen's 7 kernels, so the 64-row greedy-partials reference the PR
        # unpacked from it is compiled directly, as 1107's head declared it.
        cfg64 = vocab.tile_config(64, K)
        gker64 = vocab.packed_vocab_greedy_partials(
            n, K, packed.exponent_base, block_M=64, out_dtype=vocab._OUT[out_dtype], **cfg64
        )
        fker = vocab.finish_vocab_greedy_pairs((n + cfg64["block_N"] - 1) // cfg64["block_N"])

        dtype_res = {}
        for m in MS:
            x = x_all[:m]
            out = out_all[:m]
            pairs = pairs_all[:m]
            pv = p_vals[:m]
            pi = p_idxs[:m]
            sv = sp_vals[:m]
            si = sp_idxs[:m]

            # Verify exact bit-identity across all three configurations
            out_64 = torch.empty((m, n), dtype=out_dtype, device=DEVICE)
            ker64(
                x,
                layer_cand128.weight,
                layer_cand128.weight_exception_rows,
                layer_cand128.weight_exception_entries,
                layer_cand128.weight_row_offsets,
                out_64,
            )
            out_b128 = project_head(layer_base128, x).clone()
            out_c128 = project_head(layer_cand128, x).clone()
            assert torch.equal(out_64, out_b128), f"proj mismatch 64 vs base128 at M={m}"
            assert torch.equal(out_b128, out_c128), f"proj mismatch base128 vs cand128 at M={m}"

            g64 = torch.empty((m, 4), dtype=torch.float32, device=DEVICE)
            gker64(
                x,
                layer_cand128.weight,
                layer_cand128.weight_exception_rows,
                layer_cand128.weight_exception_entries,
                layer_cand128.weight_row_offsets,
                pv,
                pi,
                org_n,
            )
            fker(pv, pi, g64, 0)
            gb = project_head_greedy_pairs(
                layer_base128, x, vocab=org_n, vocab_start=0, pair_width=4, scratch_pairs=False
            ).clone()
            gc = project_head_greedy_pairs(
                layer_cand128, x, vocab=org_n, vocab_start=0, pair_width=4, scratch_pairs=False
            ).clone()
            ref_pairs = torch.empty((m, 4), dtype=torch.float32, device=DEVICE)
            greedy_shard_pair(out_c128, org_n, 0, ref_pairs, sv, si)
            assert torch.equal(g64[:, :2], gb[:, :2]), f"greedy mismatch 64 vs base128 at M={m}"
            assert torch.equal(gb[:, :2], gc[:, :2]), f"greedy mismatch base128 vs cand128 at M={m}"
            assert torch.equal(gc[:, :2], ref_pairs[:, :2]), f"greedy mismatch cand128 vs greedy_shard_pair at M={m}"

            row_entry = {
                "M": m,
                "row_tile": row_tile(m),
                "bit_equal": True,
                "proj_hash": sha(out_c128),
                "greedy_hash": sha(gc[:, :2]),
                "proj_tile64_us": bench_graph(
                    lambda: ker64(
                        x,
                        layer_cand128.weight,
                        layer_cand128.weight_exception_rows,
                        layer_cand128.weight_exception_entries,
                        layer_cand128.weight_row_offsets,
                        out,
                    )
                ),
                "proj_base1107_us": bench_graph(
                    lambda: project_head(layer_base128, x, out=out)
                ),
                "proj_candidate_us": bench_graph(
                    lambda: project_head(layer_cand128, x, out=out)
                ),
                "proj_plus_argmax_base1107_us": bench_graph(
                    lambda: (
                        project_head(layer_base128, x, out=out),
                        greedy_shard_pair(out, org_n, 0, pairs, sv, si),
                    )
                ),
                "proj_plus_argmax_candidate_us": bench_graph(
                    lambda: (
                        project_head(layer_cand128, x, out=out),
                        greedy_shard_pair(out, org_n, 0, pairs, sv, si),
                    )
                ),
                "greedy_epilogue_tile64_us": bench_graph(
                    lambda: (
                        gker64(
                            x,
                            layer_cand128.weight,
                            layer_cand128.weight_exception_rows,
                            layer_cand128.weight_exception_entries,
                            layer_cand128.weight_row_offsets,
                            pv,
                            pi,
                            org_n,
                        ),
                        fker(pv, pi, pairs, 0),
                    )
                ),
                "greedy_epilogue_base1107_us": bench_graph(
                    lambda: project_head_greedy_pairs(
                        layer_base128, x, vocab=org_n, vocab_start=0, pair_width=4, scratch_pairs=True
                    )
                ),
                "greedy_epilogue_candidate_us": bench_graph(
                    lambda: project_head_greedy_pairs(
                        layer_cand128, x, vocab=org_n, vocab_start=0, pair_width=4, scratch_pairs=True
                    )
                ),
            }
            # c51 own marks: eager loops of the four headline cells
            row_entry["eager_proj_base1107_us"] = bench_eager(lambda: project_head(layer_base128, x, out=out))
            row_entry["eager_proj_candidate_us"] = bench_eager(lambda: project_head(layer_cand128, x, out=out))
            row_entry["eager_greedy_epilogue_base1107_us"] = bench_eager(
                lambda: project_head_greedy_pairs(
                    layer_base128, x, vocab=org_n, vocab_start=0, pair_width=4, scratch_pairs=True
                )
            )
            row_entry["eager_greedy_epilogue_candidate_us"] = bench_eager(
                lambda: project_head_greedy_pairs(
                    layer_cand128, x, vocab=org_n, vocab_start=0, pair_width=4, scratch_pairs=True
                )
            )
            row_entry["image_tile_configs_128"] = image_cfg128
            print(f"[{shape_name}][{dtype_name}] M={m:3d}: {row_entry}", flush=True)
            dtype_res[str(m)] = row_entry

        results[dtype_name] = dtype_res
    return results


image_cfg128 = dict(vocab.TILE_CONFIGS[128])  # c51 own marks: the image's own config, before the bench sets it


def main():
    out_path = sys.argv[1]
    payload = {
        "device": torch.cuda.get_device_name(0),
        "K": K,
        "M_values": MS,
        "shapes": {},
    }
    for shape_name, (n, org_n) in SHAPES.items():
        payload["shapes"][shape_name] = {
            "N": n,
            "org_N": org_n,
            "K": K,
            "dtypes": run_shape(shape_name, n, org_n),
        }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
