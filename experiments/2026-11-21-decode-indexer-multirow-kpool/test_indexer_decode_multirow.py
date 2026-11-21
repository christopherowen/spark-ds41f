import pytest
import torch

from vllm.models.deepseek_v4_1.tilelang import indexer as idx_mod

assert hasattr(idx_mod, "_score_decode_call"), "Patched indexer.py not mounted!"
assert idx_mod._DECODE_SCORE_ROW_GROUP == 3
assert idx_mod._DECODE_SCORE_MAX_ROWS == 96


def test_plan_calls_and_cta_caps():
    assert idx_mod._decode_score_max_ctas(4096) == 144
    assert idx_mod._decode_score_max_ctas(65536) == 144
    assert idx_mod._decode_score_max_ctas(262144) == 144
    assert idx_mod._decode_score_max_ctas(1048576) == 288

    dense_calls = idx_mod.IndexerV41Plan.calls(65536, 64, candidates=False)
    score_calls = [c for c in dense_calls if c[0] is idx_mod.indexer_score_v41]
    assert len(score_calls) == 2
    assert {c[2].get("row_group") for c in score_calls} == {3, 4}

    cand_calls = idx_mod.IndexerV41Plan.calls(16384, 64, candidates=True)
    cand_score_calls = [c for c in cand_calls if c[0] is idx_mod.indexer_score_v41]
    assert len(cand_score_calls) == 1


@pytest.mark.parametrize(
    "name,req_vs,width,ctx,dead_rows",
    [
        ("1x1", [1], 4096, 4096, ()),
        ("1x2_partial", [2], 8192, 4103, ()),
        ("1x3_partial", [3], 65536, 65531, ()),
        ("1x5_partial", [5], 65536, 12349, ()),
        ("1x6_partial", [6], 65536, 65535, ()),
        ("4x6", [6] * 4, 65536, 8197, ()),
        ("8x3", [3] * 8, 65536, 8201, ()),
        ("8x6", [6] * 8, 65536, 4111, ()),
        ("16x6_boundary", [6] * 16, 8192, 4099, ()),
        ("adapt8", [3, 5, 2, 4, 6, 1, 5, 3], 16384, 9001, ()),
        ("dead_rows", [6, 3, 4], 8192, 4105, (0, 2, 5, 8, 12)),
        ("prefill_97_cross", [97], 8192, 4107, ()),
        ("prefill_130", [130], 8192, 4109, ()),
        ("1M_1x3", [3], 1048576, 32771, ()),
        ("1M_8x3", [3] * 8, 1048576, 16391, ()),
    ],
)
def test_decode_multirow_bit_exact(name, req_vs, width, ctx, dead_rows):
    device = "cuda"
    torch.manual_seed(1234 + len(req_vs))
    page_size = 64
    nbytes = idx_mod.page_bytes(page_size)
    rows = sum(req_vs)
    pages_per_req = (ctx + page_size - 1) // page_size + 2
    total_pages = max(pages_per_req * 2, len(req_vs) * pages_per_req)
    pool = torch.randint(0, 255, (total_pages, nbytes), dtype=torch.uint8, device=device)
    q = torch.randint(-128, 127, (rows, 32, 64), dtype=torch.int8, device=device)
    # Keep finite UE8M0 scales + include some 0xFF NaN scales to verify exact NaN parity too
    qs = torch.randint(100, 140, (rows, 32), dtype=torch.int32, device=device).view(torch.uint32)
    w = torch.randn((rows, 32), dtype=torch.bfloat16, device=device)
    pt = torch.zeros((rows, pages_per_req), dtype=torch.int32, device=device)
    lengths = torch.zeros((rows,), dtype=torch.int32, device=device)
    cur = 0
    for ri, v in enumerate(req_vs):
        base_pg = (ri * pages_per_req) % max(1, total_pages - pages_per_req + 1)
        perm = base_pg + torch.randperm(pages_per_req, dtype=torch.int32, device=device)
        base_len = max(1, ctx - ri * 173)
        for j in range(v):
            pt[cur] = perm
            lengths[cur] = 0 if cur in dead_rows else min(base_len + j, ctx)
            cur += 1
    active = torch.tensor([ctx], dtype=torch.int32, device=device)

    # Reference: legacy row_group=0 kernel
    k_ref = idx_mod.indexer_score_v41(width, page_size, False)
    s_ref = torch.full((rows, width), -77.0, dtype=torch.bfloat16, device=device)
    k_ref(q, qs, w, pool.view(torch.int8), pool.view(torch.uint32), pt, lengths, active, s_ref)

    # Plan path (routes <=96 to row_group=3 decode kernel, >96 to row_group=4 prefill kernel)
    plan = idx_mod.IndexerV41Plan(width, page_size, candidates=False, candidate_blocks=True)
    cand_out = torch.empty((rows, idx_mod.CANDIDATE_BLOCKS), dtype=torch.int32, device=device)
    cand_lens = torch.empty((rows,), dtype=torch.int32, device=device)
    out_idx, s_plan = plan.run(
        q, qs, w, pool, pt, lengths, active,
        candidate_out=cand_out, candidate_out_lengths=cand_lens,
    )
    for r in range(rows):
        e = min(int(lengths[r]), ctx, width)
        if e > 0:
            assert torch.equal(
                s_ref[r, :e].view(torch.uint16), s_plan[r, :e].view(torch.uint16)
            ), f"{name} row={r} ext={e} bit mismatch"

    # Also verify TopK output parity against legacy kernel scores
    out_ref = torch.empty_like(out_idx)
    vals_ref = torch.empty((rows, idx_mod.INDEX_TOPK), dtype=torch.bfloat16, device=device)
    cand_ref = torch.empty_like(cand_out)
    cand_vals_ref = torch.empty((rows, idx_mod.CANDIDATE_BLOCKS), dtype=torch.bfloat16, device=device)
    cand_lens_ref = torch.empty_like(cand_lens)
    plan.select_blocks.run(s_ref, lengths, active, cand_ref, cand_vals_ref, counts=cand_lens_ref)
    if plan.select.gather_scores:
        plan.select.run(s_ref, cand_lens_ref, None, out_ref, vals_ref, gather=cand_ref)
    else:
        plan.select.run(s_ref, lengths, active, out_ref, vals_ref)
    assert torch.equal(cand_out, cand_ref), f"{name} candidate_blocks mismatch"
    assert torch.equal(cand_lens, cand_lens_ref), f"{name} candidate_lengths mismatch"
    assert torch.equal(out_idx, out_ref), f"{name} topk indices mismatch"
