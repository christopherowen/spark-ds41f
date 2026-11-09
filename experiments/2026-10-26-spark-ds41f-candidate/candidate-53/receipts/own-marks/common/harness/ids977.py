#!/usr/bin/env python3
"""#977 draft identity (candidate-53 own marks): copy_match's output (the `sources` table that
copy_alloc/copy_apply turn into copy drafts) for every case of the PR's
bench_copy_match_long_context.py (same scenarios, seeds, contexts 4K..1M, 1/8/16 requests,
UVA history), as a SHA-256 per case, to compare across images. No timing."""
import argparse, hashlib, json
import torch
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.spec_decode import tilelang_copy

VOCAB, LOOKAHEAD = 129280, 16
CONTEXTS = (4096, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)
REQ_COUNTS = (1, 8, 16)
SCENARIOS = (("no_match", 16), ("recent_match_8", 8), ("recent_match_16", 16), ("recent_match_32", 16),
             ("recent_match_256", 16), ("deep_match_256", 16), ("boilerplate_16", 16))
p = argparse.ArgumentParser(); p.add_argument("--label", required=True); p.add_argument("--out", required=True)
a = p.parse_args()
max_ctx = max(CONTEXTS)
hb = StagedWriteTensor((max(REQ_COUNTS), max_ctx + 64), dtype=torch.int32, device=torch.device("cuda"), uva_instead_of_gpu=True)
history = hb.gpu
offsets16 = torch.arange(-16, 0).unsqueeze(0)
with open(a.out, "w") as fh:
    for scenario, min_match in SCENARIOS:
        levels = tilelang_copy.num_levels(min_match)
        for num_reqs in REQ_COUNTS:
            idx = torch.arange(num_reqs, dtype=torch.int64, device="cuda")
            sources = torch.empty((num_reqs, tilelang_copy.CHUNKS, levels), dtype=torch.int32, device="cuda")
            for ctx in CONTEXTS:
                g = torch.Generator(device="cpu").manual_seed(1234 + num_reqs * 17 + ctx)
                total_len = torch.zeros(max(REQ_COUNTS), dtype=torch.int32, device="cuda")
                for slot in range(num_reqs):
                    tokens = torch.randint(10, VOCAB, (ctx,), generator=g, dtype=torch.int32)
                    if scenario == "recent_match_8":
                        pos = max(32, ctx - 200); tokens[pos - 8:pos] = tokens[ctx - 8:ctx]; tokens[pos - 9] = (tokens[ctx - 9] + 1) % VOCAB
                    elif scenario == "recent_match_16":
                        pos = max(32, ctx - 500); tokens[pos - 16:pos] = tokens[ctx - 16:ctx]; tokens[pos - 17] = (tokens[ctx - 17] + 1) % VOCAB
                    elif scenario == "recent_match_32":
                        pos = max(64, ctx - 1000); tokens[pos - 32:pos] = tokens[ctx - 32:ctx]; tokens[pos - 33] = (tokens[ctx - 33] + 1) % VOCAB
                    elif scenario == "recent_match_256":
                        pos = max(300, ctx - 1000); tokens[pos - 256:pos] = tokens[ctx - 256:ctx]
                    elif scenario == "deep_match_256":
                        start = int(torch.randint(0, ctx // 4 + 1, (1,), generator=g)); tail = min(600, ctx // 2)
                        tokens[ctx - tail:] = tokens[start:start + tail]
                    elif scenario == "boilerplate_16":
                        phrase = tokens[ctx - 16:ctx].clone(); guard = int((tokens[ctx - 17].item() + 1) % VOCAB)
                        idxs = torch.arange(32, ctx - 320, 48); tokens[idxs - 17] = guard
                        tokens[idxs.unsqueeze(1) + offsets16] = phrase.unsqueeze(0)
                    history[slot, :ctx].copy_(tokens.to(history.device)); total_len[slot] = ctx
                sources.fill_(-7)
                tilelang_copy.copy_match(idx, history, total_len, sources, min_match, LOOKAHEAD)
                torch.cuda.synchronize()
                s = sources.cpu()
                rec = {"label": a.label, "scenario": scenario, "reqs": num_reqs, "ctx": ctx,
                       "sources_sha256": hashlib.sha256(s.numpy().tobytes()).hexdigest()[:16],
                       "found_entries": int((s >= 0).sum()), "max_level_found": [int(x) for x in (s >= 0).any(dim=1).sum(dim=0).tolist()]}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
print("done")
