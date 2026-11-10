#!/usr/bin/env python3
"""#995 guard (round 3): supplementary copy-step identity, image A/B. Usage: ids995b.py --label L --out OUT.jsonl

guard995.py's identity section took every offered copy (its fresh copy state trusts a copy at
the 0.95 prior, above any DSpark yield), so copy_apply's refusal branch (start >= 0, the copy's
expected yield below DSpark's: DSpark's drafts, the slot's pending state cleared), whose
condition the guard edits, did not run there. This is the same seven-step loop, scenarios and
seeds, with contexts 4K and 128K, the copy state started from a random one (doublings 0..4,
pending 0..16, hits/trials giving estimates of about 0.3..0.95) and DSpark confidences high
(0.9..0.999) on half the rows, so copies are both taken and refused. SHA-256 a step as there.
"""

import argparse, hashlib, importlib.util, json, statistics, time
import torch
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.spec_decode import tilelang_copy as tlc

p = argparse.ArgumentParser()
p.add_argument("--label", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
dev = torch.device("cuda")
spec = importlib.util.spec_from_file_location("bcd", "benchmarks/kernels/benchmark_copy_drafts.py")
bcd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bcd)
VOCAB, LOOKAHEAD, BLOCK, INITIAL = 129280, 16, 5, 8
fh = open(a.out, "w")
def emit(**r):
    r = {"label": a.label, **r}
    fh.write(json.dumps(r) + "\n"); fh.flush(); print(json.dumps(r), flush=True)
emit(kind="identity_env", gpu=torch.cuda.get_device_name(), torch=torch.__version__, tilelang_copy=tlc.__file__,
     bench=bcd.__file__, src_sha256=hashlib.sha256(open(tlc.__file__, "rb").read()).hexdigest())

# identity (guard995.py's, with the changes in the docstring)
SCENARIOS = (("no_match", 16), ("recent_match_8", 8), ("recent_match_16", 16), ("recent_match_32", 16),
             ("recent_match_256", 16), ("deep_match_256", 16), ("boilerplate_16", 16))
CONTEXTS = (4096, 131072)
REQ_COUNTS = (1, 8, 16)
max_ctx = max(CONTEXTS)
hb = StagedWriteTensor((16, max_ctx + 64), dtype=torch.int32, device=dev, uva_instead_of_gpu=True)
history = hb.gpu
offsets16 = torch.arange(-16, 0).unsqueeze(0)

def build(scenario, num_reqs, ctx):  # bench_copy_match_long_context.py's construction and seeds
    g = torch.Generator(device="cpu").manual_seed(1234 + num_reqs * 17 + ctx)
    total_len = torch.zeros(16, dtype=torch.int32, device=dev)
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
    return total_len

def digest(ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()

all_cells = hashlib.sha256()
for si, (scenario, min_match) in enumerate(SCENARIOS):
    levels = tlc.num_levels(min_match)
    for num_reqs in REQ_COUNTS:
        for ctx in CONTEXTS:
            total_len = build(scenario, num_reqs, ctx)
            g = torch.Generator(device="cpu").manual_seed(995000 + si * 1000 + num_reqs * 37 + ctx % 997)
            perm = torch.randperm(num_reqs, generator=g)
            idx = perm.to(dev, torch.int64)
            g.manual_seed(995500 + si * 1000 + num_reqs * 37 + ctx % 997)
            int_state = torch.stack((torch.randint(0, 5, (16,), generator=g, dtype=torch.int32),
                                     torch.randint(0, 17, (16,), generator=g, dtype=torch.int32)), 1).to(dev)
            trials = 40 * torch.rand(16, generator=g)
            hits = trials * torch.rand(16, generator=g)
            float_state = torch.stack((hits, trials), 1).to(dev)
            n = num_reqs
            sources = torch.empty((n, tlc.CHUNKS, levels), dtype=torch.int32, device=dev)
            chosen = torch.empty((n, tlc.CHOSEN_COLUMNS), dtype=torch.int32, device=dev)
            estimates = torch.empty(n, dtype=torch.float32, device=dev)
            alloc = torch.empty(n, dtype=torch.int32, device=dev)
            drafts = torch.empty((n, LOOKAHEAD), dtype=torch.int64, device=dev)
            confs = torch.empty((n, LOOKAHEAD), dtype=torch.float32, device=dev)
            planned = torch.empty((n, LOOKAHEAD), dtype=torch.float32, device=dev)
            steps = []
            for step in range(7):
                pend = int_state[idx, 1].cpu()
                full = torch.rand(n, generator=g) < 0.5
                verified = torch.where(torch.rand(n, generator=g) < 0.75, pend,
                                       torch.randint(0, LOOKAHEAD + 1, (n,), generator=g, dtype=torch.int32))
                partial = (torch.rand(n, generator=g) * (verified.float() + 1)).floor().to(torch.int32) + 1
                num_sampled = torch.where(full, verified + 1, torch.minimum(partial, verified + 1))
                if step == 0:
                    num_sampled = torch.ones(n, dtype=torch.int32)
                cut = torch.where(torch.rand(n, generator=g) < 0.8, verified,
                                  (torch.rand(n, generator=g) * (verified.float() + 1)).floor().to(torch.int32))
                dspark = torch.randint(0, VOCAB, (n, BLOCK), generator=g, dtype=torch.int64).to(dev)
                high = (torch.arange(n) % 2 == 0).unsqueeze(1)
                dconf = torch.where(high, 0.9 + 0.099 * torch.rand((n, BLOCK), generator=g),
                                    0.3 + 0.3 * torch.rand((n, BLOCK), generator=g)).to(dev)
                match_idx = idx
                if step == 6:
                    mask = torch.arange(n) % 3 == 1
                    match_idx = torch.where(mask, torch.full_like(perm, -1), perm).to(dev, torch.int64)
                for t in (sources, chosen, alloc):
                    t.fill_(-5)
                for t in (estimates, drafts, confs, planned):
                    t.fill_(-5)
                tlc.copy_match(match_idx, history, total_len, sources, min_match, LOOKAHEAD)
                tlc.copy_alloc(idx, total_len, sources, num_sampled.to(dev), verified.to(dev), cut.to(dev),
                               int_state, float_state, chosen, estimates, alloc, min_match, LOOKAHEAD, BLOCK, INITIAL)
                tlc.copy_apply(idx, history, chosen, estimates, int_state, dspark, dconf, drafts, confs, planned)
                torch.cuda.synchronize()
                d = digest((sources, chosen, estimates, alloc, drafts, confs, planned, int_state, float_state))
                steps.append({"sha": d[:16], "offered_rows": int((chosen[:, 1] > 0).sum()),
                              "copy_rows": int((int_state[idx, 1] > 0).sum()),
                              "refused_rows": int(((chosen[:, 1] > 0) & (int_state[idx, 1] == 0)).sum()),
                              "max_doublings": int(int_state[idx, 0].max()),
                              "alloc_sum": int(alloc.sum())})
            cell_sha = hashlib.sha256("".join(s["sha"] for s in steps).encode()).hexdigest()[:16]
            all_cells.update(cell_sha.encode())
            emit(kind="identity", scenario=scenario, min_match=min_match, reqs=num_reqs, ctx=ctx, cell_sha=cell_sha, steps=steps)
emit(kind="identity_all", cells=len(SCENARIOS) * len(REQ_COUNTS) * len(CONTEXTS), sha=all_cells.hexdigest()[:16])
fh.close()
print("done", flush=True)
