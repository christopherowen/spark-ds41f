#!/usr/bin/env python3
"""#995 guard (round 3, candidate-55 rework): copy-step identity and timing, image A/B.
Usage: guard995.py --label L --out OUT.jsonl

The copy step is what the guard edits: copy_alloc and copy_apply, after copy_match.
The base arm is candidate-54's tree, the candidate arm candidate-54 + the 0781 guard.

1. identity (kind=identity): #977's own-mark scenarios (candidate-53 receipts: its
   bench_copy_match_long_context.py's 7 scenarios, history construction and seeds, UVA
   history), 1/8/16 requests, contexts 4K/32K/128K/1M. Per cell, seven copy steps
   (copy_match, copy_alloc, copy_apply) with the copy state carried from step to step, the
   slots in a permuted order, verification outcomes (num_sampled, verified, cut) drawn from a
   fixed generator (full acceptances, partial ones and none), and DSpark confidences that
   admit and refuse copies. Step 7 is a mid-prefill step as the runner makes it: copy_match
   gets -1 for every third row (match_idx_mapping) and copy_alloc/copy_apply the real slots.
   Every output and state tensor (sources, chosen, estimates, alloc, drafts, confidences,
   planned, int_state, float_state) is hashed a step; output buffers are refilled with -5
   before each step, so an unwritten entry also shows.
2. timing (kind=timing): the 24 cells of benchmark_copy_drafts.py's copy case (the copy-step
   bench #977's own mark ran as its gate: UVA and device history, 1/8/16 requests, 4K/64K/
   256K/1M, the same construction and seeds), with that module's timed() imported unmodified
   (CUDA-graph replay, 64 MiB flush before each replay, median):
   - step: copy_match + copy_alloc + copy_apply (the gate's launch), 21 replays (its default);
   - alloc_apply: copy_alloc + copy_apply alone (the two guarded kernels), 201 replays;
   - eager_alloc_apply: the serving path launches the copy kernels eagerly from
     sample_tokens: wall microseconds a step over 200 back-to-back eager copy_alloc +
     copy_apply launches (one synchronize at the end), median of 5 repeats.
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

# 1. identity
SCENARIOS = (("no_match", 16), ("recent_match_8", 8), ("recent_match_16", 16), ("recent_match_32", 16),
             ("recent_match_256", 16), ("deep_match_256", 16), ("boilerplate_16", 16))
CONTEXTS = (4096, 32768, 131072, 1048576)
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
            int_state = torch.zeros((16, tlc.STATE_COLUMNS), dtype=torch.int32, device=dev)
            float_state = torch.zeros((16, tlc.STATE_COLUMNS), dtype=torch.float32, device=dev)
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
                dconf = (0.3 + 0.69 * torch.rand((n, BLOCK), generator=g)).to(dev)
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
                              "max_doublings": int(int_state[idx, 0].max()),
                              "alloc_sum": int(alloc.sum())})
            cell_sha = hashlib.sha256("".join(s["sha"] for s in steps).encode()).hexdigest()[:16]
            all_cells.update(cell_sha.encode())
            emit(kind="identity", scenario=scenario, min_match=min_match, reqs=num_reqs, ctx=ctx, cell_sha=cell_sha, steps=steps)
emit(kind="identity_all", cells=len(SCENARIOS) * len(REQ_COUNTS) * len(CONTEXTS), sha=all_cells.hexdigest()[:16])
del hb, history
torch.cuda.empty_cache()

# 2. timing: benchmark_copy_drafts.py's copy cells
MIN_MATCH = bcd.MIN_MATCH
for uva in (True, False):
    for num_reqs in (1, 8, 16):
        for context in bcd.CONTEXTS:
            g = torch.Generator(device="cpu").manual_seed(num_reqs * 7 + context)
            shape = (16, context + 64)
            if uva:
                buf = StagedWriteTensor(shape, dtype=torch.int32, device=dev, uva_instead_of_gpu=True); hist = buf.gpu
            else:
                buf = None; hist = torch.empty(shape, dtype=torch.int32, device=dev)
            total_len = torch.zeros(16, dtype=torch.int32, device=dev)
            for slot in range(num_reqs):
                tokens = torch.randint(0, VOCAB, (context,), generator=g, dtype=torch.int32)
                start = int(torch.randint(0, context // 4 + 1, (1,), generator=g))
                tail = min(600, context // 2)
                tokens[context - tail:] = tokens[start:start + tail]
                hist[slot, :context].copy_(tokens.to(hist.device)); total_len[slot] = context
            idx = torch.arange(num_reqs, dtype=torch.int64, device=dev)
            levels = tlc.num_levels(MIN_MATCH)
            sources = torch.empty((num_reqs, tlc.CHUNKS, levels), dtype=torch.int32, device=dev)
            int_state = torch.zeros((16, 2), dtype=torch.int32, device=dev)
            float_state = torch.zeros((16, 2), dtype=torch.float32, device=dev)
            dspark = torch.zeros((num_reqs, BLOCK), dtype=torch.int64, device=dev)
            dconf = torch.full((num_reqs, BLOCK), 0.5, device=dev)
            ones = torch.ones(num_reqs, dtype=torch.int32, device=dev)
            chosen = torch.empty((num_reqs, tlc.CHOSEN_COLUMNS), dtype=torch.int32, device=dev)
            estimates = torch.empty(num_reqs, dtype=torch.float32, device=dev)
            outs = (torch.empty((num_reqs, LOOKAHEAD), dtype=torch.int64, device=dev),
                    torch.empty((num_reqs, LOOKAHEAD), device=dev), torch.empty((num_reqs, LOOKAHEAD), device=dev),
                    torch.empty(num_reqs, dtype=torch.int32, device=dev))
            def match():
                tlc.copy_match(idx, hist, total_len, sources, MIN_MATCH, LOOKAHEAD)
            def alloc_apply():
                tlc.copy_alloc(idx, total_len, sources, ones, ones, ones, int_state, float_state,
                               chosen, estimates, outs[3], MIN_MATCH, LOOKAHEAD, BLOCK, INITIAL)
                tlc.copy_apply(idx, hist, chosen, estimates, int_state, dspark, dconf, *outs[:3])
            def step():
                match(); alloc_apply()
            step_med, step_min = bcd.timed(step)
            match(); torch.cuda.synchronize()
            aa_med, aa_min = bcd.timed(alloc_apply, replays=201)
            found = int((outs[3] > BLOCK).sum())
            eager = []
            for _ in range(5):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(200):
                    alloc_apply()
                torch.cuda.synchronize(); eager.append((time.perf_counter() - t0) / 200 * 1e6)
            emit(kind="timing", history="uva" if uva else "device", reqs=num_reqs, ctx=context, found=found,
                 step_us=round(step_med, 3), step_min_us=round(step_min, 3),
                 alloc_apply_us=round(aa_med, 3), alloc_apply_min_us=round(aa_min, 3),
                 eager_alloc_apply_us=round(statistics.median(eager), 3), eager_samples_us=[round(x, 3) for x in eager])
            del buf, hist
            torch.cuda.empty_cache()
fh.close()
print("done", flush=True)
