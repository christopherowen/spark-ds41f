# SPDX-License-Identifier: Apache-2.0
"""Two-boot GPU benchmark for CopyDrafter bounds guards, level-anchor pruning,
gentler partial-acceptance cap decay, and wide lookahead (16 vs 31 vs 63)."""

import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

from vllm.config.speculative import CopyDraftsConfig
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.spec_decode import tilelang_copy as cand_tlc
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)
from vllm.v1.worker.gpu.spec_decode.copy_drafts import CopyDrafter
from vllm.v1.worker.gpu.spec_decode.greedy_verification import (
    greedy_rejection_sample,
    vocab_parallel_pairs,
)
from vllm.v1.worker.gpu.states import RequestState

VOCAB = 129280
MIN_MATCH = 16
BLOCK = 5
INITIAL = 8
MAX_LEVEL = CopyDraftsConfig.MAX_LEVEL
FLUSH = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")


def load_base_tlc(path: str):
    spec = importlib.util.spec_from_file_location("base_tilelang_copy", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def timed_graph(launch, replays: int = 21, flush: bool = True) -> tuple[float, float]:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        launch()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    samples = []
    for _ in range(replays):
        if flush:
            FLUSH.zero_()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0)
    return statistics.median(samples), min(samples)


def bench_boilerplate_match(base_tlc, num_reqs: int, context: int, uva: bool) -> dict:
    rng = np.random.default_rng(20261108 + num_reqs * 13 + context)
    shape = (16, context + 64)
    if uva:
        history_buf = StagedWriteTensor(
            shape,
            dtype=torch.int32,
            device=torch.device("cuda"),
            uva_instead_of_gpu=True,
        )
        history = history_buf.gpu
    else:
        history = torch.empty(shape, dtype=torch.int32, device="cuda")
    total_len = torch.zeros(16, dtype=torch.int32, device="cuda")
    tail = rng.integers(10, VOCAB, MAX_LEVEL, dtype=np.int32)
    for slot in range(num_reqs):
        S = rng.integers(10, VOCAB, context, dtype=np.int32)
        boiler_len = (16, 32, 64, 128)[slot % 4]
        phrase = tail[MAX_LEVEL - boiler_len :]
        guard = int((tail[MAX_LEVEL - boiler_len - 1] + 1) % VOCAB)
        # Stamp repeated structural phrases every 48 tokens in the upper half of
        # the context, with an earlier 256-token full match in the first quarter.
        for pos in range(context // 2, context - MAX_LEVEL - 64, 48):
            S[pos - boiler_len - 1] = guard
            S[pos - boiler_len : pos] = phrase
        early = max(MAX_LEVEL + 32, context // 8)
        S[early : early + MAX_LEVEL] = tail
        S[context - MAX_LEVEL :] = tail
        history[slot, :context].copy_(torch.from_numpy(S).to(history.device))
        total_len[slot] = context

    idx = torch.arange(num_reqs, dtype=torch.int64, device="cuda")
    levels = cand_tlc.num_levels(MIN_MATCH)
    sources_base = torch.empty(
        (num_reqs, cand_tlc.CHUNKS, levels), dtype=torch.int32, device="cuda"
    )
    sources_cand = torch.empty_like(sources_base)
    int_base = torch.zeros((16, 2), dtype=torch.int32, device="cuda")
    float_base = torch.zeros((16, 2), dtype=torch.float32, device="cuda")
    int_cand = torch.zeros((16, 2), dtype=torch.int32, device="cuda")
    float_cand = torch.zeros((16, 2), dtype=torch.float32, device="cuda")
    dspark = torch.zeros((num_reqs, BLOCK), dtype=torch.int64, device="cuda")
    dconf = torch.full((num_reqs, BLOCK), 0.5, device="cuda")
    ones = torch.ones(num_reqs, dtype=torch.int32, device="cuda")
    chosen_base = torch.empty(
        (num_reqs, cand_tlc.CHOSEN_COLUMNS), dtype=torch.int32, device="cuda"
    )
    chosen_cand = torch.empty_like(chosen_base)
    est_base = torch.empty(num_reqs, dtype=torch.float32, device="cuda")
    est_cand = torch.empty_like(est_base)
    outs_base = (
        torch.empty((num_reqs, 16), dtype=torch.int64, device="cuda"),
        torch.empty((num_reqs, 16), device="cuda"),
        torch.empty((num_reqs, 16), device="cuda"),
        torch.empty(num_reqs, dtype=torch.int32, device="cuda"),
    )
    outs_cand = tuple(torch.empty_like(t) for t in outs_base)

    def run_base():
        base_tlc.copy_match(idx, history, total_len, sources_base, MIN_MATCH, 16)
        base_tlc.copy_alloc(
            idx, total_len, sources_base, ones, ones, ones, int_base, float_base,
            chosen_base, est_base, outs_base[3], MIN_MATCH, 16, BLOCK, INITIAL,
        )  # fmt: skip
        base_tlc.copy_apply(
            idx, history, chosen_base, est_base, int_base, dspark, dconf, *outs_base[:3]
        )

    def run_cand():
        cand_tlc.copy_match(idx, history, total_len, sources_cand, MIN_MATCH, 16)
        cand_tlc.copy_alloc(
            idx, total_len, sources_cand, ones, ones, ones, int_cand, float_cand,
            chosen_cand, est_cand, outs_cand[3], MIN_MATCH, 16, BLOCK, INITIAL,
        )  # fmt: skip
        cand_tlc.copy_apply(
            idx, history, chosen_cand, est_cand, int_cand, dspark, dconf, *outs_cand[:3]
        )

    run_base()
    run_cand()
    torch.cuda.synchronize()
    bit_exact = bool(
        torch.equal(sources_base, sources_cand)
        and torch.equal(chosen_base, chosen_cand)
        and torch.equal(outs_base[0], outs_cand[0])
        and torch.equal(outs_base[3], outs_cand[3])
    )
    assert bit_exact, "Boilerplate match output mismatch between base and candidate"

    base_us, base_min = timed_graph(run_base)
    cand_us, cand_min = timed_graph(run_cand)
    return {
        "case": f"boilerplate_B{num_reqs}_ctx{context}_{'uva' if uva else 'device'}",
        "requests": num_reqs,
        "context": context,
        "history": "uva" if uva else "device",
        "base_us": round(base_us, 2),
        "cand_us": round(cand_us, 2),
        "speedup": round(base_us / cand_us, 3),
        "bit_exact": bit_exact,
    }


def bench_wide_lookahead_kernels(lookahead: int, num_reqs: int, context: int = 65536) -> dict:
    g = torch.Generator(device="cpu").manual_seed(900 + lookahead * 31 + num_reqs)
    history = torch.empty((16, context + 128), dtype=torch.int32, device="cuda")
    total_len = torch.zeros(16, dtype=torch.int32, device="cuda")
    for slot in range(num_reqs):
        tokens = torch.randint(0, VOCAB, (context,), generator=g, dtype=torch.int32)
        start = int(torch.randint(0, context // 4 + 1, (1,), generator=g))
        tail = 600
        tokens[context - tail :] = tokens[start : start + tail]
        history[slot, :context].copy_(tokens.to("cuda"))
        total_len[slot] = context
    idx = torch.arange(num_reqs, dtype=torch.int64, device="cuda")
    levels = cand_tlc.num_levels(MIN_MATCH)
    sources = torch.empty(
        (num_reqs, cand_tlc.CHUNKS, levels), dtype=torch.int32, device="cuda"
    )
    int_state = torch.zeros((16, 2), dtype=torch.int32, device="cuda")
    float_state = torch.zeros((16, 2), dtype=torch.float32, device="cuda")
    dspark = torch.zeros((num_reqs, BLOCK), dtype=torch.int64, device="cuda")
    dconf = torch.full((num_reqs, BLOCK), 0.5, device="cuda")
    ones = torch.ones(num_reqs, dtype=torch.int32, device="cuda")
    chosen = torch.empty(
        (num_reqs, cand_tlc.CHOSEN_COLUMNS), dtype=torch.int32, device="cuda"
    )
    estimates = torch.empty(num_reqs, dtype=torch.float32, device="cuda")
    outs = (
        torch.empty((num_reqs, lookahead), dtype=torch.int64, device="cuda"),
        torch.empty((num_reqs, lookahead), device="cuda"),
        torch.empty((num_reqs, lookahead), device="cuda"),
        torch.empty(num_reqs, dtype=torch.int32, device="cuda"),
    )

    def launch_copy():
        cand_tlc.copy_match(idx, history, total_len, sources, MIN_MATCH, lookahead)
        cand_tlc.copy_alloc(
            idx, total_len, sources, ones, ones, ones, int_state, float_state,
            chosen, estimates, outs[3], MIN_MATCH, lookahead, BLOCK, INITIAL,
        )  # fmt: skip
        cand_tlc.copy_apply(
            idx, history, chosen, estimates, int_state, dspark, dconf, *outs[:3]
        )

    copy_us, _ = timed_graph(launch_copy)

    # Greedy verification at wide=True (one copying request at 1 + lookahead rows)
    counts = [1 + lookahead] + [1 + BLOCK] * (num_reqs - 1)
    rows = sum(counts)
    vg = torch.Generator(device="cuda").manual_seed(7123 + lookahead + num_reqs)
    logits = torch.randn(rows, VOCAB, generator=vg, device="cuda") * 4
    drafts = torch.randint(0, VOCAB, (rows,), generator=vg, device="cuda").int()
    cu = torch.tensor([0, *np.cumsum(counts)], dtype=torch.int32, device="cuda")
    pairs = vocab_parallel_pairs(logits, VOCAB, 0, 1)

    def launch_greedy_verify():
        return greedy_rejection_sample(pairs, drafts, cu, lookahead)

    verify_us, _ = timed_graph(launch_greedy_verify, replays=15)
    return {
        "lookahead": lookahead,
        "requests": num_reqs,
        "wide_rows": rows,
        "copy_us": round(copy_us, 2),
        "greedy_verify_us": round(verify_us, 2),
    }


def simulate_copy_workload(
    tlc_mod,
    lookahead: int,
    doc_len: int = 2048,
    num_splices: int = 0,
    seed: int = 2026,
) -> dict:
    """Drive CopyDrafter + AdaptiveVerificationManager + greedy_rejection_sample
    over a document copy workload (verbatim when num_splices=0, or with in-document
    branch splices that trigger partial acceptances >= INITIAL)."""
    rng = np.random.default_rng(seed)
    doc = rng.integers(10, VOCAB, doc_len, dtype=np.int32)
    # Build reply from doc with num_splices branch jumps to earlier spans of doc
    # so after accepting 12..28 tokens of a span, the continuation jumps to another
    # span of doc that is already in the prompt!
    if num_splices == 0:
        reply = doc.copy()
    else:
        chunks = []
        pos = 0
        stride = doc_len // (num_splices + 1)
        for s in range(num_splices):
            # Take a span of length ~stride (chosen so partial acceptance happens
            # mid-window at offset >= INITIAL)
            span_len = max(32, stride - (s % 3) * 5)
            src = (s * 137) % (doc_len - span_len - 32)
            chunks.append(doc[src : src + span_len])
            pos += span_len
        rem = max(32, doc_len - pos)
        chunks.append(doc[:rem])
        reply = np.concatenate(chunks)[:doc_len]

    prompt = np.concatenate(
        [rng.integers(10, VOCAB, 64, dtype=np.int32), doc, reply[:24]]
    )
    full_text = np.concatenate([prompt, reply[24:]]).astype(np.int64)
    start = len(prompt)
    end = len(full_text)

    states = RequestState(
        max_num_reqs=4,
        max_model_len=8192,
        max_num_batched_tokens=4096,
        num_speculative_steps=lookahead,
        vocab_size=VOCAB,
        device=torch.device("cuda"),
    )
    config = CopyDraftsConfig(
        min_match=MIN_MATCH, max_drafts=lookahead, initial_drafts=INITIAL
    )
    drafter = CopyDrafter(config, states, num_draft_steps=BLOCK, device=torch.device("cuda"))
    manager = AdaptiveVerificationManager(
        states,
        torch.zeros(5, dtype=torch.int32, device="cuda"),
        num_bonus_tokens=1,
        max_total_logits=4096,
        row_cost=0.02,
        planned_confidences=True,
        num_draft_steps=BLOCK,
    )
    manager.dead_rows_tau = 0.0

    req_id = "bench_req"
    states.add_request(req_id, start, full_text[:start].tolist(), 0, 1 << 20)
    slot = states.req_id_to_index[req_id]
    drafter.add_request(slot)
    manager.add_request(slot)
    states.apply_staged_writes()
    drafter.apply_staged_writes()
    manager.apply_staged_writes()

    idx = torch.tensor([slot], dtype=torch.int64, device="cuda")
    dspark = (
        torch.zeros((1, BLOCK), dtype=torch.int64, device="cuda"),
        torch.full((1, BLOCK), 0.4, device="cuda"),
    )
    p = start
    num_sampled = torch.ones(1, dtype=torch.int32, device="cuda")
    verified = torch.zeros(1, dtype=torch.int32, device="cuda")
    allocs = [1 + BLOCK]
    steps = 0
    total_rows = 0
    total_cost_units = 0.0
    max_offered = 0

    while p < end:
        states.all_token_ids.stage_write(slot, 0, full_text[:p].tolist())
        states.total_len.stage_write_elem(slot, p)
        states.apply_staged_writes()

        # Run using the requested tlc_mod (base_tlc or cand_tlc)
        sources = drafter.sources[:1]
        tlc_mod.copy_match(
            idx, states.all_token_ids.gpu, states.total_len.gpu, sources, MIN_MATCH, lookahead
        )
        tlc_mod.copy_alloc(
            idx, states.total_len.gpu, sources, num_sampled, verified, verified,
            drafter.int_state, drafter.float_state, drafter.chosen[:1],
            drafter.estimates[:1], drafter.alloc[:1], MIN_MATCH, lookahead, BLOCK, INITIAL,
        )  # fmt: skip
        tlc_mod.copy_apply(
            idx, states.all_token_ids.gpu, drafter.chosen[:1], drafter.estimates[:1],
            drafter.int_state, dspark[0], dspark[1], drafter.drafts[:1],
            drafter.confidences[:1], drafter.planned[:1],
        )  # fmt: skip

        offered = int(drafter.int_state[slot, 1])
        max_offered = max(max_offered, offered)
        allocs.append(1 + int(drafter.alloc[0]))
        step_alloc = allocs[-2]

        if offered > 0:
            d = min(offered, step_alloc - 1)
            proposal = drafter.drafts[0, :d].cpu().numpy()
            accepted = 0
            while (
                accepted < d
                and p + accepted < end
                and proposal[accepted] == full_text[p + accepted]
            ):
                accepted += 1
            committed = min(accepted + 1, end - p)
            live_rows = 1 + d
            num_sampled = torch.tensor([committed], dtype=torch.int32, device="cuda")
            verified = torch.tensor([d], dtype=torch.int32, device="cuda")
        else:
            # DSpark fallback step (commits 2 tokens on average)
            committed = min(2, end - p)
            live_rows = 1 + BLOCK
            num_sampled = torch.tensor([committed], dtype=torch.int32, device="cuda")
            verified = torch.tensor([BLOCK], dtype=torch.int32, device="cuda")

        p += committed
        steps += 1
        total_rows += live_rows
        total_cost_units += float(manager._step_cost(np.array([live_rows]))[0])

    generated = end - start
    return {
        "workload": "verbatim_2048" if num_splices == 0 else f"spliced_2048_edits{num_splices}",
        "lookahead": lookahead,
        "generated_tokens": generated,
        "steps": steps,
        "mean_tokens_per_step": round(generated / steps, 2),
        "total_verified_rows": total_rows,
        "relative_cost_units": round(total_cost_units, 2),
        "tokens_per_cost_unit": round(generated / total_cost_units, 2),
        "max_offered": max_offered,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, required=True)
    parser.add_argument("--base-tlc", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_tlc = load_base_tlc(args.base_tlc)

    boilerplate_results = []
    for uva in (True, False):
        for num_reqs in (1, 8, 16):
            for context in (16384, 65536, 262144):
                res = bench_boilerplate_match(base_tlc, num_reqs, context, uva)
                print(json.dumps(res), flush=True)
                boilerplate_results.append(res)

    wide_kernel_results = []
    for lookahead in (16, 31, 63):
        for num_reqs in (1, 8, 16):
            res = bench_wide_lookahead_kernels(lookahead, num_reqs)
            print(json.dumps(res), flush=True)
            wide_kernel_results.append(res)

    walk_results = []
    for num_splices in (0, 8, 16):
        # Baseline 0761 at lookahead=16
        base_res = simulate_copy_workload(base_tlc, 16, doc_len=2048, num_splices=num_splices)
        base_res["variant"] = "base_0761_k16"
        print(json.dumps(base_res), flush=True)
        walk_results.append(base_res)
        # Candidate 0762 across lookahead = 16, 31, 63
        for lookahead in (16, 31, 63):
            cand_res = simulate_copy_workload(
                cand_tlc, lookahead, doc_len=2048, num_splices=num_splices
            )
            cand_res["variant"] = f"cand_0762_k{lookahead}"
            cand_res["step_reduction_vs_base_k16"] = round(
                base_res["steps"] / cand_res["steps"], 3
            )
            cand_res["throughput_gain_vs_base_k16"] = round(
                cand_res["tokens_per_cost_unit"] / base_res["tokens_per_cost_unit"], 3
            )
            print(json.dumps(cand_res), flush=True)
            walk_results.append(cand_res)

    summary = {
        "boot": args.boot,
        "device": torch.cuda.get_device_name(0),
        "boilerplate_match": boilerplate_results,
        "wide_lookahead_kernels": wide_kernel_results,
        "end_to_end_walks": walk_results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
