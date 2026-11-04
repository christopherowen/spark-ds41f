#!/usr/bin/env python3
"""Copy drafting's step work on one GB10: candidate-37's kernels against this patch's.

Run in the candidate-37 image with the patch's files over the image's vLLM tree and
candidate-37's own ``tilelang_copy.py`` beside this script as ``tilelang_copy_base.py``
(run_bench.sh does both). Two measurements, every case base and patch alternated
(ABBA) in one process:

- ``kernels``: GPU time of each launch, from a CUDA graph of 50 back-to-back
  launches replayed 20 times (fewer for slow launches; launch overhead excluded). candidate-37 runs
  ``copy_match``, ``copy_alloc`` (read-only state) and ``copy_select`` (the state's
  update again and the proposal); the patch runs ``copy_match`` (empty chunks exit),
  ``copy_alloc`` (the state's update and the choice) and ``copy_apply``.
- ``step``: the drafting tail of a decode step as the runner issues it, eagerly,
  from the end of postprocess to the proposals, with a stand-in for DSpark's draft
  forward on the main stream (a CUDA graph of 40 bandwidth-bound BF16 GEMMs over
  distinct 5120 x 2048 weights, ~3 ms, every SM busy: the worst case for kernels
  beside it). candidate-37: match and alloc on the main stream, then DSpark, then
  select, the allocation's D2H on the copy stream after alloc. The patch: match and
  alloc on the copy stream beside DSpark, its D2H after them, apply after DSpark.
  Reported: the main stream's time (CUDA events) and the host's wall time per step.

Workloads: 1, 8 and 16 requests, contexts of 2K to 1M tokens, of random tokens over
the 129,280-token vocabulary (``prose``: few first-token hits), of Zipf(1.1) ranks
(``zipf``: frequent tokens, as text has, many short runs) or a 50-token phrase
repeated (``loop``: every end matches deep, the slowest match).

``--c1``: one request only, the ``step`` measurement alone at 1,000 alternated
steps per case (prose and zipf, 2K to 1M), for the single-stream verdict.

usage: bench_copy_kernels.py OUT.json [--quick | --c1]
"""

import json
import statistics
import sys
import time

import numpy as np
import torch

import tilelang_copy_base as base_tlc
from vllm.v1.worker.gpu.spec_decode import tilelang_copy as tlc

DEVICE = torch.device("cuda")
VOCAB = 129280
MIN_MATCH, LOOKAHEAD, BLOCK, INITIAL = 16, 16, 5, 8


class Case:
    def __init__(self, num_reqs: int, length: int, kind: str, seed: int = 0):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        self.num_reqs, self.length, self.kind = num_reqs, length, kind
        rows = 32
        self.history = torch.full(
            (rows, length + 64), -9, dtype=torch.int32, device=DEVICE
        )
        slots = torch.randperm(rows, generator=g, device=DEVICE)[:num_reqs]
        for slot in slots.tolist():
            if kind == "prose":
                tokens = torch.randint(
                    0, VOCAB, (length,), generator=g, device=DEVICE, dtype=torch.int32
                )
            elif kind == "zipf":
                ranks = np.random.default_rng(seed + slot).zipf(1.1, length) - 1
                tokens = torch.from_numpy((ranks % VOCAB).astype(np.int32)).to(DEVICE)
            else:
                phrase = torch.randint(
                    0, VOCAB, (50,), generator=g, device=DEVICE, dtype=torch.int32
                )
                tokens = phrase.repeat(length // 50 + 1)[:length]
            self.history[slot, :length] = tokens
        self.total_len = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
        self.total_len[slots] = length
        self.idx = slots.to(torch.int64)
        n = num_reqs
        levels = tlc.num_levels(MIN_MATCH)
        self.sources = torch.empty(
            (n, tlc.CHUNKS, levels), dtype=torch.int32, device=DEVICE
        )
        self.ones = torch.ones(n, dtype=torch.int32, device=DEVICE)
        self.int_state = torch.zeros(
            (rows, tlc.STATE_COLUMNS), dtype=torch.int32, device=DEVICE
        )
        self.float_state = torch.zeros(
            (rows, tlc.STATE_COLUMNS), dtype=torch.float32, device=DEVICE
        )
        self.chosen = torch.zeros(
            (n, tlc.CHOSEN_COLUMNS), dtype=torch.int32, device=DEVICE
        )
        self.estimates = torch.zeros(n, device=DEVICE)
        self.alloc = torch.zeros(n, dtype=torch.int32, device=DEVICE)
        self.alloc_cpu = torch.zeros(n, dtype=torch.int32, pin_memory=True)
        self.limits = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
        self.select_alloc = torch.zeros(n, dtype=torch.int32, device=DEVICE)
        self.dspark = (
            torch.randint(0, VOCAB, (n, BLOCK), device=DEVICE),
            torch.rand((n, BLOCK), device=DEVICE),
        )
        self.drafts = torch.zeros((n, LOOKAHEAD), dtype=torch.int64, device=DEVICE)
        self.confidences = torch.zeros((n, LOOKAHEAD), device=DEVICE)
        self.planned = torch.zeros((n, LOOKAHEAD), device=DEVICE)

    # candidate-37's launches.
    def base_match(self):
        base_tlc.copy_match(
            self.idx, self.history, self.total_len, self.sources, MIN_MATCH
        )

    def base_alloc(self):
        base_tlc.copy_alloc(
            self.idx, self.total_len, self.sources, self.ones, self.ones, self.ones,
            self.int_state, self.alloc, self.limits,
            MIN_MATCH, LOOKAHEAD, BLOCK, INITIAL,
        )  # fmt: skip

    def base_select(self):
        base_tlc.copy_select(
            self.idx, self.history, self.total_len, self.sources, *self.dspark,
            self.ones, self.ones, self.ones, self.int_state, self.float_state,
            self.drafts, self.confidences, self.planned, self.select_alloc,
            MIN_MATCH, INITIAL,
        )  # fmt: skip

    # The patch's launches.
    def new_match(self):
        tlc.copy_match(self.idx, self.history, self.total_len, self.sources, MIN_MATCH)

    def new_alloc(self):
        tlc.copy_alloc(
            self.idx, self.total_len, self.sources, self.ones, self.ones, self.ones,
            self.int_state, self.float_state, self.chosen, self.estimates,
            self.alloc, self.limits, MIN_MATCH, LOOKAHEAD, BLOCK, INITIAL,
        )  # fmt: skip

    def new_apply(self):
        tlc.copy_apply(
            self.idx, self.history, self.chosen, self.estimates, *self.dspark,
            self.drafts, self.confidences, self.planned,
        )  # fmt: skip


def graph_us(fn, launches=50, replays=20) -> float:
    """GPU microseconds of one ``fn`` call, from a CUDA graph of ``launches`` calls
    (fewer for a slow call: about 0.2 s of replays in all)."""
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    once = time.perf_counter() - t0
    if once * launches * replays > 0.2:
        launches = max(1, min(launches, int(0.2 / once / 4)))
        replays = 4
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (launches * replays)


class StandIn:
    """DSpark's draft forward, for the streams: 40 bandwidth-bound GEMMs."""

    def __init__(self, rows: int):
        self.x = torch.randn((rows, 5120), dtype=torch.bfloat16, device=DEVICE)
        self.weights = [
            torch.randn((5120, 2048), dtype=torch.bfloat16, device=DEVICE)
            for _ in range(40)
        ]
        self.out = torch.empty((rows, 2048), dtype=torch.bfloat16, device=DEVICE)
        self.run()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.run()

    def run(self):
        for w in self.weights:
            torch.matmul(self.x, w, out=self.out)


def step_base(case: Case, standin: StandIn, main, side):
    case.base_match()
    case.base_alloc()
    side.wait_stream(main)
    with torch.cuda.stream(side):
        case.alloc_cpu.copy_(case.alloc, non_blocking=True)
    standin.graph.replay()
    case.base_select()


def step_new(case: Case, standin: StandIn, main, side, done):
    side.wait_stream(main)
    with torch.cuda.stream(side):
        case.new_match()
        case.new_alloc()
        done.record(side)
        case.alloc_cpu.copy_(case.alloc, non_blocking=True)
    standin.graph.replay()
    main.wait_event(done)
    case.new_apply()


def measure_steps(case: Case, standin: StandIn, iterations: int) -> dict:
    main = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    done = torch.cuda.Event()
    variants = {
        "base": lambda: step_base(case, standin, main, side),
        "new": lambda: step_new(case, standin, main, side, done),
    }
    gpu = {name: [] for name in variants}
    wall = {name: [] for name in variants}
    for fn in variants.values():  # warm
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    for i in range(iterations):
        order = ("base", "new") if i % 2 == 0 else ("new", "base")
        for name in order:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            start.record()
            variants[name]()
            end.record()
            side.synchronize()
            end.synchronize()
            wall[name].append((time.perf_counter() - t0) * 1e6)
            gpu[name].append(start.elapsed_time(end) * 1000)
    start.record()
    for _ in range(20):
        standin.graph.replay()
    end.record()
    end.synchronize()
    standin_us = start.elapsed_time(end) * 1000 / 20
    out = {"standin_us": round(standin_us, 1)}
    for name in variants:
        out[name] = {
            "gpu_us_median": round(statistics.median(gpu[name]), 2),
            "gpu_us_mean": round(statistics.fmean(gpu[name]), 2),
            "gpu_us_stdev": round(statistics.stdev(gpu[name]), 2),
            "wall_us_median": round(statistics.median(wall[name]), 2),
        }
    paired = [n - b for n, b in zip(gpu["new"], gpu["base"])]
    out["delta_gpu_us_median"] = round(statistics.median(paired), 2)
    out["delta_gpu_us_mean"] = round(statistics.fmean(paired), 2)
    out["delta_gpu_us_ci95"] = round(
        1.96 * statistics.stdev(paired) / len(paired) ** 0.5, 2
    )
    return out


def check_equal(case: Case) -> bool:
    """The patch's proposals equal candidate-37's on the case (one step from zero
    state)."""
    case.int_state.zero_()
    case.float_state.zero_()
    case.base_match()
    case.base_select()
    reference = [
        t.clone()
        for t in (case.drafts, case.confidences, case.planned, case.select_alloc)
    ]
    state = (case.int_state.clone(), case.float_state.clone())
    case.int_state.zero_()
    case.float_state.zero_()
    case.new_match()
    case.new_alloc()
    case.new_apply()
    same = all(
        torch.equal(a, b)
        for a, b in zip(
            reference + list(state),
            (case.drafts, case.confidences, case.planned, case.alloc,
             case.int_state, case.float_state),
        )
    )  # fmt: skip
    case.int_state.zero_()
    case.float_state.zero_()
    return same


def equivalence_walk(steps: int = 400, seed: int = 7) -> dict:
    """candidate-37's step (match, read-only alloc, select) and the patch's (match,
    alloc, apply) over the same turn, each on its own copy state: 8 requests over
    texts that copy a document with edits, repeat a phrase, or never repeat, the
    committed tokens and verified, accepted and cut counts varied every step. Every
    step's proposals, allocations, slot limits and copy state must be identical."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, rows, width = 8, 16, 6000
    texts = []
    for r in range(n):
        if r % 3 == 0:
            doc = torch.randint(0, VOCAB, (900,), generator=g)
            reply = doc.clone()
            for at in torch.randint(0, 900, (6,), generator=g).tolist():
                reply[at] = int(torch.randint(0, VOCAB, (1,), generator=g))
            texts.append(torch.cat([torch.randint(0, VOCAB, (40,), generator=g), doc, reply]))
        elif r % 3 == 1:
            phrase = torch.randint(0, 64, (37,), generator=g)
            texts.append(phrase.repeat(60)[:2000])
        else:
            texts.append(torch.randint(0, VOCAB, (2000,), generator=g))
    history = torch.full((rows, width), -9, dtype=torch.int32, device=DEVICE)
    slots = torch.randperm(rows, generator=g)[:n]
    for slot, text in zip(slots.tolist(), texts):
        history[slot, : len(text)] = text.to(torch.int32).to(DEVICE)
    lengths = [940 if r % 3 == 0 else 300 for r in range(n)]
    idx = slots.to(torch.int64).to(DEVICE)
    levels = tlc.num_levels(MIN_MATCH)

    def buffers():
        return {
            "sources": torch.empty((n, tlc.CHUNKS, levels), dtype=torch.int32, device=DEVICE),
            "int_state": torch.zeros((rows, 2), dtype=torch.int32, device=DEVICE),
            "float_state": torch.zeros((rows, 2), dtype=torch.float32, device=DEVICE),
            "alloc": torch.zeros(n, dtype=torch.int32, device=DEVICE),
            "limits": torch.full((rows,), LOOKAHEAD, dtype=torch.int32, device=DEVICE),
            "drafts": torch.zeros((n, LOOKAHEAD), dtype=torch.int64, device=DEVICE),
            "confidences": torch.zeros((n, LOOKAHEAD), device=DEVICE),
            "planned": torch.zeros((n, LOOKAHEAD), device=DEVICE),
        }

    old, new = buffers(), buffers()
    old["select_alloc"] = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    new["chosen"] = torch.zeros((n, tlc.CHOSEN_COLUMNS), dtype=torch.int32, device=DEVICE)
    new["estimates"] = torch.zeros(n, device=DEVICE)
    num_sampled = [1] * n
    verified = [0] * n
    cut = [0] * n
    copies = mismatches = 0
    for step in range(steps):
        total_len = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
        total_len[idx] = torch.tensor(lengths, dtype=torch.int32, device=DEVICE)
        dspark = (
            torch.randint(0, VOCAB, (n, BLOCK), generator=g).to(DEVICE),
            torch.rand((n, BLOCK), generator=g).to(DEVICE),
        )
        args = [torch.tensor(v, dtype=torch.int32, device=DEVICE) for v in (num_sampled, verified, cut)]
        base_tlc.copy_match(idx, history, total_len, old["sources"], MIN_MATCH)
        base_tlc.copy_alloc(
            idx, total_len, old["sources"], *args, old["int_state"], old["alloc"],
            old["limits"], MIN_MATCH, LOOKAHEAD, BLOCK, INITIAL,
        )  # fmt: skip
        base_tlc.copy_select(
            idx, history, total_len, old["sources"], *dspark, *args, old["int_state"],
            old["float_state"], old["drafts"], old["confidences"], old["planned"],
            old["select_alloc"], MIN_MATCH, INITIAL,
        )  # fmt: skip
        tlc.copy_match(idx, history, total_len, new["sources"], MIN_MATCH)
        tlc.copy_alloc(
            idx, total_len, new["sources"], *args, new["int_state"], new["float_state"],
            new["chosen"], new["estimates"], new["alloc"], new["limits"],
            MIN_MATCH, LOOKAHEAD, BLOCK, INITIAL,
        )  # fmt: skip
        tlc.copy_apply(
            idx, history, new["chosen"], new["estimates"], *dspark, new["drafts"],
            new["confidences"], new["planned"],
        )  # fmt: skip
        pairs = [
            (old[k], new[k])
            for k in ("drafts", "confidences", "planned", "alloc", "limits",
                      "int_state", "float_state")
        ] + [(old["select_alloc"], new["alloc"])]  # fmt: skip
        if not all(torch.equal(a, b) for a, b in pairs):
            mismatches += 1
        # The next step: verify a random share of the offered drafts (or DSpark's
        # block), accept a prefix of them against the text, cut now and then.
        offered = (old["confidences"] > 0).sum(dim=1).tolist()
        pending = old["int_state"][idx, 1].tolist()
        drafts = old["drafts"].tolist()
        for r in range(n):
            text, p = texts[r], lengths[r]
            v = int(torch.randint(0, offered[r] + 1, (1,), generator=g))
            accepted = 0
            while accepted < v and p + accepted < len(text) and drafts[r][accepted] == int(text[p + accepted]):
                accepted += 1
            take = min(accepted + 1, len(text) - p - 1, width - 64 - p)
            if take <= 0:
                lengths[r] = 300
                take = 1
            copies += pending[r] > 0
            lengths[r] += take
            num_sampled[r] = take if step % 11 else 0
            verified[r] = v
            cut[r] = v if step % 5 else max(v - 1, 0)
    return {"steps": steps, "requests": n, "copy_proposals": copies, "mismatched_steps": mismatches}


def main_c1(out_path: str) -> None:
    torch.manual_seed(0)
    results = {"device": torch.cuda.get_device_name(), "steps": []}
    standin = StandIn(BLOCK)
    for length in (2048, 32768, 131072, 1048576):
        for kind in ("prose", "zipf"):
            case = Case(1, length, kind)
            row = {"num_reqs": 1, "length": length, "kind": kind, "equal": check_equal(case)}
            results["steps"].append({**row, **measure_steps(case, standin, 1000)})
            print(json.dumps(results["steps"][-1]), flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
        f.write("\n")


def main(out_path: str, quick: bool) -> None:
    torch.manual_seed(0)
    lengths = [2048, 32768, 131072, 1048576]
    reqs = [1, 8, 16]
    if quick:
        lengths, reqs = [2048, 131072], [1, 8]
    results = {
        "device": torch.cuda.get_device_name(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "equivalence": equivalence_walk(),
        "kernels": [],
        "steps": [],
    }
    print(json.dumps(results["equivalence"]), flush=True)
    for num_reqs in reqs:
        standin = StandIn(num_reqs * BLOCK)
        for length in lengths:
            if num_reqs * (length + 64) * 4 > 2 << 30:
                continue
            for kind in ("prose", "zipf", "loop"):
                case = Case(num_reqs, length, kind)
                row = {"num_reqs": num_reqs, "length": length, "kind": kind}
                row["equal"] = check_equal(case)
                timings = {}
                for name in ("match", "alloc"):
                    for variant in ("base", "new", "new", "base"):
                        fn = getattr(case, f"{variant}_{name}")
                        timings.setdefault(f"{variant}_{name}", []).append(graph_us(fn))
                for variant, fn in (
                    ("base_select", case.base_select),
                    ("new_apply", case.new_apply),
                    ("new_apply", case.new_apply),
                    ("base_select", case.base_select),
                ):
                    timings.setdefault(variant, []).append(graph_us(fn))
                kernels = {k: round(statistics.fmean(v), 2) for k, v in timings.items()}
                kernels["base_total"] = round(
                    kernels["base_match"] + kernels["base_alloc"] + kernels["base_select"],
                    2,
                )
                kernels["new_total"] = round(
                    kernels["new_match"] + kernels["new_alloc"] + kernels["new_apply"], 2
                )
                results["kernels"].append({**row, **kernels})
                print(json.dumps(results["kernels"][-1]), flush=True)
                if kind != "loop" or length >= 131072:
                    steps = measure_steps(case, standin, 60 if quick else 200)
                    results["steps"].append({**row, **steps})
                    print(json.dumps(results["steps"][-1]), flush=True)
                del case
                torch.cuda.empty_cache()
        del standin
        torch.cuda.empty_cache()
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    if "--c1" in sys.argv[2:]:
        main_c1(sys.argv[1])
    else:
        main(sys.argv[1], "--quick" in sys.argv[2:])
