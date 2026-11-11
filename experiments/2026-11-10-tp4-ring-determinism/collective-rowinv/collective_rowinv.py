#!/usr/bin/env python3
"""Row invariance and run-to-run determinism of serving's TP collectives, model-free,
one rank per node, through the communicator decode calls (``get_tp_group()``).

usage (a lab "fabric" job; the runner adds the probe's rank arguments):
    collective_rowinv.py --rank R --world-size N --master-addr A --master-port P
        [--iters 4] [--replays 24] [--skip-stress]
harness self-test (CPU, gloo, no GPU, no vLLM; checks this script's own power):
    collective_rowinv.py --rank R --world-size 4 --master-addr 127.0.0.1 --master-port P
        --selftest {contract,reversed,ring,stale}

Every rank generates every rank's partials from seeds, so each rank knows the exact
answer: the one-shot contract, ``bf16(((p0 + p1) + p2) + p3)`` in FP32 (BF16 inputs)
or ``((p0 + p1) + p2) + p3`` (FP32). A fixed target row (the same partials in every
cell) sits at position 0, M // 2 and M - 1 of an M-row message among neighbours of
four kinds (random, garbage with NaN/Inf/1e4, near copies of the target, zeros), for
M = 1..17, 24, 32, 48, 64, 96, 128, 136, 204 (the last one-shot size at 5120 BF16),
205, 208, 256, 272 (the rank-order path past the 2 MiB dispatch).

Phases (each checks every element against the contract on every rank):
  eager     group.all_reduce: the serving dispatch (one-shot <= 2 MiB, rank order past it),
            and the same input again (run to run)
  graph     a captured graph of 86 all-reduces (two a layer, 43 layers) at one M, replayed
            with fresh inputs each replay, as a FULL decode graph replays
  gather    all_gather (dim 0 and -1, BF16/FP32/int64 decode shapes): an exact copy
  scatter   reduce_scatter, one-shot and past the slot: the same contract per element
  stress    the eager and graph checks again with per-rank random GPU delays before each
            collective (arrival skew), long host idles (the proxy's nanosleep path and a late
            doorbell), and busy loops on ranks 1 and 3's proxy CPUs (the relay and the
            missed-doorbell catch-up while the proxy thread is descheduled)

Pass: zero element mismatches, zero repeat mismatches and one target-row hash per dtype
in every phase. The summary also reports how many of the target row's elements another
order would change (NCCL's ring order from each owner, reversed FP32), so a pass is
evidence against any other order. Rank 0 prints one JSON line per phase, then the
summary; every rank exits 1 on any failure.
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import timedelta

HIDDEN = 5120
M_VALUES = list(range(1, 18)) + [24, 32, 48, 64, 96, 128, 136, 204, 205, 208, 256, 272]
GRAPH_M = [1, 2, 5, 6, 9, 13, 16, 17, 32, 64, 136]
GRAPH_CALLS = 86
NEIGHBOURS = ("random", "garbage", "copies", "zeros")


class SelfTestGroup:
    """A CPU stand-in for the TP group (gloo): all-reduce by an exact rank-order sum, or
    by a deliberately wrong one, so the harness shows it catches the wrong ones."""

    def __init__(self, torch, dist, mode: str, world: int, rank: int):
        self.torch, self.dist, self.mode, self.world, self.rank = torch, dist, mode, world, rank
        self.cpu_group = dist.group.WORLD
        self.calls = 0
        self.previous = []  # the stale mode returns a pack from two calls back

    def _gather(self, x):
        parts = [self.torch.empty_like(x) for _ in range(self.world)]
        self.dist.all_gather(parts, x.contiguous())
        return parts

    def _sum(self, parts):
        torch = self.torch
        if self.mode == "reversed":
            total = parts[-1].float()
            for p in reversed(parts[:-1]):
                total = total + p.float()
            return total.to(parts[0].dtype)
        if self.mode == "ring":  # NCCL-like: each 1/world of the flat buffer from its own start
            flat = [p.reshape(-1) for p in parts]
            n = flat[0].numel()
            out = torch.empty_like(flat[0])
            bounds = [n * i // self.world for i in range(self.world + 1)]
            for owner in range(self.world):
                lo, hi = bounds[owner], bounds[owner + 1]
                order = [(owner + 1 + i) % self.world for i in range(self.world)]
                acc = flat[order[0]][lo:hi]
                for idx in order[1:]:
                    acc = (acc.float() + flat[idx][lo:hi].float()).to(acc.dtype)
                out[lo:hi] = acc
            return out.view(parts[0].shape)
        total = parts[0].float()
        for p in parts[1:]:
            total = total + p.float()
        return total.to(parts[0].dtype)

    def all_reduce(self, x):
        out = self._sum(self._gather(x))
        self.calls += 1
        if self.mode == "stale" and self.calls % 97 == 0 and len(self.previous) >= 2:
            old = self.previous[-2].reshape(-1)
            flat = out.reshape(-1)
            k = min(8, old.numel(), flat.numel())
            flat[:k] = old[:k]  # one 16-byte pack from the op two slots back
        self.previous = (self.previous + [out.clone()])[-2:]
        return out

    def all_gather(self, x, dim=-1):
        return self.torch.cat(self._gather(x), dim=dim)

    def reduce_scatter(self, x, dim=0):
        local = x.shape[0] // self.world
        return self._sum(self._gather(x))[self.rank * local:(self.rank + 1) * local]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--replays", type=int, default=24)
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--selftest", choices=("contract", "reversed", "ring", "stale"))
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    world, rank = args.world_size, args.rank
    selftest = args.selftest is not None
    started = time.monotonic()
    if selftest:
        device = torch.device("cpu")
        dist.init_process_group("gloo", init_method=f"tcp://{args.master_addr}:{args.master_port}",
                                world_size=world, rank=rank, timeout=timedelta(seconds=120))
        group = SelfTestGroup(torch, dist, args.selftest, world, rank)
        adapter = runtime = None
        topology = f"selftest-{args.selftest}"
        graph_capture = None
    else:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed.parallel_state import (
            get_tp_group,
            graph_capture,
            init_distributed_environment,
            initialize_model_parallel,
            set_custom_all_reduce,
        )

        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        topology = os.environ.get("SPARKNET_ROCE_TOPOLOGY")
        if topology is None:
            raise RuntimeError("the arm's environment must select sparknet's one-shot collectives")
        set_custom_all_reduce(True)
        init_distributed_environment(
            world_size=world, rank=rank, local_rank=0,
            distributed_init_method=f"tcp://{args.master_addr}:{args.master_port}",
            timeout=timedelta(seconds=120),
        )
        with set_current_vllm_config(VllmConfig()):
            initialize_model_parallel(tensor_model_parallel_size=world)
        group = get_tp_group()
        comm = group.device_communicator
        adapter = comm.b12x_ar_comm
        if adapter is None or adapter.disabled or comm.pynccl_comm is None or comm.pynccl_comm.disabled:
            raise RuntimeError("sparknet's one-shot collectives and PyNCCL are both required")
        runtime = adapter._runtime
        if topology != "direct" and runtime.topology != topology:
            raise RuntimeError(f"the runtime runs {runtime.topology}, not the arm's {topology}")
    cpu_group = group.cpu_group

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def gpu_sleep(cycles):
        if cycles and device.type == "cuda":
            torch.cuda._sleep(int(cycles))

    def gen(seed):
        return torch.Generator(device=device).manual_seed(int(seed))

    def name_of(dtype):
        return str(dtype).replace("torch.", "")

    def rows_bits(t):
        return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)

    def contract(parts):
        total = parts[0].float()
        for p in parts[1:]:
            total = total + p.float()
        return total.to(parts[0].dtype)

    def mismatches(out, expected):
        """Elements whose bits differ (NaN counts equal to NaN: garbage rows), on this rank."""
        if out.shape != expected.shape:
            return max(out.numel(), expected.numel())
        differ = rows_bits(out) != rows_bits(expected)
        both_nan = torch.isnan(out) & torch.isnan(expected)
        return int((differ & ~both_nan).sum())

    # The target row's partials: order-sensitive even in FP32 (a term 2^24 times larger
    # than the next absorbs its low bits, then cancels exactly), so any other summation
    # order changes most of its elements, after the BF16 rounding as well.
    def target_parts(dtype):
        g = gen(77)
        big = (torch.randn(HIDDEN, generator=g, device=device) * 65536).to(dtype).float()
        base = [big,
                torch.randn(HIDDEN, generator=g, device=device) / 256,
                -big,
                torch.randn(HIDDEN, generator=g, device=device) / 1024]
        base += [torch.randn(HIDDEN, generator=g, device=device) / 8 for _ in range(max(0, world - 4))]
        return [p.to(dtype) for p in base[:world]]

    def neighbour_parts(kind, rows, seed, target, dtype):
        parts = []
        for r in range(world):
            if rows == 0:
                parts.append(torch.empty((0, HIDDEN), device=device, dtype=dtype))
                continue
            g = gen(seed * 64 + r)
            if kind == "random":
                p = torch.randn((rows, HIDDEN), generator=g, device=device) * 4
            elif kind == "garbage":
                p = (torch.rand((rows, HIDDEN), generator=g, device=device) * 2 - 1) * 1e4
                mask = torch.rand((rows, HIDDEN), generator=g, device=device)
                p = torch.where(mask < 0.01, torch.full_like(p, float("nan")), p)
                p = torch.where(mask > 0.995, torch.full_like(p, float("inf")), p)
            elif kind == "copies":
                p = target[r].float().expand(rows, HIDDEN) + torch.randn(
                    (rows, HIDDEN), generator=g, device=device) * 1e-3
            else:
                p = torch.zeros((rows, HIDDEN), device=device)
            parts.append(p.to(dtype))
        return parts

    def message(kind, m, pos, seed, dtype, target):
        """Every rank's [m, HIDDEN] partial, the target row at ``pos``."""
        before = neighbour_parts(kind, pos, seed, target, dtype)
        after = neighbour_parts(kind, m - pos - 1, seed + 7919, target, dtype)
        return [torch.cat([before[r], target[r][None], after[r]]).contiguous() for r in range(world)]

    def target_hash(out, pos):
        return hashlib.sha256(out[pos].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]

    totals: dict = {}

    def entry_of(phase):
        return totals.setdefault(phase, {"cells": 0, "element_mismatches": 0, "repeat_mismatches": 0,
                                         "target_hashes": {}, "failures": [], "notes": []})

    def tally(phase, *, cells=0, elements=0, hashes=None, repeats=0, note=None):
        entry = entry_of(phase)
        entry["cells"] += cells
        entry["element_mismatches"] += elements
        entry["repeat_mismatches"] += repeats
        for dtype, seen in (hashes or {}).items():
            entry["target_hashes"].setdefault(dtype, set()).update(seen)
        if note:
            entry["notes"].append(note)

    def failure(phase, detail):
        entry = entry_of(phase)
        if len(entry["failures"]) < 8:
            entry["failures"].append(detail)

    dtypes = (torch.bfloat16, torch.float32)
    targets = {dtype: target_parts(dtype) for dtype in dtypes}

    # Sensitivity: how many target elements another order would change.
    sens = {}
    for dtype in dtypes:
        parts = targets[dtype]
        ref = contract(parts)
        total = parts[world - 1].float()
        for idx in reversed(range(world - 1)):
            total = total + parts[idx].float()
        changed = {"reversed_fp32": int((rows_bits(total.to(dtype)) != rows_bits(ref)).sum())}
        for owner in range(world):  # NCCL's ring: summed from after the chunk's owner, rounded every hop
            order = [(owner + 1 + i) % world for i in range(world)]
            acc = parts[order[0]]
            for idx in order[1:]:
                acc = (acc.float() + parts[idx].float()).to(dtype)
            changed[f"ring_after_rank{owner}"] = int((rows_bits(acc) != rows_bits(ref)).sum())
        sens[name_of(dtype)] = {"elements": HIDDEN, **changed}

    # ---------------------------------------------------------------- eager
    def eager_phase(phase, delay=None, idle_every=0, idle_s=0.0, iters=None):
        iters = args.iters if iters is None else iters
        hashes = {name_of(d): set() for d in dtypes}
        cells = elements = repeats = 0
        rng = random.Random(1000 + rank)
        cell = 0
        for dtype in dtypes:
            target = targets[dtype]
            for m in M_VALUES:
                for pos in sorted({0, m // 2, m - 1}):
                    for kind in NEIGHBOURS:
                        for it in range(iters):
                            cell += 1
                            parts = message(kind, m, pos, cell * 131 + it, dtype, target)
                            expected = contract(parts)
                            if idle_every and cell % idle_every == 0:
                                dist.barrier(group=cpu_group)
                                time.sleep(idle_s)
                            if delay is not None:
                                gpu_sleep(delay(rng))
                            out = group.all_reduce(parts[rank])
                            again = group.all_reduce(parts[rank].clone()) if it == 0 else None
                            sync()
                            bad = mismatches(out, expected)
                            elements += bad
                            if bad:
                                failure(phase, {"dtype": name_of(dtype), "m": m, "pos": pos, "kind": kind,
                                                "iter": it, "elements": bad, "rank": rank})
                            if again is not None and not torch.equal(rows_bits(again), rows_bits(out)):
                                repeats += 1
                            hashes[name_of(dtype)].add(target_hash(out, pos))
                            cells += 1
        tally(phase, cells=cells, elements=elements, hashes=hashes, repeats=repeats)

    # ---------------------------------------------------------------- graph
    def graph_phase(phase, delay_cycles=None):
        if device.type != "cuda":
            tally(phase, note="skipped: no CUDA graphs in the self-test")
            return
        hashes = {name_of(d): set() for d in dtypes}
        cells = elements = 0
        rng = random.Random(2000 + rank)
        for dtype in dtypes:
            target = targets[dtype]
            for m in GRAPH_M:
                pos = m // 2
                static = [torch.zeros((m, HIDDEN), device=device, dtype=dtype) for _ in range(GRAPH_CALLS)]
                group.all_reduce(static[0])  # this size's first use outside the capture
                sync()
                cycles = [delay_cycles(rng) if delay_cycles else 0 for _ in range(GRAPH_CALLS)]
                graph = torch.cuda.CUDAGraph()
                with graph_capture(device=device) as context:
                    with torch.cuda.graph(graph, stream=context.stream):
                        outs = []
                        for i, buf in enumerate(static):
                            gpu_sleep(cycles[i])
                            outs.append(group.all_reduce(buf))
                for rep in range(args.replays):
                    expected = []
                    for i, buf in enumerate(static):
                        kind = NEIGHBOURS[(i + rep) % len(NEIGHBOURS)]
                        parts = message(kind, m, pos, 10**6 + (rep * GRAPH_CALLS + i) * 17 + m, dtype, target)
                        buf.copy_(parts[rank])
                        expected.append(contract(parts))
                    sync()
                    dist.barrier(group=cpu_group)
                    graph.replay()
                    sync()
                    for i, out in enumerate(outs):
                        bad = mismatches(out, expected[i])
                        elements += bad
                        if bad:
                            failure(phase, {"dtype": name_of(dtype), "m": m, "call": i, "replay": rep,
                                            "elements": bad, "rank": rank})
                        hashes[name_of(dtype)].add(target_hash(out, pos))
                        cells += 1
                del graph, outs, static, expected
        tally(phase, cells=cells, elements=elements, hashes=hashes)

    # ---------------------------------------------------------------- gather / scatter
    def gather_phase(phase):
        cells = elements = 0
        shapes = [((1, 32), torch.float32), ((5, 2), torch.float32), ((17, 4), torch.int64),
                  ((17, 16), torch.float32), ((64, 4), torch.int64), ((17, HIDDEN), torch.bfloat16),
                  ((136, 8), torch.float32)]
        for (rows, cols), dtype in shapes:
            for dim in (0, -1):
                for it in range(args.iters):
                    shards = []
                    for r in range(world):
                        g = gen(5000 + it * 97 + r * 13 + rows)
                        if dtype == torch.int64:
                            shards.append(torch.randint(-2**40, 2**40, (rows, cols), generator=g, device=device))
                        else:
                            shards.append((torch.randn((rows, cols), generator=g, device=device) * 3).to(dtype))
                    out = group.all_gather(shards[rank], dim=dim)
                    sync()
                    expected = torch.cat(shards, dim=dim)
                    same = tuple(out.shape) == tuple(expected.shape) and torch.equal(
                        out.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
                    if not same:
                        elements += 1
                        failure(phase, {"shape": [rows, cols], "dtype": name_of(dtype), "dim": dim, "rank": rank})
                    cells += 1
        tally(phase, cells=cells, elements=elements)

    def scatter_phase(phase):
        cells = elements = 0
        for dtype in dtypes:
            for local in (1, 8, 51, 52, 64, 128):
                for it in range(args.iters):
                    parts = [(torch.randn((world * local, HIDDEN), generator=gen(9000 + it * 31 + r * 7 + local),
                                          device=device) * 4).to(dtype) for r in range(world)]
                    out = group.reduce_scatter(parts[rank], dim=0)
                    sync()
                    expected = contract(parts)[rank * local:(rank + 1) * local]
                    bad = mismatches(out, expected)
                    elements += bad
                    if bad:
                        failure(phase, {"dtype": name_of(dtype), "local_rows": local, "rank": rank, "elements": bad})
                    cells += 1
        tally(phase, cells=cells, elements=elements)

    def busy_loops(cpus):
        code = "import os,sys\nos.sched_setaffinity(0,{int(sys.argv[1])})\nx=0\nwhile True:\n x+=1\n"
        return [subprocess.Popen([sys.executable, "-c", code, str(cpu)]) for cpu in cpus]

    def gpu_delay(rng):
        return rng.choice((0, 0, 2000, 20000, 200000, 600000))

    stats_before = runtime.stats() if runtime is not None else {}
    eager_phase("eager")
    graph_phase("graph")
    gather_phase("gather")
    scatter_phase("scatter")
    if not args.skip_stress:
        eager_phase("stress_eager_skew", delay=gpu_delay)
        graph_phase("stress_graph_skew", delay_cycles=gpu_delay)
        # Long idles: the proxy naps after ROCE_IDLE_SPINS polls; the next doorbell finds it asleep.
        eager_phase("stress_idle", idle_every=97, idle_s=2.5 if not selftest else 0.05, iters=1)
        cpus = list(stats_before.get("proxy_cpus") or [])
        loops = busy_loops(cpus) if (rank in (1, 3) and cpus) else []
        try:
            dist.barrier(group=cpu_group)
            eager_phase("stress_contended", delay=gpu_delay)
            graph_phase("stress_contended_graph")
        finally:
            for proc in loops:
                proc.kill()
        tally("stress_contended", note=f"rank {rank}: busy loops on proxy cpus {cpus}" if loops
              else f"rank {rank}: no busy loops")
    if adapter is not None:
        adapter.check_health()
    stats_after = runtime.stats() if runtime is not None else {}

    # ---------------------------------------------------------------- verdict
    local = {phase: {**entry, "target_hashes": {d: sorted(h) for d, h in entry["target_hashes"].items()}}
             for phase, entry in totals.items()}
    keys = ("last_seq", "ops_posted", "proxy_cpus", "proxy_cpu_observed", "stripe_count", "topology")
    report = {"rank": rank, "phases": local,
              "proxy": {"before": {k: stats_before.get(k) for k in keys},
                        "after": {k: stats_after.get(k) for k in keys}},
              "seconds": round(time.monotonic() - started, 1)}
    gathered = [None] * world
    dist.all_gather_object(gathered, report, group=cpu_group)
    failures = []
    if rank == 0:
        for phase in totals:
            mism = sum(g["phases"][phase]["element_mismatches"] for g in gathered)
            reps = sum(g["phases"][phase]["repeat_mismatches"] for g in gathered)
            per_dtype: dict = {}
            for g in gathered:
                for d, h in g["phases"][phase]["target_hashes"].items():
                    per_dtype.setdefault(d, set()).update(h)
            distinct = {d: len(h) for d, h in per_dtype.items()}
            print(json.dumps({"phase": phase, "cells_per_rank": totals[phase]["cells"],
                              "element_mismatches": mism, "repeat_mismatches": reps,
                              "distinct_target_row_hashes": distinct,
                              "failures": [f for g in gathered for f in g["phases"][phase]["failures"]][:8],
                              "notes": [n for g in gathered for n in g["phases"][phase]["notes"]]}), flush=True)
            if mism or reps or any(v > 1 for v in distinct.values()):
                failures.append(phase)
        print(json.dumps({"summary": "collective row invariance and run-to-run determinism",
                          "topology": topology, "world": world, "hidden": HIDDEN,
                          "sensitivity_target_elements_changed_by_other_orders": sens,
                          "m_values": M_VALUES, "graph_m": GRAPH_M, "graph_calls": GRAPH_CALLS,
                          "proxy": [g["proxy"] for g in gathered], "seconds": [g["seconds"] for g in gathered],
                          "failures": failures, "passed": not failures}), flush=True)
    verdict = torch.tensor([1 if failures else 0], dtype=torch.int32)
    dist.all_reduce(verdict, group=cpu_group)
    raise SystemExit(1 if int(verdict.item()) else 0)


if __name__ == "__main__":
    main()
