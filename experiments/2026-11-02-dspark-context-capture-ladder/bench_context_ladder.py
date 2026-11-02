#!/usr/bin/env python3
"""DSpark context graphs on one GB10 rank: replay time per row count and capture memory
for two capture ladders of the same ``DSparkContextCudaGraphs``.

Runs inside a spark-ds41f image with the patched ``dspark.py`` mounted. Both arms are the
patched class; the ``base`` arm replaces ``_context_capture_sizes`` with candidate-37's
powers of two plus the limit, so the capacities are the only difference.

Per rank, as at the shape's TP degree (``--shape``):
- ``main_proj``: the block-32 FP8 GEMM of this rank's padded column shard of the
  [5120, 15360] projection (TP3 1728 rows, TP4 1280), its per-token MXFP8 cast, a
  device-side stand-in for the all_gather (the shard copied ``tp`` times side by side:
  the gathered width without RoCE), and TP3's slice back to 5120 columns;
- ``main_norm``: TileLang RMSNorm over 5120 columns emitting main_x's MXFP8 cast;
- three draft layers: the [512, 5120] KV rows of the fused Q-A/KV weight (split-K, read
  from main_x's cast) and ``context_kv_front`` (norm, RoPE, SWA record write);
- the context input launch (``dspark_context.context_inputs``) ahead of every replay.
Weights are random; no checkpoint is read. The RoCE all_gather is not measured.
"""

import argparse
import gc
import json
import os
import socket
import statistics
import sys
import time
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn

WIDTH = 5120
NUM_AUX = 3
KV = 512
Q_LORA = 1024
EPS = 1e-6
PAGE = 128
PAGES = 1024
POSITIONS = 1 << 16
ROPE_DIM = 64
DEVICE = torch.device("cuda")
SHAPES = {"tp3": (3, 8), "tp4": (4, 16)}  # TP degree, max_num_seqs
DECODE_QUERY_LEN = 6  # 1 + 5 drafts


def recipe_ladder(rows):
    """scripts/recipe.py capture_sizes."""
    ladder = [1, 2, 3, 4, 6, 8, *range(12, 33, 4), *range(40, rows + 1, 8)]
    return [*(size for size in ladder if size < rows), rows]


def powers_of_two(vllm_config, limit):
    """candidate-37's DSparkContextCudaGraphs capacities."""
    capacities = []
    capacity = 1
    while capacity < limit:
        capacities.append(capacity)
        capacity *= 2
    if limit > 0:
        capacities.append(limit)
    return capacities


def block32(n, k, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    weight = (torch.randn((n, k), generator=gen, device=DEVICE) * 2).to(
        torch.float8_e4m3fn
    )
    exps = torch.randint(
        120, 125, (n // 32, k // 32), generator=gen, device=DEVICE
    ).to(torch.uint8)
    layer = nn.Module()
    layer.weight, layer.weight_scale_inv = weight, exps
    return layer


class MainProj:
    def __init__(self, tp):
        from vllm.models.deepseek_v4_1.tilelang.linear import TileLangBlock32Rows

        quantum = tp * 32
        self.padded = -(-WIDTH // quantum) * quantum
        self.shard = self.padded // tp
        self.tp = tp
        self.layer = block32(self.shard, NUM_AUX * WIDTH, 1)
        self.local = TileLangBlock32Rows(self.layer, 0)
        self.local.declare_kernels()

    def __call__(self, x):
        out = self.local(x)
        if self.tp > 1:
            out = torch.cat([out] * self.tp, dim=-1)
        if self.padded != WIDTH:
            out = out[..., :WIDTH].contiguous()
        return out


class Attn:
    def __init__(self, seed, table):
        from vllm.models.deepseek_v4_1.tilelang.cache_writer import RECORD_BYTES

        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        self.kv_weight = (
            torch.randn(KV, generator=gen, device=DEVICE) * 0.3 + 1
        ).bfloat16()
        self.table = table
        self.cache = torch.zeros(
            (PAGES, PAGE * RECORD_BYTES["swa"]), dtype=torch.uint8, device=DEVICE
        )

    def insert_raw_context_kv(self, kv_raw, positions, slot_mapping):
        from vllm.models.deepseek_v4_1.tilelang.front import context_kv_front

        context_kv_front(
            kv_raw, self.kv_weight, EPS, positions, self.table, self.cache,
            slot_mapping, PAGE,
        )


def rope_table():
    inv = 1.0 / (
        10000 ** (torch.arange(0, ROPE_DIM, 2, device=DEVICE).double() / ROPE_DIM)
    )
    angles = torch.arange(POSITIONS, device=DEVICE).double()[:, None] * inv
    return torch.cat((angles.cos(), angles.sin()), dim=-1).float()


def build_model(tp):
    from vllm.models.deepseek_v4_1.nvidia.dspark import DSparkDeepseekV4Model
    from vllm.models.deepseek_v4_1.tilelang.linear import TileLangBlock32Rows
    from vllm.models.deepseek_v4_1.tilelang.norm import TileLangRMSNorm

    gen = torch.Generator(device=DEVICE).manual_seed(7)
    model = DSparkDeepseekV4Model.__new__(DSparkDeepseekV4Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=WIDTH, dspark_target_layer_ids=[0, 1, 2])
    model.hidden_size = WIDTH
    model.main_proj = MainProj(tp)
    norm = TileLangRMSNorm(WIDTH, EPS).to(DEVICE)
    norm.with_cast = True
    norm.weight.data.copy_(torch.rand(WIDTH, generator=gen, device=DEVICE) + 0.5)
    model.main_norm = norm
    model._kv_layers = [block32(Q_LORA + KV, WIDTH, 10 + i) for i in range(3)]
    projections = []
    for layer in model._kv_layers:
        projection = TileLangBlock32Rows(layer, Q_LORA)
        projection.tilelang_mxfp8_input = True
        projection.declare_kernels()
        projections.append(projection)
    model._context_kv_projections = projections
    table = rope_table()
    model.layers = [SimpleNamespace(attn=Attn(20 + i, table)) for i in range(3)]
    model._context_main_cast = None
    return model


def decode_config(max_num_seqs, rows):
    from vllm.config import CompilationConfig, ParallelConfig, SchedulerConfig, VllmConfig

    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_AND_PIECEWISE",
        cudagraph_capture_sizes=recipe_ladder(rows),
    )
    compilation_config.max_cudagraph_capture_size = rows
    compilation_config.post_init_cudagraph_sizes()
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(
        max_num_seqs=max_num_seqs
    )
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.cache_config = SimpleNamespace(use_kda_recoverssm=False)
    vllm_config.num_speculative_tokens = DECODE_QUERY_LEN - 1
    vllm_config.num_verification_tokens = DECODE_QUERY_LEN - 1
    speculative_config = MagicMock()
    speculative_config.uses_acceptance_length_adaptation.return_value = False
    speculative_config.uses_batch_size_dynamic_speculative_decoding.return_value = False
    vllm_config.speculative_config = speculative_config
    return vllm_config


def mib(value):
    return round(value / (1 << 20), 3)


def settle():
    gc.collect()
    torch.accelerator.synchronize()
    torch.cuda.empty_cache()


def capture_arm(arm, model, vllm_config, buffers, rows):
    from vllm.models.deepseek_v4_1.nvidia import dspark

    original = dspark._context_capture_sizes
    if arm == "base":
        dspark._context_capture_sizes = powers_of_two
    try:
        # Cached free blocks released on both sides: what remains reserved after
        # capture is the graphs' private pool (and the context aux buffer).
        settle()
        free_before, _ = torch.cuda.mem_get_info()
        reserved_before = torch.cuda.memory_reserved()
        start = time.perf_counter()
        graphs = dspark.DSparkContextCudaGraphs(
            model, vllm_config, *buffers, None, rows
        )
        graphs.capture()
        torch.accelerator.synchronize()
        seconds = time.perf_counter() - start
        settle()
        free_after, _ = torch.cuda.mem_get_info()
        reserved_after = torch.cuda.memory_reserved()
    finally:
        dspark._context_capture_sizes = original
    sizes = list(graphs.manager.compilation_config.cudagraph_capture_sizes)
    return graphs, {
        "capacities": sizes,
        "graphs": len(graphs.manager.graphs),
        "capture_seconds": round(seconds, 3),
        "device_free_delta_mib": mib(free_before - free_after),
        "torch_reserved_delta_mib": mib(reserved_after - reserved_before),
    }


def capacity_of(graphs, num_tokens):
    return graphs.manager.dispatch(1, num_tokens, None, 0).num_tokens


def time_hot(graphs, aux, num_tokens, iters):
    for _ in range(20):
        graphs.run(aux, num_tokens)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graphs.run(aux, num_tokens)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def time_cold(graphs, aux, num_tokens, iters, flush):
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in events:
        flush.zero_()
        start.record()
        graphs.run(aux, num_tokens)
        end.record()
    torch.accelerator.synchronize()
    return statistics.median(s.elapsed_time(e) * 1000 for s, e in events)


def step_rows(rng, max_num_seqs, rows):
    """A step's target rows: a decode batch (1..max_num_seqs requests, each verifying
    1..6 rows: adaptive verification trims drafts), or with a third's chance a
    prefill chunk beside it, its total kept within the graphs' limit."""
    requests = rng.randint(1, max_num_seqs)
    total = sum(rng.randint(1, DECODE_QUERY_LEN) for _ in range(requests))
    if rng.random() < 1 / 3 and total < rows:
        total += rng.randint(1, rows - total)
    return min(total, rows)


def soak(args, graphs, arms, model, buffers, aux, scratch, rows, gen):
    """Random steps, each replayed under both ladders from the same state on the same
    inputs, with everything outside the step's live rows poisoned: NaN in the shared
    workspace (stale rows the decode tiles read past a capacity), in the sources' and
    the context buffer's rows past the step, in each graph's aux buffer, and valid
    cache slots past the step (a padded row that kept one would overwrite a record).
    Each step's context rows and all three caches must match bit for bit, and the
    context rows must be finite."""
    import random

    hidden_states, positions, slot_mappings = buffers
    max_num_seqs = SHAPES[args.shape][1]
    rng = random.Random(11)
    nan_bf16 = torch.tensor(float("nan"), dtype=torch.bfloat16, device=DEVICE)
    caches = [layer.attn.cache for layer in model.layers]
    for cache in caches:
        cache.random_(generator=gen)
    capacities = defaultdict(int)
    mismatches, nonfinite = [], []
    sizes = []
    for step in range(args.soak):
        num_tokens = step_rows(rng, max_num_seqs, rows)
        sizes.append(num_tokens)
        live_positions = torch.randint(0, POSITIONS, (rows,), generator=gen, device=DEVICE)
        live_slots = torch.randperm(PAGES * PAGE, generator=gen, device=DEVICE)[:rows]
        for i in range(NUM_AUX):
            aux[i][:num_tokens].normal_(generator=gen)
        state = [cache.clone() for cache in caches]
        outputs = {}
        for arm in arms:
            for cache, saved in zip(caches, state):
                cache.copy_(saved)
            scratch.fill_(0xFF)  # NaN as E4M3, BF16 pairs and UE8M0 scales
            for i in range(NUM_AUX):
                aux[i][num_tokens : rows + 64].fill_(nan_bf16)
            hidden_states[num_tokens:].fill_(nan_bf16)
            for graph in graphs.values():
                graph.aux.fill_(nan_bf16)
            positions.copy_(torch.randint(0, POSITIONS, positions.shape, generator=gen, device=DEVICE))
            positions[:rows].copy_(live_positions)
            slot_mappings[0].copy_(
                torch.randint(0, PAGES * PAGE, slot_mappings[0].shape, generator=gen, device=DEVICE)
            )
            slot_mappings[0, :rows].copy_(live_slots)
            if args.soak_control and arm == "ladder":
                # Negative control: without the input launch the graph reads the
                # poisoned aux buffer, and the soak must report it.
                from vllm.models.deepseek_v4_1.nvidia import dspark

                launch = dspark.dspark_context.context_inputs
                dspark.dspark_context.context_inputs = lambda *_: None
                try:
                    graphs[arm].run(aux, num_tokens)
                finally:
                    dspark.dspark_context.context_inputs = launch
            else:
                graphs[arm].run(aux, num_tokens)
            torch.accelerator.synchronize()
            capacities[(arm, capacity_of(graphs[arm], num_tokens))] += 1
            context = hidden_states[:num_tokens].clone()
            if not torch.isfinite(context.float()).all():
                nonfinite.append((step, arm, num_tokens))
            outputs[arm] = (context, [cache.clone() for cache in caches])
        (h0, c0), (h1, c1) = outputs[arms[0]], outputs[arms[1]]
        if not (
            torch.equal(h0.view(torch.int16), h1.view(torch.int16))
            and all(torch.equal(a, b) for a, b in zip(c0, c1))
        ):
            mismatches.append((step, num_tokens))
    counts = defaultdict(int)
    for size in sizes:
        counts[size] += 1
    ladder_capacities = sorted(c for (arm, c) in capacities if arm == "ladder")
    return {
        "summary": {
            "steps": args.soak,
            "mismatched_steps": len(mismatches),
            "nonfinite_context_rows": len(nonfinite),
            "row_counts_seen": len(counts),
            "ladder_capacities_replayed": len(ladder_capacities),
            "ladder_capacities_captured": len(
                graphs["ladder"].manager.compilation_config.cudagraph_capture_sizes
            ),
        },
        "mismatches": mismatches[:50],
        "nonfinite": nonfinite[:50],
        "row_counts": dict(sorted(counts.items())),
        "replays_by_capacity": {
            f"{arm}:{capacity}": count for (arm, capacity), count in sorted(capacities.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    parser.add_argument("--first", choices=("base", "ladder"), required=True)
    parser.add_argument("--hot-iters", type=int, default=300)
    parser.add_argument("--cold-iters", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--soak",
        type=int,
        default=0,
        help="this many random steps on NaN-poisoned buffers, both arms, no timing",
    )
    parser.add_argument(
        "--soak-control",
        action="store_true",
        help="with --soak: skip the ladder arm's input launch (must be detected)",
    )
    parser.add_argument(
        "--memory-copies",
        type=int,
        default=0,
        help="capture only: this many copies of --first's ladder, no replay",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.deepseek_v4_1.tilelang import prepare
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.workspace import use_preallocated_workspace

    tp, max_num_seqs = SHAPES[args.shape]
    rows = max_num_seqs * DECODE_QUERY_LEN
    arms = [args.first, "ladder" if args.first == "base" else "base"]
    torch.cuda.set_device(0)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    record = {
        "shape": args.shape,
        "tp": tp,
        "max_num_seqs": max_num_seqs,
        "limit": rows,
        "order": arms,
        "device": torch.cuda.get_device_name(0),
        "hostname": os.environ.get("BENCH_HOST", socket.gethostname()),
        "hot_iters": args.hot_iters,
        "cold_iters": args.cold_iters,
        "rounds": args.rounds,
    }
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        model = build_model(tp)
        record["main_proj_shard_rows"] = model.main_proj.shard
        vllm_config = decode_config(max_num_seqs, rows)
        max_tokens = 4096
        hidden_states = torch.zeros(max_tokens, WIDTH, dtype=torch.bfloat16, device=DEVICE)
        positions = torch.zeros(max_tokens, dtype=torch.int64, device=DEVICE)
        slot_mappings = torch.full(
            (1, max_tokens), PAD_SLOT_ID, dtype=torch.int64, device=DEVICE
        )
        buffers = (hidden_states, positions, slot_mappings)
        gen = torch.Generator(device=DEVICE).manual_seed(3)
        aux = [
            torch.randn(max_tokens, WIDTH, generator=gen, device=DEVICE).bfloat16()
            for _ in range(NUM_AUX)
        ]
        live_positions = torch.randint(
            0, POSITIONS, (max_tokens,), generator=gen, device=DEVICE
        )
        live_slots = torch.randperm(PAGES * PAGE, generator=gen, device=DEVICE)[
            :max_tokens
        ]
        scratch = torch.empty(256 << 20, dtype=torch.uint8, device=DEVICE)
        flush = torch.empty(256 << 20, dtype=torch.uint8, device=DEVICE)

        def set_live_rows():
            positions[:rows].copy_(live_positions[:rows])
            slot_mappings[0, :rows].copy_(live_slots[:rows])

        with use_preallocated_workspace(scratch):
            # Every kernel the base ladder's sizes need, eagerly, then refuse any
            # TileLang kernel not yet built: a new ladder size that needed one fails.
            for capacity in powers_of_two(None, rows):
                joined = torch.cat([a[:capacity] for a in aux], dim=-1)
                model.combine_hidden_states(joined, out=hidden_states[:capacity])
                set_live_rows()
                model.precompute_and_store_context_kv(
                    hidden_states[:capacity], positions[:capacity],
                    slot_mappings[0, :capacity],
                )
            from vllm.models.deepseek_v4_1.tilelang import dspark_context

            probe_aux = torch.zeros(rows, NUM_AUX * WIDTH, dtype=torch.bfloat16, device=DEVICE)
            dspark_context.context_inputs(probe_aux, aux, positions, slot_mappings, 1, 2)
            torch.accelerator.synchronize()
            prepare.freeze("context ladder bench: kernels of the base sizes only")
            record["frozen_before_capture"] = True
            print("TileLang cache frozen", flush=True)

            if args.memory_copies:
                # Device memory is shared with the host on GB10, so one capture's
                # delta drowns in system noise: capture the first arm's ladder
                # ``memory_copies`` times (one shared graph pool, as in serving)
                # and divide. Each copy also allocates its aux buffer.
                settle()
                free_before, _ = torch.cuda.mem_get_info()
                reserved_before = torch.cuda.memory_reserved()
                copies = [
                    capture_arm(args.first, model, vllm_config, buffers, rows)
                    for _ in range(args.memory_copies)
                ]
                settle()
                free_after, _ = torch.cuda.mem_get_info()
                reserved_after = torch.cuda.memory_reserved()
                aux_bytes = copies[0][0].aux.numel() * copies[0][0].aux.element_size()
                record["memory"] = {
                    "arm": args.first,
                    "copies": args.memory_copies,
                    "graphs_per_copy": copies[0][1]["graphs"],
                    "aux_mib_per_copy": mib(aux_bytes),
                    "device_free_delta_mib": mib(free_before - free_after),
                    "torch_reserved_delta_mib": mib(reserved_after - reserved_before),
                    "capture_seconds": [c[1]["capture_seconds"] for c in copies],
                }
                print("memory", json.dumps(record["memory"]), flush=True)
                for copy, _ in copies:
                    copy.close()
                with open(args.out, "w") as f:
                    json.dump(record, f, indent=1)
                print("wrote", args.out)
                return 0

            graphs = {}
            record["capture"] = {}
            for arm in arms:
                graphs[arm], record["capture"][arm] = capture_arm(
                    arm, model, vllm_config, buffers, rows
                )
                print(arm, json.dumps(record["capture"][arm]), flush=True)

            if args.soak:
                record["soak"] = soak(
                    args, graphs, arms, model, buffers, aux, scratch, rows, gen
                )
                print("soak", json.dumps(record["soak"]["summary"]), flush=True)
                for arm in arms:
                    graphs[arm].close()
                with open(args.out, "w") as f:
                    json.dump(record, f, indent=1)
                print("wrote", args.out)
                return 0

            # Live rows' outputs are the same bits under either ladder.
            mismatches = []
            for num_tokens in range(1, rows + 1):
                outputs = {}
                for arm in arms:
                    for layer in model.layers:
                        layer.attn.cache.zero_()
                    set_live_rows()
                    graphs[arm].run(aux, num_tokens)
                    torch.accelerator.synchronize()
                    outputs[arm] = (
                        hidden_states[:num_tokens].clone(),
                        [layer.attn.cache.clone() for layer in model.layers],
                    )
                (h0, c0), (h1, c1) = outputs[arms[0]], outputs[arms[1]]
                same = torch.equal(h0.view(torch.int16), h1.view(torch.int16)) and all(
                    torch.equal(a, b) for a, b in zip(c0, c1)
                )
                if not same:
                    mismatches.append(num_tokens)
            record["bit_mismatch_rows"] = mismatches
            print("bit mismatches", mismatches, flush=True)

            replay = {arm: {} for arm in arms}
            for round_index in range(args.rounds):
                order = arms if round_index % 2 == 0 else arms[::-1]
                for num_tokens in range(1, rows + 1):
                    for arm in order:
                        set_live_rows()
                        entry = replay[arm].setdefault(
                            str(num_tokens),
                            {"capacity": capacity_of(graphs[arm], num_tokens),
                             "hot_us": [], "cold_us": []},
                        )
                        entry["hot_us"].append(
                            round(time_hot(graphs[arm], aux, num_tokens, args.hot_iters), 3)
                        )
                        entry["cold_us"].append(
                            round(
                                time_cold(
                                    graphs[arm], aux, num_tokens, args.cold_iters, flush
                                ),
                                3,
                            )
                        )
                print(f"round {round_index} done", flush=True)
            record["replay"] = replay
            for arm in arms:
                graphs[arm].close()
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
