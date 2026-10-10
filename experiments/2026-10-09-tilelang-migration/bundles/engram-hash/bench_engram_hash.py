"""Port B9: the Engram n-gram hash for both Engram layers (1 and 14), B12X's path as
NgramHashState.run_native drives it (vLLM's _prepare_metadata, then compress, request
ids, hash and a copy per layer) against the one-launch TileLang kernel, at TP4's 16
requests and 8192 tokens.

1. Bits: every step's hashes and live counts equal B12X's, exactly.
2. Time per step, eager (production hashes outside CUDA graphs, before each forward:
   host launch cost included, measured to the end of the GPU work) and under CUDA
   graphs (GPU time), at decode steps (1-16 requests x 1-6 tokens) and prefill.

Exits 1 if any hash differs or TileLang is slower than B12X at any step.
"""
import statistics
import sys
import time

import torch

sys.path.insert(0, "/b")
from kbench import CAPACITY, DEVICE, REPLAYS, per_call, repeatable  # noqa: E402

import triton  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from b12x.preparation.types import require_prepared  # noqa: E402
from b12x.sequence import engram as native  # noqa: E402
from b12x.sequence.engram._impl import _bind_state  # noqa: E402

from vllm.models.deepseek_v4_1.common.engram import _prepare_metadata  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.engram_hash import (  # noqa: E402
    IMAGE_SENTINEL, engram_hash, geometry_tensors,
)

VOCAB, COMPRESSED, SEQS, LAYERS = 129280, 99092, 16, (1, 14)
STEPS = {  # name: tokens per request
    "c1 x1": [1], "c1 x6": [6], "c8 x6": [6] * 8, "c16 x6": [6] * 16, "c16 mixed": [1, 6, 3, 2] * 4,
    "prefill 2048": [2048], "prefill 8192 / 4": [2048] * 4,
}
failures = []
gen = torch.Generator(device=DEVICE).manual_seed(91)
geometry = native.build_geometry(layer_ids=LAYERS, base_table_size=16_000_000, compressed_vocab_size=COMPRESSED)
token_map = torch.randint(0, COMPRESSED, (VOCAB,), generator=gen, device=DEVICE)
token_map_list = token_map.tolist()
# B12X requires the map to cover the whole compressed vocabulary.
token_map_list[:COMPRESSED] = list(range(COMPRESSED))
token_map = torch.tensor(token_map_list, dtype=torch.int64, device=DEVICE)
tl_geometry = geometry_tensors(geometry, DEVICE)

caps = [native.Caps(device=DEVICE, max_tokens=CAPACITY, max_seqs=SEQS, max_requests=SEQS, vocab_size=VOCAB,
                    layer_id=layer, tp_size=4, tp_rank=0) for layer in LAYERS]
plans = [native.plan(c, token_map=token_map_list, geometry=geometry) for c in caps]
buffers = {name: torch.empty(shape, dtype=dtype, device=DEVICE) for name, shape, dtype in (
    ("ids", (CAPACITY,), torch.int64), ("mask", (CAPACITY,), torch.bool), ("starts", (SEQS + 1,), torch.int32),
    ("history", (SEQS, 3), torch.int64), ("slots", (SEQS,), torch.int32), ("num_seqs", (1,), torch.int32),
    ("num_tokens", (1,), torch.int32))}
tl_counts = (torch.empty(1, dtype=torch.int32, device=DEVICE), torch.empty(1, dtype=torch.int32, device=DEVICE))


def prepare(state):
    (spec,) = state.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    hashes = torch.empty((CAPACITY, 24), dtype=torch.int64, device=DEVICE)
    binding = _bind_state(state, scratch=scratch, token_ids=buffers["ids"], token_mask=buffers["mask"],
                          query_start_loc=buffers["starts"], request_slots=buffers["slots"],
                          committed_history=buffers["history"], num_seqs=buffers["num_seqs"],
                          num_tokens=buffers["num_tokens"], hash_ids=hashes)
    return PreparedCall(run=lambda: state.run(binding, 1))


with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
    session.prepare(tuple(p.request(name=f"engram/hash/{i}", prepare_call=prepare) for i, p in enumerate(plans)))
    # Bound once per layer, as NgramHashState._ensure_bindings binds them.
    bindings = []
    for plan in plans:
        (spec,) = require_prepared(plan, "sequence.engram").scratch_specs()
        bindings.append(native.bind(plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=DEVICE),
                                    token_ids=buffers["ids"], token_mask=buffers["mask"],
                                    query_start_loc=buffers["starts"], request_slots=buffers["slots"],
                                    committed_history=buffers["history"], num_seqs=buffers["num_seqs"],
                                    num_tokens=buffers["num_tokens"],
                                    hash_ids=torch.empty((CAPACITY, 24), dtype=torch.int64, device=DEVICE)))

    def b12x(ids, keep, starts, history, out):
        seqs = starts.numel() - 1
        work = max(CAPACITY, SEQS * 3, SEQS + 1)
        _prepare_metadata[(triton.cdiv(work, 256),)](
            ids, keep, starts, history, token_map, buffers["ids"], buffers["mask"], buffers["starts"],
            buffers["history"], buffers["slots"], buffers["num_seqs"], buffers["num_tokens"], ids.numel(), seqs,
            CAPACITY, SEQS, history.stride(0), VOCAB, 256)
        for i, binding in enumerate(bindings):
            native.run(binding, token_count=ids.numel())
            out[:, i].copy_(binding.hash_ids[: ids.numel()])

    def tilelang(ids, image, starts, history, out):
        engram_hash(ids, image, starts, history, token_map, *tl_geometry, out, *tl_counts)

    def eager(fn, reps=200):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            begin = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - begin) * 1e6)
        return statistics.median(times)

    print("step: eager us (host + GPU) b12x / tilelang; graph us (GPU) b12x / tilelang", flush=True)
    for name, lengths in STEPS.items():
        live = sum(lengths)
        padded = live if live > 96 else live + 3  # captured decode steps carry padding
        ids = torch.randint(0, VOCAB, (padded,), generator=gen, device=DEVICE, dtype=torch.int64)
        ids[torch.rand(padded, generator=gen, device=DEVICE) < 0.05] = IMAGE_SENTINEL
        image = ids == IMAGE_SENTINEL
        keep = ~image
        starts = torch.tensor([0, *lengths], dtype=torch.int32, device=DEVICE).cumsum(0, dtype=torch.int32)
        history = torch.randint(0, VOCAB, (len(lengths), 3), generator=gen, device=DEVICE)
        history[torch.rand(len(lengths), 3, generator=gen, device=DEVICE) < 0.2] = -1
        b_out = torch.empty((padded, len(LAYERS), 24), dtype=torch.int64, device=DEVICE)
        t_out = torch.full_like(b_out, 7)
        b12x(ids, keep, starts, history, b_out)
        tilelang(ids, image, starts, history, t_out)
        torch.cuda.synchronize()
        if not torch.equal(b_out[:live], t_out[:live]) or not (t_out[live:] == -1).all():
            failures.append(f"{name}: hashes differ")
        if (int(tl_counts[0]), int(tl_counts[1])) != (int(buffers["num_seqs"]), int(buffers["num_tokens"])):
            failures.append(f"{name}: counts differ")
        if repeatable(lambda: tilelang(ids, image, starts, history, t_out), lambda: [t_out, *tl_counts]):
            failures.append(f"{name}: not repeatable over {REPLAYS} replays")
        b_eager = eager(lambda: b12x(ids, keep, starts, history, b_out))
        t_eager = eager(lambda: tilelang(ids, image, starts, history, t_out))
        b_graph = per_call(lambda: b12x(ids, keep, starts, history, b_out), padded)[0]
        t_graph = per_call(lambda: tilelang(ids, image, starts, history, t_out), padded)[0]
        print(f"  {name:18s} eager {b_eager:8.1f} / {t_eager:8.1f}  graph {b_graph:8.2f} / {t_graph:8.2f}", flush=True)
        if t_eager > b_eager or t_graph > b_graph * 1.03:
            failures.append(f"{name}: TileLang slower")

for failure in failures:
    print("FAIL", failure)
sys.exit(1 if failures else 0)
