"""End-of-turn cost of the shadow turn learner at serving shapes: rank 0 of four
(vocabulary shard 32,320 of 129,280, W1 129,280 x 256, five positions, 16 block maxima
a position, 512 draft steps kept a request, 16 request slots), the learner's own
``finish`` on a synthetic packed W2 shard and rings filled like finished turns.

Three timings a case, each the median of the repeats:
- ``gpu_ms``: the GPU's time for the end of turn alone (a spin queued ahead of the call
  keeps the host's launches ahead of the GPU);
- ``idle_ms``: what rank 0 logs (its events around the end of turn) with the GPU idle
  at the call, as at a step boundary;
- ``host_ms``: the host's time inside ``finish``.
Then, for one case, the GPU time and host time a launch.

The collectives are one-GPU stand-ins: the all-gather repeats rank 0's summaries four
times (the gathered layout) and the all-reduce returns its input. Their network time is
not in these numbers."""
import json
import statistics
import time
import types

import torch

from vllm.models.deepseek_v4_1.tilelang import embedding as emb
from vllm.models.deepseek_v4_1.tilelang.vocab import pack_bf16_weight
from vllm.v1.worker.gpu.spec_decode.dspark import turn_learning_shadow as sh

DEV = torch.device("cuda")
N, DIM, RANKS = 5, 256, 4
VOCAB = 129280
SHARD = VOCAB // RANKS
PARTS = -(-SHARD // sh.GREEDY_BLOCK)
SLOTS, CAP = 16, 512
MAX_LEN = 8192
REPS = 15
g = torch.Generator().manual_seed(0)
torch.manual_seed(0)

# Rank 0's W2 shard, packed as the checkpoint's is at load; W1 replicated.
w2_rows = (torch.randn(SHARD, DIM, device=DEV) * 0.02).to(torch.bfloat16)
packed = pack_bf16_weight(w2_rows.clone())
del w2_rows
w2 = types.SimpleNamespace(
    weight=packed.values,
    weight_exception_entries=packed.exception_entries,
    weight_row_offsets=packed.row_offsets,
    packed_bf16_in_features=DIM,
    packed_bf16_exponent_base=packed.exponent_base,
    shard_indices=types.SimpleNamespace(
        org_vocab_start_index=0,
        org_vocab_end_index=SHARD,
        num_org_vocab_padding=0,
        added_vocab_start_index=SHARD,
        added_vocab_end_index=SHARD,
    ),
)
table = (torch.randn(VOCAB, DIM, device=DEV) * 0.02).to(torch.bfloat16)


def gather(t: torch.Tensor) -> torch.Tensor:
    return t.repeat(1, RANKS)


def reduce(t: torch.Tensor) -> torch.Tensor:
    return t


learner = sh.DSparkShadowLearner(
    model=types.SimpleNamespace(draft_vocab_shard=(0, SHARD)),
    table=table,
    markov_w2=w2,
    draft_tokens=torch.zeros((SLOTS, N), dtype=torch.int64, device=DEV),
    input_ids=torch.zeros(SLOTS * 8, dtype=torch.int32, device=DEV),
    anchor_idx=torch.zeros(SLOTS, dtype=torch.int64, device=DEV),
    sample_idx_mapping=torch.zeros(SLOTS * N, dtype=torch.int32, device=DEV),
    sample_pos=torch.zeros(SLOTS * N, dtype=torch.int64, device=DEV),
    all_gather=lambda t: gather(t),
    all_reduce=lambda t: reduce(t),
    max_num_reqs=SLOTS,
    num_speculative_steps=N,
    tp_size=RANKS,
    tp_rank=0,
    device=DEV,
    capacity=CAP,
    lr=0.3,
)
assert learner.parts == PARTS


def fill(kept: list[int], accept: float = 0.7, has: float = 0.85):
    """Rings of ``len(kept)`` finished turns: each kept draft step drafted from the
    committed anchor, its maxima holding the committed token in that token's block
    (``has``), the draft right where ``accept`` says."""
    tokens = torch.zeros((SLOTS, MAX_LEN), dtype=torch.int32)
    total = torch.zeros(SLOTS, dtype=torch.int32)
    for s, k in enumerate(kept):
        length = 100 + 3 * k + N + 2
        seq = torch.randint(0, SHARD, (length,), generator=g, dtype=torch.int64)
        tokens[s, :length] = seq.int()
        total[s] = length
        pos = 100 + 3 * torch.arange(k)
        meta = torch.stack([pos, torch.full_like(pos, N), seq[pos]], dim=1)
        label = seq[pos[:, None] + 1 + torch.arange(N)]
        block = label // sh.GREEDY_BLOCK
        base = torch.arange(PARTS) * sh.GREEDY_BLOCK
        width = (SHARD - base).clamp(max=sh.GREEDY_BLOCK)
        cand = base + (torch.rand(k, N, PARTS, generator=g) * width).long()
        hit = torch.rand(k, N, generator=g) < has
        cand.scatter_(2, block[..., None], torch.where(hit, label, cand.gather(2, block[..., None])[..., 0])[..., None])
        val = torch.randn(k, N, PARTS, generator=g)
        right = (torch.rand(k, N, generator=g) < accept) & hit
        other = (block + 1 + (torch.rand(k, N, generator=g) * (PARTS - 1)).long()) % PARTS
        val.scatter_(2, torch.where(right, block, other)[..., None], (val.amax(-1) + 1)[..., None])
        val = val.to(torch.bfloat16).float()
        best = val.amax(-1, keepdim=True)
        draft = torch.where(val == best, cand, torch.full_like(cand, 1 << 30)).amin(-1)
        drafts = torch.full((k, N), -1, dtype=torch.int32)
        drafts[:, : N - 1] = draft[:, : N - 1].int()
        learner.ring_values[s, :k] = val.to(DEV)
        learner.ring_index[s, :k] = cand.int().to(DEV)
        learner.ring_meta[s, :k] = meta.to(DEV)
        learner.ring_drafts[s, :k] = drafts.to(DEV)
    return tokens.to(DEV), total.to(DEV)


def restore(kept: list[int]) -> None:
    learner.ring_count[: len(kept)] = torch.tensor(kept, dtype=torch.int64, device=DEV)
    learner.host_count[: len(kept)] = kept
    learner.calls_since_check = 0


def logged_ms() -> float:
    start, end = learner._log_queue[-1][:2]
    learner._log_queue.clear()
    return start.elapsed_time(end)


# The spin: long enough for the host to queue the largest case behind it.
spin_a, spin_b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
spin_a.record()
torch.cuda._sleep(10_000_000)
spin_b.record()
torch.cuda.synchronize()
SPIN = int(10_000_000 * 60.0 / spin_a.elapsed_time(spin_b))  # about 60 ms
_warm = torch.randn(1 << 22, device=DEV)
for _ in range(2000):
    _warm.mul_(1.0)
torch.cuda.synchronize()


def run(kept: list[int]) -> dict:
    tokens, total = fill(kept)
    slots = list(range(len(kept)))
    gpu, idle, host = [], [], []
    for rep in range(REPS + 2):
        restore(kept)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        learner.finish(slots, [], tokens, total)
        h = time.perf_counter() - t0
        torch.cuda.synchronize()
        i = logged_ms()
        restore(kept)
        torch.cuda.synchronize()
        torch.cuda._sleep(SPIN)
        t0 = time.perf_counter()
        learner.finish(slots, [], tokens, total)
        h2 = time.perf_counter() - t0
        torch.cuda.synchronize()
        q = logged_ms()
        assert h2 * 1000 < 50, "the host did not stay ahead of the spin"
        if rep >= 2:
            gpu.append(q)
            idle.append(i)
            host.append(h * 1000)
    stats = torch.zeros(sh._COUNTS + 1, dtype=torch.float64)
    stats.copy_(learner.call_totals.cpu())
    return {
        "turns": len(kept),
        "kept": kept[0] if len(set(kept)) == 1 else kept,
        "chunks": -(-sum(kept) // learner.chunk),
        "gpu_ms": round(statistics.median(gpu), 3),
        "idle_ms": round(statistics.median(idle), 3),
        "host_ms": round(statistics.median(host), 3),
        "idle_range_ms": [round(min(idle), 3), round(max(idle), 3)],
        "last_call": sh.ShadowStats.as_dict([int(v) for v in stats[: sh._COUNTS].tolist()]),
    }


CASES = [[64], [128], [256], [384], [512], [512, 512], [512] * 4, [512] * 8]
for kept in CASES:
    print(json.dumps(run(kept)), flush=True)


# GPU and host time a launch, one turn of 512 kept steps (two chunks).
class Timer:
    def __init__(self) -> None:
        self.events: list = []

    def wrap(self, label: str, fn):
        def timed(*args, **kwargs):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            a.record()
            out = fn(*args, **kwargs)
            b.record()
            self.events.append((label, a, b, time.perf_counter() - t0))
            return out

        return timed


names = {
    id(sh.labels_kernel): "labels",
    id(sh.reset_kernel): "reset",
    id(sh.shadow_local_kernel): "local",
    id(sh.shadow_combine_kernel): "combine",
    id(sh.shadow_update_kernel): "update",
    id(sh.shadow_totals_kernel): "totals",
}
timer = Timer()
orig_kernel, orig_h2d, orig_decode = sh._kernel, sh.async_tensor_h2d, emb.vocab_embedding_packed
sh._kernel = lambda factory, *key: timer.wrap(names.get(id(factory), "other"), orig_kernel(factory, *key))
sh.async_tensor_h2d = timer.wrap("h2d", orig_h2d)
emb.vocab_embedding_packed = timer.wrap("decode", orig_decode)
gather_plain, reduce_plain = gather, reduce
gather = timer.wrap("all_gather (stand-in)", gather_plain)
reduce = timer.wrap("all_reduce (stand-in)", reduce_plain)

kept = [512]
tokens, total = fill(kept)
per: dict[str, list] = {}
for rep in range(REPS + 2):
    restore(kept)
    timer.events.clear()
    torch.cuda.synchronize()
    torch.cuda._sleep(SPIN)
    learner.finish([0], [], tokens, total)
    torch.cuda.synchronize()
    learner._log_queue.clear()
    if rep < 2:
        continue
    sums: dict[str, list[float]] = {}
    for label, a, b, h in timer.events:
        s = sums.setdefault(label, [0.0, 0.0, 0])
        s[0] += a.elapsed_time(b)
        s[1] += h * 1000
        s[2] += 1
    for label, s in sums.items():
        per.setdefault(label, []).append(s)
print(
    json.dumps(
        {
            "per_launch": {
                label: {
                    "launches": v[0][2],
                    "gpu_ms": round(statistics.median(x[0] for x in v), 4),
                    "host_ms": round(statistics.median(x[1] for x in v), 4),
                }
                for label, v in per.items()
            },
            "kept": 512,
            "note": "events around each launch; gpu_ms includes each event pair's own overhead",
        }
    ),
    flush=True,
)
