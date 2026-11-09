"""DeepSeek V4.1 attention, layers 20 and 24, from the serving code on one GPU at a TP
rank's shapes, with the sequence-parallel all-gather simulated across ranks.

The modules are vLLM's own ``DeepseekV41TileLangAttention`` (the class
``_select_dsv4_attn_cls`` picks for ``--attention-backend TILELANG``), built under a
``VllmConfig`` that ``EngineArgs`` makes from the served checkpoint's config.json with
the recipe's flags (TileLang attention/linear/MoE backends, block size 256,
max_model_len 1048576, max_num_seqs 16, max_num_batched_tokens 16384) and the TP4
recipe's speculative geometry (DSpark, 5 drafts, copy drafts verify 16, so the decode
threshold is 17 rows a request and the decode arena 16 x 17 rows). Single process, TP
group of 1: a TP-W rank's attention shapes come from the per-rank head and output-group
counts (TP3: 72 padded heads / 3 = 24, 3 groups; TP4: 16, 2; TP8: 8, 1). The indexer is
replicated (32 heads at every TP), as in serving.

Layer 20 is the KV, index and candidate source (compressor, ratio 1). Layer 24 is an
index source without a compressor: the path #1009 changes (``_sp_front``). Random
weights in checkpoint dtypes go through ``process_weights_after_loading`` and
``prepare.declare_modules`` (kernels compiled, plans bound, staging allocated), the KV
caches are allocated at the specs' page sizes and bound, and each step's metadata is
built by the real ``DeepseekV41B12xMetadataBuilder`` from ``BlockTables`` slot
mappings, one builder per cache group.

A step: layer 20 runs on all T rows (no SP), writing the step's main KV, index keys,
SWA records and candidates, as the earlier layer of the same forward does. Then layer
24 runs once per TP rank r under ``sp_prefill.activate(SPRows(T, W, r))`` with that
rank's L local hidden rows (zero padding past T), from the same cache state. The TP
all-gather (``sp_prefill.tensor_model_parallel_all_gather``; with a TP group of 1 the
grouped PyNCCL path is off, so every SP gather goes through it) returns the
concatenation of every rank's input to the same call. Each rank's inputs are recorded
and the ranks are re-run until every recorded input is unchanged (a fixed point: the
real TP result). WO (``_o_proj``) is replaced by the identity, so the step's result is
WO's input: every rank's attention output for all T rows. No PR changes WO.
"""

import os
import sys
import time
import traceback
import zlib
from types import SimpleNamespace

import torch

MODEL = "/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
DEV = torch.device("cuda")
LAYERS = (20, 24)
TARGET = 24


def per_rank_heads(world: int) -> tuple[int, int]:
    groups = -(-8 // world) * world  # update_model_config_for_parallelism
    heads = groups * 8
    return heads // world, groups // world


def make_config(world: int):
    from vllm.engine.arg_utils import EngineArgs

    heads, groups = per_rank_heads(world)
    args = EngineArgs(
        model=MODEL,
        tokenizer=MODEL,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        load_format="dummy",
        max_model_len=1048576,
        max_num_seqs=16,
        max_num_batched_tokens=16384,
        block_size=256,
        attention_backend="TILELANG",
        linear_backend="tilelang",
        moe_backend="tilelang",
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"]},
        hf_overrides={"num_attention_heads": heads, "o_groups": groups},
        enforce_eager=False,
    )
    vc = args.create_engine_config()
    vc.speculative_config = SimpleNamespace(
        method="dspark",
        num_speculative_tokens=5,
        num_verification_tokens=16,
        parallel_drafting=False,
        use_dspark=lambda: True,
        use_dflash=lambda: False,
    )
    return vc


_DIST = False


def init_dist(vc):
    global _DIST
    if _DIST:
        return
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    with set_current_vllm_config(vc):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:29531",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(DEV)
    _DIST = True


def _fill(name, p, g):
    t = p.data
    if t.dtype == torch.float8_e4m3fn:
        t.copy_((torch.randn(t.shape, generator=g, device=DEV) * 0.6).to(t.dtype))
    elif t.dtype == torch.float8_e8m0fnu:
        t.view(torch.uint8).copy_(
            torch.randint(120, 123, t.shape, generator=g, device=DEV, dtype=torch.int32).to(torch.uint8)
        )
    elif t.dtype in (torch.uint8, torch.int8):
        if "scale" in name:
            t.view(torch.uint8).copy_(
                torch.randint(120, 123, t.shape, generator=g, device=DEV, dtype=torch.int32).to(torch.uint8)
            )
        else:
            t.view(torch.uint8).copy_(
                torch.randint(0, 256, t.shape, generator=g, device=DEV, dtype=torch.int32).to(torch.uint8)
            )
    elif t.dtype in (torch.int32, torch.int64):
        t.zero_()
    elif t.is_floating_point():
        if name.endswith("norm.weight"):
            t.copy_(1.0 + 0.05 * torch.randn(t.shape, generator=g, device=DEV))
        elif "attn_sink" in name:
            t.copy_(0.5 * torch.randn(t.shape, generator=g, device=DEV))
        elif "bias" in name:
            t.copy_(0.01 * torch.randn(t.shape, generator=g, device=DEV))
        else:
            fan_in = t.shape[-1] if t.ndim > 1 else 1
            t.copy_(torch.randn(t.shape, generator=g, device=DEV) / max(fan_in, 1) ** 0.5)
    else:
        raise RuntimeError(f"unhandled {name} {t.dtype}")


def build_attention(vc, seed: int = 1, verbose: bool = False):
    import vllm.models.deepseek_v4_1.nvidia.model as m
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.deepseek_v4_1.tilelang import prepare
    from vllm.utils.torch_utils import set_default_torch_dtype

    cls = m._select_dsv4_attn_cls(vc)
    hf = vc.model_config.hf_config
    mods = {}
    with set_current_vllm_config(vc), set_default_torch_dtype(torch.bfloat16), torch.device(DEV):
        topk = torch.empty(
            vc.scheduler_config.max_num_batched_tokens, hf.index_topk, dtype=torch.int32
        )
        for layer in LAYERS:
            a = cls(vc, prefix=f"model.layers.{layer}.attn", topk_indices_buffer=topk)
            g = torch.Generator(device=DEV)
            g.manual_seed(seed * 1000 + layer)
            for name, p in a.named_parameters():
                if verbose:
                    print("param", layer, name, tuple(p.shape), p.dtype, flush=True)
                _fill(name, p, g)
            process_weights_after_loading(a, vc.model_config, DEV)
            # WO's input is the step's result here (see the module docstring).
            a._o_proj = lambda o, sp_rows=None: o
            mods[layer] = a
    # As the worker does after loading: outside the model's default dtype (binding
    # allocates FP32 placeholders with the default dtype).
    with set_current_vllm_config(vc):
        prepare.declare_modules(*mods.values())
    return mods


def bind_caches(vc, mods, pages_per_group: int):
    """Allocate and bind every owned cache: one cache group per spec (the two SWA caches
    share the SWA group, as in serving)."""
    groups = {}
    for a in mods.values():
        for c in a._owned_caches():
            spec = c.get_kv_cache_spec(vc)
            key = (type(spec).__name__, spec.block_size, spec.page_size_bytes,
                   getattr(spec, "tokens_per_state", None), getattr(spec, "head_size", None))
            group = groups.setdefault(key, SimpleNamespace(spec=spec, layers=[], caches={}))
            group.layers.append(c.prefix)
            t = torch.zeros((pages_per_group, spec.page_size_bytes), dtype=torch.uint8, device=DEV)
            c.bind_kv_cache(t)
            group.caches[c.prefix] = t
    return list(groups.values())


def build_metadata(vc, groups, query_lens, context_lens):
    from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadataBuilder
    from vllm.v1.worker.gpu.block_table import BlockTables, slot_mapping_mode

    specs = [g.spec for g in groups]
    n = len(query_lens)
    seq = [c + q for c, q in zip(context_lens, query_lens)]
    enabled, circular = zip(*(slot_mapping_mode(s) for s in specs))
    blocks = []
    for s, circ in zip(specs, circular):
        need = [1 if circ else -(-L // s.block_size) for L in seq]
        blocks.append(need)
    max_blocks = [max(b) for b in blocks]
    tables = BlockTables(
        block_sizes=[s.block_size for s in specs],
        max_num_reqs=vc.scheduler_config.max_num_seqs,
        max_num_batched_tokens=vc.scheduler_config.max_num_batched_tokens,
        max_num_blocks_per_group=[max(m, 1) for m in max_blocks],
        device=DEV,
        kernel_block_sizes=[s.block_size for s in specs],
        slot_mapping_enabled=list(enabled),
        slot_mapping_circular=list(circular),
    )
    nxt = [1] * len(specs)  # page 0 stays unused
    for i in range(n):
        ids = []
        for gi in range(len(specs)):
            k = blocks[gi][i]
            ids.append(list(range(nxt[gi], nxt[gi] + k)))
            nxt[gi] += k
        tables.append_block_ids(req_index=i, new_block_ids=tuple(ids), overwrite=True)
    tables.apply_staged_writes()
    for gi, g in enumerate(groups):
        for t in g.caches.values():
            assert nxt[gi] <= t.shape[0], (nxt[gi], t.shape)
    idx_mapping = torch.arange(n, dtype=torch.int32, device=DEV)
    starts = [0]
    for q in query_lens:
        starts.append(starts[-1] + q)
    T = starts[-1]
    query_start_loc = torch.tensor(starts, dtype=torch.int32, device=DEV)
    positions = torch.cat(
        [torch.arange(c, c + q, dtype=torch.int64, device=DEV) for c, q in zip(context_lens, query_lens)]
    )
    slot_mappings = tables.compute_slot_mappings(idx_mapping, query_start_loc, positions, num_tokens_padded=T)
    block_tables = tables.gather_block_tables(idx_mapping, n)
    metadata = {}
    for gi, g in enumerate(groups):
        cm = SimpleNamespace(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens_cpu=torch.tensor(seq, dtype=torch.int32),
            is_prefilling=None,
            seq_lens=torch.tensor(seq, dtype=torch.int32, device=DEV),
            block_table_tensor=block_tables[gi],
            slot_mapping=slot_mappings[gi],
            num_reqs=n,
            num_actual_tokens=T,
            max_query_len=max(query_lens),
            max_seq_len=max(seq),
        )
        builder = DeepseekV41B12xMetadataBuilder(g.spec, list(g.layers), vc, DEV)
        md = builder.build(0, cm)
        for name in g.layers:
            metadata[name] = md
    torch.cuda.synchronize()
    return metadata, positions


class FakeTP:
    """The TP all-gather across ``world`` ranks run one after another on one GPU."""

    def __init__(self, world):
        self.world = world
        self.inputs = {}
        self.changed = False
        self.rank = 0
        self.k = 0

    def begin(self, rank, real):
        self.rank, self.real, self.k = rank, real, 0

    def all_gather(self, x, dim=0):
        assert dim == 0
        x = x.contiguous()
        k = self.k
        self.k += 1
        slot = self.inputs.setdefault(k, {})
        old = slot.get(self.rank)
        # Rows past the rank's real rows are SP padding (left unwritten by design):
        # the fixed point is judged on the real rows.
        real = self.real
        if (
            old is None
            or old.shape != x.shape
            or not torch.equal(old[:real].view(torch.uint8), x[:real].view(torch.uint8))
        ):
            self.changed = True
        slot[self.rank] = x.clone()
        parts = [x if r == self.rank else slot.get(r, torch.zeros_like(x)) for r in range(self.world)]
        return torch.cat(parts, dim=0)


def run_step(vc, mods, groups, world, query_lens, context_lens, seed, max_passes=6):
    from vllm.forward_context import set_forward_context
    from vllm.models.deepseek_v4_1 import sp_prefill

    hf = vc.model_config.hf_config
    metadata, positions = build_metadata(vc, groups, query_lens, context_lens)
    T = positions.shape[0]
    swa = metadata[mods[TARGET].swa_cache_layer.prefix]
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    h20 = (torch.randn((T, hf.hidden_size), generator=g, device=DEV) * 0.5).bfloat16()
    h24 = (torch.randn((T, hf.hidden_size), generator=g, device=DEV) * 0.5).bfloat16()
    for gr in groups:
        for t in gr.caches.values():
            t.zero_()
    result = SimpleNamespace(T=T, world=world, is_decode=bool(swa.is_decode),
                             num_decode_tokens=int(swa.num_decode_tokens), ranks=[], error=None,
                             passes=0, converged=False)
    with set_forward_context(metadata, vc, num_tokens=T):
        out20 = mods[20].forward(positions, h20)
        torch.cuda.synchronize()
        result.out20 = out20.clone()
        swa24 = mods[TARGET].swa_cache_layer.kv_cache
        swa_before = swa24.clone()
        fake = FakeTP(world)
        orig = sp_prefill.tensor_model_parallel_all_gather
        sp_prefill.tensor_model_parallel_all_gather = fake.all_gather
        try:
            for p in range(max_passes):
                fake.changed = False
                ranks = []
                for r in range(world):
                    rows = sp_prefill.SPRows(T, world, r)
                    L = rows.local_rows
                    local = torch.zeros((L, hf.hidden_size), dtype=torch.bfloat16, device=DEV)
                    real = rows.real_rows
                    if real:
                        local[:real].copy_(h24[rows.start : rows.start + real])
                    swa24.copy_(swa_before)
                    fake.begin(r, real)
                    with sp_prefill.activate(rows):
                        out = mods[TARGET].forward(positions, local)
                    torch.cuda.synchronize()
                    ranks.append(SimpleNamespace(
                        rank=r, local_rows=L, real_rows=real, start=rows.start,
                        out=out.clone(), topk=mods[TARGET].topk_indices_buffer[:T].clone(),
                        swa=swa24.clone(), gathers=fake.k,
                    ))
                result.passes = p + 1
                result.ranks = ranks
                if not fake.changed:
                    result.converged = True
                    break
        except Exception as e:  # noqa: BLE001
            result.error = f"{type(e).__name__}: {e}"
            result.traceback = traceback.format_exc()
            result.error_rank = fake.rank
        finally:
            sp_prefill.tensor_model_parallel_all_gather = orig
    return result


def decode_tail_lens(num_tokens, num_reqs=16):
    """``_dummy_run``'s request lengths for ``num_tokens`` (min(T, max_num_reqs) requests)."""
    n = min(num_tokens, num_reqs)
    return [num_tokens // n + (i >= n - num_tokens % n) for i in range(n)]


def scenarios():
    """(name, query lens, context lens). The first is TP4's failing warm-up step."""
    tail = decode_tail_lens(214)
    return [
        ("decode-tail-214", tail, [0] * len(tail)),
        ("decode-ctx-216", [13] * 8 + [14] * 8, [200 + 17 * i for i in range(16)]),
        ("prefill-4096", [4096], [0]),
        ("prefill-4097", [4097], [0]),
        ("mixed-90+4006", [6] * 15 + [4006], [300 + 11 * i for i in range(15)] + [0]),
    ]


def digest(t: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    t0 = time.time()
    vc = make_config(args.world)
    init_dist(vc)
    mods = build_attention(vc)
    groups = bind_caches(vc, mods, pages_per_group=96)
    print(f"built in {time.time() - t0:.1f}s; heads/rank {mods[TARGET].n_local_heads}, "
          f"groups {[ (type(g.spec).__name__, g.spec.block_size, g.spec.page_size_bytes, g.layers) for g in groups]}",
          flush=True)
    saved = {}
    summary = []
    for name, q, c in scenarios():
        if args.only and name not in args.only.split(","):
            continue
        res = run_step(vc, mods, groups, args.world, q, c, seed=zlib.crc32(name.encode()) % 100003)
        entry = dict(scenario=name, T=res.T, world=res.world, is_decode=res.is_decode,
                     num_decode_tokens=res.num_decode_tokens, passes=res.passes,
                     converged=res.converged, error=res.error,
                     out20=digest(res.out20),
                     ranks=[dict(rank=r.rank, local_rows=r.local_rows, real_rows=r.real_rows,
                                 gathers=r.gathers, out=digest(r.out), topk=digest(r.topk),
                                 swa=digest(r.swa)) for r in res.ranks])
        if res.error:
            entry["error_rank"] = res.error_rank
            print(res.traceback, flush=True)
        else:
            outs = {digest(r.out) for r in res.ranks}
            entry["ranks_agree_on_output"] = len(outs) == 1
            entry["ranks_agree_on_topk"] = len({digest(r.topk) for r in res.ranks}) == 1
        summary.append(entry)
        print(json.dumps(entry), flush=True)
        # Rank 0's tensors in full (every rank's digest is in the JSON).
        saved[name] = dict(
            out20=res.out20.cpu(),
            rank0=(
                dict(out=res.ranks[0].out.cpu(), topk=res.ranks[0].topk.cpu(),
                     swa=res.ranks[0].swa.cpu())
                if res.ranks else None
            ),
            error=res.error,
        )
    torch.save(saved, f"{args.out}.pt")
    with open(f"{args.out}.json", "w") as f:
        json.dump(dict(arm=args.arm, world=args.world, scenarios=summary,
                       heads_per_rank=mods[TARGET].n_local_heads,
                       seconds=round(time.time() - t0, 1)), f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
