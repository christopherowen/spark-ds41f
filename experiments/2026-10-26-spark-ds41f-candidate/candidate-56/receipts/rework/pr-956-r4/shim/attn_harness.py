"""DeepSeek V4.1 attention, compressor layers 2 (ratio 2) and 20 (ratio 1), from the
serving code on one GPU at a TP rank's shapes, with the sequence-parallel all-gather
simulated across ranks. Adapted from #1009 round 1's harness
(candidate-50/receipts/rework/pr-1009-r1/shim/attn_harness.py).

The modules are vLLM's own ``DeepseekV41TileLangAttention`` (the class
``_select_dsv4_attn_cls`` picks for ``--attention-backend TILELANG``), built under a
``VllmConfig`` that ``EngineArgs`` makes from the served checkpoint's config.json with
the recipe's flags (TileLang attention/linear/MoE backends, block size 256,
max_model_len 1048576, max_num_seqs 16, max_num_batched_tokens 16384) and the TP4
recipe's speculative geometry (DSpark, 5 drafts, copy drafts verify 16). Single
process, TP group of 1: a TP-W rank's attention shapes come from the per-rank head and
output-group counts (TP3: 24 heads, 3 groups; TP4: 16, 2). The indexer is replicated.

Round 3 without #957 part 1 (candidate-54 as the base): a TP-W rank's compressor is
column-sharded and 12-bit packed, and its projection without SP all-gathers its column
shards. One process cannot hold W ranks' shards in one module, so the compressor is
built in one of two modes (``--compressor``):
- ``tp1`` (bit-exactness, this script's default): built as TP1 builds it (the whole
  packed weight), with ``_sp_enabled`` set by the arm's own rule at world size W (a
  probe compressor is constructed with ``get_tensor_model_parallel_world_size() == W``
  and dropped). The TP1 projection gives the sharded projection plus the column
  gather's bits (tests/models/test_deepseek_v4_1_compressor.py checks both against
  each other at TP2..TP8, and the rebase's test_sp_ratio2_projection_matches_full_rows
  checks the SP projection against both at 129..16,384 rows), so every rank's result
  here is a TP-W rank's.
- ``rank`` (timing, bench_layers.py): built as TP-W rank 0 builds it (the rank's packed
  column shard); the compressor's own all-gathers (bind()'s weight gather and the column
  gather without SP) return W copies of the rank's tensor: shapes and bytes as on a
  TP-W rank, values not.

Decode check (``decode-*`` scenarios): without SP, the target layer runs one decode
step at engine-grid shapes (1, 4, 8 and 16 requests of 6 rows: the bonus token and 5
drafts); recorded are the output, top-k and cache digests and the CUDA kernels it
launches, in order (torch.profiler), which must equal the base's.

Layer 2 is a KV, index and candidate source with a ratio-2 compressor (state ring);
layer 20 is the ratio-1 KV, index and candidate source (here run as an ordinary
SP layer, as it runs in an SP step without CED compaction). The rebased #956 changes
both under SP: the local front (``_sp_front``) and the compressor's local-row
projection. Random weights in checkpoint dtypes go through
``process_weights_after_loading`` and ``prepare.declare_modules``; the KV caches are
allocated at the specs' page sizes and bound; each step's metadata is built by the real
``DeepseekV41B12xMetadataBuilder`` from ``BlockTables`` slot mappings.

A step: the target layer runs once per TP rank r under
``sp_prefill.activate(SPRows(T, W, r))`` with that rank's L local hidden rows (zero
padding past T), every cache restored to the step's start state before each rank. The
TP all-gather (``sp_prefill.tensor_model_parallel_all_gather``; with a TP group of 1
the grouped PyNCCL path is off, so every SP gather goes through it) returns the
concatenation of every rank's input to the same call. The ranks are re-run until every
recorded input is unchanged (a fixed point: the real TP result). WO (``_o_proj``) is
the identity, so the step's result is WO's input. Recorded per rank: the output, the
top-k indices, every cache the target layer owns (SWA records, main KV, index keys,
the ratio-2 state ring), and the shapes of the all-gathers it made. A chunked
scenario carries the caches of its first step (all ranks must agree) into the next.
"""

import hashlib
import os
import sys
import time
import traceback
import zlib
from types import SimpleNamespace

import torch

MODEL = "/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
DEV = torch.device("cuda")
TARGETS = (2, 20)


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
            distributed_init_method="tcp://127.0.0.1:29541",
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


class rank_world_compressor:
    """Build the compressor as described in the module docstring (``mode`` tp1/rank)."""

    def __init__(self, world, mode):
        self.world, self.mode = world, mode

    def __enter__(self):
        import vllm.models.deepseek_v4_1.compressor as cm

        self.cm = cm
        self.orig = cm.DeepseekCompressor.__init__
        world, orig, mode = self.world, self.orig, self.mode

        def at_world(fn):
            saved = cm.linear_layers.get_tensor_model_parallel_world_size
            cm.linear_layers.get_tensor_model_parallel_world_size = lambda: world
            try:
                return fn()
            finally:
                cm.linear_layers.get_tensor_model_parallel_world_size = saved

        def init(comp, vllm_config, *args, **kwargs):
            if mode == "rank":
                at_world(lambda: orig(comp, vllm_config, *args, **kwargs))
                return
            orig(comp, vllm_config, *args, **kwargs)
            probe = cm.DeepseekCompressor.__new__(cm.DeepseekCompressor)
            probe_kwargs = dict(kwargs, prefix=f"{kwargs.get('prefix', '')}.probe")
            at_world(lambda: orig(probe, vllm_config, *args, **probe_kwargs))
            comp._sp_enabled = bool(probe._sp_enabled)
            context = vllm_config.compilation_config.static_forward_context
            context.pop(f"{probe_kwargs['prefix']}.state_cache", None)
            del probe

        cm.DeepseekCompressor.__init__ = init
        return self

    def __exit__(self, *exc):
        self.cm.DeepseekCompressor.__init__ = self.orig


def copies_all_gather(world):
    """The compressor's all-gather in ``rank`` mode: W copies of this rank's tensor."""

    def gather(tensor, dim=-1):
        assert dim == 0
        return torch.cat([tensor] * world, dim=0)

    return gather


def build_attention(vc, world, seed: int = 1, mode: str = "tp1"):
    import vllm.models.deepseek_v4_1.nvidia.model as m
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.deepseek_v4_1.tilelang import prepare
    from vllm.utils.torch_utils import set_default_torch_dtype

    import vllm.models.deepseek_v4_1.compressor as cm

    if mode == "rank":
        cm.tensor_model_parallel_all_gather = copies_all_gather(world)
    cls = m._select_dsv4_attn_cls(vc)
    hf = vc.model_config.hf_config
    mods = {}
    with set_current_vllm_config(vc), set_default_torch_dtype(torch.bfloat16), torch.device(DEV):
        topk = torch.empty(
            vc.scheduler_config.max_num_batched_tokens, hf.index_topk, dtype=torch.int32
        )
        for layer in TARGETS:
            with rank_world_compressor(world, mode):
                a = cls(vc, prefix=f"model.layers.{layer}.attn", topk_indices_buffer=topk)
            g = torch.Generator(device=DEV)
            g.manual_seed(seed * 1000 + layer)
            for name, p in a.named_parameters():
                _fill(name, p, g)
            process_weights_after_loading(a, vc.model_config, DEV)
            # WO's input is the step's result here (see the module docstring).
            a._o_proj = lambda o, sp_rows=None: o
            mods[layer] = a
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    with set_current_vllm_config(vc):
        prepare.declare_modules(*mods.values())
    torch.cuda.synchronize()
    info = {"_memory": dict(mode=mode, allocated_before_declare=before,
                            allocated_after_declare=torch.cuda.memory_allocated())}
    for layer, a in mods.items():
        c = a.compressor
        info[layer] = dict(
            compress_ratio=int(c.compress_ratio),
            sp_enabled=bool(c._sp_enabled),
            sp_kernel_bound=c._sp_kernel is not None,
            wkv_tp_size=int(c.fused_wkv_wgate.tp_size),
            packed=hasattr(c.fused_wkv_wgate, "packed_bf16_exponent_base"),
            sp_weight_is_layer_weight=(
                c._sp_weight is not None
                and c._sp_weight.data_ptr() == c.fused_wkv_wgate.weight.data_ptr()
            ),
            sp_weight_bytes=(
                0 if c._sp_weight is None
                else c._sp_weight.numel() * c._sp_weight.element_size()
            ),
            sp_weight_shape=None if c._sp_weight is None else list(c._sp_weight.shape),
            heads=int(a.n_local_heads),
            indexer=a.indexer is not None,
            caches=[x.prefix for x in a._owned_caches()],
        )
    return mods, info


def bind_caches(vc, mods, pages_per_group: int):
    """Allocate and bind every owned cache: one cache group per spec."""
    groups = {}
    for a in mods.values():
        for c in a._owned_caches():
            spec = c.get_kv_cache_spec(vc)
            key = (type(spec).__name__, spec.block_size, spec.page_size_bytes,
                   getattr(spec, "tokens_per_state", None), getattr(spec, "head_size", None),
                   getattr(spec, "compress_ratio", None))
            group = groups.setdefault(key, SimpleNamespace(spec=spec, layers=[], caches={}))
            group.layers.append(c.prefix)
            t = torch.zeros((pages_per_group, spec.page_size_bytes), dtype=torch.uint8, device=DEV)
            # The worker views each group's bytes in its spec's dtype (the ratio-2
            # state ring is FP32); snapshots and digests use the bytes.
            dtype = getattr(spec, "dtype", torch.uint8)
            c.bind_kv_cache(t if dtype in (torch.uint8, None) else t.view(dtype))
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
        self.shapes = []

    def begin(self, rank, real):
        self.rank, self.real, self.k = rank, real, 0
        self.shapes = []

    def all_gather(self, x, dim=0):
        assert dim == 0
        x = x.contiguous()
        k = self.k
        self.k += 1
        self.shapes.append((tuple(x.shape), str(x.dtype).removeprefix("torch.")))
        slot = self.inputs.setdefault(k, {})
        old = slot.get(self.rank)
        # Rows past the rank's real rows are SP padding: the fixed point is judged on
        # the real rows.
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


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def target_caches(mod):
    return {c.prefix: c.kv_cache for c in mod._owned_caches()}


def run_layer_step(vc, mod, groups, world, query_lens, context_lens, seed, start_state,
                   max_passes=6):
    """One step of ``mod`` under SP on every rank, from ``start_state`` (a dict of
    cache prefix -> uint8 tensor for every bound cache)."""
    from vllm.forward_context import set_forward_context
    from vllm.models.deepseek_v4_1 import sp_prefill

    hf = vc.model_config.hf_config
    metadata, positions = build_metadata(vc, groups, query_lens, context_lens)
    T = positions.shape[0]
    swa = metadata[mod.swa_cache_layer.prefix]
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    h = (torch.randn((T, hf.hidden_size), generator=g, device=DEV) * 0.5).bfloat16()
    all_caches = {name: t for gr in groups for name, t in gr.caches.items()}
    mine = target_caches(mod)
    res = SimpleNamespace(T=T, world=world, is_decode=bool(swa.is_decode),
                          num_decode_tokens=int(swa.num_decode_tokens), ranks=[], error=None,
                          passes=0, converged=False)
    with set_forward_context(metadata, vc, num_tokens=T):
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
                        local[:real].copy_(h[rows.start : rows.start + real])
                    for name, t in all_caches.items():
                        t.copy_(start_state[name])
                    fake.begin(r, real)
                    with sp_prefill.activate(rows):
                        out = mod.forward(positions, local)
                    torch.cuda.synchronize()
                    ranks.append(SimpleNamespace(
                        rank=r, local_rows=L, real_rows=real, start=rows.start,
                        out=digest(out), out_shape=tuple(out.shape),
                        out_finite=bool(torch.isfinite(out.float()).all()),
                        topk=digest(mod.topk_indices_buffer[:T]),
                        caches={name: digest(t) for name, t in mine.items()},
                        state={name: t.clone() for name, t in all_caches.items()} if r == 0 else None,
                        gathers=list(fake.shapes),
                    ))
                res.passes = p + 1
                res.ranks = ranks
                if not fake.changed:
                    res.converged = True
                    break
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {e}"
            res.traceback = traceback.format_exc()
            res.error_rank = fake.rank
        finally:
            sp_prefill.tensor_model_parallel_all_gather = orig
    return res


def run_layer_grid_decode(vc, mod, groups, query_lens, context_lens, seed):
    """One decode step of ``mod`` without SP (an engine-grid shape): a warm-up forward,
    then a profiled one; the output, top-k and owned-cache digests after it, and the
    CUDA kernels it launched (names and counts; per stream in launch order)."""
    from collections import Counter

    from torch.profiler import ProfilerActivity, profile

    from vllm.forward_context import set_forward_context

    hf = vc.model_config.hf_config
    metadata, positions = build_metadata(vc, groups, query_lens, context_lens)
    T = positions.shape[0]
    swa = metadata[mod.swa_cache_layer.prefix]
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    h = (torch.randn((T, hf.hidden_size), generator=g, device=DEV) * 0.5).bfloat16()
    for gr in groups:
        for t in gr.caches.values():
            t.zero_()
    with set_forward_context(metadata, vc, num_tokens=T):
        mod.forward(positions, h)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            out = mod.forward(positions, h)
            torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type.name == "CUDA"]
    kernels.sort(key=lambda e: e.time_range.start)
    streams = {}
    for e in kernels:
        streams.setdefault(str(getattr(e, "device_resource_id", 0)), []).append(e.name)
    names = Counter(e.name for e in kernels)
    return dict(
        T=T, is_decode=bool(swa.is_decode), num_decode_tokens=int(swa.num_decode_tokens),
        out=digest(out), out_finite=bool(torch.isfinite(out.float()).all()),
        topk=digest(mod.topk_indices_buffer[:T]),
        caches={c.prefix: digest(c.kv_cache) for c in mod._owned_caches()},
        kernels=len(kernels),
        kernel_names=dict(sorted(names.items())),
        kernel_streams=len(streams),
        kernel_sequence_digest={k: hashlib.sha256("\n".join(v).encode()).hexdigest()[:16]
                                for k, v in sorted(streams.items())},
        kernel_names_digest=hashlib.sha256(
            "\n".join(f"{k} {v}" for k, v in sorted(names.items())).encode()
        ).hexdigest()[:16],
    )


def grid_scenarios():
    """Engine-grid decode shapes without SP: 1, 4, 8 and 16 requests of 6 rows."""
    return [
        (f"grid-decode-c{n}", [6] * n, [1500 + 97 * i for i in range(n)])
        for n in (1, 4, 8, 16)
    ]


def decode_tail_lens(num_tokens, num_reqs=16):
    n = min(num_tokens, num_reqs)
    return [num_tokens // n + (i >= n - num_tokens % n) for i in range(n)]


def scenarios():
    """(name, [steps of (query lens, context lens)]). A multi-step scenario carries the
    caches from one step into the next (a chunked prefill)."""
    tail = decode_tail_lens(214)
    return [
        ("prefill-4096", [([4096], [0])]),
        ("prefill-4097", [([4097], [0])]),
        ("prefill-16384", [([16384], [0])]),
        ("chunked-4097+4096+2051", [([4097], [0]), ([4096], [4097]), ([2051], [8193])]),
        ("mixed-90+4006", [([6] * 15 + [4006], [300 + 11 * i for i in range(15)] + [0])]),
        ("decode-tail-214", [(tail, [0] * len(tail))]),
        ("decode-ctx-216", [([13] * 8 + [14] * 8, [200 + 17 * i for i in range(16)])]),
    ]


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--compressor", default="tp1", choices=("tp1", "rank"))
    args = ap.parse_args()
    t0 = time.time()
    vc = make_config(args.world)
    init_dist(vc)
    mods, info = build_attention(vc, args.world, mode=args.compressor)
    groups = bind_caches(vc, mods, pages_per_group=320)
    print(f"built in {time.time() - t0:.1f}s; layers {json.dumps(info)}; groups "
          f"{[(type(g.spec).__name__, g.spec.block_size, g.spec.page_size_bytes, g.layers) for g in groups]}",
          flush=True)
    summary = []
    for name, steps in scenarios():
        if args.only and name not in args.only.split(","):
            continue
        for layer, mod in mods.items():
            if not isinstance(layer, int):
                continue
            state = {n: torch.zeros_like(t) for gr in groups for n, t in gr.caches.items()}
            for si, (q, c) in enumerate(steps):
                seed = zlib.crc32(f"{name}/{si}/{layer}".encode()) % 100003
                res = run_layer_step(vc, mod, groups, args.world, q, c, seed, state)
                entry = dict(scenario=name, step=si, layer=layer, T=res.T, world=res.world,
                             is_decode=res.is_decode, num_decode_tokens=res.num_decode_tokens,
                             passes=res.passes, converged=res.converged, error=res.error,
                             ranks=[dict(rank=r.rank, local_rows=r.local_rows,
                                         real_rows=r.real_rows, out=r.out,
                                         out_shape=r.out_shape, out_finite=r.out_finite,
                                         topk=r.topk, caches=r.caches,
                                         gathers=r.gathers) for r in res.ranks])
                if res.error:
                    entry["error_rank"] = res.error_rank
                    print(res.traceback, flush=True)
                    summary.append(entry)
                    print(json.dumps(entry), flush=True)
                    break
                entry["ranks_agree_on_output"] = len({r.out for r in res.ranks}) == 1
                entry["ranks_agree_on_topk"] = len({r.topk for r in res.ranks}) == 1
                entry["ranks_agree_on_caches"] = len(
                    {json.dumps(r.caches, sort_keys=True) for r in res.ranks}) == 1
                summary.append(entry)
                print(json.dumps(entry), flush=True)
                # The next step starts from rank 0's caches (all ranks agree on them).
                state = res.ranks[0].state
    grid = []
    if not args.only or "grid" in args.only:
        for name, q, c in grid_scenarios():
            for layer, mod in mods.items():
                seed = zlib.crc32(f"{name}/{layer}".encode()) % 100003
                entry = dict(scenario=name, layer=layer)
                try:
                    entry.update(run_layer_grid_decode(vc, mod, groups, q, c, seed))
                except Exception as e:  # noqa: BLE001
                    entry["error"] = f"{type(e).__name__}: {e}"
                    print(traceback.format_exc(), flush=True)
                grid.append(entry)
                print(json.dumps({k: v for k, v in entry.items() if k != "kernel_names"}),
                      flush=True)
    with open(f"{args.out}.json", "w") as f:
        json.dump(dict(arm=args.arm, world=args.world, compressor_mode=args.compressor,
                       layers=info, scenarios=summary, grid=grid,
                       seconds=round(time.time() - t0, 1)), f, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
