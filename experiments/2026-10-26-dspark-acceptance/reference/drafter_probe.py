"""DeepSeek's reference DSpark drafter, teacher-forced through given texts on CPU.

usage (inside the serving image, repository at /repo, snapshot read-only):
  python3 -I drafter_probe.py SNAPSHOT CASES.json OUT.json [--block 5] [--deadline ISO]
  python3 -I drafter_probe.py --tiny REFDIR OUT.json      (self-test, no checkpoint)

CASES.json: {"cases": [{"id": ..., "prompt_ids": [...], "output_ids": [...]}, ...]}, the
engine's own prompt and temperature-0 reply of each request, or ../per_request_acceptance.py's
JSON lines, which also carry the engine's own counters for each request. A case may instead
carry "messages" (and "thinking") plus "output_text"; the probe then renders the prompt with
the checkpoint's encoding and tokenizes the reply.

For each case, one prefill at position 0 over prompt + reply gives the target's hidden states
at every position, exactly as incremental decoding computes them (routes_probe.py's method).
The drafter then runs at every anchor of the reply at once: the anchor is the token at
position a, its context is the projected target states of positions a-128 .. a-1 (the
reference's 128-slot window ring), and the block is the anchor plus block-1 noise tokens at
positions a .. a+block-1. Its arithmetic is DSparkAttention's and DSparkBlock's own, with the
window gathered per anchor instead of read from the ring; ``--tiny`` checks the batched form
against the reference's sequential forward_spec loop.

Recorded per anchor: the drafts, the second candidate at every position, a second chain
branching at position 1 (its Markov head fed the second candidate), the confidence logits, and
the target's greedy token at every reply position. The summary chains anchors as a
verify-all engine would (next anchor = a + accepted + 1) and gives tokens a step and
survival by position; drafter_summary.py compares them with the engine's counters.

The loader is routes_probe.py's (O_DIRECT reads, only the routed experts a layer uses, one
layer resident at a time) with json-count's CPU kernels (shim/). Nothing here is DeepSeek's
code: the reference modules are imported from the snapshot unchanged.
"""

import argparse
import json
import math
import mmap
import os
import sys
import time
import types
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIM = HERE.parents[1] / "2026-10-18-json-count" / "reference" / "shim"

parser = argparse.ArgumentParser()
parser.add_argument("snapshot", help="checkpoint snapshot (with --tiny: a directory holding inference/)")
parser.add_argument("cases", nargs="?", help="cases JSON (not with --tiny)")
parser.add_argument("out")
parser.add_argument("--block", type=int, default=5, help="drafter block (the checkpoint's is 5)")
parser.add_argument("--deadline", help="UTC time (ISO 8601) after which no further case starts")
parser.add_argument("--anchor-chunk", type=int, default=256, help="anchors a drafter pass holds")
parser.add_argument("--max-cases", type=int, default=0)
parser.add_argument("--tiny", action="store_true", help="self-test on a small random model")
ARGV = parser.parse_args()
if ARGV.tiny:
    ARGV.out, ARGV.cases = ARGV.cases or ARGV.out, None

SNAP = os.path.abspath(ARGV.snapshot)
sys.path.insert(0, os.path.join(SNAP, "encoding"))
sys.path.insert(0, os.path.join(SNAP, "inference"))
sys.path.insert(0, str(SHIM))  # first, so model.py's `from kernel import` gets the shim

if ARGV.tiny:
    # image_processor imports PIL at module level; the tiny model has no vision path.
    pil = types.ModuleType("PIL")
    pil.Image = pil.ImageOps = types.ModuleType("PIL.Image")
    sys.modules.setdefault("PIL", pil)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

import kernel as shim  # noqa: E402
import model as M  # noqa: E402
from engram import EngramLayout, NgramHashState  # noqa: E402

assert M.act_quant is shim.act_quant, "model.py did not pick up the CPU kernels"

ALIGN = 4096
VOCAB_CHUNK = 8192
DT = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I32": torch.int32,
    "I64": torch.int64,
}


class Shards:
    """Reads tensors and tensor rows from the snapshot's safetensors files with O_DIRECT."""

    def __init__(self, snap):
        self.snap = snap
        self.where = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
        self.headers, self.fds = {}, {}
        self.bytes_read = 0

    def meta(self, name):
        fname = self.where[name]
        if fname not in self.headers:
            path = os.path.realpath(os.path.join(self.snap, fname))
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(n))
            self.headers[fname] = (8 + n, header)
            self.fds[fname] = os.open(path, os.O_RDONLY | os.O_DIRECT)
        start, header = self.headers[fname]
        m = header[name]
        b, e = m["data_offsets"]
        return fname, start + b, e - b, DT[m["dtype"]], m["shape"]

    def _read(self, fname, off, length):
        a0 = off // ALIGN * ALIGN
        a1 = (off + length + ALIGN - 1) // ALIGN * ALIGN
        buf = mmap.mmap(-1, a1 - a0)
        done = 0
        while done < a1 - a0:
            chunk = min(a1 - a0 - done, 1 << 30)
            got = os.preadv(self.fds[fname], [memoryview(buf)[done : done + chunk]], a0 + done)
            if got <= 0:
                break
            done += got
        assert done >= off + length - a0, (fname, off, length, done)
        view = torch.frombuffer(buf, dtype=torch.uint8, count=length, offset=off - a0)
        out = view.clone()
        del view
        buf.close()
        self.bytes_read += a1 - a0
        return out

    def get(self, name):
        fname, off, length, dt, shape = self.meta(name)
        return self._read(fname, off, length).view(dt).reshape(shape)

    def get_rows(self, name, rows):
        fname, off, length, dt, shape = self.meta(name)
        row_bytes = length // shape[0]
        out = torch.empty(len(rows), row_bytes, dtype=torch.uint8)
        for k, r in enumerate(rows):
            assert 0 <= r < shape[0]
            out[k] = self._read(fname, off + r * row_bytes, row_bytes)
        return out.view(dt).reshape(len(rows), *shape[1:])

    def get_row_range(self, name, start, n):
        """Rows start .. start+n of a 2-D tensor, in one read."""
        fname, off, length, dt, shape = self.meta(name)
        row_bytes = length // shape[0]
        return self._read(fname, off + start * row_bytes, n * row_bytes).view(dt).reshape(n, *shape[1:])


class TinyWeights:
    """Deterministic random tensors named like the checkpoint's, for --tiny. Every value is
    BF16-representable (as the checkpoint's BF16 tensors are) before any widening."""

    def __init__(self):
        self.bytes_read = 0
        self.cache = {}

    def tensor(self, name, shape, dtype):
        key = (name, tuple(shape), dtype)
        if key in self.cache:
            return self.cache[key]
        g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
        leaf = name.rsplit(".", 1)[-1]
        if dtype == torch.float4_e2m1fn_x2:
            t = torch.randint(0, 256, shape, generator=g, dtype=torch.uint8).view(torch.int8)
        elif dtype == torch.float8_e8m0fnu:
            t = torch.full(shape, 2.0**-9).to(torch.float8_e8m0fnu)
        elif dtype == torch.float8_e4m3fn:
            t = (torch.randn(shape, generator=g) * 16).clamp(-448, 448).to(torch.float8_e4m3fn)
        elif name.endswith("norm.weight"):
            t = torch.ones(shape).to(dtype)
        elif leaf in ("attn_sink", "bias", "bias_vl") or "_base" in leaf:
            t = torch.zeros(shape).to(dtype)
        elif "_scale" in leaf:
            t = torch.full(shape, 0.5).to(dtype)
        elif "_fn" in leaf:
            t = (torch.randn(shape, generator=g) * 0.01).bfloat16().to(dtype)
        else:
            t = (torch.randn(shape, generator=g) * 0.05).bfloat16().to(dtype)
        self.cache[key] = t
        return t


TINY = TinyWeights() if ARGV.tiny else None
shards = None if ARGV.tiny else Shards(SNAP)


def assign(p: nn.Parameter, t: torch.Tensor, name: str):
    if p.dtype == torch.float4_e2m1fn_x2:
        assert t.dtype == torch.int8, (name, t.dtype)
        t = t.view(torch.float4_e2m1fn_x2)
    elif t.dtype != p.dtype:
        widening = (t.dtype, p.dtype) in {(torch.bfloat16, torch.float32), (torch.float16, torch.float32)}
        assert widening, f"narrowing load {name}: {t.dtype}->{p.dtype}"
        t = t.to(p.dtype)
    assert tuple(t.shape) == tuple(p.shape), (name, tuple(t.shape), tuple(p.shape))
    p.data = t


def dequant_wo_a(prefix):
    """convert.py's treatment of wo_a: FP8 blocks times their scales, stored as BF16."""
    weight = shards.get(prefix + "weight")
    scale = shards.get(prefix + "scale")
    ob, ib = weight.size(0) // scale.size(0), weight.size(1) // scale.size(1)
    assert (ob, ib) in ((32, 32), (128, 128))
    w = weight.unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)).float() * scale[:, None, :, None].float()
    return w.flatten(2, 3).flatten(0, 1).bfloat16()


def load_module(mod: nn.Module, prefix: str, skip=()):
    for name, p in mod.named_parameters():
        if any(s in name for s in skip):
            continue
        full = prefix + name
        if TINY is not None:
            assign(p, TINY.tensor(full, p.shape, p.dtype).clone(), full)
            continue
        t = dequant_wo_a(full[: -len("weight")]) if full.endswith("attn.wo_a.weight") else shards.get(full)
        assign(p, t, full)


_expert_init = M.Expert.__init__


def _lazy_expert_init(self, dim, inter_dim, dtype=None, swiglu_limit=0.0):
    if dtype == torch.float4_e2m1fn_x2:
        with torch.device("meta"):
            _expert_init(self, dim, inter_dim, dtype, swiglu_limit)
    else:
        _expert_init(self, dim, inter_dim, dtype, swiglu_limit)


M.Expert.__init__ = _lazy_expert_init


class RowReadEngramEmbedding(nn.Module):
    """ParallelEngramEmbedding.forward at world size 1, reading only the looked-up rows."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = M.fp8_block_size
        self.prefix = None

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.flatten().tolist()
        rows = sorted(set(flat))
        w = shards.get_rows(self.prefix + "weight", rows)
        s = shards.get_rows(self.prefix + "scale", rows)
        pos = {r: k for k, r in enumerate(rows)}
        sel = torch.tensor([pos[r] for r in flat])
        values = w[sel].view(*indices.shape, self.dim)
        scales = s[sel].view(*indices.shape, -1)
        values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16)


if not ARGV.tiny:
    M.ParallelEngramEmbedding = RowReadEngramEmbedding


def expert_prefix(layer_id: int) -> str:
    if layer_id < ARGS.n_layers:
        return f"layers.{layer_id}.ffn.experts."
    return f"mtp.{layer_id - ARGS.n_layers}.ffn.experts."


EXPERTS = {}  # (layer_id, expert) -> Expert, kept for the current drafter stage only


def moe_forward(self, x, image_mask=None):
    """MoE.forward at world size 1, with each routed expert built and loaded when first used."""
    shape = x.size()
    x = x.view(-1, self.dim)
    weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
    y = torch.zeros_like(x, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
    for i in range(self.experts_start_idx, self.experts_end_idx):
        if counts[i] == 0:
            continue
        expert = EXPERTS.get((self.layer_id, i))
        if expert is None:
            expert = M.Expert.__new__(M.Expert)
            _expert_init(expert, self.dim, ARGS.moe_inter_dim, torch.float4_e2m1fn_x2, ARGS.swiglu_limit)
            load_module(expert, f"{expert_prefix(self.layer_id)}{i}.")
            if self.layer_id >= ARGS.n_layers:
                EXPERTS[(self.layer_id, i)] = expert
        idx, top = torch.where(indices == i)
        y[idx] += expert(x[idx], weights[idx, top, None])
        del expert
    y += self.shared_experts(x)
    return y.type_as(x).view(shape)


M.MoE.forward = moe_forward

torch.set_default_dtype(torch.bfloat16)
torch.set_num_threads(int(os.environ.get("REF_THREADS", "8")))
torch.manual_seed(33377335)

if ARGV.tiny:
    ARGS = M.ModelArgs(
        vocab_size=4096, n_layers=5, n_mtp_layers=3, compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0),
        dspark_block_size=ARGV.block, dspark_noise_token_id=4000, dspark_target_layer_ids=(3, 4),
        dspark_markov_rank=64, max_batch_size=1, max_seq_len=512, temperature=0.0,
    )
else:
    with open(os.path.join(SNAP, "inference", "config.json")) as f:
        ARGS = M.ModelArgs(**json.load(f))
    ARGS.max_batch_size = 1
    ARGS.max_seq_len = 8192
    ARGS.temperature = 0.0
    ARGS.dspark_block_size = ARGV.block
TARGETS = tuple(ARGS.dspark_target_layer_ids)
BLOCK = ARGS.dspark_block_size
NOISE = ARGS.dspark_noise_token_id


# --- weights the probe reads in pieces --------------------------------------------------------


def embed_rows(ids: list[int]) -> torch.Tensor:
    if TINY is not None:
        return TINY.tensor("embed.weight", (ARGS.vocab_size, ARGS.dim), torch.bfloat16)[torch.tensor(ids)]
    return shards.get_rows("embed.weight", ids)


def head_chunks() -> list[tuple[int, torch.Tensor]]:
    """(first row, BF16 rows) of the vocabulary head; callers widen a chunk to FP32 at use, as
    ParallelHead keeps the checkpoint's BF16 values in FP32."""
    if TINY is not None:
        w = TINY.tensor("head.weight", (ARGS.vocab_size, ARGS.dim), torch.float32).bfloat16()
        return [(r, w[r : r + VOCAB_CHUNK]) for r in range(0, ARGS.vocab_size, VOCAB_CHUNK)]
    _, _, _, dt, shape = shards.meta("head.weight")
    assert dt == torch.bfloat16 and shape == [ARGS.vocab_size, ARGS.dim], (dt, shape)
    return [(r, shards.get_row_range("head.weight", r, min(VOCAB_CHUNK, shape[0] - r)))
            for r in range(0, shape[0], VOCAB_CHUNK)]


# --- the target ---------------------------------------------------------------------------------

layout = EngramLayout.from_args(ARGS)
engram_hash = None
if layout is not None:
    from transformers import AutoTokenizer

    engram_hash = NgramHashState(ARGS, layout, AutoTokenizer.from_pretrained(SNAP))
final_norm = M.RMSNorm(ARGS.dim, ARGS.norm_eps)
load_module(final_norm, "norm.")


@torch.inference_mode()
def target_forward(ids: list[int]):
    """Transformer.forward at position 0, text only, one layer resident at a time. Returns the
    DSpark context states (the mean over hc copies of each target layer's input, joined) and
    the final-normed hidden states, for every position."""
    input_ids = torch.tensor([ids])
    hashes = engram_hash(input_ids, 0, None) if engram_hash is not None else None
    h = embed_rows(ids).view(1, len(ids), ARGS.dim)
    h = h.unsqueeze(2).repeat(1, 1, ARGS.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, ARGS.hc_mult)
    mains = []
    for i in range(ARGS.n_layers):
        t0 = time.time()
        block = M.Block(i, ARGS, layout)
        if block.engram is not None:
            block.engram.embed.prefix = f"layers.{i}.engram.embed."
        load_module(block, f"layers.{i}.", skip=("ffn.experts.", "engram.embed."))
        if block.engram is not None:
            h = block.engram(h, hashes[:, :, block.engram.layer_hash_index, :], None)
        if i in TARGETS:
            mains.append(h.mean(dim=2))  # the reference: the target layer's input, not its output
        h, pre_mix = block(h, 0, pre_mix, None)
        if i == ARGS.n_layers - 1:
            h = block.hc_pre(h, pre_mix)
        del block
        if not ARGV.tiny:
            print(f"  layer {i:2d} {time.time() - t0:6.1f}s read {shards.bytes_read / 2**30:8.2f} GiB", flush=True)
    return torch.cat(mains, dim=-1)[0], final_norm(h)[0]


# --- the drafter, at many anchors at once -------------------------------------------------------


def rotate(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """apply_rotary_emb with per-row frequencies: ``freqs`` broadcasts against x's complex pairs."""
    z = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs = freqs.conj()
    x.copy_(torch.view_as_real(z * freqs).flatten(-2))
    return x


def context_kv(attn, main_x: torch.Tensor) -> torch.Tensor:
    """DSparkAttention's main_kv for every position: wkv, kv_norm, RoPE at the position, FP8."""
    rd = attn.rope_head_dim
    kv = attn.kv_norm(attn.wkv(main_x.unsqueeze(0)))[0]
    rotate(kv[..., -rd:], attn.freqs_cis[: kv.size(0)])
    M.act_quant(kv, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    return kv


def dspark_attention_many(attn, x: torch.Tensor, anchors: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
    """DSparkAttention.forward for one block per anchor. x: [A, block, dim] (attention input);
    anchors: [A] positions of each block's first row; ctx: [L, head_dim] the context KV. Each
    block attends to its context window a-win .. a-1 and to every row of its own block."""
    A, B, _ = x.shape
    win, rd = attn.window_size, attn.rope_head_dim
    pos = anchors[:, None] + torch.arange(B)
    freqs = attn.freqs_cis[pos]  # [A, B, rd/2]
    qr = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
    rotate(q[..., -rd:], freqs.unsqueeze(2))
    kv = attn.kv_norm(attn.wkv(x))
    rotate(kv[..., -rd:], freqs)
    M.act_quant(kv, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    # The ring as the reference holds it: slot s keeps the context position p = s (mod win) in
    # a-win .. a-1. Its index list is the filled slots in slot order, then the block, as
    # get_dspark_topk_idxs builds it; the -1 padding that makes the rows one width goes last,
    # where sparse_attn's online softmax ignores it, so every 64-index block sums the same keys.
    slot = torch.arange(win)
    wpos = anchors[:, None] - 1 - (anchors[:, None] - 1 - slot) % win
    window = ctx[wpos.clamp_min(0)]
    kvs = torch.cat([window, kv], dim=1)  # [A, win + B, head_dim]
    filled = torch.minimum(anchors, torch.tensor(win))[:, None]
    j = torch.arange(win + B)
    idx = torch.where(j < filled, j, torch.where(j < filled + B, win + j - filled, -1))
    topk = idx[:, None, :].expand(A, B, win + B).int().contiguous()
    o = M.sparse_attn(q, kvs, attn.attn_sink, topk, attn.softmax_scale)
    rotate(o[..., -rd:], freqs.unsqueeze(2), inverse=True)
    o = o.view(A, B, attn.n_local_groups, -1)
    M.act_quant(o, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return attn.wo_b(o.flatten(2))


def build_stages():
    stages = []
    for j in range(ARGS.n_mtp_layers):
        blk = M.DSparkBlock(ARGS.n_layers + j, ARGS)
        load_module(blk, f"mtp.{j}.", skip=("ffn.experts.",))
        attn = blk.attn
        attn.forward = types.MethodType(
            lambda self, x, start_pos, anchors, ctx: dspark_attention_many(self, x, anchors, ctx), attn
        )
        stages.append(blk)
    return stages


@torch.inference_mode()
def draft(stages, main_hidden: torch.Tensor, seq: list[int], anchors: list[int], head) -> dict:
    """The drafter at every anchor: drafts, second candidates, the position-1 branch chain,
    and the confidence logits. ``seq`` supplies each anchor token (the token at its position);
    ``head`` is head_chunks()'s."""
    first, last = stages[0], stages[-1]
    main_x = first.main_norm(first.main_proj(main_hidden.unsqueeze(0)))[0]
    ctxs = [context_kv(blk.attn, main_x) for blk in stages]
    out = {"drafts": [], "second": [], "margin": [], "branch": [], "confidence": []}
    markov_embed = last.markov_head.embed.weight  # [V, rank], BF16
    markov_head = last.markov_head.head.weight  # [V, rank], FP32
    for c0 in range(0, len(anchors), ARGV.anchor_chunk):
        a = torch.tensor(anchors[c0 : c0 + ARGV.anchor_chunk])
        A = a.numel()
        ids = torch.full((A, BLOCK), NOISE, dtype=torch.long)
        ids[:, 0] = torch.tensor([seq[p] for p in a.tolist()])
        x = embed_rows(ids.flatten().tolist()).view(A, BLOCK, ARGS.dim)
        x = x.unsqueeze(2).repeat(1, 1, ARGS.hc_mult, 1)
        pre_mix = M.make_identity_pre_mix(x, ARGS.hc_mult)
        for blk, ctx in zip(stages, ctxs):
            EXPERTS.clear()
            x, pre_mix = M.Block.forward(blk, x, 1, pre_mix, None, a, ctx)
        EXPERTS.clear()
        xh = last.hc_pre(x, pre_mix)  # [A, B, dim]: forward_head's x, before the norm
        normed = last.norm(xh).float().view(A * BLOCK, -1)
        base = torch.empty(A * BLOCK, ARGS.vocab_size)
        for r, w in head:
            base[:, r : r + w.size(0)] = F.linear(normed, w.float())
        base = base.view(A, BLOCK, -1)
        prev, prev_branch = ids[:, 0], None
        drafts, second, margin, branch, embeds = [], [], [], [], []
        for i in range(BLOCK):
            emb = F.embedding(prev, markov_embed)
            logits = base[:, i] + F.linear(emb.float(), markov_head)
            top = logits.topk(2, dim=-1)
            drafts.append(top.indices[:, 0])
            second.append(top.indices[:, 1])
            margin.append(top.values[:, 0] - top.values[:, 1])
            embeds.append(emb)
            if i == 0:
                prev_branch = top.indices[:, 1]
                branch.append(prev_branch)
            else:
                bl = base[:, i] + F.linear(F.embedding(prev_branch, markov_embed).float(), markov_head)
                prev_branch = bl.argmax(dim=-1)
                branch.append(prev_branch)
            prev = top.indices[:, 0]
        conf = last.confidence_head(xh, torch.stack(embeds, dim=1))  # [A, B] logits
        out["drafts"] += torch.stack(drafts, 1).tolist()
        out["second"] += torch.stack(second, 1).tolist()
        out["margin"] += [[round(v, 4) for v in row] for row in torch.stack(margin, 1).tolist()]
        out["branch"] += torch.stack(branch, 1).tolist()
        out["confidence"] += [[round(v, 4) for v in row] for row in conf.float().tolist()]
        del base
    return out


@torch.inference_mode()
def target_greedy(final: torch.Tensor, positions: list[int], head) -> list[int]:
    """The FP32 head's argmax after each of ``positions``."""
    x = final[positions].float()
    best_v = torch.full((len(positions),), -float("inf"))
    best_i = torch.zeros(len(positions), dtype=torch.long)
    for r, w in head:
        v, i = F.linear(x, w.float()).max(dim=-1)
        better = v > best_v
        best_v = torch.where(better, v, best_v)
        best_i = torch.where(better, i + r, best_i)
    return best_i.tolist()


# --- what an engine verifying every draft would accept ------------------------------------------


def accepted(drafts: list[int], seq: list[int], a: int) -> tuple[int, int]:
    """(leading drafts that match the text after anchor a, drafts the text can judge)."""
    judged = min(len(drafts), len(seq) - a - 1)
    k = 0
    while k < judged and drafts[k] == seq[a + 1 + k]:
        k += 1
    return k, judged


def summarize(case_id, seq, start, rec) -> dict:
    index = {a: n for n, a in enumerate(rec["anchors"])}
    steps = tokens = 0
    by_pos = [[0, 0] for _ in range(BLOCK)]  # (judged, accepted) at each draft position
    second_right = [[0, 0] for _ in range(BLOCK)]  # (first wrong here, second right)
    branch_extra = []
    a = start
    while a < len(seq) - 1:
        n = index[a]
        k, judged = accepted(rec["drafts"][n], seq, a)
        steps += 1
        tokens += min(k + 1, len(seq) - 1 - a)
        for i in range(judged):
            by_pos[i][0] += 1
            by_pos[i][1] += int(i < k)
        if k < judged:
            second_right[k][0] += 1
            second_right[k][1] += int(rec["second"][n][k] == seq[a + 1 + k])
            if k == 0 and rec["branch"][n][0] == seq[a + 1]:
                kb, _ = accepted(rec["branch"][n], seq, a)
                branch_extra.append(kb)
        a += k + 1
    every = [[0, 0] for _ in range(BLOCK)]
    for a in rec["anchors"]:
        k, judged = accepted(rec["drafts"][index[a]], seq, a)
        for i in range(judged):
            every[i][0] += 1
            every[i][1] += int(i < k)
    return {
        "id": case_id,
        "steps": steps,
        "tokens": tokens,
        "tokens_per_step": round(tokens / steps, 4) if steps else None,
        "accepted_by_position": [p[1] for p in by_pos],
        "judged_by_position": [p[0] for p in by_pos],
        "every_anchor_accepted": [p[1] for p in every],
        "every_anchor_judged": [p[0] for p in every],
        "first_wrong_second_right": second_right,
        "branch_accepted_when_second_right": branch_extra,
    }


# --- cases --------------------------------------------------------------------------------------


def load_cases() -> list[dict]:
    """A {"cases": [...]} file, or per_request_acceptance.py's JSON lines (its summary line and
    any request it marked with an error are left out)."""
    text = open(ARGV.cases).read()
    if text.lstrip().startswith("{\"cases\""):
        cases = json.loads(text)["cases"]
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        cases = [r for r in rows if "id" in r and "error" not in r]
    tokenizer = None
    for c in cases:
        if "prompt_ids" in c and "output_ids" in c:
            continue
        if tokenizer is None:
            from encoding import encode_messages
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(SNAP)
        mode = "thinking" if c.get("thinking") else "chat"
        c["prompt_ids"] = tokenizer.encode(encode_messages(c["messages"], thinking_mode=mode))
        c["output_ids"] = tokenizer.encode(c["output_text"], add_special_tokens=False)
        c["rendered"] = True
    return cases[: ARGV.max_cases] if ARGV.max_cases else cases


def run_case(case, stages) -> dict:
    seq = list(case["prompt_ids"]) + list(case["output_ids"])
    start = len(case["prompt_ids"])
    assert len(seq) <= ARGS.max_seq_len, (case["id"], len(seq))
    t0 = time.time()
    main_hidden, final = target_forward(seq)
    head = head_chunks()  # BF16, held for this case's head passes
    greedy = target_greedy(final, list(range(start - 1, len(seq) - 1)), head)
    agree = sum(g == t for g, t in zip(greedy, seq[start:]))
    anchors = list(range(start, len(seq) - 1))
    rec = {"anchors": anchors, **draft(stages, main_hidden, seq, anchors, head)}
    del head
    summary = summarize(case["id"], seq, start, rec)
    summary["greedy_agrees"] = [agree, len(greedy)]
    summary["seconds"] = round(time.time() - t0, 1)
    if "engine" in case:
        summary["engine"] = case["engine"]
    return {"id": case["id"], "prompt_len": start, "output_ids": case["output_ids"],
            "greedy_after": greedy, "summary": summary, **rec}


# --- the self-test: the batched drafter against the reference's own sequential loop -------------


def tiny_selftest() -> dict:
    model = M.Transformer(ARGS)
    load_module(model, "", skip=("vision.", "aligner.", "ffn.experts."))
    model.eval()
    g = torch.Generator().manual_seed(7)
    seq = torch.randint(0, 3900, (200,), generator=g).tolist()
    prompt = 140  # past the 128-slot window, so the ring wraps during the loop
    with torch.inference_mode():
        _, _, mh = model(torch.tensor([seq[:prompt]]))
        model.forward_spec(torch.tensor([seq[prompt]]), mh)
        hidden = [mh[0]]
        ref = {}
        for i in range(prompt, len(seq) - 1):
            _, _, mh = model(torch.tensor([seq[i : i + 1]]), i)
            hidden.append(mh[0])
            out, _, conf = model.forward_spec(torch.tensor([seq[i + 1]]), mh, i)
            ref[i + 1] = (out[0, 1:].tolist(), conf[0].float().tolist())
    main_hidden = torch.cat(hidden, dim=0)  # positions 0 .. len-2, as the loop computed them
    stages = build_stages()
    anchors = sorted(ref)
    mine = draft(stages, main_hidden, seq, anchors, head_chunks())
    same = sum(mine["drafts"][n] == ref[a][0] for n, a in enumerate(anchors))
    conf_err = max(abs(x - y) for n, a in enumerate(anchors) for x, y in zip(mine["confidence"][n], ref[a][1]))
    # And the prefill path for the context states against the loop's incremental ones.
    prefill_hidden, _ = target_forward(seq[:-1])
    drift = (prefill_hidden.float() - main_hidden.float()).abs().max().item()
    return {"anchors": len(anchors), "drafts_identical": same, "max_confidence_logit_error": conf_err,
            "prefill_vs_incremental_context_max_abs": drift}


def main() -> None:
    t_start = time.time()
    if ARGV.tiny:
        result = tiny_selftest()
        print(json.dumps(result), flush=True)
        json.dump(result, open(ARGV.out, "w"), indent=1)
        return
    deadline = None
    if ARGV.deadline:
        from datetime import datetime

        deadline = datetime.fromisoformat(ARGV.deadline.replace("Z", "+00:00")).timestamp()
    cases = load_cases()
    stages = build_stages()
    records = []
    for case in cases:
        if deadline is not None and time.time() > deadline:
            print(f"deadline passed: {case['id']} and later cases not started", flush=True)
            break
        print(f"{case['id']}: prompt {len(case['prompt_ids'])}, reply {len(case['output_ids'])}", flush=True)
        records.append(run_case(case, stages))
        s = records[-1]["summary"]
        print(f"{case['id']}: {s['tokens_per_step']} tokens a step, greedy {s['greedy_agrees']}, "
              f"{s['seconds']}s", flush=True)
        with open(ARGV.out, "w") as f:  # after every case, so a later failure keeps the earlier ones
            json.dump({"snapshot": SNAP, "block": BLOCK, "cases": records,
                       "bytes_read": shards.bytes_read, "seconds": round(time.time() - t_start, 1),
                       "torch": torch.__version__, "threads": torch.get_num_threads()}, f)
    print("wrote", ARGV.out, flush=True)


if __name__ == "__main__":
    main()
