"""The routed experts of copied and ordinary text, from DeepSeek's V4.1 reference forward on CPU.

usage (inside the serving image, repository at /repo, snapshot read-only):
  python3 -I routes_probe.py SNAPSHOT OUT.json [--cases copy,edit,tool,prose] [--tokens 1000]

Each case is one conversation rendered by the checkpoint's own encoding: a user turn and the
assistant reply a copy-heavy request produces (copy_probe.py's categories, 2026-10-23). The
reply is forced: one prefill at position 0 over prompt + reply, which routes every position
exactly as incremental decoding would, without one forward per token. The record keeps every
layer's top-6 experts at every position, and the reference's greedy token at each reply
position, which says how close the forced reply is to the model's own.

What the routes answer: how many distinct experts r consecutive reply rows of one request reach
(the bytes a verification block of r rows reads at one stream), on copied text and on prose.

The loader is ../../2026-10-18-json-count/reference/ref_logits.py's (O_DIRECT reads, only the
routed experts a layer uses, one layer resident at a time), with its CPU kernels (shim/).
"""

import argparse
import json
import mmap
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIM = HERE.parents[1] / "2026-10-18-json-count" / "reference" / "shim"

parser = argparse.ArgumentParser()
parser.add_argument("snapshot")
parser.add_argument("out")
parser.add_argument("--cases", default="copy,edit,tool,prose")
parser.add_argument("--tokens", type=int, default=1000, help="source tokens of the copied file")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--repo", default="/repo", help="spark-ds41f checkout (servebench prompts)")
ARGV = parser.parse_args()

SNAP = os.path.abspath(ARGV.snapshot)
sys.path.insert(0, os.path.join(SNAP, "encoding"))
sys.path.insert(0, os.path.join(SNAP, "inference"))
sys.path.insert(0, str(SHIM))  # first, so model.py's `from kernel import` gets the shim
sys.path.insert(1, ARGV.repo)
sys.path.insert(1, str(HERE.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

import kernel as shim  # noqa: E402
import model as M  # noqa: E402
from encoding import encode_messages  # noqa: E402
from engram import EngramLayout, NgramHashState  # noqa: E402
from servebench import prompts  # noqa: E402

import cases  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

assert M.act_quant is shim.act_quant, "model.py did not pick up the CPU kernels"

ALIGN = 4096
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


shards = Shards(SNAP)


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
        if any(name.startswith(s) for s in skip):
            continue
        full = prefix + name
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


M.ParallelEngramEmbedding = RowReadEngramEmbedding

ROUTES = {}


def moe_forward(self, x, image_mask=None):
    """MoE.forward at world size 1, with each routed expert built and loaded when first used."""
    shape = x.size()
    x = x.view(-1, self.dim)
    weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
    ROUTES[self.layer_id] = indices.tolist()
    y = torch.zeros_like(x, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
    for i in range(self.experts_start_idx, self.experts_end_idx):
        if counts[i] == 0:
            continue
        expert = M.Expert.__new__(M.Expert)
        _expert_init(expert, self.dim, ARGS.moe_inter_dim, torch.float4_e2m1fn_x2, ARGS.swiglu_limit)
        load_module(expert, f"layers.{self.layer_id}.ffn.experts.{i}.")
        idx, top = torch.where(indices == i)
        y[idx] += expert(x[idx], weights[idx, top, None])
        del expert
    y += self.shared_experts(x)
    return y.type_as(x).view(shape)


M.MoE.forward = moe_forward

torch.set_default_dtype(torch.bfloat16)
torch.set_num_threads(int(os.environ.get("REF_THREADS", "8")))
torch.manual_seed(33377335)
with open(os.path.join(SNAP, "inference", "config.json")) as f:
    ARGS = M.ModelArgs(**json.load(f))
ARGS.max_batch_size = 1
ARGS.max_seq_len = 4096
ARGS.temperature = 0.0

tokenizer = AutoTokenizer.from_pretrained(SNAP)
layout = EngramLayout.from_args(ARGS)
engram_hash = NgramHashState(ARGS, layout, tokenizer)
final_norm = M.RMSNorm(ARGS.dim, ARGS.norm_eps)
load_module(final_norm, "norm.")


def render(name: str) -> tuple[list[int], int]:
    """Token ids of prompt + reply, and where the reply starts."""
    code = cases.document(prompts, ARGV.tokens, ARGV.seed * 1000)
    prose = None
    if name == "prose":
        text = (Path(ARGV.repo) / "docs" / "methodology.md").read_text(encoding="utf-8")
        prose = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[: ARGV.tokens])
    messages, reply = cases.conversation(name, code, prose_text=prose)
    ids, start = cases.render(encode_messages, tokenizer.encode, messages, reply)
    assert len(ids) <= ARGS.max_seq_len, (name, len(ids))
    return ids, start


@torch.inference_mode()
def forward(input_ids: torch.Tensor) -> torch.Tensor:
    """Transformer.forward at position 0, text only, one layer resident at a time; returns the
    final-normed hidden states of every position."""
    hashes = engram_hash(input_ids, 0, None)
    rows = input_ids.flatten().tolist()
    h = shards.get_rows("embed.weight", rows).view(*input_ids.shape, ARGS.dim)
    h = h.unsqueeze(2).repeat(1, 1, ARGS.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, ARGS.hc_mult)
    for i in range(ARGS.n_layers):
        t0 = time.time()
        block = M.Block(i, ARGS, layout)
        if block.engram is not None:
            block.engram.embed.prefix = f"layers.{i}.engram.embed."
        load_module(block, f"layers.{i}.", skip=("ffn.experts.", "engram.embed."))
        if block.engram is not None:
            h = block.engram(h, hashes[:, :, block.engram.layer_hash_index, :], None)
        h, pre_mix = block(h, 0, pre_mix, None)
        if i == ARGS.n_layers - 1:
            h = block.hc_pre(h, pre_mix)
        del block
        print(f"  layer {i:2d} {time.time() - t0:6.1f}s read {shards.bytes_read / 2**30:8.2f} GiB", flush=True)
    return final_norm(h)


@torch.inference_mode()
def greedy(hidden: torch.Tensor, positions: list[int]) -> list[int]:
    """The FP32 head's argmax after each of ``positions`` (BF16 rows read in chunks)."""
    x = hidden[0, positions].float()
    fname, off, length, dt, shape = shards.meta("head.weight")
    row_bytes = length // shape[0]
    best_v = torch.full((len(positions),), -float("inf"))
    best_i = torch.zeros(len(positions), dtype=torch.long)
    for r in range(0, shape[0], 8192):
        n = min(8192, shape[0] - r)
        w = shards._read(fname, off + r * row_bytes, n * row_bytes).view(dt).reshape(n, shape[1]).float()
        v, i = F.linear(x, w).max(dim=-1)
        better = v > best_v
        best_v = torch.where(better, v, best_v)
        best_i = torch.where(better, i + r, best_i)
    return best_i.tolist()


records = []
t_start = time.time()
for name in ARGV.cases.split(","):
    ids, reply_start = render(name)
    print(f"{name}: {len(ids)} tokens, reply from {reply_start} ({len(ids) - reply_start} tokens)", flush=True)
    ROUTES.clear()
    t0 = time.time()
    hidden = forward(torch.tensor([ids]))
    predicted = greedy(hidden, list(range(reply_start - 1, len(ids) - 1)))
    agree = sum(p == t for p, t in zip(predicted, ids[reply_start:]))
    print(f"{name}: greedy agrees at {agree}/{len(predicted)} reply positions, {time.time() - t0:.0f}s", flush=True)
    records.append({
        "case": name,
        "token_ids": ids,
        "reply_start": reply_start,
        "greedy_after": predicted,
        "routes": [ROUTES[layer] for layer in range(ARGS.n_layers)],
        "seconds": round(time.time() - t0, 1),
    })
    with open(ARGV.out, "w") as f:  # after every case, so a later failure keeps the earlier ones
        json.dump({"snapshot": SNAP, "tokens": ARGV.tokens, "seed": ARGV.seed, "cases": records,
                   "bytes_read": shards.bytes_read, "seconds": round(time.time() - t_start, 1),
                   "torch": torch.__version__, "threads": torch.get_num_threads()}, f)
print("wrote", ARGV.out, flush=True)
