"""Expert routes of real text through DeepSeek's V4.1 reference forward (the checkpoint's
inference/model.py), on CPU one layer at a time: the real routers with the checkpoint's gate
weights, after every earlier layer's real computation.

Usage: python3 -I ref_routes.py <snapshot-dir> <sequences.json> <out.pt>

The driver is experiments/2026-10-18-json-count/reference/ref_logits.py's (the same O_DIRECT
shard reads, lazily built experts and row-read Engram tables, the same six CPU kernels in
shim/kernel.py), with its json_count prompt replaced by a batch of equal-length sequences
(build_grid_tokens.py) prefilled in one forward. It keeps every MoE layer's top-6 expert ids
for every position and writes them as int16, [layers, sequences, positions, 6]. No head runs.
"""

import json
import mmap
import os
import sys
import time

SNAP = os.path.abspath(sys.argv[1])
SEQS = sys.argv[2]
OUT = sys.argv[3]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SNAP, "encoding"))
sys.path.insert(0, os.path.join(SNAP, "inference"))
sys.path.insert(0, os.path.join(HERE, "shim"))  # first, so model.py's `from kernel import` gets the shim

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

import kernel as shim  # noqa: E402
import model as M  # noqa: E402
from engram import EngramLayout, NgramHashState  # noqa: E402
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
CONVERSIONS = {}


def assign(p: nn.Parameter, t: torch.Tensor, name: str):
    if p.dtype == torch.float4_e2m1fn_x2:
        assert t.dtype == torch.int8, (name, t.dtype)
        t = t.view(torch.float4_e2m1fn_x2)
    elif t.dtype != p.dtype:
        key = f"{t.dtype}->{p.dtype}"
        widening = (t.dtype, p.dtype) in {(torch.bfloat16, torch.float32), (torch.float16, torch.float32)}
        assert widening, f"narrowing load {name}: {key}"
        CONVERSIONS.setdefault(key, set()).add(name.split(".", 2)[-1] if name.startswith("layers.") else name)
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


# Routed experts and the Engram tables are never allocated whole: experts are built on demand,
# table rows are read on lookup.
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
        _, _, _, _, shape = shards.meta(self.prefix + "weight")
        assert shape == [self.num_embeddings, self.dim], (shape, self.num_embeddings)
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
    ROUTES[self.layer_id] = indices.to(torch.int16).clone()
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
with open(SEQS) as f:
    SPEC = json.load(f)
SEQUENCES = SPEC["sequences"]
LENGTH = len(SEQUENCES[0])
assert all(len(s) == LENGTH for s in SEQUENCES)
ARGS.max_batch_size = len(SEQUENCES)
ARGS.max_seq_len = 256
ARGS.temperature = 0.0
LAYERS = int(os.environ.get("REF_LAYERS", ARGS.n_layers))  # fewer for a smoke test

tokenizer = AutoTokenizer.from_pretrained(SNAP)
layout = EngramLayout.from_args(ARGS)
engram_hash = NgramHashState(ARGS, layout, tokenizer)


@torch.inference_mode()
def forward(input_ids: torch.Tensor, start_pos: int):
    """Transformer.forward's blocks, text only, one layer resident at a time; no head."""
    hashes = engram_hash(input_ids, start_pos, None)
    rows = input_ids.flatten().tolist()
    h = shards.get_rows("embed.weight", rows).view(*input_ids.shape, ARGS.dim)
    h = h.unsqueeze(2).repeat(1, 1, ARGS.hc_mult, 1)
    pre_mix = M.make_identity_pre_mix(h, ARGS.hc_mult)
    for i in range(LAYERS):
        t0 = time.time()
        block = M.Block(i, ARGS, layout)
        if block.engram is not None:
            block.engram.embed.prefix = f"layers.{i}.engram.embed."
        load_module(block, f"layers.{i}.", skip=("ffn.experts.", "engram.embed."))
        if block.engram is not None:
            h = block.engram(h, hashes[:, :, block.engram.layer_hash_index, :], None)
        h, pre_mix = block(h, start_pos, pre_mix, None)
        del block
        r = ROUTES[i].view(input_ids.shape[0], input_ids.shape[1], -1)
        print(f"  layer {i:2d} {time.time() - t0:6.1f}s read {shards.bytes_read / 2**30:7.2f} GiB "
              f"experts hit {int(torch.unique(r).numel())} of {ARGS.n_routed_experts}", flush=True)


t_start = time.time()
print(f"prefill {len(SEQUENCES)} sequences of {LENGTH} tokens", flush=True)
forward(torch.tensor(SEQUENCES), 0)
routes = torch.stack([ROUTES[i].view(len(SEQUENCES), LENGTH, -1) for i in range(LAYERS)])
torch.save({"routes": routes, "labels": SPEC.get("labels"), "sequences": SEQUENCES,
            "context": SPEC.get("context"), "window": SPEC.get("window"),
            "snapshot": SNAP, "bytes_read": shards.bytes_read,
            "seconds": round(time.time() - t_start, 1), "torch": str(torch.__version__),
            "threads": torch.get_num_threads(), "kernel_stats": shim.STATS}, OUT)
print("wrote", OUT, f"in {time.time() - t_start:.0f} s", flush=True)
