#!/usr/bin/env python3
"""vLLM's Engram helpers against B12X's (a kernel-lab script, run in candidate 6's image,
which has B12X, with the vLLM branch mounted).

usage: engram_compare.py CHECKPOINT

Compares, and prints one JSON line each:
- the compressed-token map of the checkpoint's tokenizer (every entry and the compressed
  vocabulary size; the size must also be the config's engram_compressed_vocab_size);
- the hash geometry of the checkpoint's config (layer ids, primes, offsets, multipliers,
  table rows, compressed size; the table rows must be the config's engram_num_embeddings);
- the per-layer caps of every TP3 and TP4 rank at the recipe's capacities, and the errors
  both raise for invalid capacities;
- a resident table's gather: vLLM's TileLang engram_table_gather against B12X's Triton
  lookup kernel (b12x.sequence.engram._kernels._lookup, resident rows, as B12X's
  bind_lookup/run_lookup launch it for a prepared max_tokens), bit for bit, on synthetic
  shards (every E4M3 code, scale bytes 0 and 255, rows outside the shard and past the
  table, dead lookups, live counts 0, partial, full, past capacity and negative) and on a
  slice of layer 1's checkpoint table and scales (real rows).
Exits 1 on any difference.
"""

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import torch

DEVICE = "cuda"
HEADS = 24


def record(name: str, equal: bool, failures: list, **fields) -> None:
    if not equal:
        failures.append(name)
    print(json.dumps({"check": name, "equal": equal, **fields}), flush=True)


def gather_pair(table, scales, ids, live, start, end, table_rows, max_tokens):
    """(vLLM's, B12X's) BF16 rows [max_tokens, 6144] of a resident shard."""
    from b12x.sequence.engram._kernels import _lookup

    from vllm.models.deepseek_v4_1.tilelang.engram_rows import engram_table_gather

    num_tokens = torch.tensor([live], dtype=torch.int32, device=DEVICE)
    ours = torch.full((max_tokens, HEADS * 256), 7, dtype=torch.bfloat16, device=DEVICE)
    theirs = torch.full_like(ours, 7)
    engram_table_gather(ids, num_tokens, table, scales, ours, start, min(end, table_rows))
    # B12X's run_lookup with token_count None: prepared = max_tokens, nothing to clear.
    _lookup[(max_tokens, HEADS, 1)](
        table, scales, ids, num_tokens, theirs, max_tokens,
        T=max_tokens, ROWS=table_rows, START=start, END=end, COMPACT=False,
        RESIDENT_SCALES=False, num_warps=4)
    return ours, theirs


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    failures: list = []

    from b12x.sequence.engram import _impl as b12x_impl
    from b12x.sequence.engram import geometry as b12x_geometry
    from transformers import AutoTokenizer

    from vllm.models.deepseek_v4_1.common import engram_geometry as ours

    config = json.loads((args.checkpoint / "config.json").read_text())["text_config"]

    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))
    their_map, their_size = b12x_geometry.build_compressed_token_map(tokenizer)
    our_map, our_size = ours.build_compressed_token_map(tokenizer)
    digest = hashlib.sha256(json.dumps(our_map).encode()).hexdigest()
    record("token map", our_map == their_map and our_size == their_size
           and our_size == config["engram_compressed_vocab_size"], failures,
           entries=len(our_map), compressed_size=our_size, sha256=digest)

    geometry_args = dict(layer_ids=config["engram_layer_ids"],
                         base_table_size=config["engram_vocab_size"],
                         compressed_vocab_size=config["engram_compressed_vocab_size"])
    theirs_geometry = b12x_geometry.build_geometry(**geometry_args)
    our_geometry = ours.build_geometry(**geometry_args)
    names = [field.name for field in dataclasses.fields(theirs_geometry)]
    record("geometry", all(getattr(our_geometry, n) == getattr(theirs_geometry, n) for n in names)
           and list(our_geometry.num_embeddings) == config["engram_num_embeddings"], failures,
           fields=names, num_embeddings=list(our_geometry.num_embeddings),
           multipliers=[list(m) for m in our_geometry.multipliers])

    caps_equal = True
    for tp in (3, 4):
        for rank in range(tp):
            for layer in config["engram_layer_ids"]:
                values = dict(device=DEVICE, max_tokens=16384, max_seqs=16, max_requests=16,
                              vocab_size=config["vocab_size"], layer_id=layer, tp_size=tp,
                              tp_rank=rank)
                a, b = ours.EngramCaps(**values), b12x_impl.Caps(**values)
                caps_equal &= all(getattr(a, f.name) == getattr(b, f.name)
                                  for f in dataclasses.fields(b))
                caps_equal &= [f.name for f in dataclasses.fields(a)] == [
                    f.name for f in dataclasses.fields(b)]
    refusals = []
    for bad in (dict(max_tokens=0), dict(tp_rank=4), dict(device="cpu"),
                dict(max_tokens=2**31), dict(max_seqs=1.5)):
        values = dict(device=DEVICE, max_tokens=8, max_seqs=2, max_requests=2, tp_size=4,
                      tp_rank=0) | bad
        kinds = []
        for caps in (ours.EngramCaps, b12x_impl.Caps):
            try:
                caps(**values)
                kinds.append(None)
            except Exception as error:  # the type and message are compared
                kinds.append(f"{type(error).__name__}: {error}")
        refusals.append(kinds[0] == kinds[1] and kinds[0] is not None)
    record("caps", caps_equal and all(refusals), failures, ranks="TP3 and TP4, layers "
           f"{config['engram_layer_ids']}", refusals=refusals)

    # Synthetic resident shards: every E4M3 code with random scales, the scale edges, rows
    # around and outside the shard and past the table, dead lookups.
    generator = torch.Generator().manual_seed(17)
    table_rows, max_tokens = 5000, 16
    codes = torch.randint(0, 256, (table_rows, 256), generator=generator, dtype=torch.uint8)
    codes[0] = torch.arange(256, dtype=torch.uint8)
    scale_bytes = torch.randint(100, 150, (table_rows, 8), generator=generator, dtype=torch.uint8)
    scale_bytes[1, 0], scale_bytes[2, 3], scale_bytes[3, 7] = 0, 255, 254
    for start, end in ((0, table_rows), (1000, 3000), (3750, 5000), (3750, 6250)):
        shard = torch.zeros((end - start, 256), dtype=torch.uint8)
        shard_scales = torch.full((end - start, 8), 127, dtype=torch.uint8)
        present = min(end, table_rows) - start
        shard[:present], shard_scales[:present] = codes[start:start + present], scale_bytes[start:start + present]
        table = shard.to(DEVICE).view(torch.float8_e4m3fn)
        scales = shard_scales.to(DEVICE)
        ids = torch.randint(-2, table_rows + 1500, (max_tokens, HEADS), generator=generator)
        ids[0, :4] = torch.tensor([0, 1, 2, 3])
        ids[1, :4] = torch.tensor([start - 1, start, min(end, table_rows) - 1, min(end, table_rows)])
        ids[2, :2] = torch.tensor([end - 1, end])
        ids = ids.to(DEVICE)
        for live in (0, 1, 9, max_tokens, max_tokens + 1, -1):
            a, b = gather_pair(table, scales, ids, live, start, end, table_rows, max_tokens)
            record(f"resident gather shard [{start}, {end}) live {live}", same_bits(a, b), failures,
                   nonzero=int(torch.count_nonzero(a)), nan=int(a.isnan().sum()))

    # Real rows: a 65,536-row slice of layer 1's table and scales, as a shard starting there.
    from safetensors import safe_open

    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    name = f"layers.{config['engram_layer_ids'][0]}.engram.embed"
    with safe_open(args.checkpoint / index["weight_map"][f"{name}.weight"], framework="pt") as archive:
        rows = archive.get_slice(f"{name}.weight")
        total = rows.get_shape()[0]
        start = total // 3
        weight = rows[start:start + 65536].contiguous()
    with safe_open(args.checkpoint / index["weight_map"][f"{name}.scale"], framework="pt") as archive:
        scale = archive.get_slice(f"{name}.scale")[start:start + 65536].contiguous()
    table = weight.view(torch.uint8).to(DEVICE).view(torch.float8_e4m3fn)
    scales = scale.view(torch.uint8).to(DEVICE)
    ids = torch.randint(start - 100, start + 65536 + 100, (max_tokens, HEADS), generator=generator)
    a, b = gather_pair(table, scales, ids.to(DEVICE), 13, start, start + 65536, total, max_tokens)
    record("resident gather, checkpoint rows", same_bits(a, b), failures, table=name, rows=total,
           shard=[start, start + 65536], nonzero=int(torch.count_nonzero(a)))

    print(json.dumps({"summary": "Engram helpers vs B12X", "failures": failures}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
