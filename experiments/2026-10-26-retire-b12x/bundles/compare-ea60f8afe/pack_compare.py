#!/usr/bin/env python3
"""vLLM's TileLang packer against B12X's (patch 0012), byte for byte (a kernel-lab script,
run in candidate 6's image, which has B12X).

usage: pack_compare.py CHECKPOINT [--logits-rows 1,6,64] [--port-only]

With --port-only (an image without B12X) only vLLM's packer runs: each case still checks the
round trip and prints the packed values' and the logits' SHA-256, for comparison with a run
that had B12X.

Cases:
- synthetic weights: B12X's own test shapes and value classes (zeros, subnormals, values
  far outside the window, infinities, rows with many exceptions), K not a multiple of 256,
  a weight without exceptions, in place across a partial 1,024-row block;
- the checkpoint's head.weight as vLLM shards it (VocabParallelEmbedding: rank r's rows of
  the vocabulary padded to a multiple of lcm(64, tp), the padding rows zero) at TP1, TP3
  and TP4, every rank, packed out of place and in place.
Each case compares every field of the two packed weights (values, exponent base, exception
rows and entries, row offsets, exception count, shapes and dtypes) and checks that the
port's unpack gives the source bits. For the checkpoint shards, the TileLang projection
(vocab.packed_vocab_projection, the served head) then runs on both packed weights with the
same hidden states, for BF16 and FP32 logits, and the logits must be the same bits. Shapes
the packer refuses must raise the same exception type in both. Prints one JSON line a case
and a summary; exits 1 on any difference.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

DEVICE = "cuda"


def synthetic(rows: int, k: int, *, seed: int = 0, infinities: bool = False) -> torch.Tensor:
    """B12X's test weights (tests/gemm/test_packed_bf16_vocab_projection.py ``_weight``)."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    weight = (torch.randn(rows, k, device=DEVICE, generator=generator) * 0.02).to(torch.bfloat16)
    bits = weight.view(torch.int16)
    specials = [0x0000, -0x8000, 0x0001, -0x7FFF, 0x007F, 0x0080, 0x2000, 0x1000, 0x3F80, -0x4080,
                0x4500]
    if infinities:
        specials += [0x7F7F, 0x7F80, -0x0080, 0x7FC0]
    specials = torch.tensor(specials, dtype=torch.int16, device=DEVICE)
    bits[3, : specials.numel()] = specials
    bits[7, 64 : 64 + specials.numel()] = specials
    bits[rows - 2, ::3] = 0x4100
    bits[rows - 5, 1::4] = 0x2000
    return weight


def shards(head: torch.Tensor, tp: int):
    """(rank, shard) as vLLM's ParallelLMHead loads them: padded vocabulary, zero padding."""
    vocab = head.shape[0]
    padded = -(-vocab // math.lcm(64, tp)) * math.lcm(64, tp)
    per_rank = padded // tp
    for rank in range(tp):
        start = rank * per_rank
        end = min(start + per_rank, vocab)
        shard = torch.zeros((per_rank, head.shape[1]), dtype=head.dtype, device=DEVICE)
        shard[: end - start].copy_(head[start:end])
        yield rank, shard


def fields(packed) -> dict:
    return {
        "values": packed.values,
        "exception_rows": packed.exception_rows,
        "exception_entries": packed.exception_entries,
        "row_offsets": packed.row_offsets,
    }


def same(a, b) -> list[str]:
    differ = [name for name in ("exponent_base", "exceptions") if getattr(a, name) != getattr(b, name)]
    for name, left in fields(a).items():
        right = fields(b)[name]
        if (left.shape != right.shape or left.dtype != right.dtype or left.stride() != right.stride()
                or not torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8))):
            differ.append(name)
    return differ


def sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def timed(function, *args, **kwargs):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = function(*args, **kwargs)
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def logits(packed, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    import tilelang.language as T

    from vllm.models.deepseek_v4_1.tilelang.vocab import packed_vocab_projection, row_tile, tile_config

    n, k = packed.out_features, packed.in_features
    block_M = row_tile(x.shape[0])
    kernel = packed_vocab_projection(
        n, k, packed.exponent_base, block_M=block_M,
        out_dtype={torch.bfloat16: T.bfloat16, torch.float32: T.float32}[out_dtype],
        **tile_config(block_M, k))
    out = torch.empty((x.shape[0], n), dtype=out_dtype, device=DEVICE)
    kernel(x, packed.values, packed.exception_rows, packed.exception_entries, packed.row_offsets, out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--logits-rows", default="1,6,64")
    parser.add_argument("--port-only", action="store_true")
    args = parser.parse_args()

    if args.port_only:
        b12x_format = None
    else:
        from b12x.gemm.packed_bf16_vocab_projection import _format as b12x_format

    from vllm.models.deepseek_v4_1.tilelang import vocab_pack

    failures = []

    def case(name: str, weight: torch.Tensor, *, in_place: bool, x_rows=()) -> None:
        source = weight.clone()
        reference = b12x_seconds = None
        if b12x_format is not None:
            reference, b12x_seconds = timed(b12x_format.pack, weight.clone(), in_place=in_place)
        ported_source = weight.clone()
        ported, port_seconds = timed(vocab_pack.pack, ported_source, in_place=in_place)
        differ = same(reference, ported) if reference is not None else []
        round_trip = torch.equal(vocab_pack.unpack(ported).view(torch.int16), source.view(torch.int16))
        in_storage = (not in_place) or ported.values.data_ptr() == ported_source.data_ptr()
        record = {
            "case": name, "in_place": in_place, "shape": list(weight.shape),
            "equal": not differ, "differ": differ, "round_trip": round_trip, "in_storage": in_storage,
            "exponent_base": ported.exponent_base, "exceptions": ported.exceptions,
            "values_sha256": sha256(ported.values),
            "exceptions_sha256": sha256(torch.cat([ported.exception_rows, ported.exception_entries,
                                                   ported.row_offsets])),
            "b12x_seconds": b12x_seconds and round(b12x_seconds, 4),
            "port_seconds": round(port_seconds, 4),
        }
        if reference is None:
            record["compared"] = "port only"
        logit_records = []
        for rows in x_rows:
            generator = torch.Generator(device=DEVICE).manual_seed(rows)
            x = (torch.randn(rows, weight.shape[1], device=DEVICE, generator=generator) * 0.5).bfloat16()
            for dtype in (torch.bfloat16, torch.float32):
                b = logits(ported, x, dtype)
                a = logits(reference, x, dtype) if reference is not None else b
                logit_records.append({"rows": rows, "dtype": str(dtype).removeprefix("torch."),
                                      "equal": torch.equal(a.view(torch.uint8), b.view(torch.uint8)),
                                      "sha256": sha256(b)})
        if logit_records:
            record["logits"] = logit_records
        ok = not differ and round_trip and in_storage and all(r["equal"] for r in logit_records)
        if not ok:
            failures.append(name + (" in place" if in_place else ""))
        print(json.dumps(record), flush=True)

    def refused(name: str, weight: torch.Tensor) -> None:
        kinds = []
        packs = (vocab_pack.pack,) if b12x_format is None else (b12x_format.pack, vocab_pack.pack)
        for pack in packs:
            try:
                pack(weight)
                kinds.append(None)
            except Exception as error:  # the type is what is compared
                kinds.append(type(error).__name__)
        equal = kinds[0] == kinds[-1] and kinds[0] is not None
        if not equal:
            failures.append(name)
        print(json.dumps({"case": name, "refused": kinds, "equal": equal}), flush=True)

    case("b12x-test-weight-infinities", synthetic(512, 640, infinities=True), in_place=False)
    case("b12x-test-weight-infinities", synthetic(512, 640, infinities=True), in_place=True)
    case("partial-block", synthetic(2_500, 5_120, seed=13, infinities=True), in_place=True)
    case("partial-block", synthetic(2_500, 5_120, seed=13, infinities=True), in_place=False)
    case("many-exceptions-k384", synthetic(1_000, 384, seed=5), in_place=False)
    case("no-exceptions", torch.full((16, 128), 0.5, device=DEVICE, dtype=torch.bfloat16), in_place=False)
    case("zeros", torch.zeros((64, 256), device=DEVICE, dtype=torch.bfloat16), in_place=True)
    one = torch.full((512, 256), 0.01, dtype=torch.bfloat16, device=DEVICE)
    one[5, 17] = 1.0e30
    case("one-exception", one, in_place=False)
    refused("k-not-multiple-of-128", torch.zeros((16, 96), device=DEVICE, dtype=torch.bfloat16))
    refused("float16", torch.zeros((16, 128), device=DEVICE, dtype=torch.float16))
    refused("non-contiguous", torch.zeros((128, 16), device=DEVICE, dtype=torch.bfloat16).t())

    from safetensors import safe_open

    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    with safe_open(args.checkpoint / index["weight_map"]["head.weight"], framework="pt") as archive:
        head = archive.get_tensor("head.weight").to(DEVICE)
    print(json.dumps({"head": list(head.shape), "dtype": str(head.dtype), "sha256": sha256(head)}),
          flush=True)
    rows = tuple(int(value) for value in args.logits_rows.split(",") if value)
    for tp in (1, 3, 4):
        for rank, shard in shards(head, tp):
            case(f"head-tp{tp}-rank{rank}", shard, in_place=True, x_rows=rows)
            if tp == 4:
                case(f"head-tp{tp}-rank{rank}", shard, in_place=False)
            del shard
    print(json.dumps({"summary": "pack port vs B12X", "failures": failures}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
