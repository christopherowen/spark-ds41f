#!/usr/bin/env python3
"""Two lossless-byte questions about the checkpoint, read on CPU (standard library only).

usage: weight_stats.py SNAPSHOT_DIR [--sample-mib 8] [--expert-stride 32] [--out weight_stats.json]

1. The mHC mix projections (``*hc*fn*``, FP32): is every value exactly a BF16 (its low 16
   bits zero)? If so, storing them as BF16 and widening in the kernel reads half the bytes
   with identical arithmetic.
2. The empirical entropy of the stored codes, which bounds any lossless coding of them:
   E4M3 bytes of the FP8 weights, E2M1 nibbles of the packed FP4 experts, the exponent
   byte of BF16 weights. ``--sample-mib`` bytes are read from the middle of each tensor
   (the whole tensor if smaller), and of the routed experts only every
   ``--expert-stride``-th one (each expert tensor is smaller than the sample, so all of
   them would be about 130 GB); the fn tensors are always read whole.

Tensors are read from the safetensors shards by offset, so a run touches only the sampled
bytes. It sends nothing anywhere and writes one JSON file.
"""
import argparse
import glob
import json
import math
import os
import re
import struct
from collections import Counter


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def read(path, base, offsets, limit=None):
    start, end = offsets
    size = end - start
    if limit and size > limit:
        start += (size - limit) // 2 // 64 * 64
        size = limit
    with open(path, "rb") as f:
        f.seek(base + start)
        return f.read(size)


def entropy(counts):
    total = sum(counts.values())
    return -sum(c / total * math.log2(c / total) for c in counts.values() if c)


def kind(name, dtype):
    if "hc" in name and name.endswith("fn"):
        return "mhc_fn"
    if dtype == "F8_E4M3":
        return "fp8"
    if dtype in ("U8", "I8", "F4") and ".experts." in name and "scale" not in name:
        return "fp4_packed"
    if dtype == "BF16":
        return "bf16"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--sample-mib", type=float, default=8)
    ap.add_argument("--expert-stride", type=int, default=32)
    ap.add_argument("--out", default="weight_stats.json")
    args = ap.parse_args()
    limit = int(args.sample_mib * (1 << 20))
    hist = {k: Counter() for k in ("fp8", "fp4_nibbles", "bf16_exponent")}
    sampled = Counter()
    stored = Counter()
    fn = {"tensors": 0, "values": 0, "not_bf16": 0, "names_not_bf16": [], "dtypes": Counter()}
    for path in sorted(glob.glob(os.path.join(args.snapshot, "*.safetensors"))):
        base, meta = header(path)
        for name, t in meta.items():
            if name == "__metadata__":
                continue
            k = kind(name, t["dtype"])
            if k is None:
                continue
            stored[k] += t["data_offsets"][1] - t["data_offsets"][0]
            if k == "mhc_fn":
                fn["dtypes"][t["dtype"]] += 1
                fn["tensors"] += 1
                if t["dtype"] != "F32":  # already stored narrower than the FP32 the kernel reads
                    continue
                data = read(path, base, t["data_offsets"])
                words = struct.unpack(f"<{len(data) // 4}I", data)
                bad = sum(1 for w in words if w & 0xFFFF)
                fn["values"] += len(words)
                fn["not_bf16"] += bad
                if bad and len(fn["names_not_bf16"]) < 8:
                    fn["names_not_bf16"].append(name)
                continue
            expert = re.search(r"\.experts\.(\d+)\.", name)
            if expert and int(expert.group(1)) % args.expert_stride:
                continue
            data = read(path, base, t["data_offsets"], limit)
            sampled[k] += len(data)
            if k == "fp8":
                hist["fp8"].update(data)
            elif k == "fp4_packed":
                c = Counter(data)
                for byte, n in c.items():
                    hist["fp4_nibbles"][byte & 15] += n
                    hist["fp4_nibbles"][byte >> 4] += n
            else:  # BF16 little-endian: the high byte holds the sign and seven exponent bits
                hist["bf16_exponent"].update(data[1::2])
    out = {"snapshot": args.snapshot, "sample_mib_per_tensor": args.sample_mib,
           "expert_stride": args.expert_stride,
           "stored_bytes": dict(stored), "sampled_bytes": dict(sampled), "mhc_fn": fn,
           "fn_exactly_bf16": fn["tensors"] > 0 and fn["not_bf16"] == 0,  # true too when stored as BF16
           "entropy_bits": {"fp8 (of 8)": round(entropy(hist["fp8"]), 3),
                            "fp4 nibble (of 4)": round(entropy(hist["fp4_nibbles"]), 3),
                            "bf16 sign+exponent byte (of 8)": round(entropy(hist["bf16_exponent"]), 3)}}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps({k: out[k] for k in ("fn_exactly_bf16", "entropy_bits")}))


if __name__ == "__main__":
    main()
