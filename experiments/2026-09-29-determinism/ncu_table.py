#!/usr/bin/env python3
"""Summarize the Nsight Compute sections of run20's output.  usage: ncu_table.py FILE

Each bench call under ncu is three eager MoE calls per (rows, dead) shape, in
the order the bench prints them; kernels are grouped per shape and averaged
per call. Reports duration and DRAM read/write and L2 bytes for the fused
MoE kernel and the top-k sum.
"""
import csv
import io
import re
import sys
from collections import defaultdict

text = open(sys.argv[1]).read()
sections = re.split(r"^== ncu (\w+)\n", text, flags=re.M)
for mode, body in zip(sections[1::2], sections[2::2]):
    body = body.split("\n== ")[0]
    lines = [l for l in body.splitlines() if l.startswith('"')]
    if not lines:
        print(mode, "no ncu rows")
        continue
    reader = list(csv.reader(io.StringIO("\n".join(lines))))
    header, units, data = reader[0], reader[1], reader[2:]
    col = {name: i for i, name in enumerate(header)}
    shapes = re.findall(r'\{"mode": "\w+", "rows": (\d+), "dead": ([0-9.]+), "ncu": true\}', body)
    kinds = []
    for row in data:
        name = row[col["Kernel Name"]]
        kind = "topk_sum" if "TopKSum" in name else "fused_moe"
        kinds.append((kind, row))
    # the first kernel pair is the warm-up call (1 call) + 3 calls per shape
    per_shape = defaultdict(lambda: defaultdict(list))
    fused = [r for k, r in kinds if k == "fused_moe"]
    topk = [r for k, r in kinds if k == "topk_sum"]
    calls = 4  # warm-up + three measured calls per shape
    for i, shape in enumerate(shapes):
        for kind, rows in (("fused_moe", fused), ("topk_sum", topk)):
            for r in rows[i * calls + 1:(i + 1) * calls]:
                per_shape[shape][kind].append(r)
    print(f"== {mode}  (duration unit {units[col['gpu__time_duration.sum']]}, L2 unit {units[col['lts__t_bytes.sum']]})")
    for shape, kinds_ in per_shape.items():
        parts = []
        for kind, rows in kinds_.items():
            if not rows:
                continue
            avg = lambda key: sum(float(r[col[key]].replace(",", "")) for r in rows) / len(rows)
            parts.append(f"{kind}: {avg('gpu__time_duration.sum') / 1000:.1f} us, sysmem fill "
                         f"{avg('lts__d_sectors_fill_sysmem.sum') * 32 / 2**20:.2f} MiB, read miss "
                         f"{avg('lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum') * 32 / 2**20:.2f} MiB, "
                         f"write {avg('lts__t_sectors_aperture_sysmem_op_write.sum') * 32 / 2**20:.2f} MiB, "
                         f"L2 {avg('lts__t_bytes.sum') / 2**20:.2f} MiB")
        print(f"  rows {shape[0]:>2} dead {shape[1]}: " + " | ".join(parts))
