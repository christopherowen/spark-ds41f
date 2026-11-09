#!/usr/bin/env python3
"""Own-mark row-identity check for PR #969 (ours, not the PR's). The PR bench asserts that the four modes
agree within one image on its hot working set only; here every step's gathered rows of all three of its
workloads (hot M 1..128, Zipfian M 4..128 over 80 steps, cold disjoint M 4..32 over 20 steps), in all four
modes (two_plane/packed x nocache/cache64k), are hashed, so the base and candidate arms can be compared row
for row, cold misses (the direct-insert path) and in-step duplicate misses included. It also runs a
duplicate-heavy workload: 32 tokens that all look up the same 24 rows per column, two steps (miss, then hit).
Same table, seeds and id generators as bench_and_verify.run_cache_and_packed_benchmark.
usage: rows969.py LABEL  (writes /out/rows-LABEL.json)
"""

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/bench969")
import bench_and_verify as b  # noqa: E402
from vllm.models.deepseek_v4_1.engram_rows import (  # noqa: E402
    DEFAULT_CACHE_ROWS, EngramDiskRows, column_compaction_map, pack_engram_columns)
from vllm.models.deepseek_v4_1.tilelang.engram_rows import HEADS, ROW, SCALES  # noqa: E402

LABEL = sys.argv[1]


def main():
    base_dir = Path("/mnt/nvme/engram_rows_identity")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)
    table_rows = 24000
    offsets = tuple((i * table_rows) // HEADS for i in range(HEADS))
    tp3_cols = tuple(range(1, HEADS, 3))
    tp3_mask = sum(1 << c for c in tp3_cols)
    gen = torch.Generator().manual_seed(2026)
    weights = torch.randint(0, 120, (table_rows, ROW), generator=gen, dtype=torch.uint8)
    scales = torch.randint(115, 135, (table_rows, SCALES), generator=gen, dtype=torch.uint8)
    w_path, s_path, p_path = (str(base_dir / n) for n in ("weight.bin", "scale.bin", "packed.bin"))
    open(w_path, "wb").write(weights.numpy().tobytes())
    open(s_path, "wb").write(scales.numpy().tobytes())
    col_start, col_end, col_delta, owned_starts, owned_rows = column_compaction_map(offsets, table_rows, tp3_cols)
    pack_engram_columns(w_path, 0, s_path, 0, p_path, owned_starts, owned_rows)
    modes = [("two_plane_nocache", False, 0), ("two_plane_cache64k", False, DEFAULT_CACHE_ROWS),
             ("packed_nocache", True, 0), ("packed_cache64k", True, DEFAULT_CACHE_ROWS)]
    stream = torch.cuda.current_stream()

    def workloads():
        for m in (1, 4, 8, 16, 32, 48, 64, 128):
            rng = np.random.default_rng(100 + m)
            ids = np.zeros((m, HEADS), dtype=np.int64)
            for c in range(HEADS):
                lo = offsets[c]
                hi = offsets[c + 1] if c + 1 < HEADS else table_rows
                ids[:, c] = rng.integers(lo, min(lo + 64, hi), size=m)
            yield f"hot_M{m}", [ids] * 6
        for m in (4, 16, 48, 128):
            rng = np.random.default_rng(500 + m)
            ranks = np.arange(1, 257, dtype=np.float64)
            probs = ranks ** -1.35
            probs /= probs.sum()
            steps = []
            for _ in range(80):
                arr = np.zeros((m, HEADS), dtype=np.int64)
                for c in range(HEADS):
                    arr[:, c] = offsets[c] + rng.choice(256, size=m, p=probs)
                steps.append(arr)
            yield f"zipf_M{m}", steps
        for m in (4, 16, 32):
            steps = []
            for s in range(20):
                arr = np.zeros((m, HEADS), dtype=np.int64)
                for c in range(HEADS):
                    arr[:, c] = offsets[c] + s * m + np.arange(m)
                steps.append(arr)
            yield f"cold_M{m}", steps
        arr = np.zeros((32, HEADS), dtype=np.int64)
        for c in range(HEADS):
            arr[:, c] = offsets[c] + (np.arange(32) % 24) * 15 + 14
        yield "dup_M32", [arr, arr]

    out = {"label": LABEL, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "workloads": {}}
    try:
        for name, steps in workloads():
            m = steps[0].shape[0]
            gpu = [torch.from_numpy(a).cuda() for a in steps]
            live = torch.tensor([m], dtype=torch.int32, device="cuda:0")
            rec = {}
            for mode, packed, cache_rows in modes:
                r = EngramDiskRows(device="cuda:0", table_rows=table_rows, shard_start=0, shard_end=table_rows,
                                   max_lookups=128 * HEADS, columns=tp3_mask, cache_rows=cache_rows)
                try:
                    if packed:
                        r.add_packed_source(p_path, 0, col_start=col_start, col_end=col_end, col_delta=col_delta)
                    else:
                        r.add_source(w_path, 0, scale=False)
                        r.add_source(s_path, 0, scale=True)
                    h = hashlib.sha256()
                    o = torch.empty((m, 8 * ROW), dtype=torch.bfloat16, device="cuda:0")
                    for a, g in zip(steps, gpu):
                        r.stage(stream, g, m * HEADS, a)
                        r.gather(o, g, live)
                        torch.cuda.synchronize()
                        h.update(o.view(torch.uint16).cpu().numpy().tobytes())
                    st = b._wait_reader_idle(r)
                    rec[mode] = {"sha": h.hexdigest()[:16], "cache_hits": st["cache_hits"],
                                 "cache_misses": st["cache_misses"], "read_calls": st["read_calls"]}
                finally:
                    r.close()
            rec["modes_agree"] = len({v["sha"] for v in rec.values() if isinstance(v, dict)}) == 1
            out["workloads"][name] = rec
            print(f"[{LABEL}] {name}: " + " ".join(f"{k}={v['sha']}" for k, v in rec.items() if isinstance(v, dict))
                  + f" agree={rec['modes_agree']}", flush=True)
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
    out["all_modes_agree"] = all(v["modes_agree"] for v in out["workloads"].values())
    out["digest"] = hashlib.sha256(json.dumps({k: {m: v[m]["sha"] for m in v if isinstance(v[m], dict)}
                                               for k, v in out["workloads"].items()}, sort_keys=True).encode()).hexdigest()[:16]
    out["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"/out/rows-{LABEL}.json", "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps({"all_modes_agree": out["all_modes_agree"], "digest": out["digest"]}))


if __name__ == "__main__":
    main()
