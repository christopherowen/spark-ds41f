#!/usr/bin/env python3
"""Build overlay gemv-geom: B12X's TMA prefill GEMV with a configurable launch geometry.

usage: make_overlay_gemvgeom.py BUILD_B12X_DIR   (writes ~/spark3-overlay/gemv-geom2/b12x/gemm/bf16_gemv/)

BUILD_B12X_DIR must be the serving image's b12x package (r5o: /opt/spark3/candidate/b12x/b12x).
Overlay gemv-geom (run61, run62) was built from a pre-r5o tree and lacks r5o's proxy fence before
the TMA refill; gemv-geom2 is built from the r5o image.

Bf16PrefillKernel accumulates every output over the whole K in one CTA: per 16-wide K segment one
tensor-core product, then a compensated FP32 addition, segments in K order. The row tile (16 per
compute warp), column tile, K tile and pipeline depth only choose which outputs a CTA computes
and how loads are staged, not that sequence. The production geometry (64x64 tiles, 64-deep K
tiles, two stages) stays the default; GEOMETRIES names the others, each with its own compile
cache key, and backend "prefill_small" selects small_geometry(N) (decode sizes: fewer padded rows,
narrower column tiles for more CTAs, a deeper pipeline for the latency-bound K loop). run61:
every geometry bit-identical to production's at 1-4000 rows for all three served shapes.
"""
import os
import re
import sys

SRC = os.path.join(sys.argv[1], "gemm", "bf16_gemv")
OUT = os.environ.get("GEMVGEOM_OUT", os.path.expanduser("~/spark3-overlay/gemv-geom2/b12x/gemm/bf16_gemv"))
os.makedirs(OUT, exist_ok=True)

p = open(os.path.join(SRC, "_prefill.py")).read()
old = """class Bf16PrefillKernel:
    \"\"\"TMA-fed BF16 matrix product with FP32 accumulators and runtime rows.\"\"\"

    num_threads = 160
    num_compute_warps = 4
    producer_warp = 4
    tile_m = 64
    tile_n = 64
    tile_k = 64
    num_stages = 2
"""
new = """DEFAULT_GEOMETRY = (64, 64, 64, 2)  # tile_m, tile_n, tile_k, num_stages (production)
GEOMETRIES = {DEFAULT_GEOMETRY, (16, 64, 128, 4), (16, 64, 64, 4), (16, 32, 128, 4), (32, 64, 128, 3),
              (16, 64, 256, 2), (16, 16, 128, 4)}


def small_geometry(n):
    \"\"\"Decode sizes (measured at 1-48 rows): 16-row tiles, 128-deep K tiles, four stages; 16 output
    columns per CTA for the 384-wide router gate, 32 for the wider compressor projections.\"\"\"
    return (16, 16 if n <= 384 else 32, 128, 4)


SMALL_GEOMETRY = small_geometry


class Bf16PrefillKernel:
    \"\"\"TMA-fed BF16 matrix product with FP32 accumulators and runtime rows.\"\"\"

    num_threads = 160
    num_compute_warps = 4
    producer_warp = 4
    tile_m = 64
    tile_n = 64
    tile_k = 64
    num_stages = 2
"""
assert old in p
p = p.replace(old, new)
old = """    def __init__(self, n: int, k: int):
        self.n, self.k = int(n), int(k)
        if self.n <= 0 or self.k <= 0 or self.k % self.tile_k:
            raise ValueError("BF16 prefill needs positive N and K divisible by 64")
"""
new = """    def __init__(self, n: int, k: int, geometry: tuple = DEFAULT_GEOMETRY):
        self.n, self.k = int(n), int(k)
        if tuple(geometry) not in GEOMETRIES:
            raise ValueError(f"unsupported BF16 prefill geometry {geometry}")
        self.tile_m, self.tile_n, self.tile_k, self.num_stages = (int(v) for v in geometry)
        self.num_compute_warps = self.tile_m // 16
        self.producer_warp = self.num_compute_warps
        self.num_threads = 32 * (self.num_compute_warps + 1)
        if self.n <= 0 or self.k <= 0 or self.k % self.tile_k:
            raise ValueError("BF16 prefill needs positive N and K divisible by the K tile")
"""
assert old in p
p = p.replace(old, new)
old = """@cache
def _kernel(n, k):
    return Bf16PrefillKernel(n, k)
"""
new = """@cache
def _kernel(n, k, geometry=DEFAULT_GEOMETRY):
    return Bf16PrefillKernel(n, k, geometry)


def _spec_name(geometry):
    if tuple(geometry) == DEFAULT_GEOMETRY:
        return "gemm.bf16_prefill"
    return "gemm.bf16_prefill.m{}n{}k{}s{}".format(*geometry)
"""
assert old in p
p = p.replace(old, new)
old = """@program_cache
def compile_prefill(ordinal, max_rows, n, k, output_dtype):"""
new = """@program_cache
def compile_prefill(ordinal, max_rows, n, k, output_dtype, geometry=DEFAULT_GEOMETRY):"""
assert old in p
p = p.replace(old, new)
old = """        raw = b12x_compile(
            _kernel(n, k), *args,
            compile_spec=KernelCompileSpec.from_key("gemm.bf16_prefill", 3, key),
        )"""
new = """        raw = b12x_compile(
            _kernel(n, k, tuple(geometry)), *args,
            compile_spec=KernelCompileSpec.from_key(_spec_name(geometry), 3, key),
        )"""
assert old in p
p = p.replace(old, new)
open(os.path.join(OUT, "_prefill.py"), "w").write(p)

t = open(os.path.join(SRC, "_tuning.py")).read()
old = """    if config.backend == "prefill" and _prefill_eligible(query):
        return
"""
new = """    if config.backend in ("prefill", "prefill_small") and _prefill_eligible(query):
        return
"""
assert old in t
t = t.replace(old, new)
open(os.path.join(OUT, "_tuning.py"), "w").write(t)

r = open(os.path.join(SRC, "_preparation.py")).read()
m = re.search(r"    if config.backend == \"prefill\":\n        from \._prefill import compile_prefill\n        launcher = compile_prefill\(ordinal, query\.max_rows, query\.out_features,([^\n]*)\n([^\n]*)\n", r)
assert m, "preparation dispatch"
block = m.group(0)
new_block = block.replace('if config.backend == "prefill":', 'if config.backend in ("prefill", "prefill_small"):')
new_block = new_block.replace("from ._prefill import compile_prefill", "from ._prefill import SMALL_GEOMETRY, DEFAULT_GEOMETRY, compile_prefill")
new_block = new_block.replace("query.in_features, query.output_dtype)",
                              "query.in_features, query.output_dtype,\n                                   SMALL_GEOMETRY(query.out_features) if config.backend == \"prefill_small\" else DEFAULT_GEOMETRY)")
assert "SMALL_GEOMETRY(query.out_features) if config.backend" in new_block, new_block
open(os.path.join(OUT, "_preparation.py"), "w").write(r.replace(block, new_block))
print(block)
print("wrote", OUT)
