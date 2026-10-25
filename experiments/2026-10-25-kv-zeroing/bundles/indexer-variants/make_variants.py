"""Write the indexer variants for the read-bound diagnosis from vLLM's 0503 indexer.py.

usage: make_variants.py INDEXER_0503 OUT_DIR
"""
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

LOADS = '''            for p, d in T.Parallel(block_P, D):
                pos = position(p0, T.min(p, extent - 1 - p0))
                page = T.min(
                    T.max(PageTable[row, T.min(pos // page_size, table_width - 1)], 0),
                    pool_pages - 1,
                )
                K_shared[p, d] = KPool[page, (pos % page_size) * D + d]
            for p in T.Parallel(block_P):
                pos = position(p0, T.min(p, extent - 1 - p0))
                page = T.min(
                    T.max(PageTable[row, T.min(pos // page_size, table_width - 1)], 0),
                    pool_pages - 1,
                )
                SFK_shared[p, 0] = KScale[page, scale_word0 + pos % page_size]
'''
assert source.count(LOADS) == 1

# v2: a dense tile lies inside one page, so its page is read once a tile; the row
# index is clamped as in v1. Candidate mode keeps v1's per-row loads.
HOISTED = '''            if dense:
                tile_page = T.alloc_var(
                    T.int32,
                    init=T.min(
                        T.max(PageTable[row, T.min(p0 // page_size, table_width - 1)], 0),
                        pool_pages - 1,
                    ),
                )
                for p, d in T.Parallel(block_P, D):
                    slot = (p0 + T.min(p, extent - 1 - p0)) % page_size
                    K_shared[p, d] = KPool[tile_page, slot * D + d]
                for p in T.Parallel(block_P):
                    slot = (p0 + T.min(p, extent - 1 - p0)) % page_size
                    SFK_shared[p, 0] = KScale[tile_page, scale_word0 + slot]
            else:
''' + "".join("    " + line + "\n" for line in LOADS.splitlines())

# v3: the release's loads (the row's own position and page), guarded by the extent:
# rows past it keep whatever the tile's shared memory held, and nothing past the
# extent is read from the cache.
GUARDED = '''            for p, d in T.Parallel(block_P, D):
                if p0 + p < extent:
                    pos = position(p0, p)
                    page = T.min(
                        T.max(PageTable[row, T.min(pos // page_size, table_width - 1)], 0),
                        pool_pages - 1,
                    )
                    K_shared[p, d] = KPool[page, (pos % page_size) * D + d]
            for p in T.Parallel(block_P):
                if p0 + p < extent:
                    pos = position(p0, p)
                    page = T.min(
                        T.max(PageTable[row, T.min(pos // page_size, table_width - 1)], 0),
                        pool_pages - 1,
                    )
                    SFK_shared[p, 0] = KScale[page, scale_word0 + pos % page_size]
'''

SIGNATURE = '''        extent,
        position,
        slot_valid,
        Scores,
    ):
        K_shared = T.alloc_shared([block_P, D], fp4)'''
assert source.count(SIGNATURE) == 1
DENSE_CALL = '''                    lambda p0, p: p0 + p,
                    lambda p0, p: True,
                    Scores,
                )'''
CANDIDATE_CALL = '''                lambda p0, p: Candidates[row, T.min(p0 + p, width - 1)] >= 0,
                Scores,
            )'''
assert source.count(DENSE_CALL) == 1 and source.count(CANDIDATE_CALL) == 1

(out / "v1.py").write_text(source)
v2 = source.replace(LOADS, HOISTED).replace(
    SIGNATURE, SIGNATURE.replace("        Scores,\n    ):", "        Scores,\n        dense,\n    ):")
).replace(DENSE_CALL, DENSE_CALL.replace("Scores,\n                )", "Scores,\n                    True,\n                )")
).replace(CANDIDATE_CALL, CANDIDATE_CALL.replace("Scores,\n            )", "Scores,\n                False,\n            )"))
(out / "v2.py").write_text(v2)
(out / "v3.py").write_text(source.replace(LOADS, GUARDED))
print("wrote v1 v2 v3")
