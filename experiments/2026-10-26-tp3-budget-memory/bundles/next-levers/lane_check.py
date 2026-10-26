#!/usr/bin/env python3
"""The workspace each V4.1 TileLang attention layer reserves, by layer role, at TP3's
and TP4's shapes and budgets: what the drafter's lane held before 0506 (every slot
sized to the largest reservation of any layer) and what it holds after (the drafter's
own layers' largest), from the release's own plan functions on the GPU.

Layer roles (the checkpoint's compress_ratios, index and candidate source layers):
window-only 0-1 and the drafter's 40-42; ratio 2, layers 2-19, with indexers at 2, 8 and
14; ratio 1, layers 20-39, with layer 20 the candidate source and 24, 28, 32 and 36
re-indexing its candidates. A layer reserves, as one borrow each (``reserve_all`` before,
``reserve_profile_scratch`` after): every index mode's scratch for its chunk, and every
attention mode's metadata arena with the MLA scratch for the decode route's split rows.
"""
import json

import torch

from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.models.deepseek_v4_1.tilelang import attention as tla
from vllm.models.deepseek_v4_1.tilelang.indexer import CANDIDATE_BLOCK, CANDIDATE_BLOCKS
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backends.mla.compressor_utils import get_dspark_swa_index_width

BLOCK, WINDOW, MAX_MODEL_LEN, DRAFTS = 256, 128, 1048576, 5
SHAPES = {"TP3": (24, 8), "TP4": (16, 16)}  # local heads (TP3 pads 64 to 72), streams
BUDGETS = (8192, 12288, 16384)
ROLES = {  # role: (compress ratio, index width or None, re-index, candidate source)
    "window (0-1)": (0, None, False, False),
    "ratio 2 (2-19)": (2, None, False, False),
    "ratio 2 index (2, 8, 14)": (2, "full", False, False),
    "ratio 1 (21-39)": (1, None, False, False),
    "candidate source (20)": (1, "full", False, True),
    "re-index (24, 28, 32, 36)": (1, "candidates", True, False),
    "drafter (40-42)": (0, None, False, False),
}


def nbytes(specs) -> int:
    return sum(round_up(torch.Size(s).numel() * d.itemsize, 256) for s, d in specs)


def reservations(role, heads, streams, capacity, device) -> list[int]:
    ratio, index, reindex, candidate_source = ROLES[role]
    draft = role.startswith("drafter")
    swa_width = get_dspark_swa_index_width(WINDOW, DRAFTS) if draft else WINDOW
    main_page = BLOCK // max(ratio, 1)
    main_width = -(-MAX_MODEL_LEN // BLOCK)
    mla = tla._mla_plan(heads, swa_width, 512 if ratio else 0, main_page, BLOCK,
                        capacity, device)
    out = []
    if index is not None:
        width = CANDIDATE_BLOCKS * CANDIDATE_BLOCK if reindex else main_width * main_page
        plan = tla._index_plan(width, main_page, reindex, candidate_source)
        for chunk in (DeepseekV4Attention.DECODE_CHUNK, DeepseekV4Attention.INDEX_CHUNK):
            out.append(nbytes(plan.scratch_specs(chunk)))
    decode_rows = streams * (1 + DRAFTS)  # the largest CUDA graph's rows
    split_rows = min(mla.decode_max_rows, capacity)
    for rows in (min(capacity, decode_rows), capacity):
        specs = [((rows, swa_width), torch.int32), ((rows,), torch.int32),
                 ((rows,), torch.int32)]
        if ratio:
            specs.append(((rows, main_width), torch.int32))
        out.append(nbytes([*specs, *mla.scratch_specs(split_rows)]))
    return out


def main() -> None:
    device = torch.device("cuda")
    record = {}
    for shape, (heads, streams) in SHAPES.items():
        for capacity in BUDGETS:
            per_role = {role: max(reservations(role, heads, streams, capacity, device))
                        for role in ROLES}
            before = max(per_role.values())
            after = per_role["drafter (40-42)"]
            record[f"{shape} {capacity}"] = {
                "largest reservation by role, MiB": {
                    role: round(v / 2**20, 2) for role, v in per_role.items()},
                "drafter lane before, MiB": round(before / 2**20, 2),
                "drafter lane after (its attention), MiB": round(after / 2**20, 2),
                "saved, GiB": round((before - after) / 2**30, 3),
            }
            print(shape, capacity, json.dumps(record[f"{shape} {capacity}"]), flush=True)
    print("RECORD " + json.dumps(record))


if __name__ == "__main__":
    main()
