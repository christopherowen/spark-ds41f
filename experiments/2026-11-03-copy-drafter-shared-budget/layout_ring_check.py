"""Check of overlay/layout-ring's per-step record (the ring-c37-m8-nodead arm's)."""
import os
import struct
import tempfile
from types import SimpleNamespace

import numpy as np

from vllm.v1.worker.gpu import model_runner as mr

lines = []
mr.logger.warning = lambda fmt, *args: lines.append(fmt % args)
mr.get_tensor_model_parallel_rank = lambda: 2
mr.GPUModelRunner._LAYOUT_RING_DIR = tempfile.mkdtemp()
RECORD = mr.GPUModelRunner._LAYOUT_RING_RECORD
assert RECORD.size == 40


def runner(steps):
    # The runner's class constants, which the stub stands in for.
    r = SimpleNamespace(
        _LAYOUT_RING_DIR=mr.GPUModelRunner._LAYOUT_RING_DIR,
        _LAYOUT_RING_RECORD=RECORD,
        adaptive_verification=SimpleNamespace(_batch_budget=None),
        copy_drafter=SimpleNamespace(_reads={"reads": 0, "stale": 0}),
    )
    for eligible, budget, tokens, padded, scheduled, drafts in steps:
        if eligible is None:
            r.adaptive_verification._batch_budget = None
        else:
            r.adaptive_verification._batch_budget = (eligible, {}, budget)
            r.copy_drafter._reads["reads"] += 1
            r.copy_drafter._reads["stale"] += 1
        mr.GPUModelRunner._layout_ring_record(
            r, tokens, padded, len(scheduled), np.array(scheduled, np.int32),
            None if drafts is None else np.array(drafts, np.int32),
        )
    os.fsync(r._layout_ring_fd)
    return r


def records(r):
    with open(f"{mr.GPUModelRunner._LAYOUT_RING_DIR}/rank2.bin", "rb") as handle:
        blob = handle.read()
    return [RECORD.unpack_from(blob, i) for i in range(0, len(blob), RECORD.size)]


steps = [({"a": 5, "b": 16}, 20, 23, 24, [10, 13], [8, 16]), ({"a": 8}, 8, 9, 12, [9], [8]),
         (None, 0, 3, 3, [1, 1, 1], None)] * 3
one = records(runner(steps))
assert len(one) == 9 and one[0][0] == 0 and one[8][0] == 8, one
step, tokens, padded, reqs, _, _, _, budget, reads, stale = one[0]
assert (tokens, padded, reqs, budget, reads, stale) == (23, 24, 2, 20, 1, 1), one[0]
assert one[2][7] == -1 and one[2][5] == 0 and one[2][6] == 0, one[2]  # no drafts: no budget
assert one[5][8] == 4 and one[5][9] == 4, one[5]  # cumulative counter reads
two = records(runner(steps))
assert one == two
other = records(runner([({"a": 5, "b": 15}, 20, 23, 24, [10, 13], [8, 16])] + steps[1:]))
assert other[0][6] != one[0][6] and other[0][4:6] == one[0][4:6] and other[1:] == one[1:]
assert len(lines) == 3 and lines[0].startswith("LAYOUT-RING "), lines
print(lines[0], "| layout ring OK:", len(one), "records of", RECORD.size, "bytes")
